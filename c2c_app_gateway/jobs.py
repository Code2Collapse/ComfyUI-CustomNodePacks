"""Jobs: run submission, safe progress projection, cancel / timeout, declared-output collection, restart recovery.

A runner only ever sees ``project(job)``: public job id, app id, release, state, a 0..1 progress, an error CATEGORY with
a fixed sentence, and the declared outputs. The worker prompt id, node ids, class names, exception text, tracebacks and
file paths stay in the job's ``_private`` part (owner-only, never served to a runner).
"""
from __future__ import annotations

import asyncio
import logging
import re
import secrets
import shutil
import time
from pathlib import Path

from . import manifest as M
from .store import Store
from .worker import Worker, WorkerError

log = logging.getLogger("c2c_app_gateway.jobs")

TERMINAL = ("succeeded", "failed", "cancelled")
ERRORS = {
    "out_of_memory": "The server ran out of memory for this run. Try a smaller input or a lower resolution.",
    "invalid_input": "The app could not accept these inputs.",
    "cancelled": "The run was cancelled.",
    "timeout": "The run took longer than this app allows and was stopped.",
    "failed": "The run failed on the server. The app owner can see the details.",
    "lost": "The server lost track of this run (it was restarted). Please run it again.",
}
OUT_EXT = {"png", "jpg", "jpeg", "webp", "gif", "mp4", "webm", "mov", "wav", "flac", "mp3", "ogg", "exr", "json", "txt"}


class LimitError(RuntimeError):
    """A per-user / per-app / global limit; the message is safe for the runner."""


def _category(exc_type: str, message: str) -> str:
    t, m = (exc_type or "").lower(), (message or "").lower()
    if "outofmemory" in t or "out of memory" in m or "oom" in t.split(".")[-1]:
        return "out_of_memory"
    return "failed"


class JobManager:
    def __init__(self, store: Store, worker: Worker, *, poll_s: float = 2.0, queue_wait_s: float = 3600.0,
                 max_active_per_user: int = 4, max_active_total: int = 32, max_output_mb: int = 512,
                 retention_h: float = 72.0, worker_input_dir: str | None = None):
        self.store, self.worker = store, worker
        self.retention_s = float(retention_h) * 3600.0                   # finished jobs + outputs kept this long
        self.worker_input_dir = Path(worker_input_dir) if worker_input_dir else None
        self._last_sweep = 0.0
        self.poll_s, self.queue_wait_s = float(poll_s), float(queue_wait_s)
        self.max_active_per_user, self.max_active_total = int(max_active_per_user), int(max_active_total)
        self.max_output_bytes = int(max_output_mb) * 1024 * 1024
        self._jobs: dict[str, dict] = {}
        self._by_pid: dict[str, str] = {}
        self._live: dict[str, dict] = {}                                  # job_id -> {cur, done, frac, total, missing}
        self._subs: dict[str, list[asyncio.Queue]] = {}
        self._poll_task: asyncio.Task | None = None
        self._lock = asyncio.Lock()
        from .auth import RateLimiter
        self.rate = RateLimiter()
        worker.add_listener(self._on_event)

    # ── lifecycle ───────────────────────────────────────────────────────────────────────────────────────────
    def start(self) -> None:
        for j in self.store.list_jobs():
            j.pop("_settling", None)                                      # saved mid-settle before a restart
            self._jobs[j["job_id"]] = j
            pid = (j.get("_private") or {}).get("prompt_id")
            if pid:
                self._by_pid[pid] = j["job_id"]
            if j["state"] == "submitting":                                # died mid-submit: never re-submit blindly
                self._finish(j, "failed", "lost")
            elif j["state"] not in TERMINAL:
                self._live[j["job_id"]] = {"cur": None, "done": set(), "frac": 0.0,
                                           "total": j["_private"].get("nodes", 1), "missing": 0}
        self.worker.start_ws()
        self._poll_task = asyncio.ensure_future(self._poll_loop())

    async def stop(self) -> None:
        if self._poll_task:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass

    # ── public view ─────────────────────────────────────────────────────────────────────────────────────────
    @staticmethod
    def project(job: dict) -> dict:
        err = job.get("error")
        return {"job_id": job["job_id"], "app_id": job["app_id"], "release": job["release"], "state": job["state"],
                "progress": round(float(job.get("progress", 0.0)), 3),
                "error": {"category": err, "message": ERRORS[err]} if err else None,
                "created": job["created"], "finished": job.get("finished"),
                "outputs_ready": job["state"] == "succeeded"}

    def get(self, job_id: str) -> dict | None:
        return self._jobs.get(job_id)

    def subscribe(self, job_id: str) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=256)
        self._subs.setdefault(job_id, []).append(q)
        return q

    def unsubscribe(self, job_id: str, q: asyncio.Queue) -> None:
        subs = self._subs.get(job_id) or []
        if q in subs:
            subs.remove(q)

    def _publish(self, job: dict) -> None:
        self.store.save_job(job)
        view = self.project(job)
        for q in list(self._subs.get(job["job_id"], ())):
            try:
                q.put_nowait(view)
            except asyncio.QueueFull:
                pass

    def active(self, user: str | None = None, app_id: str | None = None) -> list[dict]:
        return [j for j in self._jobs.values() if j["state"] not in TERMINAL
                and (user is None or j["user"] == user) and (app_id is None or j["app_id"] == app_id)]

    # ── submit ──────────────────────────────────────────────────────────────────────────────────────────────
    async def submit(self, user: str, app: dict, manifest: dict, private: dict, resolved: dict) -> dict:
        lim = manifest["limits"]
        async with self._lock:
            if len(self.active()) >= self.max_active_total:
                raise LimitError("The server is busy. Try again in a few minutes.")
            if len(self.active(user)) >= self.max_active_per_user \
                    or len(self.active(user, app["app_id"])) >= lim["max_concurrent_per_user"]:
                raise LimitError("You already have the maximum number of runs in progress for this app.")
            if not self.rate.hit(f"runs:{user}:{app['app_id']}", lim["runs_per_hour"], 3600):
                raise LimitError("You reached this app's hourly run limit.")
            job_id = secrets.token_urlsafe(16)
            job = {"job_id": job_id, "app_id": app["app_id"], "release": manifest["release"], "user": user,
                   "state": "submitting", "progress": 0.0, "error": None, "created": time.time(), "finished": None,
                   "outputs": [], "max_runtime_s": lim["max_runtime_s"], "cancel_requested": False,
                   "_private": {"nodes": max(1, len(private["prompt"]))}}
            self._jobs[job_id] = job
            self.store.save_job(job)
        uploaded = {}
        try:
            for iid, v in resolved.items():
                if isinstance(v, dict) and "upload" in v:
                    uploaded[iid] = await self.worker.upload_image(v["upload"], f"c2cgw_{job_id}_{iid}.{v['ext']}")
            job["_private"]["uploads"] = sorted(uploaded.values())
            prompt = M.build_prompt(private, resolved, uploaded)
            pid = await self.worker.submit(prompt)
        except WorkerError as exc:
            job["_private"]["worker_error"] = exc.private if exc.private is not None else str(exc)
            self._finish(job, "failed", "invalid_input" if exc.status == 400 or exc.private else "failed")
            return job
        except Exception as exc:  # noqa: BLE001 - worker unreachable etc.
            job["_private"]["worker_error"] = repr(exc)
            self._finish(job, "failed", "failed")
            return job
        job["_private"]["prompt_id"] = pid
        self._by_pid[pid] = job_id
        self._live[job_id] = {"cur": None, "done": set(), "frac": 0.0, "total": job["_private"]["nodes"], "missing": 0}
        job["state"] = "queued"
        self._publish(job)
        return job

    # ── cancel ──────────────────────────────────────────────────────────────────────────────────────────────
    async def cancel(self, job: dict, reason: str = "cancelled") -> None:
        if job["state"] in TERMINAL:
            return
        job["cancel_requested"] = reason
        pid = job["_private"].get("prompt_id")
        if not pid:
            self._finish(job, "cancelled" if reason == "cancelled" else "failed", reason)
            return
        try:
            running, pending = await self.worker.queue()
            if pid in pending:
                await self.worker.delete_queued(pid)
                self._finish(job, "cancelled" if reason == "cancelled" else "failed", reason)
            elif pid in running:
                await self.worker.interrupt(pid)                          # targeted: never someone else's run
                self.store.save_job(job)
            # neither: it already finished; the poll loop settles it
        except WorkerError as exc:
            job["_private"]["cancel_error"] = str(exc)
            self.store.save_job(job)

    # ── worker events -> projection ─────────────────────────────────────────────────────────────────────────
    def _on_event(self, ev: dict) -> None:
        t, d = ev.get("type"), ev.get("data") or {}
        job_id = self._by_pid.get(d.get("prompt_id")) if isinstance(d, dict) else None
        job = self._jobs.get(job_id) if job_id else None
        if not job or job["state"] in TERMINAL:
            return
        live = self._live.setdefault(job_id, {"cur": None, "done": set(), "frac": 0.0,
                                              "total": job["_private"]["nodes"], "missing": 0})
        if t == "execution_start":
            job["state"], job["started"] = "running", time.time()
        elif t == "execution_cached":
            live["done"].update(map(str, d.get("nodes") or ()))
        elif t == "executing":
            node = d.get("node")
            if job["state"] == "queued":
                job["state"], job["started"] = "running", time.time()
            if live["cur"] is not None and live["cur"] != node:
                live["done"].add(str(live["cur"]))
            live["cur"], live["frac"] = node, 0.0
            if node is None:                                              # older cores: end of prompt
                asyncio.ensure_future(self._settle(job))
                return
        elif t == "progress":
            mx = d.get("max") or 0
            live["frac"] = max(0.0, min(1.0, (d.get("value") or 0) / mx)) if mx else 0.0
        elif t == "execution_success":
            asyncio.ensure_future(self._settle(job))
            return
        elif t == "execution_error":
            job["_private"]["worker_error"] = {k: d.get(k) for k in ("node_id", "node_type", "exception_type",
                                                                     "exception_message")}
            self._finish(job, "failed", _category(d.get("exception_type"), d.get("exception_message")))
            return
        elif t == "execution_interrupted":
            self._finish_interrupted(job)
            return
        else:
            return
        p = min(0.99, (len(live["done"]) + live["frac"]) / max(1, live["total"]))
        if job["state"] == "running" and abs(p - job.get("progress", 0.0)) >= 0.01 or t in ("execution_start",
                                                                                          "executing"):
            job["progress"] = max(job.get("progress", 0.0), p)
            self._publish(job)

    def _finish_interrupted(self, job: dict) -> None:
        reason = job.get("cancel_requested")
        if reason == "timeout":
            self._finish(job, "failed", "timeout")
        elif reason:
            self._finish(job, "cancelled", "cancelled")
        else:
            self._finish(job, "failed", "failed")

    def _finish(self, job: dict, state: str, error: str | None) -> None:
        if job["state"] in TERMINAL:
            return
        job["state"], job["error"], job["finished"] = state, (None if state == "succeeded" else error), time.time()
        if state == "succeeded":
            job["progress"] = 1.0
        self._live.pop(job["job_id"], None)
        self._remove_uploads(job)
        self._publish(job)
        self.store.audit("job_end", actor=job["user"], app=job["app_id"], release=job["release"], job=job["job_id"],
                         state=state, error=job["error"])

    # ── completion: fetch ONLY the declared outputs ─────────────────────────────────────────────────────────
    async def _settle(self, job: dict) -> None:
        if job["state"] in TERMINAL or job.get("_settling"):
            return
        job["_settling"] = True
        try:
            h = await self.worker.history(job["_private"]["prompt_id"])
            if not h:
                return
            st = h.get("status") or {}
            if st.get("status_str") == "error" or not st.get("completed", True):
                msgs = {m[0]: m[1] for m in st.get("messages", []) if isinstance(m, list) and len(m) == 2}
                if "execution_interrupted" in msgs:
                    self._finish_interrupted(job)
                else:
                    e = msgs.get("execution_error") or {}
                    job["_private"]["worker_error"] = {k: e.get(k) for k in ("node_id", "node_type", "exception_type",
                                                                             "exception_message")}
                    self._finish(job, "failed", _category(e.get("exception_type"), e.get("exception_message")))
                return
            private = self.store.read_private(job["app_id"], job["release"])
            outs, d, total = [], self.store.output_dir(job["job_id"]), 0
            for oid, o in private["outputs"].items():
                items = ((h.get("outputs") or {}).get(o["node"]) or {}).get(o["kind"]) or []
                for i, item in enumerate(items):
                    if not isinstance(item, dict):
                        continue
                    ext = (item.get("filename", "").rsplit(".", 1)[-1] or "").lower()
                    ext = ext if ext in OUT_EXT and re.fullmatch(r"[a-z0-9]{1,5}", ext) else "bin"
                    data = await self.worker.view(item, self.max_output_bytes - total)
                    total += len(data)
                    name = f"{oid}_{i}.{ext}"
                    (d / name).write_bytes(data)
                    outs.append({"id": oid, "index": i, "name": name, "size": len(data)})
            job["outputs"] = outs
            self._finish(job, "succeeded", None)
        except WorkerError as exc:
            job["_private"]["settle_error"] = str(exc)
            self._finish(job, "failed", "failed")
        finally:
            job.pop("_settling", None)

    # ── retention: uploads on the worker, finished jobs and their outputs ──────────────────────────────────
    def _remove_uploads(self, job: dict) -> None:
        """Delete this job's runner uploads from the worker's input folder (only when the gateway shares that disk:
        ``worker_input_dir``). ComfyUI has no delete route, so a remote worker keeps them (deployment guide)."""
        if not self.worker_input_dir:
            return
        base = (self.worker_input_dir / "c2c_gateway").resolve()
        for name in job["_private"].get("uploads") or ():
            leaf = name.split("/", 1)[-1]
            if not re.fullmatch(r"c2cgw_[A-Za-z0-9_-]+\.(png|jpg|webp)", leaf):
                continue
            p = (base / leaf).resolve()
            if p.parent == base:
                try:
                    p.unlink()
                except FileNotFoundError:
                    pass
                except OSError as exc:                                    # file in use: the next sweep retries
                    log.debug("upload cleanup: %s", exc)

    def sweep(self, now: float | None = None) -> int:
        """Forget finished jobs older than the retention: their outputs, record and signed links go with them."""
        now = time.time() if now is None else now
        gone = 0
        for job in list(self._jobs.values()):
            if job["state"] in TERMINAL and job.get("finished") and now - job["finished"] > self.retention_s:
                shutil.rmtree(self.store.root / "outputs" / job["job_id"], ignore_errors=True)
                try:
                    self.store.job_path(job["job_id"]).unlink()
                except FileNotFoundError:
                    pass
                self._jobs.pop(job["job_id"], None)
                pid = job["_private"].get("prompt_id")
                if pid:
                    self._by_pid.pop(pid, None)
                self._remove_uploads(job)
                gone += 1
        if gone:
            self.store.audit("jobs_expired", count=gone)
        self.store.rotate_audit()
        return gone

    # ── poll: completion fallback, timeouts, lost prompts ───────────────────────────────────────────────────
    async def poll_once(self) -> None:
        now = time.time()
        if now - self._last_sweep > 600:
            self._last_sweep = now
            self.sweep(now)
        for job in list(self.active()):
            pid = job["_private"].get("prompt_id")
            if not pid:
                continue
            started = job.get("started")
            over = (started and now - started > job["max_runtime_s"]) or \
                   (not started and now - job["created"] > job["max_runtime_s"] + self.queue_wait_s)
            if over and not job.get("cancel_requested"):
                await self.cancel(job, "timeout")
                continue
            try:
                h = await self.worker.history(pid)
                if h:
                    await self._settle(job)
                    continue
                running, pending = await self.worker.queue()
            except WorkerError:
                continue
            except Exception:  # noqa: BLE001 - worker down: try again next round
                continue
            live = self._live.setdefault(job["job_id"], {"cur": None, "done": set(), "frac": 0.0,
                                                         "total": job["_private"]["nodes"], "missing": 0})
            if pid in running and job["state"] == "queued":
                job["state"], job["started"] = "running", time.time()
                self._publish(job)
            if pid in running or pid in pending:
                live["missing"] = 0
            else:
                live["missing"] += 1
                if live["missing"] >= 3:
                    self._finish(job, "failed", "lost")

    async def _poll_loop(self) -> None:
        while True:
            try:
                await self.poll_once()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001
                log.exception("poll failed")
            await asyncio.sleep(self.poll_s)
