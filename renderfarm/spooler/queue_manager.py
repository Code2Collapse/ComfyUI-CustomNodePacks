"""C2C Farm queue manager — the Tractor-style spooler.

Jobs are held LOCALLY when backends are at capacity and dispatched by a
single scheduler thread. Ordering: priority (desc), then submitted_at (asc),
then a monotonic sequence for absolute stability. Per-user concurrency is
enforced from users.json (`max_concurrent_jobs`), per-backend concurrency
from backends.json.

The manager is dependency-injected (adapter factory, user-limit fn, audit db)
so pytest can drive it with fakes; production wiring lives in
`get_queue_manager()`.
"""

from __future__ import annotations

import itertools
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field

log = logging.getLogger("C2C.Farm.spooler")

# Terminal states never leave the history; everything else is "live".
TERMINAL = {"complete", "failed", "cancelled"}


@dataclass
class Job:
    prompt_json: dict
    backend_name: str
    user: str
    project_name: str = ""
    priority: int = 5
    compute_profile: str = ""
    job_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    submitted_at: float = field(default_factory=time.time)
    seq: int = 0
    status: str = "queued"          # queued|paused|dispatching|running|complete|failed|cancelled
    remote_id: str | None = None
    error: str | None = None
    progress: float | None = None   # 0..1 while running, None when unknown
    result_paths: list = field(default_factory=list)
    done: threading.Event = field(default_factory=threading.Event, repr=False)

    def sort_key(self):
        return (-int(self.priority), self.submitted_at, self.seq)

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id, "backend_name": self.backend_name, "user": self.user,
            "project_name": self.project_name, "priority": self.priority,
            "compute_profile": self.compute_profile, "submitted_at": self.submitted_at,
            "status": self.status, "remote_id": self.remote_id, "error": self.error,
            "progress": self.progress, "result_paths": list(self.result_paths),
        }


class QueueManager:
    def __init__(self, adapter_factory, user_limit_fn, backend_limit_fn,
                 audit=None, poll_interval: float = 1.0, autostart: bool = True):
        """
        adapter_factory(backend_name) -> BackendAdapter
        user_limit_fn(user) -> int          (max concurrent jobs for the user)
        backend_limit_fn(backend_name) -> int
        audit: AuditDB or None
        """
        self._adapter_factory = adapter_factory
        self._user_limit = user_limit_fn
        self._backend_limit = backend_limit_fn
        self._audit = audit
        self._poll = poll_interval
        self._lock = threading.RLock()
        self._pending: list[Job] = []
        self._active: dict[str, Job] = {}
        self._history: list[Job] = []          # most-recent-first, capped
        self._seq = itertools.count()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="c2c-farm-spooler", daemon=True)
        if autostart:
            self._thread.start()

    # ── public API ───────────────────────────────────────────────────
    def submit(self, job: Job) -> Job:
        with self._lock:
            job.seq = next(self._seq)
            self._pending.append(job)
            self._pending.sort(key=Job.sort_key)
        if self._audit:
            self._audit.record_submitted(job.job_id, job.backend_name, job.user,
                                         job.project_name, job.priority,
                                         job.compute_profile, job.submitted_at)
        log.info("spooled job %s (user=%s prio=%s backend=%s)",
                 job.job_id, job.user, job.priority, job.backend_name)
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            for j in self._pending:
                if j.job_id == job_id:
                    return j
            if job_id in self._active:
                return self._active[job_id]
            for j in self._history:
                if j.job_id == job_id:
                    return j
        return None

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "pending": [j.to_dict() for j in sorted(self._pending, key=Job.sort_key)],
                "active": [j.to_dict() for j in self._active.values()],
                "recent": [j.to_dict() for j in self._history[:50]],
            }

    def bump(self, job_id: str, new_priority: int) -> bool:
        with self._lock:
            for j in self._pending:
                if j.job_id == job_id:
                    j.priority = max(1, min(10, int(new_priority)))
                    self._pending.sort(key=Job.sort_key)
                    if self._audit:
                        self._audit.set_priority(job_id, j.priority)
                    return True
        return False

    def pause(self, job_id: str) -> bool:
        """Only locally-queued jobs can pause; running jobs must be cancelled."""
        with self._lock:
            for j in self._pending:
                if j.job_id == job_id and j.status == "queued":
                    j.status = "paused"
                    self._set_status(j, "paused")
                    return True
        return False

    def resume(self, job_id: str) -> bool:
        with self._lock:
            for j in self._pending:
                if j.job_id == job_id and j.status == "paused":
                    j.status = "queued"
                    self._set_status(j, "queued")
                    return True
        return False

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            for j in list(self._pending):
                if j.job_id == job_id:
                    self._pending.remove(j)
                    self._finish(j, "cancelled", "cancelled while queued locally")
                    return True
            j = self._active.get(job_id)
        if j is not None:
            ok = True
            if j.remote_id:
                try:
                    ok = self._adapter_factory(j.backend_name).cancel(j.remote_id)
                except Exception as exc:  # noqa: BLE001
                    log.warning("remote cancel of %s failed: %s", job_id, exc)
                    ok = False
            with self._lock:
                self._active.pop(job_id, None)
            self._finish(j, "cancelled", "cancelled by user")
            return ok
        return False

    def wait(self, job_id: str, timeout: float | None = None) -> Job:
        job = self.get(job_id)
        if job is None:
            raise RuntimeError(f"C2C Farm: unknown job {job_id}")
        if not job.done.wait(timeout):
            raise TimeoutError(f"C2C Farm: job {job_id} still {job.status} after {timeout}s "
                               f"(it keeps running — track it in the C2C Farm dashboard)")
        return job

    def stop(self):
        self._stop.set()

    # ── scheduler ────────────────────────────────────────────────────
    def _counts(self):
        by_user: dict[str, int] = {}
        by_backend: dict[str, int] = {}
        for j in self._active.values():
            by_user[j.user] = by_user.get(j.user, 0) + 1
            by_backend[j.backend_name] = by_backend.get(j.backend_name, 0) + 1
        return by_user, by_backend

    def schedule_once(self) -> list[Job]:
        """One scheduling pass; returns jobs dispatched (called by loop + tests)."""
        dispatched = []
        with self._lock:
            by_user, by_backend = self._counts()
            for j in sorted(self._pending, key=Job.sort_key):
                if j.status != "queued":
                    continue
                if by_user.get(j.user, 0) >= self._user_limit(j.user):
                    continue
                if by_backend.get(j.backend_name, 0) >= self._backend_limit(j.backend_name):
                    continue
                self._pending.remove(j)
                j.status = "dispatching"
                self._active[j.job_id] = j
                by_user[j.user] = by_user.get(j.user, 0) + 1
                by_backend[j.backend_name] = by_backend.get(j.backend_name, 0) + 1
                dispatched.append(j)
        for j in dispatched:
            threading.Thread(target=self._run_job, args=(j,),
                             name=f"c2c-farm-job-{j.job_id}", daemon=True).start()
        return dispatched

    def _loop(self):
        while not self._stop.is_set():
            try:
                self.schedule_once()
            except Exception as exc:  # noqa: BLE001 — scheduler must survive anything
                log.exception("scheduler pass failed: %s", exc)
            self._stop.wait(self._poll)

    # ── job lifecycle (one worker thread per dispatched job) ────────
    def _run_job(self, job: Job):
        try:
            adapter = self._adapter_factory(job.backend_name)
            job.remote_id = adapter.submit(job.prompt_json, job.compute_profile, job.priority)
            job.status = "running"
            self._set_status(job, "running")
            while True:
                st = adapter.get_status(job.remote_id)
                job.progress = st.get("progress_pct")
                s = st.get("status")
                if s == "complete":
                    job.result_paths = adapter.get_result(job.remote_id) or []
                    self._done(job, "complete", None)
                    return
                if s == "error":
                    self._done(job, "failed", st.get("error") or "backend reported error")
                    return
                if s == "unknown":
                    self._done(job, "failed",
                               "backend lost track of the job (no queue/history entry)")
                    return
                time.sleep(self._poll)
        except Exception as exc:  # noqa: BLE001
            log.exception("job %s failed: %s", job.job_id, exc)
            self._done(job, "failed", str(exc))

    def _done(self, job: Job, status: str, error: str | None):
        with self._lock:
            self._active.pop(job.job_id, None)
        self._finish(job, status, error)

    def _finish(self, job: Job, status: str, error: str | None):
        job.status = status
        job.error = error
        with self._lock:
            self._history.insert(0, job)
            del self._history[200:]
        if self._audit:
            self._audit.mark_completed(job.job_id, status, error)
        job.done.set()

    def _set_status(self, job: Job, status: str):
        if self._audit:
            self._audit.update_status(job.job_id, status)


# ── production wiring ────────────────────────────────────────────────
_qm_lock = threading.Lock()
_qm: QueueManager | None = None


def get_queue_manager() -> QueueManager:
    global _qm
    with _qm_lock:
        if _qm is None:
            from ..backends import get_adapter
            from ..logging.audit_db import get_audit_db
            from ..user_config import get_backend, get_user

            def user_limit(u):
                try:
                    return int(get_user(u)["max_concurrent_jobs"])
                except RuntimeError:
                    return 1

            def backend_limit(b):
                try:
                    return int(get_backend(b).get("max_concurrent_jobs", 1))
                except RuntimeError:
                    return 0  # disabled/unknown backend: hold jobs locally

            _qm = QueueManager(get_adapter, user_limit, backend_limit,
                               audit=get_audit_db())
        return _qm
