"""Model download manager (L5.05): a queue of model downloads with progress, speed, ETA, pause / resume / cancel and
a SHA-256 check.

- Each job streams to `<name>.part`. Pause keeps the part; resume asks the server for the rest
  (`Range: bytes=<have>-`, 206 appends, 200 restarts from zero). Cancel deletes the part.
- 429 / 5xx retry up to 3 times with backoff (Retry-After honoured, capped).
- When the expected SHA-256 is known (Civitai lists it; Hugging Face LFS files carry it), the finished file must
  match it, or it is removed and the job fails with both hashes. Without a hash the job says "not verified".
- Only `<name>.part` -> `<name>` on success: a broken download never leaves a file that looks complete.
- Secrets travel in the request headers only and are never part of a job's public view.

The destination is decided (and checked to be inside ComfyUI's model folders) by model_browser_routes before a job
is added; this module only writes where it is told.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

log = logging.getLogger("C2C.ModelDownloads")

CHUNK = 1 << 20
RETRIES = 3
BACKOFF_S = (2.0, 5.0, 12.0)
KEEP_FINISHED = 50
ACTIVE = ("queued", "running")


@dataclass
class DownloadJob:
    url: str
    dest: Path
    headers: dict = field(repr=False)
    sha256: Optional[str] = None
    label: str = ""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    status: str = "queued"             # queued | running | paused | done | error | cancelled
    total: int = 0
    done: int = 0
    speed: float = 0.0                 # bytes / s (smoothed)
    error: str = ""
    verified: Optional[bool] = None    # True = hash matched, None = no hash to check
    created: float = field(default_factory=time.time)
    finished: float = 0.0
    _want: str = field(default="", repr=False)    # "" | "pause" | "cancel"

    @property
    def part(self) -> Path:
        return self.dest.with_name(self.dest.name + ".part")

    def public(self) -> dict:
        eta = (self.total - self.done) / self.speed if self.speed > 0 and self.total > self.done else None
        return {"id": self.id, "label": self.label or self.dest.name, "dest": str(self.dest), "status": self.status,
                "total": self.total, "done": self.done, "speed": round(self.speed), "eta_s": None if eta is None
                else round(eta), "progress": round(self.done / self.total, 4) if self.total else None,
                "error": self.error, "verified": self.verified, "sha256_expected": bool(self.sha256)}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(CHUNK * 4), b""):
            h.update(chunk)
    return h.hexdigest()


class DownloadManager:
    def __init__(self, max_parallel: int = 2, session_factory: Optional[Callable] = None,
                 sleep: Callable = asyncio.sleep):
        self.jobs: dict[str, DownloadJob] = {}
        self._sem: Optional[asyncio.Semaphore] = None
        self._max = max(1, int(max_parallel))
        self._session_factory = session_factory
        self._sleep = sleep
        self._tasks: dict[str, asyncio.Task] = {}

    # ── public API (call on the server's event loop) ─────────────────────────────────────────────────────────
    def add(self, url: str, dest: Path, headers: Optional[dict] = None, sha256: Optional[str] = None,
            label: str = "") -> DownloadJob:
        for j in self.jobs.values():                     # the same file twice: hand back the live job
            if j.dest == Path(dest) and j.status in ACTIVE + ("paused",):
                return j
        job = DownloadJob(url=url, dest=Path(dest), headers=dict(headers or {}),
                          sha256=(sha256 or "").strip().lower() or None, label=label)
        self.jobs[job.id] = job
        self._prune()
        self._start(job)
        return job

    def list(self) -> list:
        return [j.public() for j in sorted(self.jobs.values(), key=lambda j: j.created, reverse=True)]

    def pause(self, job_id: str) -> bool:
        j = self.jobs.get(job_id)
        if not j or j.status not in ACTIVE:
            return False
        j._want = "pause"
        if j.status == "queued":
            j.status = "paused"
        return True

    def resume(self, job_id: str) -> bool:
        j = self.jobs.get(job_id)
        if not j or j.status not in ("paused", "error"):
            return False
        j._want, j.error, j.status = "", "", "queued"
        self._start(j)
        return True

    def cancel(self, job_id: str) -> bool:
        j = self.jobs.get(job_id)
        if not j or j.status in ("done", "cancelled"):
            return False
        if j.status == "running":
            j._want = "cancel"                           # the running task removes the part
        else:
            j.status = "cancelled"
            j.finished = time.time()
            self._remove_part(j)
        return True

    def clear_finished(self) -> int:
        gone = [k for k, j in self.jobs.items() if j.status in ("done", "error", "cancelled")]
        for k in gone:
            self.jobs.pop(k, None)
        return len(gone)

    async def wait(self, job_id: str) -> None:
        t = self._tasks.get(job_id)
        if t:
            await t

    # ── internals ─────────────────────────────────────────────────────────────────────────────────────────────
    def _start(self, job: DownloadJob) -> None:
        if self._sem is None:
            self._sem = asyncio.Semaphore(self._max)
        self._tasks[job.id] = asyncio.ensure_future(self._run(job))

    def _prune(self) -> None:
        old = [j for j in self.jobs.values() if j.status in ("done", "error", "cancelled")]
        for j in sorted(old, key=lambda j: j.finished)[:max(0, len(old) - KEEP_FINISHED)]:
            self.jobs.pop(j.id, None)

    @staticmethod
    def _remove_part(job: DownloadJob) -> None:
        try:
            job.part.unlink(missing_ok=True)
        except OSError:
            pass

    def _session(self):
        if self._session_factory:
            return self._session_factory()
        import aiohttp
        return aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None, sock_connect=30, sock_read=120))

    async def _run(self, job: DownloadJob) -> None:
        async with self._sem:
            if job._want == "pause" or job.status != "queued":
                return
            job.status = "running"
            attempt = 0
            while True:
                try:
                    outcome = await self._fetch(job)
                except Exception as exc:                 # network trouble: same retry budget as 5xx
                    outcome = ("retry", f"{type(exc).__name__}: {exc}", None)
                if outcome[0] == "retry" and attempt < RETRIES:
                    wait = outcome[2] if outcome[2] is not None else BACKOFF_S[attempt]
                    attempt += 1
                    log.info("download %s: %s - retry %d in %.0fs", job.id, outcome[1], attempt, wait)
                    await self._sleep(min(float(wait), 60.0))
                    if job._want:
                        outcome = ("stopped", "", None)
                    else:
                        continue
                break
            kind, msg = outcome[0], outcome[1]
            if kind == "stopped":
                if job._want == "cancel":
                    self._remove_part(job)
                    job.status, job.finished = "cancelled", time.time()
                else:
                    job.status = "paused"
                job._want = ""
                return
            if kind in ("error", "retry"):
                job.status, job.error, job.finished = "error", msg, time.time()
                return
            await self._finish(job)

    async def _fetch(self, job: DownloadJob):
        have = job.part.stat().st_size if job.part.exists() else 0
        hdrs = dict(job.headers)
        if have:
            hdrs["Range"] = f"bytes={have}-"
        job.dest.parent.mkdir(parents=True, exist_ok=True)
        async with self._session() as sess:
            async with sess.get(job.url, headers=hdrs) as r:
                if r.status == 416 and have:             # we already have every byte
                    job.total = job.done = have
                    return ("ok", "", None)
                if r.status == 429 or r.status >= 500:
                    ra = r.headers.get("Retry-After")
                    return ("retry", f"HTTP {r.status}", float(ra) if ra and ra.isdigit() else None)
                if r.status in (401, 403):
                    return ("error", f"HTTP {r.status}: this file needs an account or token (Civitai API key / "
                                     "HF_TOKEN), or its licence must be accepted on the site first", None)
                if r.status == 404:
                    return ("error", "HTTP 404: the file is not at this address any more", None)
                if r.status not in (200, 206):
                    return ("error", f"HTTP {r.status}", None)
                if r.status == 200:
                    have = 0                             # the server ignored Range: start again
                length = int(r.headers.get("Content-Length") or 0)
                job.total = have + length if length else job.total
                job.done = have
                t0, d0 = time.monotonic(), have
                with open(job.part, "ab" if have else "wb") as fh:
                    async for chunk in r.content.iter_chunked(CHUNK):
                        if job._want:
                            return ("stopped", "", None)
                        fh.write(chunk)
                        job.done += len(chunk)
                        dt = time.monotonic() - t0
                        if dt >= 0.5:
                            inst = (job.done - d0) / dt
                            job.speed = inst if job.speed == 0 else 0.7 * job.speed + 0.3 * inst
                            t0, d0 = time.monotonic(), job.done
                if job.total and job.done < job.total:
                    return ("retry", f"connection closed at {job.done} of {job.total} bytes", None)
                job.total = job.total or job.done
                return ("ok", "", None)

    async def _finish(self, job: DownloadJob) -> None:
        if job.sha256:
            got = await asyncio.to_thread(_sha256, job.part)
            if got != job.sha256:
                self._remove_part(job)
                job.status, job.finished = "error", time.time()
                job.verified = False
                job.error = (f"SHA-256 mismatch: expected {job.sha256}, got {got}. The download was removed; try "
                             "again (a proxy or a changed file on the site can cause this).")
                return
            job.verified = True
        try:
            os.replace(job.part, job.dest)
        except OSError as exc:
            job.status, job.error, job.finished = "error", f"could not move the finished file into place: {exc}", \
                time.time()
            return
        job.status, job.finished, job.speed = "done", time.time(), 0.0
        log.info("Model downloaded: %s%s", job.dest, " (SHA-256 verified)" if job.verified else "")


_MANAGER: Optional[DownloadManager] = None


def get_manager() -> DownloadManager:
    global _MANAGER
    if _MANAGER is None:
        _MANAGER = DownloadManager()
    return _MANAGER
