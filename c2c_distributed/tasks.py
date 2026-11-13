"""C2C GPU worker + typed tile-task service (ledger L9.214, A26 4.3). Design: docs/research/L9.214_gpu_worker_service_
design.md (workspace).

Independent tiles / windows of one job go to GPU workers that each own their model copies (problem 2 of A26 4: more
throughput; NOT weight pooling, NOT sequence parallelism). One worker thread per device - the mechanism C2C Multi-GPU
Split uses, and the only one Windows torch supports (no NCCL, no multi-process gloo; measured in L9.213).

The service is generic: a task's work is a registered executor ``fn(task, ctx) -> tensor``. It provides registration,
heartbeats, capacity-aware scheduling with VRAM reservations, model residency, leases with deadlines, retries, worker
loss, cancellation, progress, deterministic ordering, idempotent results, cleanup and per-worker diagnostics.
Other repos use it through ``SERVICE_API_VERSION`` and keep a serial fallback.
"""
from __future__ import annotations

import collections
import hashlib
import itertools
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

SERVICE_API_VERSION = 1
TERMINAL = ("DONE", "FAILED", "CANCELLED")


class TaskCancelled(Exception):
    """Raised by an executor (or ctx.check_cancel) when its job was cancelled."""


class TaskQCError(RuntimeError):
    """The executor's output does not match the task's declared shape / dtype, or is not finite."""


@dataclass(frozen=True)
class TaskManifest:
    job_id: str
    task_id: str
    order: int
    task_type: str
    model_fingerprint: str = ""
    workflow_release_id: str = ""
    input_asset_hash: str = ""
    tile_or_window_id: str = ""
    spatial_bounds: Optional[tuple] = None           # (y0, y1, x0, x1) of the core, in source pixels / latents
    temporal_bounds: Optional[tuple] = None          # (t0, t1)
    halo_bounds: Optional[tuple] = None              # (top, bottom, left, right, before, after) of context
    frame_mapping: Optional[tuple] = None
    seed: Optional[int] = None
    sampling_step_or_sigma: Optional[float] = None
    expected_output_shape: Optional[tuple] = None
    expected_dtype: Optional[str] = None             # "float32", "float16", "bfloat16", ...
    colour_space: str = ""
    alpha_semantics: str = ""                        # "", "straight", "premultiplied"
    vram_bytes: int = 0                              # reserved on the worker while the task runs
    payload: Any = None

    def validate(self) -> "TaskManifest":
        if not self.job_id or not self.task_id or not self.task_type:
            raise ValueError("job_id, task_id and task_type are required")
        if self.spatial_bounds is not None:
            y0, y1, x0, x1 = self.spatial_bounds
            if not (0 <= y0 < y1 and 0 <= x0 < x1):
                raise ValueError(f"task {self.task_id}: empty or negative spatial bounds {self.spatial_bounds}")
        if self.temporal_bounds is not None and not 0 <= self.temporal_bounds[0] < self.temporal_bounds[1]:
            raise ValueError(f"task {self.task_id}: bad temporal bounds {self.temporal_bounds}")
        if self.alpha_semantics not in ("", "straight", "premultiplied"):
            raise ValueError(f"task {self.task_id}: alpha_semantics must be straight or premultiplied")
        if self.vram_bytes < 0:
            raise ValueError("vram_bytes cannot be negative")
        return self


@dataclass
class TaskResult:
    task_id: str
    job_id: str
    order: int
    output: Any
    spatial_bounds: Optional[tuple]
    temporal_bounds: Optional[tuple]
    halo_bounds: Optional[tuple]
    shape: tuple
    dtype: str
    colour_space: str
    alpha_semantics: str
    duration_s: float
    worker_id: str
    device: str
    attempt: int
    checksum: str
    qc: dict


def tensor_checksum(t) -> str:
    """sha256 over shape, dtype and bytes; a tuple / list of tensors (or None) is hashed part by part."""
    import torch
    h = hashlib.sha256()
    for x in (t if isinstance(t, (tuple, list)) else (t,)):
        if x is None:
            h.update(b"None|")
            continue
        x = x.detach().to("cpu").contiguous()
        h.update(f"{tuple(x.shape)}|{x.dtype}|".encode())
        h.update(x.flatten().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


class WorkerContext:
    """What an executor gets: its device, resident models, the cancel flag, a log."""

    def __init__(self, service: "TileService", worker: "_Worker", job_id: str, attempt: int):
        self._svc, self._w, self.job_id, self.attempt = service, worker, job_id, attempt
        self.worker_id, self.device = worker.id, worker.device

    def get_model(self, fingerprint: str, loader: Callable[[Any], Any]):
        """The worker's own instance of a model: loaded once with loader(device), then reused (LRU-bounded)."""
        w = self._w
        with w.res_lock:
            if fingerprint in w.resident:
                w.resident.move_to_end(fingerprint)
                return w.resident[fingerprint]
        model = loader(self.device)
        with w.res_lock:
            w.resident[fingerprint] = model
            w.loads[fingerprint] += 1
            while len(w.resident) > self._svc.max_resident:
                old, inst = w.resident.popitem(last=False)
                self._svc._log(w, "evict", "", old)
                unload = getattr(inst, "c2c_unload", None)
                if callable(unload):
                    unload()
        self._svc._log(w, "load", "", fingerprint)
        return model

    def cancelled(self) -> bool:
        return self._svc._job_cancelled(self.job_id)

    def check_cancel(self) -> None:
        if self.cancelled():
            raise TaskCancelled(self.job_id)

    def log(self, msg: str) -> None:
        self._svc._log(self._w, "note", "", msg)


class _Worker:
    def __init__(self, wid: str, device, vram_total: int, slots: int):
        self.id, self.device, self.vram_total, self.slots = wid, device, int(vram_total), int(slots)
        self.reserved, self.busy = 0, 0
        self.leases, self.completed, self.failed = 0, 0, 0              # lifetime counters (diagnostics)
        self.alive, self.killed, self.stop, self.stuck = True, False, False, False
        self.last_beat = time.monotonic()
        self.inbox: collections.deque = collections.deque()
        self.resident: "collections.OrderedDict[str, Any]" = collections.OrderedDict()
        self.loads: collections.Counter = collections.Counter()
        self.res_lock = threading.Lock()
        self.log: collections.deque = collections.deque(maxlen=400)
        self.threads: list = []                          # one per slot, sharing the inbox
        self.hb_thread: Optional[threading.Thread] = None


@dataclass
class _Entry:
    task: TaskManifest
    attempt: int = 0
    not_before: float = 0.0
    excluded: set = field(default_factory=set)
    worker: Optional[str] = None
    started: float = 0.0
    deadline: float = 0.0
    lease_id: str = ""


@dataclass
class _Job:
    job_id: str
    total: int
    state: str = "RUNNING"
    cancelled: bool = False
    error: Optional[str] = None
    results: dict = field(default_factory=dict)
    duplicates_ignored: int = 0
    conflicts: list = field(default_factory=list)
    retries: int = 0
    failures: list = field(default_factory=list)


class TileService:
    def __init__(self, devices, *, vram_total=None, slots: int = 1, heartbeat_interval: float = 0.2,
                 heartbeat_timeout: float = 2.0, task_timeout: float = 600.0, max_attempts: int = 3,
                 retry_backoff: float = 0.05, max_resident: int = 2, progress: Optional[Callable] = None):
        self.heartbeat_interval, self.heartbeat_timeout = float(heartbeat_interval), float(heartbeat_timeout)
        self.task_timeout, self.max_attempts = float(task_timeout), int(max_attempts)
        self.retry_backoff, self.max_resident, self.progress = float(retry_backoff), int(max_resident), progress
        self._lock = threading.RLock()
        self._cond = threading.Condition(self._lock)
        self._executors: dict = {}
        self._pending: list = []
        self._running: dict = {}                       # task_id -> _Entry (leased)
        self._jobs: dict = {}
        self._stopped = False
        self.workers: dict = {}
        totals = list(vram_total) if vram_total is not None else [None] * len(devices)
        for i, (dev, tot) in enumerate(zip(devices, totals)):
            self.register_worker(f"w{i}", dev, tot, slots)
        self._monitor = threading.Thread(target=self._monitor_loop, name="c2c-tasks-monitor", daemon=True)
        self._monitor.start()

    # ── registration ────────────────────────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _probe_vram(dev) -> int:
        try:
            import torch
            d = torch.device(dev)
            if d.type == "cuda":
                return int(torch.cuda.mem_get_info(d)[1])
        except Exception:  # noqa: BLE001 - CPU / unknown: no limit
            pass
        return 1 << 62

    def register_worker(self, wid: str, device, vram_total=None, slots: int = 1) -> None:
        w = _Worker(wid, device, vram_total if vram_total is not None else self._probe_vram(device), slots)
        with self._lock:
            if wid in self.workers:
                raise ValueError(f"worker {wid} is already registered")
            self.workers[wid] = w
        w.threads = [threading.Thread(target=self._worker_loop, args=(w,), name=f"c2c-worker-{wid}-{k}", daemon=True)
                     for k in range(max(1, int(slots)))]
        w.hb_thread = threading.Thread(target=self._heartbeat_loop, args=(w,), name=f"c2c-hb-{wid}", daemon=True)
        for th in w.threads:
            th.start()
        w.hb_thread.start()
        self._log(w, "register", "", f"{device} vram={w.vram_total} slots={slots}")

    def register_executor(self, task_type: str, fn: Callable, version: int = 1) -> None:
        self._executors[task_type] = (fn, int(version))

    # ── jobs ────────────────────────────────────────────────────────────────────────────────────────────────────
    def submit(self, manifests) -> str:
        tasks = [m.validate() for m in manifests]
        if not tasks:
            raise ValueError("a job needs at least one task")
        job_ids = {t.job_id for t in tasks}
        if len(job_ids) != 1:
            raise ValueError("all tasks of one submit must share a job_id")
        ids = [t.task_id for t in tasks]
        if len(set(ids)) != len(ids):
            raise ValueError("task ids must be unique within a job")
        unknown = sorted({t.task_type for t in tasks} - set(self._executors))
        if unknown:
            raise ValueError(f"no executor registered for task type(s) {unknown}")
        job_id = tasks[0].job_id
        with self._cond:
            if job_id in self._jobs:
                raise ValueError(f"job {job_id} already exists")
            if not any(w.alive and w.vram_total >= max(t.vram_bytes for t in tasks) for w in self.workers.values()):
                raise ValueError("no registered worker has enough memory for the largest task")
            self._jobs[job_id] = _Job(job_id, len(tasks))
            self._pending.extend(_Entry(t) for t in tasks)
            self._assign()
            self._cond.notify_all()
        return job_id

    def wait(self, job_id: str, timeout: Optional[float] = None) -> str:
        end = None if timeout is None else time.monotonic() + timeout
        with self._cond:
            while self._jobs[job_id].state not in TERMINAL:
                left = None if end is None else end - time.monotonic()
                if left is not None and left <= 0:
                    break
                self._cond.wait(timeout=0.1 if left is None else min(0.1, left))
            return self._jobs[job_id].state

    def results(self, job_id: str) -> list:
        with self._lock:
            return sorted(self._jobs[job_id].results.values(), key=lambda r: r.order)

    def job(self, job_id: str) -> dict:
        with self._lock:
            j = self._jobs[job_id]
            return {"state": j.state, "done": len(j.results), "total": j.total, "error": j.error,
                    "retries": j.retries, "failures": list(j.failures), "duplicates_ignored": j.duplicates_ignored,
                    "conflicts": list(j.conflicts)}

    def cancel(self, job_id: str) -> None:
        with self._cond:
            j = self._jobs[job_id]
            if j.state in TERMINAL:
                return
            j.cancelled = True
            self._pending = [e for e in self._pending if e.task.job_id != job_id]
            if not any(e.task.job_id == job_id for e in self._running.values()):
                j.state = "CANCELLED"
            self._cond.notify_all()

    def close(self, job_id: str) -> None:
        """Forget a finished job (results, counters). Resident models stay for the next job."""
        with self._cond:
            j = self._jobs.get(job_id)
            if j is None:
                return
            if j.state not in TERMINAL:
                raise RuntimeError(f"job {job_id} is still {j.state}: cancel or wait first")
            del self._jobs[job_id]

    def release_models(self) -> None:
        for w in self.workers.values():
            with w.res_lock:
                for inst in w.resident.values():
                    unload = getattr(inst, "c2c_unload", None)
                    if callable(unload):
                        unload()
                w.resident.clear()

    def shutdown(self) -> None:
        with self._cond:
            self._stopped = True
            for w in self.workers.values():
                w.stop = True
            self._cond.notify_all()
        for w in self.workers.values():
            for th in w.threads:
                th.join(timeout=5)
        self._monitor.join(timeout=5)

    # ── test / ops hooks ────────────────────────────────────────────────────────────────────────────────────────
    def kill_worker(self, wid: str) -> None:
        """Simulate a crashed worker: it stops heartbeating and never reports again."""
        with self._cond:
            self.workers[wid].killed = True
            self._cond.notify_all()

    def snapshot(self) -> dict:
        with self._lock:
            return {"api_version": SERVICE_API_VERSION,
                    "workers": {w.id: {"device": str(w.device), "alive": w.alive, "stuck": w.stuck, "busy": w.busy,
                                       "slots": w.slots,
                                       "vram_total": w.vram_total, "reserved": w.reserved,
                                       "resident": list(w.resident), "loads": dict(w.loads),
                                       "leases": w.leases, "completed": w.completed, "failed": w.failed,
                                       "last_beat_age_s": round(time.monotonic() - w.last_beat, 3),
                                       "log": list(w.log)[-20:]} for w in self.workers.values()},
                    "pending": len(self._pending), "running": len(self._running),
                    "jobs": {k: v.state for k, v in self._jobs.items()}}

    # ── internals ───────────────────────────────────────────────────────────────────────────────────────────────
    def _log(self, w: _Worker, event: str, task_id: str, detail: str = "") -> None:
        w.log.append((round(time.time(), 3), event, task_id, detail))

    def _job_cancelled(self, job_id: str) -> bool:
        j = self._jobs.get(job_id)
        return j is None or j.cancelled or j.state == "FAILED"         # a failed job stops its other tasks too

    def _assign(self) -> None:
        """Under the lock: hand ready tasks to the best idle worker (resident model, most free memory, lowest id)."""
        now = time.monotonic()
        keep = []
        for e in sorted(self._pending, key=lambda x: (x.not_before, x.task.order)):
            if e.not_before > now:
                keep.append(e)
                continue
            cands = [w for w in self.workers.values()
                     if w.alive and not w.killed and not w.stuck and w.busy < w.slots and w.id not in e.excluded
                     and w.vram_total - w.reserved >= e.task.vram_bytes]
            if not cands:
                alive = [w for w in self.workers.values() if w.alive and not w.killed]
                if alive and all(w.id in e.excluded for w in alive):
                    e.excluded = set()                   # every worker tried once: allow a second round anywhere
                keep.append(e)
                continue
            fp = e.task.model_fingerprint
            best = min(cands, key=lambda w: (fp not in w.resident, -(w.vram_total - w.reserved), w.id))
            best.busy += 1
            best.leases += 1
            best.reserved += e.task.vram_bytes
            e.worker, e.started = best.id, now
            e.deadline, e.lease_id = now + self.task_timeout, uuid.uuid4().hex
            self._running[e.task.task_id] = e
            best.inbox.append(e)
            self._log(best, "lease", e.task.task_id, f"attempt {e.attempt + 1}")
        self._pending = keep

    def _release(self, e: _Entry) -> None:
        w = self.workers.get(e.worker)
        if w is not None:
            w.busy = max(0, w.busy - 1)
            w.reserved = max(0, w.reserved - e.task.vram_bytes)
        self._running.pop(e.task.task_id, None)

    def _requeue(self, e: _Entry, why: str, exclude: bool) -> None:
        j = self._jobs.get(e.task.job_id)
        if j is None or j.cancelled:
            return
        e.attempt += 1
        j.retries += 1
        j.failures.append({"task_id": e.task.task_id, "worker": e.worker, "attempt": e.attempt, "error": why})
        if e.attempt >= self.max_attempts:
            j.state, j.error = "FAILED", f"task {e.task.task_id} failed {e.attempt} times; last: {why}"
            self._pending = [p for p in self._pending if p.task.job_id != j.job_id]
            return
        if exclude and e.worker:
            e.excluded.add(e.worker)
        e.worker, e.lease_id = None, ""
        e.not_before = time.monotonic() + self.retry_backoff * (2 ** (e.attempt - 1))
        self._pending.append(e)

    def _finish_job_if_done(self, j: _Job) -> None:
        if j.state not in TERMINAL:
            if j.cancelled and not any(e.task.job_id == j.job_id for e in self._running.values()):
                j.state = "CANCELLED"
            elif len(j.results) == j.total:
                j.state = "DONE"

    def _heartbeat_loop(self, w: _Worker) -> None:
        while not w.stop and not self._stopped:
            if not w.killed:
                with self._lock:
                    w.last_beat = time.monotonic()
            time.sleep(self.heartbeat_interval)

    def _monitor_loop(self) -> None:
        while not self._stopped:
            time.sleep(min(self.heartbeat_interval, 0.1))
            with self._cond:
                now = time.monotonic()
                for w in self.workers.values():
                    if w.alive and now - w.last_beat > self.heartbeat_timeout:
                        w.alive = False                         # LOST: its leases go back to the queue
                        self._log(w, "lost", "", f"no heartbeat for {now - w.last_beat:.1f} s")
                        for e in [e for e in self._running.values() if e.worker == w.id]:
                            self._release(e)
                            self._requeue(e, f"worker {w.id} lost", exclude=True)
                for e in [e for e in self._running.values() if now > e.deadline]:
                    w = self.workers.get(e.worker)
                    if w is not None:
                        w.stuck = True                          # its thread is still inside the executor
                        self._log(w, "timeout", e.task.task_id, f"{self.task_timeout:g} s")
                    self._release(e)
                    self._requeue(e, f"task took longer than {self.task_timeout:g} s", exclude=True)
                for j in self._jobs.values():
                    self._finish_job_if_done(j)
                self._assign()
                self._cond.notify_all()

    def _worker_loop(self, w: _Worker) -> None:
        while True:
            with self._cond:
                while not w.inbox and not w.stop and not self._stopped:
                    self._cond.wait(timeout=self.heartbeat_interval)
                if w.stop or self._stopped:
                    return
                if w.killed:                                    # crashed: never reports, never takes more
                    return
                e = w.inbox.popleft()
                lease = e.lease_id
            self._run(w, e, lease)

    def _run(self, w: _Worker, e: _Entry, lease: str) -> None:
        t = e.task
        fn, _v = self._executors[t.task_type]
        ctx = WorkerContext(self, w, t.job_id, e.attempt + 1)
        t0 = time.perf_counter()
        out, err = None, None
        try:
            if ctx.cancelled():
                raise TaskCancelled(t.job_id)
            out = fn(t, ctx)
            qc = self._qc(t, out)
        except TaskCancelled:
            err = "cancelled"
        except Exception as exc:  # noqa: BLE001 - recorded and retried
            err = f"{type(exc).__name__}: {exc}"[:400]
        dur = time.perf_counter() - t0
        if w.killed:
            return                                              # a crashed worker reports nothing
        notify = None
        with self._cond:
            w.stuck = False                                     # the thread is back: it can take work again
            j = self._jobs.get(t.job_id)
            current = self._running.get(t.task_id)
            stale = current is None or current.lease_id != lease   # lease expired and the task moved on
            if not stale:
                self._release(e)
            if j is None:
                pass
            elif err == "cancelled" or j.cancelled:
                self._log(w, "cancelled", t.task_id)
            elif err is not None:
                w.failed += 1
                self._log(w, "fail", t.task_id, err)
                if not stale:
                    self._requeue(e, err, exclude=len(self.workers) > 1)
            else:
                first = out[0] if isinstance(out, (tuple, list)) else out
                res = TaskResult(t.task_id, t.job_id, t.order, out, t.spatial_bounds, t.temporal_bounds,
                                 t.halo_bounds, tuple(first.shape), str(first.dtype).replace("torch.", ""),
                                 t.colour_space, t.alpha_semantics, dur, w.id, str(w.device), e.attempt + 1,
                                 tensor_checksum(out), qc)
                prev = j.results.get(t.task_id)
                if prev is None:
                    j.results[t.task_id] = res
                    w.completed += 1
                    self._log(w, "done", t.task_id, f"{dur:.3f} s")
                    notify = (len(j.results), j.total)
                elif prev.checksum == res.checksum:
                    j.duplicates_ignored += 1                   # idempotent: the same result twice is one result
                    self._log(w, "duplicate", t.task_id)
                else:
                    j.conflicts.append({"task_id": t.task_id, "kept": prev.worker_id, "other": w.id})
                    self._log(w, "conflict", t.task_id)
                if stale and prev is None:
                    self._pending = [p for p in self._pending if p.task.task_id != t.task_id]
            if j is not None:
                self._finish_job_if_done(j)
            self._assign()
            self._cond.notify_all()
        if notify and self.progress:
            try:
                self.progress(t.job_id, *notify)
            except Exception:  # noqa: BLE001 - a progress callback never breaks the job
                pass

    @staticmethod
    def _qc(t: TaskManifest, out) -> dict:
        import torch
        if isinstance(out, (tuple, list)):                      # several outputs (e.g. video + audio of one tile)
            parts = [x for x in out if x is not None]
            if not parts or not all(isinstance(x, torch.Tensor) for x in parts):
                raise TaskQCError("the executor must return tensors")
            if any(x.is_floating_point() and not bool(torch.isfinite(x).all()) for x in parts):
                raise TaskQCError("non-finite values in the output")
            return {"shape_ok": t.expected_output_shape is None, "dtype_ok": True, "finite": True,
                    "parts": len(out)}
        if not isinstance(out, torch.Tensor):
            raise TaskQCError("the executor must return a tensor")
        if t.expected_output_shape is not None and tuple(out.shape) != tuple(t.expected_output_shape):
            raise TaskQCError(f"shape {tuple(out.shape)} != expected {tuple(t.expected_output_shape)}")
        if t.expected_dtype and str(out.dtype).replace("torch.", "") != t.expected_dtype:
            raise TaskQCError(f"dtype {out.dtype} != expected {t.expected_dtype}")
        if out.is_floating_point() and not bool(torch.isfinite(out).all()):
            raise TaskQCError("non-finite values in the output")
        return {"shape_ok": True, "dtype_ok": True, "finite": True}


_ids = itertools.count()


def new_job_id(prefix: str = "job") -> str:
    return f"{prefix}-{next(_ids)}-{uuid.uuid4().hex[:8]}"
