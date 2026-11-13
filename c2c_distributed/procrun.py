"""Multi-process ranks for C2C Distributed: the Linux / NCCL path (A27 GPU-008).

The C2C Multi-GPU Split node runs its ranks as THREADS of the ComfyUI process (comm.ThreadComm): that works on
Windows, needs no NCCL, and moves tensors with device copies. Server GPUs are normally driven the other way - one
PROCESS per GPU talking through NCCL (raylight does this with Ray actors). This module is that second path, without
Ray: plain worker processes and torch.distributed.

Design
------
* ``ProcessRanks(world)`` makes a fresh temp directory with two file stores: one for torch.distributed's rendezvous
  and one for COMMANDS. It starts ``world - 1`` worker processes (``python -c`` + this package on ``sys.path``; no
  ``multiprocessing`` spawn, which would re-import ComfyUI's ``main.py`` in the child). The calling process is
  rank 0.
* Backend: NCCL when the build has it and the ranks are CUDA devices (rank r on ``cuda:r``), otherwise gloo on CPU.
  ``backend=`` / ``device_kind=`` override that. The choice is reported, never hidden.
* ``run(fn, payload)`` runs ``fn(comm, payload)`` on EVERY rank at once. ``comm`` is a ``comm.TorchComm``, so the
  exact Ulysses / Ring / USP code in attention.py runs unchanged across processes. ``fn`` must be a module-level
  function of this package (workers import it by name). The payload is pickled into the command store, and each
  rank's return value comes back the same way, so every rank's result returns to rank 0.
* Idle workers wait on the command store, not inside a collective, so being idle never trips a collective timeout.
* Failure handling: the process group gets ``timeout_s``, so a hung collective raises instead of blocking. A worker
  that dies or stops answering is detected (exit code or missing result within ``timeout_s``). After ANY failure the
  group is treated as corrupted: every worker is killed, the group is destroyed, and the object refuses further
  work. A new ``ProcessRanks`` is the recovery.
* Cleanup: ``close()`` sends "stop", waits, kills stragglers, destroys the process group and deletes the temp
  directory. It also runs at interpreter exit and through ``with``. Workers exit on their own when the parent
  process disappears, so no worker is ever orphaned.

Status: the code is complete. Verified with gloo on CPU ranks under Linux (WSL), which counts as SIMULATED. The NCCL
path on real GPUs is CODE_COMPLETE_HARDWARE_UNVERIFIED until a machine with two or more GPUs runs ``selftest``.
"""
from __future__ import annotations

import atexit
import datetime
import importlib
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from typing import Any, Callable, List, Optional

import torch

_PKG = "c2c_distributed"
_PKG_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_WORKER = r"""
import datetime, os, pickle, sys, time, traceback
rank, world, root, backend, kind, timeout_s, parent, pkg_parent = (int(sys.argv[1]), int(sys.argv[2]), sys.argv[3],
    sys.argv[4], sys.argv[5], float(sys.argv[6]), int(sys.argv[7]), sys.argv[8])
sys.path.insert(0, pkg_parent)
import torch, torch.distributed as dist
from c2c_distributed import procrun
cmds = dist.FileStore(os.path.join(root, "cmds"), -1)
def alive(pid):
    if os.name == "posix":
        return os.getppid() == pid
    try:
        import psutil
        return psutil.pid_exists(pid)
    except Exception:
        return True
try:
    procrun._init(rank, world, root, backend, kind, timeout_s)
    cmds.set(f"ready/{rank}", b"1")
except Exception as exc:
    cmds.set(f"ready/{rank}", pickle.dumps(("err", f"{type(exc).__name__}: {exc}")))
    os._exit(3)
seq = 0
while True:
    key = f"cmd/{seq}"
    try:
        cmds.wait([key], datetime.timedelta(seconds=2))
    except Exception:
        if not alive(parent):
            os._exit(0)
        continue
    cmd = pickle.loads(cmds.get(key))
    if cmd["op"] == "stop":
        break
    try:
        out = ("ok", procrun._resolve(cmd["fn"])(procrun.comm(), cmd["payload"]))
    except BaseException as exc:
        out = ("err", f"{type(exc).__name__}: {exc}", traceback.format_exc()[-2000:])
    try:
        blob = pickle.dumps(out)
    except Exception as exc:
        blob = pickle.dumps(("err", f"result of rank {rank} cannot be pickled: {exc}", ""))
    cmds.set(f"res/{seq}/{rank}", blob)
    if out[0] != "ok":
        os._exit(4)             # a failed rank leaves: its peers see the closed connection at once, not at the timeout
    seq += 1
try:
    import torch.distributed as dist
    dist.destroy_process_group()
except Exception:
    pass
os._exit(0)
"""


# ── shared by the parent and the workers ───────────────────────────────────────────────────────────────────────

def _init(rank: int, world: int, root: str, backend: str, kind: str, timeout_s: float) -> None:
    import torch.distributed as dist
    if kind == "cuda":
        torch.cuda.set_device(rank)
    dist.init_process_group(backend, init_method="file:///" + os.path.join(root, "rdzv").replace(os.sep, "/"),
                            rank=rank, world_size=world, timeout=datetime.timedelta(seconds=timeout_s))


def comm():
    from .comm import TorchComm
    return TorchComm()


def device() -> torch.device:
    """This rank's device (cuda:rank for CUDA ranks, else cpu)."""
    import torch.distributed as dist
    if dist.get_backend() == "nccl":
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def _ref(fn: Callable) -> str:
    mod = fn.__module__
    i = mod.find(_PKG + ".")
    if i < 0 and mod != _PKG:
        raise ValueError(f"{fn.__module__}.{fn.__qualname__}: rank functions must live in the {_PKG} package "
                         f"(the worker processes import them by name)")
    if "<locals>" in fn.__qualname__:
        raise ValueError(f"{fn.__qualname__}: rank functions must be module-level")
    return f"{mod[i:] if i >= 0 else mod}:{fn.__qualname__}"


def _resolve(ref: str) -> Callable:
    mod, _, name = ref.partition(":")
    obj: Any = importlib.import_module(mod)
    for part in name.split("."):
        obj = getattr(obj, part)
    return obj


def choose(world: int, backend: Optional[str] = None, device_kind: Optional[str] = None):
    """(backend, device kind, reason) - NCCL on CUDA ranks when possible, else gloo on CPU."""
    import torch.distributed as dist
    if not dist.is_available():
        raise RuntimeError("this torch build has no torch.distributed")
    ngpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if device_kind is None:
        device_kind = "cuda" if ngpu >= world and dist.is_nccl_available() and backend in (None, "nccl") else "cpu"
    if device_kind == "cuda" and ngpu < world:
        raise RuntimeError(f"{world} CUDA ranks need {world} visible GPUs; {ngpu} visible")
    if backend is None:
        backend = "nccl" if device_kind == "cuda" and dist.is_nccl_available() else "gloo"
    if backend == "nccl" and not dist.is_nccl_available():
        raise RuntimeError("NCCL is not in this torch build (Windows wheels have none): use backend='gloo' on Linux "
                           "for CPU ranks, or the thread path (C2C Multi-GPU Split)")
    reason = f"{backend} on {device_kind} ranks" + ("" if device_kind == "cuda" else
                                                     " (CPU ranks: correctness only - SIMULATED, not a GPU result)")
    return backend, device_kind, reason


class RankFailure(RuntimeError):
    """A rank failed, died or stalled; the group was torn down."""


class ProcessRanks:
    """``world`` ranks: this process is rank 0, the others are worker processes. See the module docstring."""

    def __init__(self, world: int, backend: Optional[str] = None, device_kind: Optional[str] = None,
                 timeout_s: float = 600.0, init_timeout_s: float = 120.0):
        if world < 2:
            raise ValueError("ProcessRanks needs at least two ranks")
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            raise RuntimeError("torch.distributed is already initialised in this process; one ProcessRanks at a time")
        self.world, self.timeout_s, self.init_timeout_s = int(world), float(timeout_s), float(init_timeout_s)
        self.backend, self.device_kind, self.reason = choose(self.world, backend, device_kind)
        self.root = tempfile.mkdtemp(prefix="c2c_procrun_")
        self.procs: List[subprocess.Popen] = []
        self.seq = 0
        self.broken: Optional[str] = None
        self.closed = False
        self._cmds = None
        atexit.register(self.close)

    # ---- lifetime ------------------------------------------------------------------------------------------------
    def start(self) -> "ProcessRanks":
        import torch.distributed as dist
        self._cmds = dist.FileStore(os.path.join(self.root, "cmds"), -1)
        env = dict(os.environ)
        env.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")    # NCCL: a hung collective raises after the timeout
        for r in range(1, self.world):
            self.procs.append(subprocess.Popen(
                [sys.executable, "-c", _WORKER, str(r), str(self.world), self.root, self.backend, self.device_kind,
                 str(self.timeout_s), str(os.getpid()), _PKG_PARENT],
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, cwd=tempfile.gettempdir(), env=env))
        try:
            _init(0, self.world, self.root, self.backend, self.device_kind, self.timeout_s)
            deadline = time.monotonic() + self.init_timeout_s
            for r in range(1, self.world):
                self._wait_key(f"ready/{r}", r, deadline)
                v = self._cmds.get(f"ready/{r}")
                if v != b"1":
                    raise RankFailure(f"rank {r} could not join: {pickle.loads(v)[1]}")
        except BaseException as exc:
            self._teardown(f"start failed: {exc}")
            raise RankFailure(f"the {self.reason} group did not start: {exc}") from exc
        return self

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.close()

    def _wait_key(self, key: str, rank: int, deadline: float) -> None:
        while True:
            try:
                self._cmds.wait([key], datetime.timedelta(seconds=0.5))
                return
            except Exception:  # noqa: BLE001 - not there yet
                pass
            p = self.procs[rank - 1]
            if p.poll() is not None:
                err = (p.stderr.read() if p.stderr else b"").decode(errors="replace")[-600:]
                raise RankFailure(f"rank {rank} exited with code {p.returncode}" + (f": {err}" if err else ""))
            if time.monotonic() > deadline:
                raise RankFailure(f"rank {rank} gave no answer within {self.timeout_s:.0f} s (stalled)")

    # ---- work ----------------------------------------------------------------------------------------------------
    def run(self, fn: Callable, payload: Any = None, timeout_s: Optional[float] = None) -> list:
        """fn(comm, payload) on every rank at once -> [result of rank 0, rank 1, ...]."""
        if self.closed or self.broken:
            raise RankFailure(f"this rank group is not usable ({self.broken or 'closed'}); start a new one")
        if self._cmds is None:
            raise RuntimeError("call start() first")
        seq = self.seq
        self.seq += 1
        self._cmds.set(f"cmd/{seq}", pickle.dumps({"op": "run", "fn": _ref(fn), "payload": payload}))
        results: list = [None] * self.world
        try:
            try:
                results[0] = fn(comm(), payload)
            except BaseException as exc:
                cause = self._worker_error(seq)     # usually rank 0 only saw a peer leave: name the real cause
                raise RankFailure(cause or f"rank 0 failed: {type(exc).__name__}: {exc}") from exc
            deadline = time.monotonic() + float(timeout_s or self.timeout_s)
            for r in range(1, self.world):
                self._wait_key(f"res/{seq}/{r}", r, deadline)
                out = pickle.loads(self._cmds.get(f"res/{seq}/{r}"))
                if out[0] != "ok":
                    raise RankFailure(f"rank {r} failed: {out[1]}\n{out[2] if len(out) > 2 else ''}")
                results[r] = out[1]
        except BaseException as exc:
            self._teardown(str(exc).splitlines()[0][:300])
            raise
        return results

    _KNOCK_ON = ("Connection closed", "Connection reset", "Timed out", "timed out", "gloo/transport", "NCCL error",
                 "ProcessGroupNCCL", "Broken pipe")

    def _worker_error(self, seq: int) -> Optional[str]:
        """The root cause among the workers: a rank's own error beats a peer's 'connection closed' knock-on."""
        time.sleep(0.3)                             # a failing worker writes its result just before it exits
        found: List[str] = []
        for r in range(1, self.world):
            key = f"res/{seq}/{r}"
            try:
                if self._cmds.check([key]):
                    out = pickle.loads(self._cmds.get(key))
                    if out[0] != "ok":
                        found.append(f"rank {r} failed: {out[1]}")
                        continue
            except Exception:  # noqa: BLE001
                pass
            p = self.procs[r - 1]
            if p.poll() is not None and p.returncode not in (0, 4):
                found.append(f"rank {r} exited with code {p.returncode} during the run")
        own = [f for f in found if not any(k in f for k in self._KNOCK_ON)]
        return (own or found or [None])[0]

    # ---- cleanup -------------------------------------------------------------------------------------------------
    def _teardown(self, why: str) -> None:
        """After a failure: the group's state is unknown, so nothing is reused."""
        self.broken = self.broken or why
        for p in self.procs:
            if p.poll() is None:
                p.kill()
        for p in self.procs:
            try:
                p.wait(10)
            except Exception:  # noqa: BLE001
                pass
        self._destroy()

    def _destroy(self) -> None:
        try:
            import torch.distributed as dist
            if dist.is_initialized():
                dist.destroy_process_group()
        except Exception:  # noqa: BLE001
            pass

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        if self._cmds is not None and not self.broken:
            try:
                self._cmds.set(f"cmd/{self.seq}", pickle.dumps({"op": "stop"}))
            except Exception:  # noqa: BLE001
                pass
        for p in self.procs:
            try:
                p.wait(15)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait(10)
        self._destroy()
        shutil.rmtree(self.root, ignore_errors=True)
        try:
            atexit.unregister(self.close)
        except Exception:  # noqa: BLE001
            pass

    def alive(self) -> List[bool]:
        return [p.poll() is None for p in self.procs]


# ── rank functions for the self-test (module-level: workers import them by name) ─────────────────────────────────

def rank_collectives(c, payload):
    """all_gather, uneven all_to_all, ring step and broadcast with known answers."""
    dev = device()
    r, w = c.rank, c.world
    got = c.all_gather(torch.full((3,), float(r), device=dev))
    ok_gather = all(bool((t == i).all()) for i, t in enumerate(got))
    chunks = [torch.full((j + 1, 2), float(r * 10 + j), device=dev) for j in range(w)]
    back = c.all_to_all(chunks)
    ok_a2a = all(tuple(t.shape) == (r + 1, 2) and bool((t == s * 10 + r).all()) for s, t in enumerate(back))
    ring = c.sendrecv_ring(torch.full((4,), float(r), device=dev))
    ok_ring = bool((ring == (r - 1) % w).all())
    b = c.broadcast(torch.full((2,), float(r) + 7, device=dev), src=0)
    ok_b = bool((b == 7).all())
    return {"all_gather": ok_gather, "all_to_all": ok_a2a, "ring": ok_ring, "broadcast": ok_b}


def rank_usp_attention(c, payload):
    """Ulysses x Ring attention over processes on this rank's shard; gathered back to the full sequence."""
    from . import attention as A
    q, k, v = (t.to(device()) for t in payload["qkv"])
    mesh = A.Mesh.build(c, payload["ulysses"], payload["ring"])
    ql, kl, vl = (A.shard(t, c.rank, c.world) for t in (q, k, v))
    mask = payload.get("mask")
    out = A.unshard(c, A.usp_attention(mesh, ql, kl, vl, mask=None if mask is None else mask.to(q.device)))
    return out.cpu()


def rank_fail(c, payload):
    if c.rank == payload["rank"]:
        raise ValueError("deliberate failure")
    c.barrier()                     # the others wait in a collective: the failure must not hang them for ever
    return "unreachable"


def rank_exit(c, payload):
    if c.rank == payload["rank"]:
        os._exit(9)                 # a rank that dies mid-run
    c.barrier()
    return "unreachable"


def selftest(world: int = 2, backend: Optional[str] = None, device_kind: Optional[str] = None,
             timeout_s: float = 20.0) -> dict:
    """Run the whole check list; returns a JSON-able report with an honest label."""
    from . import attention as A
    report: dict = {"world": world, "torch": torch.__version__, "platform": sys.platform}
    t0 = time.perf_counter()
    with ProcessRanks(world, backend, device_kind, timeout_s=timeout_s) as pr:
        report.update(backend=pr.backend, device_kind=pr.device_kind, path=pr.reason,
                      start_s=round(time.perf_counter() - t0, 2))
        report["collectives"] = pr.run(rank_collectives)
        g = torch.Generator().manual_seed(7)
        heads = 4 * world
        qkv = [torch.randn(2, heads, 37, 16, generator=g, dtype=torch.float64) for _ in range(3)]
        want = A.full_attention(*qkv)
        layouts = [(world, 1), (1, world)] + ([(2, world // 2)] if world % 2 == 0 and world > 2 else [])
        usp = {}
        for u, r in layouts:
            outs = pr.run(rank_usp_attention, {"qkv": qkv, "ulysses": u, "ring": r})
            usp[f"ulysses{u}xring{r}"] = max(float((o - want).abs().max()) for o in outs)
        report["usp_max_abs_err_float64"] = usp
        pids = [p.pid for p in pr.procs]
    report["workers_exited_after_close"] = all(not _pid_alive(p) for p in pids)
    # failure handling: a raising rank and a dying rank are both reported, never a hang, never orphans
    for name, fn in (("rank_raises", rank_fail), ("rank_dies", rank_exit)):
        pr = ProcessRanks(world, backend, device_kind, timeout_s=timeout_s).start()
        pids = [p.pid for p in pr.procs]
        t1 = time.perf_counter()
        try:
            pr.run(fn, {"rank": world - 1})
            report[name] = {"detected": False}
        except RankFailure as exc:
            report[name] = {"detected": True, "seconds": round(time.perf_counter() - t1, 2),
                            "message": str(exc).splitlines()[0][:200]}
        try:
            pr.run(rank_collectives)
            report[name]["refuses_reuse"] = False
        except RankFailure:
            report[name]["refuses_reuse"] = True
        pr.close()
        report[name]["no_orphans"] = all(not _pid_alive(p) for p in pids)
    ok = (all(all(v.values()) for v in report["collectives"])
          and all(e < 1e-12 for e in usp.values())
          and report["workers_exited_after_close"]
          and all(report[n]["detected"] and report[n]["refuses_reuse"] and report[n]["no_orphans"]
                  for n in ("rank_raises", "rank_dies")))
    report["passed"] = ok
    report["label"] = ("MULTI_GPU_VERIFIED" if ok and report["device_kind"] == "cuda" and _distinct_gpus(world)
                       else "SIMULATED" if report["device_kind"] == "cpu" else "FAILED" if not ok else
                       "CUDA ranks without distinct physical GPUs: not verified")
    return report


def _distinct_gpus(world: int) -> bool:
    try:
        uuids = {str(torch.cuda.get_device_properties(i).uuid) for i in range(world)}
        return len(uuids) == world
    except Exception:  # noqa: BLE001
        return False


def _pid_alive(pid: int) -> bool:
    if os.name == "posix":
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        try:                        # a zombie child (exited, not reaped) is not running
            with open(f"/proc/{pid}/stat", encoding="utf-8") as fh:
                return fh.read().split(")")[-1].split()[0] != "Z"
        except OSError:
            return True
    try:
        import psutil
        return psutil.pid_exists(pid)
    except Exception:  # noqa: BLE001
        return False


if __name__ == "__main__":
    import argparse
    import json
    ap = argparse.ArgumentParser(description="C2C multi-process rank self-test (gloo CPU / NCCL GPU)")
    ap.add_argument("--world", type=int, default=2)
    ap.add_argument("--backend", default=None)
    ap.add_argument("--device-kind", default=None, choices=[None, "cpu", "cuda"])
    ap.add_argument("--timeout", type=float, default=20.0)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    sys.path.insert(0, _PKG_PARENT)
    from c2c_distributed import procrun as _pkg     # rank functions must carry the package name, not __main__
    rep = _pkg.selftest(a.world, a.backend, a.device_kind, a.timeout)
    text = json.dumps(rep, indent=1, default=str)
    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            fh.write(text)
    print(text)
    raise SystemExit(0 if rep["passed"] else 1)
