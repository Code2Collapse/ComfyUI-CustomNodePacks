"""Collective communication for C2C Distributed, behind one small interface.

The parallel algorithms (attention.py, cfg.py) are written against `Comm` only, so the same code runs:

* `TorchComm`  - torch.distributed (NCCL on Linux GPU boxes; gloo where it works),
* `ThreadComm` - ranks as threads in one process: exact, CPU, any OS. Used by the tests and by anything that wants
                 the maths without a process group (on Windows torch 2.9 has no usable gloo transport and no NCCL -
                 measured 2026-10-24, docs/c2c-distributed.md).

Only what the algorithms need: all_gather, all_to_all (lists, uneven sizes allowed), send / recv (for the ring),
broadcast, barrier.
"""
from __future__ import annotations

import queue
import threading
from typing import Callable, List, Optional

import torch


class Comm:
    """One rank's view of a group."""

    rank: int
    world: int

    def all_gather(self, t: torch.Tensor) -> List[torch.Tensor]:
        raise NotImplementedError

    def all_to_all(self, chunks: List[torch.Tensor]) -> List[torch.Tensor]:
        """chunks[j] goes to rank j; returns what every rank sent to this one, by source rank."""
        raise NotImplementedError

    def send(self, t: torch.Tensor, dst: int) -> None:
        raise NotImplementedError

    def recv(self, like: torch.Tensor, src: int) -> torch.Tensor:
        raise NotImplementedError

    def broadcast(self, t: torch.Tensor, src: int = 0) -> torch.Tensor:
        raise NotImplementedError

    def barrier(self) -> None:
        raise NotImplementedError

    def split(self, groups: List[List[int]]) -> "Comm":
        """Sub-groups: every rank passes the SAME full list; returns the comm of the group holding this rank."""
        raise NotImplementedError

    def sendrecv_ring(self, t: torch.Tensor) -> torch.Tensor:
        """Send `t` to rank+1 and receive rank-1's tensor of the same shape (one ring step)."""
        nxt, prv = (self.rank + 1) % self.world, (self.rank - 1) % self.world
        if self.world == 1:
            return t
        if self.rank % 2 == 0:              # even ranks send first: no deadlock on blocking backends
            self.send(t, nxt)
            return self.recv(t, prv)
        out = self.recv(t, prv)
        self.send(t, nxt)
        return out


# ── in-process ranks ────────────────────────────────────────────────────────────────────────────────────────

class GroupAborted(RuntimeError):
    """Another rank failed; this rank's collective was cancelled."""


STEP_TIMEOUT_S = 600.0          # a collective waiting longer than this means a rank stalled (design: watchdog)


class ThreadGroup:
    """Ranks as threads of one process. Each rank may own a device; tensors move between devices on receipt
    (cudaMemcpyPeer over NVLink / PCIe - no NCCL or gloo needed, so this works on Windows too)."""

    def __init__(self, world: int, timeout: float = STEP_TIMEOUT_S):
        if world < 1:
            raise ValueError("a group needs at least one rank")
        self.world, self.timeout = world, float(timeout)
        self.barrier = threading.Barrier(world, timeout=self.timeout)
        self.slots: List[object] = [None] * world
        self.pipes = {(s, d): queue.Queue() for s in range(world) for d in range(world)}
        self.lock = threading.Lock()
        self.subgroups: dict = {}
        self.aborted = False

    def subgroup(self, ranks) -> "ThreadGroup":
        with self.lock:
            key = tuple(ranks)
            if key not in self.subgroups:
                self.subgroups[key] = ThreadGroup(len(key), self.timeout)
                if self.aborted:
                    self.subgroups[key].abort()
            return self.subgroups[key]

    def abort(self):
        """Release every rank blocked in this group or any sub-group (they raise GroupAborted)."""
        self.aborted = True
        self.barrier.abort()
        with self.lock:
            subs = list(self.subgroups.values())
        for g in subs:
            g.abort()


class ThreadComm(Comm):
    def __init__(self, group: ThreadGroup, rank: int, device=None):
        self.group, self.rank, self.world = group, int(rank), group.world
        self.device = torch.device(device) if device is not None else None

    def _own(self, t: torch.Tensor) -> torch.Tensor:
        if self.device is not None and t.device != self.device:
            return t.to(self.device)
        return t.clone()

    def _wait(self):
        try:
            self.group.barrier.wait()
        except threading.BrokenBarrierError:
            raise GroupAborted("another rank failed or stalled; collective cancelled") from None

    def split(self, groups):
        _check_groups(groups, self.world)
        mine = next(g for g in groups if self.rank in g)
        return ThreadComm(self.group.subgroup(mine), list(mine).index(self.rank), self.device)

    def _exchange(self, value):
        g = self.group
        g.slots[self.rank] = value
        self._wait()
        everyone = list(g.slots)
        self._wait()                        # nobody overwrites a slot before all have read
        return everyone

    def all_gather(self, t):
        return [self._own(x) for x in self._exchange(t)]

    def all_to_all(self, chunks):
        if len(chunks) != self.world:
            raise ValueError(f"all_to_all needs {self.world} chunks, got {len(chunks)}")
        everyone = self._exchange(list(chunks))
        return [self._own(everyone[src][self.rank]) for src in range(self.world)]

    def send(self, t, dst):
        self.group.pipes[(self.rank, int(dst))].put(t)

    def recv(self, like, src):
        q = self.group.pipes[(int(src), self.rank)]
        waited = 0.0
        while True:
            if self.group.aborted:
                raise GroupAborted("another rank failed or stalled; receive cancelled")
            try:
                return self._own(q.get(timeout=0.25))
            except queue.Empty:
                waited += 0.25
                if waited >= self.group.timeout:
                    self.group.abort()
                    raise GroupAborted(f"rank {src} sent nothing for {self.group.timeout:.0f} s") from None

    def broadcast(self, t, src=0):
        return self._own(self._exchange(t)[int(src)])

    def barrier(self):
        self._wait()


def run_ranks(world: int, fn: Callable[[Comm], object], devices: Optional[list] = None,
              submit: Optional[Callable] = None, timeout: float = STEP_TIMEOUT_S) -> list:
    """Run fn(comm) on `world` ranks; returns every rank's result, re-raising the first real failure.

    Rank 0 runs in the calling thread (it is already on its device). The others run on `submit(rank, job)` when
    given (e.g. ComfyUI's per-device MultiGPUThreadPool, so kernel caches stay warm), else on fresh threads.
    """
    group = ThreadGroup(world, timeout)
    results: list = [None] * world
    errors: list = [None] * world
    done = [threading.Event() for _ in range(world)]

    def body(r):
        try:
            results[r] = fn(ThreadComm(group, r, devices[r] if devices else None))
        except BaseException as exc:  # noqa: BLE001 - reported below
            errors[r] = exc
            group.abort()                   # release ranks stuck in a collective
        finally:
            done[r].set()

    threads = []
    for r in range(1, world):
        if submit is not None:
            submit(r, lambda r=r: body(r))
        else:
            t = threading.Thread(target=body, args=(r,), daemon=True, name=f"c2c-rank{r}")
            t.start()
            threads.append(t)
    body(0)
    for r in range(1, world):
        done[r].wait(timeout + 60)
    real = [(r, e) for r, e in enumerate(errors) if e is not None and not isinstance(e, GroupAborted)]
    for r, e in real or [(r, e) for r, e in enumerate(errors) if e is not None]:
        if r == 0 and not isinstance(e, GroupAborted):
            raise e                         # rank 0's own exception (e.g. ComfyUI's interrupt) keeps its type
        raise RuntimeError(f"rank {r} failed: {type(e).__name__}: {e}") from e
    missing = [r for r in range(world) if not done[r].is_set()]
    if missing:
        raise RuntimeError(f"ranks {missing} did not finish")
    return results


# ── torch.distributed ───────────────────────────────────────────────────────────────────────────────────────

class TorchComm(Comm):
    """torch.distributed with an initialised default (or given) process group."""

    def __init__(self, group=None):
        import torch.distributed as dist
        if not dist.is_initialized():
            raise RuntimeError("torch.distributed is not initialised - start the C2C Distributed pool first.")
        self.dist, self.group = dist, group
        self.rank, self.world = dist.get_rank(group), dist.get_world_size(group)

    def all_gather(self, t):
        out = [torch.empty_like(t) for _ in range(self.world)]
        self.dist.all_gather(out, t.contiguous(), group=self.group)
        return out

    def all_to_all(self, chunks):
        shapes = self._shapes_of(chunks)
        outs = [torch.empty(shapes[src], dtype=chunks[0].dtype, device=chunks[0].device) for src in range(self.world)]
        self.dist.all_to_all(outs, [c.contiguous() for c in chunks], group=self.group)
        return outs

    def _shapes_of(self, chunks):
        """Every rank's chunk shape for this rank (uneven splits): exchanged once as a small tensor."""
        mine = torch.tensor([list(c.shape) for c in chunks], dtype=torch.long, device=chunks[0].device)
        allshapes = self.all_gather(mine)                      # [world] x [world, ndim]
        return [tuple(int(v) for v in allshapes[src][self.rank]) for src in range(self.world)]

    def send(self, t, dst):
        self.dist.send(t.contiguous(), dst, group=self.group)

    def recv(self, like, src):
        out = torch.empty_like(like)
        self.dist.recv(out, src, group=self.group)
        return out

    def sendrecv_ring(self, t):
        if self.world == 1:
            return t
        nxt, prv = (self.rank + 1) % self.world, (self.rank - 1) % self.world
        out = torch.empty_like(t)
        ops = [self.dist.P2POp(self.dist.isend, t.contiguous(), nxt, self.group),
               self.dist.P2POp(self.dist.irecv, out, prv, self.group)]
        for req in self.dist.batch_isend_irecv(ops):
            req.wait()
        return out

    def broadcast(self, t, src=0):
        t = t.contiguous().clone()
        self.dist.broadcast(t, src, group=self.group)
        return t

    def barrier(self):
        self.dist.barrier(group=self.group)

    _groups: dict = {}

    def split(self, groups):
        _check_groups(groups, self.world)
        mine = None
        for g in groups:                    # new_group is collective: every rank creates every group, same order
            key = (id(self.group), tuple(int(r) for r in g))
            if key not in TorchComm._groups:
                glob = [self.dist.get_global_rank(self.group, r) for r in key[1]] if self.group is not None else list(key[1])
                TorchComm._groups[key] = self.dist.new_group(glob)
            if self.rank in key[1]:
                mine = TorchComm._groups[key]
        return TorchComm(mine)


def _check_groups(groups, world):
    flat = sorted(int(r) for g in groups for r in g)
    if flat != list(range(world)):
        raise ValueError(f"groups must cover ranks 0..{world - 1} exactly once, got {groups}")


def local_comm() -> Optional[Comm]:
    """The torch.distributed comm when a group is up in this process, else None."""
    try:
        return TorchComm()
    except Exception:  # noqa: BLE001
        return None
