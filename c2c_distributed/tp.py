"""Tensor-parallel primitives (ledger L9.216 milestone 1, A26 4.7). Design: docs/research/L9.216_model_sharding_
design.md (workspace).

MODEL-WEIGHT SHARDING, not sequence parallelism and not full replicas: each rank keeps 1/N of a layer's weights.
- column-parallel: rank r keeps slice r of the OUTPUT rows; ``chunks`` > 1 splits a concatenated output (q|k|v or
  SwiGLU gate|up) chunk by chunk, so every rank holds matching slices of each part (whole heads when the slice is a
  multiple of head_dim);
- row-parallel: rank r keeps slice r of the INPUT columns; partial results are summed with one all-reduce, the bias is
  added once after the sum;
- a plan is DATA: {module-name pattern: (rule, options)}; ``scale_attr`` divides an attribute (e.g. the head count) by
  N. Models ship their own plan; this module knows no model.

Milestone 1 proves exactness (one H3-shaped block, TP=2 == unsplit) in-process and over real gloo processes; NCCL on a
multi-GPU Linux box is milestone 2 / 3 (hardware).
"""
from __future__ import annotations

import fnmatch
from typing import Callable, Optional

import torch
import torch.nn as nn


def _slice_out(w: torch.Tensor, rank: int, n: int, chunks: int) -> torch.Tensor:
    out = w.shape[0]
    if out % chunks or (out // chunks) % n:
        raise ValueError(f"output size {out} does not split into {chunks} chunk(s) x {n} ranks")
    per, k = out // chunks, out // chunks // n
    return w.reshape(chunks, per, *w.shape[1:])[:, rank * k:(rank + 1) * k].reshape(chunks * k, *w.shape[1:])


def _slice_in(w: torch.Tensor, rank: int, n: int) -> torch.Tensor:
    if w.shape[1] % n:
        raise ValueError(f"input size {w.shape[1]} does not split over {n} ranks")
    k = w.shape[1] // n
    return w[:, rank * k:(rank + 1) * k]


class ColumnParallelLinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, n: int, chunks: int = 1):
        super().__init__()
        self.weight = nn.Parameter(_slice_out(base.weight.detach(), rank, n, chunks).clone(), requires_grad=False)
        self.bias = None if base.bias is None else nn.Parameter(
            _slice_out(base.bias.detach()[:, None], rank, n, chunks)[:, 0].clone(), requires_grad=False)
        self.rank, self.n, self.chunks = rank, n, chunks

    def forward(self, x):
        return nn.functional.linear(x, self.weight.to(x.dtype), None if self.bias is None else self.bias.to(x.dtype))


class RowParallelLinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, n: int, all_reduce: Callable[[torch.Tensor], torch.Tensor]):
        super().__init__()
        self.weight = nn.Parameter(_slice_in(base.weight.detach(), rank, n).clone(), requires_grad=False)
        self.bias = None if base.bias is None else nn.Parameter(base.bias.detach().clone(), requires_grad=False)
        self.rank, self.n, self.all_reduce = rank, n, all_reduce

    def forward(self, x):
        y = self.all_reduce(nn.functional.linear(x, self.weight.to(x.dtype)))
        return y if self.bias is None else y + self.bias.to(y.dtype)


def apply_plan(model: nn.Module, plan: dict, rank: int, n: int, all_reduce: Callable) -> dict:
    """Shard ``model`` IN PLACE for ``rank`` of ``n``. Returns {rule: [module names]} (what was done)."""
    done: dict = {"column": [], "row": [], "scale_attr": [], "replicate": []}
    targets = []
    for name, mod in model.named_modules():
        for pattern, (rule, opts) in plan.items():
            if fnmatch.fnmatchcase(name, pattern):
                targets.append((name, mod, rule, opts or {}))
                break
    for name, mod, rule, opts in targets:
        parent_name, _, leaf = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        if rule == "column":
            if not isinstance(mod, nn.Linear):
                raise TypeError(f"{name}: column-parallel needs a Linear, got {type(mod).__name__}")
            setattr(parent, leaf, ColumnParallelLinear(mod, rank, n, int(opts.get("chunks", 1))))
        elif rule == "row":
            if not isinstance(mod, nn.Linear):
                raise TypeError(f"{name}: row-parallel needs a Linear, got {type(mod).__name__}")
            setattr(parent, leaf, RowParallelLinear(mod, rank, n, all_reduce))
        elif rule == "scale_attr":
            for attr in opts.get("attrs", ()):
                v = getattr(mod, attr)
                if v % n:
                    raise ValueError(f"{name}.{attr} = {v} does not split over {n} ranks")
                setattr(mod, attr, v // n)
        elif rule != "replicate":
            raise ValueError(f"unknown sharding rule {rule!r}")
        done[rule].append(name)
    return done


def param_bytes(model: nn.Module) -> int:
    return sum(p.numel() * p.element_size() for p in model.parameters())


class ThreadAllReduce:
    """An in-process all-reduce for N rank threads (tests and single-box simulation): sums the ranks' tensors in RANK
    ORDER at a barrier, so the result is deterministic. Not a performance path."""

    def __init__(self, n: int):
        import threading
        self.n = n
        self._barrier = threading.Barrier(n)
        self._slots: list = [None] * n
        self._result: Optional[torch.Tensor] = None
        self._lock = threading.Lock()

    def for_rank(self, rank: int) -> Callable[[torch.Tensor], torch.Tensor]:
        def reduce(t: torch.Tensor) -> torch.Tensor:
            self._slots[rank] = t
            if self._barrier.wait() == 0:
                acc = self._slots[0].clone()
                for s in self._slots[1:]:
                    acc = acc + s
                self._result = acc
            self._barrier.wait()
            out = self._result.clone()
            self._barrier.wait()
            return out
        return reduce


def torch_all_reduce(group=None) -> Callable[[torch.Tensor], torch.Tensor]:
    """all-reduce (sum) over a torch.distributed process group."""
    import torch.distributed as dist

    def reduce(t: torch.Tensor) -> torch.Tensor:
        t = t.contiguous()
        dist.all_reduce(t, group=group)
        return t
    return reduce
