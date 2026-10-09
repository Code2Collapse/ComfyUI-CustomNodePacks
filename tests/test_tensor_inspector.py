"""L2.42: the tensor inspector's per-output statistics are exact and memory-bounded.

It summarises EVERY node output. The old whole-tensor reduction (`value[torch.isfinite(value)]`) peaked at 6.1 GB
for one 2K x 24 IMAGE batch (0.59 GB); now it reduces in chunks (docs/evidence/L7.59 in the work area).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

PACK = Path(__file__).resolve().parents[1]


@pytest.fixture
def ti(monkeypatch):
    if str(PACK) not in sys.path:
        sys.path.insert(0, str(PACK))
    import importlib

    mod = importlib.import_module("nodes.tensor_inspector")
    monkeypatch.setattr(mod, "_STAT_CHUNK", 1000)          # many chunks: exercises the merge arithmetic
    return mod


def _ref(x):
    finite = x[torch.isfinite(x)].double()
    return {"min": float(finite.min()), "max": float(finite.max()), "mean": float(finite.mean()),
            "std": float(finite.std()), "n_nan": int(torch.isnan(x).sum()), "n_inf": int(torch.isinf(x).sum())}


def test_float_stats_exact_with_nan_and_inf(ti):
    g = torch.Generator().manual_seed(3)
    x = torch.randn(7, 33, 41, 3, generator=g) * 4 + 1
    x[0, 0, 0, 0] = float("nan")
    x[3, 5, 7, 1] = float("inf")
    x[6, 1, 2, 2] = float("-inf")
    got, ref = ti._summarize_value(x), _ref(x)
    for k in ("min", "max", "mean", "std"):
        assert math.isclose(got[k], ref[k], rel_tol=1e-9, abs_tol=1e-9), k
    assert (got["n_nan"], got["n_inf"]) == (1, 2)
    assert got["shape"] == [7, 33, 41, 3] and got["kind"] == "tensor"


def test_strided_and_int_and_scalar_tensors(ti):
    x = torch.rand(5, 20, 30, 3).permute(0, 3, 1, 2)          # not contiguous
    assert not x.is_contiguous()
    got, ref = ti._summarize_value(x), _ref(x)
    assert math.isclose(got["mean"], ref["mean"], rel_tol=1e-9) and math.isclose(got["std"], ref["std"], rel_tol=1e-9)
    i = torch.arange(-50, 2950, dtype=torch.int64).view(30, 100)
    gi = ti._summarize_value(i)
    assert (gi["min"], gi["max"]) == (-50, 2949) and math.isclose(gi["mean"], 1449.5) and "n_nan" not in gi
    s = ti._summarize_value(torch.tensor(2.5))
    assert s["min"] == s["max"] == s["mean"] == 2.5 and "std" not in s


def test_all_nan_reports_no_range(ti):
    got = ti._summarize_value(torch.full((4, 4), float("nan")))
    assert got["n_nan"] == 16 and got["min"] is None and got["mean"] is None


def test_reduction_never_holds_more_than_a_chunk(ti):
    x = torch.rand(3, 50, 70)                                   # 10500 elements, chunk 1000
    sizes = [c.numel() for c in ti._flat_chunks(x)]
    assert max(sizes) <= 1000 and sum(sizes) == x.numel()
    y = x.permute(2, 0, 1)                                      # strided: one leading slice at a time
    sizes = [c.numel() for c in ti._flat_chunks(y)]
    assert max(sizes) <= 1000 and sum(sizes) == y.numel()
