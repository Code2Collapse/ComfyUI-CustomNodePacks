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


# ── L2.51: zero work per node at run time; exact statistics on request ──────────────────────────────────────────

def test_expanded_view_from_mask_to_image_is_exact_and_fast(ti):
    import time
    mask = (torch.rand(16, 180, 320, generator=torch.Generator().manual_seed(1)) > 0.5).float()
    view = mask.reshape((-1, 1, 180, 320)).movedim(1, -1).expand(-1, -1, -1, 3)   # what core Mask to Image returns
    assert not view.is_contiguous() and 0 in view.stride()
    t0 = time.perf_counter()
    got = ti._summarize_value(view)
    assert time.perf_counter() - t0 < 2.0                      # it used to walk every element in Python (> 60 s)
    ref = _ref(view.contiguous())
    for k in ("min", "max", "mean"):
        assert math.isclose(got[k], ref[k], rel_tol=1e-9, abs_tol=1e-12), k
    assert math.isclose(got["std"], ref["std"], rel_tol=1e-4)  # n-1 of the deduplicated copy: a 1e-6 relative change


def test_broadcast_nan_counts_are_of_the_full_tensor(ti):
    base = torch.zeros(4, 5, 1)
    base[1, 2, 0] = float("nan")
    view = base.expand(-1, -1, 3)
    assert ti._summarize_value(view)["n_nan"] == 3


def test_record_does_no_reductions_and_keeps_nothing_alive(ti):
    import gc
    import time
    import weakref
    big = torch.rand(97, 360, 640, 3)
    t0 = time.perf_counter()
    ti._record("7", "p1", [[big]])
    assert time.perf_counter() - t0 < 0.05
    slot = ti._STORE["7"]["slots"][0]
    assert slot["stats"] == "on request" and "mean" not in slot and slot["shape"] == [97, 360, 640, 3]
    data = ti.snapshot_with_stats("7")
    assert data["slots"][0]["stats"] == "exact"
    assert math.isclose(data["slots"][0]["mean"], float(big.double().mean()), rel_tol=1e-9)
    assert "mean" not in ti._STORE["7"]["slots"][0]            # the stored snapshot stays metadata only
    probe = weakref.ref(big)
    del big
    gc.collect()
    assert probe() is None, "the inspector must not keep node outputs alive"
    assert ti.snapshot_with_stats("7")["slots"][0]["stats"] == "no longer in memory"


def test_latent_dict_stats_on_request(ti):
    lat = {"samples": torch.ones(1, 4, 8, 8) * 2}
    ti._record("9", "p2", [[lat]])
    got = ti.snapshot_with_stats("9", 0)["slots"][0]
    assert got["kind"] == "dict" and got["samples"]["mean"] == 2.0 and got["samples"]["stats"] == "exact"


def test_kill_switch_installs_nothing(ti, monkeypatch):
    monkeypatch.setenv("C2C_TENSOR_INSPECTOR", "0")
    monkeypatch.setattr(ti, "_INSTALLED", False)
    assert ti.install() is False
