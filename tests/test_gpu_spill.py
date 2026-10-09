"""L7.59: GPU spill detection - peak above the card's size means the driver spilled into system RAM."""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

PACK = Path(__file__).resolve().parents[1]


@pytest.fixture
def gs(monkeypatch):
    if str(PACK) not in sys.path:
        sys.path.insert(0, str(PACK))
    import importlib
    return importlib.import_module("nodes._c2c_gpu_spill")


def _fake_torch(monkeypatch, total, peak, available=True):
    calls = {"reset": 0}
    cuda = types.SimpleNamespace(
        is_available=lambda: available, current_device=lambda: 0,
        get_device_properties=lambda d: types.SimpleNamespace(total_memory=total),
        max_memory_allocated=lambda d=0: peak, get_device_name=lambda d=0: "Fake GPU",
        reset_peak_memory_stats=lambda d=0: calls.__setitem__("reset", calls["reset"] + 1))
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(cuda=cuda))
    return calls


def test_spill_is_peak_above_physical(gs, monkeypatch):
    calls = _fake_torch(monkeypatch, total=8 * 2**30, peak=12.4 * 2**30)
    r = gs.peak_report(reset=True)
    assert r["spilled"] is True and r["total_gb"] == 8.0 and r["peak_gb"] == 12.4 and calls["reset"] == 1
    _fake_torch(monkeypatch, total=8 * 2**30, peak=7.9 * 2**30)
    assert gs.peak_report()["spilled"] is False


def test_no_cuda_reports_unavailable(gs, monkeypatch):
    _fake_torch(monkeypatch, 0, 0, available=False)
    assert gs.peak_report() == {"available": False}
