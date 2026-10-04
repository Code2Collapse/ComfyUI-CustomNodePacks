"""The integrity badge's health poll must never stall the ComfyUI server.

Measured 2026-09-27 on the isolated test server: /system_stats (6 ms normally)
took 20-60 s. py-spy showed the main thread inside os.walk, called from
POST /c2c/int/health -> aggregate() -> c2c_doctor.collect_disk(), on the
aiohttp event loop. The badge polls every 4 s, and this box's output folder
holds ~1,000,000 sub-folders, so the whole server - queueing, the progress
websocket, every route - froze for most of every minute.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import threading
import time
import types
from pathlib import Path

import pytest

NODES = Path(__file__).resolve().parents[1] / "nodes"


def _load(name):
    spec = importlib.util.spec_from_file_location(f"_t_{name}", NODES / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def doctor(tmp_path, monkeypatch):
    d = _load("c2c_doctor")
    root = tmp_path / "comfy"
    for sub in ("models/loras", "user", "custom_nodes", "temp", "output/a/b", "input"):
        (root / sub).mkdir(parents=True)
    (root / "models/loras/x.safetensors").write_bytes(b"0" * 1000)
    (root / "output/a/b/img.png").write_bytes(b"0" * 250)
    monkeypatch.setattr(d, "_comfy_root", lambda: root)
    d._DISK_CACHE.update({"ts": 0.0, "data": None})
    return d


def test_scan_counts_bytes_and_files(doctor, tmp_path):
    total, count, complete = doctor._dir_size_scan(tmp_path / "comfy")
    assert (total, count, complete) == (1250, 2, True)
    assert doctor._dir_size_fast(tmp_path / "comfy") == (1250, 2)


def test_scan_stops_at_its_deadline(doctor, tmp_path):
    total, count, complete = doctor._dir_size_scan(tmp_path / "comfy", deadline=time.monotonic() - 1)
    assert complete is False and count == 0


def test_the_badge_path_returns_at_once_while_a_walk_is_slow(doctor, monkeypatch):
    gate = threading.Event()
    real = doctor._dir_size_scan

    def slow_scan(p, deadline=None):
        gate.wait(5)
        return real(p, deadline)

    monkeypatch.setattr(doctor, "_dir_size_scan", slow_scan)
    t0 = time.perf_counter()
    first = doctor.collect_disk_nowait()
    took = time.perf_counter() - t0
    assert took < 0.5, f"the poller waited {took:.2f}s on a folder walk"
    assert first.get("pending") is True and "drive" in first
    gate.set()
    for _ in range(100):
        if doctor._DISK_CACHE["data"] is not None:
            break
        time.sleep(0.05)
    later = doctor.collect_disk_nowait()
    assert later["sections"]["models"]["bytes"] == 1000
    assert later["cached"] is True


def test_only_one_walk_runs_at_a_time(doctor, monkeypatch):
    walks = []
    real = doctor._walk_disk

    def counting_walk():
        walks.append(1)
        time.sleep(0.3)
        return real()

    monkeypatch.setattr(doctor, "_walk_disk", counting_walk)
    ts = [threading.Thread(target=doctor.collect_disk) for _ in range(4)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert len(walks) == 1, f"{len(walks)} walks for 4 simultaneous callers"


def test_package_list_is_cached(doctor, monkeypatch):
    calls = []
    monkeypatch.setattr(doctor, "_collect_pyenv_uncached", lambda: calls.append(1) or {"ok": 1})
    doctor._PYENV_CACHE.update({"ts": 0.0, "data": None})
    for _ in range(5):
        doctor.collect_pyenv()
    assert len(calls) == 1


def test_the_lint_of_an_unchanged_workflow_is_reused(monkeypatch):
    agg = _load("c2c_int_aggregator")
    runs = []
    monkeypatch.setattr(agg, "_run_doctor_uncached", lambda wf: runs.append(1) or {"ran": True})
    agg._DOCTOR_CACHE.update({"key": None, "data": None, "ts": 0.0})
    for _ in range(3):
        agg._run_doctor({"nodes": []}, key="abc")
    agg._run_doctor({"nodes": [1]}, key="def")
    assert len(runs) == 2


def test_health_routes_run_off_the_event_loop():
    src = (NODES / "c2c_int_aggregator.py").read_text(encoding="utf-8")
    routes = src[src.index('@routes.get("/c2c/int/health")'):src.index('@routes.get("/c2c/int/runs")')]
    assert "web.json_response(aggregate(" not in routes, "aggregate() called on the event loop"
    assert routes.count("await _aggregate_off_loop(") == 2
    assert "run_in_executor" in src
    env = src[src.index("def _read_environment"):src.index("def _read_registry_summary")]
    assert "collect_disk_nowait()" in env


# ── the level must mean something ────────────────────────────────────────
PIP_THIS_BOX = """omnivoice 0.1.5 requires gradio, which is not installed.
mediapipe 0.10.21 has requirement numpy<2, but you have numpy 2.4.6.
onnx-weekly 1.23.0.dev20260713 has requirement protobuf>=6.31.1, but you have protobuf 4.25.8."""


def _ig(detail):
    return {"available": True, "pip_check_ok": False, "pip_check_detail": detail}


def test_everyday_pip_conflicts_are_degraded_not_critical():
    """Measured 2026-09-27: this box fails pip check on six lines (unused
    extras, mediapipe's numpy<2 pin) and ComfyUI runs fine. Any failure used
    to mean Critical, so the badge was purple forever."""
    agg = _load("c2c_int_aggregator")
    core, other = agg._pip_conflicts(_ig(PIP_THIS_BOX))
    assert core == [] and len(other) == 3


def test_a_broken_core_package_is_critical_in_pip_and_uv_wording():
    agg = _load("c2c_int_aggregator")
    pip_line = "torchvision 0.20.0 has requirement torch==2.5.0, but you have torch 2.6.0."
    uv = "Found 2 incompatibilities\nThe package `comfyui_frontend_package` requires `comfyui-workflow-templates>=0.1`, but `0.0.9` is installed\nThe package `mediapipe` requires `numpy<2`, but `2.4.6` is installed"
    assert agg._pip_conflicts(_ig(pip_line))[0] == [pip_line]
    core, other = agg._pip_conflicts(_ig(uv))
    assert len(core) == 1 and "comfyui_frontend_package" in core[0]
    assert len(other) == 1 and "mediapipe" in other[0]


def test_aggregate_level_follows_the_split(monkeypatch):
    agg = _load("c2c_int_aggregator")
    quiet = {"available": True}
    monkeypatch.setattr(agg, "_read_runtime_buffer", lambda w: quiet)
    monkeypatch.setattr(agg, "_read_registry_summary", lambda: quiet)
    monkeypatch.setattr(agg, "_read_environment", lambda: quiet)
    monkeypatch.setattr(agg, "_read_integrity_report", lambda: _ig(PIP_THIS_BOX))
    out = agg.aggregate()
    assert out["level"] == "warn" and out["counts"]["pip_conflicts"] == 3
    monkeypatch.setattr(agg, "_read_integrity_report",
                        lambda: _ig("torch 2.6.0 has requirement sympy==1.13.1, but you have sympy 1.14.0."))
    out = agg.aggregate()
    assert out["level"] == "crit" and out["counts"]["pip_core_conflicts"] == 1
