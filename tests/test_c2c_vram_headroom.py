"""Browser VRAM headroom route logic — CPU-only with fake comfy modules."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

PACK = Path(__file__).resolve().parents[1]
MOD_PATH = PACK / "nodes" / "_c2c_vram_headroom.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("_c2c_vram_headroom_test", MOD_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod._CORE_DEFAULT_BYTES = None
    mod._MODE = None
    mod._ROUTES_REGISTERED = False
    return mod


@pytest.fixture
def dynamic_env(monkeypatch):
    calls = {"set": [], "get": [600 * 1024 * 1024]}

    def get_simple_vram_headroom():
        return calls["get"][-1]

    def set_simple_vram_headroom(n):
        calls["set"].append(int(n))
        calls["get"].append(int(n))

    aimdo = types.SimpleNamespace(
        get_simple_vram_headroom=get_simple_vram_headroom,
        set_simple_vram_headroom=set_simple_vram_headroom,
    )
    mm = types.SimpleNamespace(
        EXTRA_RESERVED_VRAM=400 * 1024 * 1024,
        extra_reserved_memory=lambda: mm.EXTRA_RESERVED_VRAM,
        get_torch_device=lambda: types.SimpleNamespace(type="cuda"),
        get_total_memory=lambda _d: 16 * 1024 ** 3,
        total_vram=16 * 1024,
    )
    monkeypatch.setitem(sys.modules, "comfy_aimdo", types.SimpleNamespace(control=aimdo))
    monkeypatch.setitem(sys.modules, "comfy_aimdo.control", aimdo)
    monkeypatch.setitem(sys.modules, "comfy", types.SimpleNamespace(model_management=mm))
    monkeypatch.setitem(sys.modules, "comfy.model_management", mm)
    mod = _load_module()
    return mod, calls, mm


@pytest.fixture
def classic_env(monkeypatch):
    mm = types.SimpleNamespace(
        EXTRA_RESERVED_VRAM=500 * 1024 * 1024,
        extra_reserved_memory=lambda: mm.EXTRA_RESERVED_VRAM,
        get_torch_device=lambda: types.SimpleNamespace(type="cuda"),
        get_total_memory=lambda _d: 8 * 1024 ** 3,
        total_vram=8 * 1024,
    )
    monkeypatch.delitem(sys.modules, "comfy_aimdo", raising=False)
    monkeypatch.delitem(sys.modules, "comfy_aimdo.control", raising=False)
    monkeypatch.setitem(sys.modules, "comfy", types.SimpleNamespace(model_management=mm))
    monkeypatch.setitem(sys.modules, "comfy.model_management", mm)
    mod = _load_module()
    return mod, mm


def test_get_captures_core_default_before_any_apply(dynamic_env):
    mod, calls, _mm = dynamic_env
    status = mod.build_status()
    assert status["ok"] is True
    assert mod._CORE_DEFAULT_BYTES == 600 * 1024 * 1024
    assert calls["set"] == []


def test_dynamic_apply_uses_aimdo(dynamic_env):
    mod, calls, _mm = dynamic_env
    mod.apply_browser_headroom_gb(1.0)
    expected = max(600 * 1024 * 1024, 1 * 1024 ** 3)
    assert calls["set"] == [expected]


def test_classic_apply_sets_extra_reserved_vram(classic_env):
    mod, mm = classic_env
    mod.apply_browser_headroom_gb(1.5)
    expected = max(500 * 1024 * 1024, int(1.5 * 1024 ** 3))
    assert mm.EXTRA_RESERVED_VRAM == expected


def test_never_below_core_default(dynamic_env):
    mod, calls, _mm = dynamic_env
    mod.apply_browser_headroom_gb(0.1)
    assert calls["set"] == [600 * 1024 * 1024]


def test_zero_restores_core_default(dynamic_env):
    mod, calls, _mm = dynamic_env
    mod.apply_browser_headroom_gb(2.0)
    mod.apply_browser_headroom_gb(0)
    assert calls["set"][-1] == 600 * 1024 * 1024


def test_bad_input_plain_error_never_raises(dynamic_env):
    mod, _calls, _mm = dynamic_env
    for bad in ("nope", -1, 5):
        out = mod.apply_browser_headroom_gb(bad)
        assert out["ok"] is False
        assert "error" in out


def test_build_status_fields(dynamic_env):
    mod, _calls, _mm = dynamic_env
    mod.apply_browser_headroom_gb(1.0)
    s = mod.build_status()
    assert s["mode"] == "dynamic"
    assert s["core_default_gb"] == pytest.approx(600 / 1024, rel=0.01)
    assert s["applied_gb"] >= s["core_default_gb"]
    assert s["vram_total_gb"] == 16.0


def test_register_routes_idempotent(monkeypatch):
    mod = _load_module()
    class _Routes:
        def __init__(self):
            self._gets, self._posts = [], []

        def get(self, path):
            def deco(fn):
                self._gets.append((path, fn))
                return fn
            return deco

        def post(self, path):
            def deco(fn):
                self._posts.append((path, fn))
                return fn
            return deco

    routes = _Routes()
    inst = types.SimpleNamespace(routes=routes)
    ps = types.SimpleNamespace(instance=inst)
    monkeypatch.setitem(sys.modules, "server", types.SimpleNamespace(PromptServer=ps))
    monkeypatch.setitem(sys.modules, "aiohttp", types.SimpleNamespace(web=types.SimpleNamespace(json_response=lambda d, status=200: (d, status))))
    assert mod.register_routes() is True
    assert mod.register_routes() is True
    assert len(routes._gets) == 1
    assert len(routes._posts) == 1
