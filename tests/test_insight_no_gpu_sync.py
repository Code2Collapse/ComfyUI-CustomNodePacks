"""L2.51 (owner 2026-10-15: "every process has to be very fast"): the Insight executor hook (nodes/insight.py) wraps
core's per-node execute(). It used to call torch.cuda.synchronize() before and after EVERY node on a GPU, so no node's
GPU work could overlap the next node. Now it never syncs unless C2C_INSIGHT_SYNC=1, and C2C_INSIGHT=0 skips the hook.
"""
from __future__ import annotations

import asyncio
import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch

PACK = Path(__file__).resolve().parents[1]


def _fresh_insight():
    spec = importlib.util.spec_from_file_location("c2c_insight_under_test", PACK / "nodes" / "insight.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def fake_core(monkeypatch):
    """A stand-in for core's execution module (async execute) and a GPU that counts synchronize() calls."""
    calls = {"sync": 0, "execute": 0}
    ex = types.ModuleType("execution")

    class PromptExecutor:  # noqa: D401 - only its presence matters
        pass

    async def execute(server, dynprompt, caches, current_item, *a, **k):
        calls["execute"] += 1
        return ("SUCCESS", None, None)

    ex.PromptExecutor = PromptExecutor
    ex.execute = execute
    monkeypatch.setitem(sys.modules, "execution", ex)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: calls.__setitem__("sync", calls["sync"] + 1))
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda *a, **k: 0)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *a, **k: 0)
    return ex, calls


def _run_nodes(ex, n):
    async def go():
        for i in range(n):
            await ex.execute(None, None, None, str(i))
    asyncio.run(go())


def test_no_gpu_sync_per_node_by_default(fake_core, monkeypatch):
    ex, calls = fake_core
    monkeypatch.delenv("C2C_INSIGHT_SYNC", raising=False)
    monkeypatch.delenv("C2C_INSIGHT", raising=False)
    mod = _fresh_insight()
    monkeypatch.setattr(mod, "_emit", lambda ev: None)
    assert mod.install() is True
    _run_nodes(ex, 5)
    assert calls["execute"] == 5
    assert calls["sync"] == 0


def test_sync_only_when_asked(fake_core, monkeypatch):
    ex, calls = fake_core
    monkeypatch.setenv("C2C_INSIGHT_SYNC", "1")
    mod = _fresh_insight()
    monkeypatch.setattr(mod, "_emit", lambda ev: None)
    assert mod.install() is True
    _run_nodes(ex, 3)
    assert calls["sync"] == 6                      # before and after each of the 3 nodes


def test_off_switch_leaves_core_alone(fake_core, monkeypatch):
    ex, calls = fake_core
    monkeypatch.setenv("C2C_INSIGHT", "0")
    original = ex.execute
    mod = _fresh_insight()
    assert mod.install() is False
    assert ex.execute is original
