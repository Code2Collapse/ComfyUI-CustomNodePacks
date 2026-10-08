"""The C2C AI plumbing that made "AI things are not working" (A9, L2.29, 2026-10-08).

1. Saving the AI settings removed the built-in offline rule pack, leaving zero backends until a restart.
2. The workflow builder and the error diagnoser read only the old Error Assistant tiers, so a model configured
   in Settings > C2C > AI was never asked.
Measured end to end against a stand-in model server: docs/evidence/L2.29 (work area).
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

PACK = Path(__file__).resolve().parents[1]
if str(PACK) not in sys.path:
    sys.path.insert(0, str(PACK))

pytest.importorskip("httpx", reason="the C2C AI backends need httpx")

from c2c_ai import bootstrap as bs  # noqa: E402
from c2c_ai.types import Tier  # noqa: E402


class _Info:
    def __init__(self, bid, tier, enabled=True):
        self.id, self.tier, self.enabled = bid, tier, enabled


class _Backend:
    def __init__(self, bid, tier):
        self.info = _Info(bid, tier)


class _Router:
    def __init__(self, backends):
        self._b = {b.info.id: b for b in backends}
        self.asked = []

    def all_backends(self):
        return list(self._b.values())

    def unregister(self, bid):
        self._b.pop(bid, None)

    def register(self, b):
        self._b[b.info.id] = b

    def ask(self, feature, msgs, **kw):
        self.asked.append((feature, [m.content for m in msgs], kw))
        model = next(b for b in self._b.values() if b.info.tier != Tier.DETERMINISTIC)
        return types.SimpleNamespace(text='{"nodes": []}', backend_id=model.info.id)


def test_saving_the_settings_keeps_the_offline_rule_pack():
    router = _Router([_Backend("deterministic.rulepack", Tier.DETERMINISTIC), _Backend("local.old", Tier.LOCAL)])
    bs.reapply_backends(router, {"backends": []})
    assert [b.info.id for b in router.all_backends()] == ["deterministic.rulepack"]


def test_saving_the_settings_registers_the_saved_backends():
    router = _Router([_Backend("deterministic.rulepack", Tier.DETERMINISTIC)])
    bs.reapply_backends(router, {"backends": [{"kind": "local", "id": "local.x", "base_url": "http://127.0.0.1:9"}]})
    assert {b.info.id for b in router.all_backends()} == {"deterministic.rulepack", "local.x"}


def _builder():
    from nodes import ai_workflow_builder as wb
    return wb


def test_the_builder_asks_the_c2c_ai_router_first(monkeypatch):
    wb = _builder()
    router = _Router([_Backend("deterministic.rulepack", Tier.DETERMINISTIC), _Backend("local.fake", Tier.LOCAL)])
    import c2c_ai.router as rmod
    monkeypatch.setattr(rmod, "get_router", lambda: router)
    text, provider = wb._complete_router("Request: x", "JSON only", "workflow_builder")
    assert provider == "local.fake" and text == '{"nodes": []}'
    assert router.asked and router.asked[0][0] == "workflow_builder"
    assert router.asked[0][1][0] == "JSON only"          # the system prompt went first


def test_the_rule_pack_alone_is_not_a_model(monkeypatch):
    wb = _builder()
    router = _Router([_Backend("deterministic.rulepack", Tier.DETERMINISTIC)])
    import c2c_ai.router as rmod
    monkeypatch.setattr(rmod, "get_router", lambda: router)
    assert wb._complete_router("Request: x", None, "workflow_builder") == (None, "none")
    assert not router.asked
