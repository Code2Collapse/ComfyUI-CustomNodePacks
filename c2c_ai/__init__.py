"""C2C.AI — dual-backend AI spine for ComfyUI-CustomNodePacks.

Public surface:
    from c2c_ai import ask, stream, get_status, get_cost_today

Every AI-powered feature in C2C goes through this module. Features must NOT
talk to model providers directly — they MUST call ``ask()`` or ``stream()``
which routes through the policy layer, redactor, cost meter and health
monitor.

Architecture (one-liner each):
    backends/          adapters per provider (anthropic, openai, qwen, openai_compat, bundled)
    router.py          picks the best healthy backend that satisfies the request
    policy.py          per-feature rules (AUTO / PREFER_LOCAL / CLOUD_ONLY / ...)
    redactor.py        scrubs local paths, emails, keys, UUIDs before cloud calls
    cost_meter.py      tallies tokens, enforces daily caps, persists usage history
    keychain.py        OS keyring wrapper (Windows Credential Manager / macOS Keychain / Linux Secret Service)
    api_routes.py      /c2c/ai/* HTTP endpoints consumed by the JS layer
    prompts/           versioned Jinja2 templates (one per feature, regression-tested)
"""

from __future__ import annotations

from .router import ask, stream, get_status, get_cost_today, get_router

__all__ = ["ask", "stream", "get_status", "get_cost_today", "get_router"]

__version__ = "2.0.0-dev"
