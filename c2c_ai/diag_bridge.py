"""The Diagnostics sidebar's Tier 2 / Tier 3 choices, served through the AI spine (L6.01 S5).

The Diagnostics sidebar has its own AI settings (``error_assistant.load_settings()``: tier2 llama.cpp / Ollama model,
tier3 cloud provider + model, keys in ``secrets_store``). They used to drive separate client code that skipped the
spine (no timeout / cancel, health, cost cap; the cloud path could skip redaction). Here they become ordinary router
backends with ``diag.*`` ids, scoped to the features those tiers served before (``FEATURES``), so they reach exactly
as far as they did and no further. Re-applied at start-up and on every Diagnostics settings save.
"""
from __future__ import annotations

import logging
from typing import Callable, Optional

log = logging.getLogger("c2c_ai.diag_bridge")

PREFIX = "diag."
# The features the legacy Diagnostics tiers answered before this bridge: Diagnose (feature error_assistant) and the
# workflow builder (whose _complete fell back to those tiers).
FEATURES = frozenset({"error_assistant", "workflow_builder"})
# Diagnostics cloud provider -> spine backend kind (bootstrap.build_backend)
CLOUD_KINDS = {"openai": "openai", "anthropic": "anthropic", "gemini": "gemini", "openrouter": "openrouter",
               "groq": "groq", "deepseek": "deepseek"}


KEY_NAMES = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY", "gemini": "GEMINI_API_KEY",
             "openrouter": "OPENROUTER_API_KEY", "groq": "GROQ_API_KEY", "deepseek": "DEEPSEEK_API_KEY"}


def entries_from_settings(st: dict, resolve_gguf: Optional[Callable[[str], Optional[str]]] = None,
                          has_key: Optional[Callable[[str], bool]] = None) -> list[dict]:
    """Spine backend entries (bootstrap.build_backend format) for the enabled Diagnostics tiers. Pure. The cloud
    tier is only bridged when its provider has a key (the legacy path also did nothing without one; an unusable cloud
    backend would otherwise be offered in the Diagnose consent card)."""
    out: list[dict] = []
    if st.get("tier2_enabled"):
        if (st.get("tier2_backend") or "llamacpp").strip().lower() == "ollama":
            out.append({"kind": "local", "id": PREFIX + "local.ollama", "display_name": "Ollama (Diagnostics)",
                        "base_url": (st.get("ollama_url") or "http://localhost:11434").rstrip("/"),
                        "model": st.get("ollama_model") or "qwen3:4b"})
        else:
            path = resolve_gguf(st.get("local_model") or "") if resolve_gguf else None
            if path:
                out.append({"kind": "text_encoder_local", "id": PREFIX + "local.llamacpp",
                            "display_name": "Local GGUF (Diagnostics)", "model_file": path, "n_gpu_layers": 0})
    if st.get("tier3_enabled"):
        prov = (st.get("cloud_provider") or "").strip().lower()
        kind = CLOUD_KINDS.get(prov)
        if kind and has_key is not None and not has_key(prov):
            log.info("[diag_bridge] %s has no API key; the Diagnostics cloud tier is not bridged", prov)
        elif kind:
            e = {"kind": kind, "id": f"{PREFIX}cloud.{prov}", "display_name": f"{prov} (Diagnostics)"}
            if st.get("cloud_model"):
                e["model"] = st["cloud_model"]
            out.append(e)
        elif prov:
            log.info("[diag_bridge] cloud provider %r has no spine backend; not bridged", prov)
    return out


def _resolve_gguf(model_id: str) -> Optional[str]:
    try:
        try:
            from ..nodes import local_llm  # type: ignore
        except Exception:
            import importlib
            local_llm = importlib.import_module("nodes.local_llm")
        return local_llm._resolve_any_model(model_id)
    except Exception as exc:  # noqa: BLE001
        log.info("[diag_bridge] GGUF resolver unavailable: %s", exc)
        return None


def _has_key(prov: str) -> bool:
    try:
        from .keychain import get
        return bool(get(KEY_NAMES.get(prov, "")))
    except Exception:  # noqa: BLE001
        return False


def apply(router, settings: dict, resolve_gguf: Optional[Callable[[str], Optional[str]]] = None,
          has_key: Optional[Callable[[str], bool]] = None) -> list[str]:
    """Replace the router's diag.* backends with the ones the settings describe. Returns the registered ids."""
    from .bootstrap import build_backend
    for b in list(router.all_backends()):
        if b.info.id.startswith(PREFIX):
            router.unregister(b.info.id)
    ids = []
    for entry in entries_from_settings(settings, resolve_gguf or _resolve_gguf, has_key or _has_key):
        try:
            b = build_backend(entry)
            b.info.features = FEATURES
            router.register(b)
            ids.append(b.info.id)
        except Exception as exc:  # noqa: BLE001 - one bad tier must not stop the others
            log.warning("[diag_bridge] %s not registered: %s", entry.get("id"), exc)
    return ids


def apply_current() -> list[str]:
    """apply() with the saved Diagnostics settings and the global router. Never raises."""
    try:
        try:
            from ..nodes import error_assistant as ea  # type: ignore
        except Exception:
            import importlib
            ea = importlib.import_module("nodes.error_assistant")
        from .router import get_router
        return apply(get_router(), ea.load_settings())
    except Exception as exc:  # noqa: BLE001
        log.info("[diag_bridge] not applied: %s", exc)
        return []
