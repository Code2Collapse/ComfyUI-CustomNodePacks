"""Tier-3 backend for ErrorAssistantMEC: cloud LLM dispatcher.

Supports OpenAI, Anthropic, Gemini, OpenRouter via plain HTTP — no SDK
dependencies required. Keys are pulled from `secrets_store.get_key(provider)`.

Public API
----------
generate(provider, model, prompt, max_tokens=512) -> str | None
    Synchronous. Returns the model's text or None on failure.

AI-routing contract (MegaPlan §3.2)
-----------------------------------
Every AI call in the pack must flow through the unified C2C AI spine
(`c2c_ai.router`) when a backend matching the caller's `provider` is
registered. This file is the legacy back-compat path: it first tries
the spine (so calls get redaction, cost gating, health metrics, and
local-backend fallback for free), and only drops down to direct urllib
HTTP if the spine has no matching backend or the call fails transiently.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Optional

import urllib.error
import urllib.request

log = logging.getLogger("MEC.cloud_llm")

SYSTEM_PROMPT = (
    "You explain Python tracebacks for ComfyUI users. Be concise. "
    "Always answer with two sections labelled exactly CAUSE: and FIXES: "
    "(bulleted, 2-4 fixes)."
)


# ── Router (c2c_ai spine) integration ──────────────────────────────────────
# Sentinel for "spine had no matching backend; caller should use legacy path".
_ROUTER_MISS = object()


def _try_router(provider: str, model: str, prompt: str, max_tokens: int,
                system: Optional[str], **extra_kwargs) -> object:
    """Route via the C2C AI spine if a backend matching `provider` is registered.

    Returns:
        - The response text (str), possibly empty, on a successful spine call.
        - None on a hard failure (spine present, backend matched, but call
          raised) — caller should still attempt legacy as a fallback so users
          aren't broken by a transient spine error.
        - _ROUTER_MISS sentinel if the spine has no backend matching `provider`,
          which signals the caller to use the legacy direct-HTTP dispatch.
    """
    try:
        from .. import c2c_ai  # type: ignore
        from ..c2c_ai import router as router_mod  # type: ignore
        from ..c2c_ai.types import (  # type: ignore
            AskRequest, Capability, Message, Sensitivity,
        )
        from ..c2c_ai import redactor as redactor_mod  # type: ignore
    except Exception as e:
        log.debug("[cloud_llm] router path unavailable: %s", e)
        return _ROUTER_MISS

    try:
        router = router_mod.get_router()
    except Exception as e:
        log.debug("[cloud_llm] get_router failed: %s", e)
        return _ROUTER_MISS

    backend_id = f"cloud.{(provider or '').strip().lower()}"
    backend = router.get(backend_id)
    if backend is None:
        return _ROUTER_MISS

    info = getattr(backend, "info", None)
    if info is None or not getattr(info, "enabled", True):
        return _ROUTER_MISS
    health = getattr(backend, "health", None)
    if health is not None and getattr(health, "ok", True) is False:
        log.debug("[cloud_llm] router backend %s is unhealthy; using legacy",
                  backend_id)
        return _ROUTER_MISS

    msgs = [
        Message(role="system", content=system if system is not None else SYSTEM_PROMPT),
        Message(role="user",   content=prompt),
    ]
    sensitivity = extra_kwargs.get("sensitivity") or Sensitivity.NORMAL
    try:
        redacted_msgs, redacted_flag = redactor_mod.redact_messages(msgs, sensitivity)
    except Exception:
        redacted_msgs, redacted_flag = msgs, False

    req = AskRequest(
        feature=extra_kwargs.get("feature") or "cloud_llm",
        messages=redacted_msgs,
        sensitivity=sensitivity,
        required={Capability.CHAT},
        max_tokens=int(max_tokens),
        temperature=float(extra_kwargs.get("temperature", 0.2)),
    )

    try:
        resp = backend.ask(req)
    except Exception as e:
        log.warning("[cloud_llm] spine call to %s failed: %s — falling back",
                    backend_id, e)
        return None  # signal legacy fallback

    try:
        from ..c2c_ai.cost_meter import get_meter, make_entry  # type: ignore
        get_meter().record(make_entry(
            feature=req.feature, backend_info=info, model=info.model,
            input_tokens=getattr(resp, "input_tokens", 0),
            output_tokens=getattr(resp, "output_tokens", 0),
            latency_ms=getattr(resp, "latency_ms", 0.0),
        ))
    except Exception:
        pass

    text = getattr(resp, "text", "") or ""
    return text.strip()


# ── Legacy direct-HTTP dispatch ────────────────────────────────────────────
def _http_post(url: str, headers: dict, body: dict, timeout: float = 30.0) -> Optional[dict]:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        try:
            payload = e.read().decode("utf-8", errors="replace")
        except Exception:
            payload = ""
        log.warning("[cloud_llm] HTTP %s on %s: %s", e.code, url, payload[:400])
        return None
    except Exception as e:
        log.warning("[cloud_llm] request to %s failed: %s", url, e)
        return None


def _get_key(provider: str) -> Optional[str]:
    try:
        from . import secrets_store  # type: ignore
    except Exception:
        import importlib
        secrets_store = importlib.import_module(
            "ComfyUI-CustomNodePacks.nodes.secrets_store")
    return secrets_store.get_key(provider)


def _openai(model: str, prompt: str, max_tokens: int,
            system: Optional[str] = None) -> Optional[str]:
    key = _get_key("openai")
    if not key:
        return None
    body = {
        "model": model or "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": system if system is not None else SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": int(max_tokens),
        "temperature": 0.2,
    }
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    j = _http_post("https://api.openai.com/v1/chat/completions", headers, body)
    if not j:
        return None
    try:
        return (j["choices"][0]["message"]["content"] or "").strip()
    except Exception:
        return None


def _anthropic(model: str, prompt: str, max_tokens: int,
               system: Optional[str] = None) -> Optional[str]:
    key = _get_key("anthropic")
    if not key:
        return None
    body = {
        "model": model or "claude-3-5-haiku-latest",
        "max_tokens": int(max_tokens),
        "system": system if system is not None else SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
    }
    headers = {
        "x-api-key": key,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
    }
    j = _http_post("https://api.anthropic.com/v1/messages", headers, body)
    if not j:
        return None
    try:
        return (j["content"][0]["text"] or "").strip()
    except Exception:
        return None


def _gemini(model: str, prompt: str, max_tokens: int,
            system: Optional[str] = None) -> Optional[str]:
    key = _get_key("gemini")
    if not key:
        return None
    model = model or "gemini-1.5-flash-latest"
    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"{model}:generateContent?key={key}")
    body = {
        "system_instruction": {"parts": [{"text": system if system is not None else SYSTEM_PROMPT}]},
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": int(max_tokens)},
    }
    headers = {"Content-Type": "application/json"}
    j = _http_post(url, headers, body)
    if not j:
        return None
    try:
        return (j["candidates"][0]["content"]["parts"][0]["text"] or "").strip()
    except Exception:
        return None


def _openrouter(model: str, prompt: str, max_tokens: int,
                system: Optional[str] = None) -> Optional[str]:
    key = _get_key("openrouter")
    if not key:
        return None
    body = {
        "model": model or "openai/gpt-4o-mini",
        "messages": [
            {"role": "system", "content": system if system is not None else SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": int(max_tokens),
        "temperature": 0.2,
    }
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/Code2Collapse/ComfyUI-CustomNodePacks",
        "X-Title": "ComfyUI ErrorAssistantMEC",
    }
    j = _http_post("https://openrouter.ai/api/v1/chat/completions", headers, body)
    if not j:
        return None
    try:
        return (j["choices"][0]["message"]["content"] or "").strip()
    except Exception:
        return None


_DISPATCH = {
    "openai":     _openai,
    "anthropic":  _anthropic,
    "gemini":     _gemini,
    "openrouter": _openrouter,
}


def _groq(model: str, prompt: str, max_tokens: int,
          system: Optional[str] = None) -> Optional[str]:
    key = _get_key("groq")
    if not key:
        return None
    body = {
        "model": model or "llama-3.3-70b-versatile",
        "messages": [
            {"role": "system", "content": system if system is not None else SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": int(max_tokens),
        "temperature": 0.2,
    }
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    j = _http_post("https://api.groq.com/openai/v1/chat/completions", headers, body)
    if not j:
        return None
    try:
        return (j["choices"][0]["message"]["content"] or "").strip()
    except Exception:
        return None


def _deepseek(model: str, prompt: str, max_tokens: int,
              system: Optional[str] = None) -> Optional[str]:
    key = _get_key("deepseek")
    if not key:
        return None
    body = {
        "model": model or "deepseek-chat",
        "messages": [
            {"role": "system", "content": system if system is not None else SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": int(max_tokens),
        "temperature": 0.2,
    }
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    j = _http_post("https://api.deepseek.com/v1/chat/completions", headers, body)
    if not j:
        return None
    try:
        return (j["choices"][0]["message"]["content"] or "").strip()
    except Exception:
        return None


_DISPATCH["groq"]     = _groq
_DISPATCH["deepseek"] = _deepseek


# ─── P10.1: Extended providers (Azure OpenAI, Cohere, Custom OpenAI-compatible) ───
def _azure_openai(model: str, prompt: str, max_tokens: int,
                  system: Optional[str] = None,
                  endpoint: Optional[str] = None,
                  api_version: str = "2024-02-15-preview") -> Optional[str]:
    """Azure OpenAI Service.

    Requires:
      - key (stored as 'azure_openai' in secrets_store)
      - endpoint URL (e.g. https://my-resource.openai.azure.com)
      - deployment name passed as `model` (NOT the underlying model id)

    The `endpoint` is read from the router config (or kwargs) since it is
    user-specific. Falls back to AZURE_OPENAI_ENDPOINT env var.
    """
    key = _get_key("azure_openai")
    if not key:
        return None
    ep = endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT")
    if not ep:
        log.warning("[cloud_llm] azure_openai: no endpoint configured")
        return None
    deployment = model or os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")
    url = (f"{ep.rstrip('/')}/openai/deployments/{deployment}"
           f"/chat/completions?api-version={api_version}")
    body = {
        "messages": [
            {"role": "system", "content": system if system is not None else SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": int(max_tokens),
        "temperature": 0.2,
    }
    headers = {"api-key": key, "Content-Type": "application/json"}
    j = _http_post(url, headers, body)
    if not j:
        return None
    try:
        return (j["choices"][0]["message"]["content"] or "").strip()
    except Exception:
        return None


def _cohere(model: str, prompt: str, max_tokens: int,
            system: Optional[str] = None) -> Optional[str]:
    """Cohere Chat API v2 (https://docs.cohere.com/reference/chat)."""
    key = _get_key("cohere")
    if not key:
        return None
    messages = []
    if system is not None or SYSTEM_PROMPT:
        messages.append({"role": "system",
                         "content": system if system is not None else SYSTEM_PROMPT})
    messages.append({"role": "user", "content": prompt})
    body = {
        "model": model or "command-r-08-2024",
        "messages": messages,
        "max_tokens": int(max_tokens),
        "temperature": 0.2,
    }
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    j = _http_post("https://api.cohere.com/v2/chat", headers, body)
    if not j:
        return None
    try:
        # v2 returns: {"message": {"content": [{"type":"text","text":"..."}]}}
        parts = j["message"]["content"]
        return "".join(p.get("text", "") for p in parts if p.get("type") == "text").strip()
    except Exception:
        return None


def _custom_openai(model: str, prompt: str, max_tokens: int,
                   system: Optional[str] = None,
                   base_url: Optional[str] = None,
                   api_key_override: Optional[str] = None,
                   extra_headers: Optional[dict] = None) -> Optional[str]:
    """Custom OpenAI-compatible endpoint (vLLM, LM Studio, LiteLLM, etc.).

    Reads base_url from kwargs (router-provided) or CUSTOM_LLM_BASE_URL env.
    Key is optional — many local servers don't require it.
    """
    base = base_url or os.environ.get("CUSTOM_LLM_BASE_URL")
    if not base:
        log.warning("[cloud_llm] custom: no base_url configured")
        return None
    key = api_key_override or _get_key("custom") or "not-needed"
    body = {
        "model": model or "default",
        "messages": [
            {"role": "system", "content": system if system is not None else SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": int(max_tokens),
        "temperature": 0.2,
    }
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    if extra_headers:
        headers.update(extra_headers)
    j = _http_post(f"{base.rstrip('/')}/chat/completions", headers, body)
    if not j:
        return None
    try:
        return (j["choices"][0]["message"]["content"] or "").strip()
    except Exception:
        return None


_DISPATCH["azure_openai"] = _azure_openai
_DISPATCH["cohere"]       = _cohere
_DISPATCH["custom"]       = _custom_openai


def generate(provider: str, model: str, prompt: str,
             max_tokens: int = 512,
             system: Optional[str] = None,
             **extra_kwargs) -> Optional[str]:
    """Dispatch to the named provider.

    Prefers the unified C2C AI spine (`c2c_ai.router`) so calls receive
    redaction, cost-cap enforcement, health-aware fallback, and metrics. If
    the spine has no backend registered for `provider`, falls back to a
    direct urllib HTTP call (the legacy path). MegaPlan §3.2 compliance.

    `extra_kwargs` are forwarded to providers that accept them (azure_openai,
    custom). They are silently ignored by providers that don't.
    """
    # 1) Spine-first path.
    spine = _try_router(provider, model, prompt, max_tokens, system, **extra_kwargs)
    if spine is not _ROUTER_MISS:
        # spine attempted; spine returned text (possibly "") or None on failure.
        if isinstance(spine, str):
            return spine or None
        # spine == None → backend matched but call failed; try legacy as a
        # fallback so the caller still gets a response if the keys/URL work.

    # 2) Legacy direct-HTTP dispatch.
    fn = _DISPATCH.get((provider or "").strip().lower())
    if fn is None:
        log.warning("[cloud_llm] unknown provider %r", provider)
        return None
    import inspect
    sig = inspect.signature(fn)
    accepted = {k: v for k, v in extra_kwargs.items() if k in sig.parameters}
    return fn(model, prompt, max_tokens, system, **accepted)

