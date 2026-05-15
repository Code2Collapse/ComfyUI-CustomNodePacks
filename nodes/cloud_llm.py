"""Tier-3 backend for ErrorAssistantMEC: cloud LLM dispatcher.

Supports OpenAI, Anthropic, Gemini, OpenRouter via plain HTTP — no SDK
dependencies required. Keys are pulled from `secrets_store.get_key(provider)`.

Public API
----------
generate(provider, model, prompt, max_tokens=512) -> str | None
    Synchronous. Returns the model's text or None on failure.
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

    `extra_kwargs` are forwarded to providers that accept them (azure_openai,
    custom). They are silently ignored by providers that don't.
    """
    fn = _DISPATCH.get((provider or "").strip().lower())
    if fn is None:
        log.warning("[cloud_llm] unknown provider %r", provider)
        return None
    # Filter kwargs by what the function accepts to remain backward-compatible.
    import inspect
    sig = inspect.signature(fn)
    accepted = {k: v for k, v in extra_kwargs.items() if k in sig.parameters}
    return fn(model, prompt, max_tokens, system, **accepted)

