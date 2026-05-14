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


def generate(provider: str, model: str, prompt: str,
             max_tokens: int = 512,
             system: Optional[str] = None) -> Optional[str]:
    fn = _DISPATCH.get((provider or "").strip().lower())
    if fn is None:
        log.warning("[cloud_llm] unknown provider %r", provider)
        return None
    return fn(model, prompt, max_tokens, system)
