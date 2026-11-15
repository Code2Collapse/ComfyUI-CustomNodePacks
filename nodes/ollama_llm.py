"""Tier-2 (advanced local) backend: Ollama HTTP client.

Ollama runs an OpenAI-compatible server on localhost (default :11434) and
serves ANY local model the user has pulled (`ollama pull qwen3:4b`,
`ollama pull llama3.2:3b`, etc). No Python deps required — pure stdlib HTTP.

This is the "Tier 2 advanced" path: when the user has Ollama running, they
get access to dozens of curated local models at any size, with hot-swap
between them, instead of a single hand-picked GGUF file.

Public API
----------
is_available(url=None) -> bool
    True iff an Ollama daemon is reachable.

list_models(url=None) -> list[str]
    Returns the names of models installed locally in Ollama. [] on failure.

Chat generation is not here any more (L6.10): the Diagnostics Tier 2 Ollama choice is the AI spine's
diag.local.ollama backend (c2c_ai.diag_bridge -> the OpenAI-compatible backend), asked through the router.
"""
from __future__ import annotations

import json
import logging
from typing import List, Optional

import urllib.error
import urllib.request

log = logging.getLogger("MEC.ollama_llm")

DEFAULT_URL = "http://localhost:11434"

def _http(url: str, *, method: str = "GET", body: Optional[dict] = None,
          timeout: float = 60.0) -> Optional[dict]:
    data = json.dumps(body).encode("utf-8") if body is not None else None
    headers = {"Content-Type": "application/json"} if body is not None else {}
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            try:
                return json.loads(raw)
            except Exception:
                return {"_raw": raw}
    except urllib.error.HTTPError as e:
        try:
            payload = e.read().decode("utf-8", errors="replace")
        except Exception:
            payload = ""
        log.warning("[ollama] HTTP %s on %s: %s", e.code, url, payload[:400])
        return None
    except Exception as e:
        log.info("[ollama] request to %s failed: %s", url, e)
        return None


def _normalize_url(url: Optional[str]) -> str:
    u = (url or DEFAULT_URL).strip().rstrip("/")
    if not u:
        u = DEFAULT_URL
    if not u.startswith(("http://", "https://")):
        u = "http://" + u
    return u


def is_available(url: Optional[str] = None, timeout: float = 1.5) -> bool:
    u = _normalize_url(url)
    j = _http(f"{u}/api/tags", timeout=timeout)
    return isinstance(j, dict) and "models" in j


def list_models(url: Optional[str] = None) -> List[str]:
    u = _normalize_url(url)
    j = _http(f"{u}/api/tags", timeout=3.0)
    if not isinstance(j, dict):
        return []
    out: List[str] = []
    for m in j.get("models") or []:
        name = m.get("name") or m.get("model")
        if name:
            out.append(str(name))
    return out
