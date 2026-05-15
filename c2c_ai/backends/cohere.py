"""Cohere Chat v2 backend (https://docs.cohere.com/reference/chat).

The v2 API is OpenAI-style with a "messages" array. Both sync and SSE
streaming are supported. Auth header is ``Authorization: Bearer ...``.

Recommended models:
    cheap   → command-r-08-2024
    smart   → command-r-plus-08-2024
"""

from __future__ import annotations

import json
import logging
import time
from typing import Generator

import httpx

from .base import Backend, now_ms
from ..keychain import get as kc_get, KEY_COHERE
from ..types import (
    AskRequest,
    AskResponse,
    BackendInfo,
    Capability,
    HealthState,
    Tier,
)

log = logging.getLogger("c2c_ai.cohere")

ENDPOINT = "https://api.cohere.com/v2/chat"


class CohereBackend(Backend):

    @classmethod
    def build(cls,
              backend_id: str = "cloud.cohere",
              model: str = "command-r-08-2024",
              display_name: str = "Command R (Cohere)",
              cost_per_1k_input: float = 0.00015,
              cost_per_1k_output: float = 0.0006,
              max_context: int = 128_000) -> "CohereBackend":
        info = BackendInfo(
            id=backend_id,
            tier=Tier.CLOUD,
            display_name=display_name,
            model=model,
            capabilities={Capability.CHAT, Capability.STREAMING,
                          Capability.JSON_MODE, Capability.TOOL_USE},
            max_context=max_context,
            cost_per_1k_input=cost_per_1k_input,
            cost_per_1k_output=cost_per_1k_output,
        )
        return cls(info)

    def _headers(self, key: str) -> dict:
        return {"Authorization": f"Bearer {key}",
                "Content-Type": "application/json"}

    def _body(self, req: AskRequest, stream: bool = False) -> dict:
        msgs = [{"role": m.role, "content": m.content} for m in req.messages]
        return {
            "model": self.info.model,
            "messages": msgs,
            "temperature": req.temperature,
            "max_tokens": req.max_tokens,
            "stream": stream,
        }

    @staticmethod
    def _extract_text(data: dict) -> str:
        try:
            parts = data["message"]["content"]
            return "".join(p.get("text", "") for p in parts
                           if p.get("type") == "text")
        except (KeyError, TypeError):
            return ""

    # ------------------------------------------------------------------ ask
    def ask(self, req: AskRequest) -> AskResponse:
        key = kc_get(KEY_COHERE)
        if not key:
            raise RuntimeError("COHERE_API_KEY not in keychain — "
                               "open Settings → C2C → AI Backends")
        t0 = now_ms()
        with httpx.Client(timeout=60.0) as client:
            r = client.post(ENDPOINT, headers=self._headers(key),
                            json=self._body(req))
        latency = now_ms() - t0
        if r.status_code != 200:
            raise RuntimeError(f"Cohere API {r.status_code}: {r.text[:400]}")
        data = r.json()
        text = self._extract_text(data)
        usage = (data.get("usage") or {}).get("tokens") or {}
        in_tok = int(usage.get("input_tokens", 0))
        out_tok = int(usage.get("output_tokens", 0))
        cost = ((in_tok / 1000.0) * self.info.cost_per_1k_input
                + (out_tok / 1000.0) * self.info.cost_per_1k_output)
        return AskResponse(
            text=text,
            backend_id=self.info.id,
            model=self.info.model,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cost_usd=cost,
            latency_ms=latency,
        )

    # ---------------------------------------------------------------- stream
    def stream(self, req: AskRequest) -> Generator[str, None, AskResponse]:
        key = kc_get(KEY_COHERE)
        if not key:
            raise RuntimeError("COHERE_API_KEY not in keychain")
        t0 = now_ms()
        text_buf: list[str] = []
        in_tok = 0
        out_tok = 0
        with httpx.Client(timeout=120.0) as client:
            with client.stream("POST", ENDPOINT, headers=self._headers(key),
                               json=self._body(req, stream=True)) as r:
                if r.status_code != 200:
                    raise RuntimeError(f"Cohere stream {r.status_code}: "
                                       f"{r.read()[:400]!r}")
                # Cohere v2 SSE: events are JSON lines with "type" field.
                for line in r.iter_lines():
                    if not line:
                        continue
                    payload = line.strip()
                    if payload.startswith("data: "):
                        payload = payload[6:].strip()
                    if not payload or payload == "[DONE]":
                        continue
                    try:
                        ev = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    et = ev.get("type", "")
                    if et == "content-delta":
                        chunk = ((ev.get("delta") or {})
                                 .get("message", {})
                                 .get("content", {})
                                 .get("text", ""))
                        if chunk:
                            text_buf.append(chunk)
                            yield chunk
                    elif et == "message-end":
                        usage = ((ev.get("delta") or {})
                                 .get("usage", {})
                                 .get("tokens", {}))
                        in_tok = int(usage.get("input_tokens", in_tok))
                        out_tok = int(usage.get("output_tokens", out_tok))
        latency = now_ms() - t0
        cost = ((in_tok / 1000.0) * self.info.cost_per_1k_input
                + (out_tok / 1000.0) * self.info.cost_per_1k_output)
        return AskResponse(
            text="".join(text_buf),
            backend_id=self.info.id,
            model=self.info.model,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cost_usd=cost,
            latency_ms=latency,
        )

    # ----------------------------------------------------------------- probe
    def probe(self, timeout: float = 5.0) -> HealthState:
        key = kc_get(KEY_COHERE)
        if not key:
            self.health = HealthState(
                ok=False, last_rtt_ms=0.0,
                last_error="no API key in keychain",
                last_probe_at=time.time())
            return self.health
        body = {
            "model": self.info.model,
            "messages": [{"role": "user", "content": "."}],
            "max_tokens": 1,
        }
        t0 = now_ms()
        try:
            with httpx.Client(timeout=timeout) as client:
                r = client.post(ENDPOINT, headers=self._headers(key), json=body)
            rtt = now_ms() - t0
            ok = r.status_code == 200
            err = None if ok else f"HTTP {r.status_code}: {r.text[:120]}"
        except Exception as exc:
            rtt = now_ms() - t0
            ok = False
            err = type(exc).__name__ + ": " + str(exc)[:200]
        self.health = HealthState(ok=ok, last_rtt_ms=rtt,
                                  last_error=err, last_probe_at=time.time())
        return self.health
