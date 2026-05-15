"""Google Gemini backend.

Uses Generative Language REST API (v1beta) directly via httpx — no SDK.
Auth is via the ``GEMINI_API_KEY`` in the keychain (also accepted as
``GOOGLE_API_KEY`` by the txt parser). Key is passed as a ``?key=...``
query param per Google convention.

Recommended models:
    cheap   → gemini-1.5-flash-latest
    smart   → gemini-1.5-pro-latest
    smartest→ gemini-2.0-pro-exp  (when GA)
"""

from __future__ import annotations

import json
import logging
import time
from typing import Generator

import httpx

from .base import Backend, now_ms
from ..keychain import get as kc_get, KEY_GEMINI
from ..types import (
    AskRequest,
    AskResponse,
    BackendInfo,
    Capability,
    HealthState,
    Tier,
)

log = logging.getLogger("c2c_ai.gemini")

BASE = "https://generativelanguage.googleapis.com/v1beta/models"


class GeminiBackend(Backend):

    @classmethod
    def build(cls,
              backend_id: str = "cloud.gemini",
              model: str = "gemini-1.5-flash-latest",
              display_name: str = "Gemini (Google)",
              cost_per_1k_input: float = 0.000075,
              cost_per_1k_output: float = 0.0003,
              max_context: int = 1_000_000) -> "GeminiBackend":
        info = BackendInfo(
            id=backend_id,
            tier=Tier.CLOUD,
            display_name=display_name,
            model=model,
            capabilities={Capability.CHAT, Capability.STREAMING,
                          Capability.JSON_MODE, Capability.VISION},
            max_context=max_context,
            cost_per_1k_input=cost_per_1k_input,
            cost_per_1k_output=cost_per_1k_output,
        )
        return cls(info)

    # ---------------------------------------------------------------- helpers
    def _build_body(self, req: AskRequest) -> dict:
        system_parts: list[str] = []
        contents: list[dict] = []
        for m in req.messages:
            if m.role == "system":
                system_parts.append(m.content)
            else:
                role = "user" if m.role == "user" else "model"
                contents.append({"role": role,
                                 "parts": [{"text": m.content}]})
        body: dict = {
            "contents": contents,
            "generationConfig": {
                "temperature": req.temperature,
                "maxOutputTokens": req.max_tokens,
            },
        }
        if system_parts:
            body["systemInstruction"] = {
                "parts": [{"text": "\n\n".join(system_parts)}]
            }
        return body

    # ------------------------------------------------------------------ ask
    def ask(self, req: AskRequest) -> AskResponse:
        key = kc_get(KEY_GEMINI)
        if not key:
            raise RuntimeError("GEMINI_API_KEY not in keychain — "
                               "open Settings → C2C → AI Backends")
        url = f"{BASE}/{self.info.model}:generateContent?key={key}"
        t0 = now_ms()
        with httpx.Client(timeout=60.0) as client:
            r = client.post(url, json=self._build_body(req))
        latency = now_ms() - t0
        if r.status_code != 200:
            raise RuntimeError(f"Gemini API {r.status_code}: {r.text[:400]}")
        data = r.json()
        try:
            text = "".join(
                p.get("text", "")
                for p in data["candidates"][0]["content"]["parts"]
            )
        except (KeyError, IndexError) as exc:
            raise RuntimeError(f"Gemini response missing candidates: "
                               f"{exc!r}; raw={str(data)[:300]}")
        usage = data.get("usageMetadata", {})
        in_tok = int(usage.get("promptTokenCount", 0))
        out_tok = int(usage.get("candidatesTokenCount", 0))
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

    # -------------------------------------------------------------- stream
    def stream(self, req: AskRequest) -> Generator[str, None, AskResponse]:
        key = kc_get(KEY_GEMINI)
        if not key:
            raise RuntimeError("GEMINI_API_KEY not in keychain")
        url = (f"{BASE}/{self.info.model}:streamGenerateContent"
               f"?alt=sse&key={key}")
        t0 = now_ms()
        text_buf: list[str] = []
        in_tok = 0
        out_tok = 0
        with httpx.Client(timeout=120.0) as client:
            with client.stream("POST", url, json=self._build_body(req),
                               headers={"accept": "text/event-stream"}) as r:
                if r.status_code != 200:
                    raise RuntimeError(f"Gemini stream {r.status_code}: "
                                       f"{r.read()[:400]!r}")
                for line in r.iter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    payload = line[6:].strip()
                    if not payload or payload == "[DONE]":
                        continue
                    try:
                        ev = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    try:
                        parts = ev["candidates"][0]["content"]["parts"]
                        for p in parts:
                            chunk = p.get("text", "")
                            if chunk:
                                text_buf.append(chunk)
                                yield chunk
                    except (KeyError, IndexError):
                        pass
                    usage = ev.get("usageMetadata") or {}
                    if usage:
                        in_tok = int(usage.get("promptTokenCount", in_tok))
                        out_tok = int(usage.get("candidatesTokenCount", out_tok))
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

    # --------------------------------------------------------------- probe
    def probe(self, timeout: float = 5.0) -> HealthState:
        key = kc_get(KEY_GEMINI)
        if not key:
            self.health = HealthState(
                ok=False, last_rtt_ms=0.0,
                last_error="no API key in keychain",
                last_probe_at=time.time())
            return self.health
        url = f"{BASE}/{self.info.model}:generateContent?key={key}"
        body = {
            "contents": [{"role": "user", "parts": [{"text": "."}]}],
            "generationConfig": {"maxOutputTokens": 1},
        }
        t0 = now_ms()
        try:
            with httpx.Client(timeout=timeout) as client:
                r = client.post(url, json=body)
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
