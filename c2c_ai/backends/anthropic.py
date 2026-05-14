"""Anthropic (Claude) backend.

Uses ``httpx`` directly against the official REST API rather than the SDK so
we have one consistent retry / timeout / streaming story across all
backends. Authentication header is ``x-api-key`` per Anthropic spec.

Model is configurable at registration. Recommended defaults:
    cheap   → claude-3-5-haiku-latest
    smart   → claude-3-5-sonnet-latest
    smartest→ claude-3-opus-latest  (or claude-4-x when GA)
"""

from __future__ import annotations

import json
import logging
import time
from typing import Generator

import httpx

from .base import Backend, now_ms
from ..keychain import get as kc_get, KEY_ANTHROPIC
from ..types import (
    AskRequest,
    AskResponse,
    BackendInfo,
    Capability,
    HealthState,
    Tier,
)

log = logging.getLogger("c2c_ai.anthropic")

ENDPOINT = "https://api.anthropic.com/v1/messages"
API_VERSION = "2023-06-01"


class AnthropicBackend(Backend):

    @classmethod
    def build(cls,
              backend_id: str = "cloud.anthropic",
              model: str = "claude-3-5-sonnet-latest",
              display_name: str = "Claude (Anthropic)",
              cost_per_1k_input: float = 0.003,
              cost_per_1k_output: float = 0.015,
              max_context: int = 200_000) -> "AnthropicBackend":
        info = BackendInfo(
            id=backend_id,
            tier=Tier.CLOUD,
            display_name=display_name,
            model=model,
            capabilities={Capability.CHAT, Capability.STREAMING,
                          Capability.JSON_MODE, Capability.TOOL_USE,
                          Capability.VISION},
            max_context=max_context,
            cost_per_1k_input=cost_per_1k_input,
            cost_per_1k_output=cost_per_1k_output,
        )
        return cls(info)

    # ------------------------------------------------------------------ ask
    def ask(self, req: AskRequest) -> AskResponse:
        api_key = kc_get(KEY_ANTHROPIC)
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not in keychain — open Settings → C2C → AI Backends")

        system_prompt, msg_list = _split_system(req.messages)
        body: dict = {
            "model": self.info.model,
            "max_tokens": req.max_tokens,
            "temperature": req.temperature,
            "messages": msg_list,
        }
        if system_prompt:
            body["system"] = system_prompt

        headers = {
            "x-api-key": api_key,
            "anthropic-version": API_VERSION,
            "content-type": "application/json",
        }

        t0 = now_ms()
        with httpx.Client(timeout=60.0) as client:
            r = client.post(ENDPOINT, headers=headers, json=body)
        latency = now_ms() - t0

        if r.status_code != 200:
            raise RuntimeError(f"Anthropic API {r.status_code}: {r.text[:400]}")

        data = r.json()
        # content is a list of {type, text} blocks
        text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
        usage = data.get("usage", {})
        in_tok = int(usage.get("input_tokens", 0))
        out_tok = int(usage.get("output_tokens", 0))
        cost = (in_tok / 1000.0) * self.info.cost_per_1k_input + \
               (out_tok / 1000.0) * self.info.cost_per_1k_output

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
        api_key = kc_get(KEY_ANTHROPIC)
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not in keychain")

        system_prompt, msg_list = _split_system(req.messages)
        body: dict = {
            "model": self.info.model,
            "max_tokens": req.max_tokens,
            "temperature": req.temperature,
            "messages": msg_list,
            "stream": True,
        }
        if system_prompt:
            body["system"] = system_prompt

        headers = {
            "x-api-key": api_key,
            "anthropic-version": API_VERSION,
            "content-type": "application/json",
            "accept": "text/event-stream",
        }

        t0 = now_ms()
        text_buf: list[str] = []
        in_tok = 0
        out_tok = 0
        with httpx.Client(timeout=120.0) as client:
            with client.stream("POST", ENDPOINT, headers=headers, json=body) as r:
                if r.status_code != 200:
                    raise RuntimeError(f"Anthropic stream {r.status_code}: {r.read()[:400]!r}")
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
                    et = ev.get("type")
                    if et == "content_block_delta":
                        delta = ev.get("delta", {})
                        if delta.get("type") == "text_delta":
                            chunk = delta.get("text", "")
                            if chunk:
                                text_buf.append(chunk)
                                yield chunk
                    elif et == "message_start":
                        usage = ev.get("message", {}).get("usage", {})
                        in_tok = int(usage.get("input_tokens", 0))
                    elif et == "message_delta":
                        usage = ev.get("usage", {})
                        out_tok = int(usage.get("output_tokens", out_tok))
        latency = now_ms() - t0
        cost = (in_tok / 1000.0) * self.info.cost_per_1k_input + \
               (out_tok / 1000.0) * self.info.cost_per_1k_output
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
        api_key = kc_get(KEY_ANTHROPIC)
        if not api_key:
            self.health = HealthState(
                ok=False, last_rtt_ms=0.0,
                last_error="no API key in keychain", last_probe_at=time.time())
            return self.health
        # Cheap probe: 1-token request. Yes, costs ~$0.00001. Worth it for accurate health.
        body = {
            "model": self.info.model,
            "max_tokens": 1,
            "messages": [{"role": "user", "content": "."}],
        }
        headers = {
            "x-api-key": api_key,
            "anthropic-version": API_VERSION,
            "content-type": "application/json",
        }
        t0 = now_ms()
        try:
            with httpx.Client(timeout=timeout) as client:
                r = client.post(ENDPOINT, headers=headers, json=body)
            rtt = now_ms() - t0
            ok = r.status_code == 200
            err = None if ok else f"HTTP {r.status_code}"
        except Exception as exc:
            rtt = now_ms() - t0
            ok = False
            err = type(exc).__name__ + ": " + str(exc)[:200]
        self.health = HealthState(ok=ok, last_rtt_ms=rtt, last_error=err, last_probe_at=time.time())
        return self.health


def _split_system(messages: list) -> tuple[str | None, list[dict]]:
    """Anthropic wants ``system`` as a top-level string, not in the messages list."""
    system_parts: list[str] = []
    msg_list: list[dict] = []
    for m in messages:
        if m.role == "system":
            system_parts.append(m.content)
        else:
            msg_list.append({"role": m.role, "content": m.content})
    return ("\n\n".join(system_parts) if system_parts else None), msg_list
