"""Azure OpenAI backend.

Same chat-completions shape as OpenAI but routed through your Azure
resource's deployment URL. Auth header is ``api-key: <KEY>`` (NOT
``Authorization: Bearer``). The ``deployment`` is Azure-specific — it's
the name you gave when you deployed the model in Azure portal, not the
underlying model id.

The URL pattern is:
    {endpoint}/openai/deployments/{deployment}/chat/completions?api-version=...

Configuration (in ai_config.json::backends[]):
    {
        "kind": "azure_openai",
        "id": "cloud.azure_openai",
        "endpoint": "https://my-resource.openai.azure.com",
        "deployment": "gpt-4o-mini-deployment",
        "api_version": "2024-02-15-preview",
        "model": "gpt-4o-mini",   // logical name for logs/UI only
        "max_context": 128000,
        "cost_per_1k_input": 0.00015,
        "cost_per_1k_output": 0.0006,
        "enabled": true
    }
"""

from __future__ import annotations

import json
import logging
import time
from typing import Generator

import httpx

from .base import Backend, now_ms
from ..keychain import get as kc_get, KEY_AZURE_OPENAI
from ..types import (
    AskRequest,
    AskResponse,
    BackendInfo,
    Capability,
    HealthState,
    Tier,
)

log = logging.getLogger("c2c_ai.azure_openai")


class AzureOpenAIBackend(Backend):
    """Azure OpenAI Service (chat-completions deployment)."""

    def __init__(self, info: BackendInfo, *,
                 endpoint: str, deployment: str,
                 api_version: str = "2024-02-15-preview"):
        super().__init__(info)
        self.endpoint = endpoint.rstrip("/")
        self.deployment = deployment
        self.api_version = api_version

    @classmethod
    def build(cls, *,
              backend_id: str = "cloud.azure_openai",
              endpoint: str,
              deployment: str,
              api_version: str = "2024-02-15-preview",
              model: str = "gpt-4o-mini",
              display_name: str = "Azure OpenAI",
              cost_per_1k_input: float = 0.00015,
              cost_per_1k_output: float = 0.0006,
              max_context: int = 128_000) -> "AzureOpenAIBackend":
        if not endpoint:
            raise ValueError("AzureOpenAIBackend requires endpoint URL")
        if not deployment:
            raise ValueError("AzureOpenAIBackend requires deployment name")
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
        return cls(info, endpoint=endpoint, deployment=deployment,
                   api_version=api_version)

    def _url(self) -> str:
        return (f"{self.endpoint}/openai/deployments/{self.deployment}"
                f"/chat/completions?api-version={self.api_version}")

    def _headers(self, key: str) -> dict:
        return {"api-key": key, "Content-Type": "application/json"}

    def _body(self, req: AskRequest, stream: bool = False) -> dict:
        return {
            "messages": [{"role": m.role, "content": m.content}
                         for m in req.messages],
            "temperature": req.temperature,
            "max_tokens": req.max_tokens,
            "stream": stream,
        }

    # ------------------------------------------------------------------ ask
    def ask(self, req: AskRequest) -> AskResponse:
        key = kc_get(KEY_AZURE_OPENAI)
        if not key:
            raise RuntimeError("AZURE_OPENAI_API_KEY not in keychain — "
                               "open Settings → C2C → AI Backends")
        t0 = now_ms()
        with httpx.Client(timeout=60.0) as client:
            r = client.post(self._url(), headers=self._headers(key),
                            json=self._body(req))
        latency = now_ms() - t0
        if r.status_code != 200:
            raise RuntimeError(f"Azure OpenAI {r.status_code}: {r.text[:400]}")
        data = r.json()
        try:
            text = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError) as exc:
            raise RuntimeError(f"Azure OpenAI malformed response: {exc}; "
                               f"raw={str(data)[:300]}")
        usage = data.get("usage", {})
        in_tok = int(usage.get("prompt_tokens", 0))
        out_tok = int(usage.get("completion_tokens", 0))
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
        key = kc_get(KEY_AZURE_OPENAI)
        if not key:
            raise RuntimeError("AZURE_OPENAI_API_KEY not in keychain")
        t0 = now_ms()
        text_buf: list[str] = []
        in_tok = 0
        out_tok = 0
        with httpx.Client(timeout=120.0) as client:
            with client.stream("POST", self._url(),
                               headers=self._headers(key),
                               json=self._body(req, stream=True)) as r:
                if r.status_code != 200:
                    raise RuntimeError(f"Azure stream {r.status_code}: "
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
                        choices = ev.get("choices") or []
                        if not choices:
                            usage = ev.get("usage") or {}
                            if usage:
                                in_tok = int(usage.get("prompt_tokens", in_tok))
                                out_tok = int(usage.get("completion_tokens", out_tok))
                            continue
                        delta = choices[0].get("delta") or {}
                        chunk = delta.get("content") or ""
                        if chunk:
                            text_buf.append(chunk)
                            yield chunk
                    except (KeyError, IndexError):
                        continue
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
        key = kc_get(KEY_AZURE_OPENAI)
        if not key:
            self.health = HealthState(
                ok=False, last_rtt_ms=0.0,
                last_error="no API key in keychain",
                last_probe_at=time.time())
            return self.health
        body = {"messages": [{"role": "user", "content": "."}],
                "max_tokens": 1}
        t0 = now_ms()
        try:
            with httpx.Client(timeout=timeout) as client:
                r = client.post(self._url(), headers=self._headers(key), json=body)
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
