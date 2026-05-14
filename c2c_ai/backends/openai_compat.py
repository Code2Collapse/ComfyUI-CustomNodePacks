"""OpenAI-compatible backend — works with Ollama, LM Studio, llama.cpp's
``llama-server``, vLLM, text-generation-webui (with the openai extension),
and OpenAI itself.

The detection helpers below probe well-known local ports so the first-run
wizard can offer "auto-detect" without the user typing URLs.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Generator, Iterable

import httpx

from .base import Backend, now_ms
from ..keychain import get as kc_get
from ..types import (
    AskRequest,
    AskResponse,
    BackendInfo,
    Capability,
    HealthState,
    Tier,
)

log = logging.getLogger("c2c_ai.openai_compat")


class OpenAICompatBackend(Backend):
    """Generic adapter — same code path for cloud OpenAI and local servers.

    ``base_url`` should NOT include the trailing ``/v1`` — we add it. (Some
    servers expose ``/v1/chat/completions`` and a sibling ``/v1/models``.)
    """

    def __init__(self, info: BackendInfo, *, base_url: str, api_key_name: str | None,
                 api_key_inline: str | None = None) -> None:
        super().__init__(info)
        self.base_url = base_url.rstrip("/")
        self.api_key_name = api_key_name
        self.api_key_inline = api_key_inline

    # ----------------------------------------------- factory: cloud OpenAI
    @classmethod
    def openai(cls, model: str = "gpt-4o-mini",
               cost_per_1k_input: float = 0.00015,
               cost_per_1k_output: float = 0.0006) -> "OpenAICompatBackend":
        info = BackendInfo(
            id="cloud.openai",
            tier=Tier.CLOUD,
            display_name="OpenAI (GPT)",
            model=model,
            capabilities={Capability.CHAT, Capability.STREAMING,
                          Capability.JSON_MODE, Capability.TOOL_USE,
                          Capability.VISION},
            max_context=128_000,
            cost_per_1k_input=cost_per_1k_input,
            cost_per_1k_output=cost_per_1k_output,
        )
        return cls(info, base_url="https://api.openai.com", api_key_name="OPENAI_API_KEY")

    # -------------------------------------------- factory: Qwen cloud (DashScope-compat)
    @classmethod
    def qwen(cls, model: str = "qwen-max-latest",
             cost_per_1k_input: float = 0.0008,
             cost_per_1k_output: float = 0.002) -> "OpenAICompatBackend":
        info = BackendInfo(
            id="cloud.qwen",
            tier=Tier.CLOUD,
            display_name="Qwen (Alibaba Cloud)",
            model=model,
            capabilities={Capability.CHAT, Capability.STREAMING, Capability.JSON_MODE},
            max_context=32_000,
            cost_per_1k_input=cost_per_1k_input,
            cost_per_1k_output=cost_per_1k_output,
        )
        # DashScope's OpenAI-compat endpoint
        return cls(info,
                   base_url="https://dashscope-intl.aliyuncs.com/compatible-mode",
                   api_key_name="QWEN_API_KEY")

    # -------------------------------------------- factory: OpenRouter
    @classmethod
    def openrouter(cls, model: str = "anthropic/claude-3.5-sonnet") -> "OpenAICompatBackend":
        info = BackendInfo(
            id="cloud.openrouter",
            tier=Tier.CLOUD,
            display_name="OpenRouter",
            model=model,
            capabilities={Capability.CHAT, Capability.STREAMING},
            max_context=200_000,
            cost_per_1k_input=0.003,
            cost_per_1k_output=0.015,
        )
        return cls(info, base_url="https://openrouter.ai/api",
                   api_key_name="OPENROUTER_API_KEY")

    # -------------------------------------------- factory: local server
    @classmethod
    def local(cls,
              backend_id: str,
              display_name: str,
              base_url: str,
              model: str,
              max_context: int = 32_768) -> "OpenAICompatBackend":
        info = BackendInfo(
            id=backend_id,
            tier=Tier.LOCAL,
            display_name=display_name,
            model=model,
            capabilities={Capability.CHAT, Capability.STREAMING, Capability.JSON_MODE},
            max_context=max_context,
            cost_per_1k_input=0.0,
            cost_per_1k_output=0.0,
        )
        return cls(info, base_url=base_url, api_key_name=None)

    # =========================================================== headers
    def _headers(self) -> dict[str, str]:
        h = {"content-type": "application/json"}
        key = self.api_key_inline
        if not key and self.api_key_name:
            key = kc_get(self.api_key_name)
        if key:
            h["authorization"] = f"Bearer {key}"
        # OpenRouter polite identification (optional but recommended)
        if self.info.id == "cloud.openrouter":
            h["http-referer"] = "https://github.com/Code2Collapse/ComfyUI-CustomNodePacks"
            h["x-title"] = "C2C ComfyUI"
        return h

    # =============================================================== ask
    def ask(self, req: AskRequest) -> AskResponse:
        url = f"{self.base_url}/v1/chat/completions"
        body: dict = {
            "model": self.info.model,
            "messages": [{"role": m.role, "content": m.content} for m in req.messages],
            "max_tokens": req.max_tokens,
            "temperature": req.temperature,
        }
        if req.json_schema and Capability.JSON_MODE in self.info.capabilities:
            body["response_format"] = {"type": "json_object"}

        t0 = now_ms()
        try:
            with httpx.Client(timeout=120.0) as client:
                r = client.post(url, headers=self._headers(), json=body)
        except httpx.RequestError as exc:
            raise RuntimeError(f"{self.info.id}: connection failed: {exc}")
        latency = now_ms() - t0

        if r.status_code != 200:
            raise RuntimeError(f"{self.info.id} {r.status_code}: {r.text[:400]}")

        data = r.json()
        choice = (data.get("choices") or [{}])[0]
        text = (choice.get("message") or {}).get("content", "") or ""
        usage = data.get("usage") or {}
        in_tok = int(usage.get("prompt_tokens", 0))
        out_tok = int(usage.get("completion_tokens", 0))
        if in_tok == 0 and out_tok == 0:
            # Local servers sometimes omit usage; rough estimate so cost meter still records calls
            in_tok = max(1, sum(len(m.content) for m in req.messages) // 4)
            out_tok = max(1, len(text) // 4)
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

    # ============================================================ stream
    def stream(self, req: AskRequest) -> Generator[str, None, AskResponse]:
        url = f"{self.base_url}/v1/chat/completions"
        body: dict = {
            "model": self.info.model,
            "messages": [{"role": m.role, "content": m.content} for m in req.messages],
            "max_tokens": req.max_tokens,
            "temperature": req.temperature,
            "stream": True,
        }
        t0 = now_ms()
        text_buf: list[str] = []
        in_tok = 0
        out_tok = 0
        with httpx.Client(timeout=180.0) as client:
            with client.stream("POST", url, headers=self._headers(), json=body) as r:
                if r.status_code != 200:
                    raise RuntimeError(f"{self.info.id} stream {r.status_code}: {r.read()[:400]!r}")
                for line in r.iter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    payload = line[6:].strip()
                    if payload == "[DONE]":
                        break
                    try:
                        ev = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    delta = (ev.get("choices") or [{}])[0].get("delta", {})
                    chunk = delta.get("content", "") or ""
                    if chunk:
                        text_buf.append(chunk)
                        yield chunk
                    usage = ev.get("usage")
                    if usage:
                        in_tok = int(usage.get("prompt_tokens", in_tok))
                        out_tok = int(usage.get("completion_tokens", out_tok))
        latency = now_ms() - t0
        if in_tok == 0 and out_tok == 0:
            in_tok = max(1, sum(len(m.content) for m in req.messages) // 4)
            out_tok = max(1, len("".join(text_buf)) // 4)
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

    # ============================================================= probe
    def probe(self, timeout: float = 3.0) -> HealthState:
        url = f"{self.base_url}/v1/models"
        t0 = now_ms()
        try:
            with httpx.Client(timeout=timeout) as client:
                r = client.get(url, headers=self._headers())
            rtt = now_ms() - t0
            ok = r.status_code == 200
            err = None if ok else f"HTTP {r.status_code}"
        except Exception as exc:
            rtt = now_ms() - t0
            ok = False
            err = type(exc).__name__ + ": " + str(exc)[:200]
        self.health = HealthState(ok=ok, last_rtt_ms=rtt, last_error=err, last_probe_at=time.time())
        return self.health


# ============================================================ discovery

# Well-known local server endpoints, in priority order.
KNOWN_LOCAL_SERVERS: tuple[dict, ...] = (
    {"id": "local.ollama",     "name": "Ollama",            "base_url": "http://127.0.0.1:11434"},
    {"id": "local.lmstudio",   "name": "LM Studio",         "base_url": "http://127.0.0.1:1234"},
    {"id": "local.llamacpp",   "name": "llama.cpp server",  "base_url": "http://127.0.0.1:8080"},
    {"id": "local.vllm",       "name": "vLLM",              "base_url": "http://127.0.0.1:8000"},
    {"id": "local.tgw",        "name": "text-generation-webui", "base_url": "http://127.0.0.1:5000"},
    {"id": "local.c2c",        "name": "C2C bundled llama.cpp", "base_url": "http://127.0.0.1:8765"},
)


def detect_local_servers(timeout: float = 1.5) -> list[dict]:
    """Probe each well-known port, return the list of servers that respond.

    Each entry: ``{"id": str, "name": str, "base_url": str, "models": [str, ...]}``
    """
    found: list[dict] = []
    for entry in KNOWN_LOCAL_SERVERS:
        try:
            with httpx.Client(timeout=timeout) as client:
                r = client.get(entry["base_url"].rstrip("/") + "/v1/models")
            if r.status_code != 200:
                continue
            data = r.json()
            models = [m.get("id") for m in data.get("data", []) if m.get("id")]
            found.append({**entry, "models": models})
        except Exception:
            continue
    return found
