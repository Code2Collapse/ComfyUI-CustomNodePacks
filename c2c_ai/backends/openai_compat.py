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
                 api_key_inline: str | None = None,
                 api_path: str = "/v1") -> None:
        super().__init__(info)
        self.base_url = base_url.rstrip("/")
        self.api_key_name = api_key_name
        self.api_key_inline = api_key_inline
        # Path prefix before `/chat/completions` and `/models`. Most providers
        # use `/v1` but Perplexity (and a handful of niche servers) omit it.
        self.api_path = "/" + api_path.strip("/") if api_path else ""

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

    # -------------------------------------------- factory: xAI (Grok)
    @classmethod
    def xai(cls, model: str = "grok-3-mini-fast",
            cost_per_1k_input: float = 0.0003,
            cost_per_1k_output: float = 0.0005) -> "OpenAICompatBackend":
        info = BackendInfo(
            id="cloud.xai",
            tier=Tier.CLOUD,
            display_name="Grok (xAI)",
            model=model,
            capabilities={Capability.CHAT, Capability.STREAMING,
                          Capability.JSON_MODE, Capability.TOOL_USE},
            max_context=131_072,
            cost_per_1k_input=cost_per_1k_input,
            cost_per_1k_output=cost_per_1k_output,
        )
        return cls(info, base_url="https://api.x.ai", api_key_name="XAI_API_KEY")

    # -------------------------------------------- factory: Groq (fast inference)
    @classmethod
    def groq(cls, model: str = "llama-3.3-70b-versatile",
             cost_per_1k_input: float = 0.00059,
             cost_per_1k_output: float = 0.00079) -> "OpenAICompatBackend":
        info = BackendInfo(
            id="cloud.groq",
            tier=Tier.CLOUD,
            display_name="Groq (LPU)",
            model=model,
            capabilities={Capability.CHAT, Capability.STREAMING,
                          Capability.JSON_MODE, Capability.TOOL_USE},
            max_context=131_072,
            cost_per_1k_input=cost_per_1k_input,
            cost_per_1k_output=cost_per_1k_output,
        )
        return cls(info, base_url="https://api.groq.com/openai",
                   api_key_name="GROQ_API_KEY")

    # -------------------------------------------- factory: Mistral
    @classmethod
    def mistral(cls, model: str = "mistral-small-latest",
                cost_per_1k_input: float = 0.0002,
                cost_per_1k_output: float = 0.0006) -> "OpenAICompatBackend":
        info = BackendInfo(
            id="cloud.mistral",
            tier=Tier.CLOUD,
            display_name="Mistral",
            model=model,
            capabilities={Capability.CHAT, Capability.STREAMING,
                          Capability.JSON_MODE, Capability.TOOL_USE},
            max_context=128_000,
            cost_per_1k_input=cost_per_1k_input,
            cost_per_1k_output=cost_per_1k_output,
        )
        return cls(info, base_url="https://api.mistral.ai",
                   api_key_name="MISTRAL_API_KEY")

    # -------------------------------------------- factory: DeepSeek
    @classmethod
    def deepseek(cls, model: str = "deepseek-chat",
                 cost_per_1k_input: float = 0.00014,
                 cost_per_1k_output: float = 0.00028) -> "OpenAICompatBackend":
        info = BackendInfo(
            id="cloud.deepseek",
            tier=Tier.CLOUD,
            display_name="DeepSeek",
            model=model,
            capabilities={Capability.CHAT, Capability.STREAMING,
                          Capability.JSON_MODE, Capability.TOOL_USE},
            max_context=65_536,
            cost_per_1k_input=cost_per_1k_input,
            cost_per_1k_output=cost_per_1k_output,
        )
        return cls(info, base_url="https://api.deepseek.com",
                   api_key_name="DEEPSEEK_API_KEY")

    # -------------------------------------------- factory: Together.ai
    @classmethod
    def together(cls, model: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo",
                 cost_per_1k_input: float = 0.00088,
                 cost_per_1k_output: float = 0.00088) -> "OpenAICompatBackend":
        info = BackendInfo(
            id="cloud.together",
            tier=Tier.CLOUD,
            display_name="Together.ai",
            model=model,
            capabilities={Capability.CHAT, Capability.STREAMING, Capability.JSON_MODE},
            max_context=131_072,
            cost_per_1k_input=cost_per_1k_input,
            cost_per_1k_output=cost_per_1k_output,
        )
        return cls(info, base_url="https://api.together.xyz",
                   api_key_name="TOGETHER_API_KEY")

    # -------------------------------------------- factory: Fireworks.ai
    @classmethod
    def fireworks(cls, model: str = "accounts/fireworks/models/llama-v3p3-70b-instruct",
                  cost_per_1k_input: float = 0.0009,
                  cost_per_1k_output: float = 0.0009) -> "OpenAICompatBackend":
        info = BackendInfo(
            id="cloud.fireworks",
            tier=Tier.CLOUD,
            display_name="Fireworks.ai",
            model=model,
            capabilities={Capability.CHAT, Capability.STREAMING, Capability.JSON_MODE},
            max_context=131_072,
            cost_per_1k_input=cost_per_1k_input,
            cost_per_1k_output=cost_per_1k_output,
        )
        return cls(info, base_url="https://api.fireworks.ai/inference",
                   api_key_name="FIREWORKS_API_KEY")

    # -------------------------------------------- factory: Perplexity (web-grounded)
    @classmethod
    def perplexity(cls, model: str = "sonar",
                   cost_per_1k_input: float = 0.001,
                   cost_per_1k_output: float = 0.001) -> "OpenAICompatBackend":
        info = BackendInfo(
            id="cloud.perplexity",
            tier=Tier.CLOUD,
            display_name="Perplexity",
            model=model,
            capabilities={Capability.CHAT, Capability.STREAMING},
            max_context=127_072,
            cost_per_1k_input=cost_per_1k_input,
            cost_per_1k_output=cost_per_1k_output,
        )
        return cls(info, base_url="https://api.perplexity.ai",
                   api_key_name="PERPLEXITY_API_KEY", api_path="")


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
        url = f"{self.base_url}{self.api_path}/chat/completions"
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
        url = f"{self.base_url}{self.api_path}/chat/completions"
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
        url = f"{self.base_url}{self.api_path}/models"
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

    # ========================================================= list_models
    def list_models(self, timeout: float = 5.0) -> list[str]:
        """Live-fetch ``GET {base_url}{api_path}/models`` and return the IDs.

        The OpenAI-compatible response shape is ``{"data":[{"id":...}, ...]}``
        for OpenAI/Qwen/OpenRouter/xAI/Groq/Mistral/DeepSeek/Together/Fireworks/
        Perplexity AND for local Ollama (via its OpenAI-compat shim) and
        LM Studio. Any failure (no key, network, non-200, bad shape) falls
        back to the single configured model so the picker is never empty.
        """
        url = f"{self.base_url}{self.api_path}/models"
        try:
            with httpx.Client(timeout=timeout) as client:
                r = client.get(url, headers=self._headers())
            if r.status_code != 200:
                return [self.info.model] if self.info.model else []
            data = r.json()
        except Exception:
            return [self.info.model] if self.info.model else []
        items = data.get("data") if isinstance(data, dict) else None
        if not isinstance(items, list):
            return [self.info.model] if self.info.model else []
        ids: list[str] = []
        for it in items:
            mid = (it or {}).get("id") if isinstance(it, dict) else None
            if isinstance(mid, str) and mid:
                ids.append(mid)
        # Pin configured model at position 0 so the active selection is always
        # present (some providers' /v1/models returns embeddings + image
        # models the chat endpoint can't actually serve).
        out: list[str] = []
        if self.info.model:
            out.append(self.info.model)
        for m in ids:
            if m not in out:
                out.append(m)
        return out


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
