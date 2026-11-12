"""Google Gemini backend.

Uses Generative Language REST API (v1beta) directly via httpx — no SDK.
Auth is via the ``GEMINI_API_KEY`` in the keychain (also accepted as
``GOOGLE_API_KEY`` by the txt parser). The key travels in the ``x-goog-api-key`` header (documented by Google), never
in the URL: httpx error messages carry the URL, so a ``?key=`` ended up in logs and error text (L6.03).

Models: the settings picker lists them live (GET v1beta/models). Stable at aistudio.google.com/docs/models on
2026-11-12: gemini-3.6-flash, gemini-3.5-flash-lite, gemini-3.1-flash-lite; Gemini 1.5 / 2.0 are gone.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Generator

import httpx

from .base import Backend, check_cancel, now_ms, req_timeout
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
              model: str = "gemini-3.6-flash",
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
                parts = [{"text": m.content}] if m.content else []
                parts += [{"inlineData": {"mimeType": im.mime, "data": im.b64()}}
                          for im in getattr(m, "images", ())]
                contents.append({"role": role, "parts": parts})
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
        url = f"{BASE}/{self.info.model}:generateContent"
        t0 = now_ms()
        check_cancel(req)
        with httpx.Client(timeout=req_timeout(req, 60.0)) as client:
            r = client.post(url, json=self._build_body(req), headers={"x-goog-api-key": key})
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
               "?alt=sse")
        t0 = now_ms()
        text_buf: list[str] = []
        in_tok = 0
        out_tok = 0
        check_cancel(req)
        with httpx.Client(timeout=req_timeout(req, 120.0)) as client:
            with client.stream("POST", url, json=self._build_body(req),
                               headers={"accept": "text/event-stream", "x-goog-api-key": key}) as r:
                if r.status_code != 200:
                    raise RuntimeError(f"Gemini stream {r.status_code}: "
                                       f"{r.read()[:400]!r}")
                for line in r.iter_lines():
                    check_cancel(req)
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
        url = f"{BASE}/{self.info.model}:generateContent"
        body = {
            "contents": [{"role": "user", "parts": [{"text": "."}]}],
            "generationConfig": {"maxOutputTokens": 1},
        }
        t0 = now_ms()
        try:
            with httpx.Client(timeout=timeout) as client:
                r = client.post(url, json=body, headers={"x-goog-api-key": key})
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

    # ---------------------------------------------------------- list_models
    # Offline fallback (stable models on aistudio.google.com/docs/models, checked 2026-11-12; Gemini 1.5 / 2.0 are
    # gone). The picker normally shows the live list.
    KNOWN_MODELS: tuple[str, ...] = (
        "gemini-3.6-flash",
        "gemini-3.5-flash-lite",
        "gemini-3.1-flash-lite",
    )

    def list_models(self, timeout: float = 5.0) -> list[str]:
        """Live: GET v1beta/models (models[].name = "models/<id>"), only models whose supportedGenerationMethods
        include generateContent (the list also holds embedding / TTS models). Key in the x-goog-api-key header."""
        live: list[str] = []
        key = kc_get(KEY_GEMINI)
        if key:
            try:
                with httpx.Client(timeout=timeout) as client:
                    r = client.get(BASE, params={"pageSize": 1000}, headers={"x-goog-api-key": key})
                if r.status_code == 200:
                    for m in r.json().get("models") or []:
                        name = str(m.get("name") or "")
                        if name and "generateContent" in (m.get("supportedGenerationMethods") or []):
                            live.append(name.split("/", 1)[1] if name.startswith("models/") else name)
            except Exception as exc:  # noqa: BLE001
                log.info("gemini list_models failed: %s", type(exc).__name__)
        return _merge_models(self.info.model, live, self.KNOWN_MODELS)


def _merge_models(configured: str, live: list, known: tuple) -> list:
    """Configured model first (never dropped), then the live list, else the curated fallback."""
    out = [configured] if configured else []
    for m in (live or known):
        if m and m not in out:
            out.append(m)
    return out
