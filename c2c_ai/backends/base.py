"""Abstract base class for backends."""

from __future__ import annotations

import abc
import time
from typing import Generator

from ..types import (
    AskRequest,
    AskResponse,
    BackendInfo,
    Capability,
    HealthState,
    Message,
)


class Backend(abc.ABC):
    info: BackendInfo
    health: HealthState

    def __init__(self, info: BackendInfo) -> None:
        self.info = info
        self.health = HealthState()

    # ---- core operations -----------------------------------------------------

    @abc.abstractmethod
    def ask(self, req: AskRequest) -> AskResponse:
        """Single non-streaming request. MUST return ``AskResponse`` on success
        and raise on failure. Implementations should populate latency_ms and
        token counts as accurately as possible."""

    def stream(self, req: AskRequest) -> Generator[str, None, AskResponse]:
        """Optional streaming. Default implementation degrades to ask()."""
        resp = self.ask(req)
        yield resp.text
        return resp

    @abc.abstractmethod
    def probe(self, timeout: float = 5.0) -> HealthState:
        """Refresh health state. MUST update ``self.health`` and return it."""

    # ---- model discovery -----------------------------------------------------

    def list_models(self, timeout: float = 5.0) -> list[str]:
        """Return the list of model identifiers this backend can serve.

        Default returns ``[self.info.model]`` — the single currently configured
        model. Concrete backends override this to:
          * Cloud providers with stable curated lists (Anthropic, Gemini,
            Cohere) — return a static tuple of currently supported models.
          * OpenAI-compatible servers (cloud + local Ollama/LM Studio) —
            live-fetch ``GET {base_url}/v1/models``.
          * llama.cpp local — enumerate GGUFs in the ``text_encoders`` folder.

        Implementations MUST always include ``self.info.model`` in the result
        so the currently-selected model is never dropped from the picker (even
        if the live remote list temporarily excludes it).

        Failures (network error, no key) MUST NOT raise — return
        ``[self.info.model]`` so the picker always has at least one choice.
        """
        return [self.info.model] if self.info.model else []

    # ---- helpers -------------------------------------------------------------

    def supports(self, cap: Capability) -> bool:
        return cap in self.info.capabilities

    def __repr__(self) -> str:
        return f"<Backend {self.info.id} model={self.info.model} ok={self.health.ok}>"


def now_ms() -> float:
    return time.monotonic() * 1000.0
