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

    # ---- helpers -------------------------------------------------------------

    def supports(self, cap: Capability) -> bool:
        return cap in self.info.capabilities

    def __repr__(self) -> str:
        return f"<Backend {self.info.id} model={self.info.model} ok={self.health.ok}>"


def now_ms() -> float:
    return time.monotonic() * 1000.0
