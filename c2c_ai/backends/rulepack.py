"""RulePackBackend — Tier 1 deterministic backend (Track D.1).

Wraps the existing 47-rule pack in ``nodes/error_assistant.py`` and presents
it through the standard ``Backend`` ABC. Cost = $0, latency typically <10 ms,
runs entirely offline. Selected by the router whenever:

  * ``Policy.DETERMINISTIC_ONLY`` is requested explicitly, OR
  * No LLM backend is healthy AND the feature is one of the error-explanation
    features (``doctor.explain``, ``error.explain``, ``plain.english``).

This makes the previously-failing ``/c2c/doctor/explain`` route return 200
with a useful plain-English answer even when zero AI backends are configured.

Input contract
--------------
The router passes a list of ``Message`` objects. For error-explanation calls
the convention (used by ``error_translator.translate_message`` and the new
``/c2c/doctor/explain`` rewire) is:

  * ``messages[-1].content`` — the raw error string (multi-line traceback OK).
  * Optional first system message of the form ``"exc_type: RuntimeError"`` to
    constrain pattern matching. If absent the backend probes the user message
    for the conventional ``"<ExcType>: <message>"`` prefix and falls back to
    ``"Exception"``.

The backend formats the matched ``Pattern`` (cause + fix steps) into a
single plain-English block. On *no match* it returns an empty-text response
so the router can escalate to the next tier without raising.
"""

from __future__ import annotations

import re
import time
from typing import Generator, List

from ..types import (
    AskRequest,
    AskResponse,
    BackendInfo,
    Capability,
    HealthState,
    Tier,
)
from .base import Backend, now_ms


# Importing the rule-pack lazily inside methods so importing this module never
# triggers the (relatively heavy) JSON loader at router-registration time.

_EXC_PREFIX_RE = re.compile(r"^\s*([A-Za-z_][\w\.]*)\s*:\s*", re.MULTILINE)


class RulePackBackend(Backend):
    """Pattern-pack-driven, zero-cost, deterministic explanation backend."""

    BACKEND_ID = "deterministic.rulepack"

    def __init__(self) -> None:
        super().__init__(
            BackendInfo(
                id=self.BACKEND_ID,
                tier=Tier.DETERMINISTIC,
                display_name="Built-in rule pack (offline)",
                model="error_assistant.patterns",
                capabilities={Capability.CHAT},
                max_context=8_192,
                cost_per_1k_input=0.0,
                cost_per_1k_output=0.0,
                enabled=True,
            )
        )
        # The pack is always "healthy" — it's a local JSON file. probe() will
        # downgrade this only if the pack file can't be loaded.
        self.health = HealthState(ok=True, last_rtt_ms=0.0, last_probe_at=time.time())

    # -------------------------------------------------------------- ask
    def ask(self, req: AskRequest) -> AskResponse:
        t0 = now_ms()
        raw, exc_type = self._extract(req)
        # Lazy import — the rule pack lives in nodes/ which depends on
        # folder_paths and other ComfyUI bits. Importing at module top would
        # block c2c_ai from loading before ComfyUI is ready.
        try:
            from ...nodes import error_assistant as _ea  # type: ignore[import-not-found]
        except Exception:
            _ea = None  # rule pack unreachable — return no-match so router escalates
        match = _ea.match_pattern(exc_type or "Exception", raw) if _ea is not None else None
        text = self._format(match, exc_type, raw)
        # Rough token estimates so the cost meter records a meaningful entry.
        in_tokens = max(1, len(raw) // 4)
        out_tokens = max(1, len(text) // 4) if text else 0
        return AskResponse(
            text=text,
            backend_id=self.info.id,
            model=self.info.model,
            input_tokens=in_tokens,
            output_tokens=out_tokens,
            cost_usd=0.0,
            latency_ms=now_ms() - t0,
            redacted=False,
        )

    def stream(self, req: AskRequest) -> Generator[str, None, AskResponse]:
        # Deterministic output — chunk per fix-step so the UI can render
        # progressively just like an LLM stream.
        resp = self.ask(req)
        if resp.text:
            for line in resp.text.splitlines(keepends=True):
                yield line
        return resp

    # ------------------------------------------------------------ probe
    def probe(self, timeout: float = 5.0) -> HealthState:
        """Lightweight: confirm the rule pack loads and returns >=1 pattern."""
        t0 = now_ms()
        try:
            from ...nodes import error_assistant as _ea  # type: ignore[import-not-found]
            count = len(_ea._get_patterns())  # noqa: SLF001 — module-private but stable
            self.health = HealthState(
                ok=count > 0,
                last_rtt_ms=now_ms() - t0,
                last_error=None if count > 0 else "rule pack returned 0 patterns",
                last_probe_at=time.time(),
            )
        except Exception as exc:  # pragma: no cover — defensive
            self.health = HealthState(
                ok=False,
                last_rtt_ms=now_ms() - t0,
                last_error=f"{type(exc).__name__}: {exc}",
                last_probe_at=time.time(),
            )
        return self.health

    def list_models(self, timeout: float = 5.0) -> list[str]:
        return [self.info.model]

    # ----------------------------------------------------------- helpers
    @staticmethod
    def _extract(req: AskRequest) -> tuple[str, str | None]:
        """Pull (raw_error_text, exc_type) out of the message list.

        Convention: caller may set the first system message to
        ``"exc_type: RuntimeError"`` to constrain matching; otherwise we try
        to parse the conventional ``"<ExcType>: <msg>"`` prefix from the
        user message.
        """
        raw = ""
        exc_type: str | None = None
        for m in req.messages:
            if m.role == "system" and m.content.startswith("exc_type:"):
                exc_type = m.content.split(":", 1)[1].strip()
            elif m.role == "user":
                raw = m.content
        if not exc_type and raw:
            mat = _EXC_PREFIX_RE.search(raw)
            if mat:
                exc_type = mat.group(1)
        return raw, exc_type

    @staticmethod
    def _format(match, exc_type: str | None, raw: str) -> str:
        """Render a matched Pattern as a single plain-English answer.

        Returns ``""`` on no-match so the router can escalate to the next
        tier (introspector / cloud LLM) without us having to raise.
        """
        if match is None:
            return ""
        lines: List[str] = []
        head = exc_type or "Error"
        lines.append(f"**{head}** \u2014 {match.cause}".rstrip())
        if match.fixes:
            lines.append("")
            lines.append("**Try this:**")
            for step in match.fixes:
                lines.append(f"\u2022 {step}")
        if match.category and match.category != "uncategorized":
            lines.append("")
            lines.append(f"_Category: {match.category} \u00b7 confidence: {match.confidence:.0%}_")
        return "\n".join(lines).strip()
