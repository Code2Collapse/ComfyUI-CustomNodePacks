"""Payload redactor — strips private identifiers before cloud calls.

Run automatically by the router whenever the active backend is in
``Tier.CLOUD`` and the request's :class:`Sensitivity` is ``NORMAL`` or
``SENSITIVE``. ``PUBLIC`` requests pass through untouched.

What we scrub:
    * Absolute Windows paths     C:\\Users\\name\\... → <PATH>
    * Absolute POSIX home paths  /home/name/...     → <PATH>
    * macOS home paths           /Users/name/...    → <PATH>
    * Email addresses            foo@bar.com        → <EMAIL>
    * Bearer / API keys          sk-ant-..., sk-..., hf_..., gho_..., etc. → <KEY>
    * UUID v4                    11111111-2222-...  → <UUID>
    * IPv4                       192.168.x.y        → <IP>
    * Long hex hashes (>=32)     a1b2c3d4...        → <HASH>
    * Windows usernames inside env-var-style refs (%USERNAME%, $env:UserProfile)

What we deliberately do NOT scrub:
    * Code identifiers, node class names, model names — these are essential
      for the AI to understand the question
    * Numeric widget values, resolutions, seeds — same reason

Sensitivity == SENSITIVE additionally:
    * Blocks the call entirely unless ``allow_cloud=True`` is asserted by
      the caller after explicit user confirmation
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .types import Message, Sensitivity


# Order matters — longer patterns first.
_PATTERNS: tuple[tuple[str, re.Pattern[str], str], ...] = (
    ("PATH",  re.compile(r"[A-Za-z]:\\(?:[^\\\s\"'<>|*?:]+\\)+[^\\\s\"'<>|*?:]+"), "<PATH>"),
    ("PATH",  re.compile(r"/(?:home|Users)/[A-Za-z0-9._-]+(?:/[^\s\"'<>|*?:]+)*"), "<PATH>"),
    ("EMAIL", re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"), "<EMAIL>"),
    # API keys — common prefixes across providers
    ("KEY",   re.compile(r"\bsk-(?:ant-)?[A-Za-z0-9_\-]{20,}\b"), "<KEY>"),
    ("KEY",   re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"), "<KEY>"),
    ("KEY",   re.compile(r"\bgh[opsu]_[A-Za-z0-9]{20,}\b"), "<KEY>"),
    ("KEY",   re.compile(r"\bxoxb-[A-Za-z0-9-]{20,}\b"), "<KEY>"),
    ("UUID",  re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"), "<UUID>"),
    ("IP",    re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"), "<IP>"),
    ("HASH",  re.compile(r"\b[0-9a-fA-F]{32,}\b"), "<HASH>"),
    ("ENV",   re.compile(r"%USER(?:NAME|PROFILE)%", re.IGNORECASE), "<USER>"),
    ("ENV",   re.compile(r"\$env:User(?:Name|Profile)", re.IGNORECASE), "<USER>"),
)


@dataclass
class RedactionResult:
    text: str
    counts: dict[str, int]    # category → number of substitutions

    @property
    def any_redacted(self) -> bool:
        return sum(self.counts.values()) > 0


def redact_text(s: str) -> RedactionResult:
    counts: dict[str, int] = {}
    out = s
    for label, pat, repl in _PATTERNS:
        out, n = pat.subn(repl, out)
        if n:
            counts[label] = counts.get(label, 0) + n
    return RedactionResult(text=out, counts=counts)


def redact_messages(messages: list[Message], sensitivity: Sensitivity) -> tuple[list[Message], bool]:
    """Return a redacted copy of ``messages`` plus a flag if anything changed.

    ``PUBLIC`` sensitivity bypasses redaction entirely.
    """
    if sensitivity == Sensitivity.PUBLIC:
        return messages, False
    any_changed = False
    out: list[Message] = []
    for m in messages:
        r = redact_text(m.content)
        if r.any_redacted:
            any_changed = True
        out.append(Message(role=m.role, content=r.text))
    return out, any_changed


def preview(text: str, sensitivity: Sensitivity = Sensitivity.NORMAL) -> RedactionResult:
    """Useful for the 'Show me what would be sent' button in the UI."""
    if sensitivity == Sensitivity.PUBLIC:
        return RedactionResult(text=text, counts={})
    return redact_text(text)
