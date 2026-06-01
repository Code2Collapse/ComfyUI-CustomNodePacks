"""Qwen3-style ``<think>...</think>`` stripping.

Qwen3 / DeepSeek-R1 / o1-style reasoning models emit chain-of-thought wrapped
in ``<think>...</think>`` tags. End users almost never want that in the final
answer, and it confuses JSON parsers + token-cost accounting.

This module centralises the pattern so every backend response gets cleaned at
exactly one place (the router) and we never have to remember to add the regex
to a new backend.
"""
from __future__ import annotations

import re

# Non-greedy, DOTALL — handles multi-line reasoning blocks and multiple
# occurrences. Case-insensitive because some fine-tunes capitalise tag names.
THINK_RE = re.compile(r"<think\b[^>]*>.*?</think\s*>", re.DOTALL | re.IGNORECASE)


def strip_think(text: str) -> str:
    """Remove every ``<think>...</think>`` block from *text*.

    Idempotent. Safe on empty strings. Trims surrounding whitespace that the
    removal might leave behind, but preserves internal formatting.
    """
    if not text or "<think" not in text.lower():
        return text
    cleaned = THINK_RE.sub("", text)
    # Collapse the blank lines left behind by tag removal, but only at the
    # leading edge — keep internal paragraph breaks intact.
    return cleaned.lstrip()
