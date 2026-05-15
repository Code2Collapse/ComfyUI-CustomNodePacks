# Prompt Relay — refined parser (inline + block syntax).
#
# Refined port of Gordon Chen's Prompt Relay smart-prompt parser. Algorithm
# unchanged; renamed private helpers and tightened type signatures.
# See ``NOTICE.md`` for full attribution.

from __future__ import annotations

import re
from typing import List, Optional, Tuple

_INLINE_TAG_RE = re.compile(r"\[([\d\.]+)(?:[:\-]([\d\.]+))?\]")
_DIGIT_RANGE_TAIL_RE = re.compile(r"([\d]+(?:\.\d+)?)\s*[-\u2013]\s*([\d]+(?:\.\d+)?)\s*$")


def _try_parse_num(s: str) -> Optional[int]:
    s = s.strip()
    try:
        return int(float(s))
    except (ValueError, TypeError):
        pass
    try:
        from word2number import w2n
        return int(w2n.word_to_num(s))
    except Exception:
        return None


def _parse_header(line: str) -> Optional[Tuple[int, Optional[int]]]:
    """Detect a block-syntax header line. Returns ``(start, end_or_None)`` or None."""
    line = line.strip()
    if not line.endswith(":"):
        return None
    body = line[:-1].rstrip()
    tokens = body.split()
    if len(tokens) < 2:
        return None
    m = _DIGIT_RANGE_TAIL_RE.search(body)
    if m and body[: m.start()].strip():
        start = _try_parse_num(m.group(1))
        end = _try_parse_num(m.group(2))
        if start is not None and end is not None:
            return (start, end)
    max_num_tokens = min(4, len(tokens) - 1)
    for n in range(max_num_tokens, 0, -1):
        candidate = " ".join(tokens[-n:])
        val = _try_parse_num(candidate)
        if val is not None:
            return (val, None)
    return None


def _extract_inline_tag(text: str) -> Tuple[str, Optional[float]]:
    m = _INLINE_TAG_RE.search(text)
    if not m:
        return text.strip(), None
    val1 = float(m.group(1))
    val2 = float(m.group(2)) if m.group(2) else None
    weight = (val2 - val1) if val2 is not None else val1
    clean = _INLINE_TAG_RE.sub("", text).strip()
    return clean, weight


def _parse_inline_syntax(text: str) -> List[dict]:
    segments: List[dict] = []
    for part in text.split("|"):
        clean, weight = _extract_inline_tag(part)
        if clean:
            segments.append({"text": clean, "weight": weight if weight is not None else 1.0})
    return segments


def _parse_block_syntax(text: str) -> List[dict]:
    lines = text.splitlines(keepends=True)
    raw_segments: List[Tuple[Optional[Tuple[int, Optional[int]]], str]] = []
    current_header: Optional[Tuple[int, Optional[int]]] = None
    current_body: List[str] = []
    for line in lines:
        h = _parse_header(line)
        if h is not None:
            if current_body or current_header is not None:
                raw_segments.append((current_header, "".join(current_body)))
            current_header = h
            current_body = []
        else:
            current_body.append(line)
    if current_body or current_header is not None:
        raw_segments.append((current_header, "".join(current_body)))

    segments: List[dict] = []
    for header, body in raw_segments:
        clean, inline_weight = _extract_inline_tag(body)
        if not clean:
            continue
        if inline_weight is not None:
            weight = inline_weight
        elif header is not None:
            start, end = header
            weight = (end - start) if end is not None else 1.0
        else:
            weight = 1.0
        segments.append({"text": clean, "weight": float(weight)})
    return segments


def parse_smart_prompt(text: str) -> List[dict]:
    """Parse ``smart_prompt`` text into ``[{"text": str, "weight": float}, ...]``.

    Auto-detects inline (``|``-separated) vs. block (header-line) syntax.
    """
    lines = text.splitlines()
    has_blocks = any(_parse_header(line) is not None for line in lines)
    if has_blocks:
        return _parse_block_syntax(text)
    return _parse_inline_syntax(text)
