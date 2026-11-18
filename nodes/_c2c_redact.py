"""One secret redactor for every C2C pack surface (L3 S6): logs, registry status, diagnostics reports.

Only SECRETS are removed here: provider key shapes, Authorization / Bearer / x-api-key / api_key= values, and the exact
values of the key-like environment variables that are set (``*_API_KEY``, ``*_KEY``, ``*_TOKEN``, ``*_SECRET``,
``HF_TOKEN`` ...). Paths, e-mails and ids stay readable here; the AI router's payload redactor (c2c_ai/redactor.py)
adds those for cloud calls and takes its key rules from this module.

``install_log_filter()`` adds one filter to the root logger's handlers. It acts only on records whose logger name
starts with ``C2C`` / ``c2c`` / ``MEC`` (a filter on a parent logger would miss its children's records) and rewrites
the message and any traceback text. It never raises: a failing redaction leaves the record as it was.
"""
from __future__ import annotations

import logging
import os
import re
import time

KEY = "<KEY>"

# provider key shapes (order: longer / more specific first)
KEY_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bsk-(?:ant-|proj-|or-v1-)?[A-Za-z0-9_\-]{20,}"),       # Anthropic, OpenAI, OpenRouter
    re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"),                              # Hugging Face
    re.compile(r"\bgh[opsu]_[A-Za-z0-9]{20,}\b"),                        # GitHub
    re.compile(r"\bxox[abp]-[A-Za-z0-9-]{20,}\b"),                       # Slack
    re.compile(r"\bAIza[0-9A-Za-z_\-]{35}"),                             # Google AI Studio / Gemini
    re.compile(r"\bgsk_[A-Za-z0-9]{20,}\b"),                             # Groq
    re.compile(r"\bxai-[A-Za-z0-9]{20,}\b"),                             # xAI
    re.compile(r"\bpplx-[A-Za-z0-9]{20,}\b"),                            # Perplexity
    re.compile(r"\bfw_[A-Za-z0-9]{20,}\b"),                              # Fireworks
    re.compile(r"\br8_[A-Za-z0-9]{20,}\b"),                              # Replicate
)
# a value after an auth header / parameter name, whatever its shape
VALUE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?i)(\bBearer\s+)[A-Za-z0-9._~+/=\-]{8,}"),
    re.compile(r"(?i)(\b(?:authorization|x-api-key|x-goog-api-key|api[_-]?key|access[_-]?token|token)"
               r"[\"']?\s*[:=]\s*[\"']?)(?!<KEY>)[A-Za-z0-9._~+/=\-]{8,}"),
)
_ENV_NAME = re.compile(r"(?i)(?:^|_)(?:API_?KEY|KEY|TOKEN|SECRET|PASSWORD)$")
_ENV_CACHE: dict = {"t": 0.0, "values": ()}


def secret_env_values(min_len: int = 12) -> tuple[str, ...]:
    """Values of the key-like environment variables that are set (cached 5 s; longest first)."""
    now = time.monotonic()
    if now - _ENV_CACHE["t"] > 5.0:
        vals = {v.strip() for k, v in os.environ.items() if _ENV_NAME.search(k) and v and len(v.strip()) >= min_len}
        _ENV_CACHE["values"] = tuple(sorted(vals, key=len, reverse=True))
        _ENV_CACHE["t"] = now
    return _ENV_CACHE["values"]


def redact_secrets(text, extra_values=()) -> tuple[str, int]:
    """``text`` with every secret replaced by <KEY>, and the number of replacements. Non-strings pass through."""
    if not isinstance(text, str) or not text:
        return text, 0
    n_total = 0
    out = text
    for v in (*extra_values, *secret_env_values()):
        if v and v in out:
            n_total += out.count(v)
            out = out.replace(v, KEY)
    for pat in KEY_PATTERNS:
        out, n = pat.subn(KEY, out)
        n_total += n
    for pat in VALUE_PATTERNS:
        out, n = pat.subn(lambda m: m.group(1) + KEY, out)
        n_total += n
    return out, n_total


def redact(text):
    """``redact_secrets(text)[0]``."""
    return redact_secrets(text)[0]


_PREFIXES = ("C2C", "c2c", "MEC")


class SecretLogFilter(logging.Filter):
    """Rewrites C2C / MEC log records: message (args merged in) and traceback text. Always lets the record through."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            if not str(record.name).startswith(_PREFIXES):
                return True
            msg = record.getMessage()
            red, n = redact_secrets(msg)
            if n:
                record.msg, record.args = red, ()
            if record.exc_info and not record.exc_text:
                record.exc_text = logging.Formatter().formatException(record.exc_info)
            if record.exc_text:
                record.exc_text = redact(record.exc_text)
        except Exception:  # noqa: BLE001 - a log line must never be lost to the redactor
            pass
        return True


_FILTER = SecretLogFilter()


def install_log_filter(root: logging.Logger | None = None) -> int:
    """Add the filter to every handler of ``root`` (default: the root logger) that lacks it. Returns how many."""
    root = root or logging.getLogger()
    n = 0
    for h in list(root.handlers):
        if not any(isinstance(f, SecretLogFilter) for f in h.filters):
            h.addFilter(_FILTER)
            n += 1
    return n
