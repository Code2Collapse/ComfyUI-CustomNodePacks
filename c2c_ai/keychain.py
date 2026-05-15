"""Secret storage backed by the OS credential store.

Uses the ``keyring`` library so we transparently get:
    - Windows  → Credential Manager (DPAPI)
    - macOS    → Keychain
    - Linux    → Secret Service (libsecret) or KWallet

If ``keyring`` is unavailable (user hasn't installed C2C[ai]) we refuse to
fall back to plaintext on disk — instead we prompt at runtime so users
never silently lose security guarantees.

Service name in the keychain is ``c2c-comfy``. Each key is stored as a
separate entry so revoking one doesn't disturb the others.
"""

from __future__ import annotations

import logging
from typing import Iterable

log = logging.getLogger("c2c_ai.keychain")

SERVICE = "c2c-comfy"


# We import lazily so the module loads even when keyring is missing —
# callers see a clear error only when they actually try to use it.
def _kr():
    try:
        import keyring  # type: ignore
        return keyring
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "The 'keyring' package is required for secure API key storage. "
            "Install it with:  pip install keyring   (or: pip install \"c2c[ai]\")"
        ) from exc


# Known canonical key names — features should reference these constants
# rather than hard-coding strings.
KEY_ANTHROPIC = "ANTHROPIC_API_KEY"
KEY_OPENAI = "OPENAI_API_KEY"
KEY_QWEN = "QWEN_API_KEY"
KEY_OPENROUTER = "OPENROUTER_API_KEY"
KEY_HF = "HUGGINGFACE_API_KEY"
# P10.1 — extended providers
KEY_AZURE_OPENAI = "AZURE_OPENAI_API_KEY"
KEY_GEMINI = "GEMINI_API_KEY"
KEY_COHERE = "COHERE_API_KEY"

ALL_KNOWN_KEYS: tuple[str, ...] = (
    KEY_ANTHROPIC,
    KEY_OPENAI,
    KEY_QWEN,
    KEY_OPENROUTER,
    KEY_HF,
    KEY_AZURE_OPENAI,
    KEY_GEMINI,
    KEY_COHERE,
)


def get(name: str) -> str | None:
    """Return the secret, or ``None`` if not set. Never raises on missing."""
    try:
        return _kr().get_password(SERVICE, name)
    except Exception as exc:
        log.warning("keychain get(%s) failed: %s", name, exc)
        return None


def set_(name: str, value: str) -> None:
    """Store / overwrite a secret. Raises on backend failure."""
    if not value:
        raise ValueError("refusing to store empty secret")
    _kr().set_password(SERVICE, name, value)
    log.info("keychain: stored %s (%d chars)", name, len(value))


def delete(name: str) -> bool:
    """Remove a secret. Returns True if it existed, False if not."""
    try:
        _kr().delete_password(SERVICE, name)
        return True
    except Exception:
        return False


def list_set_keys() -> list[str]:
    """Return canonical key names that currently have a value in the keychain."""
    return [k for k in ALL_KNOWN_KEYS if get(k)]


# ---------- import-from-txt --------------------------------------------------

def parse_keys_txt(path: str) -> dict[str, str]:
    """Best-effort parser for ``All API Keys Of Comfy.txt``.

    Accepts any of these line formats (one per line, blanks/comments ignored):
        ANTHROPIC_API_KEY=sk-ant-...
        ANTHROPIC_API_KEY: sk-ant-...
        anthropic = sk-ant-...
        # comment
    Returns a dict mapping canonical key names → values for keys we recognise.
    Unknown lines are silently skipped so this is safe to run on noisy files.
    """
    import os
    import re

    if not os.path.isfile(path):
        return {}

    # Aliases — what users might write → canonical key
    alias = {
        "anthropic": KEY_ANTHROPIC,
        "anthropic_api_key": KEY_ANTHROPIC,
        "claude": KEY_ANTHROPIC,
        "openai": KEY_OPENAI,
        "openai_api_key": KEY_OPENAI,
        "gpt": KEY_OPENAI,
        "qwen": KEY_QWEN,
        "qwen_api_key": KEY_QWEN,
        "dashscope": KEY_QWEN,
        "openrouter": KEY_OPENROUTER,
        "openrouter_api_key": KEY_OPENROUTER,
        "hf": KEY_HF,
        "huggingface": KEY_HF,
        "huggingface_api_key": KEY_HF,
        "hf_token": KEY_HF,
        # P10.1 — extended providers
        "azure": KEY_AZURE_OPENAI,
        "azure_openai": KEY_AZURE_OPENAI,
        "azure_openai_api_key": KEY_AZURE_OPENAI,
        "gemini": KEY_GEMINI,
        "gemini_api_key": KEY_GEMINI,
        "google_api_key": KEY_GEMINI,
        "google_ai": KEY_GEMINI,
        "cohere": KEY_COHERE,
        "cohere_api_key": KEY_COHERE,
    }

    pat = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*[:=]\s*(.+?)\s*$")
    found: dict[str, str] = {}
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#") or line.startswith("//"):
                continue
            m = pat.match(line)
            if not m:
                continue
            name = m.group(1).lower()
            value = m.group(2).strip().strip('"').strip("'")
            if not value or value.lower() in {"none", "null", "todo", "..."}:
                continue
            canonical = alias.get(name)
            if canonical:
                found[canonical] = value
    return found


def import_from_txt(path: str) -> list[str]:
    """Import recognised keys from a txt file into the keychain.

    Returns the list of canonical keys that were written. Caller is
    responsible for asking the user whether to delete the source file.
    """
    found = parse_keys_txt(path)
    written: list[str] = []
    for k, v in found.items():
        try:
            set_(k, v)
            written.append(k)
        except Exception as exc:
            log.error("failed to store %s: %s", k, exc)
    return written
