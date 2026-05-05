"""
Encrypted local secrets store for cloud LLM API keys.

Keys are stored at:
    <pack_root>/user/secrets.enc

Encryption: AES-256-GCM (via the `cryptography` package if available, else
a derived-key Fernet fallback). The encryption key itself is derived from
a machine-bound seed (hostname + uname + a per-install salt persisted at
`<pack_root>/user/.salt`). This is *not* protection against a determined
local attacker — it's protection against accidental disclosure (commits,
logs, screenshots). The keys never leave the user's machine, and never go
into git: `<pack_root>/user/` is gitignored by the pack.

Public API
----------
set_key(provider, api_key)     persist a key (encrypted)
get_key(provider) -> str|None  decrypt and return
has_key_for(provider) -> bool
list_providers() -> [str]      providers with keys set
delete_key(provider)
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import platform
import secrets
from typing import Optional

log = logging.getLogger("MEC.secrets")

PROVIDERS = ("openai", "anthropic", "gemini", "openrouter", "groq", "deepseek")


def _user_dir() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    pack_root = os.path.dirname(here)
    p = os.path.join(pack_root, "user")
    os.makedirs(p, exist_ok=True)
    # Make sure user/ is gitignored at the pack root
    gi = os.path.join(pack_root, ".gitignore")
    if os.path.exists(gi):
        try:
            with open(gi, "r", encoding="utf-8") as f:
                txt = f.read()
            if "user/" not in txt:
                with open(gi, "a", encoding="utf-8") as f:
                    f.write("\n# MEC error assistant runtime data\nuser/\n")
        except Exception:
            pass
    return p


def _salt_path() -> str:
    return os.path.join(_user_dir(), ".salt")


def _store_path() -> str:
    return os.path.join(_user_dir(), "secrets.enc")


def _machine_seed() -> bytes:
    return f"{platform.node()}|{platform.system()}|{platform.machine()}".encode()


def _get_or_create_salt() -> bytes:
    p = _salt_path()
    if os.path.exists(p):
        with open(p, "rb") as f:
            return f.read()
    salt = secrets.token_bytes(32)
    with open(p, "wb") as f:
        f.write(salt)
    return salt


def _derive_key() -> bytes:
    """Return a 32-byte key derived from machine seed + persisted salt."""
    return hashlib.scrypt(
        password=_machine_seed(),
        salt=_get_or_create_salt(),
        n=2 ** 14, r=8, p=1, dklen=32,
    )


# ── Encryption backend (prefers cryptography, falls back to AES-CTR+HMAC) ──
def _encrypt(plaintext: bytes) -> bytes:
    key = _derive_key()
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        nonce = secrets.token_bytes(12)
        ct = AESGCM(key).encrypt(nonce, plaintext, None)
        return b"GCM1" + nonce + ct
    except Exception:
        # Fallback: AES-CTR + HMAC-SHA256 via stdlib only
        from Crypto.Cipher import AES  # type: ignore
        nonce = secrets.token_bytes(16)
        cipher = AES.new(key, AES.MODE_CTR, nonce=nonce[:8],
                         initial_value=nonce[8:])
        ct = cipher.encrypt(plaintext)
        tag = hmac.new(key, nonce + ct, hashlib.sha256).digest()
        return b"CTR1" + nonce + tag + ct


def _decrypt(blob: bytes) -> bytes:
    key = _derive_key()
    if blob.startswith(b"GCM1"):
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        nonce, ct = blob[4:16], blob[16:]
        return AESGCM(key).decrypt(nonce, ct, None)
    if blob.startswith(b"CTR1"):
        from Crypto.Cipher import AES  # type: ignore
        nonce, tag, ct = blob[4:20], blob[20:52], blob[52:]
        if not hmac.compare_digest(tag, hmac.new(key, nonce + ct, hashlib.sha256).digest()):
            raise ValueError("HMAC mismatch — secrets file tampered or wrong machine")
        cipher = AES.new(key, AES.MODE_CTR, nonce=nonce[:8],
                         initial_value=nonce[8:])
        return cipher.decrypt(ct)
    raise ValueError("Unknown secrets blob version")


# ── Public API ────────────────────────────────────────────────────────
def _load() -> dict:
    p = _store_path()
    if not os.path.exists(p):
        return {}
    try:
        with open(p, "rb") as f:
            blob = f.read()
        return json.loads(_decrypt(blob).decode("utf-8"))
    except Exception as e:
        log.warning("[secrets] could not decrypt store: %s", e)
        return {}


def _save(d: dict) -> None:
    blob = _encrypt(json.dumps(d).encode("utf-8"))
    p = _store_path()
    tmp = p + ".tmp"
    with open(tmp, "wb") as f:
        f.write(blob)
    os.replace(tmp, p)
    try:  # tighten perms on POSIX
        os.chmod(p, 0o600)
    except Exception:
        pass


def set_key(provider: str, api_key: str) -> None:
    if provider not in PROVIDERS:
        raise ValueError(f"unknown provider: {provider}")
    if not api_key or not isinstance(api_key, str):
        raise ValueError("api_key must be a non-empty string")
    d = _load()
    d[provider] = api_key.strip()
    _save(d)


def get_key(provider: str) -> Optional[str]:
    return _load().get(provider)


def has_key_for(provider: str) -> bool:
    return bool(get_key(provider))


def list_providers() -> list:
    return sorted(_load().keys())


def delete_key(provider: str) -> None:
    d = _load()
    d.pop(provider, None)
    _save(d)
