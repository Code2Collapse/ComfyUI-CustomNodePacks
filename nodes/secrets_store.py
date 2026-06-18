"""
Encrypted local secrets store for cloud LLM API keys.

Keys are stored at:
    <pack_root>/user/secrets.enc

Encryption: three-tier backend — preferred AES-256-GCM via `cryptography`,
fallback AES-256-CTR+HMAC via `pycryptodome`, final stdlib-only fallback
using HMAC-SHA256 counter-mode keystream + HMAC-SHA256 authentication tag
(encrypt-then-MAC with HKDF-derived twin keys). The first available tier
wins on every write; reads dispatch by the 4-byte format prefix. The
encryption key itself is derived (scrypt) from a machine-bound seed
(hostname + uname + a per-install salt persisted at `<pack_root>/user/.salt`). This is *not* protection against a determined
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

PROVIDERS = (
    "openai", "anthropic", "gemini", "openrouter", "groq", "deepseek",
    # P10.1 — extended providers
    "azure_openai", "cohere", "custom",
)


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


def _derive_twin_keys() -> tuple:
    """Derive (enc_key, mac_key) for the stdlib-only HMAC-CTR fallback.

    Both 32 bytes, produced via HKDF-Expand-SHA256 from the scrypt-derived
    master key with distinct info labels so the two outputs are unrelated.
    """
    master = _derive_key()
    # HKDF-Expand(SHA256, prk=master, info, L=32) -> 1 block (32B) each.
    def _hkdf_expand_32(info: bytes) -> bytes:
        return hmac.new(master, info + b"\x01", hashlib.sha256).digest()
    return _hkdf_expand_32(b"c2c-secrets:enc"), _hkdf_expand_32(b"c2c-secrets:mac")


def _hmac_ctr_keystream(enc_key: bytes, nonce: bytes, nbytes: int) -> bytes:
    """Counter-mode keystream from HMAC-SHA256. 32 bytes per counter block."""
    out = bytearray()
    ctr = 0
    while len(out) < nbytes:
        block = hmac.new(enc_key, nonce + ctr.to_bytes(8, "big"), hashlib.sha256).digest()
        out.extend(block)
        ctr += 1
    return bytes(out[:nbytes])


# ── Encryption backend (prefers cryptography, falls back to AES-CTR+HMAC) ──
def _encrypt(plaintext: bytes) -> bytes:
    # Tier 1: AES-256-GCM via `cryptography` (preferred).
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        key = _derive_key()
        nonce = secrets.token_bytes(12)
        ct = AESGCM(key).encrypt(nonce, plaintext, None)
        return b"GCM1" + nonce + ct
    except ImportError:
        pass
    except Exception as e:
        log.warning("[secrets] AES-GCM backend failed (%s); trying next tier", e)

    # Tier 2: AES-256-CTR + HMAC via pycryptodome.
    try:
        from Crypto.Cipher import AES  # type: ignore
        key = _derive_key()
        nonce = secrets.token_bytes(16)
        cipher = AES.new(key, AES.MODE_CTR, nonce=nonce[:8],
                         initial_value=nonce[8:])
        ct = cipher.encrypt(plaintext)
        tag = hmac.new(key, nonce + ct, hashlib.sha256).digest()
        return b"CTR1" + nonce + tag + ct
    except ImportError:
        pass
    except Exception as e:
        log.warning("[secrets] AES-CTR backend failed (%s); trying next tier", e)

    # Tier 3: stdlib-only HMAC-SHA256 counter-mode + HMAC tag.
    # Cryptographically sound encrypt-then-MAC with twin HKDF-derived keys.
    # No third-party deps required.
    enc_key, mac_key = _derive_twin_keys()
    nonce = secrets.token_bytes(16)
    ks = _hmac_ctr_keystream(enc_key, nonce, len(plaintext))
    ct = bytes(p ^ k for p, k in zip(plaintext, ks))
    tag = hmac.new(mac_key, b"HMC1" + nonce + ct, hashlib.sha256).digest()
    return b"HMC1" + nonce + tag + ct


def _decrypt(blob: bytes) -> bytes:
    if blob.startswith(b"GCM1"):
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        key = _derive_key()
        nonce, ct = blob[4:16], blob[16:]
        return AESGCM(key).decrypt(nonce, ct, None)
    if blob.startswith(b"CTR1"):
        from Crypto.Cipher import AES  # type: ignore
        key = _derive_key()
        nonce, tag, ct = blob[4:20], blob[20:52], blob[52:]
        if not hmac.compare_digest(tag, hmac.new(key, nonce + ct, hashlib.sha256).digest()):
            raise ValueError("HMAC mismatch — secrets file tampered or wrong machine")
        cipher = AES.new(key, AES.MODE_CTR, nonce=nonce[:8],
                         initial_value=nonce[8:])
        return cipher.decrypt(ct)
    if blob.startswith(b"HMC1"):
        enc_key, mac_key = _derive_twin_keys()
        nonce, tag, ct = blob[4:20], blob[20:52], blob[52:]
        expected = hmac.new(mac_key, b"HMC1" + nonce + ct, hashlib.sha256).digest()
        if not hmac.compare_digest(tag, expected):
            raise ValueError("HMAC mismatch — secrets file tampered or wrong machine")
        ks = _hmac_ctr_keystream(enc_key, nonce, len(ct))
        return bytes(c ^ k for c, k in zip(ct, ks))
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
