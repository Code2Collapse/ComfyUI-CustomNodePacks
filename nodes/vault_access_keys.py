"""C2C Vault access keys (V6-B, owner 2026-11-21; design docs/h2vault/V6_cnp_vault_enhancements.md section 3).

A Sealed vault can require an ACCESS KEY on the installations it was sealed for: a short text the vault's author
signs (Ed25519) and sends, bound to one or more installations and valid until a date. Everything is offline - no
licence server. The vault's encrypted contents carry the policy `access_keys = {required, issuers, gen}`; the run
check (secure_subgraph.open_for_run) asks this module whether this installation holds a key that passes.

Key text: `c2cak1.<b64url(payload)>.<b64url(signature)>`, payload = compact JSON
`{v, vid, iss, iat, exp, ins, gen, jti, note}`, signature = Ed25519 over DOMAIN + payload bytes.

Honest limits (said in the UI and the howto): a key cannot be withdrawn from a copy someone already has - revoking
raises the vault's generation, so old keys stop opening NEW copies, and expiry ends old ones. The check runs in
ComfyUI on the recipient's machine; someone who changes that code can skip it (the same boundary as every vault).

The author's signing key is created once and kept in the local secrets store (vault_store); received keys are not
secrets (signed, bound to an installation) and live in `<user dir>/c2c/vault_access_keys.json`.

License: Apache-2.0
"""
from __future__ import annotations

import base64
import binascii
import hashlib
import json
import logging
import os
import re
import secrets
import threading
import time
from pathlib import Path
from typing import Optional

from . import vault_store
from .vault_crypto import VaultError

log = logging.getLogger("c2c.vault")

PREFIX = "c2cak1"
VERSION = 1
DOMAIN = b"c2c-vault-access-key-v1\x00"
STORE_FILE: Optional[Path] = None      # tests point this at a temp file; None = the ComfyUI user directory
MAX_KEYS_PER_VAULT = 16
MAX_TOKEN_CHARS = 8192
MAX_INSTALLS = 64
NOTE_MAX = 60
TEN_YEARS = 10 * 366 * 86400
CLOCK_SLACK = 86400                    # time zones and small drift are not a "clock set back"
SEEN_STEP = 60                         # the high-water mark is written only when it moves this much
_NAME = "author_ed25519_private"
_lock = threading.Lock()
_KID_RE = re.compile(r"^[0-9a-f]{16}$")

NOT_A_KEY = f"This is not a C2C vault access key (it starts with {PREFIX}.)."
NOT_ISSUER = "This access key was not issued by this vault's author."
OTHER_VAULT = "This access key is for a different vault."
REVOKED = "This access key was revoked by the vault's author. Ask its author for a new key."
OTHER_INSTALL = ("This access key is for other installations. Send the vault's author your installation code "
                 "(C2C menu > Vault > My installation code).")
NEEDS_KEY = "This vault needs an access key from its author. Right-click the vault > Enter access key."
CLOCK = ("This computer's clock is set earlier than the last time a vault access key was checked here. Correct the "
         "date and time to run this vault.")


def _day(ts: int) -> str:
    return time.strftime("%Y-%m-%d", time.gmtime(int(ts)))


def _expired(ts: int) -> str:
    return f"This vault's access key expired on {_day(ts)}. Ask its author for a new key."


def _b64u(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _unb64u(text: str) -> bytes:
    if not isinstance(text, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", text):
        raise ValueError("not base64url")
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


# ── the author's signing key ────────────────────────────────────────────────────────────────────────────────────
def _author_private():
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    with _lock:
        stored = vault_store.get(_NAME)
        if stored:
            return Ed25519PrivateKey.from_private_bytes(base64.b64decode(stored))
        key = Ed25519PrivateKey.generate()
        raw = key.private_bytes(serialization.Encoding.Raw, serialization.PrivateFormat.Raw,
                                serialization.NoEncryption())
        vault_store.put(_NAME, base64.b64encode(raw).decode("ascii"))
        return key


def _public_raw(private) -> bytes:
    from cryptography.hazmat.primitives import serialization

    return private.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)


def author_issuer() -> str:
    """This installation's issuer id: standard base64 of its raw Ed25519 public key (public, safe to show)."""
    return base64.b64encode(_public_raw(_author_private())).decode("ascii")


def install_key_id(public_raw: bytes) -> str:
    """The short id a key is bound to: the first 8 bytes of SHA-256 of an installation's X25519 public key, hex."""
    return hashlib.sha256(bytes(public_raw)).digest()[:8].hex()


# ── keys ────────────────────────────────────────────────────────────────────────────────────────────────────────
def issue(*, vault_id: str, installs: list, expires: int, gen: int, note: str = "", now: Optional[float] = None) -> str:
    """Sign a key for these installations (raw X25519 public keys) of this vault until `expires` (unix seconds)."""
    now = time.time() if now is None else now
    expires = int(expires)
    if expires <= now:
        raise VaultError("Choose an expiry date in the future.")
    if expires > now + TEN_YEARS:
        raise VaultError("Choose an expiry date within ten years.")
    kids = []
    for pub in installs or []:
        kid = install_key_id(pub)
        if kid not in kids:
            kids.append(kid)
    if not kids:
        raise VaultError("Choose at least one installation for this key.")
    if len(kids) > MAX_INSTALLS:
        raise VaultError(f"A key can name at most {MAX_INSTALLS} installations.")
    key = _author_private()
    payload = {"v": VERSION, "vid": str(vault_id), "iss": base64.b64encode(_public_raw(key)).decode("ascii"),
               "iat": int(now), "exp": expires, "ins": kids, "gen": int(gen), "jti": secrets.token_hex(8),
               "note": str(note or "").strip()[:NOTE_MAX]}
    body = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return ".".join([PREFIX, _b64u(body), _b64u(key.sign(DOMAIN + body))])


def parse(token: str) -> dict:
    """The payload of a well-formed key whose signature matches the issuer it names. Anything else: NOT_A_KEY."""
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    try:
        text = str(token or "").strip()
        if len(text) > MAX_TOKEN_CHARS:
            raise ValueError("too long")
        head, body_b64, sig_b64 = text.split(".")
        if head != PREFIX:
            raise ValueError("prefix")
        body, sig = _unb64u(body_b64), _unb64u(sig_b64)
        p = json.loads(body.decode("utf-8"))
        if not isinstance(p, dict) or p.get("v") != VERSION:
            raise ValueError("version")
        for k, t in (("vid", str), ("iss", str), ("jti", str), ("note", str)):
            if not isinstance(p.get(k), t):
                raise ValueError(k)
        for k in ("iat", "exp", "gen"):
            if not isinstance(p.get(k), int) or isinstance(p.get(k), bool):
                raise ValueError(k)
        ins = p.get("ins")
        if not isinstance(ins, list) or not ins or not all(isinstance(k, str) and _KID_RE.match(k) for k in ins):
            raise ValueError("ins")
        pub = base64.b64decode(p["iss"], validate=True)
        if len(pub) != 32 or len(sig) != 64:
            raise ValueError("lengths")
        Ed25519PublicKey.from_public_bytes(pub).verify(sig, DOMAIN + body)
    except (ValueError, TypeError, UnicodeDecodeError, binascii.Error, InvalidSignature):
        raise VaultError(NOT_A_KEY) from None
    return p


def verify(token: str, *, vault_id: str, policy: dict, install_kid: str, now: float) -> dict:
    """The payload when this key lets `install_kid` run `vault_id` under `policy` at `now`; else a plain refusal."""
    p = parse(token)
    if p["iss"] not in list((policy or {}).get("issuers") or []):
        raise VaultError(NOT_ISSUER)
    if p["vid"] != str(vault_id):
        raise VaultError(OTHER_VAULT)
    if p["gen"] < int((policy or {}).get("gen") or 0):
        raise VaultError(REVOKED)
    if str(install_kid) not in p["ins"]:
        raise VaultError(OTHER_INSTALL)
    if p["exp"] <= now:
        raise VaultError(_expired(p["exp"]))
    return p


def policy_of(plain: dict) -> Optional[dict]:
    """The access policy of decrypted vault contents, or None when the vault does not require a key."""
    pol = (plain or {}).get("access_keys") if isinstance(plain, dict) else None
    return pol if isinstance(pol, dict) and pol.get("required") is True else None


# ── the store of received keys (+ the clock high-water mark) ────────────────────────────────────────────────────
def store_path() -> Path:
    if STORE_FILE is not None:
        return Path(STORE_FILE)
    import folder_paths

    return Path(folder_paths.get_user_directory()) / "c2c" / "vault_access_keys.json"


def _load() -> dict:
    try:
        data = json.loads(store_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"version": 1, "keys": {}, "seen": 0}
    if not isinstance(data, dict):
        return {"version": 1, "keys": {}, "seen": 0}
    keys = data.get("keys") if isinstance(data.get("keys"), dict) else {}
    keys = {str(v): [t for t in ts if isinstance(t, str)] for v, ts in keys.items() if isinstance(ts, list)}
    seen = data.get("seen") if isinstance(data.get("seen"), (int, float)) else 0
    return {"version": 1, "keys": keys, "seen": int(seen)}


def _save(data: dict) -> None:
    path = store_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def tokens_for(vault_id: str) -> list:
    with _lock:
        return list(_load()["keys"].get(str(vault_id), []))


def add_token(vault_id: str, token: str) -> None:
    def iat(t):
        try:
            return parse(t)["iat"]
        except VaultError:
            return -1
    with _lock:
        data = _load()
        lst = [t for t in data["keys"].get(str(vault_id), []) if t != token] + [str(token)]
        lst.sort(key=iat)
        data["keys"][str(vault_id)] = lst[-MAX_KEYS_PER_VAULT:]
        _save(data)


def last_seen() -> int:
    with _lock:
        return _load()["seen"]


def note_seen(now: float) -> None:
    with _lock:
        data = _load()
        if now - data["seen"] > SEEN_STEP:
            data["seen"] = int(now)
            try:
                _save(data)
            except OSError as exc:                            # a read-only user dir must not stop a run
                log.warning("C2C Vault: could not record the access-key clock mark: %s", exc)


def _check_clock(now: float) -> None:
    if now < last_seen() - CLOCK_SLACK:
        raise VaultError(CLOCK)


def check_run(vault_id: str, policy: dict, *, install_public: bytes, now: Optional[float] = None) -> dict:
    """Does a stored key let this installation run the vault now? -> the best key's payload, or a plain refusal."""
    now = time.time() if now is None else now
    _check_clock(now)
    stored = tokens_for(vault_id)
    if not stored:
        raise VaultError(NEEDS_KEY)
    kid = install_key_id(install_public)
    best, refusal = None, None
    for tok in stored:
        try:
            p = verify(tok, vault_id=vault_id, policy=policy, install_kid=kid, now=now)
        except VaultError as exc:
            try:
                exp = parse(tok)["exp"]
            except VaultError:
                exp = -1
            if refusal is None or exp > refusal[0]:
                refusal = (exp, str(exc))
            continue
        if best is None or p["exp"] > best["exp"]:
            best = p
    if best is None:
        raise VaultError(refusal[1] if refusal else NEEDS_KEY)
    note_seen(now)
    return best


def accept(token: str, *, vault_id: str, policy: dict, install_public: bytes, now: Optional[float] = None) -> dict:
    """A recipient enters a key: checked exactly like a run, then kept. Nothing is stored for a refused key."""
    now = time.time() if now is None else now
    _check_clock(now)
    token = str(token or "").strip()
    p = verify(token, vault_id=vault_id, policy=policy, install_kid=install_key_id(install_public), now=now)
    add_token(vault_id, token)
    note_seen(now)
    return {"expires": _day(p["exp"]), "exp": p["exp"], "note": p.get("note", "")}
