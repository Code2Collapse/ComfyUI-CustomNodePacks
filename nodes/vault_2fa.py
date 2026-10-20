"""Second factor for opening a vault to edit: TOTP (RFC 6238) per person per install (A17 s2.6, owner D0.25).

The `cryptography` package's TOTP (SHA-1, 6 digits, 30 s - what authenticator apps use). A 160-bit secret is created
at enrolment and stored only in the local secrets store (vault_store), never in a workflow and never derived from a
password. Enrolment needs one correct code before it takes effect, so a mistyped setup cannot lock anyone out.
Eight single-use recovery codes are shown once; only their SHA-256 hashes are kept. A code is accepted once (replay
protection); five failures lock the second factor for five minutes (persisted, so a restart does not reset it).

Honest scope: this protects editing against other users of this ComfyUI and casual access. The machine's
administrator can delete the stored state (that is the documented reset when both the app and the recovery codes
are lost).
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import threading
import time
from typing import Optional

from . import vault_store

DIGITS, STEP, WINDOW = 6, 30, 1
MAX_FAILS, LOCKOUT_SECONDS = 5, 5 * 60
_SECRET, _LAST, _RECOVERY, _FAILS = "totp_secret", "totp_last_step", "totp_recovery", "totp_fails"
_lock = threading.Lock()
_pending: dict = {}                    # enrolment waiting for its first code: {"secret": bytes, "codes": [...]}


def _totp(secret: bytes):
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.twofactor.totp import TOTP

    return TOTP(secret, DIGITS, hashes.SHA1(), STEP, enforce_key_length=False)


def enrolled() -> bool:
    return bool(vault_store.get(_SECRET))


def _hash(code: str) -> str:
    return hashlib.sha256(code.strip().upper().replace("-", "").encode("ascii")).hexdigest()


def _recovery_code() -> str:
    alphabet = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"            # no 0/O/1/I/L
    raw = "".join(secrets.choice(alphabet) for _ in range(10))
    return f"{raw[:4]}-{raw[4:8]}-{raw[8:]}"


def begin_enrolment(account: str = "ComfyUI vault") -> dict:
    """A new secret + recovery codes, shown ONCE; active only after confirm_enrolment(first code)."""
    with _lock:
        if enrolled():
            raise PermissionError("A second factor is already set up on this install. Reset it first (needs a code "
                                  "or a recovery code).")
        secret = os.urandom(20)
        codes = [_recovery_code() for _ in range(8)]
        _pending.clear()
        _pending.update({"secret": secret, "codes": codes, "at": time.time()})
    uri = _totp(secret).get_provisioning_uri(account, "C2C Vault")
    return {"secret_base32": base64.b32encode(secret).decode("ascii").rstrip("="), "otpauth_uri": uri,
            "recovery_codes": codes}


def confirm_enrolment(code: str, now: Optional[float] = None) -> bool:
    with _lock:
        if not _pending or time.time() - _pending.get("at", 0) > 15 * 60:
            _pending.clear()
            raise PermissionError("No enrolment is waiting (or it expired). Start the setup again.")
        step = _match(_pending["secret"], code, now)
        if step is None:
            return False
        vault_store.put(_SECRET, base64.b64encode(_pending["secret"]).decode("ascii"))
        vault_store.put(_RECOVERY, json.dumps([_hash(c) for c in _pending["codes"]]))
        vault_store.put(_LAST, str(step))
        vault_store.delete(_FAILS)
        _pending.clear()
        return True


def _match(secret: bytes, code: str, now: Optional[float]) -> Optional[int]:
    from cryptography.hazmat.primitives.twofactor import InvalidToken

    code = (code or "").strip().replace(" ", "")
    if len(code) != DIGITS or not code.isdigit():
        return None
    t = time.time() if now is None else now
    for offset in range(-WINDOW, WINDOW + 1):
        when = t + offset * STEP
        try:
            _totp(secret).verify(code.encode("ascii"), int(when))
            return int(when // STEP)
        except InvalidToken:
            continue
    return None


def locked_out(now: Optional[float] = None) -> float:
    state = json.loads(vault_store.get(_FAILS) or '{"count": 0, "until": 0}')
    if state.get("count", 0) < MAX_FAILS:
        return 0.0
    remaining = float(state.get("until", 0)) - (time.time() if now is None else now)
    if remaining <= 0:
        vault_store.delete(_FAILS)
        return 0.0
    return remaining


def _fail(now: Optional[float]) -> None:
    state = json.loads(vault_store.get(_FAILS) or '{"count": 0, "until": 0}')
    state["count"] = int(state.get("count", 0)) + 1
    state["until"] = (time.time() if now is None else now) + LOCKOUT_SECONDS
    vault_store.put(_FAILS, json.dumps(state))


def verify(code: str, now: Optional[float] = None) -> bool:
    """A current authenticator code (once) or an unused recovery code (consumed). Never raises on a wrong code."""
    with _lock:
        stored = vault_store.get(_SECRET)
        if not stored:
            raise PermissionError("No second factor is set up on this install yet. Set it up in the C2C Vault menu.")
        if locked_out(now) > 0:
            raise PermissionError("Too many wrong codes. Try again in a few minutes.")
        step = _match(base64.b64decode(stored), code, now)
        if step is not None:
            last = int(vault_store.get(_LAST) or "-1")
            if step <= last:
                _fail(now)
                return False                                    # a code is good once
            vault_store.put(_LAST, str(step))
            vault_store.delete(_FAILS)
            return True
        hashes = json.loads(vault_store.get(_RECOVERY) or "[]")
        h = _hash(code or "")
        for k, known in enumerate(hashes):
            if hmac.compare_digest(known, h):
                hashes.pop(k)
                vault_store.put(_RECOVERY, json.dumps(hashes))
                vault_store.delete(_FAILS)
                return True
        _fail(now)
        return False


def reset(code: str, now: Optional[float] = None) -> bool:
    """Remove the second factor (to enrol again) - only with a valid code or recovery code."""
    if not verify(code, now):
        return False
    with _lock:
        for name in (_SECRET, _RECOVERY, _LAST, _FAILS):
            vault_store.delete(name)
    return True


def recovery_codes_left() -> int:
    return len(json.loads(vault_store.get(_RECOVERY) or "[]"))
