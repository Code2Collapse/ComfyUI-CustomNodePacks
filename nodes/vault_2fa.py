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
        if _pending and time.time() - _pending.get("at", 0) <= 15 * 60:
            secret, codes = _pending["secret"], _pending["codes"]   # same setup while it is pending (QR <-> key)
        else:
            secret = os.urandom(20)
            codes = [_recovery_code() for _ in range(8)]
            _pending.clear()
            _pending.update({"secret": secret, "codes": codes, "at": time.time()})
    uri = _totp(secret).get_provisioning_uri(account, "C2C Vault")
    b32 = base64.b32encode(secret).decode("ascii").rstrip("=")
    return {"secret_base32": b32, "secret_grouped": " ".join(b32[i:i + 4] for i in range(0, len(b32), 4)),
            "otpauth_uri": uri, "qr_svg": qr_svg(uri), "recovery_codes": codes, "issuer": "C2C Vault",
            "account": account, "algorithm": "SHA1", "digits": DIGITS, "period": STEP}


def qr_svg(text: str, border: int = 4) -> str:
    """The QR code of `text` as a small standalone SVG (vendored Nayuki qrcodegen, MIT). '' if it cannot."""
    try:
        from ._vendor.qrcodegen import QrCode
    except Exception:  # noqa: BLE001
        return ""
    qr = QrCode.encode_text(text, QrCode.Ecc.MEDIUM)
    n = qr.get_size()
    parts = [f"M{x + border},{y + border}h1v1h-1z" for y in range(n) for x in range(n) if qr.get_module(x, y)]
    size = n + 2 * border
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {size} {size}" shape-rendering="crispEdges">'
            f'<rect width="100%" height="100%" fill="#ffffff"/><path d="{" ".join(parts)}" fill="#000000"/></svg>')


def cancel_enrolment() -> None:
    with _lock:
        _pending.clear()


def last_enrolment_error() -> str:
    return str(_pending.get("last_error") or "")


def confirm_enrolment(code: str, now: Optional[float] = None) -> bool:
    with _lock:
        if not _pending or time.time() - _pending.get("at", 0) > 15 * 60:
            _pending.clear()
            raise PermissionError("No enrolment is waiting (or it expired). Start the setup again.")
        step = _match(_pending["secret"], code, now)
        if step is None:
            skew = _probe_clock(_pending["secret"], code, now)
            _pending["last_error"] = (
                f"That code matches a time {abs(skew)} s away: this computer's clock and your phone's disagree. Turn "
                "on automatic time sync on both, then enter a fresh code." if skew is not None else
                "That code is not right. Enter the current code of the new 'C2C Vault' entry (scan the QR code or "
                "type the setup key again if the app shows no entry).")
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


CLOCK_PROBE_STEPS = 20                 # how far (x 30 s) to look for a code from a drifted clock - never accepted


def _probe_clock(secret: bytes, code: str, now: Optional[float]) -> Optional[int]:
    """Seconds the server clock is AHEAD (+) or BEHIND (-) the app's, when the code belongs to a nearby time step
    outside the accepted window. Diagnosis only: such a code is never accepted."""
    from cryptography.hazmat.primitives.twofactor import InvalidToken

    code = (code or "").strip().replace(" ", "")
    if len(code) != DIGITS or not code.isdigit():
        return None
    t = time.time() if now is None else now
    for k in range(WINDOW + 1, CLOCK_PROBE_STEPS + 1):
        for offset in (k, -k):
            try:
                _totp(secret).verify(code.encode("ascii"), int(t + offset * STEP))
                return -offset * STEP
            except InvalidToken:
                continue
    return None


def verify_detail(code: str, now: Optional[float] = None) -> dict:
    """{"ok": bool, "reason": "ok" | "recovery" | "reused" | "clock" | "wrong", "message": str, ...}.

    "reused": the code was right but already used (a code works ONCE - replay protection); the next one is due in
    `next_in` seconds. Not counted as a failed attempt: it is not a guess.
    "clock": the code belongs to a time step `skew_seconds` away from this server's clock - the clocks disagree.
    Never accepted, not counted. "wrong": counted (5 failures lock the factor for 5 minutes)."""
    with _lock:
        stored = vault_store.get(_SECRET)
        if not stored:
            raise PermissionError("No second factor is set up on this install yet. Set it up in the C2C Vault menu.")
        if locked_out(now) > 0:
            raise PermissionError("Too many wrong codes. Try again in a few minutes.")
        secret = base64.b64decode(stored)
        t = time.time() if now is None else now
        step = _match(secret, code, now)
        if step is not None:
            last = int(vault_store.get(_LAST) or "-1")
            if step <= last:
                nxt = int(STEP - (t % STEP)) + 1
                return {"ok": False, "reason": "reused", "next_in": nxt,
                        "message": f"That code was already used (each code works once). Wait for the next code in "
                                   f"your app - about {nxt} s - and enter that one."}
            vault_store.put(_LAST, str(step))
            vault_store.delete(_FAILS)
            return {"ok": True, "reason": "ok", "message": "Verified."}
        hashes = json.loads(vault_store.get(_RECOVERY) or "[]")
        h = _hash(code or "")
        for k, known in enumerate(hashes):
            if hmac.compare_digest(known, h):
                hashes.pop(k)
                vault_store.put(_RECOVERY, json.dumps(hashes))
                vault_store.delete(_FAILS)
                return {"ok": True, "reason": "recovery", "message": f"Recovery code used ({len(hashes)} left)."}
        skew = _probe_clock(secret, code, now)
        if skew is not None:
            side = "ahead of" if skew > 0 else "behind"
            return {"ok": False, "reason": "clock", "skew_seconds": skew,
                    "message": f"That code matches a time {abs(skew)} s away: this computer's clock is about "
                               f"{abs(skew)} s {side} your phone's. Codes depend on the clock (not the time zone): "
                               "turn on automatic time sync on both, then try a fresh code."}
        _fail(now)
        return {"ok": False, "reason": "wrong",
                "message": "That code is not right. Use the current 6-digit code of the 'C2C Vault' entry in your "
                           "authenticator app."}


def verify(code: str, now: Optional[float] = None) -> bool:
    """A current authenticator code (once) or an unused recovery code (consumed). Never raises on a wrong code."""
    return bool(verify_detail(code, now)["ok"])


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
