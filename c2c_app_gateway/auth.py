"""Sessions (browser), signed scoped tokens (API clients), CSRF / Origin checks and rate limits.

Sessions live in memory only: a gateway restart signs everyone out (nothing to steal from disk, nothing to restore
wrongly). Every request re-reads the user record, so disabling a user or removing them from an app ACL takes effect on
the next request. Tokens: ``c2t1.<b64url payload>.<b64url HMAC-SHA256>`` with user, scopes, app ids, expiry and a
token id that the owner can revoke.
"""
from __future__ import annotations

import base64
import collections
import hashlib
import hmac
import json
import secrets
import threading
import time

COOKIE = "c2c_gw"
CSRF_HEADER = "X-C2C-CSRF"
SCOPES = ("run", "publish", "admin")
ROLE_SCOPES = {"owner": {"run", "publish", "admin"}, "publisher": {"run", "publish"}, "runner": {"run"}}


def _b64e(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode("ascii")


def _b64d(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


class Sessions:
    def __init__(self, idle_s: int = 12 * 3600, absolute_s: int = 7 * 24 * 3600):
        self.idle_s, self.absolute_s = int(idle_s), int(absolute_s)
        self._s: dict[str, dict] = {}
        self._lock = threading.Lock()

    def create(self, user: str) -> dict:
        now = time.time()
        s = {"sid": secrets.token_urlsafe(32), "user": user, "csrf": secrets.token_urlsafe(24), "created": now,
             "last": now}
        with self._lock:
            self._s[s["sid"]] = s
        return s

    def get(self, sid: str | None) -> dict | None:
        if not sid:
            return None
        now = time.time()
        with self._lock:
            s = self._s.get(sid)
            if not s:
                return None
            if now - s["last"] > self.idle_s or now - s["created"] > self.absolute_s:
                self._s.pop(sid, None)
                return None
            s["last"] = now
            return dict(s)

    def drop(self, sid: str) -> None:
        with self._lock:
            self._s.pop(sid, None)

    def drop_user(self, user: str) -> None:
        with self._lock:
            for k in [k for k, v in self._s.items() if v["user"] == user]:
                self._s.pop(k, None)


class Tokens:
    def __init__(self, key: bytes, revoked_fn):
        self._key, self._revoked = key, revoked_fn

    def issue(self, user: str, scopes, apps=None, ttl_s: int = 3600) -> tuple[str, dict]:
        scopes = sorted(set(scopes or ()))
        if not scopes or any(s not in SCOPES for s in scopes):
            raise ValueError(f"scopes must be a non-empty subset of {', '.join(SCOPES)}.")
        ttl_s = int(ttl_s)
        if not 60 <= ttl_s <= 30 * 24 * 3600:
            raise ValueError("ttl_s must be between 60 seconds and 30 days.")
        p = {"u": user, "s": scopes, "a": sorted(set(apps)) if apps else None, "exp": int(time.time()) + ttl_s,
             "jti": secrets.token_urlsafe(12)}
        body = _b64e(json.dumps(p, sort_keys=True, separators=(",", ":")).encode())
        sig = _b64e(hmac.new(self._key, body.encode(), hashlib.sha256).digest())
        return f"c2t1.{body}.{sig}", p

    def verify(self, tok: str) -> dict | None:
        try:
            ver, body, sig = tok.split(".")
            if ver != "c2t1":
                return None
            want = hmac.new(self._key, body.encode(), hashlib.sha256).digest()
            if not hmac.compare_digest(want, _b64d(sig)):
                return None
            p = json.loads(_b64d(body))
        except Exception:  # noqa: BLE001 - any malformed token is simply not a token
            return None
        if p.get("exp", 0) < time.time() or p.get("jti") in self._revoked():
            return None
        return p


class Signer:
    """Short-lived signed download links: job id + output file + user + expiry, HMAC-SHA256."""

    def __init__(self, key: bytes):
        self._key = key

    def sign(self, job_id: str, name: str, user: str, ttl_s: int = 600) -> str:
        body = _b64e(json.dumps([job_id, name, user, int(time.time()) + int(ttl_s)],
                                separators=(",", ":")).encode())
        return body + "." + _b64e(hmac.new(self._key, body.encode(), hashlib.sha256).digest()[:24])

    def verify(self, tok: str) -> tuple[str, str, str] | None:
        try:
            body, sig = tok.split(".")
            want = hmac.new(self._key, body.encode(), hashlib.sha256).digest()[:24]
            if not hmac.compare_digest(want, _b64d(sig)):
                return None
            job_id, name, user, exp = json.loads(_b64d(body))
        except Exception:  # noqa: BLE001
            return None
        if exp < time.time():
            return None
        return job_id, name, user


class RateLimiter:
    """Sliding-window counter per key."""

    def __init__(self):
        self._hits: dict[str, collections.deque] = collections.defaultdict(collections.deque)
        self._lock = threading.Lock()

    def hit(self, key: str, limit: int, window_s: float, *, record: bool = True) -> bool:
        """True when the action is allowed (and, with ``record``, counts it)."""
        now = time.time()
        with self._lock:
            q = self._hits[key]
            while q and now - q[0] > window_s:
                q.popleft()
            if len(q) >= limit:
                return False
            if record:
                q.append(now)
            return True

    def count(self, key: str, window_s: float) -> int:
        now = time.time()
        with self._lock:
            q = self._hits.get(key) or ()
            return sum(1 for t in q if now - t <= window_s)

    def clear(self, key: str) -> None:
        with self._lock:
            self._hits.pop(key, None)
