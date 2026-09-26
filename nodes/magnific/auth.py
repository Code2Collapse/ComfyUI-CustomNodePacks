"""OAuth 2.0 Device Authorization Grant (RFC 8628) against the Magnific
Keycloak realm — the Python port of editor-plugins/src/api/deviceAuth.ts.

The device flow is the one grant that works here: the ComfyUI server may be
headless or remote, so it can't host a redirect URI; the user approves on any
browser via the verification URL.
"""

import json
import os
import stat
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Optional

from . import config, tls_trust


class AuthError(RuntimeError):
    pass


class NotSignedInError(AuthError):
    def __init__(self) -> None:
        super().__init__(
            "Not signed in to Magnific. Use the ComfyUI menu (Magnific → Sign in) and retry."
        )


def _post_form(url: str, params: dict) -> dict:
    body = urllib.parse.urlencode(params).encode()
    request = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            # Same per-host attribution mcp.py sends on every MCP request.
            "X-Pikaso-Client": config.CLIENT_TAG,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=config.HTTP_TIMEOUT_SECONDS, context=tls_trust.get_ssl_context()) as response:
            return json.loads(response.read().decode())
    except urllib.error.HTTPError as error:
        # OAuth error responses (authorization_pending, invalid_grant…) arrive
        # as 4xx with a JSON body — surface them as data, not transport errors.
        try:
            return json.loads(error.read().decode())
        except (ValueError, OSError):
            raise AuthError(f"OAuth endpoint failed (HTTP {error.code})") from error


_endpoints_cache: Optional[dict] = None


def _discover_endpoints() -> dict:
    global _endpoints_cache
    if _endpoints_cache:
        return _endpoints_cache
    url = config.OAUTH_ISSUER.rstrip("/") + "/.well-known/openid-configuration"
    request = urllib.request.Request(url, headers={"X-Pikaso-Client": config.CLIENT_TAG})
    try:
        with urllib.request.urlopen(request, timeout=config.HTTP_TIMEOUT_SECONDS, context=tls_trust.get_ssl_context()) as response:
            metadata = json.loads(response.read().decode())
    except (urllib.error.URLError, ValueError) as error:
        raise AuthError(f"OIDC discovery failed: {error}") from error
    if not metadata.get("device_authorization_endpoint") or not metadata.get("token_endpoint"):
        raise AuthError("Authorization server does not advertise the device grant")
    _endpoints_cache = metadata
    return metadata


def _to_tokens(payload: dict) -> dict:
    # Untrusted JSON from the token endpoint: a malformed 2xx must not yield a
    # half-valid blob that leaves every MCP call sending a broken bearer.
    access_token = payload.get("access_token")
    expires_in = payload.get("expires_in")
    if not isinstance(access_token, str) or not access_token:
        raise AuthError("Token response missing access_token")
    if not isinstance(expires_in, (int, float)):
        raise AuthError("Token response missing a valid expires_in")
    tokens = {
        "access_token": access_token,
        "expires_at": time.time() + float(expires_in),
    }
    if isinstance(payload.get("refresh_token"), str):
        tokens["refresh_token"] = payload["refresh_token"]
    return tokens


# ---------------------------------------------------------------------------
# Token persistence


_lock = threading.Lock()
_memory_tokens: Optional[dict] = None


def _read_tokens() -> Optional[dict]:
    global _memory_tokens
    if _memory_tokens:
        return _memory_tokens
    try:
        with open(config.AUTH_FILE, encoding="utf-8") as handle:
            _memory_tokens = json.load(handle)
    except (OSError, ValueError):
        return None
    return _memory_tokens


def _write_tokens(tokens: Optional[dict]) -> None:
    global _memory_tokens
    _memory_tokens = tokens
    if tokens is None:
        try:
            os.remove(config.AUTH_FILE)
        except OSError:
            pass
        return
    os.makedirs(os.path.dirname(config.AUTH_FILE), exist_ok=True)
    # Create 0600 before writing the secret, never after.
    fd = os.open(config.AUTH_FILE, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, stat.S_IRUSR | stat.S_IWUSR)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(tokens, handle)


def is_signed_in() -> bool:
    tokens = _read_tokens()
    return bool(tokens and tokens.get("refresh_token"))


def sign_out() -> None:
    with _lock:
        _write_tokens(None)


def invalidate_access_token() -> None:
    """Force a refresh on the next get_access_token(). Used when the server
    rejects a bearer that still looks fresh locally (revoked session, clock
    skew) — otherwise every call would repeat the same 401 until expiry."""
    with _lock:
        tokens = _read_tokens()
        if tokens:
            _write_tokens({**tokens, "expires_at": 0.0})


# ---------------------------------------------------------------------------
# Device flow


def start_device_authorization() -> dict:
    """Returns {device_code, user_code, verification_uri, verification_uri_complete,
    interval, expires_at} — the caller shows the URL to the user and then calls
    poll_for_tokens()."""
    endpoints = _discover_endpoints()
    payload = _post_form(
        endpoints["device_authorization_endpoint"],
        {"client_id": config.OAUTH_CLIENT_ID, "scope": config.OAUTH_SCOPE},
    )
    if "device_code" not in payload:
        raise AuthError(payload.get("error_description") or "Device authorization failed")
    return {
        "device_code": payload["device_code"],
        "user_code": payload.get("user_code", ""),
        "verification_uri": payload.get("verification_uri", ""),
        "verification_uri_complete": payload.get("verification_uri_complete")
        or payload.get("verification_uri", ""),
        "interval": int(payload.get("interval", 5)),
        "expires_at": time.time() + float(payload.get("expires_in", 600)),
    }


def poll_for_tokens(authorization: dict, should_stop=lambda: False) -> None:
    """Blocks until the user approves (persisting the tokens), the code expires,
    or should_stop() returns True."""
    endpoints = _discover_endpoints()
    interval = max(1, int(authorization["interval"]))
    while time.time() < authorization["expires_at"]:
        if should_stop():
            raise AuthError("Sign-in cancelled")
        time.sleep(interval)
        payload = _post_form(
            endpoints["token_endpoint"],
            {
                "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                "device_code": authorization["device_code"],
                "client_id": config.OAUTH_CLIENT_ID,
            },
        )
        error = payload.get("error")
        if error == "authorization_pending":
            continue
        if error == "slow_down":
            interval += 5
            continue
        if error:
            raise AuthError(payload.get("error_description") or f"Sign-in failed ({error})")
        # Re-check after the (blocking) token call: a sign-out or a newer
        # sign-in during approval must not have its cleared state re-persisted.
        if should_stop():
            raise AuthError("Sign-in cancelled")
        with _lock:
            _write_tokens(_to_tokens(payload))
        return
    raise AuthError("Sign-in code expired — start again")


# ---------------------------------------------------------------------------
# Access-token supply for the MCP client


# Serializes refreshes among themselves (parallel refreshes with a rotating
# refresh token would invalidate each other) WITHOUT holding `_lock` during the
# HTTP round-trip — sign_out() only needs `_lock`, so it never waits on the
# token endpoint.
_refresh_lock = threading.Lock()


def _refresh(tokens: dict) -> Optional[dict]:
    refresh_token = tokens.get("refresh_token")
    if not refresh_token:
        return None
    endpoints = _discover_endpoints()
    payload = _post_form(
        endpoints["token_endpoint"],
        {
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": config.OAUTH_CLIENT_ID,
        },
    )
    if payload.get("error"):
        # invalid_grant = the session is dead; anything else is transient and
        # keeps the stored tokens so the next call retries.
        if payload["error"] == "invalid_grant":
            with _lock:
                _write_tokens(None)
            return None
        raise AuthError(payload.get("error_description") or f"Token refresh failed ({payload['error']})")
    fresh = _to_tokens(payload)
    # Keycloak may omit the rotated refresh token; keep the previous one then.
    fresh.setdefault("refresh_token", refresh_token)
    with _lock:
        # A sign-out while this refresh was in flight must win — never
        # resurrect a session the user just ended.
        if _read_tokens() is None:
            return None
        _write_tokens(fresh)
    return fresh


def get_access_token() -> str:
    """Valid access token, refreshing when within 60s of expiry. Raises
    NotSignedInError when there is no session."""
    with _lock:
        tokens = _read_tokens()
    if not tokens:
        raise NotSignedInError()
    if tokens.get("expires_at", 0) - time.time() > 60:
        return tokens["access_token"]
    with _refresh_lock:
        # Re-check under the refresh lock: another thread may have refreshed
        # (or a sign-out may have cleared the session) while we waited.
        with _lock:
            tokens = _read_tokens()
        if not tokens:
            raise NotSignedInError()
        if tokens.get("expires_at", 0) - time.time() > 60:
            return tokens["access_token"]
        fresh = _refresh(tokens)
        if not fresh:
            raise NotSignedInError()
        return fresh["access_token"]
