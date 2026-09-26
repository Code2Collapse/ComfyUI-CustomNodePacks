"""Local-UI guard for the Magnific HTTP routes (TH-651).

The routes in routes.py read from and act on the signed-in Magnific account, so
they must only ever be reachable from this ComfyUI process's own web UI — never
from another origin the victim's browser happens to visit, and never from
another host on the network. ComfyUI's own origin_only_middleware cannot deliver
that: it resolves the `Host` *header* through DNS (CWE-350), so a DNS rebind
satisfies it; `--enable-cors-header` replaces it with a permissive `*` handler;
and it disengages entirely when no `Origin` header is present. The account-facing
routes therefore carry their own guard, independent of any upstream middleware.

Three checks, applied per request (they are cheap and every route runs them):

1. The TCP *peer* must be loopback. This is the socket's real source address,
   not a header, so a non-browser client on the network cannot forge it (a TCP
   handshake cannot complete from a spoofed source). This is what actually
   closes the LAN/`--listen` case: `Host` and `Sec-Fetch-Site` are both
   client-controlled and prove nothing on their own.
2. `Host` must be a loopback *literal* (127.0.0.0/8, ::1, localhost). Parsed as
   an address, never resolved, so `attacker.example` — the Host a DNS rebind
   leaves in place while the peer is the victim's own loopback — is rejected.
3. A per-process token in a custom request header. A custom header forces a CORS
   preflight (defeating the text/plain simple-request CSRF the upstream control
   missed), and the token cannot be read by another origin — including a rebound
   same-origin page, because the token is only ever handed to a caller that
   already passed checks 1-2 (see the csrf-token bootstrap route).

Fetch-metadata is checked too as a belt: an explicit cross-site/same-site
`Sec-Fetch-Site` is refused, and any `Origin` that is present must name a
loopback host. A browser attaches `Origin` to every cross-origin request even
when it omits `Sec-Fetch-Site` (older Safari, non-secure contexts), so the
Origin check — not fetch-metadata alone — is what stops a cross-origin page
reading the token under the permissive CORS flag.

The peer check (1) is never relaxed — a genuine local UI always connects over
loopback (the browser on the same machine, or a same-box reverse proxy).
MAGNIFIC_TRUSTED_HOSTS relaxes only the Host *header* literal (2), for a local
proxy terminating a custom hostname; it does not let a remote peer in.
"""

import ipaddress
import os
import secrets
import urllib.parse

# Sent by web/magnific.js on every route call; absent means "not the local UI".
TOKEN_HEADER = "X-Magnific-Token"

# Regenerated on every process start: a session that outlives one ComfyUI run is
# never a valid reason to accept a stale token, and there is no need to persist.
UI_TOKEN = secrets.token_urlsafe(32)

# Escape hatch for a deployment that KNOWINGLY serves this ComfyUI over a
# non-loopback host (a personal reverse proxy or tunnel with its own auth in
# front). Comma-separated hostnames; empty by default. The token (check 2) is
# still required for these hosts, so relaxing check 1 does not open the routes
# to other origins — it only lets the local UI reach them under that hostname.
_TRUSTED_HOSTS = {
    name.strip().lower()
    for name in os.environ.get("MAGNIFIC_TRUSTED_HOSTS", "").split(",")
    if name.strip()
}

# Sec-Fetch-Site values that prove the request did NOT originate from this
# origin's own page. `same-origin` and `none` (typed URL, non-browser client)
# are allowed; the two below never are.
_FORBIDDEN_FETCH_SITES = frozenset({"cross-site", "same-site"})


def _hostname_from_host_header(value: str) -> str:
    """The bare hostname from a `Host` header, port and IPv6 brackets removed.
    No DNS: this only splits the literal the client sent."""
    value = (value or "").strip()
    if not value:
        return ""
    if value[0] == "[":  # bracketed IPv6, e.g. [::1]:8188
        end = value.find("]")
        return value[1:end] if end != -1 else ""
    # A valid IPv4/hostname Host has at most one colon (the port); a bare
    # (unbracketed) IPv6 is malformed and correctly fails the loopback parse.
    if value.count(":") == 1:
        value = value.split(":", 1)[0]
    return value


def _is_loopback_ip(value: str) -> bool:
    """True for an IPv4/IPv6 loopback literal, including an IPv4-mapped IPv6
    peer (::ffff:127.0.0.1, how some stacks report a loopback socket). Never
    resolves a name."""
    if not value:
        return False
    try:
        address = ipaddress.ip_address(value)
    except ValueError:
        return False
    if address.is_loopback:
        return True
    mapped = getattr(address, "ipv4_mapped", None)
    return bool(mapped is not None and mapped.is_loopback)


def _is_loopback_name(hostname: str) -> bool:
    """True for `localhost`, a trusted host, or a loopback IP literal. Never
    resolves a name."""
    if not hostname:
        return False
    if hostname.lower() in _TRUSTED_HOSTS:
        return True
    if hostname == "localhost":
        return True
    return _is_loopback_ip(hostname)


def _is_loopback_host(host_header: str) -> bool:
    return _is_loopback_name(_hostname_from_host_header(host_header))


def _fetch_site_ok(headers) -> bool:
    return headers.get("Sec-Fetch-Site") not in _FORBIDDEN_FETCH_SITES


def _origin_ok(headers) -> bool:
    """`Origin`, when present, must name a loopback host. A browser attaches
    Origin to every cross-origin request (even where it omits Sec-Fetch-Site, as
    older Safari does), so this — not fetch-metadata alone — is what refuses a
    cross-origin page reading the token under --enable-cors-header. Absent Origin
    (a same-origin GET, or a non-browser client) is left to the peer + Host +
    token checks. An opaque `null` origin has no hostname and is refused."""
    origin = headers.get("Origin")
    if not origin:
        return True
    return _is_loopback_name(urllib.parse.urlsplit(origin).hostname or "")


def is_local_ui_origin(headers, peer_ip) -> bool:
    """Peer + Host + fetch-metadata (Origin and Sec-Fetch-Site). Used by the
    csrf-token bootstrap route, which cannot require the token it exists to hand
    out — the loopback-peer check is what keeps a network client from obtaining
    it, and the Origin check keeps a cross-origin browser page from reading it."""
    if not _is_loopback_ip(peer_ip):
        return False
    if not _is_loopback_host(headers.get("Host", "")):
        return False
    return _fetch_site_ok(headers) and _origin_ok(headers)


def is_authorized(headers, peer_ip) -> bool:
    """Full guard for the account-facing routes: local-UI origin AND the token."""
    if not is_local_ui_origin(headers, peer_ip):
        return False
    return secrets.compare_digest(headers.get(TOKEN_HEADER, ""), UI_TOKEN)
