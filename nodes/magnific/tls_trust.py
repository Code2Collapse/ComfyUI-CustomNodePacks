"""OS trust-store merge so corporate TLS-inspection proxies work.

Twin implementation: hosts/blender/magnific_blender/net.py — keep in sync.
"""

import base64
import re
import ssl
import subprocess
import sys
import threading
from typing import Optional

_PEM_CERT_BODY = re.compile(
    r"-----BEGIN CERTIFICATE-----(.+?)-----END CERTIFICATE-----", re.S
)
# DER encoding of the basicConstraints OID (2.5.29.19).
_BASIC_CONSTRAINTS_OID = b"\x06\x03\x55\x1d\x13"


def _der_is_ca(der: bytes) -> bool:
    # Everything merged into the context becomes a TLS trust anchor, and the
    # macOS keychain sweep lists certs PRESENT, not only ones trusted to issue.
    # stdlib has no X.509 parser, so walk the basicConstraints DER structure:
    # OID, then an OPTIONAL critical flag whose encoding (01 01 ff) is
    # byte-identical to the cA:TRUE boolean (a naive scan would accept every
    # "critical, CA:FALSE" leaf), then the OCTET STRING wrapping a SEQUENCE
    # whose first element is cA:TRUE — absent for CA:FALSE, which DER omits
    # as the DEFAULT value.
    # Only a structurally-valid cA:TRUE returns; anything else keeps scanning,
    # so a coincidental OID byte pattern earlier in the cert (key material,
    # serial) can never mask the real extension further down.
    for match in re.finditer(re.escape(_BASIC_CONSTRAINTS_OID), der):
        at = match.end()
        if der[at : at + 3] == b"\x01\x01\xff":  # critical flag
            at += 3
        # basicConstraints content is tiny, so both lengths are short-form.
        if at + 2 > len(der) or der[at] != 0x04:  # OCTET STRING
            continue
        at += 2
        if at + 2 > len(der) or der[at] != 0x30:  # inner SEQUENCE
            continue
        sequence_length = der[at + 1]
        at += 2
        if sequence_length >= 3 and der[at : at + 3] == b"\x01\x01\xff":
            return True
    return False


def _create_ssl_context() -> tuple[ssl.SSLContext, bool]:
    # Python verifies against its own CA bundle, not the OS trust store, so
    # corporate TLS-inspection roots (deployed via MDM) fail every request
    # with "unable to get local issuer certificate". Only macOS needs the
    # merge below: on Windows create_default_context() already loads the
    # system ROOT/CA stores (SSLContext.load_default_certs), and Linux
    # follows SSL_CERT_FILE/DIR. The boolean reports whether the merge
    # resolved, so a transient keychain failure can be retried by the caller.
    context = ssl.create_default_context()
    if sys.platform != "darwin":
        return context, True
    try:
        # The default keychain search list covers the login AND System
        # keychains, which is where MDM-deployed corporate roots land. Known
        # accepted gap: per-cert trust settings ("Never Trust") are not
        # consulted — that costs one `security verify-cert` subprocess per
        # certificate. Same trade-off as the CEP twin (systemCa.ts).
        result = subprocess.run(
            ["security", "find-certificate", "-a", "-p"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        # A non-zero exit means the export did not run (locked keychain,
        # missing binary shim) — retryable, NOT an empty-but-successful sweep.
        if result.returncode != 0:
            return context, False
        cas = 0
        loaded = 0
        for body in _PEM_CERT_BODY.findall(result.stdout):
            try:
                der = base64.b64decode("".join(body.split()), validate=True)
            except ValueError:
                continue
            if not _der_is_ca(der):
                continue
            cas += 1
            try:
                # The PEM text, not the DER bytes: cadata accepts either, but
                # only the ASCII form is unambiguous across OpenSSL builds.
                context.load_verify_locations(
                    cadata=f"-----BEGIN CERTIFICATE-----{body}-----END CERTIFICATE-----\n"
                )
                loaded += 1
            except ssl.SSLError:
                continue  # one malformed cert must not drop the rest
        # Every anchor rejected is a failed merge, not an empty-but-successful
        # sweep: reporting success here would pin a context with no OS roots.
        if cas and not loaded:
            return context, False
    except Exception:
        # The bundled defaults still apply on non-intercepted networks.
        return context, False
    return context, True


# Built lazily on first use so importing the node pack never blocks ComfyUI
# startup on the keychain export subprocess. A failed merge (locked keychain,
# EDR stall) gets ONE retry before pinning, mirroring the CEP twin
# (systemCa.ts): unbounded retries would re-spawn the subprocess on every
# request.
_MAX_MERGE_ATTEMPTS = 2

# Optional[…] rather than `ssl.SSLContext | None`: pyproject allows Python 3.9,
# where a module-level annotation is evaluated at import time and `|` on types
# raises TypeError — taking auth and MCP down with it.
_ssl_context: Optional[ssl.SSLContext] = None
_ssl_context_lock = threading.Lock()
_failed_merges = 0


def get_ssl_context() -> ssl.SSLContext:
    # The lock serializes concurrent first requests (one keychain export
    # instead of N) and stops a losing failed attempt from overwriting a
    # winning merged context with the default bundle.
    global _ssl_context, _failed_merges
    with _ssl_context_lock:
        if _ssl_context is not None:
            return _ssl_context
        context, merged = _create_ssl_context()
        if not merged:
            _failed_merges += 1
            if _failed_merges < _MAX_MERGE_ATTEMPTS:
                return context  # retry the export on the next request
        _ssl_context = context
        return context
