"""C2C Vault — authenticated encryption for a locked subgraph payload.

Threat model, stated up front so nobody over-trusts this
--------------------------------------------------------
This protects a distributed workflow against CASUAL inspection and copying: a
recipient who opens the .json, or loads the graph in ComfyUI, sees an opaque
blob instead of your node wiring.

It does NOT protect against a determined attacker who can run Python in the same
process. To execute, the subgraph must be decrypted into memory, and the session
key lives in that same process. Anyone able to attach a debugger, edit the pack,
or dump memory can recover the graph. That is inherent to running the workflow
at all, not a flaw in the cipher — the same limit applies to every DRM scheme
that must eventually execute the thing it protects.

Treat it as a lock on a door, not a safe.

Construction
------------
    key   = PBKDF2-HMAC-SHA256(password, salt, iterations)   # stdlib hashlib
    blob  = header || nonce || AESGCM(key).encrypt(nonce, plaintext, aad=header)

* AES-GCM is authenticated, so a wrong password, a flipped byte, or a payload
  moved between vaults all fail the SAME way: the GCM tag check raises. There is
  no padding oracle and no partial decryption to probe, because GCM verifies the
  tag before returning any plaintext.
* The header is passed as AAD, so `vault_id`, the KDF parameters and the
  boundary hash are all authenticated even though they are stored in clear. An
  attacker cannot swap a blob into a different vault, or lower the iteration
  count, without invalidating the tag.
* The boundary hash binds the ciphertext to the node's declared input/output
  sockets, so a payload cannot be transplanted into a vault with a different
  interface and silently mis-execute.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import struct
from typing import Any

MAGIC = b"C2CVLT1"
MODE_LOCKED = 0      # key derived from the password only
MODE_SEALED = 1      # data key ALSO wrapped under a pack constant (runs unprompted)
KDF_ITERATIONS = 600_000          # OWASP 2023 floor for PBKDF2-HMAC-SHA256
SALT_LEN = 16
NONCE_LEN = 12                    # 96-bit, the GCM-recommended size
KEY_LEN = 32                      # AES-256

_INSTALL_HINT = (
    "C2C Vault needs the `cryptography` package for AES-GCM.\n"
    "Install it into the ComfyUI python:\n"
    "    <ComfyUI python> -m pip install cryptography\n"
    "Vault nodes stay registered but cannot lock or unlock until it is present."
)


class VaultError(RuntimeError):
    """Any vault failure. The message never distinguishes wrong-password from
    tampering — telling them apart would hand an attacker an oracle."""


def _aesgcm():
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    except Exception as exc:  # pragma: no cover - environment dependent
        raise VaultError(_INSTALL_HINT) from exc
    return AESGCM


def derive_key(password: str, salt: bytes, iterations: int = KDF_ITERATIONS) -> bytes:
    """PBKDF2-HMAC-SHA256. stdlib only, so no extra dependency for the KDF."""
    if not isinstance(password, str) or not password:
        raise VaultError("A vault password is required.")
    if len(salt) != SALT_LEN:
        raise VaultError("Malformed vault: bad salt length.")
    if not (10_000 <= int(iterations) <= 10_000_000):
        # Bounded so a tampered header cannot force a 1-iteration derivation
        # (trivially brute-forced) or a denial of service.
        raise VaultError("Malformed vault: KDF iteration count out of range.")
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, int(iterations), KEY_LEN)


def boundary_hash(boundary_in: list, boundary_out: list) -> str:
    """Stable digest of the vault's declared interface.

    Included in the authenticated header so a payload cannot be moved into a
    vault whose sockets differ - that would execute the wrong graph against the
    wrong wiring and produce plausible-looking nonsense.
    """
    blob = json.dumps(
        {"in": list(boundary_in or []), "out": list(boundary_out or [])},
        sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _pack_header(vault_id: str, salt: bytes, iterations: int, bhash: str, mode: int = 0) -> bytes:
    vid = vault_id.encode("utf-8")
    bh = bhash.encode("ascii")
    if len(vid) > 255 or len(bh) != 64:
        raise VaultError("Malformed vault header fields.")
    return b"".join([
        MAGIC,
        struct.pack("!B", int(mode)),
        struct.pack("!B", len(vid)), vid,
        struct.pack("!I", int(iterations)),
        salt,
        bh,
    ])


def _unpack_header(blob: bytes) -> tuple[bytes, str, bytes, int, str, int, int]:
    if not blob.startswith(MAGIC):
        raise VaultError("Not a C2C vault payload.")
    i = len(MAGIC)
    (mode,) = struct.unpack_from("!B", blob, i); i += 1
    (vid_len,) = struct.unpack_from("!B", blob, i); i += 1
    vault_id = blob[i:i + vid_len].decode("utf-8", "replace"); i += vid_len
    (iterations,) = struct.unpack_from("!I", blob, i); i += 4
    salt = blob[i:i + SALT_LEN]; i += SALT_LEN
    bhash = blob[i:i + 64].decode("ascii", "replace"); i += 64
    if len(salt) != SALT_LEN or len(bhash) != 64:
        raise VaultError("Truncated vault header.")
    return blob[:i], vault_id, salt, iterations, bhash, i, mode


def lock_subgraph(
    subgraph: dict[str, Any],
    password: str,
    *,
    vault_id: str,
    iterations: int = KDF_ITERATIONS,
) -> str:
    """Encrypt a subgraph dict -> base64 payload string for the node widget."""
    if not isinstance(subgraph, dict):
        raise VaultError("Subgraph must be an object.")
    for key in ("nodes", "links", "boundary_in", "boundary_out"):
        if key not in subgraph:
            raise VaultError(f"Subgraph missing required key {key!r}.")

    bhash = boundary_hash(subgraph["boundary_in"], subgraph["boundary_out"])
    salt = os.urandom(SALT_LEN)
    nonce = os.urandom(NONCE_LEN)
    header = _pack_header(vault_id, salt, iterations, bhash)
    key = derive_key(password, salt, iterations)

    plaintext = json.dumps(subgraph, separators=(",", ":")).encode("utf-8")
    ct = _aesgcm()(key).encrypt(nonce, plaintext, header)   # header is AAD
    return base64.b64encode(header + nonce + ct).decode("ascii")


def unlock_subgraph(payload: str, password: str, *, vault_id: str | None = None) -> dict[str, Any]:
    """Decrypt a payload. Raises VaultError on ANY failure, indistinguishably."""
    try:
        blob = base64.b64decode(payload.encode("ascii"), validate=True)
    except Exception as exc:
        raise VaultError("Vault payload is not valid base64.") from exc

    header, hdr_vault_id, salt, iterations, _bhash, hlen, _mode = _unpack_header(blob)
    if vault_id is not None and not hmac.compare_digest(hdr_vault_id, vault_id):
        # Constant-time so the id cannot be probed character by character.
        raise VaultError("Vault payload does not belong to this vault.")

    nonce = blob[hlen:hlen + NONCE_LEN]
    ct = blob[hlen + NONCE_LEN:]
    if len(nonce) != NONCE_LEN or not ct:
        raise VaultError("Truncated vault payload.")

    key = derive_key(password, salt, iterations)
    try:
        plaintext = _aesgcm()(key).decrypt(nonce, ct, header)
    except VaultError:
        raise
    except Exception as exc:
        # ONE message for wrong password AND for tampering. Distinguishing them
        # is exactly the oracle an attacker wants.
        raise VaultError(
            "Could not open the vault: wrong password, or the payload has been "
            "modified. (These are deliberately reported the same way.)"
        ) from exc

    try:
        out = json.loads(plaintext.decode("utf-8"))
    except Exception as exc:
        raise VaultError("Vault opened but its contents are not valid JSON.") from exc
    if not isinstance(out, dict):
        raise VaultError("Vault contents are not a subgraph object.")
    return out


def payload_vault_id(payload: str) -> str:
    """Read the vault id from a payload WITHOUT the password.

    Safe: the id is stored in clear (it is authenticated, not secret) so the UI
    can tell which vault a blob belongs to before prompting.
    """
    try:
        blob = base64.b64decode(payload.encode("ascii"), validate=True)
    except Exception as exc:
        raise VaultError("Vault payload is not valid base64.") from exc
    return _unpack_header(blob)[1]


# ─────────────────────── SEALED mode (envelope encryption) ───────────────────
#
# A sealed vault RUNS WITHOUT A PASSWORD. That is the point, and it is also the
# limit: if the recipient's machine can decrypt unaided, the unwrapping secret is
# ON that machine. `_PACK_SECRET` below is a constant in this file - anyone who
# reads the source can recover any sealed payload.
#
# So: sealing is OBFUSCATION. It stops someone opening your .json and reading the
# graph, or pasting your nodes into their own workflow. It does not stop a
# determined extractor, and no scheme that must execute unattended can. Say so in
# the UI rather than letting a user believe otherwise.
#
# One data key, wrapped twice (standard envelope form, two "recipients"):
#   data_key                       = random 32 bytes, encrypts the subgraph
#   wrap_pack     = AESGCM(KDF(_PACK_SECRET, salt)).encrypt(data_key)   -> auto-run
#   wrap_password = AESGCM(KDF(password,     salt)).encrypt(data_key)   -> open/edit
#
# The password wrap is why an author can re-open a sealed vault on ANY machine,
# including one where they never sealed it. Losing the original graph does not
# lose the vault.
_PACK_SECRET = "c2c-vault-sealed-v1"     # deliberately not a secret; see above


def seal_subgraph(
    subgraph: dict[str, Any],
    password: str,
    *,
    vault_id: str,
    iterations: int = KDF_ITERATIONS,
) -> str:
    """Encrypt so it runs unattended, but only opens with the password."""
    if not isinstance(subgraph, dict):
        raise VaultError("Subgraph must be an object.")
    for key in ("nodes", "links", "boundary_in", "boundary_out"):
        if key not in subgraph:
            raise VaultError(f"Subgraph missing required key {key!r}.")

    bhash = boundary_hash(subgraph["boundary_in"], subgraph["boundary_out"])
    salt = os.urandom(SALT_LEN)
    header = _pack_header(vault_id, salt, iterations, bhash, mode=MODE_SEALED)

    data_key = os.urandom(KEY_LEN)
    aes = _aesgcm()

    n_pack, n_pw, n_ct = os.urandom(NONCE_LEN), os.urandom(NONCE_LEN), os.urandom(NONCE_LEN)
    wrap_pack = aes(derive_key(_PACK_SECRET, salt, iterations)).encrypt(n_pack, data_key, header)
    wrap_pw = aes(derive_key(password, salt, iterations)).encrypt(n_pw, data_key, header)
    ct = aes(data_key).encrypt(n_ct, json.dumps(subgraph, separators=(",", ":")).encode("utf-8"), header)

    body = b"".join([
        struct.pack("!H", len(wrap_pack)), n_pack, wrap_pack,
        struct.pack("!H", len(wrap_pw)), n_pw, wrap_pw,
        n_ct, ct,
    ])
    return base64.b64encode(header + body).decode("ascii")


def _sealed_parts(payload: str, vault_id: str | None):
    blob = base64.b64decode(payload.encode("ascii"), validate=True)
    header, hdr_id, salt, iters, _bh, i, mode = _unpack_header(blob)
    if mode != MODE_SEALED:
        raise VaultError("This payload is not a sealed vault.")
    if vault_id is not None and not hmac.compare_digest(hdr_id, vault_id):
        raise VaultError("Vault payload does not belong to this vault.")

    (n1,) = struct.unpack_from("!H", blob, i); i += 2
    np_, wp = blob[i:i + NONCE_LEN], blob[i + NONCE_LEN:i + NONCE_LEN + n1]
    i += NONCE_LEN + n1
    (n2,) = struct.unpack_from("!H", blob, i); i += 2
    npw, wpw = blob[i:i + NONCE_LEN], blob[i + NONCE_LEN:i + NONCE_LEN + n2]
    i += NONCE_LEN + n2
    n_ct, ct = blob[i:i + NONCE_LEN], blob[i + NONCE_LEN:]
    if not ct:
        raise VaultError("Truncated vault payload.")
    return header, salt, iters, (np_, wp), (npw, wpw), (n_ct, ct)


def _open_sealed(payload: str, unwrap_key: bytes, which: str, vault_id: str | None) -> dict[str, Any]:
    header, _salt, _it, pack, pw, body = _sealed_parts(payload, vault_id)
    nonce, wrapped = pack if which == "pack" else pw
    aes = _aesgcm()
    try:
        data_key = aes(unwrap_key).decrypt(nonce, wrapped, header)
        plain = aes(data_key).decrypt(body[0], body[1], header)
    except Exception as exc:
        raise VaultError(
            "Could not open the vault: wrong password, or the payload has been "
            "modified. (These are deliberately reported the same way.)"
        ) from exc
    out = json.loads(plain.decode("utf-8"))
    if not isinstance(out, dict):
        raise VaultError("Vault contents are not a subgraph object.")
    return out


def unseal_for_run(payload: str, *, vault_id: str | None = None) -> dict[str, Any]:
    """Unwrap with the pack constant. No password - this is the auto-run path."""
    _h, salt, iters, _p, _w, _b = _sealed_parts(payload, vault_id)
    return _open_sealed(payload, derive_key(_PACK_SECRET, salt, iters), "pack", vault_id)


def unseal_with_password(payload: str, password: str, *, vault_id: str | None = None) -> dict[str, Any]:
    """Unwrap with the author's password - the open/edit path."""
    _h, salt, iters, _p, _w, _b = _sealed_parts(payload, vault_id)
    return _open_sealed(payload, derive_key(password, salt, iters), "password", vault_id)


def payload_mode(payload: str) -> int:
    """MODE_LOCKED or MODE_SEALED, readable without any key."""
    blob = base64.b64decode(payload.encode("ascii"), validate=True)
    return _unpack_header(blob)[6]
