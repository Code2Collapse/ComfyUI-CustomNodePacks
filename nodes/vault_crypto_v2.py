"""C2C Vault envelope v2 (`C2CVLT2`, owner directive A17): Argon2id + AES-256-GCM. Design: docs/VAULT_DESIGN_A17.md.

A fresh random data key (DEK) encrypts the graph; the DEK is wrapped for whoever may open or run it:
  password wrap  - KEK = Argon2id(password, salt): opening for editing (both modes) and running a Locked vault
  install wraps  - X25519 to each chosen install's public key: passwordless runs of a Sealed vault on those installs
  portable wrap  - a public pack constant: "Sealed - portable (obfuscation only)", runs anywhere, protects nothing
The header (version, mode, vault id, KDF parameters, salt, interface hashes) is the associated data of every AES-GCM
operation, so changing any of it fails authentication. Decryption is all or nothing: GCM checks the tag before it
returns a byte. Wrong password and tampering produce the same message.

Honest boundary: whoever can RUN a vault on a machine can decrypt it there (the graph must be plaintext to execute).
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import struct
import unicodedata
from typing import Any, Optional

from .vault_crypto import VaultError

MAGIC2 = b"C2CVLT2"
VERSION = 2
MODE_LOCKED = 0
MODE_SEALED_INSTALLS = 1
MODE_SEALED_PORTABLE = 2
MODES = (MODE_LOCKED, MODE_SEALED_INSTALLS, MODE_SEALED_PORTABLE)
KDF_ARGON2ID = 1
ARGON2_V1 = {"memory_kib": 131072, "iterations": 3, "lanes": 4}      # measured 0.11 s on the build machine
MEMORY_KIB_BOUNDS = (65536, 1048576)
ITERATION_BOUNDS = (2, 10)
LANE_BOUNDS = (1, 16)
SALT_LEN, NONCE_LEN, KEY_LEN, TAG_LEN = 16, 12, 32, 16
MAX_PAYLOAD_BYTES = 64 * 1024 * 1024
MAX_RECIPIENTS = 64
# deliberately public: a portable seal is obfuscation, and the UI says so
_PORTABLE_SECRET = b"c2c-vault-portable-v2-not-a-secret"
_SAME_ERROR = ("Could not open the vault: wrong password, or the payload has been modified. (These are deliberately "
               "reported the same way.)")


def _aes():
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    return AESGCM


def _hkdf(material: bytes, salt: bytes, info: bytes) -> bytes:
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF

    return HKDF(algorithm=hashes.SHA256(), length=KEY_LEN, salt=salt, info=info).derive(material)


def normalize_password(password: str) -> bytes:
    """NFC on both sides, never truncated: the same passphrase always derives the same key."""
    if not isinstance(password, str) or not password:
        raise VaultError("A vault password is required.")
    return unicodedata.normalize("NFC", password).encode("utf-8")


def kek_from_password(password: str, salt: bytes, kdf: dict) -> bytes:
    from cryptography.hazmat.primitives.kdf.argon2 import Argon2id

    check_kdf(kdf)
    return Argon2id(salt=salt, length=KEY_LEN, iterations=int(kdf["iterations"]), lanes=int(kdf["lanes"]),
                    memory_cost=int(kdf["memory_kib"])).derive(normalize_password(password))


def check_kdf(kdf: dict) -> None:
    """A header can neither weaken the KDF below the floor nor demand absurd memory / time."""
    m, t, p = int(kdf["memory_kib"]), int(kdf["iterations"]), int(kdf["lanes"])
    if not (MEMORY_KIB_BOUNDS[0] <= m <= MEMORY_KIB_BOUNDS[1] and ITERATION_BOUNDS[0] <= t <= ITERATION_BOUNDS[1]
            and LANE_BOUNDS[0] <= p <= LANE_BOUNDS[1]):
        raise VaultError("Malformed vault: KDF parameters out of range.")


def canonical_hash(obj: Any) -> bytes:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")).digest()


def interface_struct(iface: dict) -> dict:
    """The authenticated part of the public interface (the node's vault_interface manifest, js buildInterfaceManifest):
    socket names / types and promoted parameter ids / names / types. Never values, never which socket a promoted
    parameter was converted to - those change in normal use after locking."""
    iface = iface or {}
    return {"in": [[str(s.get("name")), str(s.get("type"))] for s in iface.get("in", []) or []],
            "out": [[str(s.get("name")), str(s.get("type"))] for s in iface.get("out", []) or []],
            "params": [[str(p.get("id")), str(p.get("name")), str(p.get("type"))]
                       for p in iface.get("params", []) or []]}


def key_id(public_raw: bytes) -> bytes:
    return hashlib.sha256(public_raw).digest()[:8]


# ── the header ──────────────────────────────────────────────────────────────────────────────────────────────────────

def _header(mode: int, vault_id: str, kdf: dict, salt: bytes, bhash: bytes, ihash: bytes) -> bytes:
    vid = vault_id.encode("utf-8")
    if mode not in MODES or not vid or len(vid) > 255 or len(salt) != SALT_LEN or len(bhash) != 32 or len(ihash) != 32:
        raise VaultError("Malformed vault header fields.")
    return b"".join([MAGIC2, struct.pack("!BB", VERSION, mode), struct.pack("!B", len(vid)), vid,
                     struct.pack("!BIIB", KDF_ARGON2ID, int(kdf["memory_kib"]), int(kdf["iterations"]),
                                 int(kdf["lanes"])), salt, bhash, ihash])


def is_v2(payload: str) -> bool:
    try:
        return base64.b64decode(payload[:16].encode("ascii") + b"==", validate=False).startswith(MAGIC2)
    except Exception:  # noqa: BLE001
        return False


def parse(payload: str) -> dict:
    """Every structural check BEFORE any key is derived or anything decrypted."""
    if not isinstance(payload, str) or not payload.strip():
        raise VaultError("Vault payload is empty.")
    if len(payload) > MAX_PAYLOAD_BYTES * 4 // 3 + 4:
        raise VaultError("Vault payload is too large.")
    try:
        blob = base64.b64decode(payload.encode("ascii"), validate=True)
    except Exception as exc:
        raise VaultError("Vault payload is not valid base64.") from exc
    if not blob.startswith(MAGIC2):
        raise VaultError("Not a C2C vault v2 payload.")
    try:
        i = len(MAGIC2)
        version, mode = struct.unpack_from("!BB", blob, i); i += 2
        if version != VERSION:
            raise VaultError(f"Unsupported vault format version {version}.")
        if mode not in MODES:
            raise VaultError("Malformed vault: unknown mode.")
        (vlen,) = struct.unpack_from("!B", blob, i); i += 1
        vault_id = blob[i:i + vlen].decode("utf-8"); i += vlen
        kdf_id, mem, it, lanes = struct.unpack_from("!BIIB", blob, i); i += 10
        if kdf_id != KDF_ARGON2ID:
            raise VaultError("Malformed vault: unknown KDF.")
        kdf = {"memory_kib": mem, "iterations": it, "lanes": lanes}
        check_kdf(kdf)
        salt = blob[i:i + SALT_LEN]; i += SALT_LEN
        bhash = blob[i:i + 32]; i += 32
        ihash = blob[i:i + 32]; i += 32
        if len(ihash) != 32:
            raise VaultError("Truncated vault header.")
        header = blob[:i]
        pw = (blob[i:i + NONCE_LEN], blob[i + NONCE_LEN:i + NONCE_LEN + KEY_LEN + TAG_LEN])
        i += NONCE_LEN + KEY_LEN + TAG_LEN
        (count,) = struct.unpack_from("!H", blob, i); i += 2
        if count > MAX_RECIPIENTS:
            raise VaultError("Malformed vault: too many recipients.")
        installs = []
        for _ in range(count):
            kid, eph = blob[i:i + 8], blob[i + 8:i + 40]
            nonce = blob[i + 40:i + 52]
            wrapped = blob[i + 52:i + 52 + KEY_LEN + TAG_LEN]
            if len(wrapped) != KEY_LEN + TAG_LEN:
                raise VaultError("Truncated vault payload.")
            installs.append((kid, eph, nonce, wrapped))
            i += 52 + KEY_LEN + TAG_LEN
        (has_portable,) = struct.unpack_from("!B", blob, i); i += 1
        portable = None
        if has_portable:
            portable = (blob[i:i + NONCE_LEN], blob[i + NONCE_LEN:i + NONCE_LEN + KEY_LEN + TAG_LEN])
            i += NONCE_LEN + KEY_LEN + TAG_LEN
        body = (blob[i:i + NONCE_LEN], blob[i + NONCE_LEN:])
    except VaultError:
        raise
    except Exception as exc:  # struct.error, UnicodeDecodeError: a truncated or mangled blob
        raise VaultError("Truncated or malformed vault payload.") from exc
    if len(pw[1]) != KEY_LEN + TAG_LEN or len(body[0]) != NONCE_LEN or len(body[1]) <= TAG_LEN:
        raise VaultError("Truncated vault payload.")
    if (mode == MODE_SEALED_PORTABLE) != (portable is not None):
        raise VaultError("Malformed vault: mode and wraps disagree.")
    return {"header": header, "mode": mode, "vault_id": vault_id, "kdf": kdf, "salt": salt, "bhash": bhash,
            "ihash": ihash, "pw": pw, "installs": installs, "portable": portable, "body": body}


def info(payload: str) -> dict:
    """What anyone may read without a key: mode, vault id, KDF, the installs it was sealed for."""
    p = parse(payload)
    return {"version": VERSION, "mode": p["mode"], "vault_id": p["vault_id"], "kdf": p["kdf"],
            "installs": [kid.hex() for kid, *_ in p["installs"]]}


# ── encrypting ─────────────────────────────────────────────────────────────────────────────────────────────────────

def _encrypt(plain: dict, *, mode: int, vault_id: str, kdf: dict, salt: bytes, kek: bytes, recipients: list,
             bhash: bytes, ihash: bytes) -> str:
    from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey, X25519PublicKey

    aes = _aes()
    header = _header(mode, vault_id, kdf, salt, bhash, ihash)
    dek = os.urandom(KEY_LEN)                                   # fresh for every lock: nonces never repeat under it
    parts = []
    n = os.urandom(NONCE_LEN)
    parts.append(n + aes(kek).encrypt(n, dek, header))
    installs = list(recipients or []) if mode == MODE_SEALED_INSTALLS else []
    if mode == MODE_SEALED_INSTALLS and not installs:
        raise VaultError("A vault sealed for installs needs at least one install key.")
    if len(installs) > MAX_RECIPIENTS:
        raise VaultError(f"At most {MAX_RECIPIENTS} installs per vault.")
    parts.append(struct.pack("!H", len(installs)))
    for pub in installs:
        eph = X25519PrivateKey.generate()
        shared = eph.exchange(X25519PublicKey.from_public_bytes(pub))
        wk = _hkdf(shared, salt, b"c2c-vault-v2 install wrap|" + vault_id.encode("utf-8"))
        from cryptography.hazmat.primitives import serialization

        eph_pub = eph.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
        n = os.urandom(NONCE_LEN)
        parts.append(key_id(pub) + eph_pub + n + aes(wk).encrypt(n, dek, header))
    if mode == MODE_SEALED_PORTABLE:
        n = os.urandom(NONCE_LEN)
        parts.append(b"\x01" + n + aes(_hkdf(_PORTABLE_SECRET, salt, b"c2c-vault-v2 portable")).encrypt(n, dek, header))
    else:
        parts.append(b"\x00")
    n = os.urandom(NONCE_LEN)
    data = json.dumps(plain, separators=(",", ":")).encode("utf-8")
    parts.append(n + aes(dek).encrypt(n, data, header))
    blob = header + b"".join(parts)
    if len(blob) > MAX_PAYLOAD_BYTES:
        raise VaultError("This graph is too large for a vault (64 MB).")
    return base64.b64encode(blob).decode("ascii")


def _check_plain(plain: dict) -> dict:
    if not isinstance(plain, dict) or not isinstance(plain.get("exec"), dict):
        raise VaultError("Vault contents must be {'exec': ..., 'layout': ...}.")
    ex = plain["exec"]
    for key in ("nodes", "links", "boundary_in", "boundary_out"):
        if key not in ex:
            raise VaultError(f"Subgraph missing required key {key!r}.")
    return ex


def new_kek(password: str, kdf: Optional[dict] = None) -> dict:
    """A fresh salt and the KEK the password derives with it: {kek, salt, kdf}."""
    kdf = dict(kdf or ARGON2_V1)
    salt = os.urandom(SALT_LEN)
    return {"kek": kek_from_password(password, salt, kdf), "salt": salt, "kdf": kdf}


def lock(plain: dict, password: str, *, vault_id: str, mode: int, iface: dict, recipients: Optional[list] = None,
         kdf: Optional[dict] = None) -> tuple:
    """New payload + the KEK (for the creator's session). Fresh salt, DEK and nonces."""
    k = new_kek(password, kdf)
    return lock_with(plain, k, vault_id=vault_id, mode=mode, iface=iface, recipients=recipients), k["kek"]


def lock_with(plain: dict, k: dict, *, vault_id: str, mode: int, iface: dict, recipients: Optional[list] = None) -> str:
    """Encrypt with an existing {kek, salt, kdf} (fresh DEK and nonces)."""
    from .vault_crypto import boundary_hash

    ex = _check_plain(plain)
    kdf, salt, kek = k["kdf"], k["salt"], k["kek"]
    bhash = bytes.fromhex(boundary_hash(ex["boundary_in"], ex["boundary_out"]))
    # the installs' PUBLIC keys ride inside the ciphertext: a re-seal after an edit wraps for the same installs,
    # and the list of who may run it is not readable from the file
    inner = dict(plain, format=2, recipients=[base64.b64encode(k).decode("ascii") for k in (recipients or [])]
                 if mode == MODE_SEALED_INSTALLS else [])
    return _encrypt(inner, mode=mode, vault_id=vault_id, kdf=kdf, salt=salt, kek=kek,
                    recipients=recipients, bhash=bhash, ihash=canonical_hash(interface_struct(iface)))


def recipients_of(plain: dict) -> list:
    """The install public keys a decrypted Sealed vault was sealed for (raw 32-byte keys)."""
    return [base64.b64decode(k) for k in (plain.get("recipients") or [])]


def relock(old_payload: str, kek: bytes, plain: dict, *, iface: dict, recipients: Optional[list] = None,
           mode: Optional[int] = None) -> str:
    """Re-encrypt after an edit with the session's KEK: same vault id / salt / KDF, fresh DEK and nonces."""
    from .vault_crypto import boundary_hash

    old = parse(old_payload)
    _unwrap_password(old, kek)                                  # the KEK must belong to this vault
    ex = _check_plain(plain)
    mode = old["mode"] if mode is None else mode
    bhash = bytes.fromhex(boundary_hash(ex["boundary_in"], ex["boundary_out"]))
    inner = dict(plain, format=2, recipients=[base64.b64encode(k).decode("ascii") for k in (recipients or [])]
                 if mode == MODE_SEALED_INSTALLS else [])
    return _encrypt(inner, mode=mode, vault_id=old["vault_id"], kdf=old["kdf"], salt=old["salt"],
                    kek=kek, recipients=recipients, bhash=bhash, ihash=canonical_hash(interface_struct(iface)))


# ── decrypting ─────────────────────────────────────────────────────────────────────────────────────────────────────

def _decrypt_body(p: dict, dek: bytes) -> dict:
    try:
        data = _aes()(dek).decrypt(p["body"][0], p["body"][1], p["header"])
    except Exception as exc:
        raise VaultError(_SAME_ERROR) from exc
    try:
        out = json.loads(data.decode("utf-8"))
    except Exception as exc:
        raise VaultError("Vault opened but its contents are not valid JSON.") from exc
    _check_plain(out)
    return out


def _unwrap_password(p: dict, kek: bytes) -> bytes:
    try:
        return _aes()(kek).decrypt(p["pw"][0], p["pw"][1], p["header"])
    except Exception as exc:
        raise VaultError(_SAME_ERROR) from exc


def _check_vault(p: dict, vault_id: Optional[str]) -> None:
    if vault_id is not None and not hmac.compare_digest(p["vault_id"].encode("utf-8"), str(vault_id).encode("utf-8")):
        raise VaultError("Vault payload does not belong to this vault.")


def check_interface(p: dict, iface: dict) -> None:
    """The clear-text interface on the node must be the one that was locked."""
    if not hmac.compare_digest(p["ihash"], canonical_hash(interface_struct(iface))):
        raise VaultError("The vault's public interface was changed after it was locked (socket names / types or "
                         "promoted parameters). Re-lock the vault, or restore the interface.")


def open_with_password(payload: str, password: str, *, vault_id: Optional[str] = None) -> tuple:
    """(plaintext, KEK, parsed) - for editing, and for a Locked vault's run session."""
    p = parse(payload)
    _check_vault(p, vault_id)
    kek = kek_from_password(password, p["salt"], p["kdf"])
    return _decrypt_body(p, _unwrap_password(p, kek)), kek, p


def open_with_kek(payload: str, kek: bytes, *, vault_id: Optional[str] = None) -> dict:
    p = parse(payload)
    _check_vault(p, vault_id)
    return _decrypt_body(p, _unwrap_password(p, kek))


def open_with_install_key(payload: str, private_raw: bytes, *, vault_id: Optional[str] = None) -> dict:
    """A Sealed vault's passwordless run on an install it was sealed for."""
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey, X25519PublicKey

    p = parse(payload)
    _check_vault(p, vault_id)
    if p["mode"] != MODE_SEALED_INSTALLS:
        raise VaultError("This vault is not sealed for installs.")
    priv = X25519PrivateKey.from_private_bytes(private_raw)
    mine = key_id(priv.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw))
    for kid, eph, nonce, wrapped in p["installs"]:
        if hmac.compare_digest(kid, mine):
            wk = _hkdf(priv.exchange(X25519PublicKey.from_public_bytes(eph)), p["salt"],
                       b"c2c-vault-v2 install wrap|" + p["vault_id"].encode("utf-8"))
            try:
                dek = _aes()(wk).decrypt(nonce, wrapped, p["header"])
            except Exception as exc:
                raise VaultError(_SAME_ERROR) from exc
            return _decrypt_body(p, dek)
    raise VaultError("This vault was not sealed for this install, so it cannot run here without the password. Ask "
                     "its author to seal it for this install's public key (C2C Vault menu), or open it with the "
                     "password.")


def open_portable(payload: str, *, vault_id: Optional[str] = None) -> dict:
    p = parse(payload)
    _check_vault(p, vault_id)
    if p["mode"] != MODE_SEALED_PORTABLE or p["portable"] is None:
        raise VaultError("This vault is not a portable seal.")
    try:
        dek = _aes()(_hkdf(_PORTABLE_SECRET, p["salt"], b"c2c-vault-v2 portable")).decrypt(
            p["portable"][0], p["portable"][1], p["header"])
    except Exception as exc:
        raise VaultError(_SAME_ERROR) from exc
    return _decrypt_body(p, dek)


def wipe(buf) -> None:
    """Best effort: overwrite a mutable key buffer (Python copies bytes freely - not a guarantee)."""
    if isinstance(buf, bytearray):
        for k in range(len(buf)):
            buf[k] = 0
