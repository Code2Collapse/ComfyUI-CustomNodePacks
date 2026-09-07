"""C2C Vault node + session-key store + HTTP routes.

WHY THE PASSWORD IS NOT A WIDGET
--------------------------------
A widget value is serialised into the workflow JSON and into every queued prompt,
which is logged, cached, and often shared alongside the .png. A password there
would travel with the very file the vault exists to protect.

So the password only ever moves over an HTTP route, is used immediately to
derive a key, and only the DERIVED KEY is held - in memory, in the server
process, with a TTL. The node itself sees no password at all; at execute() time
it asks the store for a key and refuses if there is none.

The blob in `vault_payload` IS serialised with the workflow. That is the point:
it is ciphertext.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Any

from .vault_crypto import (
    MODE_SEALED,
    VaultError,
    lock_subgraph,
    payload_mode,
    seal_subgraph,
    unlock_subgraph,
    unseal_for_run,
    unseal_with_password,
)
from .vault_exec import VaultExecError, execute_subgraph

log = logging.getLogger("c2c.vault")

# How long an unlock lasts. 15 minutes was too short in practice - it expires
# mid-render and re-prompts during a batch, which is exactly when you cannot
# stop to type. Default is a working day; 0 means "until ComfyUI restarts".
# Override with C2C_VAULT_TTL_MINUTES (0 = no expiry).
def _ttl_seconds() -> float:
    raw = os.environ.get("C2C_VAULT_TTL_MINUTES", "480")
    try:
        minutes = float(raw)
    except ValueError:
        minutes = 480.0
    return float("inf") if minutes <= 0 else minutes * 60.0


SESSION_TTL_SECONDS = _ttl_seconds()
MAX_FAILED_ATTEMPTS = 5
LOCKOUT_SECONDS = 5 * 60


class _SessionStore:
    """Derived keys, in memory only, per vault id. Never written to disk.

    Rate limiting is per vault id rather than per client: the attacker we care
    about already has the file, so throttling their IP is meaningless - what
    matters is capping guesses against a given blob.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._keys: dict[str, tuple[bytes, float]] = {}
        self._fails: dict[str, tuple[int, float]] = {}

    def put(self, vault_id: str, key: bytes) -> None:
        with self._lock:
            self._keys[vault_id] = (key, time.monotonic() + SESSION_TTL_SECONDS)
            self._fails.pop(vault_id, None)

    def get(self, vault_id: str) -> bytes | None:
        with self._lock:
            entry = self._keys.get(vault_id)
            if not entry:
                return None
            key, expiry = entry
            if time.monotonic() > expiry:
                del self._keys[vault_id]
                return None
            return key

    def drop(self, vault_id: str | None = None) -> None:
        with self._lock:
            if vault_id is None:
                self._keys.clear()
            else:
                self._keys.pop(vault_id, None)

    def note_failure(self, vault_id: str) -> None:
        with self._lock:
            count, _ = self._fails.get(vault_id, (0, 0.0))
            self._fails[vault_id] = (count + 1, time.monotonic() + LOCKOUT_SECONDS)

    def locked_out(self, vault_id: str) -> float:
        """Seconds remaining in lockout, or 0."""
        with self._lock:
            count, until = self._fails.get(vault_id, (0, 0.0))
            if count < MAX_FAILED_ATTEMPTS:
                return 0.0
            remaining = until - time.monotonic()
            if remaining <= 0:
                self._fails.pop(vault_id, None)
                return 0.0
            return remaining


SESSIONS = _SessionStore()


class C2C_VaultLocked:
    """A password-locked subgraph. Will not RUN without the password."""

    CATEGORY = "C2C/Vault"
    FUNCTION = "execute"
    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("output",)
    DESCRIPTION = (
        "Run a password-locked subgraph. The wiring inside travels with the "
        "workflow as AES-GCM ciphertext, so a recipient sees an opaque blob "
        "instead of your node graph.\n\n"
        "SCOPE, honestly: this stops casual inspection and copying. It cannot "
        "stop someone who can run Python in this process - the subgraph must be "
        "decrypted to execute, so the plaintext exists in memory while it runs. "
        "A lock on a door, not a safe."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vault_id": ("STRING", {
                    "default": "", "multiline": False,
                    "tooltip": "Identifies which unlocked session this node may use. "
                               "Stored in clear and authenticated, never secret."}),
                "vault_payload": ("STRING", {
                    "default": "", "multiline": True,
                    "tooltip": "The encrypted subgraph. Base64 AES-GCM ciphertext; "
                               "saved with the workflow. Not editable by hand."}),
            },
            "optional": {
                "input_0": ("*", {"tooltip": "Wired to the vault's first boundary input."}),
                "input_1": ("*", {}),
                "input_2": ("*", {}),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        from ._is_changed_util import hash_args_and_kwargs
        return hash_args_and_kwargs(**kwargs)

    def execute(self, vault_id: str, vault_payload: str, **inputs):
        if not (vault_payload or "").strip():
            raise RuntimeError("C2C Vault: no payload. Lock a selection first.")

        key = SESSIONS.get(vault_id)
        if key is None:
            # Deliberately says nothing about the contents. Before unlock the node
            # must not hint at what is inside, how many nodes, or what they need.
            raise RuntimeError(
                "Vault locked. Open the vault and enter its password to run this "
                "workflow. The unlock then lasts for the rest of the session."
            )

        try:
            subgraph = _unlock_with_key(vault_payload, key, vault_id)
        except VaultError as exc:
            raise RuntimeError(f"C2C Vault: {exc}") from exc

        return self._run(subgraph, inputs, "C2C Vault")

    def _run(self, subgraph, inputs, label):
        """Shared boundary wiring + execution, used by both vault modes."""
        names = [b["name"] for b in subgraph.get("boundary_in", [])]
        supplied = {}
        for i, name in enumerate(names):
            val = inputs.get(f"input_{i}")
            if val is None:
                raise RuntimeError(f"{label}: input {i} ({name!r}) is not connected.")
            supplied[name] = val

        try:
            out = execute_subgraph(subgraph, supplied)
        except VaultExecError as exc:
            # Safe to be specific: we are past the access check.
            raise RuntimeError(f"{label}: {exc}") from exc

        outs = list(subgraph.get("boundary_out", []))
        if not outs:
            raise RuntimeError(f"{label}: the subgraph declares no outputs.")
        return (out[outs[0]["name"]],)


def _unlock_with_key(payload: str, key: bytes, vault_id: str) -> dict[str, Any]:
    """Decrypt using an already-derived key (the store never holds passwords)."""
    import base64

    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    from .vault_crypto import MAGIC, NONCE_LEN, _unpack_header

    blob = base64.b64decode(payload.encode("ascii"), validate=True)
    if not blob.startswith(MAGIC):
        raise VaultError("Not a C2C vault payload.")
    header, hdr_id, _salt, _iters, _bh, hlen = _unpack_header(blob)
    if hdr_id != vault_id:
        raise VaultError("Vault payload does not belong to this vault.")
    nonce = blob[hlen:hlen + NONCE_LEN]
    ct = blob[hlen + NONCE_LEN:]
    try:
        plain = AESGCM(key).decrypt(nonce, ct, header)
    except Exception as exc:
        raise VaultError(
            "Could not open the vault: wrong password, or the payload has been "
            "modified. (These are deliberately reported the same way.)"
        ) from exc
    return json.loads(plain.decode("utf-8"))


# ─────────────────────────── HTTP routes ────────────────────────────
_ROUTES_REGISTERED = False


def register_routes(server) -> None:
    """Idempotent registration of /c2c_vault/* on the PromptServer."""
    global _ROUTES_REGISTERED
    if _ROUTES_REGISTERED:
        return
    try:
        from aiohttp import web
    except Exception as exc:  # pragma: no cover
        log.warning("[C2C Vault] aiohttp missing; vault routes skipped: %s", exc)
        return

    routes = server.routes

    @routes.post("/c2c_vault/unlock")
    async def _unlock(request):
        body = await request.json()
        vault_id = str(body.get("vault_id") or "")
        password = str(body.get("password") or "")
        payload = str(body.get("payload") or "")

        wait = SESSIONS.locked_out(vault_id)
        if wait > 0:
            return web.json_response(
                {"ok": False, "error": f"Too many failed attempts. Try again in "
                                       f"{int(wait // 60) + 1} minute(s)."},
                status=429,
            )
        try:
            unlock_subgraph(payload, password, vault_id=vault_id)
        except VaultError as exc:
            SESSIONS.note_failure(vault_id)
            return web.json_response({"ok": False, "error": str(exc)}, status=403)

        from .vault_crypto import _unpack_header, derive_key
        import base64
        blob = base64.b64decode(payload.encode("ascii"), validate=True)
        _hdr, _vid, salt, iters, _bh, _n = _unpack_header(blob)
        SESSIONS.put(vault_id, derive_key(password, salt, iters))
        return web.json_response({"ok": True, "ttl_seconds": SESSION_TTL_SECONDS})

    @routes.post("/c2c_vault/lock")
    async def _lock(request):
        body = await request.json()
        mode = str(body.get("mode") or "locked").lower()
        fn = seal_subgraph if mode == "sealed" else lock_subgraph
        try:
            payload = fn(
                body.get("subgraph") or {},
                str(body.get("password") or ""),
                vault_id=str(body.get("vault_id") or ""),
            )
        except VaultError as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, "payload": payload})

    @routes.post("/c2c_vault/open")
    async def _open(request):
        """Return the decrypted subgraph for EDITING - unlocked sessions only."""
        body = await request.json()
        vault_id = str(body.get("vault_id") or "")
        payload = str(body.get("payload") or "")
        # A SEALED vault runs without a password but must NOT open without one -
        # otherwise "runs freely" would silently mean "readable by anyone", which
        # is the whole thing sealing is supposed to prevent.
        try:
            sealed = payload_mode(payload) == MODE_SEALED
        except VaultError as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)

        if sealed:
            wait = SESSIONS.locked_out(vault_id)
            if wait > 0:
                return web.json_response(
                    {"ok": False, "error": f"Too many failed attempts. Try again in "
                                           f"{int(wait // 60) + 1} minute(s)."},
                    status=429)
            password = str(body.get("password") or "")
            try:
                sub = unseal_with_password(payload, password, vault_id=vault_id)
            except VaultError as exc:
                SESSIONS.note_failure(vault_id)
                return web.json_response({"ok": False, "error": str(exc)}, status=403)
            return web.json_response({"ok": True, "subgraph": sub})

        key = SESSIONS.get(vault_id)
        if key is None:
            return web.json_response({"ok": False, "error": "Vault locked."}, status=403)
        try:
            return web.json_response({"ok": True, "subgraph": _unlock_with_key(payload, key, vault_id)})
        except VaultError as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=403)

    @routes.post("/c2c_vault/lock_session")
    async def _lock_session(request):
        body = await request.json()
        SESSIONS.drop(str(body.get("vault_id")) if body.get("vault_id") else None)
        return web.json_response({"ok": True})

    _ROUTES_REGISTERED = True
    log.info("[C2C Vault] routes registered")


class C2C_VaultSealed(C2C_VaultLocked):
    """A sealed subgraph: RUNS with no password, opens only with one.

    The difference from C2C_VaultLocked is who needs the secret. Locked = the
    recipient must have your password to run at all. Sealed = they run it
    freely and simply cannot read it.
    """

    CATEGORY = "C2C/Vault"
    DESCRIPTION = (
        "Sealed subgraph - RUNS WITHOUT A PASSWORD, so you can hand a workflow "
        "to someone and they just use it. The password is only needed to OPEN "
        "the vault for editing.\n\n"
        "HONEST SCOPE: because it runs unattended, the unwrapping secret ships "
        "with the pack. Sealing deters copying - it stops someone reading your "
        "graph out of the .json or pasting your nodes into their workflow. It "
        "does NOT stop a determined extractor, and nothing that must run "
        "unattended can. Use C2C_VaultLocked when the recipient should need "
        "your password to run it at all."
    )

    def execute(self, vault_id: str, vault_payload: str, **inputs):
        if not (vault_payload or "").strip():
            raise RuntimeError("C2C Vault (sealed): no payload. Seal a selection first.")
        try:
            subgraph = unseal_for_run(vault_payload, vault_id=vault_id)
        except VaultError as exc:
            raise RuntimeError(f"C2C Vault (sealed): {exc}") from exc
        return self._run(subgraph, inputs, "C2C Vault (sealed)")


NODE_CLASS_MAPPINGS = {
    "C2C_VaultLocked": C2C_VaultLocked,
    "C2C_VaultSealed": C2C_VaultSealed,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "C2C_VaultLocked": "C2C Vault — Locked (password to run)",
    "C2C_VaultSealed": "C2C Vault — Sealed (runs without password)",
}
