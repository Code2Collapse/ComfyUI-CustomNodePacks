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

Positional sockets (input_0..input_9, output_0..output_7) are the wire protocol.
Derived names live in vault_interface and are applied to the node instance by JS.
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import threading
import time
from typing import Any

from .vault_crypto import (
    MODE_SEALED,
    VaultError,
    lock_subgraph,
    payload_mode,
    relock_with_key,
    seal_subgraph,
    unlock_subgraph,
    unseal_for_run,
    unseal_with_password,
)
from . import vault_2fa, vault_keys
from . import vault_crypto_v2 as v2
from .vault_exec import VaultExecError, execute_subgraph
from .vault_promote import (
    MAX_VAULT_PARAMS,
    VaultPromotionError,
    describe,
    resolve_overrides,
)

log = logging.getLogger("c2c.vault")

MAX_VAULT_INPUTS = 10
MAX_VAULT_OUTPUTS = 8

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


def _vault_input_types() -> dict[str, Any]:
    """Build optional input_0..input_9 sockets (positional wire protocol)."""
    optional: dict[str, Any] = {}
    for i in range(MAX_VAULT_INPUTS):
        tip = (
            "Positional boundary input {} — the label you see is applied by the "
            "vault interface manifest, not by this socket name. Index must match "
            "boundary_in[{}] inside the ciphertext."
        ).format(i, i)
        optional[f"input_{i}"] = ("*", {"tooltip": tip})
    # Promoted parameters that were converted to inputs read from these. They
    # are deliberately separate from input_N: input_N is the vault's boundary
    # wiring, param_N drives a widget inside it.
    for i in range(MAX_VAULT_PARAMS):
        optional[f"param_{i}"] = ("*", {
            "tooltip": (
                "Promoted parameter {} — connect this only when a promoted "
                "parameter has been converted to an input. The vault decides "
                "which internal widget it drives; that mapping lives inside "
                "the ciphertext and is never in the workflow JSON."
            ).format(i)})
    return optional


class C2C_VaultLocked:
    """A password-locked subgraph. Will not RUN without the password."""

    CATEGORY = "C2C/Vault"
    FUNCTION = "execute"
    # A18 (owner 2026-10-25): new vaults are protected native subgraphs (right-click Lock / Seal as C2C Vault,
    # nodes/secure_subgraph.py). This node stays registered so old workflows load; the front end upgrades it to the
    # new vault on load (same payload). Hidden from the node search, so nobody builds a new custom-executor vault.
    DEPRECATED = True
    RETURN_TYPES = tuple(["*"] * MAX_VAULT_OUTPUTS)
    RETURN_NAMES = tuple(f"output_{i}" for i in range(MAX_VAULT_OUTPUTS))
    DESCRIPTION = (
        "Run a password-locked subgraph. The wiring inside travels with the "
        "workflow as AES-GCM ciphertext, so a recipient sees an opaque blob "
        "instead of your node graph.\n\n"
        "Lock a canvas selection via right-click → C2C Vault. To run it, type the "
        "password in the field on this node and press Enter; the unlock lasts for "
        "the ComfyUI session and the password never enters the workflow JSON.\n\n"
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
                    "tooltip": "Public vault handle — authenticated in the "
                               "ciphertext header, never secret. Lets the UI "
                               "match an unlock session to the right blob "
                               "without revealing what is inside."}),
                "vault_payload": ("STRING", {
                    "default": "", "multiline": True,
                    "tooltip": "AES-GCM ciphertext of the internal graph. This "
                               "is what travels in the .json — opaque to anyone "
                               "without the password. Do not hand-edit."}),
                "vault_interface": ("STRING", {
                    "default": "{}", "multiline": True,
                    "tooltip": "Clear-text manifest: socket names, types, node "
                               "count, and any promoted parameters. This is the "
                               "public API of the vault and it travels in the "
                               ".json, so it carries a promoted parameter's "
                               "LABEL and value only - which internal widget it "
                               "drives is inside the ciphertext, because naming "
                               "it here would describe the contents."}),
            },
            "optional": _vault_input_types(),
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    # A Locked vault needs an open session to run; a Sealed one does not.
    NEEDS_SESSION = True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        from ._is_changed_util import hash_args_and_kwargs
        digest = hash_args_and_kwargs(**kwargs)
        if not cls.NEEDS_SESSION:
            return digest
        # The session state is part of the cache key: after "Lock session" the next queue checks the password
        # again instead of serving the cached result (which made Lock session look like it did nothing).
        state = "open" if SESSIONS.get(str(kwargs.get("vault_id") or "")) is not None else "locked"
        return f"{digest}:{state}"

    def execute(self, vault_id: str, vault_payload: str, vault_interface: str = "{}", unique_id=None, **inputs):
        if not (vault_payload or "").strip():
            raise RuntimeError("C2C Vault: no payload. Lock a selection first.")

        key = SESSIONS.get(vault_id)
        if key is None:
            # Deliberately says nothing about the contents. Before unlock the node
            # must not hint at what is inside, how many nodes, or what they need.
            raise RuntimeError(
                "Vault locked. Type the password in the vault node's password field and press Enter "
                "(or click Unlock), then queue again. The unlock lasts for this ComfyUI session."
            )

        interface = _parse_interface(vault_interface)
        try:
            if v2.is_v2(vault_payload):
                plain = v2.open_with_kek(vault_payload, key, vault_id=vault_id)
                v2.check_interface(v2.parse(vault_payload), interface)
                subgraph = plain["exec"]
            else:
                subgraph = _unlock_with_key(vault_payload, key, vault_id)
        except VaultError as exc:
            raise RuntimeError(f"C2C Vault: {exc}") from exc

        return self._run(subgraph, inputs, "C2C Vault", interface, unique_id)

    def _run(self, subgraph, inputs, label, interface=None, unique_id=None):
        """Shared boundary wiring + execution, used by both vault modes."""
        # Promoted parameters first: a mistake here should be reported before
        # anything runs, not halfway through the graph.
        socket_values = {
            i: inputs.get(f"param_{i}") for i in range(MAX_VAULT_PARAMS)
            if inputs.get(f"param_{i}") is not None
        }
        try:
            overrides = resolve_overrides(subgraph, interface or {}, socket_values)
        except VaultPromotionError as exc:
            raise RuntimeError(f"{label}: {exc}") from exc

        names = [b["name"] for b in subgraph.get("boundary_in", [])]
        supplied = {}
        for i, name in enumerate(names):
            val = inputs.get(f"input_{i}")
            if val is None:
                raise RuntimeError(f"{label}: input {i} ({name!r}) is not connected.")
            supplied[name] = val

        try:
            out = execute_subgraph(subgraph, supplied, overrides, hidden={"UNIQUE_ID": unique_id})
        except VaultExecError as exc:
            # Safe to be specific: we are past the access check.
            raise RuntimeError(f"{label}: {exc}") from exc

        outs = list(subgraph.get("boundary_out", []))
        if not outs:
            raise RuntimeError(f"{label}: the subgraph declares no outputs.")
        if len(outs) > MAX_VAULT_OUTPUTS:
            raise RuntimeError(
                f"{label}: the subgraph declares {len(outs)} outputs but the "
                f"vault node supports at most {MAX_VAULT_OUTPUTS}."
            )
        # Pad to MAX_VAULT_OUTPUTS — RETURN_TYPES is fixed-length; trailing
        # slots are hidden by JS and are never wired downstream.
        result: list[Any] = [None] * MAX_VAULT_OUTPUTS
        for i, spec in enumerate(outs):
            result[i] = out[spec["name"]]
        return tuple(result)


def _notify(unique_id, text: str) -> None:
    log.warning("[C2C Vault] %s", text)
    if unique_id is None:
        return
    try:
        from server import PromptServer

        PromptServer.instance.send_progress_text(text, unique_id)
    except Exception:  # noqa: BLE001 - outside ComfyUI
        pass


def _parse_interface(raw: str) -> dict[str, Any]:
    """The manifest is clear text and user-editable, so treat it as untrusted."""
    if not raw:
        return {}
    try:
        val = json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"C2C Vault: the interface manifest is not valid JSON ({exc}). "
            "Re-lock the vault to rebuild it."
        ) from exc
    if not isinstance(val, dict):
        raise RuntimeError("C2C Vault: the interface manifest must be an object.")
    return val


def _unlock_with_key(payload: str, key: bytes, vault_id: str) -> dict[str, Any]:
    """Decrypt using an already-derived key (the store never holds passwords)."""
    import base64

    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    from .vault_crypto import MAGIC, NONCE_LEN, _unpack_header

    blob = base64.b64decode(payload.encode("ascii"), validate=True)
    if not blob.startswith(MAGIC):
        raise VaultError("Not a C2C vault payload.")
    # _unpack_header returns (header, vault_id, salt, iterations, boundary_hash, header_len, mode): unpacking six
    # values raised ValueError since the mode byte was added, so no Locked vault could run (A9, L2.24).
    hdr = _unpack_header(blob)
    header, hdr_id, hlen = hdr[0], hdr[1], hdr[5]
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



def _open_session(vault_id: str, payload: str, password: str) -> None:
    """Hold the key derived from `password` for this vault (the store never holds the password itself)."""
    import base64

    from .vault_crypto import _unpack_header, derive_key

    if v2.is_v2(payload):
        _plain, kek, _p = v2.open_with_password(payload, password, vault_id=vault_id)
        SESSIONS.put(vault_id, kek)
        return
    blob = base64.b64decode(payload.encode("ascii"), validate=True)
    hdr = _unpack_header(blob)   # 7 fields; Unlock failed with ValueError while it unpacked 6 (L2.24)
    SESSIONS.put(vault_id, derive_key(password, hdr[2], hdr[3]))

def vault_store_name() -> str:
    from . import vault_store

    try:
        return vault_store.backend_name()
    except Exception:  # noqa: BLE001
        return "none"


_MODE_NAMES = {"locked": v2.MODE_LOCKED, "sealed": v2.MODE_SEALED_INSTALLS,
               "sealed_installs": v2.MODE_SEALED_INSTALLS, "sealed_portable": v2.MODE_SEALED_PORTABLE}


def _recipients(extra) -> list:
    """This install's own key first, then any recipient install keys the author pasted (deduplicated)."""
    keys = [vault_keys.install_public_key()]
    for text in extra or []:
        raw = vault_keys.parse_public_text(str(text))
        if raw not in keys:
            keys.append(raw)
    return keys


def lock_v2(body: dict) -> tuple:
    """POST /c2c_vault/lock -> (payload, kek, mode). The plaintext is {exec, layout}; the interface is authenticated."""
    mode_name = str(body.get("mode") or "locked").lower()
    if mode_name not in _MODE_NAMES:
        raise VaultError(f"Unknown vault mode {mode_name!r}.")
    mode = _MODE_NAMES[mode_name]
    ex = body.get("subgraph") or {}
    validate_lockable(ex)
    plain = {"exec": ex, "layout": body.get("layout")}
    recipients = _recipients(body.get("recipients")) if mode == v2.MODE_SEALED_INSTALLS else None
    payload, kek = v2.lock(plain, str(body.get("password") or ""), vault_id=str(body.get("vault_id") or ""),
                           mode=mode, iface=body.get("interface") or {}, recipients=recipients)
    return payload, kek, mode


def validate_lockable(ex: dict) -> None:
    """Refuse, at lock time, inner nodes the vault executor cannot run faithfully (A17 s2.9)."""
    from .vault_exec import _node_registry

    registry = _node_registry()
    problems = []
    for n in ex.get("nodes") or []:
        cls = registry.get(str(n.get("class_type")))
        if cls is None:
            continue                                       # reported by the executor with the install hint
        name = str(n.get("class_type"))
        if getattr(cls, "INPUT_IS_LIST", False) or any(getattr(cls, "OUTPUT_IS_LIST", ()) or ()):
            problems.append(f"{name}: list inputs / outputs")
        try:
            it = cls.INPUT_TYPES() or {}
        except Exception:  # noqa: BLE001
            it = {}
        for section in ("required", "optional"):
            for key, spec in (it.get(section) or {}).items():
                opts = spec[1] if isinstance(spec, (tuple, list)) and len(spec) > 1 and isinstance(spec[1], dict) else {}
                if opts.get("lazy"):
                    problems.append(f"{name}: lazy input {key!r}")
        hidden = set((it.get("hidden") or {}).values())
        unsupported = hidden - {"UNIQUE_ID", "PROMPT", "EXTRA_PNGINFO", "DYNPROMPT"}
        if unsupported:
            problems.append(f"{name}: hidden inputs {sorted(unsupported)}")
    if problems:
        raise VaultError("These nodes cannot run inside a vault (the vault runs its nodes itself, without ComfyUI's "
                         "scheduler): " + "; ".join(sorted(set(problems))) + ". Keep them outside the selection.")


class _EditSessions:
    """Password checked -> pending (2 min, plaintext kept server-side) -> code checked -> active (30 min idle)."""

    PENDING_SECONDS = 120
    IDLE_SECONDS = 30 * 60

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pending: dict[str, dict] = {}
        self._active: dict[str, dict] = {}

    def _sweep(self) -> None:
        now = time.monotonic()
        for store, ttl in ((self._pending, self.PENDING_SECONDS), (self._active, self.IDLE_SECONDS)):
            for k in [k for k, v in store.items() if now - v["at"] > ttl]:
                store.pop(k, None)

    def begin(self, vault_id: str, payload: str, password: str) -> str:
        if v2.is_v2(payload):
            plain, kek, p = v2.open_with_password(payload, password, vault_id=vault_id)
            session = {"vault_id": vault_id, "payload": payload, "plain": plain, "mode": p["mode"],
                       "k": {"kek": kek, "salt": p["salt"], "kdf": p["kdf"]}, "legacy": False}
        else:                                              # v1: upgraded to v2 on commit
            sealed = payload_mode(payload) == MODE_SEALED
            sub = (unseal_with_password if sealed else unlock_subgraph)(payload, password, vault_id=vault_id)
            session = {"vault_id": vault_id, "payload": payload, "plain": {"exec": sub, "layout": None},
                       "mode": v2.MODE_SEALED_INSTALLS if sealed else v2.MODE_LOCKED, "k": v2.new_kek(password),
                       "legacy": True}
        edit_id = secrets.token_urlsafe(24)
        with self._lock:
            self._sweep()
            session["at"] = time.monotonic()
            self._pending[edit_id] = session
        return edit_id

    def verify(self, edit_id: str, code: str) -> dict:
        with self._lock:
            self._sweep()
            session = self._pending.get(edit_id)
        if session is None:
            raise PermissionError("The password step expired. Open the vault again.")
        if not vault_2fa.verify(code):
            raise PermissionError("That code is not right (or was already used).")
        token = secrets.token_urlsafe(32)
        with self._lock:
            self._pending.pop(edit_id, None)
            session["at"] = time.monotonic()
            self._active[token] = session
        plain = session["plain"]
        recips = v2.recipients_of(plain) if session["mode"] == v2.MODE_SEALED_INSTALLS else []
        return {"edit_token": token, "exec": plain["exec"], "layout": plain.get("layout"), "mode": session["mode"],
                "legacy": session["legacy"], "recipients": [vault_keys.public_text(k) for k in recips]}

    def commit(self, token: str, ex: dict, layout, iface: dict, recipients, new_password=None) -> str:
        with self._lock:
            self._sweep()
            session = self._active.get(token)
        if session is None:
            raise PermissionError("The edit session expired. Open the vault again (your last locked version is "
                                  "unchanged).")
        validate_lockable(ex)
        plain = {"exec": ex, "layout": layout}
        mode = session["mode"]
        recips = None
        if mode == v2.MODE_SEALED_INSTALLS:
            recips = _recipients(recipients if recipients is not None else
                                 [vault_keys.public_text(k) for k in v2.recipients_of(session["plain"])])
        # a password change is a re-lock under a new KEK (fresh salt); otherwise the session's KEK is kept
        k = v2.new_kek(new_password) if new_password else session["k"]
        payload = v2.lock_with(plain, k, vault_id=session["vault_id"], mode=mode, iface=iface, recipients=recips)
        check = v2.open_with_kek(payload, k["kek"], vault_id=session["vault_id"])
        if check["exec"] != json.loads(json.dumps(ex)):
            raise VaultError("The re-encrypted vault did not read back identically; nothing was replaced.")
        with self._lock:
            self._active.pop(token, None)
        if mode == v2.MODE_LOCKED:
            SESSIONS.put(session["vault_id"], k["kek"])                # the editor may run it right away
        return payload

    def cancel(self, token: str) -> None:
        with self._lock:
            self._active.pop(token, None)


EDITS = _EditSessions()

_WORKFLOW_NOTE = (
    "C2C Vault encrypted workflow: open it in ComfyUI with "
    "ComfyUI-CustomNodePacks and the password."
)


def _workflow_enabled() -> bool:
    return os.environ.get("C2C_VAULT_WORKFLOW", "1").strip().lower() not in ("0", "false", "no")


def _workflow_subgraph(workflow: dict[str, Any]) -> dict[str, Any]:
    return {
        "nodes": [],
        "links": [],
        "boundary_in": [],
        "boundary_out": [],
        "workflow": workflow,
    }


def _unwrap_workflow(sub: dict[str, Any]) -> dict[str, Any]:
    wf = sub.get("workflow")
    if not isinstance(wf, dict):
        raise VaultError("Vault opened but its contents are not a workflow object.")
    return wf


def _workflow_file(vault_id: str, payload: str) -> dict[str, Any]:
    return {
        "c2c_encrypted_workflow": 1,
        "vault_id": vault_id,
        "mode": "locked",
        "payload": payload,
        "note": _WORKFLOW_NOTE,
    }


def _new_workflow_vault_id() -> str:
    return "wf-" + secrets.token_hex(4)


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
            if v2.is_v2(payload):
                _open_session(vault_id, payload, password)        # derives, decrypts, stores the KEK
            else:
                unlock_subgraph(payload, password, vault_id=vault_id)
                _open_session(vault_id, payload, password)
        except VaultError as exc:
            SESSIONS.note_failure(vault_id)
            return web.json_response({"ok": False, "error": str(exc)}, status=403)
        return web.json_response({"ok": True, "ttl_seconds": SESSION_TTL_SECONDS})

    @routes.post("/c2c_vault/lock")
    async def _lock(request):
        body = await request.json()
        try:
            payload, kek, mode = lock_v2(body)
        except VaultError as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        # The creator has just typed the password: their session opens as Unlock would (A9). TTL / Lock session apply.
        session_open = False
        if mode == v2.MODE_LOCKED:
            SESSIONS.put(str(body.get("vault_id") or ""), kek)
            session_open = True
        return web.json_response({"ok": True, "payload": payload, "session_open": session_open, "format": 2})

    @routes.post("/c2c_vault/info")
    async def _info(request):
        """What a payload is, readable without any key: format, mode, and whether THIS install may run it."""
        body = await request.json()
        payload = str(body.get("payload") or "")
        try:
            if not v2.is_v2(payload):
                return web.json_response({"ok": True, "format": 1, "mode": "sealed" if payload_mode(payload) ==
                                          MODE_SEALED else "locked", "legacy": True})
            i = v2.info(payload)
            mine = vault_keys.install_private_key(create=False)
            here = mine is not None and v2.key_id(vault_keys._raw_public(mine)).hex() in i["installs"]
            names = {v2.MODE_LOCKED: "locked", v2.MODE_SEALED_INSTALLS: "sealed", v2.MODE_SEALED_PORTABLE: "portable"}
            return web.json_response({"ok": True, "format": 2, "mode": names[i["mode"]], "legacy": False,
                                      "installs": len(i["installs"]), "runs_here": here})
        except VaultError as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)

    @routes.get("/c2c_vault/keys/public")
    async def _public_key(request):
        pub = vault_keys.install_public_key()
        return web.json_response({"ok": True, "public_key": vault_keys.public_text(pub),
                                  "key_id": v2.key_id(pub).hex(), "store": vault_store_name()})

    @routes.get("/c2c_vault/2fa/status")
    async def _tfa_status(request):
        return web.json_response({"ok": True, "enrolled": vault_2fa.enrolled(),
                                  "locked_out_seconds": int(vault_2fa.locked_out()),
                                  "recovery_codes_left": vault_2fa.recovery_codes_left()})

    @routes.post("/c2c_vault/2fa/enrol/begin")
    async def _tfa_begin(request):
        try:
            return web.json_response({"ok": True, **vault_2fa.begin_enrolment()})
        except PermissionError as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=409)

    @routes.post("/c2c_vault/2fa/enrol/confirm")
    async def _tfa_confirm(request):
        body = await request.json()
        try:
            ok = vault_2fa.confirm_enrolment(str(body.get("code") or ""))
        except PermissionError as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=409)
        return web.json_response({"ok": ok, "error": None if ok else (vault_2fa.last_enrolment_error() or
                                                                      "That code is not right.")},
                                 status=200 if ok else 403)

    @routes.post("/c2c_vault/2fa/reset")
    async def _tfa_reset(request):
        body = await request.json()
        try:
            ok = vault_2fa.reset(str(body.get("code") or ""))
        except PermissionError as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=403)
        return web.json_response({"ok": ok}, status=200 if ok else 403)

    @routes.post("/c2c_vault/edit/begin")
    async def _edit_begin(request):
        body = await request.json()
        vault_id = str(body.get("vault_id") or "")
        wait = SESSIONS.locked_out(vault_id)
        if wait > 0:
            return web.json_response({"ok": False, "error": f"Too many failed attempts. Try again in "
                                                            f"{int(wait // 60) + 1} minute(s)."}, status=429)
        if not vault_2fa.enrolled():
            return web.json_response({"ok": False, "needs_enrolment": True,
                                      "error": "Opening a vault for editing needs a second factor. Set up an "
                                               "authenticator app first (C2C Vault menu)."}, status=403)
        try:
            edit_id = EDITS.begin(vault_id, str(body.get("payload") or ""), str(body.get("password") or ""))
        except VaultError as exc:
            SESSIONS.note_failure(vault_id)
            return web.json_response({"ok": False, "error": str(exc)}, status=403)
        return web.json_response({"ok": True, "edit_id": edit_id, "second_factor": "totp"})

    @routes.post("/c2c_vault/edit/verify")
    async def _edit_verify(request):
        body = await request.json()
        try:
            result = EDITS.verify(str(body.get("edit_id") or ""), str(body.get("code") or ""))
        except PermissionError as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=403)
        return web.json_response({"ok": True, **result})

    @routes.post("/c2c_vault/edit/commit")
    async def _edit_commit(request):
        body = await request.json()
        try:
            payload = EDITS.commit(str(body.get("edit_token") or ""), body.get("exec") or {}, body.get("layout"),
                                   body.get("interface") or {}, body.get("recipients"),
                                   new_password=body.get("new_password") or None)
        except (VaultError, PermissionError) as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, "payload": payload})

    @routes.post("/c2c_vault/edit/cancel")
    async def _edit_cancel(request):
        body = await request.json()
        EDITS.cancel(str(body.get("edit_token") or ""))
        return web.json_response({"ok": True})

    @routes.post("/c2c_vault/open")
    async def _open(request):
        """Superseded (A17): opening for editing needs the password AND the second factor (/c2c_vault/edit/*). A run
        session is never enough - it used to be, which made "can run" the same as "can read"."""
        return web.json_response({"ok": False, "error": "Opening a vault for editing needs your password and your "
                                  "second factor: use C2C Vault -> Open for editing."}, status=410)

    async def _open_superseded(request):
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

    @routes.post("/c2c_vault/status")
    async def _status(request):
        """Is this vault's session open? Lets the node show the real state after a restart or an expiry (A9)."""
        body = await request.json()
        return web.json_response({"ok": True, "open": SESSIONS.get(str(body.get("vault_id") or "")) is not None})

    @routes.post("/c2c_vault/lock_session")
    async def _lock_session(request):
        body = await request.json()
        SESSIONS.drop(str(body.get("vault_id")) if body.get("vault_id") else None)
        return web.json_response({"ok": True})

    @routes.post("/c2c_vault/workflow/lock")
    async def _workflow_lock(request):
        if not _workflow_enabled():
            return web.json_response({"error": "disabled"}, status=404)
        body = await request.json()
        password = str(body.get("password") or "")
        workflow = body.get("workflow")
        if not isinstance(workflow, dict):
            return web.json_response({"ok": False, "error": "workflow must be an object."}, status=400)
        vault_id = _new_workflow_vault_id()
        try:
            payload = lock_subgraph(
                _workflow_subgraph(workflow),
                password,
                vault_id=vault_id,
            )
        except VaultError as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        try:
            _open_session(vault_id, payload, password)
        except Exception as exc:
            log.warning("[C2C Vault] could not open workflow session after lock: %s", exc)
        return web.json_response({"ok": True, "file": _workflow_file(vault_id, payload)})

    @routes.post("/c2c_vault/workflow/unlock")
    async def _workflow_unlock(request):
        if not _workflow_enabled():
            return web.json_response({"error": "disabled"}, status=404)
        body = await request.json()
        file = body.get("file") or {}
        vault_id = str(file.get("vault_id") or "")
        payload = str(file.get("payload") or "")
        password = str(body.get("password") or "")

        wait = SESSIONS.locked_out(vault_id)
        if wait > 0:
            return web.json_response(
                {"ok": False, "error": f"Too many failed attempts. Try again in "
                                       f"{int(wait // 60) + 1} minute(s)."},
                status=429,
            )
        try:
            sub = unlock_subgraph(payload, password, vault_id=vault_id)
            workflow = _unwrap_workflow(sub)
        except VaultError as exc:
            SESSIONS.note_failure(vault_id)
            return web.json_response({"ok": False, "error": str(exc)}, status=403)

        _open_session(vault_id, payload, password)
        return web.json_response({"ok": True, "workflow": workflow})

    @routes.post("/c2c_vault/workflow/relock")
    async def _workflow_relock(request):
        if not _workflow_enabled():
            return web.json_response({"error": "disabled"}, status=404)
        body = await request.json()
        vault_id = str(body.get("vault_id") or "")
        payload = str(body.get("payload") or "")
        workflow = body.get("workflow")
        if not isinstance(workflow, dict):
            return web.json_response({"ok": False, "error": "workflow must be an object."}, status=400)

        key = SESSIONS.get(vault_id)
        if key is None:
            return web.json_response({"error": "session expired"}, status=403)
        try:
            new_payload = relock_with_key(
                payload,
                key,
                _workflow_subgraph(workflow),
                vault_id=vault_id,
            )
        except VaultError as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, "file": _workflow_file(vault_id, new_payload)})

    _ROUTES_REGISTERED = True
    log.info("[C2C Vault] routes registered")


class C2C_VaultSealed(C2C_VaultLocked):
    """A sealed subgraph: RUNS with no password, opens only with one.

    The difference from C2C_VaultLocked is who needs the secret. Locked = the
    recipient must have your password to run at all. Sealed = they run it
    freely and simply cannot read it.
    """

    NEEDS_SESSION = False

    CATEGORY = "C2C/Vault"
    DESCRIPTION = (
        "Sealed subgraph - RUNS WITHOUT A PASSWORD, so you can hand a workflow "
        "to someone and they just use it. The password is only needed to OPEN "
        "the vault for editing.\n\n"
        "Sealed for installs (the default): it runs without a password only on "
        "this install and on the installs whose public keys you pasted when you "
        "sealed it; each install keeps its own private key in the OS credential "
        "store. Portable seal: runs on any install with this pack, so it is "
        "obfuscation only. A legacy (v1) seal still runs and is marked; open it "
        "and lock it again to upgrade.\n\n"
        "HONEST SCOPE: a recipient install can always decrypt what it runs, so "
        "sealing stops reading and copying your graph out of the .json, not a "
        "determined extractor on a recipient machine. Use C2C_VaultLocked when "
        "the recipient should need your password to run it at all."
    )

    def execute(self, vault_id: str, vault_payload: str, vault_interface: str = "{}", unique_id=None, **inputs):
        if not (vault_payload or "").strip():
            raise RuntimeError("C2C Vault (sealed): no payload. Seal a selection first.")
        interface = _parse_interface(vault_interface)
        try:
            if v2.is_v2(vault_payload):
                mode = v2.parse(vault_payload)["mode"]
                if mode == v2.MODE_SEALED_INSTALLS:
                    private = vault_keys.install_private_key(create=False)
                    if private is None:
                        raise VaultError("This install has no vault key yet, so a vault sealed for installs cannot "
                                         "run here. Ask the author to seal it for this install's public key (C2C "
                                         "Vault menu), or open it with the password.")
                    plain = v2.open_with_install_key(vault_payload, private, vault_id=vault_id)
                elif mode == v2.MODE_SEALED_PORTABLE:
                    plain = v2.open_portable(vault_payload, vault_id=vault_id)
                else:
                    raise VaultError("This is a Locked vault's payload: it needs a C2C Vault Locked node and the "
                                     "password.")
                v2.check_interface(v2.parse(vault_payload), interface)
                subgraph = plain["exec"]
            else:
                subgraph = unseal_for_run(vault_payload, vault_id=vault_id)
                _notify(unique_id, "legacy seal - obfuscation only: open it with the password + your code and lock "
                                   "it to upgrade")
        except VaultError as exc:
            raise RuntimeError(f"C2C Vault (sealed): {exc}") from exc
        return self._run(subgraph, inputs, "C2C Vault (sealed)", interface, unique_id)


class C2C_VaultEditInput:
    """A vault input, as shown in the vault's edit view (A17). Only exists inside that view, which never runs."""

    CATEGORY = "C2C/Vault"
    FUNCTION = "execute"
    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("value",)
    DEPRECATED = True              # hidden from the node search: created by the vault editor only
    DESCRIPTION = "A vault input in the vault's edit view. Created by the vault editor; it never runs."

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"index": ("INT", {"default": 0, "min": 0, "max": MAX_VAULT_INPUTS - 1})}}

    def execute(self, index):
        raise RuntimeError("This is a vault editor placeholder: lock the vault to use it.")


class C2C_VaultEditOutput:
    """A vault output, as shown in the vault's edit view (A17). Only exists inside that view, which never runs."""

    CATEGORY = "C2C/Vault"
    FUNCTION = "execute"
    RETURN_TYPES = ()
    OUTPUT_NODE = False
    DEPRECATED = True
    DESCRIPTION = "A vault output in the vault's edit view. Created by the vault editor; it never runs."

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"value": ("*", {}),
                             "index": ("INT", {"default": 0, "min": 0, "max": MAX_VAULT_OUTPUTS - 1})}}

    def execute(self, value, index):
        raise RuntimeError("This is a vault editor placeholder: lock the vault to use it.")


NODE_CLASS_MAPPINGS = {
    "C2C_VaultLocked": C2C_VaultLocked,
    "C2C_VaultSealed": C2C_VaultSealed,
    "C2C_VaultEditInput": C2C_VaultEditInput,
    "C2C_VaultEditOutput": C2C_VaultEditOutput,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "C2C_VaultLocked": "C2C Vault — Locked (password to run)",
    "C2C_VaultSealed": "C2C Vault — Sealed (runs without password)",
    "C2C_VaultEditInput": "Vault input (edit view)",
    "C2C_VaultEditOutput": "Vault output (edit view)",
}
