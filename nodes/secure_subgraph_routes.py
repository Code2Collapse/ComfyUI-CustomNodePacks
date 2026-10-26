"""Routes of the protected-subgraph vault (A18, docs/VAULT_DESIGN_A18.md).

  POST /c2c_vault/sg/protect     selection / subgraph -> encrypted vault (password, profile, recipients, editors)
  POST /c2c_vault/sg/unlock_run  Locked profile: password once per browser tab (key kept in memory for that tab)
  POST /c2c_vault/sg/forget_run  the tab closed: drop its run keys
  POST /c2c_vault/sg/open        password + second factor + editor check -> the inner graph for native editing
  POST /c2c_vault/sg/commit      leaving the subgraph: re-encrypt the edited graph (auto-lock)
  POST /c2c_vault/sg/cancel      leave without changes
  GET  /c2c_vault/sg/install_code  this installation's code (public key) to send to a vault author
  POST /c2c_vault/sg/info        public facts of a payload (profile, legacy, node count) - never the inside

The password travels only over these routes; it is never a widget, never in a prompt or a workflow, never stored. The
server keeps derived keys in memory only.
"""
from __future__ import annotations

import base64
import hashlib
import secrets
import threading
import time
from typing import Any, Optional

from .vault_crypto import VaultError

PROFILES = {"locked": 0, "sealed_installs": 1, "sealed_portable": 2}
PROFILE_NAMES = {v: k for k, v in PROFILES.items()}
EDIT_IDLE_SECONDS = 6 * 3600           # an edit session also ends when the user leaves the subgraph (owner A18)
_lock = threading.Lock()
_EDITS: dict[str, dict] = {}


def _hash(payload: str) -> str:
    return hashlib.sha256((payload or "").encode("ascii", "ignore")).hexdigest()


def _this_install_code() -> str:
    from . import vault_keys
    return vault_keys.public_text(vault_keys.install_public_key())


def _codes_to_keys(codes) -> list:
    from . import vault_keys
    out = []
    for c in codes or []:
        raw = vault_keys.parse_public_text(str(c))
        if raw not in out:
            out.append(raw)
    return out


def boundary_exec(interface: dict) -> dict:
    """The v2 envelope wants the old flattened block; a new vault carries an EMPTY one (no nodes), so the old
    executor can never run it, plus the boundary names / types it authenticates."""
    return {"nodes": [], "links": [],
            "boundary_in": [{"name": str(s.get("name")), "type": str(s.get("type", "*"))}
                            for s in interface.get("in") or []],
            "boundary_out": [{"name": str(s.get("name")), "type": str(s.get("type", "*"))}
                             for s in interface.get("out") or []]}


def _validate_api(api: dict, interface: dict) -> None:
    if not isinstance(api, dict) or not isinstance(api.get("nodes"), dict) or not api["nodes"]:
        raise VaultError("Nothing to protect: the selection has no nodes that run.")
    n_in = len(interface.get("in") or [])
    for nid, node in api["nodes"].items():
        if not isinstance(node, dict) or not node.get("class_type"):
            raise VaultError(f"Malformed inner node {nid!r}.")
        for name, v in (node.get("inputs") or {}).items():
            if isinstance(v, dict) and "vault_in" in v and not (0 <= int(v["vault_in"]) < n_in):
                raise VaultError(f"Inner input {name!r} reads vault input {v['vault_in']}, which does not exist.")
            if isinstance(v, (list, tuple)) and len(v) == 2 and isinstance(v[0], str) and v[0] not in api["nodes"]:
                raise VaultError(f"Inner link to unknown node {v[0]!r}.")
    for out in api.get("outputs") or []:
        if "node" in out and str(out["node"]) not in api["nodes"]:
            raise VaultError(f"Vault output {out.get('name')!r} reads an unknown node.")


def protect(body: dict) -> dict:
    from . import secure_subgraph as SS
    from . import vault_crypto_v2 as v2

    profile = str(body.get("profile") or "sealed_installs")
    if profile not in PROFILES:
        raise VaultError(f"Unknown security profile {profile!r}.")
    password = str(body.get("password") or "")
    if len(password) < 8:
        raise VaultError("Choose a vault password of at least 8 characters (it is needed to open the vault).")
    interface = body.get("interface") or {}
    api = body.get("api") or {}
    _validate_api(api, interface)
    me = _this_install_code()
    editors = [me] + [c for c in (body.get("editors") or []) if str(c) != me]
    _codes_to_keys(editors)                                      # validates every code
    recipients = None
    if profile == "sealed_installs":
        recipients = _codes_to_keys([me] + list(body.get("recipients") or []))       # this install first
    vault_id = "v" + secrets.token_hex(8)
    plain = {"exec": boundary_exec(interface), "api": api, "ui": body.get("ui"), "interface": interface,
             "editors": editors, "profile": profile, "packs": body.get("packs") or [], "created": int(time.time())}
    payload, kek = v2.lock(plain, password, vault_id=vault_id, mode=PROFILES[profile], iface=_iface(interface),
                           recipients=recipients)
    if profile == "locked" and body.get("client_id"):
        SS.RUN_SESSIONS.put(str(body["client_id"]), vault_id, kek)   # the creator can run it right away
    return {"vault_id": vault_id, "payload": payload, "profile": profile,
            "outputs_node": bool(body.get("outputs_node")), "node_count": len(api["nodes"])}


def _iface(interface: dict) -> dict:
    """The authenticated part (v2 interface_struct): socket names / types and promoted parameter ids / names."""
    return {"in": interface.get("in") or [], "out": interface.get("out") or [],
            "params": [{"id": p.get("key"), "name": p.get("name"), "type": p.get("type", "*")}
                       for p in interface.get("params") or []]}


def unlock_run(body: dict) -> dict:
    from . import secure_subgraph as SS
    from . import vault_crypto, vault_crypto_v2 as v2

    payload, vault_id = str(body.get("payload") or ""), str(body.get("vault_id") or "")
    client_id = str(body.get("client_id") or "")
    if not client_id:
        raise VaultError("No browser tab id - reload the page.")
    if v2.is_v2(payload):
        _plain, kek, _p = v2.open_with_password(payload, str(body.get("password") or ""), vault_id=vault_id)
    else:
        from .vault_crypto import _unpack_header, derive_key
        blob = base64.b64decode(payload.encode("ascii"), validate=True)
        hdr = _unpack_header(blob)
        kek = derive_key(str(body.get("password") or ""), hdr[2], hdr[3])
        from .vault_node import _unlock_with_key
        _unlock_with_key(payload, kek, vault_id)              # proves the password before keeping the key
    SS.RUN_SESSIONS.put(client_id, vault_id, kek)
    return {"ok": True}


def forget_run(body: dict) -> dict:
    from . import secure_subgraph as SS
    SS.RUN_SESSIONS.drop_client(str(body.get("client_id") or ""))
    return {"ok": True}


def open_for_edit(body: dict) -> dict:
    """Password -> second factor -> editor list. Returns the inner graph for native editing + an edit token."""
    from . import secure_subgraph as SS
    from . import vault_2fa, vault_crypto_v2 as v2

    payload, vault_id = str(body.get("payload") or ""), str(body.get("vault_id") or "")
    password = str(body.get("password") or "")
    if not vault_2fa.enrolled():
        return {"ok": False, "needs_enrolment": True,
                "error": "Opening a vault needs a second factor on this installation. Set up an authenticator app "
                         "first (C2C menu > Vault > Two-factor setup)."}
    legacy = not v2.is_v2(payload)
    if not legacy:
        plain, kek, parsed = v2.open_with_password(payload, password, vault_id=vault_id)
        k = {"kek": kek, "salt": parsed["salt"], "kdf": parsed["kdf"]}
    else:
        from .vault_crypto import unlock_subgraph
        plain = {"exec": unlock_subgraph(payload, password, vault_id=vault_id)}
        k = v2.new_kek(password)                               # the upgrade re-encrypts in the current envelope
    detail = vault_2fa.verify_detail(str(body.get("code") or ""))
    if not detail["ok"]:
        return {"ok": False, "reason": detail["reason"], "error": detail["message"],
                **({"next_in": detail["next_in"]} if "next_in" in detail else {})}
    editors = plain.get("editors")
    me = _this_install_code()
    if editors and me not in editors:
        return {"ok": False, "reason": "not_editor",
                "error": "This installation is not an editor of this vault. Ask its author to add your installation "
                         "code (C2C menu > Vault > My installation code)."}
    token = secrets.token_urlsafe(24)
    with _lock:
        _EDITS[token] = {"vault_id": vault_id, "k": k, "payload_hash": _hash(payload),
                         "profile": plain.get("profile") or PROFILE_NAMES.get(_mode_of(payload), "locked"),
                         "editors": editors or [me], "recipients": plain.get("recipients") or [],
                         "packs": plain.get("packs") or [], "at": time.time(),
                         "client_id": str(body.get("client_id") or "")}
    out = {"ok": True, "edit_token": token, "interface": plain.get("interface"), "profile": _EDITS[token]["profile"],
           "second_factor": detail["reason"]}
    if isinstance(plain.get("ui"), dict):
        out["ui"] = plain["ui"]
    else:                                                      # an old vault: the editor rebuilds it from the nodes
        out["legacy_exec"] = plain.get("exec") if isinstance(plain.get("exec"), dict) else plain
        out["api"] = SS.api_of(plain)
    return out


def _mode_of(payload: str) -> int:
    from . import vault_crypto, vault_crypto_v2 as v2
    if v2.is_v2(payload):
        return v2.parse(payload)["mode"]
    return 2 if vault_crypto.payload_mode(payload) else 0


def _session(token: str) -> dict:
    with _lock:
        s = _EDITS.get(str(token))
        if not s or time.time() - s["at"] > EDIT_IDLE_SECONDS:
            _EDITS.pop(str(token), None)
            raise VaultError("This edit session has ended (the vault re-locked). Open the vault again.")
        return s


def snapshot(body: dict) -> dict:
    """Queueing while a vault is open: encrypt the current edit WITHOUT ending the session, so the run uses the
    edited graph and nothing goes to the server in plaintext."""
    out = commit(dict(body, _keep_open=True))
    s = _session(str(body.get("edit_token") or ""))
    with _lock:
        s["payload_hash"] = _hash(out["payload"])
        s["at"] = time.time()
    return out


def commit(body: dict) -> dict:
    """Re-encrypt the edited inside under the session's key; the session ends (owner: exit = lock)."""
    from . import vault_crypto_v2 as v2

    s = _session(str(body.get("edit_token") or ""))
    if body.get("payload_hash") and body["payload_hash"] != s["payload_hash"]:
        raise VaultError("This vault was changed elsewhere while it was open here. Reload the workflow and open it "
                         "again.")
    interface = body.get("interface") or {}
    api = body.get("api") or {}
    _validate_api(api, interface)
    plain = {"exec": boundary_exec(interface), "api": api, "ui": body.get("ui"), "interface": interface,
             "editors": s["editors"], "profile": s["profile"], "packs": body.get("packs") or s["packs"],
             "updated": int(time.time())}
    recipients = None
    if s["profile"] == "sealed_installs":
        recipients = [base64.b64decode(r) for r in s["recipients"]] or _codes_to_keys([_this_install_code()])
    payload = v2.lock_with(plain, s["k"], vault_id=s["vault_id"], mode=PROFILES[s["profile"]],
                           iface=_iface(interface), recipients=recipients)
    if s["profile"] == "locked" and s.get("client_id"):
        from . import secure_subgraph as SS
        SS.RUN_SESSIONS.put(s["client_id"], s["vault_id"], s["k"]["kek"])
    if not body.get("_keep_open"):
        with _lock:
            _EDITS.pop(str(body.get("edit_token")), None)
    return {"ok": True, "payload": payload, "vault_id": s["vault_id"], "profile": s["profile"],
            "node_count": len(api["nodes"])}


def cancel(body: dict) -> dict:
    with _lock:
        _EDITS.pop(str(body.get("edit_token") or ""), None)
    return {"ok": True}


def info(body: dict) -> dict:
    from . import vault_crypto_v2 as v2
    payload = str(body.get("payload") or "")
    if not payload:
        return {"ok": False}
    legacy = not v2.is_v2(payload)
    mode = _mode_of(payload)
    return {"ok": True, "legacy_format": legacy, "profile": PROFILE_NAMES.get(mode, "locked"),
            "obfuscation_only": mode == 2,
            "this_install": _this_install_code()}


def register_routes(server) -> None:
    from aiohttp import web

    routes = server.routes

    def wrap(fn, status_on_error=403):
        async def handler(request):
            try:
                body = await request.json() if request.method == "POST" else {}
            except Exception:  # noqa: BLE001
                body = {}
            try:
                out = fn(body)
                status = 200 if out.get("ok", True) else 403
                return web.json_response({"ok": True, **out} if "ok" not in out else out, status=status)
            except VaultError as exc:
                return web.json_response({"ok": False, "error": str(exc)}, status=status_on_error)
            except PermissionError as exc:
                return web.json_response({"ok": False, "error": str(exc)}, status=429)
        return handler

    routes.post("/c2c_vault/sg/protect")(wrap(protect, 400))
    routes.post("/c2c_vault/sg/unlock_run")(wrap(unlock_run))
    routes.post("/c2c_vault/sg/forget_run")(wrap(forget_run))
    routes.post("/c2c_vault/sg/open")(wrap(open_for_edit))
    routes.post("/c2c_vault/sg/commit")(wrap(commit, 409))
    routes.post("/c2c_vault/sg/snapshot")(wrap(snapshot, 409))
    routes.post("/c2c_vault/sg/cancel")(wrap(cancel))
    routes.post("/c2c_vault/sg/info")(wrap(info, 400))
    routes.get("/c2c_vault/sg/install_code")(wrap(lambda b: {"code": _this_install_code()}, 500))
