"""C2C Vault as a protected native subgraph - the server side (A18, docs/VAULT_DESIGN_A18.md).

A locked vault is ONE node in the prompt. When it executes it decrypts its payload with the key its security profile
allows and returns a NATIVE EXPANSION: ComfyUI's own scheduler then runs the inner nodes (lazy inputs, hidden inputs,
lists, caching, async nodes, interrupts, progress). There is no custom executor here.

The vault's boundary inputs are declared lazy and never requested. Inner nodes that read a boundary input reference
the UPSTREAM node directly, so an upstream branch runs only when an inner node really needs it - the same lazy
behaviour as an open subgraph.

Leak control: messages about the inner (ephemeral) nodes are rewritten to name only the vault, and errors keep the
inner node type + message but lose inputs, outputs and traceback (owner A18: "type + message, no values").
"""
from __future__ import annotations

import logging
import os
import re
import threading
from typing import Any, Optional

from .vault_crypto import VaultError

log = logging.getLogger("c2c.vault")

KILL_SWITCH = "C2C_VAULT_NATIVE"
MAX_OUTPUTS = 256                      # RETURN_TYPES must be fixed per class; the frontend shows only the used ones
IN_PREFIX, PARAM_PREFIX = "in_", "p_"


def disabled() -> bool:
    return os.environ.get(KILL_SWITCH, "").strip().lower() in ("off", "0", "false", "no")


class _AnyType(str):
    """Matches every socket type (the vault's sockets take whatever the inside takes)."""

    def __ne__(self, other):
        return False


ANY = _AnyType("*")


class _BoundaryInputs(dict):
    """Optional inputs of any name: `in_<n>` are the vault's lazy boundary sockets, `p_<key>` promoted controls."""

    def __contains__(self, key):
        return isinstance(key, str) and (key.startswith(IN_PREFIX) or key.startswith(PARAM_PREFIX))

    def __getitem__(self, key):
        if isinstance(key, str) and key.startswith(IN_PREFIX):
            return (ANY, {"lazy": True})
        if isinstance(key, str) and key.startswith(PARAM_PREFIX):
            return (ANY, {})
        raise KeyError(key)


# ── keys: what may open a vault for a RUN ───────────────────────────────────────────────────────────────────

class RunSessions:
    """Locked profile: the key a browser tab unlocked, in memory only, per (client id, vault id) - owner A18:
    "once per browser tab". Dropped when the tab's websocket closes or ComfyUI restarts."""

    def __init__(self):
        self._lock = threading.Lock()
        self._keys: dict[tuple[str, str], bytes] = {}

    def put(self, client_id: str, vault_id: str, key: bytes) -> None:
        with self._lock:
            self._keys[(str(client_id), str(vault_id))] = key

    def get(self, client_id: str, vault_id: str) -> Optional[bytes]:
        with self._lock:
            return self._keys.get((str(client_id), str(vault_id)))

    def drop_client(self, client_id: str) -> None:
        with self._lock:
            for k in [k for k in self._keys if k[0] == str(client_id)]:
                del self._keys[k]


RUN_SESSIONS = RunSessions()


def _current_client_id() -> Optional[str]:
    try:
        from server import PromptServer
        return getattr(PromptServer.instance, "client_id", None)
    except Exception:  # noqa: BLE001
        return None


def open_for_run(payload: str, vault_id: str, client_id: Optional[str] = None) -> dict:
    """Decrypt for execution with whatever this install / tab is allowed to use. Raises VaultError with what is
    missing, never with anything from inside the vault."""
    from . import vault_crypto, vault_crypto_v2 as v2, vault_keys, vault_node

    client_id = client_id if client_id is not None else _current_client_id()
    if v2.is_v2(payload):
        mode = v2.parse(payload)["mode"]
        if mode == v2.MODE_SEALED_PORTABLE:
            return v2.open_portable(payload, vault_id=vault_id)
        if mode == v2.MODE_SEALED_INSTALLS:
            priv = vault_keys.install_private_key(create=False)
            if priv is None:
                raise VaultError("This vault is not authorized on this ComfyUI installation (it has no vault key "
                                 "yet). Send its author your installation code (C2C menu > Vault > My "
                                 "installation code).")
            try:
                return v2.open_with_install_key(payload, priv, vault_id=vault_id)
            except VaultError:
                raise VaultError("This vault is not authorized on this ComfyUI installation. Send its author your "
                                 "installation code (C2C menu > Vault > My installation code).") from None
        key = (RUN_SESSIONS.get(client_id, vault_id) if client_id else None) or vault_node.SESSIONS.get(vault_id)
        if key is None:
            raise VaultError("This vault needs its password to run. Click the vault and enter it (asked once per "
                             "browser tab).")
        return v2.open_with_kek(payload, key, vault_id=vault_id)
    # v1 (the first vault format)
    if vault_crypto.payload_mode(payload) != 0:
        return vault_crypto.unseal_for_run(payload, vault_id=vault_id)
    key = (RUN_SESSIONS.get(client_id, vault_id) if client_id else None) or vault_node.SESSIONS.get(vault_id)
    if key is None:
        raise VaultError("This vault needs its password to run. Click the vault and enter it (asked once per browser "
                         "tab).")
    return vault_node._unlock_with_key(payload, key, vault_id)


# ── the inner graph in API form ─────────────────────────────────────────────────────────────────────────────

def _slot_names(class_type: str) -> list:
    import nodes
    cls = nodes.NODE_CLASS_MAPPINGS.get(class_type)
    try:
        it = cls.INPUT_TYPES() if cls is not None else {}
    except Exception:  # noqa: BLE001
        it = {}
    return list((it.get("required") or {}).keys()) + list((it.get("optional") or {}).keys())


def api_from_legacy(ex: dict) -> dict:
    """The flattened format of v1 / v2 vaults (nodes with named widgets, links by name or slot, boundary specs) ->
    the API form the expansion uses. Old vaults therefore run on the native scheduler too."""
    nodes_in = list(ex.get("nodes") or [])
    by_id = {str(n["id"]): n for n in nodes_in}
    api = {str(n["id"]): {"class_type": str(n["class_type"]), "inputs": dict(n.get("widgets") or {})}
           for n in nodes_in}

    def name_for(node_id: str, lk: dict, key_name: str, key_slot: str) -> Optional[str]:
        if lk.get(key_name):
            return str(lk[key_name])
        names = _slot_names(by_id[node_id]["class_type"])
        slot = int(lk.get(key_slot, -1))
        return names[slot] if 0 <= slot < len(names) else None

    for lk in ex.get("links") or []:
        to = str(lk["to"])
        name = name_for(to, lk, "to_name", "to_slot")
        if name and to in api:
            api[to]["inputs"][name] = [str(lk["from"]), int(lk["from_slot"])]
    inputs = []
    for i, spec in enumerate(ex.get("boundary_in") or []):
        inputs.append({"name": spec.get("name", f"input {i}"), "type": spec.get("type", "*")})
        to = str(spec["to"])
        name = str(spec["widget"]) if spec.get("widget") else name_for(to, spec, "to_name", "to_slot")
        if name and to in api:
            api[to]["inputs"][name] = {"vault_in": i}
    outputs = [{"name": s.get("name", f"output {i}"), "type": s.get("type", "*"), "node": str(s["from"]),
                "slot": int(s["from_slot"])} for i, s in enumerate(ex.get("boundary_out") or [])]
    promoted = [{"key": str(p.get("id")), "node": str(p.get("node")), "input": str(p.get("widget"))}
                for p in (ex.get("promoted") or []) if isinstance(p, dict) and p.get("id") and p.get("node")]
    return {"nodes": api, "inputs": inputs, "outputs": outputs, "promoted": promoted}


def api_of(plain: dict) -> dict:
    if isinstance(plain.get("api"), dict):
        return plain["api"]
    return api_from_legacy(plain.get("exec") if isinstance(plain.get("exec"), dict) else plain)


def missing_classes(api: dict) -> list:
    import nodes
    return sorted({str(n.get("class_type")) for n in (api.get("nodes") or {}).values()
                   if str(n.get("class_type")) not in nodes.NODE_CLASS_MAPPINGS})


def _is_link(v) -> bool:
    return isinstance(v, (list, tuple)) and len(v) == 2 and isinstance(v[0], str) and isinstance(v[1], int)


def build_expansion(api: dict, my_inputs: dict, params: dict, prefix: str, display_id: str) -> tuple[dict, list]:
    """-> (graph for core's expansion, result list). `my_inputs` = the vault's own prompt inputs (upstream links or
    literal values per `in_<i>`), `params` = evaluated promoted-control values per key."""
    inner = api.get("nodes") or {}
    nid = lambda i: f"{prefix}{i}"  # noqa: E731
    graph: dict[str, Any] = {}
    promoted = {}
    for p in api.get("promoted") or []:
        if str(p.get("key")) in params:
            promoted.setdefault(str(p["node"]), {})[str(p["input"])] = params[str(p["key"])]
    for node_id, node in inner.items():
        inputs = {}
        for name, v in (node.get("inputs") or {}).items():
            if _is_link(v) and v[0] in inner:
                inputs[name] = [nid(v[0]), int(v[1])]
            elif isinstance(v, dict) and "vault_in" in v:
                src = my_inputs.get(f"{IN_PREFIX}{int(v['vault_in'])}")
                if src is not None:
                    inputs[name] = list(src) if _is_link(src) else src
            else:
                inputs[name] = v
        inputs.update(promoted.get(str(node_id), {}))
        graph[nid(node_id)] = {"class_type": node["class_type"], "inputs": inputs, "override_display_id": display_id}
    result = []
    for out in api.get("outputs") or []:
        if "input" in out:                                   # a boundary input wired straight to an output
            src = my_inputs.get(f"{IN_PREFIX}{int(out['input'])}")
            result.append(list(src) if _is_link(src) else src)
        else:
            result.append([nid(out["node"]), int(out["slot"])])
    return graph, result


# ── the node ────────────────────────────────────────────────────────────────────────────────────────────────

_ACTIVE_VAULTS: set = set()            # prompt ids of vaults that expanded: their inner ids start with "<id>."


class C2C_SecureSubgraph:
    """A protected native subgraph: one node outside, ComfyUI's scheduler inside."""

    CATEGORY = "C2C/Vault"
    FUNCTION = "execute"
    RETURN_TYPES = tuple([ANY] * MAX_OUTPUTS)
    RETURN_NAMES = tuple(f"out_{i}" for i in range(MAX_OUTPUTS))
    DESCRIPTION = ("A C2C Vault: a subgraph whose inside is encrypted. It runs with ComfyUI's own scheduler like any "
                   "subgraph; opening it needs the vault password and your second factor. Create one by "
                   "right-clicking a selection or a subgraph: Lock as C2C Vault / Seal as C2C Vault.")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "vault_id": ("STRING", {"default": ""}),
            "vault_payload": ("STRING", {"default": "", "multiline": False}),
        }, "optional": _BoundaryInputs(), "hidden": {"unique_id": "UNIQUE_ID", "dynprompt": "DYNPROMPT"}}

    def check_lazy_status(self, **_kw):
        return []                                             # boundary inputs are wired straight into the inside

    def execute(self, vault_id, vault_payload, unique_id=None, dynprompt=None, **kw):
        if disabled():
            raise RuntimeError(f"C2C Vault native execution is switched off ({KILL_SWITCH}=off).")
        if not (vault_payload or "").strip():
            raise RuntimeError("This C2C Vault is empty: lock a selection or a subgraph first.")
        try:
            plain = open_for_run(vault_payload, vault_id)
        except VaultError as exc:
            raise RuntimeError(str(exc)) from None
        api = api_of(plain)
        missing = missing_classes(api)
        if missing:
            _note_missing(vault_id, missing)
            raise RuntimeError("This workflow needs custom nodes that are not installed, so it cannot run. Use "
                               "'Install' on the vault node (it asks before installing anything).")
        me = dynprompt.get_node(unique_id) if dynprompt is not None else {}
        my_inputs = dict((me or {}).get("inputs") or {})
        params = {k[len(PARAM_PREFIX):]: v for k, v in kw.items() if k.startswith(PARAM_PREFIX)}
        from comfy_execution.graph_utils import GraphBuilder
        prefix = GraphBuilder.alloc_prefix()
        graph, result = build_expansion(api, my_inputs, params, prefix, str(unique_id))
        _ACTIVE_VAULTS.add(str(unique_id))
        _INNER_TYPES[str(unique_id)] = {k: str(n["class_type"]) for k, n in graph.items()}
        result += [None] * (len(self.RETURN_TYPES) - len(result))
        return {"result": tuple(result), "expand": graph}


class C2C_SecureSubgraphOutput(C2C_SecureSubgraph):
    """The same, for vaults whose inside saves or previews something: runs even with nothing wired after it."""

    OUTPUT_NODE = True


_MISSING: dict[str, list] = {}         # vault id -> missing classes on this install (for the consent install route)


def _note_missing(vault_id: str, classes: list) -> None:
    _MISSING[str(vault_id)] = list(classes)


# ── leak control on the server's messages ───────────────────────────────────────────────────────────────────

def _vault_of(node_id) -> Optional[str]:
    """The vault id an ephemeral inner node belongs to, else None. Inner ids are "<vault id>.<call>.<graph>.<inner>"
    (GraphBuilder.alloc_prefix), so a file name like "10.png" in a message is never mistaken for one."""
    s = str(node_id) if node_id is not None else ""
    if "." not in s:
        return None
    best = None
    for v in _ACTIVE_VAULTS:
        if not s.startswith(v + "."):
            continue
        parts = s[len(v) + 1:].split(".", 3)
        if len(parts) < 3 or not (parts[0].isdigit() and parts[1].isdigit() and parts[2]):
            continue
        known = _INNER_TYPES.get(v)                           # the ids this vault's expansion really made
        if known is not None and f"{v}.{parts[0]}.{parts[1]}.{parts[2]}" not in known:
            continue                                          # (their own expansions are deeper: same first part)
        if best is None or len(v) < len(best):
            best = v                                          # the OUTERMOST vault (nested vaults stay hidden)
    return best


_VAULT_CLASSES = ("C2C_SecureSubgraph", "C2C_SecureSubgraphOutput")
_INNER_TYPES: dict[str, dict[str, str]] = {}   # vault id -> {inner id: class type} of its latest expansion
_LAST_INNER: dict[str, str] = {}               # vault id -> the inner node that started last (names an inner error)
_DETAIL_KEYS = ("traceback", "trace", "current_inputs", "current_outputs")
_INNER_ID_RE = re.compile(r"[\w:]+\.\d+\.\d+\.[\w:.-]+")    # "9:10.0.0.3" inside a message text


def _inner_type(vault: str, inner: Optional[str]) -> Optional[str]:
    return (_INNER_TYPES.get(vault) or {}).get(str(inner)) if inner else None


def _deep_map(obj: Any) -> tuple[Any, bool]:
    """Every inner id inside an arbitrary payload (custom events, telemetry) -> its vault id. -> (new, changed)."""
    if isinstance(obj, str):
        v = _vault_of(obj)
        return (v, True) if v is not None else (obj, False)
    if isinstance(obj, dict):
        out, changed = {}, False
        for k, val in obj.items():
            k2, ck = _deep_map(k) if isinstance(k, str) else (k, False)
            v2, cv = _deep_map(val)
            out[k2] = v2
            changed = changed or ck or cv
        return out, changed
    if isinstance(obj, (list, tuple)):
        items = [_deep_map(x) for x in obj]
        changed = any(c for _, c in items)
        if not changed:
            return obj, False
        vals = [x for x, _ in items]
        return (tuple(vals) if isinstance(obj, tuple) else vals), True
    return obj, False


def _sanitize_error(d: dict) -> dict:
    """An error / interrupt anywhere inside a vault -> the vault, the inner node TYPE and the message only. The core
    already reports an inner failure under the vault's id (get_real_node_id), so the inner node is the one that
    started last."""
    nid = str(d.get("node_id"))
    v = _vault_of(nid)
    inner = nid if v is not None else None
    if v is None and nid in _ACTIVE_VAULTS:
        v, inner = nid, _LAST_INNER.get(nid)
    if v is not None or d.get("node_type") in _VAULT_CLASSES:
        d["node_id"] = v or nid
        d["node_type"] = _inner_type(v, inner) or d.get("node_type")
        msg = d.get("exception_message")
        if isinstance(msg, str) and v is not None:             # core names nodes by id in its own messages
            d["exception_message"] = _INNER_ID_RE.sub(lambda m: _vault_of(m.group(0)) or m.group(0), msg)
        for k in _DETAIL_KEYS:
            if k in d:
                d[k] = type(d[k])() if isinstance(d[k], (list, dict)) else None
    for k in ("executed", "current_outputs"):
        if isinstance(d.get(k), (list, tuple)):
            d[k] = [x for x in d[k] if _vault_of(x) is None]
    return d


def _sanitize_binary(event: int, data: Any) -> Any:
    """Binary websocket events: PREVIEW_IMAGE_WITH_METADATA = (image, metadata dict), TEXT = >I len + node id + text."""
    if event == 4 and isinstance(data, tuple) and len(data) == 2 and isinstance(data[1], dict):
        meta, changed = _deep_map(data[1])
        return (data[0], meta) if changed else data
    if event == 3 and isinstance(data, (bytes, bytearray)) and len(data) >= 4:
        import struct
        n = struct.unpack(">I", bytes(data[:4]))[0]
        node = bytes(data[4:4 + n]).decode("utf-8", "replace")
        v = _vault_of(node)
        if v is not None:
            vb = v.encode("utf-8")
            return struct.pack(">I", len(vb)) + vb + bytes(data[4 + n:])
    return data


def sanitize_message(event: str, data: Any, track: bool = True) -> Any:
    """`track` = a live message (it moves the 'which inner node runs' state); history replays pass False."""
    if isinstance(event, int):
        return _sanitize_binary(event, data) if _ACTIVE_VAULTS else data
    if not isinstance(data, dict):
        return data
    if event in ("execution_error", "execution_interrupted"):
        return _sanitize_error(dict(data))
    if not _ACTIVE_VAULTS:
        return data
    d = dict(data)
    if event == "execution_start" and track:
        _LAST_INNER.clear()
    if event in ("executing", "executed", "progress"):
        node = d.get("node")
        v = _vault_of(node)
        if event == "executing" and node is not None and track:
            if v is not None:
                _LAST_INNER[v] = str(node)
            elif str(node) in _ACTIVE_VAULTS:
                _LAST_INNER.pop(str(node), None)           # the vault itself starts: nothing inside has run yet
        if v is not None:
            d["node"] = v
            if "display_node" in d:
                d["display_node"] = v
    elif event == "execution_cached":
        d["nodes"] = [x for x in (d.get("nodes") or []) if _vault_of(x) is None]
    elif event == "progress_state":
        nodes_ = {}
        for k, st in (d.get("nodes") or {}).items():
            v = _vault_of(k)
            if v is None:
                nodes_[k] = st
            elif v not in nodes_:
                st = dict(st)
                st.update({"node_id": v, "display_node_id": v, "real_node_id": v, "parent_node_id": None})
                nodes_[v] = st
        d["nodes"] = nodes_
    d, changed = _deep_map(d)                                # any other event (custom packs' telemetry included)
    if changed and event not in ("executing", "executed", "progress", "progress_state", "execution_cached"):
        for k in _DETAIL_KEYS:
            d.pop(k, None)
    return d


def sanitize_history(result: dict) -> dict:
    """History of a prompt that ran vaults: inner preview outputs filed under the vault, no inner ids in meta."""
    if not isinstance(result, dict) or not _ACTIVE_VAULTS:
        return result
    out = dict(result)
    outputs = {}
    for k, v in (result.get("outputs") or {}).items():
        vault = _vault_of(k)
        if vault is None:
            outputs[k] = v
            continue
        merged = outputs.setdefault(vault, {})
        for key, val in (v or {}).items():
            merged.setdefault(key, [])
            merged[key] = list(merged[key]) + (list(val) if isinstance(val, (list, tuple)) else [val])
    out["outputs"] = outputs
    if "meta" in result:
        out["meta"] = {k: v for k, v in (result.get("meta") or {}).items() if _vault_of(k) is None}
    return out


_installed = False


def install_message_filter(server=None) -> bool:
    """Wrap PromptServer.send_sync and the history store once."""
    global _installed
    if _installed:
        return True
    try:
        if server is None:
            from server import PromptServer
            server = PromptServer.instance
        orig = server.send_sync

        def send_sync(event, data, sid=None, _orig=orig):
            try:
                data = sanitize_message(event, data)
            except Exception:  # noqa: BLE001 - never break core messaging
                log.exception("C2C Vault: message filter failed")
            return _orig(event, data, sid)

        server.send_sync = send_sync
        q = getattr(server, "prompt_queue", None)
        if q is not None and hasattr(q, "task_done"):
            orig_done = q.task_done

            def task_done(item_id, history_result, status, *a, _orig=orig_done, **k):
                try:
                    history_result = sanitize_history(history_result)
                    if status is not None and getattr(status, "messages", None):
                        msgs = [(e, sanitize_message(e, d, track=False)) for e, d in status.messages]
                        status = status._replace(messages=msgs) if hasattr(status, "_replace") else status
                except Exception:  # noqa: BLE001
                    log.exception("C2C Vault: history filter failed")
                return _orig(item_id, history_result, status, *a, **k)

            q.task_done = task_done
        _installed = True
        return True
    except Exception as exc:  # noqa: BLE001
        log.warning("C2C Vault: message filter not installed: %s", exc)
        return False


NODE_CLASS_MAPPINGS = {
    "C2C_SecureSubgraph": C2C_SecureSubgraph,
    "C2C_SecureSubgraphOutput": C2C_SecureSubgraphOutput,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "C2C_SecureSubgraph": "C2C Vault (protected subgraph)",
    "C2C_SecureSubgraphOutput": "C2C Vault (protected subgraph, saves)",
}
