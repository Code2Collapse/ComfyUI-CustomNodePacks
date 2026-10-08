"""C2C Vault — in-process topological executor for a decrypted subgraph.

Runs the vault's internal nodes by calling each class's declared FUNCTION in
dependency order, then returns the values wired to the vault's boundary outputs.

Deliberately NOT reusing ComfyUI's own executor: that one caches, sends progress
per node, and reports node ids to the client. Every one of those would leak the
structure the vault exists to hide - a progress bar naming your internal nodes
defeats the whole point.

Subgraph format (what lock_subgraph encrypts):

    {
      "nodes": [{"id": "1", "class_type": "NukeMax_Add", "widgets": {...}}, ...],
      "links": [{"from": "1", "from_slot": 0, "to": "2", "to_slot": 1, "to_name": "image"}, ...],
      "boundary_in":  [{"name": "image", "to": "1", "to_slot": 0, "to_name": "image"}],

`to_name` is the target input's name (the INPUT_TYPES key). Payloads locked before 2026-10-08 carry only
`to_slot`, which is mapped onto the class's INPUT_TYPES order as before.
      "boundary_out": [{"name": "result", "from": "2", "from_slot": 0}]
    }

Error policy: any message that could reveal internals (a missing class name, a
node id) is only ever raised AFTER a successful unlock. Before unlock the node
says "Vault locked" and nothing else.
"""

from __future__ import annotations

import asyncio
import inspect
import threading
from typing import Any


class VaultExecError(RuntimeError):
    """Raised only after a successful unlock, so it may name internals."""


def _node_registry() -> dict[str, Any]:
    """The live NODE_CLASS_MAPPINGS, looked up lazily.

    Imported at call time rather than module import time: ComfyUI populates the
    registry while loading custom nodes, so a top-level import would capture a
    half-built dict (and, for this pack, its own partially-initialised self).
    """
    try:
        import nodes as comfy_nodes  # ComfyUI's own module
        return dict(getattr(comfy_nodes, "NODE_CLASS_MAPPINGS", {}) or {})
    except Exception:
        return {}


def _run_coroutine(coro):
    """Finish an async node's coroutine. The vault runs inside ComfyUI's executor, which may already be running
    an event loop on this thread, so the coroutine then gets a loop of its own on a short-lived thread."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    box: dict[str, Any] = {}

    def runner():
        try:
            box["value"] = asyncio.run(coro)
        except BaseException as exc:  # re-raised on the caller's thread
            box["error"] = exc

    t = threading.Thread(target=runner, name="c2c-vault-async-node", daemon=True)
    t.start()
    t.join()
    if "error" in box:
        raise box["error"]
    return box.get("value")


def _normalize_output(out) -> tuple:
    """A node's return value as the plain output tuple."""
    args = getattr(out, "args", None)          # V3 NodeOutput keeps its outputs in .args
    if args is not None and not isinstance(out, (tuple, dict)):
        return tuple(args)
    if isinstance(out, dict):                  # {"ui": ..., "result": ...} form
        out = out.get("result", ())
    return tuple(out) if isinstance(out, tuple) else (out,)


def _declared_only(cls, fn_name: str, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Keep only the inputs the class declares, unless its function takes **kwargs. The front-end also keeps
    display-only widgets on a node (e.g. `$$canvas-image-preview` after an image preview): passed on, they made
    ImageInvert raise "unexpected keyword argument" inside the vault (A9, L2.24)."""
    try:
        it = cls.INPUT_TYPES() or {}
        declared = set((it.get("required") or {}).keys()) | set((it.get("optional") or {}).keys())
    except Exception:
        return kwargs
    try:
        fn = getattr(cls, fn_name)
        if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in inspect.signature(fn).parameters.values()):
            return {k: v for k, v in kwargs.items() if not str(k).startswith("$$")}
    except (TypeError, ValueError):
        pass
    return {k: v for k, v in kwargs.items() if k in declared}


def _call_node(cls, fn_name: str, kwargs: dict[str, Any]) -> tuple:
    """Call one node the way ComfyUI's executor does (core 0.36 execution.py, _async_map_node_over_list), minus
    caching, progress and hidden inputs. A V3 node (comfy_api io.ComfyNode) runs on a prepared class clone; called
    directly it returned a NodeOutput object, which the next node then received as its input."""
    is_v3 = False
    try:
        from comfy_api.internal import _ComfyNodeInternal, make_locked_method_func  # type: ignore
        from comfy_api.latest import _io  # type: ignore
        is_v3 = isinstance(cls, type) and issubclass(cls, _ComfyNodeInternal)
    except Exception:
        pass
    if is_v3:
        res = _io.get_finalized_class_inputs(cls.INPUT_TYPES(), kwargs)
        v3_data = dict((res[-1] if isinstance(res, tuple) else None) or {})
        v3_data.setdefault("hidden_inputs", {})
        cls.VALIDATE_CLASS()
        clone = cls.PREPARE_CLASS_CLONE(v3_data)
        fn = make_locked_method_func(cls, fn_name, clone)
        out = fn(**_io.build_nested_inputs(kwargs, v3_data))
    else:
        out = getattr(cls(), fn_name)(**kwargs)
    if inspect.isawaitable(out):
        out = _run_coroutine(out)
    return _normalize_output(out)


def _topo_order(nodes: list[dict], links: list[dict]) -> list[str]:
    """Kahn's algorithm. Raises on a cycle rather than looping forever."""
    ids = [str(n["id"]) for n in nodes]
    indeg = {i: 0 for i in ids}
    succ: dict[str, list[str]] = {i: [] for i in ids}
    for lk in links:
        a, b = str(lk["from"]), str(lk["to"])
        if a not in indeg or b not in indeg:
            raise VaultExecError(f"Vault link references an unknown node: {a} -> {b}")
        succ[a].append(b)
        indeg[b] += 1

    queue = [i for i in ids if indeg[i] == 0]
    order: list[str] = []
    while queue:
        cur = queue.pop(0)
        order.append(cur)
        for nxt in succ[cur]:
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                queue.append(nxt)
    if len(order) != len(ids):
        stuck = sorted(set(ids) - set(order))
        raise VaultExecError(f"Vault subgraph contains a cycle involving: {stuck}")
    return order


def execute_subgraph(
    subgraph: dict[str, Any],
    boundary_inputs: dict[str, Any],
    extra_widget_overrides: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run the subgraph. Returns {boundary_out name: value}.

    Args:
        subgraph: the decrypted dict.
        boundary_inputs: {name: value} for each declared boundary_in.
        extra_widget_overrides: {node_id: {widget: value}} from promoted
            parameters. Applied BEFORE boundary inputs, so a socket wired
            straight at a widget still wins - a wire is the more specific
            statement of intent than a value typed on the vault.
    """
    nodes = list(subgraph.get("nodes") or [])
    links = list(subgraph.get("links") or [])
    b_in = list(subgraph.get("boundary_in") or [])
    b_out = list(subgraph.get("boundary_out") or [])
    if not nodes:
        raise VaultExecError("Vault subgraph contains no nodes.")

    registry = _node_registry()
    by_id = {str(n["id"]): n for n in nodes}

    # Validate promoted targets before anything else. A manifest that points at
    # a node this vault does not contain means the clear-text manifest and the
    # ciphertext have drifted apart, and no other error message would say so.
    for node_id in (extra_widget_overrides or {}):
        if str(node_id) not in by_id:
            raise VaultExecError(
                f"A promoted parameter points at node {node_id!r}, which is not "
                "in this vault. Re-lock the vault to rebuild the manifest.")

    # Resolve every class BEFORE running anything, so a missing dependency is one
    # clear message instead of a half-executed graph.
    missing = sorted({
        str(n.get("class_type")) for n in nodes
        if str(n.get("class_type")) not in registry
    })
    if missing:
        raise VaultExecError(
            "This vault needs node types that are not installed: "
            + ", ".join(missing)
            + ". Install the packs that provide them, then re-run."
        )

    # incoming[node_id][slot] = (src_id, src_slot); incoming_named[node_id][input name] = (src_id, src_slot)
    incoming: dict[str, dict[int, tuple[str, int]]] = {i: {} for i in by_id}
    incoming_named: dict[str, dict[str, tuple[str, int]]] = {i: {} for i in by_id}
    for lk in links:
        src = (str(lk["from"]), int(lk["from_slot"]))
        if lk.get("to_name"):
            incoming_named[str(lk["to"])][str(lk["to_name"])] = src
        else:
            incoming[str(lk["to"])][int(lk["to_slot"])] = src

    # boundary inputs feed specific (node, input) pairs
    injected: dict[str, dict[int, Any]] = {i: {} for i in by_id}
    injected_named: dict[str, dict[str, Any]] = {i: {} for i in by_id}
    widget_overrides: dict[str, dict[str, Any]] = {i: {} for i in by_id}
    for node_id, widgets in (extra_widget_overrides or {}).items():
        widget_overrides[str(node_id)].update(widgets)
    for spec in b_in:
        name = spec["name"]
        if name not in boundary_inputs:
            raise VaultExecError(f"Vault input {name!r} was not supplied.")
        val = boundary_inputs[name]
        widget_name = spec.get("widget")
        if widget_name:
            widget_overrides[str(spec["to"])][str(widget_name)] = val
        elif spec.get("to_name"):
            injected_named[str(spec["to"])][str(spec["to_name"])] = val
        else:
            injected[str(spec["to"])][int(spec["to_slot"])] = val

    results: dict[str, tuple] = {}
    for node_id in _topo_order(nodes, links):
        spec = by_id[node_id]
        cls = registry[str(spec["class_type"])]
        fn_name = getattr(cls, "FUNCTION", None)
        if not fn_name or not hasattr(cls, fn_name):
            raise VaultExecError(
                f"Vault node {spec['class_type']!r} declares no callable FUNCTION."
            )

        kwargs = dict(spec.get("widgets") or {})
        kwargs.update(widget_overrides.get(node_id, {}))
        # Positional sockets are addressed by index; map them onto the class's
        # declared required-input order.
        try:
            it = cls.INPUT_TYPES()
            slot_names = list((it.get("required") or {}).keys()) + list((it.get("optional") or {}).keys())
        except Exception:
            slot_names = []
        for slot, value in injected[node_id].items():
            if slot < len(slot_names):
                kwargs[slot_names[slot]] = value
        for slot, (src, src_slot) in incoming[node_id].items():
            if slot < len(slot_names):
                kwargs[slot_names[slot]] = results[src][src_slot]
        # by name (payloads locked since 2026-10-08): independent of socket order
        kwargs.update(injected_named[node_id])
        for name, (src, src_slot) in incoming_named[node_id].items():
            kwargs[name] = results[src][src_slot]

        results[node_id] = _call_node(cls, fn_name, _declared_only(cls, fn_name, kwargs))

    final: dict[str, Any] = {}
    for spec in b_out:
        src, slot = str(spec["from"]), int(spec["from_slot"])
        if src not in results:
            raise VaultExecError(f"Vault output {spec['name']!r} reads an unrun node.")
        vals = results[src]
        if slot >= len(vals):
            raise VaultExecError(
                f"Vault output {spec['name']!r} reads slot {slot} of a node that "
                f"returned {len(vals)} value(s)."
            )
        final[spec["name"]] = vals[slot]
    return final
