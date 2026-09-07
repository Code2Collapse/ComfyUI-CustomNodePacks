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
      "links": [{"from": "1", "from_slot": 0, "to": "2", "to_slot": 1}, ...],
      "boundary_in":  [{"name": "image", "to": "1", "to_slot": 0}],
      "boundary_out": [{"name": "result", "from": "2", "from_slot": 0}]
    }

Error policy: any message that could reveal internals (a missing class name, a
node id) is only ever raised AFTER a successful unlock. Before unlock the node
says "Vault locked" and nothing else.
"""

from __future__ import annotations

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


def execute_subgraph(subgraph: dict[str, Any], boundary_inputs: dict[str, Any]) -> dict[str, Any]:
    """Run the subgraph. Returns {boundary_out name: value}.

    Args:
        subgraph: the decrypted dict.
        boundary_inputs: {name: value} for each declared boundary_in.
    """
    nodes = list(subgraph.get("nodes") or [])
    links = list(subgraph.get("links") or [])
    b_in = list(subgraph.get("boundary_in") or [])
    b_out = list(subgraph.get("boundary_out") or [])
    if not nodes:
        raise VaultExecError("Vault subgraph contains no nodes.")

    registry = _node_registry()
    by_id = {str(n["id"]): n for n in nodes}

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

    # incoming[node_id][slot] = (src_id, src_slot)
    incoming: dict[str, dict[int, tuple[str, int]]] = {i: {} for i in by_id}
    for lk in links:
        incoming[str(lk["to"])][int(lk["to_slot"])] = (str(lk["from"]), int(lk["from_slot"]))

    # boundary inputs feed specific (node, slot) pairs
    injected: dict[str, dict[int, Any]] = {i: {} for i in by_id}
    for spec in b_in:
        name = spec["name"]
        if name not in boundary_inputs:
            raise VaultExecError(f"Vault input {name!r} was not supplied.")
        injected[str(spec["to"])][int(spec["to_slot"])] = boundary_inputs[name]

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

        out = getattr(cls(), fn_name)(**kwargs)
        if isinstance(out, dict):          # {"ui":..., "result":...} form
            out = out.get("result", ())
        results[node_id] = tuple(out) if isinstance(out, tuple) else (out,)

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
