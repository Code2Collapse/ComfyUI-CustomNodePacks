"""C2C Vault — pure boundary derivation (Python mirror of js/c2c_vault.js).

Used for CPU tests and optional server-side validation. The live UI runs the
same classification in the browser before POST /c2c_vault/lock.
"""

from __future__ import annotations

from typing import Any


def derive_boundary(
    selection_ids: set[str],
    inputs_by_node: dict[str, list[dict[str, Any]]],
    outputs_by_node: dict[str, list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Classify links as internal, entering, or leaving the selection.

    Args:
        selection_ids: node ids inside the vault.
        inputs_by_node: {node_id: [{slot, origin_id, origin_slot}, ...]} for
            connected inputs only (origin_id None = unwired, skipped).
        outputs_by_node: {node_id: [{slot, targets: [{target_id, target_slot}]}, ...]}.

    Returns:
        (internal_links, boundary_in, boundary_out)
    """
    links: list[dict[str, Any]] = []
    boundary_in: list[dict[str, Any]] = []

    for nid in selection_ids:
        for inp in inputs_by_node.get(nid, []):
            origin_id = inp.get("origin_id")
            if origin_id is None:
                continue
            origin_id = str(origin_id)
            slot = int(inp["slot"])
            if origin_id in selection_ids:
                links.append({
                    "from": origin_id,
                    "from_slot": int(inp["origin_slot"]),
                    "to": nid,
                    "to_slot": slot,
                })
            else:
                boundary_in.append({
                    "name": f"in_{len(boundary_in)}",
                    "to": nid,
                    "to_slot": slot,
                })

    boundary_out: list[dict[str, Any]] = []
    for nid in selection_ids:
        for out in outputs_by_node.get(nid, []):
            slot = int(out["slot"])
            for tgt in out.get("targets") or []:
                target_id = str(tgt["target_id"])
                if target_id in selection_ids:
                    continue
                boundary_out.append({
                    "name": f"out_{len(boundary_out)}",
                    "from": nid,
                    "from_slot": slot,
                })

    if not boundary_out:
        consumed = {f"{l['from']}:{l['from_slot']}" for l in links}
        for nid in selection_ids:
            for out in outputs_by_node.get(nid, []):
                slot = int(out["slot"])
                key = f"{nid}:{slot}"
                if key in consumed:
                    continue
                boundary_out.append({
                    "name": f"out_{len(boundary_out)}",
                    "from": nid,
                    "from_slot": slot,
                })

    return links, boundary_in, boundary_out


def build_interface_manifest(
    mode: str,
    node_count: int,
    boundary_in: list[dict[str, Any]],
    boundary_out: list[dict[str, Any]],
    in_types: list[str] | None = None,
    out_types: list[str] | None = None,
) -> dict[str, Any]:
    """Clear-text public API manifest stored in vault_interface widget."""
    in_types = in_types or ["*"] * len(boundary_in)
    out_types = out_types or ["*"] * len(boundary_out)
    return {
        "mode": mode,
        "node_count": node_count,
        "in": [
            {"name": b["name"], "type": in_types[i] if i < len(in_types) else "*"}
            for i, b in enumerate(boundary_in)
        ],
        "out": [
            {"name": b["name"], "type": out_types[i] if i < len(out_types) else "*"}
            for i, b in enumerate(boundary_out)
        ],
    }
