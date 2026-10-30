"""Saved-workflow compatibility for node ids merged into the unified nodes (issue #11, ledger L2.02).

Fourteen legacy ids were folded into MaskEditMEC, SplineMaskMEC, MaskTrackerMEC, ProPainterMEC and
MaskOpsMEC without leaving anything behind, so every saved workflow that used them stopped loading.
ComfyUI core (app/node_replace_manager.py, added 2026-02-15) lets a pack declare
``old_node_id -> new_node_id`` with input and output mappings:

* the frontend (1.52.7) offers "Replace Node" for a missing type when a workflow loads, moving the
  links and the saved widget values (``old_widget_ids`` maps the positional ``widgets_values`` of the
  old node to input names) and setting the new node's mode;
* the server applies the same mapping to API prompts that still name an old id.

The table is data (``_legacy_replacements.json``) generated from the LAST version of each legacy
class and checked against the live node definitions; ``tests/test_legacy_replacements.py`` keeps it
honest. A workflow saved with an OLDER version of a legacy node whose widget list later changed can
still land values in the wrong widgets - one ``old_widget_ids`` per replacement is all core supports.

Core limitation (0.36.0 - 0.39.x, measured 2026-10-15): ``apply_replacements`` indexes
``node_struct["inputs"][old_id]`` directly, so an API prompt that omits an optional input of a legacy node raised
KeyError and the whole POST /prompt answered HTTP 500. Fixed here: an on_prompt handler (core runs those BEFORE
apply_replacements) migrates C2C's own legacy ids itself with the same semantics, skipping a mapped input the prompt
does not carry (the new node's default applies), so core finds nothing left to replace.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

_log = logging.getLogger("C2C.legacy")
TABLE_PATH = Path(__file__).with_name("_legacy_replacements.json")
# Nodes removed with no successor (L7.65 consolidation): core's NodeReplace needs a new_node_id, so these are not
# registered with core; the front end takes them out of a loading workflow and says so (js/c2c_auto_migrate.js).
REMOVED_PATH = Path(__file__).with_name("_removed_nodes.json")


def load_table() -> list[dict[str, Any]]:
    with open(TABLE_PATH, encoding="utf-8") as f:
        return json.load(f)


def load_removed() -> list[dict[str, Any]]:
    try:
        with open(REMOVED_PATH, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return []


_ROWS_BY_OLD: dict[str, dict[str, Any]] | None = None


def _rows_by_old() -> dict[str, dict[str, Any]]:
    global _ROWS_BY_OLD
    if _ROWS_BY_OLD is None:
        _ROWS_BY_OLD = {r["old_node_id"]: r for r in load_table()}
    return _ROWS_BY_OLD


def pick_case(row: dict[str, Any], inputs: Any) -> dict[str, Any]:
    """The row that applies to one old node (js/_c2c_migrate_core.js pickCase): a removed multi-mode node can send
    each mode to a different successor (``c2c_by_value: {widget, cases: {str(value): {...row fields}}}``, L7.65 P01).
    An API prompt names its inputs, so the value is ``inputs[widget]``; no matching case = the row itself."""
    by = row.get("c2c_by_value")
    if not isinstance(by, dict) or not isinstance(inputs, dict) or by.get("widget") not in inputs:
        return row
    value = inputs[by["widget"]]
    key = ("true" if value else "false") if isinstance(value, bool) else str(value)
    case = (by.get("cases") or {}).get(key)
    if not isinstance(case, dict):
        return row
    return {**{k: v for k, v in row.items() if k != "c2c_by_value"}, **case}


def _js_str(v: Any) -> str:
    """A value as JavaScript's String() writes it (the map keys of c2c_values are written that way)."""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)


def translate_value(spec: dict[str, Any], value: Any) -> tuple[bool, Any]:
    """One c2c_values entry applied to a saved value (js/_c2c_migrate_core.js translate): a map by the value's text,
    then log2, then a clamp. (False, None) when the map has no entry for it."""
    nv = value
    if isinstance(spec.get("map"), dict):
        key = _js_str(value)
        if key not in spec["map"]:
            return False, None
        nv = spec["map"][key]
    if spec.get("fn") == "log2":
        import math
        try:
            x = float(nv)
        except (TypeError, ValueError):
            return False, None
        if not x > 0:
            return False, None
        nv = math.log2(x)
    clamp = spec.get("clamp")
    if isinstance(clamp, (list, tuple)) and len(clamp) == 2 and isinstance(nv, (int, float))             and not isinstance(nv, bool):
        nv = min(clamp[1], max(clamp[0], nv))
    return True, nv


def migrate_api_prompt(prompt: Any, registered: Any = None) -> int:
    """Replace C2C legacy ids in an API prompt in place, as core's NodeReplaceManager.apply_replacements does
    (class_type, input_mapping old_id / set_value, output_mapping on every link that reads the node), except that a
    mapped input the prompt omits is skipped instead of raising KeyError. ``registered``: the live node ids; a row
    whose new id is not registered is left alone (core does the same). Returns how many nodes were replaced."""
    if not isinstance(prompt, dict):
        return 0
    rows = _rows_by_old()
    targets = {}
    for nid, node in prompt.items():
        if isinstance(node, dict) and node.get("class_type") in rows:
            row = pick_case(rows[node["class_type"]], node.get("inputs"))
            if registered is not None and row["new_node_id"] not in registered:
                continue
            targets[str(nid)] = row
    if not targets:
        return 0
    for nid, row in targets.items():
        old = prompt[nid]
        old_inputs = old.get("inputs") or {}
        new = {k: v for k, v in old.items() if k != "inputs"}
        new["class_type"] = row["new_node_id"]
        new["inputs"] = {}
        for m in row.get("input_mapping") or []:
            if "set_value" in m:
                new["inputs"][m["new_id"]] = m["set_value"]
            elif "old_id" in m and m["old_id"] in old_inputs:
                new["inputs"][m["new_id"]] = old_inputs[m["old_id"]]
        # value translations (c2c_values), as the front end applies them (L7.65 P30: a saved half_float becomes the
        # successor's bit_depth); a wired input is a link and is left alone
        for v in row.get("c2c_values") or []:
            if "set" in v:
                new["inputs"][v["new_id"]] = v["set"]
                continue
            ov = old_inputs.get(v.get("old_id"))
            if v.get("old_id") not in old_inputs or (isinstance(ov, list) and len(ov) == 2):
                continue
            ok, nv = translate_value(v, ov)
            if ok:
                new["inputs"][v["new_id"]] = nv
        prompt[nid] = new
    out_maps = {nid: {int(o["old_idx"]): int(o["new_idx"]) for o in row.get("output_mapping") or []}
                for nid, row in targets.items()}
    for node in prompt.values():
        if not isinstance(node, dict):
            continue
        for name, value in list((node.get("inputs") or {}).items()):
            if (isinstance(value, list) and len(value) == 2 and str(value[0]) in out_maps
                    and isinstance(value[1], int) and value[1] in out_maps[str(value[0])]):
                node["inputs"][name] = [value[0], out_maps[str(value[0])][value[1]]]
    return len(targets)


_WIDGET_TYPES = {"INT", "FLOAT", "STRING", "BOOLEAN", "COMBO"}


def _widget_default(spec: Any) -> tuple[bool, Any]:
    """(True, default) for a widget input spec, (False, None) for a socket."""
    if not isinstance(spec, (list, tuple)) or not spec:
        return False, None
    kind = spec[0]
    opts = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
    if opts.get("forceInput"):
        return False, None
    if isinstance(kind, (list, tuple)):                       # a combo given as its option list
        return True, opts.get("default", kind[0] if kind else "")
    if kind == "COMBO":
        options = opts.get("options") or []
        return True, opts.get("default", options[0] if options else "")
    if kind in _WIDGET_TYPES:
        if "default" in opts:
            return True, opts["default"]
        return True, {"INT": 0, "FLOAT": 0.0, "STRING": "", "BOOLEAN": False}.get(kind)
    return False, None


def fill_required_defaults(node: dict[str, Any], cls: Any) -> list[str]:
    """A migrated API node gets the successor's REQUIRED widget inputs it lacks, at their defaults (the old node had
    none of them; the front end gets them from the widgets, an API prompt does not). Sockets are never invented."""
    try:
        required = (cls.INPUT_TYPES() or {}).get("required") or {}
    except Exception:
        return []
    filled = []
    inputs = node.setdefault("inputs", {})
    for name, spec in required.items():
        if name in inputs:
            continue
        ok, value = _widget_default(spec)
        if ok:
            inputs[name] = value
            filled.append(name)
    return filled


def _on_prompt(json_data: Any) -> Any:
    """PromptServer on_prompt hook: runs before core's apply_replacements (server.post_prompt)."""
    try:
        prompt = json_data.get("prompt") if isinstance(json_data, dict) else None
        if isinstance(prompt, dict):
            try:
                import nodes as _core_nodes  # noqa: WPS433 - the merged mappings of the running server
                mappings = getattr(_core_nodes, "NODE_CLASS_MAPPINGS", None) or None
            except Exception:
                mappings = None
            registered = set(mappings) if mappings else None        # unknown -> do not filter
            before = {k: v.get("class_type") for k, v in prompt.items() if isinstance(v, dict)}
            n = migrate_api_prompt(prompt, registered)
            if n and mappings:
                for k, v in prompt.items():
                    if isinstance(v, dict) and before.get(k) != v.get("class_type") and v.get("class_type") in mappings:
                        fill_required_defaults(v, mappings[v["class_type"]])
            if n:
                _log.info("[C2C] migrated %d legacy node(s) in an API prompt", n)
    except Exception as exc:  # a migration problem must never block a prompt
        _log.warning("[C2C] API prompt migration skipped: %s", exc)
    return json_data


def migration_rows() -> list[dict[str, Any]]:
    """What the front end migrates: every replacement row, then every removal marked ``c2c_remove``."""
    return load_table() + [{**row, "c2c_remove": True} for row in load_removed()]


def register(server: Any = None) -> int:
    """Register every replacement with core's NodeReplaceManager. Returns how many were registered;
    0 (and no error) on a core that predates node replacements, so the pack still loads there."""
    if server is None:
        try:
            from server import PromptServer
            server = getattr(PromptServer, "instance", None)
        except Exception:
            server = None
    _register_route(server)          # the front end's automatic migration works even where core cannot replace
    _register_on_prompt(server)      # API prompts: migrate before core's replacement can raise KeyError
    try:
        from comfy_api.latest import io as _io  # NodeReplace lives in comfy_api.latest._io
        node_replace = getattr(_io, "NodeReplace")
    except Exception:
        _log.info("[C2C] node replacements unavailable in this ComfyUI core; legacy ids stay unmapped")
        return 0
    manager = getattr(server, "node_replace_manager", None)
    if manager is None:
        _log.info("[C2C] no node_replace_manager on this server; legacy ids stay unmapped")
        return 0
    count = 0
    # a c2c_by_value row registers its base row only (core has no per-value cases); C2C's own on_prompt handler and the
    # front-end migrator apply the cases first, so core sees nothing left to replace for them
    for row in load_table():
        try:
            manager.register(node_replace(
                new_node_id=row["new_node_id"],
                old_node_id=row["old_node_id"],
                old_widget_ids=row.get("old_widget_ids") or None,
                input_mapping=row.get("input_mapping") or None,
                output_mapping=row.get("output_mapping") or None,
            ))
            count += 1
        except Exception as exc:  # one bad row must not cost the others
            _log.warning("[C2C] could not register replacement %s -> %s: %s",
                         row.get("old_node_id"), row.get("new_node_id"), exc)
    return count


_ROUTE = False
_ON_PROMPT = False


def _register_on_prompt(server: Any) -> None:
    global _ON_PROMPT
    if _ON_PROMPT or server is None or not hasattr(server, "add_on_prompt_handler"):
        return
    server.add_on_prompt_handler(_on_prompt)
    _ON_PROMPT = True


def _register_route(server: Any) -> None:
    """GET /c2c/legacy_replacements - C2C's own rows (not every pack's, as core's /node_replacements serves) for the
    automatic migration on workflow load (js/c2c_auto_migrate.js)."""
    global _ROUTE
    if _ROUTE or server is None or getattr(server, "routes", None) is None:
        return
    try:
        from aiohttp import web
    except Exception:
        return

    async def _table(_request):
        try:
            return web.json_response(migration_rows())
        except Exception as exc:  # noqa: BLE001
            return web.json_response({"error": str(exc)[:200]}, status=500)

    server.routes.get("/c2c/legacy_replacements")(_table)
    _ROUTE = True

