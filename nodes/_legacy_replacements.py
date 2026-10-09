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

Known core limitation (0.36.0): ``apply_replacements`` indexes ``node_struct["inputs"][old_id]``
directly, so an API prompt that omits an optional input of a legacy node raises KeyError (HTTP 500)
instead of the 400 "node not found" it would get without a replacement - it failed either way.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

_log = logging.getLogger("C2C.legacy")
TABLE_PATH = Path(__file__).with_name("_legacy_replacements.json")


def load_table() -> list[dict[str, Any]]:
    with open(TABLE_PATH, encoding="utf-8") as f:
        return json.load(f)


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
            return web.json_response(load_table())
        except Exception as exc:  # noqa: BLE001
            return web.json_response({"error": str(exc)[:200]}, status=500)

    server.routes.get("/c2c/legacy_replacements")(_table)
    _ROUTE = True

