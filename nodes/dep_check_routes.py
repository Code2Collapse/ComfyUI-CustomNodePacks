# SPDX-License-Identifier: Apache-2.0
"""HTTP routes for the ComfyUI-Manager dependency-conflict modal.

The JS layer in ``js/c2c_dep_conflict_modal.js`` watches for successful
calls to ComfyUI-Manager's install endpoint. After Manager finishes
installing a pack, the JS asks this module to:

    1. List newly-added custom_node directories since session start.
    2. Parse each one's requirements.txt.
    3. Diff against the live `importlib.metadata` install index.
    4. Return a ConflictReport JSON so the JS can show a "force
       reinstall ENV?" modal listing every breaking / risky entry.

Routes
------
* ``GET  /c2c/depcheck/snapshot``      -> {"dirs": ["path", ...]}
                                          full list of custom_nodes dirs
                                          (used by JS to baseline a "what
                                          was here before install?" set)

* ``POST /c2c/depcheck/scan_new``      -> body: {"baseline": ["path", ...]}
                                          response: {"reports": [
                                            {"pack": "name",
                                             "path": "abs/path",
                                             "requirements": "abs/req.txt",
                                             "report": <ConflictReport.to_dict>,
                                             "summary": "<human text>"}]}

* ``POST /c2c/depcheck/preview``       -> body: {"requirements_path": "abs/path"}
                                                 OR  {"requirements_lines": ["pkg==1.0", ...]}
                                          response: {"report": ..., "summary": "..."}

No third-party HTTP framework — uses aiohttp surface that
``PromptServer`` already exposes (same pattern as the rest of the pack).
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import List

from ..dependency_checker import (
    check_conflicts,
    format_warning_message,
)

log = logging.getLogger("c2c.depcheck")


def _custom_nodes_root() -> Path:
    """Find the ComfyUI ``custom_nodes/`` directory by walking up from us."""
    here = Path(__file__).resolve()
    # this file: <comfy>/custom_nodes/ComfyUI-CustomNodePacks/nodes/dep_check_routes.py
    # parents[2] = <comfy>/custom_nodes
    try:
        return here.parents[2]
    except IndexError:  # pragma: no cover — extreme edge
        return here.parent


def _list_pack_dirs() -> List[str]:
    root = _custom_nodes_root()
    if not root.exists():
        return []
    out: List[str] = []
    for child in root.iterdir():
        try:
            if child.is_dir() and not child.name.startswith("."):
                out.append(str(child.resolve()))
        except OSError:
            continue
    return sorted(out)


def _find_requirements(pack_dir: Path) -> Path | None:
    """Locate the most likely requirements file for a freshly-installed pack."""
    for name in ("requirements.txt", "requirements-min.txt", "pyproject-deps.txt"):
        p = pack_dir / name
        if p.is_file():
            return p
    # Some packs ship requirements in a subfolder.
    for cand in pack_dir.glob("requirements*.txt"):
        if cand.is_file():
            return cand
    return None


def register_routes(server) -> None:
    try:
        from aiohttp import web
    except Exception as exc:  # pragma: no cover
        log.warning("aiohttp not available, depcheck routes skipped: %s", exc)
        return

    routes = server.routes

    @routes.get("/c2c/depcheck/snapshot")
    async def _snapshot(_request):
        return web.json_response({"success": True, "data": {"dirs": _list_pack_dirs()}})

    @routes.post("/c2c/depcheck/scan_new")
    async def _scan_new(request):
        try:
            payload = await request.json()
        except Exception as exc:
            return web.json_response(
                {"success": False, "error": "bad_json", "message": str(exc)},
                status=400,
            )
        baseline = set(payload.get("baseline") or [])
        current = set(_list_pack_dirs())
        new_dirs = sorted(current - baseline)

        reports = []
        for d in new_dirs:
            pack_dir = Path(d)
            req = _find_requirements(pack_dir)
            if req is None:
                reports.append({
                    "pack": pack_dir.name,
                    "path": str(pack_dir),
                    "requirements": None,
                    "report": None,
                    "summary": "No requirements.txt found — nothing to check.",
                })
                continue
            try:
                rep = check_conflicts(req)
                reports.append({
                    "pack": pack_dir.name,
                    "path": str(pack_dir),
                    "requirements": str(req),
                    "report": rep.to_dict(),
                    "summary": format_warning_message(rep),
                })
            except Exception as exc:
                log.warning("[c2c.depcheck] scan failed for %s: %s", pack_dir, exc)
                reports.append({
                    "pack": pack_dir.name,
                    "path": str(pack_dir),
                    "requirements": str(req),
                    "report": None,
                    "summary": f"Scan failed: {exc}",
                })
        return web.json_response({"success": True, "data": {"reports": reports}})

    @routes.post("/c2c/depcheck/preview")
    async def _preview(request):
        try:
            payload = await request.json()
        except Exception as exc:
            return web.json_response(
                {"success": False, "error": "bad_json", "message": str(exc)},
                status=400,
            )
        req_path = payload.get("requirements_path")
        req_lines = payload.get("requirements_lines")
        tmp_path: Path | None = None
        try:
            if req_path:
                target = Path(req_path)
            elif req_lines:
                if not isinstance(req_lines, list):
                    return web.json_response(
                        {"success": False, "error": "bad_payload",
                         "message": "requirements_lines must be a list of strings"},
                        status=400,
                    )
                tmp = tempfile.NamedTemporaryFile(
                    mode="w", suffix=".txt", delete=False, encoding="utf-8")
                tmp.write("\n".join(str(x) for x in req_lines))
                tmp.flush()
                tmp.close()
                tmp_path = Path(tmp.name)
                target = tmp_path
            else:
                return web.json_response(
                    {"success": False, "error": "bad_payload",
                     "message": "supply either requirements_path or requirements_lines"},
                    status=400,
                )
            if not target.is_file():
                return web.json_response(
                    {"success": False, "error": "not_found",
                     "message": f"requirements file not found: {target}"},
                    status=404,
                )
            rep = check_conflicts(target)
            return web.json_response({
                "success": True,
                "data": {
                    "report": rep.to_dict(),
                    "summary": format_warning_message(rep),
                    "has_breaking": rep.has_breaking,
                    "has_risky": bool(rep.risky),
                },
            })
        finally:
            if tmp_path is not None:
                try:
                    tmp_path.unlink()
                except OSError:
                    pass
