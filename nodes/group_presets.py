"""
group_presets.py — Phase 11: Node Group Presets ("Macro Nodes").

Filesystem-backed preset library. Each preset is one JSON file under
`<comfyui>/user/mec_presets/<id>.json` with:

    {
        "id":         "preset_xxx",
        "name":       "Face Detailer (SDXL)",
        "created":    1714672881.123,
        "thumbnail":  "data:image/png;base64,..."   (optional, ≤ 64 KB)
        "subgraph":   { "nodes": [...], "links": [...] }   ← LiteGraph slice
    }

Routes
------
GET    /mec/presets               → list (no subgraph payload — for gallery)
GET    /mec/presets/{id}          → full preset including subgraph + thumb
POST   /mec/presets               → create  body {name, subgraph, thumbnail?}
DELETE /mec/presets/{id}          → delete
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
import uuid
from typing import Any, Dict, List, Optional

log = logging.getLogger("MEC.group_presets")

_DIR_LOCK = threading.Lock()
_THUMB_MAX_BYTES = 64 * 1024  # 64 KB hard cap
_NAME_RE = re.compile(r"^[\w \-]{1,80}$")


def _presets_dir() -> str:
    """Resolve the preset directory under ComfyUI's user folder."""
    try:
        import folder_paths  # type: ignore
        base = getattr(folder_paths, "get_user_directory", None)
        if callable(base):
            root = base()
        else:
            root = os.path.join(folder_paths.base_path, "user")
    except Exception:
        root = os.path.join(os.getcwd(), "user")
    d = os.path.join(root, "mec_presets")
    with _DIR_LOCK:
        os.makedirs(d, exist_ok=True)
    return d


def _path_for(preset_id: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_\-]", "", preset_id)
    if not safe:
        raise ValueError("invalid preset id")
    return os.path.join(_presets_dir(), f"{safe}.json")


def _read(preset_id: str) -> Optional[Dict[str, Any]]:
    p = _path_for(preset_id)
    if not os.path.isfile(p):
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        log.warning("[group_presets] failed to read %s: %s", p, e)
        return None


def _write(data: Dict[str, Any]) -> str:
    pid = data["id"]
    p = _path_for(pid)
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, p)
    return p


def _list_all() -> List[Dict[str, Any]]:
    d = _presets_dir()
    out: List[Dict[str, Any]] = []
    try:
        for fn in sorted(os.listdir(d)):
            if not fn.endswith(".json"):
                continue
            p = os.path.join(d, fn)
            try:
                with open(p, "r", encoding="utf-8") as f:
                    obj = json.load(f)
                sub = obj.get("subgraph") or {}
                nodes_n = len(sub.get("nodes") or []) if isinstance(sub, dict) else 0
                out.append({
                    "id":         obj.get("id"),
                    "name":       obj.get("name") or obj.get("id"),
                    "created":    obj.get("created"),
                    "node_count": nodes_n,
                    "has_thumb":  bool(obj.get("thumbnail")),
                })
            except Exception as e:
                log.debug("[group_presets] skip %s: %s", fn, e)
    except FileNotFoundError:
        pass
    out.sort(key=lambda o: o.get("created") or 0, reverse=True)
    return out


def register_routes(server: Any) -> None:
    try:
        from aiohttp import web
    except Exception as e:
        log.warning("[group_presets] aiohttp unavailable: %s", e)
        return

    routes = server.routes

    @routes.get("/mec/presets")
    async def _list(_req: web.Request) -> web.Response:
        try:
            data = _list_all()
        except Exception as e:
            log.exception("[group_presets] list failed")
            return web.json_response(
                {"success": False, "error": "list_failed", "message": str(e)},
                status=500)
        return web.json_response({"success": True, "data": {"presets": data}})

    @routes.get("/mec/presets/{pid}")
    async def _get(req: web.Request) -> web.Response:
        pid = req.match_info.get("pid", "")
        obj = _read(pid)
        if obj is None:
            return web.json_response(
                {"success": False, "error": "not_found"}, status=404)
        return web.json_response({"success": True, "data": obj})

    @routes.post("/mec/presets")
    async def _create(req: web.Request) -> web.Response:
        try:
            body = await req.json()
        except Exception:
            body = {}
        if not isinstance(body, dict):
            return web.json_response(
                {"success": False, "error": "bad_body"}, status=400)

        name = (body.get("name") or "").strip()
        subgraph = body.get("subgraph")
        thumb = body.get("thumbnail")

        if not _NAME_RE.match(name):
            return web.json_response(
                {"success": False, "error": "invalid_name",
                 "message": "Name must be 1–80 chars, [A-Za-z0-9_-space]."},
                status=400)
        if not isinstance(subgraph, dict) or not subgraph.get("nodes"):
            return web.json_response(
                {"success": False, "error": "invalid_subgraph"}, status=400)

        if isinstance(thumb, str):
            # Strip data: prefix safely; enforce size cap.
            if len(thumb) > _THUMB_MAX_BYTES:
                thumb = thumb[:_THUMB_MAX_BYTES]
        else:
            thumb = None

        pid = "preset_" + uuid.uuid4().hex[:10]
        record = {
            "id":        pid,
            "name":      name,
            "created":   time.time(),
            "thumbnail": thumb,
            "subgraph":  subgraph,
        }
        try:
            _write(record)
        except Exception as e:
            log.exception("[group_presets] write failed")
            return web.json_response(
                {"success": False, "error": "write_failed", "message": str(e)},
                status=500)
        return web.json_response({
            "success": True,
            "data": {"id": pid, "name": name, "created": record["created"]},
        })

    @routes.delete("/mec/presets/{pid}")
    async def _delete(req: web.Request) -> web.Response:
        pid = req.match_info.get("pid", "")
        try:
            p = _path_for(pid)
        except Exception:
            return web.json_response(
                {"success": False, "error": "bad_id"}, status=400)
        if not os.path.isfile(p):
            return web.json_response(
                {"success": False, "error": "not_found"}, status=404)
        try:
            os.remove(p)
        except Exception as e:
            log.exception("[group_presets] delete failed")
            return web.json_response(
                {"success": False, "error": "delete_failed", "message": str(e)},
                status=500)
        return web.json_response({"success": True, "data": {"id": pid}})

    log.info("[group_presets] Routes registered: /mec/presets[/{id}]")
