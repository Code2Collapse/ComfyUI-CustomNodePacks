"""aiohttp routes for ImageMaskEditorC2C mask store.

PNG decode/encode and file I/O run in the default executor, never on the event loop: a 16-megapixel mask
takes hundreds of milliseconds to decode, and the event loop serves every other request ComfyUI has.
"""
from __future__ import annotations

import asyncio
import logging

from . import store

log = logging.getLogger("C2C.ImageMaskEditor")
_ROUTES_REGISTERED = False


def register_routes(server) -> None:
    global _ROUTES_REGISTERED
    if _ROUTES_REGISTERED:
        return
    try:
        from aiohttp import web
    except Exception:
        print("[C2C.ImageMaskEditor] aiohttp not available; routes disabled.")
        return

    routes = server.routes

    async def _off_loop(fn, *args):
        return await asyncio.get_running_loop().run_in_executor(None, fn, *args)

    def _bad(msg, status=400):
        return web.json_response({"error": str(msg)}, status=status)

    def _query(request, *names):
        q = request.rel_url.query
        return [(q.get(n) or "").strip() for n in names]

    @routes.post("/c2c/image_mask_editor/frame")
    async def post_frame(request):
        eid, frame_s = _query(request, "id", "frame")
        if not eid or not frame_s.isdigit():
            return _bad("id and frame required")
        try:
            data = await request.read()
            sha = await _off_loop(store.put_frame, eid, int(frame_s), data)
            return web.json_response({"ok": True, "sha": sha, "digest": await _off_loop(store.digest, eid)})
        except ValueError as e:
            return _bad(e)
        except Exception as e:  # noqa: BLE001 - disk full, permissions: report, do not crash the handler
            log.exception("saving mask frame failed")
            return _bad(f"could not save the mask: {e}", 500)

    @routes.get("/c2c/image_mask_editor/frame")
    async def get_frame_route(request):
        eid, frame_s = _query(request, "id", "frame")
        if not eid or not frame_s.isdigit():
            return _bad("id and frame required")
        body = await _off_loop(store.get_frame_png, eid, int(frame_s))
        if body is None:
            return web.Response(status=404)
        return web.Response(body=body, content_type="image/png", headers={"Cache-Control": "no-store"})

    @routes.delete("/c2c/image_mask_editor/frame")
    async def delete_frame_route(request):
        eid, frame_s = _query(request, "id", "frame")
        if not eid or not frame_s.isdigit():
            return _bad("id and frame required")
        try:
            await _off_loop(store.delete_frame, eid, int(frame_s))
            return web.json_response({"ok": True, "digest": await _off_loop(store.digest, eid)})
        except ValueError as e:
            return _bad(e)

    @routes.get("/c2c/image_mask_editor/state")
    async def get_state(request):
        (eid,) = _query(request, "id")
        if not eid:
            return _bad("id required")
        try:
            store.validate_editor_id(eid)
        except ValueError as e:
            return _bad(e)

        def _state():
            shape = store.get_shape(eid)
            return {"frames": store.list_frames(eid), "shape": list(shape) if shape else None,
                    "digest": store.digest(eid)}
        return web.json_response(await _off_loop(_state))

    @routes.post("/c2c/image_mask_editor/clear")
    async def clear_store(request):
        (eid,) = _query(request, "id")
        if not eid:
            return _bad("id required")
        try:
            await _off_loop(store.clear, eid)
            return web.json_response({"ok": True})
        except ValueError as e:
            return _bad(e)

    @routes.post("/c2c/image_mask_editor/copy")
    async def copy_store(request):
        src, dst = _query(request, "from", "to")
        if not src or not dst:
            return _bad("from and to required")
        try:
            await _off_loop(store.copy, src, dst)
            return web.json_response({"ok": True, "digest": await _off_loop(store.digest, dst)})
        except ValueError as e:
            return _bad(e)

    _ROUTES_REGISTERED = True
    print("[C2C.ImageMaskEditor] routes registered (/c2c/image_mask_editor/*)")
