"""
Watch Folder service — P2.1
Monitors user-specified folders for new image/video/workflow files.
Auto-imports to ComfyUI or triggers a configurable workflow.

Routes:
  GET  /c2c/watchfolder/status        → list active watchers
  POST /c2c/watchfolder/add           → add watcher {"path","action","workflow_path"}
  POST /c2c/watchfolder/remove        → remove watcher {"path"}
  GET  /c2c/watchfolder/events        → last 50 events
  POST /c2c/watchfolder/clear_events  → clear event log
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections import deque
from pathlib import Path
from typing import Any

log = logging.getLogger("C2C.WatchFolder")

# ── state ─────────────────────────────────────────────────────────────────────

_watchers: dict[str, dict] = {}          # path → {action, workflow_path, task}
_events: deque[dict] = deque(maxlen=200)
_POLL_INTERVAL = 2.0                     # seconds between fs scans

IMAGE_EXTS   = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}
VIDEO_EXTS   = {".mp4", ".mov", ".webm", ".avi", ".mkv"}
WORKFLOW_EXTS = {".json"}
ALL_EXTS     = IMAGE_EXTS | VIDEO_EXTS | WORKFLOW_EXTS


def _add_event(event_type: str, path: str, watcher_path: str) -> None:
    _events.append({
        "type"        : event_type,
        "path"        : path,
        "watcher"     : watcher_path,
        "timestamp"   : time.time(),
    })


# ── file watcher coroutine ────────────────────────────────────────────────────

async def _watch_loop(watch_path: str, action: str, workflow_path: str | None) -> None:
    root = Path(watch_path)
    if not root.exists():
        log.warning("WatchFolder: path does not exist: %s", watch_path)
        return

    seen: set[str] = set()
    # seed with existing files so we don't fire events for them
    for f in root.rglob("*"):
        if f.is_file() and f.suffix.lower() in ALL_EXTS:
            seen.add(str(f))

    log.info("WatchFolder: watching %s (action=%s)", watch_path, action)
    while watch_path in _watchers:
        try:
            current: set[str] = set()
            for f in root.rglob("*"):
                if f.is_file() and f.suffix.lower() in ALL_EXTS:
                    current.add(str(f))

            new_files = current - seen
            for fp in sorted(new_files):
                seen.add(fp)
                _add_event("new_file", fp, watch_path)
                log.info("WatchFolder: new file detected: %s", fp)
                if action == "load_workflow" and workflow_path:
                    await _trigger_workflow(fp, workflow_path)
                elif action == "notify":
                    pass  # event already logged above
                elif action == "load_image":
                    await _queue_load_image(fp)
        except Exception as exc:
            log.error("WatchFolder loop error: %s", exc)

        await asyncio.sleep(_POLL_INTERVAL)

    log.info("WatchFolder: stopped watching %s", watch_path)


async def _trigger_workflow(input_path: str, workflow_path: str) -> None:
    """Load a workflow JSON and inject the new file as the first LoadImage path."""
    try:
        wf = json.loads(Path(workflow_path).read_text(encoding="utf-8"))
        # find first LoadImage node and set its path widget
        for node in wf.get("nodes", []):
            if node.get("type") == "LoadImage":
                widgets = node.setdefault("widgets_values", [])
                if widgets:
                    widgets[0] = os.path.basename(input_path)
                else:
                    widgets.append(os.path.basename(input_path))
                break
        # POST to /prompt
        import aiohttp
        payload = {"prompt": {str(n["id"]): n for n in wf.get("nodes", [])}}
        async with aiohttp.ClientSession() as sess:
            async with sess.post(
                "http://127.0.0.1:8188/prompt",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as r:
                status = r.status
        log.info("WatchFolder: queued workflow for %s → HTTP %d", input_path, status)
    except Exception as exc:
        log.error("WatchFolder: workflow trigger failed: %s", exc)


async def _queue_load_image(input_path: str) -> None:
    """Notify frontend about new image so it can be loaded interactively."""
    try:
        import server as _srv
        srv = _srv.PromptServer.instance
        await srv.send_json(
            "c2c:watchfolder:new_image",
            {"path": input_path},
        )
    except Exception:
        pass


# ── route registration ────────────────────────────────────────────────────────

def register_routes(server) -> None:
    from aiohttp import web

    @server.routes.get("/c2c/watchfolder/status")
    async def wf_status(_request: web.Request) -> web.Response:
        result = []
        for path, info in _watchers.items():
            result.append({
                "path"          : path,
                "action"        : info.get("action"),
                "workflow_path" : info.get("workflow_path"),
                "running"       : not info["task"].done() if info.get("task") else False,
            })
        return web.json_response({"watchers": result})

    @server.routes.post("/c2c/watchfolder/add")
    async def wf_add(request: web.Request) -> web.Response:
        body = await request.json()
        path = body.get("path", "").strip()
        action = body.get("action", "notify")
        workflow_path = body.get("workflow_path") or None

        if not path:
            return web.json_response({"error": "path required"}, status=400)
        if path in _watchers:
            return web.json_response({"status": "already_watching", "path": path})

        task = asyncio.create_task(_watch_loop(path, action, workflow_path))
        _watchers[path] = {"action": action, "workflow_path": workflow_path, "task": task}
        return web.json_response({"status": "watching", "path": path})

    @server.routes.post("/c2c/watchfolder/remove")
    async def wf_remove(request: web.Request) -> web.Response:
        body = await request.json()
        path = body.get("path", "").strip()
        if path not in _watchers:
            return web.json_response({"error": "not watching"}, status=404)
        info = _watchers.pop(path)
        task = info.get("task")
        if task and not task.done():
            task.cancel()
        return web.json_response({"status": "removed", "path": path})

    @server.routes.get("/c2c/watchfolder/events")
    async def wf_events(_request: web.Request) -> web.Response:
        return web.json_response({"events": list(_events)})

    @server.routes.post("/c2c/watchfolder/clear_events")
    async def wf_clear(_request: web.Request) -> web.Response:
        _events.clear()
        return web.json_response({"status": "cleared"})

    log.info("WatchFolder routes registered (/c2c/watchfolder/*).")
