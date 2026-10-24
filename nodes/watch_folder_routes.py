"""
Watch Folder — react to new image / video files in a folder (rebuilt L9.39).

Routes:
  GET  /c2c/watchfolder/status        → watchers with state, files seen, pending, last error
  POST /c2c/watchfolder/add           → {path, action: notify|load_image|run_workflow, workflow_path | workflow_json,
                                         recursive}
  POST /c2c/watchfolder/remove        → {path}
  GET  /c2c/watchfolder/events        → last 200 events
  POST /c2c/watchfolder/clear_events  → clear the event log

Actions:
  notify        log the file (the panel lists it).
  load_image    copy it into input/c2c_watch/ and tell the browser (c2c:watchfolder:new_image) - the panel offers
                "Add Load Image" for it.
  run_workflow  put it into the first image / video loader of an API workflow and queue it on this server.

Fixed in L9.39 (the module was never registered, and):
- the scan (rglob every 2 s) ran ON the server event loop - a big folder froze every request; scans now run in a
  worker thread, top level only unless "recursive", capped at MAX_FILES entries;
- a file was fired the moment it appeared, half written; now only when its size and mtime are unchanged across two
  scans;
- "run workflow" turned a UI workflow into a fake prompt that could never validate, and posted to a hard-coded
  127.0.0.1:8188; it takes an API workflow (or the current graph) and queues on this server;
- "load image" sent an absolute path nobody listened for;
- watching ComfyUI's own output / temp folder with run_workflow would feed itself forever - refused;
- watchers survive a restart (user/default/c2c/automation.json).
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from collections import deque
from pathlib import Path

from . import _c2c_automation as A

log = logging.getLogger("C2C.WatchFolder")

_watchers: dict[str, dict] = {}            # normalised path -> watcher
_events: deque[dict] = deque(maxlen=200)
_server = None
POLL_INTERVAL = 2.0
MAX_FILES = 20000
ACTIONS = ("notify", "load_image", "run_workflow")
_ACTION_ALIAS = {"load_workflow": "run_workflow"}
WATCH_EXTS = A.IMAGE_EXTS | A.VIDEO_EXTS
_SKIP_SUFFIX = (".tmp", ".part", ".crdownload", ".partial", ".download")


def _key(path: str) -> str:
    return os.path.normcase(os.path.abspath(str(path).strip().strip('"')))


def _event(etype: str, path: str, watcher: str, msg: str = "", **extra) -> None:
    _events.append({"type": etype, "path": path, "watcher": watcher, "msg": msg, "timestamp": time.time(), **extra})


def _excluded_roots() -> list[str]:
    """input/c2c_watch (our own copies) is never scanned."""
    try:
        import folder_paths  # type: ignore
        return [_key(os.path.join(folder_paths.get_input_directory(), A.WATCH_SUBDIR))]
    except Exception:  # noqa: BLE001
        return []


def _feeds_itself(path: str, recursive: bool = False) -> str | None:
    """run_workflow on ComfyUI's output or temp folder (or a parent of it, recursively) would queue a run for every
    file it writes."""
    try:
        import folder_paths  # type: ignore
        roots = {"output": folder_paths.get_output_directory(), "temp": folder_paths.get_temp_directory()}
    except Exception:  # noqa: BLE001
        return None
    p = _key(path)
    for name, root in roots.items():
        r = _key(root)
        if p == r or p.startswith(r + os.sep) or (recursive and r.startswith(p + os.sep)):
            return name
    return None


def scan(root: str, recursive: bool = False, cap: int = MAX_FILES, skip: list[str] | None = None):
    """{path: (size, mtime_ns)} of image / video files; (files, truncated). Runs in a worker thread."""
    files: dict[str, tuple[int, int]] = {}
    skip = skip or []
    stack = [root]
    while stack:
        d = stack.pop()
        if any(_key(d) == s or _key(d).startswith(s + os.sep) for s in skip):
            continue
        try:
            with os.scandir(d) as it:
                for e in it:
                    name = e.name
                    if name.startswith((".", "~")) or name.lower().endswith(_SKIP_SUFFIX):
                        continue
                    try:
                        if e.is_dir(follow_symlinks=False):
                            if recursive:
                                stack.append(e.path)
                            continue
                        if os.path.splitext(name)[1].lower() not in WATCH_EXTS:
                            continue
                        st = e.stat()
                    except OSError:
                        continue
                    files[e.path] = (st.st_size, st.st_mtime_ns)
                    if len(files) >= cap:
                        return files, True
        except OSError:
            continue
    return files, False


class Settle:
    """A file is ready when it has bytes and the same (size, mtime) on two scans in a row."""

    def __init__(self, initial: dict):
        self.known = set(initial)
        self.pending: dict[str, tuple[int, int]] = {}

    def update(self, current: dict) -> list[str]:
        ready = []
        for p, sig in current.items():
            if p in self.known:
                continue
            if sig[0] > 0 and self.pending.get(p) == sig:
                self.pending.pop(p, None)
                self.known.add(p)
                ready.append(p)
            else:
                self.pending[p] = sig
        for p in list(self.pending):
            if p not in current:
                self.pending.pop(p)
        self.known &= set(current)          # a deleted then re-written file fires again
        return sorted(ready)


async def _handle(w: dict, path: str) -> None:
    _event("new_file", path, w["path"])
    w["files_seen"] = w.get("files_seen", 0) + 1
    action = w["action"]
    if action == "notify":
        return
    kind = A.file_kind(path)
    try:
        if action == "load_image":
            name = await asyncio.to_thread(A.copy_to_input, path)
            _event("copied", path, w["path"], f"input/{name}", name=name, kind=kind)
            if _server is not None:
                await _server.send_json("c2c:watchfolder:new_image",
                                        {"name": name, "path": path, "kind": kind, "watcher": w["path"]})
            return
        prompt = (A.load_api_workflow(w["workflow_path"]) if w.get("workflow_path")
                  else A.parse_api_workflow(w.get("workflow_json") or {}))
        prompt, target = await asyncio.to_thread(A.inject_file, prompt, path)
        res = await A.queue_prompt(prompt, client_id="c2c-watchfolder", server=_server)
        if res["ok"]:
            _event("queued", path, w["path"], f"node {target}", prompt_id=res["prompt_id"])
            w["last_error"] = None
        else:
            _event("error", path, w["path"], res["error"])
            w["last_error"] = res["error"]
    except Exception as exc:  # noqa: BLE001 - a bad file never stops the watcher
        _event("error", path, w["path"], str(exc))
        w["last_error"] = str(exc)


async def _watch(key: str) -> None:
    w = _watchers.get(key)
    if w is None:
        return
    skip = _excluded_roots()
    settle = None
    warned = False
    try:
        while _watchers.get(key) is w:
            if not os.path.isdir(w["path"]):
                w["state"] = "folder missing"
                settle = None
                await asyncio.sleep(POLL_INTERVAL * 5)
                continue
            files, truncated = await asyncio.to_thread(scan, w["path"], w.get("recursive", False), MAX_FILES, skip)
            if truncated and not warned:
                warned = True
                _event("too_many_files", w["path"], w["path"], f"only the first {MAX_FILES} files are watched")
            if settle is None:
                settle = Settle(files)             # files already there are not events
                w["state"] = "watching"
            else:
                for p in settle.update(files):
                    await _handle(w, p)
            w["pending"] = len(settle.pending)
            await asyncio.sleep(POLL_INTERVAL)
    except asyncio.CancelledError:
        pass
    finally:
        if _watchers.get(key) is w:
            w["state"] = "stopped"


def _start(key: str) -> None:
    w = _watchers[key]
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        srv_loop = getattr(_server, "loop", None)
        if srv_loop is not None:
            srv_loop.call_soon_threadsafe(_start, key)
        return
    w["task"] = loop.create_task(_watch(key))


def _persist() -> None:
    keep = ("path", "action", "workflow_path", "workflow_json", "recursive")
    try:
        A.save_section("watchers", [{k: w.get(k) for k in keep} for w in _watchers.values()])
    except Exception as exc:  # noqa: BLE001
        log.warning("WatchFolder: could not save watchers: %s", exc)


def validate(body: dict) -> dict:
    """The watcher spec from a request body, or ValueError with the reason."""
    path = str(body.get("path") or "").strip().strip('"')
    if not path:
        raise ValueError("Folder path is required.")
    if not os.path.isdir(path):
        raise ValueError(f"Folder not found: {path}")
    action = _ACTION_ALIAS.get(body.get("action"), body.get("action") or "notify")
    if action not in ACTIONS:
        raise ValueError(f"Unknown action '{action}' (one of: {', '.join(ACTIONS)}).")
    spec = {"path": os.path.abspath(path), "action": action, "recursive": bool(body.get("recursive", False)),
            "workflow_path": None, "workflow_json": None}
    if action == "run_workflow":
        own = _feeds_itself(path, spec["recursive"])
        if own:
            raise ValueError(f"This is ComfyUI's {own} folder: every file the workflow writes would queue it again.")
        wf_path = str(body.get("workflow_path") or "").strip().strip('"')
        if body.get("workflow_json"):
            spec["workflow_json"] = A.parse_api_workflow(body["workflow_json"])
        elif wf_path:
            A.load_api_workflow(wf_path)
            spec["workflow_path"] = wf_path
        else:
            raise ValueError("Run workflow needs a workflow: an API-format JSON path, or 'Use current workflow'.")
    return spec


# ── route registration ────────────────────────────────────────────────────────

def register_routes(server) -> None:
    global _server
    from aiohttp import web

    _server = server
    for spec in A.load_section("watchers"):
        if isinstance(spec, dict) and spec.get("path"):
            k = _key(spec["path"])
            _watchers[k] = {**spec, "action": _ACTION_ALIAS.get(spec.get("action"), spec.get("action", "notify")),
                            "state": "starting", "files_seen": 0}
            _start(k)

    async def body_of(request) -> dict:
        try:
            b = await request.json()
            return b if isinstance(b, dict) else {}
        except Exception:  # noqa: BLE001
            return {}

    @server.routes.get("/c2c/watchfolder/status")
    async def wf_status(_request: web.Request) -> web.Response:
        out = []
        for w in _watchers.values():
            t = w.get("task")
            out.append({"path": w["path"], "action": w["action"], "recursive": w.get("recursive", False),
                        "workflow_path": w.get("workflow_path"),
                        "workflow": "file" if w.get("workflow_path") else (
                            "current workflow" if w.get("workflow_json") else None),
                        "running": bool(t) and not t.done(), "state": w.get("state"),
                        "files_seen": w.get("files_seen", 0), "pending": w.get("pending", 0),
                        "last_error": w.get("last_error")})
        return web.json_response({"watchers": out})

    @server.routes.post("/c2c/watchfolder/add")
    async def wf_add(request: web.Request) -> web.Response:
        try:
            spec = validate(await body_of(request))
        except ValueError as exc:
            return web.json_response({"error": str(exc)}, status=400)
        k = _key(spec["path"])
        if k in _watchers:
            return web.json_response({"status": "already_watching", "path": spec["path"]})
        _watchers[k] = {**spec, "state": "starting", "files_seen": 0}
        _start(k)
        _persist()
        return web.json_response({"status": "watching", "path": spec["path"]})

    @server.routes.post("/c2c/watchfolder/remove")
    async def wf_remove(request: web.Request) -> web.Response:
        k = _key((await body_of(request)).get("path", ""))
        w = _watchers.pop(k, None)
        if w is None:
            return web.json_response({"error": "not watching"}, status=404)
        t = w.get("task")
        if t and not t.done():
            t.cancel()
        _persist()
        return web.json_response({"status": "removed", "path": w["path"]})

    @server.routes.get("/c2c/watchfolder/events")
    async def wf_events(_request: web.Request) -> web.Response:
        return web.json_response({"events": list(_events)})

    @server.routes.post("/c2c/watchfolder/clear_events")
    async def wf_clear(_request: web.Request) -> web.Response:
        _events.clear()
        return web.json_response({"status": "cleared"})

    log.info("WatchFolder routes registered (/c2c/watchfolder/*), %d watcher(s) restored.", len(_watchers))
