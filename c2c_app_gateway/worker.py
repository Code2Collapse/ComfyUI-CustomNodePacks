"""Async client for the PRIVATE ComfyUI worker (core HTTP API, verified against core 0.39 server.py).

Only the gateway talks to the worker. Raw worker answers (node ids, class names, tracebacks, paths) are never returned
to a runner: jobs.py projects them.
"""
from __future__ import annotations

import asyncio
import json
import logging
import uuid

import aiohttp

log = logging.getLogger("c2c_app_gateway.worker")


class WorkerError(RuntimeError):
    """The worker refused or failed a request. ``private`` holds its raw answer (owner-only)."""

    def __init__(self, msg: str, private=None, status: int | None = None):
        super().__init__(msg)
        self.private, self.status = private, status


class Worker:
    def __init__(self, base_url: str, *, headers: dict | None = None, timeout_s: float = 60.0):
        self.base = base_url.rstrip("/")
        self.client_id = "c2cgw-" + uuid.uuid4().hex
        self._headers = dict(headers or {})
        self._timeout = aiohttp.ClientTimeout(total=timeout_s)
        self._session: aiohttp.ClientSession | None = None
        self._ws_task: asyncio.Task | None = None
        self._listeners: list = []

    async def session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=self._timeout, headers=self._headers)
        return self._session

    async def close(self) -> None:
        if self._ws_task:
            self._ws_task.cancel()
            try:
                await self._ws_task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        if self._session and not self._session.closed:
            await self._session.close()

    async def _json(self, method: str, path: str, **kw):
        s = await self.session()
        async with s.request(method, self.base + path, **kw) as r:
            text = await r.text()
            try:
                data = json.loads(text) if text else None
            except json.JSONDecodeError:
                data = text
            if r.status >= 400:
                raise WorkerError(f"worker answered {r.status}", private=data, status=r.status)
            return data

    # ── core routes ─────────────────────────────────────────────────────────────────────────────────────────
    async def submit(self, prompt: dict) -> str:
        data = await self._json("POST", "/prompt", json={"prompt": prompt, "client_id": self.client_id})
        if not isinstance(data, dict) or not data.get("prompt_id"):
            raise WorkerError("worker did not return a prompt id", private=data)
        if data.get("node_errors"):
            raise WorkerError("worker rejected the prompt", private=data)
        return data["prompt_id"]

    async def history(self, prompt_id: str) -> dict | None:
        data = await self._json("GET", f"/history/{prompt_id}")
        return (data or {}).get(prompt_id) if isinstance(data, dict) else None

    async def queue(self) -> tuple[list[str], list[str]]:
        data = await self._json("GET", "/queue") or {}
        run = [x[1] for x in data.get("queue_running", []) if isinstance(x, list) and len(x) > 1]
        pend = [x[1] for x in data.get("queue_pending", []) if isinstance(x, list) and len(x) > 1]
        return run, pend

    async def delete_queued(self, prompt_id: str) -> None:
        await self._json("POST", "/queue", json={"delete": [prompt_id]})

    async def interrupt(self, prompt_id: str) -> None:
        """Targeted interrupt: core only interrupts when ``prompt_id`` is the running prompt."""
        await self._json("POST", "/interrupt", json={"prompt_id": prompt_id})

    async def upload_image(self, data: bytes, filename: str, subfolder: str = "c2c_gateway") -> str:
        form = aiohttp.FormData()
        form.add_field("image", data, filename=filename, content_type="application/octet-stream")
        form.add_field("type", "input")
        form.add_field("subfolder", subfolder)
        form.add_field("overwrite", "true")
        r = await self._json("POST", "/upload/image", data=form)
        name, sub = r.get("name"), r.get("subfolder") or ""
        return f"{sub}/{name}" if sub else name

    async def view(self, item: dict, max_bytes: int) -> bytes:
        params = {"filename": item.get("filename", ""), "subfolder": item.get("subfolder", ""),
                  "type": item.get("type", "output")}
        s = await self.session()
        async with s.get(self.base + "/view", params=params) as r:
            if r.status != 200:
                raise WorkerError(f"worker /view answered {r.status}", status=r.status)
            buf = bytearray()
            async for chunk in r.content.iter_chunked(1 << 16):
                buf += chunk
                if len(buf) > max_bytes:
                    raise WorkerError("output larger than the gateway's output limit")
            return bytes(buf)

    # ── progress tap ────────────────────────────────────────────────────────────────────────────────────────
    def add_listener(self, fn) -> None:
        self._listeners.append(fn)

    def start_ws(self) -> None:
        if self._ws_task is None or self._ws_task.done():
            self._ws_task = asyncio.ensure_future(self._ws_loop())

    async def _ws_loop(self) -> None:
        url = self.base.replace("http", "ws", 1) + f"/ws?clientId={self.client_id}"
        delay = 1.0
        while True:
            try:
                s = await self.session()
                async with s.ws_connect(url, heartbeat=30) as ws:
                    delay = 1.0
                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            try:
                                ev = json.loads(msg.data)
                            except json.JSONDecodeError:
                                continue
                            for fn in list(self._listeners):
                                try:
                                    fn(ev)
                                except Exception:  # noqa: BLE001 - one bad listener never stops the tap
                                    log.exception("listener failed")
                        elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                            break
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - worker down: retry with backoff, polling covers the gap
                log.debug("worker ws: %s", exc)
            await asyncio.sleep(delay)
            delay = min(delay * 2, 30.0)
