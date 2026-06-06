"""
Webhook routes — P2.1
Inbound and outbound webhook support for ComfyUI automation.

Inbound:  POST /c2c/webhook/trigger/{id} → queue a pre-registered workflow
Outbound: fired automatically on queue complete / progress events

Routes:
  GET  /c2c/webhook/list           → list registered webhooks
  POST /c2c/webhook/register       → register {id, workflow_json, secret}
  POST /c2c/webhook/unregister     → unregister {id}
  POST /c2c/webhook/trigger/{id}   → trigger registered workflow
  GET  /c2c/webhook/outbound/list  → list outbound endpoints
  POST /c2c/webhook/outbound/add   → add outbound {id, url, events[], secret}
  POST /c2c/webhook/outbound/remove→ remove outbound {id}
  GET  /c2c/webhook/log            → last 100 call records
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import time
from collections import deque

log = logging.getLogger("C2C.Webhook")

_inbound: dict[str, dict]  = {}    # id → {workflow_json, secret}
_outbound: dict[str, dict] = {}    # id → {url, events, secret}
_log: deque[dict] = deque(maxlen=100)


def _log_call(direction: str, id_: str, event: str, status: str, msg: str = "") -> None:
    _log.append({
        "direction": direction,
        "id"       : id_,
        "event"    : event,
        "status"   : status,
        "msg"      : msg,
        "ts"       : time.time(),
    })


def _verify_hmac(secret: str, body: bytes, sig_header: str) -> bool:
    if not secret:
        return True  # no secret = no verification
    expected = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(f"sha256={expected}", sig_header or "")


# ── outbound dispatch ─────────────────────────────────────────────────────────

async def _dispatch_outbound(event: str, payload: dict) -> None:
    import aiohttp
    for oid, ep in list(_outbound.items()):
        if event not in ep.get("events", [event]):
            continue
        body = json.dumps({"event": event, "data": payload}).encode()
        secret = ep.get("secret", "")
        sig = "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest() if secret else ""
        headers = {"Content-Type": "application/json", "X-C2C-Signature": sig}
        try:
            async with aiohttp.ClientSession() as sess:
                async with sess.post(
                    ep["url"], data=body, headers=headers,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as r:
                    _log_call("outbound", oid, event, str(r.status))
        except Exception as exc:
            _log_call("outbound", oid, event, "error", str(exc))
            log.warning("Webhook outbound %s failed: %s", ep["url"], exc)


def install_prompt_hooks() -> None:
    """Hook into ComfyUI's execution lifecycle to fire outbound webhooks."""
    try:
        import server
        srv = server.PromptServer.instance

        # Intercept send_json to capture execution events
        _orig_send = srv.send_json

        async def _hooked_send(event: str, data: dict, *args, **kwargs) -> None:
            await _orig_send(event, data, *args, **kwargs)
            if event in ("execution_complete", "executing", "progress", "executed", "execution_error"):
                asyncio.create_task(_dispatch_outbound(event, data))

        srv.send_json = _hooked_send
        log.info("Webhook: PromptServer hooks installed.")
    except Exception as exc:
        log.warning("Webhook: could not install PromptServer hooks: %s", exc)


# ── routes ────────────────────────────────────────────────────────────────────

def register_routes(server) -> None:
    from aiohttp import web

    install_prompt_hooks()

    @server.routes.get("/c2c/webhook/list")
    async def wh_list(_request: web.Request) -> web.Response:
        result = [{"id": k, "has_secret": bool(v.get("secret"))} for k, v in _inbound.items()]
        return web.json_response({"webhooks": result})

    @server.routes.post("/c2c/webhook/register")
    async def wh_register(request: web.Request) -> web.Response:
        body = await request.json()
        wid = body.get("id", "").strip()
        if not wid:
            return web.json_response({"error": "id required"}, status=400)
        _inbound[wid] = {
            "workflow_json": body.get("workflow_json", {}),
            "secret"       : body.get("secret", ""),
        }
        return web.json_response({"status": "registered", "id": wid})

    @server.routes.post("/c2c/webhook/unregister")
    async def wh_unregister(request: web.Request) -> web.Response:
        body = await request.json()
        wid = body.get("id", "")
        if wid not in _inbound:
            return web.json_response({"error": "not found"}, status=404)
        del _inbound[wid]
        return web.json_response({"status": "unregistered"})

    @server.routes.post("/c2c/webhook/trigger/{wid}")
    async def wh_trigger(request: web.Request) -> web.Response:
        wid = request.match_info["wid"]
        if wid not in _inbound:
            return web.json_response({"error": "unknown webhook id"}, status=404)

        ep = _inbound[wid]
        raw_body = await request.read()
        sig = request.headers.get("X-Hub-Signature-256", "")
        if not _verify_hmac(ep.get("secret", ""), raw_body, sig):
            _log_call("inbound", wid, "trigger", "unauthorized")
            return web.json_response({"error": "invalid signature"}, status=401)

        wf = ep.get("workflow_json", {})
        if not wf:
            _log_call("inbound", wid, "trigger", "error", "no workflow")
            return web.json_response({"error": "no workflow configured"}, status=400)

        import aiohttp
        try:
            async with aiohttp.ClientSession() as sess:
                async with sess.post(
                    "http://127.0.0.1:8188/prompt",
                    json={"prompt": wf},
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as r:
                    _log_call("inbound", wid, "trigger", str(r.status))
                    return web.json_response({"status": "queued" if r.status in (200, 201) else "error", "http": r.status})
        except Exception as exc:
            _log_call("inbound", wid, "trigger", "error", str(exc))
            return web.json_response({"error": str(exc)}, status=500)

    @server.routes.get("/c2c/webhook/outbound/list")
    async def wh_out_list(_request: web.Request) -> web.Response:
        result = [{"id": k, "url": v["url"], "events": v["events"]} for k, v in _outbound.items()]
        return web.json_response({"outbound": result})

    @server.routes.post("/c2c/webhook/outbound/add")
    async def wh_out_add(request: web.Request) -> web.Response:
        body = await request.json()
        oid = body.get("id", "").strip()
        url = body.get("url", "").strip()
        if not oid or not url:
            return web.json_response({"error": "id and url required"}, status=400)
        _outbound[oid] = {
            "url"   : url,
            "events": body.get("events", ["execution_complete", "execution_error"]),
            "secret": body.get("secret", ""),
        }
        return web.json_response({"status": "added", "id": oid})

    @server.routes.post("/c2c/webhook/outbound/remove")
    async def wh_out_remove(request: web.Request) -> web.Response:
        body = await request.json()
        oid = body.get("id", "")
        if oid not in _outbound:
            return web.json_response({"error": "not found"}, status=404)
        del _outbound[oid]
        return web.json_response({"status": "removed"})

    @server.routes.get("/c2c/webhook/log")
    async def wh_log(_request: web.Request) -> web.Response:
        return web.json_response({"log": list(_log)})

    log.info("Webhook routes registered (/c2c/webhook/*).")
