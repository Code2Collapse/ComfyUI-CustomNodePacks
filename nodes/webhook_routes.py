"""
Webhooks — trigger a stored workflow from outside, and notify URLs when runs finish (rebuilt L9.39).

Inbound:  POST /c2c/webhook/trigger/{id}  queues the workflow stored under {id} on this server. Optional JSON body
          {"inputs": {"<node id>": {"<input>": value}}} changes literal inputs for that run (links are never
          rewired). With a secret, the caller signs the raw body: X-Hub-Signature-256: sha256=<hex HMAC-SHA256>.
Outbound: POSTs {"event", "data", "ts", "history_url"} to each endpoint for the events it subscribed to, signed with
          X-C2C-Signature when it has a secret.

Routes:
  GET  /c2c/webhook/list            → inbound webhooks (id, label, trigger path, has_secret)
  POST /c2c/webhook/register        → {id, label, workflow_json (API prompt) | workflow (API JSON path), secret, reseed}
  POST /c2c/webhook/unregister      → {id}
  POST /c2c/webhook/trigger/{id}    → queue it
  GET  /c2c/webhook/outbound/list   → endpoints (id, url, events, has_secret)
  POST /c2c/webhook/outbound/add    → {id, url, events[], secret}
  POST /c2c/webhook/outbound/remove → {id}
  POST /c2c/webhook/outbound/test   → {id}: send a "test" event now
  GET  /c2c/webhook/log             → last 100 calls

Fixed in L9.39 (the module was never registered, and):
- it wrapped PromptServer.send_json at import and spawned a task per progress event with no endpoint configured;
  the wrapper is installed with the FIRST outbound endpoint and does nothing while there are none;
- the default "execution_complete" event does not exist in ComfyUI (it sends execution_success): it is an alias now;
- triggers posted to a hard-coded 127.0.0.1:8188; they queue on this server;
- secrets were kept in plain memory dicts; they go to the C2C secrets vault (memory only without a backend, said so);
- registrations survive a restart (user/default/c2c/automation.json, secrets never in it).
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import time
from collections import deque

from . import _c2c_automation as A

log = logging.getLogger("C2C.Webhook")

_inbound: dict[str, dict] = {}
_outbound: dict[str, dict] = {}
_log: deque[dict] = deque(maxlen=100)
_server = None
_hook_installed = False

EVENTS = ("execution_start", "execution_success", "execution_error", "execution_interrupted", "executed",
          "executing", "progress", "status")
DEFAULT_EVENTS = ["execution_success", "execution_error", "execution_interrupted"]
EVENT_ALIASES = {"execution_complete": "execution_success", "complete": "execution_success",
                 "error": "execution_error", "interrupted": "execution_interrupted"}
_SEC_IN, _SEC_OUT = "webhook_in.", "webhook_out."


def _log_call(direction: str, id_: str, event: str, status: str, msg: str = "") -> None:
    _log.append({"direction": direction, "id": id_, "event": event, "status": status, "msg": msg[:300],
                 "ts": time.time()})


def sign(secret: str, body: bytes) -> str:
    return "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


def verify(secret: str, body: bytes, header: str) -> bool:
    if not secret:
        return True
    return hmac.compare_digest(sign(secret, body), header or "")


def normalise_events(events) -> list[str]:
    if not events:
        return list(DEFAULT_EVENTS)
    if isinstance(events, str):
        events = [e for e in events.replace(";", ",").split(",")]
    out = []
    for e in events:
        e = EVENT_ALIASES.get(str(e).strip(), str(e).strip())
        if e not in EVENTS:
            raise ValueError(f"Unknown event '{e}' (one of: {', '.join(EVENTS)}).")
        if e not in out:
            out.append(e)
    return out


# ── outbound ─────────────────────────────────────────────────────────────────

async def _post(oid: str, ep: dict, event: str, data) -> None:
    import aiohttp
    payload = {"event": event, "data": data, "ts": time.time()}
    if isinstance(data, dict) and data.get("prompt_id"):
        payload["history_url"] = f"{A.own_base_url(_server)}/history/{data['prompt_id']}"
    body = json.dumps(payload, default=str).encode()
    headers = {"Content-Type": "application/json", "User-Agent": "C2C-Webhook"}
    secret = A.get_secret(_SEC_OUT + oid)
    if secret:
        headers["X-C2C-Signature"] = sign(secret, body)
    try:
        async with aiohttp.ClientSession() as sess:
            async with sess.post(ep["url"], data=body, headers=headers, timeout=aiohttp.ClientTimeout(total=10)) as r:
                _log_call("outbound", oid, event, str(r.status))
    except Exception as exc:  # noqa: BLE001
        _log_call("outbound", oid, event, "error", str(exc))
        log.warning("Webhook outbound %s failed: %s", ep["url"], exc)


def dispatch(event: str, data) -> int:
    """Schedule a POST for every endpoint subscribed to `event`; returns how many."""
    n = 0
    for oid, ep in list(_outbound.items()):
        if event in ep.get("events", DEFAULT_EVENTS):
            asyncio.ensure_future(_post(oid, ep, event, data))
            n += 1
    return n


def install_hook(server) -> bool:
    """Wrap server.send_json once; the wrapper is a no-op while no outbound endpoint exists."""
    global _hook_installed
    if _hook_installed or server is None:
        return _hook_installed
    orig = server.send_json

    async def _c2c_webhook_send_json(event, data, *args, **kwargs):
        await orig(event, data, *args, **kwargs)
        if _outbound and isinstance(event, str):
            try:
                dispatch(event, data)
            except Exception as exc:  # noqa: BLE001 - never break the websocket path
                log.debug("webhook dispatch failed: %s", exc)

    _c2c_webhook_send_json.__wrapped__ = orig
    server.send_json = _c2c_webhook_send_json
    _hook_installed = True
    log.info("Webhook: execution events hooked (first outbound endpoint).")
    return True


# ── persistence ──────────────────────────────────────────────────────────────

def _persist() -> None:
    try:
        A.save_section("webhooks_in", [{k: v for k, v in w.items()} for w in _inbound.values()])
        A.save_section("webhooks_out", [{k: v for k, v in e.items()} for e in _outbound.values()])
    except Exception as exc:  # noqa: BLE001
        log.warning("Webhook: could not save: %s", exc)


def _load() -> None:
    for w in A.load_section("webhooks_in"):
        if isinstance(w, dict) and w.get("id"):
            _inbound[str(w["id"])] = w
    for e in A.load_section("webhooks_out"):
        if isinstance(e, dict) and e.get("id") and e.get("url"):
            _outbound[str(e["id"])] = e


def _prompt_for(w: dict) -> dict:
    prompt = A.load_api_workflow(w["workflow"]) if w.get("workflow") else A.parse_api_workflow(
        w.get("workflow_json") or {})
    return A.reseed(prompt) if w.get("reseed", True) else prompt


# ── routes ────────────────────────────────────────────────────────────────────

def register_routes(server) -> None:
    global _server
    from aiohttp import web

    _server = server
    _load()
    if _outbound:
        install_hook(server)

    def err(msg: str, status: int = 400):
        return web.json_response({"error": msg}, status=status)

    async def body_of(request) -> dict:
        try:
            b = await request.json()
            return b if isinstance(b, dict) else {}
        except Exception:  # noqa: BLE001
            return {}

    def clean_id(v) -> str:
        s = "".join(ch for ch in str(v or "").strip() if ch.isalnum() or ch in "-_")
        return s[:64]

    @server.routes.get("/c2c/webhook/list")
    async def wh_list(_request: web.Request) -> web.Response:
        out = [{"id": k, "label": v.get("label") or k, "trigger": f"/c2c/webhook/trigger/{k}",
                "url": f"{A.own_base_url(_server)}/c2c/webhook/trigger/{k}",
                "source": "file" if v.get("workflow") else "current workflow", "reseed": v.get("reseed", True),
                "has_secret": bool(A.get_secret(_SEC_IN + k))} for k, v in _inbound.items()]
        return web.json_response({"webhooks": out})

    @server.routes.post("/c2c/webhook/register")
    async def wh_register(request: web.Request) -> web.Response:
        body = await body_of(request)
        wid = clean_id(body.get("id")) or __import__("uuid").uuid4().hex[:8]
        path = str(body.get("workflow") or "").strip().strip('"')
        try:
            if body.get("workflow_json"):
                wf, path = A.parse_api_workflow(body["workflow_json"]), ""
            elif path:
                A.load_api_workflow(path)
                wf = None
            else:
                return err("Pick a workflow: an API-format JSON file path, or 'Use current workflow'.")
        except ValueError as exc:
            return err(str(exc))
        _inbound[wid] = {"id": wid, "label": str(body.get("label") or "").strip() or wid, "workflow": path,
                         "workflow_json": wf, "reseed": bool(body.get("reseed", True))}
        where = A.put_secret(_SEC_IN + wid, str(body.get("secret") or ""))
        _persist()
        return web.json_response({"status": "registered", "id": wid, "trigger": f"/c2c/webhook/trigger/{wid}",
                                  "secret_storage": where})

    @server.routes.post("/c2c/webhook/unregister")
    async def wh_unregister(request: web.Request) -> web.Response:
        wid = str((await body_of(request)).get("id", ""))
        if wid not in _inbound:
            return err("not found", 404)
        del _inbound[wid]
        A.drop_secret(_SEC_IN + wid)
        _persist()
        return web.json_response({"status": "unregistered"})

    @server.routes.post("/c2c/webhook/trigger/{wid}")
    async def wh_trigger(request: web.Request) -> web.Response:
        wid = request.match_info["wid"]
        w = _inbound.get(wid)
        if w is None:
            return err("unknown webhook id", 404)
        raw = await request.read()
        if not verify(A.get_secret(_SEC_IN + wid), raw, request.headers.get("X-Hub-Signature-256", "")):
            _log_call("inbound", wid, "trigger", "unauthorized")
            return err("invalid signature", 401)
        try:
            body = json.loads(raw) if raw.strip() else {}
        except json.JSONDecodeError:
            body = {}
        try:
            prompt, ignored = A.apply_overrides(_prompt_for(w), (body or {}).get("inputs") if isinstance(body, dict)
                                                else None)
        except ValueError as exc:
            _log_call("inbound", wid, "trigger", "error", str(exc))
            return err(str(exc))
        res = await A.queue_prompt(prompt, client_id=f"c2c-webhook-{wid}", server=_server)
        _log_call("inbound", wid, "trigger", "queued" if res["ok"] else "error", res.get("error") or "")
        if not res["ok"]:
            return err(res["error"])
        return web.json_response({"status": "queued", "prompt_id": res["prompt_id"], "ignored_inputs": ignored})

    @server.routes.get("/c2c/webhook/outbound/list")
    async def wh_out_list(_request: web.Request) -> web.Response:
        out = [{"id": k, "url": v["url"], "events": v.get("events", DEFAULT_EVENTS),
                "has_secret": bool(A.get_secret(_SEC_OUT + k))} for k, v in _outbound.items()]
        return web.json_response({"outbound": out, "events": list(EVENTS), "hooked": _hook_installed})

    @server.routes.post("/c2c/webhook/outbound/add")
    async def wh_out_add(request: web.Request) -> web.Response:
        body = await body_of(request)
        url = str(body.get("url") or "").strip()
        if not url.lower().startswith(("http://", "https://")):
            return err("The URL must start with http:// or https://")
        try:
            events = normalise_events(body.get("events"))
        except ValueError as exc:
            return err(str(exc))
        oid = clean_id(body.get("id")) or __import__("uuid").uuid4().hex[:8]
        _outbound[oid] = {"id": oid, "url": url, "events": events}
        where = A.put_secret(_SEC_OUT + oid, str(body.get("secret") or ""))
        install_hook(_server)
        _persist()
        return web.json_response({"status": "added", "id": oid, "events": events, "secret_storage": where})

    @server.routes.post("/c2c/webhook/outbound/remove")
    async def wh_out_remove(request: web.Request) -> web.Response:
        oid = str((await body_of(request)).get("id", ""))
        if oid not in _outbound:
            return err("not found", 404)
        del _outbound[oid]
        A.drop_secret(_SEC_OUT + oid)
        _persist()
        return web.json_response({"status": "removed"})

    @server.routes.post("/c2c/webhook/outbound/test")
    async def wh_out_test(request: web.Request) -> web.Response:
        oid = str((await body_of(request)).get("id", ""))
        ep = _outbound.get(oid)
        if ep is None:
            return err("not found", 404)
        await _post(oid, ep, "test", {"message": "C2C webhook test"})
        last = next((r for r in reversed(_log) if r["id"] == oid), {})
        return web.json_response({"status": last.get("status"), "msg": last.get("msg")})

    @server.routes.get("/c2c/webhook/log")
    async def wh_log(_request: web.Request) -> web.Response:
        return web.json_response({"log": list(_log)})

    log.info("Webhook routes registered (/c2c/webhook/*): %d inbound, %d outbound.", len(_inbound), len(_outbound))
