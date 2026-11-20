"""Node Identity HTTP routes (design docs/research/node_identity_design.md, section 4).

GET  /c2c/node_identity/state      -> policy, is_admin, may {rename, toggle, configure}, prefs (any client)
POST /c2c/node_identity/policy     -> admin only (the ComfyUI page on this machine), validated, atomic write
POST /c2c/node_identity/authorize  -> {allowed, reason} for rename | restore | edit | toggle | configure

The admin check is nodes/_local_ui_guard.is_local (fail closed). POST bodies must be application/json: a
cross-origin page cannot send that without a CORS preflight ComfyUI does not grant.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Tuple

from . import policy as P

LOG = logging.getLogger("C2C.NodeIdentity")


def _is_admin(request) -> bool:
    try:
        from .._local_ui_guard import is_local
    except Exception:  # noqa: BLE001 - no guard, no admin (fail closed)
        return False
    try:
        return bool(is_local(request))
    except Exception:  # noqa: BLE001
        return False


def _user(request) -> str:
    user = request.headers.get("comfy-user") or "default"
    return user if user.replace("-", "").replace("_", "").isalnum() else "default"


def handle_state(is_admin: bool, user: str) -> Tuple[int, Dict[str, Any]]:
    return 200, P.effective_state(is_admin, user)


def handle_policy(is_admin: bool, body: Any) -> Tuple[int, Dict[str, Any]]:
    if not is_admin:
        return 403, {"error": "ADMIN_ONLY",
                     "message": "Only the admin (the ComfyUI page on this machine) can change the Node Identity rules."}
    try:
        pol = P.save_policy(body)
    except P.PolicyError as exc:
        return 400, {"error": "INVALID", "message": str(exc)}
    except OSError as exc:
        LOG.error("Node Identity: could not save the policy: %s", exc)
        return 500, {"error": "SAVE_FAILED", "message": f"Could not save the rules: {exc}"}
    return 200, {"policy": pol, "may": P.permissions(pol, True)}


def handle_authorize(is_admin: bool, body: Any) -> Tuple[int, Dict[str, Any]]:
    action = body.get("action") if isinstance(body, dict) else None
    if not isinstance(action, str):
        return 400, {"error": "INVALID", "message": "action is required"}
    allowed, reason = P.authorize(P.load_policy(), is_admin, action)
    return 200, {"allowed": allowed, "reason": reason, "action": action}


async def _json_body(request):
    ctype = (request.headers.get("Content-Type") or "").split(";")[0].strip().lower()
    if ctype != "application/json":
        return None, (415, {"error": "JSON_ONLY", "message": "Send the body as application/json."})
    try:
        return await request.json(), None
    except Exception:  # noqa: BLE001
        return None, (400, {"error": "INVALID", "message": "The body is not valid JSON."})


def register_routes(server) -> None:
    from aiohttp import web

    routes = server.routes

    @routes.get("/c2c/node_identity/state")
    async def _state(request):  # noqa: ANN001
        status, out = handle_state(_is_admin(request), _user(request))
        return web.json_response(out, status=status)

    @routes.post("/c2c/node_identity/policy")
    async def _policy(request):  # noqa: ANN001
        admin = _is_admin(request)
        if not admin:                                   # refuse before reading the body
            status, out = handle_policy(False, None)
            return web.json_response(out, status=status)
        body, err = await _json_body(request)
        if err:
            return web.json_response(err[1], status=err[0])
        status, out = handle_policy(admin, body)
        return web.json_response(out, status=status)

    @routes.post("/c2c/node_identity/authorize")
    async def _authorize(request):  # noqa: ANN001
        body, err = await _json_body(request)
        if err:
            return web.json_response(err[1], status=err[0])
        status, out = handle_authorize(_is_admin(request), body)
        return web.json_response(out, status=status)
