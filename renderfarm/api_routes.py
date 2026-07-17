"""C2C Farm dashboard REST routes (/c2c/farm/*) served by the local ComfyUI PromptServer.

Registered once at pack import (the Model-Browser lesson: routes that are
never registered are just 404s). All responses are JSON; role enforcement
(admin vs user) happens server-side from C2C_USER_NAME + users.json.
"""

from __future__ import annotations

import json
import logging
import os

log = logging.getLogger("C2C.Farm.api")
_registered = False


def register_routes():
    global _registered
    if _registered:
        return
    from aiohttp import web
    from server import PromptServer

    routes = PromptServer.instance.routes

    def _actor() -> dict:
        """Current local user (never raises — dashboard must render even
        when C2C_USER_NAME is unset; submits still fail loudly)."""
        from .user_config import get_user
        try:
            return get_user()
        except RuntimeError as exc:
            return {"name": os.environ.get("C2C_USER_NAME", ""), "role": "unset",
                    "error": str(exc), "projects": [], "max_concurrent_jobs": 0}

    def _qm():
        from .spooler.queue_manager import get_queue_manager
        return get_queue_manager()

    def _can_control(job_user: str) -> bool:
        from .user_config import can_control
        a = _actor()
        return bool(a.get("name")) and can_control(a["name"], job_user)

    @routes.get("/c2c/farm/user")
    async def farm_user(request):
        return web.json_response(_actor())

    @routes.get("/c2c/farm/jobs")
    async def farm_jobs(request):
        snap = _qm().snapshot()
        snap["actor"] = _actor()
        return web.json_response(snap)

    @routes.get("/c2c/farm/history")
    async def farm_history(request):
        from .logging.audit_db import get_audit_db
        q = request.rel_url.query
        rows = get_audit_db().query(
            user=q.get("user") or None, project=q.get("project") or None,
            status=q.get("status") or None, limit=int(q.get("limit", "200")))
        return web.json_response({"rows": rows})

    @routes.get("/c2c/farm/cluster")
    async def farm_cluster(request):
        import asyncio
        from .backends import get_adapter
        from .user_config import list_backends

        def probe(cfg):
            try:
                return get_adapter(cfg["name"]).capacity()
            except Exception as exc:  # noqa: BLE001
                return {"backend": cfg.get("name"), "reachable": False,
                        "error": str(exc)[:200]}

        cfgs = list_backends(enabled_only=False)
        results = await asyncio.gather(
            *(asyncio.to_thread(probe, c) for c in cfgs if c.get("enabled")),
            return_exceptions=False)
        disabled = [{"backend": c.get("name"), "reachable": False, "disabled": True}
                    for c in cfgs if not c.get("enabled")]
        return web.json_response({"backends": list(results) + disabled})

    @routes.get("/c2c/farm/preview/{job_id}")
    async def farm_preview(request):
        job = _qm().get(request.match_info["job_id"])
        if job is None or not job.remote_id:
            return web.json_response({"preview": None}, status=404)
        try:
            from .backends import get_adapter
            preview = get_adapter(job.backend_name).get_preview(job.remote_id)
        except Exception:  # noqa: BLE001
            preview = None
        return web.json_response({"preview": preview})

    async def _job_action(request, fn_name: str):
        body = await request.json()
        job_id = body.get("job_id", "")
        qm = _qm()
        job = qm.get(job_id)
        if job is None:
            return web.json_response({"ok": False, "error": f"unknown job {job_id}"}, status=404)
        if not _can_control(job.user):
            return web.json_response(
                {"ok": False, "error": "permission denied — only admins can control "
                                       "other users' jobs (see config/users.json)"},
                status=403)
        if fn_name == "bump":
            ok = qm.bump(job_id, int(body.get("priority", job.priority + 1)))
        else:
            ok = getattr(qm, fn_name)(job_id)
        return web.json_response({"ok": bool(ok)})

    @routes.post("/c2c/farm/cancel")
    async def farm_cancel(request):
        return await _job_action(request, "cancel")

    @routes.post("/c2c/farm/pause")
    async def farm_pause(request):
        return await _job_action(request, "pause")

    @routes.post("/c2c/farm/resume")
    async def farm_resume(request):
        return await _job_action(request, "resume")

    @routes.post("/c2c/farm/bump")
    async def farm_bump(request):
        return await _job_action(request, "bump")

    _registered = True
    log.info("C2C Farm dashboard routes registered under /c2c/farm/*")
