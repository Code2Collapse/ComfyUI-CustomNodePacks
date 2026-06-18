"""
Scheduled Queue — P2.1
Cron-style task scheduling: run queued workflows at specified times/intervals.

Routes:
  GET  /c2c/schedule/list          → list all jobs
  POST /c2c/schedule/add           → add job {"id","cron","workflow","enabled"}
  POST /c2c/schedule/remove        → remove job {"id"}
  POST /c2c/schedule/toggle        → toggle enabled {"id"}
  POST /c2c/schedule/run_now       → immediately fire job {"id"}
  GET  /c2c/schedule/history       → last N run records
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from collections import deque
from pathlib import Path
from typing import Any

log = logging.getLogger("C2C.ScheduledQueue")

_jobs: dict[str, dict] = {}            # id → job spec
_history: deque[dict] = deque(maxlen=100)
_scheduler_task: asyncio.Task | None = None
_TICK_INTERVAL = 30.0                   # check every 30 s


# ── cron parser (simplified: supports minute hour dom month dow) ──────────────

def _parse_cron(cron: str) -> tuple[set, set, set, set, set] | None:
    """Parse 5-field cron into (minutes, hours, doms, months, dows). * = all."""
    parts = cron.strip().split()
    if len(parts) != 5:
        return None

    def _parse_field(s: str, lo: int, hi: int) -> set[int]:
        result: set[int] = set()
        for part in s.split(","):
            if part == "*":
                result.update(range(lo, hi + 1))
            elif "/" in part:
                base, step_s = part.split("/", 1)
                step = int(step_s)
                start = lo if base == "*" else int(base)
                result.update(range(start, hi + 1, step))
            elif "-" in part:
                a, b = part.split("-", 1)
                result.update(range(int(a), int(b) + 1))
            else:
                result.add(int(part))
        return result

    try:
        return (
            _parse_field(parts[0], 0, 59),
            _parse_field(parts[1], 0, 23),
            _parse_field(parts[2], 1, 31),
            _parse_field(parts[3], 1, 12),
            _parse_field(parts[4], 0, 6),
        )
    except Exception:
        return None


def _cron_matches(cron: str, t: time.struct_time) -> bool:
    parsed = _parse_cron(cron)
    if parsed is None:
        return False
    mins, hrs, doms, months, dows = parsed
    return (
        t.tm_min  in mins and
        t.tm_hour in hrs  and
        t.tm_mday in doms and
        t.tm_mon  in months and
        t.tm_wday in dows
    )


# ── queue a workflow ──────────────────────────────────────────────────────────

async def _fire_job(job_id: str) -> None:
    job = _jobs.get(job_id)
    if not job:
        return
    log.info("ScheduledQueue: firing job %s", job_id)
    workflow_path = job.get("workflow", "")
    record = {"job_id": job_id, "started_at": time.time(), "status": "running"}
    _history.append(record)
    try:
        if workflow_path:
            wf_text = Path(workflow_path).read_text(encoding="utf-8")
            wf = json.loads(wf_text)
        else:
            wf = job.get("workflow_json", {})
        import aiohttp
        async with aiohttp.ClientSession() as sess:
            async with sess.post(
                "http://127.0.0.1:8188/prompt",
                json={"prompt": wf},
                timeout=aiohttp.ClientTimeout(total=10),
            ) as r:
                record["status"] = "queued" if r.status in (200, 201) else f"error_{r.status}"
    except Exception as exc:
        record["status"] = f"error: {exc}"
        log.error("ScheduledQueue: job %s failed: %s", job_id, exc)
    record["ended_at"] = time.time()
    job["last_run"] = record["ended_at"]
    job["last_status"] = record["status"]


async def _scheduler_loop() -> None:
    while True:
        try:
            now_struct = time.localtime()
            for job_id, job in list(_jobs.items()):
                if not job.get("enabled", True):
                    continue
                cron = job.get("cron", "")
                if cron and _cron_matches(cron, now_struct):
                    last = job.get("last_run", 0)
                    # prevent double-fire within same minute
                    if time.time() - last > 58:
                        asyncio.create_task(_fire_job(job_id))
        except Exception as exc:
            log.error("Scheduler loop error: %s", exc)
        await asyncio.sleep(_TICK_INTERVAL)


def _start_scheduler() -> None:
    global _scheduler_task
    if _scheduler_task is None or _scheduler_task.done():
        _scheduler_task = asyncio.create_task(_scheduler_loop())
        log.info("ScheduledQueue: scheduler loop started.")


# ── routes ────────────────────────────────────────────────────────────────────

def register_routes(server) -> None:
    from aiohttp import web

    _start_scheduler()

    @server.routes.get("/c2c/schedule/list")
    async def schedule_list(_request: web.Request) -> web.Response:
        result = []
        for jid, job in _jobs.items():
            result.append({
                "id"          : jid,
                "cron"        : job.get("cron"),
                "workflow"    : job.get("workflow"),
                "label"       : job.get("label", jid),
                "enabled"     : job.get("enabled", True),
                "last_run"    : job.get("last_run"),
                "last_status" : job.get("last_status"),
            })
        return web.json_response({"jobs": result})

    @server.routes.post("/c2c/schedule/add")
    async def schedule_add(request: web.Request) -> web.Response:
        body = await request.json()
        job_id = body.get("id", "").strip()
        if not job_id:
            import uuid
            job_id = str(uuid.uuid4())[:8]
        cron = body.get("cron", "").strip()
        if cron and _parse_cron(cron) is None:
            return web.json_response({"error": f"Invalid cron expression: {cron}"}, status=400)
        _jobs[job_id] = {
            "id"           : job_id,
            "cron"         : cron,
            "workflow"     : body.get("workflow", ""),
            "workflow_json": body.get("workflow_json", {}),
            "label"        : body.get("label", job_id),
            "enabled"      : body.get("enabled", True),
        }
        return web.json_response({"status": "added", "id": job_id})

    @server.routes.post("/c2c/schedule/remove")
    async def schedule_remove(request: web.Request) -> web.Response:
        body = await request.json()
        job_id = body.get("id", "")
        if job_id not in _jobs:
            return web.json_response({"error": "job not found"}, status=404)
        del _jobs[job_id]
        return web.json_response({"status": "removed", "id": job_id})

    @server.routes.post("/c2c/schedule/toggle")
    async def schedule_toggle(request: web.Request) -> web.Response:
        body = await request.json()
        job_id = body.get("id", "")
        if job_id not in _jobs:
            return web.json_response({"error": "job not found"}, status=404)
        _jobs[job_id]["enabled"] = not _jobs[job_id].get("enabled", True)
        return web.json_response({"status": "ok", "enabled": _jobs[job_id]["enabled"]})

    @server.routes.post("/c2c/schedule/run_now")
    async def schedule_run_now(request: web.Request) -> web.Response:
        body = await request.json()
        job_id = body.get("id", "")
        if job_id not in _jobs:
            return web.json_response({"error": "job not found"}, status=404)
        asyncio.create_task(_fire_job(job_id))
        return web.json_response({"status": "fired", "id": job_id})

    @server.routes.get("/c2c/schedule/history")
    async def schedule_history(_request: web.Request) -> web.Response:
        return web.json_response({"history": list(_history)})

    log.info("ScheduledQueue routes registered (/c2c/schedule/*).")
