"""
Scheduled Queue — cron-style scheduling of API workflows on this server (rebuilt L9.39).

Routes:
  GET  /c2c/schedule/list          → jobs with next run, source and last result
  POST /c2c/schedule/add           → {label, cron, workflow (API JSON path) | workflow_json (API prompt), enabled, reseed}
  POST /c2c/schedule/remove        → {"id"}
  POST /c2c/schedule/toggle        → {"id"}
  POST /c2c/schedule/run_now       → {"id"}
  GET  /c2c/schedule/history       → last 100 runs
  GET  /c2c/schedule/preview?cron= → next runs or the parse error (live check in the panel)

Fixed in L9.39 (the module was never registered, and):
- the loop started at import with no jobs; now it runs only while an enabled job exists and sleeps to the next minute;
- cron day-of-week was off by one (Python Monday = 0, cron Sunday = 0); dom + dow now follow the Vixie OR rule;
- the 58 s double-fire guard could skip a minute; runs are keyed by minute, and minutes missed while the event loop
  was blocked (up to 10) are caught up;
- it posted to a hard-coded 127.0.0.1:8188; it queues on this server's real address now;
- a repeated identical prompt is fully cached by ComfyUI and does nothing; "reseed" (default on) gives every run new
  seeds;
- jobs survive a restart (user/default/c2c/automation.json).
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta

from . import _c2c_automation as A

log = logging.getLogger("C2C.ScheduledQueue")

_jobs: dict[str, dict] = {}
_history: deque[dict] = deque(maxlen=100)
_task: asyncio.Task | None = None
_server = None
_last_tick: datetime | None = None
CATCH_UP_MINUTES = 10


# ── cron ─────────────────────────────────────────────────────────────────────

_MACROS = {"@yearly": "0 0 1 1 *", "@annually": "0 0 1 1 *", "@monthly": "0 0 1 * *", "@weekly": "0 0 * * 0",
           "@daily": "0 0 * * *", "@midnight": "0 0 * * *", "@hourly": "0 * * * *"}
_MON = {m: i + 1 for i, m in enumerate("jan feb mar apr may jun jul aug sep oct nov dec".split())}
_DOW = {d: i for i, d in enumerate("sun mon tue wed thu fri sat".split())}
_RANGES = ((0, 59, None, "minute"), (0, 23, None, "hour"), (1, 31, None, "day of month"), (1, 12, _MON, "month"),
           (0, 7, _DOW, "day of week"))


@dataclass(frozen=True)
class Cron:
    minutes: frozenset
    hours: frozenset
    doms: frozenset
    months: frozenset
    dows: frozenset          # 0 = Sunday (7 folded to 0)
    dom_star: bool
    dow_star: bool


def _field(text: str, lo: int, hi: int, names, what: str) -> set[int]:
    def num(s: str) -> int:
        s = s.strip().lower()
        if names and s in names:
            return names[s]
        if not s.isdigit():
            raise ValueError(f"{what}: '{s}' is not a number")
        v = int(s)
        if not lo <= v <= hi:
            raise ValueError(f"{what}: {v} is outside {lo}-{hi}")
        return v

    out: set[int] = set()
    for part in text.split(","):
        if not part:
            raise ValueError(f"{what}: empty list item")
        base, _, step_s = part.partition("/")
        step = 1
        if step_s:
            if not step_s.isdigit() or int(step_s) == 0:
                raise ValueError(f"{what}: step '{step_s}' must be a whole number above 0")
            step = int(step_s)
        if base == "*":
            a, b = lo, hi
        elif "-" in base:
            a_s, b_s = base.split("-", 1)
            a, b = num(a_s), num(b_s)
            if a > b:
                raise ValueError(f"{what}: range {a}-{b} runs backwards")
        else:
            a = num(base)
            b = hi if step_s else a
        out.update(range(a, b + 1, step))
    return out


def parse_cron(cron: str) -> Cron:
    """5-field cron (minute hour day-of-month month day-of-week) with lists, ranges, steps, month/day names and
    @hourly/@daily/@weekly/@monthly/@yearly. Raises ValueError with a readable reason."""
    text = _MACROS.get(str(cron or "").strip().lower(), str(cron or "").strip())
    parts = text.split()
    if len(parts) != 5:
        raise ValueError(f"cron needs 5 fields (minute hour day month weekday), got {len(parts)}")
    sets = [_field(p, lo, hi, names, what) for p, (lo, hi, names, what) in zip(parts, _RANGES)]
    dows = {0 if d == 7 else d for d in sets[4]}
    return Cron(frozenset(sets[0]), frozenset(sets[1]), frozenset(sets[2]), frozenset(sets[3]), frozenset(dows),
                parts[2].startswith("*"), parts[4].startswith("*"))


def _day_ok(c: Cron, dt: datetime) -> bool:
    if dt.month not in c.months:
        return False
    dom = dt.day in c.doms
    dow = ((dt.weekday() + 1) % 7) in c.dows      # Python Monday = 0 -> cron Monday = 1, Sunday -> 0
    # Vixie cron: a field written with a leading "*" (so "*/2" too) ANDs; both restricted -> either day matches
    if c.dom_star or c.dow_star:
        return dom and dow
    return dom or dow


def cron_matches(cron, dt: datetime) -> bool:
    c = cron if isinstance(cron, Cron) else parse_cron(cron)
    return dt.minute in c.minutes and dt.hour in c.hours and _day_ok(c, dt)


def next_runs(cron, after: datetime | None = None, count: int = 1) -> list[datetime]:
    """The next `count` minutes strictly after `after` that match (searches up to 4 years: Feb 29 jobs)."""
    c = cron if isinstance(cron, Cron) else parse_cron(cron)
    t = (after or datetime.now()).replace(second=0, microsecond=0) + timedelta(minutes=1)
    end = t + timedelta(days=366 * 4 + 1)
    hours, minutes = sorted(c.hours), sorted(c.minutes)
    out: list[datetime] = []
    while t < end and len(out) < count:
        if not _day_ok(c, t):
            t = (t + timedelta(days=1)).replace(hour=0, minute=0)
            continue
        for h in hours:
            if h < t.hour:
                continue
            for m in minutes:
                if h == t.hour and m < t.minute:
                    continue
                out.append(t.replace(hour=h, minute=m))
                if len(out) >= count:
                    return out
        t = (t + timedelta(days=1)).replace(hour=0, minute=0)
    return out


# ── jobs ─────────────────────────────────────────────────────────────────────

def _public(job: dict) -> dict:
    try:
        nxt = next_runs(job["cron"])
        next_run = nxt[0].timestamp() if nxt else None
    except ValueError:
        next_run = None
    return {
        "id": job["id"], "label": job.get("label") or job["id"], "cron": job.get("cron", ""),
        "workflow": job.get("workflow", ""),
        "source": "file" if job.get("workflow") else "current workflow",
        "nodes": len(job.get("workflow_json") or {}) or None,
        "enabled": bool(job.get("enabled", True)), "reseed": bool(job.get("reseed", True)),
        "next_run": next_run if job.get("enabled", True) else None,
        "last_run": job.get("last_run"), "last_status": job.get("last_status"),
        "last_error": job.get("last_error"), "last_prompt_id": job.get("last_prompt_id"),
    }


def _persist() -> None:
    try:
        A.save_section("schedules", [{k: v for k, v in j.items() if k != "fired_key"} for j in _jobs.values()])
    except Exception as exc:  # noqa: BLE001
        log.warning("ScheduledQueue: could not save jobs: %s", exc)


def _prompt_for(job: dict) -> dict:
    prompt = A.load_api_workflow(job["workflow"]) if job.get("workflow") else A.parse_api_workflow(
        job.get("workflow_json") or {})
    return A.reseed(prompt) if job.get("reseed", True) else prompt


async def fire_job(job_id: str, reason: str = "schedule") -> dict:
    job = _jobs.get(job_id)
    if not job:
        return {"ok": False, "error": "job not found"}
    record = {"job_id": job_id, "label": job.get("label") or job_id, "reason": reason, "started_at": time.time()}
    try:
        res = await A.queue_prompt(_prompt_for(job), client_id=f"c2c-schedule-{job_id}", server=_server)
    except ValueError as exc:
        res = {"ok": False, "prompt_id": None, "error": str(exc)}
    record.update(status="queued" if res["ok"] else "error", prompt_id=res.get("prompt_id"), error=res.get("error"))
    _history.append(record)
    job.update(last_run=record["started_at"], last_status=record["status"], last_error=res.get("error"),
               last_prompt_id=res.get("prompt_id"))
    if not res["ok"]:
        log.warning("ScheduledQueue: job %s not queued: %s", job_id, res.get("error"))
    _persist()
    return res


def _due(now: datetime) -> list[str]:
    """Job ids due in the minutes since the last tick (missed minutes caught up, at most CATCH_UP_MINUTES)."""
    global _last_tick
    now = now.replace(second=0, microsecond=0)
    start = now if _last_tick is None else max(_last_tick + timedelta(minutes=1),
                                               now - timedelta(minutes=CATCH_UP_MINUTES - 1))
    _last_tick = now
    due = []
    t = start
    while t <= now:
        key = t.strftime("%Y%m%d%H%M")
        for jid, job in _jobs.items():
            if not job.get("enabled", True) or job.get("fired_key") == key or jid in due:
                continue
            try:
                if cron_matches(job["cron"], t):
                    job["fired_key"] = key
                    due.append(jid)
            except ValueError:
                continue
        t += timedelta(minutes=1)
    return due


def _any_enabled() -> bool:
    return any(j.get("enabled", True) for j in _jobs.values())


async def _loop() -> None:
    global _task, _last_tick
    log.info("ScheduledQueue: scheduler started (%d job(s)).", len(_jobs))
    try:
        while _any_enabled():
            now = datetime.now()
            await asyncio.sleep(60.5 - now.second - now.microsecond / 1e6)
            for jid in _due(datetime.now()):
                asyncio.ensure_future(fire_job(jid))
    finally:
        _task, _last_tick = None, None
        log.info("ScheduledQueue: scheduler stopped (no enabled jobs).")


def _ensure_loop() -> None:
    """Start the loop when an enabled job exists; from import time it is handed to the server's loop."""
    global _task
    if (_task is not None and not _task.done()) or not _any_enabled():
        return
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        srv_loop = getattr(_server, "loop", None)
        if srv_loop is not None:
            srv_loop.call_soon_threadsafe(_ensure_loop)
        return
    _task = loop.create_task(_loop())


def _load() -> None:
    for j in A.load_section("schedules"):
        if isinstance(j, dict) and j.get("id") and j.get("cron"):
            _jobs[str(j["id"])] = j


# ── routes ────────────────────────────────────────────────────────────────────

def register_routes(server) -> None:
    global _server
    from aiohttp import web

    _server = server
    _load()
    _ensure_loop()

    def err(msg: str, status: int = 400):
        return web.json_response({"error": msg}, status=status)

    async def body_of(request) -> dict:
        try:
            b = await request.json()
            return b if isinstance(b, dict) else {}
        except Exception:  # noqa: BLE001
            return {}

    @server.routes.get("/c2c/schedule/list")
    async def schedule_list(_request: web.Request) -> web.Response:
        return web.json_response({"jobs": [_public(j) for j in _jobs.values()], "running": _task is not None,
                                  "server": A.own_base_url(_server)})

    @server.routes.get("/c2c/schedule/preview")
    async def schedule_preview(request: web.Request) -> web.Response:
        try:
            runs = next_runs(request.query.get("cron", ""), count=3)
        except ValueError as exc:
            return web.json_response({"ok": False, "error": str(exc)})
        return web.json_response({"ok": True, "next": [r.timestamp() for r in runs]})

    @server.routes.post("/c2c/schedule/add")
    async def schedule_add(request: web.Request) -> web.Response:
        body = await body_of(request)
        cron = str(body.get("cron", "")).strip()
        try:
            parse_cron(cron)
        except ValueError as exc:
            return err(f"Invalid cron: {exc}")
        path = str(body.get("workflow") or "").strip().strip('"')
        wf = body.get("workflow_json") or None
        try:
            if wf:
                wf = A.parse_api_workflow(wf)
                path = ""
            elif path:
                A.load_api_workflow(path)
            else:
                return err("Pick a workflow: an API-format JSON file path, or 'Use current workflow'.")
        except ValueError as exc:
            return err(str(exc))
        job_id = str(body.get("id") or "").strip() or uuid.uuid4().hex[:8]
        _jobs[job_id] = {"id": job_id, "cron": cron, "workflow": path, "workflow_json": wf if not path else None,
                         "label": str(body.get("label") or "").strip() or job_id,
                         "enabled": bool(body.get("enabled", True)), "reseed": bool(body.get("reseed", True))}
        _persist()
        _ensure_loop()
        return web.json_response({"status": "added", "id": job_id, "job": _public(_jobs[job_id])})

    @server.routes.post("/c2c/schedule/remove")
    async def schedule_remove(request: web.Request) -> web.Response:
        job_id = str((await body_of(request)).get("id", ""))
        if job_id not in _jobs:
            return err("job not found", 404)
        del _jobs[job_id]
        _persist()
        return web.json_response({"status": "removed", "id": job_id})

    @server.routes.post("/c2c/schedule/toggle")
    async def schedule_toggle(request: web.Request) -> web.Response:
        job_id = str((await body_of(request)).get("id", ""))
        if job_id not in _jobs:
            return err("job not found", 404)
        _jobs[job_id]["enabled"] = not _jobs[job_id].get("enabled", True)
        _persist()
        _ensure_loop()
        return web.json_response({"status": "ok", "enabled": _jobs[job_id]["enabled"]})

    @server.routes.post("/c2c/schedule/run_now")
    async def schedule_run_now(request: web.Request) -> web.Response:
        job_id = str((await body_of(request)).get("id", ""))
        if job_id not in _jobs:
            return err("job not found", 404)
        res = await fire_job(job_id, reason="run now")
        return web.json_response({"status": "queued" if res["ok"] else "error", **res},
                                 status=200 if res["ok"] else 400)

    @server.routes.get("/c2c/schedule/history")
    async def schedule_history(_request: web.Request) -> web.Response:
        return web.json_response({"history": list(_history)})

    log.info("ScheduledQueue routes registered (/c2c/schedule/*), %d job(s) loaded.", len(_jobs))
