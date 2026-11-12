"""aiohttp application: runner API, owner / publisher API, owner-only Studio proxy, runner pages.

Security rules this file keeps (design docs/research/L9.206_owner_app_gateway_design.md):
- An app or job the caller may not see answers exactly like one that does not exist (404 ``{"error":"not_found"}``).
- No route forwards a runner request to the worker. ``/studio/*`` is the only proxy and it requires an owner.
- Runner-visible bodies come from the manifest and ``JobManager.project`` only.
"""
from __future__ import annotations

import asyncio
import json
import logging
import mimetypes
import re
import secrets
from pathlib import Path

import aiohttp
from aiohttp import web

from . import manifest as M
from .auth import COOKIE, CSRF_HEADER, ROLE_SCOPES, RateLimiter, Sessions, Signer, Tokens
from .jobs import ERRORS, JobManager, LimitError
from .store import APP_ID_RE, Store, StoreError
from .worker import Worker

log = logging.getLogger("c2c_app_gateway")
STATIC = Path(__file__).with_name("static")
STATIC_FILES = {"runner.js": "application/javascript", "runner.css": "text/css"}
UNSAFE = ("POST", "PUT", "PATCH", "DELETE")
HOP = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailer", "transfer-encoding",
       "upgrade", "host", "cookie", "authorization", "origin", "referer", "content-length", "accept-encoding",
       CSRF_HEADER.lower(), "x-forwarded-for", "x-forwarded-proto", "x-forwarded-host"}
RESP_DROP = {"connection", "keep-alive", "transfer-encoding", "content-encoding", "content-length", "set-cookie",
             "server"}
CSP = ("default-src 'self'; img-src 'self' blob: data:; media-src 'self' blob:; style-src 'self'; script-src 'self'; "
       "connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'; object-src 'none'")

DEFAULTS = {"data_dir": None, "bind": "127.0.0.1", "port": 8390, "public_origin": None,
            "worker_url": "http://127.0.0.1:8188", "worker_headers": {}, "secure_cookies": True,
            "session_idle_s": 12 * 3600, "max_upload_mb": 50, "max_active_total": 32, "max_active_per_user": 4,
            "max_output_mb": 512, "scrypt_n": 2 ** 15, "trust_forwarded_for": False, "studio_proxy": True,
            "poll_s": 2.0, "queue_wait_s": 3600, "sse_heartbeat_s": 15.0, "download_ttl_s": 600,
            "login_limit_per_user": 10, "login_limit_per_ip": 30, "login_window_s": 900,
            "retention_h": 72, "worker_input_dir": None}


def _nf():
    return web.json_response({"error": "not_found"}, status=404)


def _bad(msg: str, status: int = 400, code: str = "invalid_input"):
    return web.json_response({"error": code, "message": msg}, status=status)


class Gateway:
    def __init__(self, cfg: dict, *, worker: Worker | None = None):
        self.cfg = dict(DEFAULTS, **{k: v for k, v in (cfg or {}).items() if v is not None})
        if not self.cfg["data_dir"]:
            raise ValueError("config: data_dir is required")
        self.store = Store(self.cfg["data_dir"], scrypt_n=self.cfg["scrypt_n"])
        self.sessions = Sessions(self.cfg["session_idle_s"])
        self.tokens = Tokens(self.store.subkey("tokens"), self.store.revoked_tokens)
        self.signer = Signer(self.store.subkey("downloads"))
        self.limits = RateLimiter()
        self.worker = worker or Worker(self.cfg["worker_url"], headers=self.cfg["worker_headers"])
        self.jobs = JobManager(self.store, self.worker, poll_s=self.cfg["poll_s"], queue_wait_s=self.cfg["queue_wait_s"],
                               max_active_per_user=self.cfg["max_active_per_user"],
                               max_active_total=self.cfg["max_active_total"], max_output_mb=self.cfg["max_output_mb"],
                               retention_h=self.cfg["retention_h"], worker_input_dir=self.cfg["worker_input_dir"])

    # ── app factory ─────────────────────────────────────────────────────────────────────────────────────────
    def make_app(self) -> web.Application:
        app = web.Application(client_max_size=(int(self.cfg["max_upload_mb"]) + 2) * 1024 * 1024,
                              middlewares=[self._mw_errors, self._mw_ident])
        app.on_response_prepare.append(self._headers)
        app.on_startup.append(self._startup)
        app.on_cleanup.append(self._cleanup)
        r = app.router
        r.add_get("/", self.page_index)
        r.add_get("/app/{app_id}", self.page_app)
        r.add_get("/static/{name}", self.static)
        r.add_post("/api/v1/login", self.login)
        r.add_post("/api/v1/logout", self.logout)
        r.add_get("/api/v1/me", self.me)
        r.add_get("/api/v1/apps", self.list_apps)
        r.add_get("/api/v1/apps/{app_id}", self.app_info)
        r.add_get("/api/v1/apps/{app_id}/manifest", self.app_manifest)
        r.add_post("/api/v1/apps/{app_id}/runs", self.run)
        r.add_get("/api/v1/jobs", self.my_jobs)
        r.add_get("/api/v1/jobs/{job_id}", self.job)
        r.add_get("/api/v1/jobs/{job_id}/events", self.job_events)
        r.add_post("/api/v1/jobs/{job_id}/cancel", self.job_cancel)
        r.add_get("/api/v1/jobs/{job_id}/outputs", self.job_outputs)
        r.add_get("/dl/{token}", self.download)
        # owner / publisher
        r.add_get("/api/v1/admin/users", self.adm_users)
        r.add_post("/api/v1/admin/users", self.adm_add_user)
        r.add_post("/api/v1/admin/users/{name}/disable", self.adm_disable_user)
        r.add_get("/api/v1/admin/apps", self.adm_apps)
        r.add_post("/api/v1/admin/apps", self.adm_create_app)
        r.add_post("/api/v1/admin/apps/{app_id}/releases", self.adm_publish)
        r.add_get("/api/v1/admin/apps/{app_id}/releases/{n}", self.adm_release)
        r.add_post("/api/v1/admin/apps/{app_id}/rollback", self.adm_rollback)
        r.add_post("/api/v1/admin/apps/{app_id}/acl", self.adm_acl)
        r.add_post("/api/v1/admin/tokens", self.adm_token)
        r.add_post("/api/v1/admin/tokens/revoke", self.adm_revoke_token)
        r.add_get("/api/v1/admin/audit", self.adm_audit)
        r.add_get("/api/v1/admin/jobs", self.adm_jobs)
        if self.cfg["studio_proxy"]:
            r.add_route("*", "/studio", self.studio_root)
            r.add_route("*", "/studio/{tail:.*}", self.studio)
        return app

    async def _startup(self, app):
        self.jobs.start()

    async def _cleanup(self, app):
        await self.jobs.stop()
        await self.worker.close()

    # ── middleware ──────────────────────────────────────────────────────────────────────────────────────────
    @web.middleware
    async def _mw_errors(self, request, handler):
        try:
            return await handler(request)
        except web.HTTPException as exc:
            if exc.status == 404:
                return _nf()
            if exc.status == 413:
                return _bad("The upload is larger than this server accepts.", 413, "too_large")
            raise
        except json.JSONDecodeError:
            return _bad("The request body is not valid JSON.")
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - never leak a traceback to any caller
            log.exception("unhandled error on %s", request.path)
            return web.json_response({"error": "server_error"}, status=500)

    @web.middleware
    async def _mw_ident(self, request, handler):
        request["ident"] = self._identify(request)
        return await handler(request)

    async def _headers(self, request, response):
        response.headers["Server"] = "c2c-gateway"
        if request.path.startswith("/studio"):
            return
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Content-Security-Policy", CSP)
        if request.path.startswith("/api/") or request.path.startswith("/dl/"):
            response.headers.setdefault("Cache-Control", "no-store")

    # ── identity ────────────────────────────────────────────────────────────────────────────────────────────
    def _ip(self, request) -> str:
        if self.cfg["trust_forwarded_for"]:
            xf = request.headers.get("X-Forwarded-For", "")
            if xf:
                return xf.split(",")[0].strip()[:64]
        return request.remote or "?"

    def _identify(self, request) -> dict | None:
        auth = request.headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            p = self.tokens.verify(auth[7:].strip())
            u = self.store.get_user(p["u"]) if p else None
            if not u or u["disabled"]:
                return None
            return {"user": u["name"], "role": u["role"], "via": "token", "jti": p["jti"],
                    "scopes": set(p["s"]) & ROLE_SCOPES[u["role"]], "apps": set(p["a"]) if p["a"] else None}
        s = self.sessions.get(request.cookies.get(COOKIE))
        if not s:
            return None
        u = self.store.get_user(s["user"])
        if not u or u["disabled"]:
            self.sessions.drop(s["sid"])
            return None
        return {"user": u["name"], "role": u["role"], "via": "session", "sid": s["sid"], "csrf": s["csrf"],
                "scopes": set(ROLE_SCOPES[u["role"]]), "apps": None}

    def _origin_ok(self, request) -> bool:
        origin = request.headers.get("Origin")
        want = self.cfg["public_origin"] or f"{request.scheme}://{request.host}"
        return origin is not None and origin.rstrip("/") == want.rstrip("/")

    def _csrf_ok(self, request, ident) -> bool:
        if ident["via"] != "session":
            return True                                                   # bearer tokens carry no ambient authority
        return self._origin_ok(request) and secrets.compare_digest(request.headers.get(CSRF_HEADER, ""),
                                                                   ident["csrf"])

    def _guard(self, request, scope: str):
        """(ident, error_response). Unsafe methods through a session need CSRF + Origin."""
        ident = request["ident"]
        if not ident:
            return None, web.json_response({"error": "login_required"}, status=401)
        if scope not in ident["scopes"]:
            return None, _nf()
        if request.method in UNSAFE and not self._csrf_ok(request, ident):
            return None, web.json_response({"error": "csrf"}, status=403)
        return ident, None

    def _runnable(self, ident, app_id: str):
        """(app, manifest) when ``ident`` may run this app, else None. Same answer for missing and forbidden."""
        if not isinstance(app_id, str) or not APP_ID_RE.match(app_id):
            return None
        if ident["apps"] is not None and app_id not in ident["apps"]:
            return None
        app = self.store.get_app(app_id)
        if not app or not app.get("current"):
            return None
        if ident["role"] != "owner" and ident["user"] != app["owner"] and ident["user"] not in app["acl"]:
            return None
        man = self.store.read_manifest(app_id, app["current"])
        return (app, man) if man else None

    def _admin_app(self, ident, app_id: str):
        app = self.store.get_app(app_id) if APP_ID_RE.match(app_id or "") else None
        if not app or (ident["role"] != "owner" and app["owner"] != ident["user"]):
            return None
        if ident["apps"] is not None and app_id not in ident["apps"]:
            return None
        return app

    def _visible_job(self, ident, job_id: str):
        job = self.jobs.get(job_id) if isinstance(job_id, str) else None
        if not job:
            return None
        if ident["role"] != "owner" and job["user"] != ident["user"]:
            return None
        if not self._runnable(ident, job["app_id"]):                       # revoked runners lose their jobs too
            return None
        return job

    # ── pages ───────────────────────────────────────────────────────────────────────────────────────────────
    async def page_index(self, request):
        return web.FileResponse(STATIC / "index.html", headers={"Content-Type": "text/html; charset=utf-8"})

    async def page_app(self, request):
        ident = request["ident"]
        app_id = request.match_info["app_id"]
        if not ident:
            raise web.HTTPFound("/?next=" + ("/app/" + app_id if APP_ID_RE.match(app_id) else "/"))
        if "run" not in ident["scopes"] or not self._runnable(ident, app_id):
            return _nf()
        return web.FileResponse(STATIC / "app.html", headers={"Content-Type": "text/html; charset=utf-8"})

    async def static(self, request):
        name = request.match_info["name"]
        if name not in STATIC_FILES:
            return _nf()
        return web.FileResponse(STATIC / name, headers={"Content-Type": STATIC_FILES[name] + "; charset=utf-8"})

    # ── session ─────────────────────────────────────────────────────────────────────────────────────────────
    async def login(self, request):
        if request.headers.get("Origin") is not None and not self._origin_ok(request):
            return web.json_response({"error": "csrf"}, status=403)
        try:
            body = await request.json()
            name, pw = str(body.get("user", ""))[:64], str(body.get("password", ""))[:1024]
        except Exception:  # noqa: BLE001
            return _bad("Send JSON {user, password}.")
        ip, w = self._ip(request), self.cfg["login_window_s"]
        if not self.limits.hit(f"login-ip:{ip}", self.cfg["login_limit_per_ip"], w, record=False) or \
                not self.limits.hit(f"login-user:{name}", self.cfg["login_limit_per_user"], w, record=False):
            self.store.audit("login_limited", actor=name, ip=ip)
            return _bad("Too many sign-in attempts. Wait a few minutes.", 429, "rate_limited")
        u = await asyncio.get_running_loop().run_in_executor(None, self.store.verify_password, name, pw)
        if not u:
            self.limits.hit(f"login-ip:{ip}", 10 ** 9, w)
            self.limits.hit(f"login-user:{name}", 10 ** 9, w)
            self.store.audit("login_failed", actor=name, ip=ip)
            return web.json_response({"error": "bad_credentials", "message": "Wrong user name or password."},
                                     status=401)
        self.limits.clear(f"login-user:{name}")
        s = self.sessions.create(u["name"])
        self.store.audit("login", actor=u["name"], ip=ip)
        resp = web.json_response({"user": u["name"], "role": u["role"], "csrf": s["csrf"]})
        resp.set_cookie(COOKIE, s["sid"], httponly=True, samesite="Strict", secure=bool(self.cfg["secure_cookies"]),
                        path="/", max_age=int(self.cfg["session_idle_s"]))
        return resp

    async def logout(self, request):
        ident = request["ident"]
        if ident and ident["via"] == "session":
            if not self._csrf_ok(request, ident):
                return web.json_response({"error": "csrf"}, status=403)
            self.sessions.drop(ident["sid"])
            self.store.audit("logout", actor=ident["user"], ip=self._ip(request))
        resp = web.json_response({"ok": True})
        resp.del_cookie(COOKIE, path="/")
        return resp

    async def me(self, request):
        ident = request["ident"]
        if not ident:
            return web.json_response({"error": "login_required"}, status=401)
        return web.json_response({"user": ident["user"], "role": ident["role"], "csrf": ident.get("csrf"),
                                  "scopes": sorted(ident["scopes"])})

    # ── runner API ──────────────────────────────────────────────────────────────────────────────────────────
    async def list_apps(self, request):
        ident, err = self._guard(request, "run")
        if err:
            return err
        out = []
        for a in self.store.list_apps():
            r = self._runnable(ident, a["app_id"])
            if r:
                out.append({"app_id": a["app_id"], "name": r[1]["name"], "release": r[1]["release"],
                            "description": r[1]["description"]})
        return web.json_response({"apps": out})

    async def app_info(self, request):
        ident, err = self._guard(request, "run")
        if err:
            return err
        r = self._runnable(ident, request.match_info["app_id"])
        if not r:
            return _nf()
        m = r[1]
        return web.json_response({"app_id": m["app_id"], "name": m["name"], "release": m["release"],
                                  "description": m["description"]})

    async def app_manifest(self, request):
        ident, err = self._guard(request, "run")
        if err:
            return err
        r = self._runnable(ident, request.match_info["app_id"])
        return web.json_response(r[1]) if r else _nf()

    async def _read_run_body(self, request, max_bytes: int):
        values, files = {}, {}
        ctype = request.content_type
        if ctype == "multipart/form-data":
            reader = await request.multipart()
            total = 0
            while True:
                part = await reader.next()
                if part is None:
                    break
                if part.name == "values" and not part.filename:
                    values = json.loads((await part.read(decode=False)).decode("utf-8") or "{}")
                    continue
                if not part.name or part.name in files:
                    raise M.InputError("Each upload needs a distinct field name.")
                buf = bytearray()
                while True:
                    chunk = await part.read_chunk(1 << 16)
                    if not chunk:
                        break
                    buf += chunk
                    total += len(chunk)
                    if total > max_bytes:
                        raise M.InputError("The uploads are larger than this server accepts.")
                files[part.name] = bytes(buf)
        elif ctype == "application/json":
            body = await request.json()
            values = body.get("values", {}) if isinstance(body, dict) else None
        else:
            raise M.InputError("Send multipart/form-data (a 'values' JSON field + files) or JSON {values}.")
        return values, files

    async def run(self, request):
        ident, err = self._guard(request, "run")
        if err:
            return err
        r = self._runnable(ident, request.match_info["app_id"])
        if not r:
            return _nf()
        app, man = r
        try:
            values, files = await self._read_run_body(request, int(self.cfg["max_upload_mb"]) * 1024 * 1024)
            resolved = M.resolve_values(man, values, files)
        except M.InputError as exc:
            return _bad(str(exc))
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
            return _bad("The run request is not valid JSON / multipart.")
        private = self.store.read_private(app["app_id"], man["release"])
        try:
            job = await self.jobs.submit(ident["user"], app, man, private, resolved)
        except LimitError as exc:
            return _bad(str(exc), 429, "rate_limited")
        self.store.audit("run", actor=ident["user"], app=app["app_id"], release=man["release"], job=job["job_id"],
                         ip=self._ip(request))
        return web.json_response(self.jobs.project(job), status=202)

    async def my_jobs(self, request):
        ident, err = self._guard(request, "run")
        if err:
            return err
        out = [self.jobs.project(j) for j in self.jobs._jobs.values()
               if j["user"] == ident["user"] and self._runnable(ident, j["app_id"])]
        out.sort(key=lambda j: -j["created"])
        return web.json_response({"jobs": out[:200]})

    async def job(self, request):
        ident, err = self._guard(request, "run")
        if err:
            return err
        job = self._visible_job(ident, request.match_info["job_id"])
        return web.json_response(self.jobs.project(job)) if job else _nf()

    async def job_events(self, request):
        ident, err = self._guard(request, "run")
        if err:
            return err
        job_id = request.match_info["job_id"]
        job = self._visible_job(ident, job_id)
        if not job:
            return _nf()
        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream", "Cache-Control": "no-store",
                                           "X-Accel-Buffering": "no"})
        await resp.prepare(request)
        q = self.jobs.subscribe(job_id)
        hb = float(self.cfg["sse_heartbeat_s"])
        try:
            view = self.jobs.project(job)
            await resp.write(f"event: job\ndata: {json.dumps(view)}\n\n".encode())
            while view["state"] not in ("succeeded", "failed", "cancelled"):
                try:
                    view = await asyncio.wait_for(q.get(), timeout=hb)
                    await resp.write(f"event: job\ndata: {json.dumps(view)}\n\n".encode())
                except asyncio.TimeoutError:
                    await resp.write(b": keep-alive\n\n")
                ident = self._identify(request)                          # revocation closes the stream
                if not ident or not self._visible_job(ident, job_id):
                    await resp.write(b"event: revoked\ndata: {}\n\n")
                    break
        except (ConnectionResetError, asyncio.CancelledError):
            pass
        finally:
            self.jobs.unsubscribe(job_id, q)
        return resp

    async def job_cancel(self, request):
        ident, err = self._guard(request, "run")
        if err:
            return err
        job = self._visible_job(ident, request.match_info["job_id"])
        if not job:
            return _nf()
        man = self.store.read_manifest(job["app_id"], job["release"]) or {}
        if man.get("cancel") == "never" and ident["role"] != "owner":
            return _bad("This app does not allow cancelling a run.", 409, "not_allowed")
        await self.jobs.cancel(job)
        self.store.audit("cancel", actor=ident["user"], app=job["app_id"], job=job["job_id"], ip=self._ip(request))
        return web.json_response(self.jobs.project(job))

    async def job_outputs(self, request):
        ident, err = self._guard(request, "run")
        if err:
            return err
        job = self._visible_job(ident, request.match_info["job_id"])
        if not job:
            return _nf()
        if job["state"] != "succeeded":
            return _bad("This run has no outputs yet.", 409, "not_ready")
        man = self.store.read_manifest(job["app_id"], job["release"]) or {}
        types = {o["id"]: o["type"] for o in man.get("outputs", [])}
        ttl = int(self.cfg["download_ttl_s"])
        out = [{"id": o["id"], "index": o["index"], "type": types.get(o["id"], "file"), "size": o["size"],
                "url": "/dl/" + self.signer.sign(job["job_id"], o["name"], ident["user"], ttl)}
               for o in job["outputs"]]
        return web.json_response({"outputs": out, "expires_in_s": ttl})

    async def download(self, request):
        v = self.signer.verify(request.match_info["token"])
        if not v:
            return _nf()
        job_id, name, user = v
        u = self.store.get_user(user)
        job = self.jobs.get(job_id)
        if not u or u["disabled"] or not job or job["user"] != user or name not in {o["name"] for o in job["outputs"]}:
            return _nf()
        ident = {"user": user, "role": u["role"], "scopes": ROLE_SCOPES[u["role"]], "apps": None}
        if not self._runnable(ident, job["app_id"]):                     # ACL re-checked at download time
            return _nf()
        path = self.store.output_dir(job_id) / name
        if not re.fullmatch(r"[a-z0-9_]{1,48}_\d{1,4}\.[a-z0-9]{1,5}", name) or not path.is_file():
            return _nf()
        self.store.audit("download", actor=user, app=job["app_id"], job=job_id, ip=self._ip(request))
        ctype = mimetypes.guess_type(name)[0] or "application/octet-stream"
        return web.FileResponse(path, headers={"Content-Type": ctype,
                                               "Content-Disposition": f'inline; filename="{name}"'})

    # ── owner / publisher API ───────────────────────────────────────────────────────────────────────────────
    async def adm_users(self, request):
        ident, err = self._guard(request, "admin")
        if err:
            return err
        return web.json_response({"users": self.store.list_users()})

    async def adm_add_user(self, request):
        ident, err = self._guard(request, "admin")
        if err:
            return err
        b = await request.json()
        try:
            u = await asyncio.get_running_loop().run_in_executor(
                None, self.store.add_user, str(b.get("name", "")), b.get("password"), str(b.get("role", "")))
        except StoreError as exc:
            return _bad(str(exc))
        self.store.audit("user_add", actor=ident["user"], target=u["name"], role=u["role"])
        return web.json_response(u, status=201)

    async def adm_disable_user(self, request):
        ident, err = self._guard(request, "admin")
        if err:
            return err
        name = request.match_info["name"]
        b = await request.json() if request.can_read_body else {}
        disabled = bool((b or {}).get("disabled", True))
        if name == ident["user"] and disabled:
            return _bad("You cannot disable your own account.")
        try:
            self.store.set_disabled(name, disabled)
        except StoreError as exc:
            return _bad(str(exc))
        if disabled:
            self.sessions.drop_user(name)
        self.store.audit("revoke" if disabled else "user_enable", actor=ident["user"], target=name)
        return web.json_response({"ok": True})

    async def adm_apps(self, request):
        ident, err = self._guard(request, "publish")
        if err:
            return err
        return web.json_response({"apps": [a for a in self.store.list_apps() if self._admin_app(ident, a["app_id"])]})

    async def adm_create_app(self, request):
        ident, err = self._guard(request, "publish")
        if err:
            return err
        b = await request.json()
        try:
            app = self.store.create_app(str(b.get("app_id", "")), ident["user"], str(b.get("name", "")))
        except StoreError as exc:
            return _bad(str(exc))
        self.store.audit("app_create", actor=ident["user"], app=app["app_id"])
        return web.json_response(app, status=201)

    async def adm_publish(self, request):
        ident, err = self._guard(request, "publish")
        if err:
            return err
        app = self._admin_app(ident, request.match_info["app_id"])
        if not app:
            return _nf()
        b = await request.json()
        try:
            man = M.validate_manifest(b.get("manifest"), app["app_id"])
            priv = M.validate_private(b.get("private"), man)
            n = self.store.write_release(app["app_id"], man, priv)
        except (M.ManifestError, StoreError) as exc:
            return _bad(str(exc))
        self.store.audit("publish", actor=ident["user"], app=app["app_id"], release=n, schema_hash=man["schema_hash"])
        return web.json_response({"app_id": app["app_id"], "release": n, "schema_hash": man["schema_hash"]},
                                 status=201)

    async def adm_release(self, request):
        ident, err = self._guard(request, "publish")
        if err:
            return err
        app = self._admin_app(ident, request.match_info["app_id"])
        try:
            n = int(request.match_info["n"])
        except ValueError:
            return _nf()
        if not app or n not in app["releases"]:
            return _nf()
        return web.json_response({"manifest": self.store.read_manifest(app["app_id"], n),
                                  "private": self.store.read_private(app["app_id"], n)})

    async def adm_rollback(self, request):
        ident, err = self._guard(request, "publish")
        if err:
            return err
        app = self._admin_app(ident, request.match_info["app_id"])
        if not app:
            return _nf()
        n = (await request.json()).get("release")
        if n not in app["releases"]:
            return _bad("No such release.")
        prev, app["current"] = app["current"], n
        self.store.save_app(app)
        self.store.audit("rollback", actor=ident["user"], app=app["app_id"], release=n, previous=prev)
        return web.json_response({"app_id": app["app_id"], "current": n})

    async def adm_acl(self, request):
        ident, err = self._guard(request, "publish")
        if err:
            return err
        app = self._admin_app(ident, request.match_info["app_id"])
        if not app:
            return _nf()
        b = await request.json()
        add, remove = list(b.get("add") or []), list(b.get("remove") or [])
        for n in add:
            if not self.store.get_user(str(n)):
                return _bad(f"No such user: {n}.")
        acl = [x for x in app["acl"] if x not in remove]
        acl += [x for x in add if x not in acl]
        app["acl"] = acl
        self.store.save_app(app)
        self.store.audit("acl", actor=ident["user"], app=app["app_id"], added=add or None, removed=remove or None)
        if remove:
            self.store.audit("revoke", actor=ident["user"], app=app["app_id"], target=remove)
        return web.json_response({"app_id": app["app_id"], "acl": acl})

    async def adm_token(self, request):
        ident = request["ident"]
        if not ident:
            return web.json_response({"error": "login_required"}, status=401)
        if request.method in UNSAFE and not self._csrf_ok(request, ident):
            return web.json_response({"error": "csrf"}, status=403)
        b = await request.json()
        scopes = set(b.get("scopes") or ())
        if not scopes or not scopes <= ident["scopes"]:
            return _bad("You can only issue tokens with scopes you have.")
        apps = b.get("apps")
        if ident["apps"] is not None and (not apps or not set(apps) <= ident["apps"]):
            return _bad("This token can only issue tokens for its own apps.")
        try:
            tok, p = self.tokens.issue(ident["user"], scopes, apps, int(b.get("ttl_s", 3600)))
        except ValueError as exc:
            return _bad(str(exc))
        self.store.audit("token_issue", actor=ident["user"], jti=p["jti"], scopes=sorted(scopes))
        return web.json_response({"token": tok, "jti": p["jti"], "expires": p["exp"]}, status=201)

    async def adm_revoke_token(self, request):
        ident, err = self._guard(request, "admin")
        if err:
            return err
        jti = str((await request.json()).get("jti", ""))[:64]
        self.store.revoke_token(jti)
        self.store.audit("revoke", actor=ident["user"], jti=jti)
        return web.json_response({"ok": True})

    async def adm_audit(self, request):
        ident, err = self._guard(request, "admin")
        if err:
            return err
        try:
            limit = int(request.query.get("limit", 200))
        except ValueError:
            limit = 200
        return web.json_response({"events": self.store.read_audit(limit)})

    async def adm_jobs(self, request):
        ident, err = self._guard(request, "admin")
        if err:
            return err
        return web.json_response({"jobs": sorted(self.jobs._jobs.values(), key=lambda j: -j["created"])[:500]})

    # ── owner Studio: the only proxy to the worker ──────────────────────────────────────────────────────────
    def _studio_ident(self, request):
        ident = request["ident"]
        if not ident or ident["role"] != "owner" or "admin" not in ident["scopes"]:
            return None
        return ident

    async def studio_root(self, request):
        if not self._studio_ident(request):
            return _nf()
        raise web.HTTPFound("/studio/")

    async def studio(self, request):
        ident = self._studio_ident(request)
        if not ident:
            return _nf()
        if request.method in UNSAFE and ident["via"] == "session" and not self._origin_ok(request):
            return web.json_response({"error": "csrf"}, status=403)
        path = "/" + request.match_info["tail"]
        qs = ("?" + request.query_string) if request.query_string else ""
        if request.headers.get("Upgrade", "").lower() == "websocket":
            return await self._studio_ws(request, path + qs)
        headers = {k: v for k, v in request.headers.items() if k.lower() not in HOP}
        sess = await self.worker.session()
        body = await request.read() if request.can_read_body else None
        async with sess.request(request.method, self.worker.base + path + qs, headers=headers, data=body,
                                allow_redirects=False, timeout=aiohttp.ClientTimeout(total=None, sock_read=600),
                                auto_decompress=True) as up:
            resp = web.StreamResponse(status=up.status)
            for k, v in up.headers.items():
                if k.lower() not in RESP_DROP:
                    if k.lower() == "location" and v.startswith("/"):
                        v = "/studio" + v
                    resp.headers[k] = v
            await resp.prepare(request)
            async for chunk in up.content.iter_chunked(1 << 16):
                await resp.write(chunk)
            await resp.write_eof()
            return resp

    async def _studio_ws(self, request, path_qs: str):
        down = web.WebSocketResponse(heartbeat=30, max_msg_size=64 * 1024 * 1024)
        await down.prepare(request)
        sess = await self.worker.session()
        url = self.worker.base.replace("http", "ws", 1) + path_qs
        try:
            async with sess.ws_connect(url, max_msg_size=64 * 1024 * 1024) as up:
                async def pump(src, dst, to_client: bool):
                    async for msg in src:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            await dst.send_str(msg.data)
                        elif msg.type == aiohttp.WSMsgType.BINARY:
                            await dst.send_bytes(msg.data)
                        else:
                            break
                t1 = asyncio.ensure_future(pump(up, down, True))
                t2 = asyncio.ensure_future(pump(down, up, False))
                await asyncio.wait({t1, t2}, return_when=asyncio.FIRST_COMPLETED)
                for t in (t1, t2):
                    t.cancel()
        except aiohttp.ClientError:
            pass
        await down.close()
        return down


def build(cfg: dict, *, worker: Worker | None = None) -> tuple[Gateway, web.Application]:
    gw = Gateway(cfg, worker=worker)
    return gw, gw.make_app()


__all__ = ["Gateway", "build", "DEFAULTS", "ERRORS"]
