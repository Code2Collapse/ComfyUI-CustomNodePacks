"""
mec_diagnostics_api.py — Backend for the MEC Diagnostics sidebar.

Doctor-style endpoints exposed under `/mec/diagnostics/*`:

    GET  /mec/diagnostics/recent           Recent execution events (ring buffer).
    GET  /mec/diagnostics/statistics       Aggregated counters per category / pattern.
    GET  /mec/diagnostics/patterns         Loaded pattern packs + counts.
    POST /mec/diagnostics/reload_patterns  Force pattern hot-reload.
    GET  /mec/diagnostics/settings         Current error_assistant settings.
    POST /mec/diagnostics/settings         Update settings (passthrough to error_assistant.save_settings).
    POST /mec/diagnostics/clear            Clear the in-memory ring buffer.
    GET  /mec/diagnostics/health           Liveness + counters (for sidebar status pill).

Coexists with ComfyUI-Doctor (`/doctor/*`) — separate namespace + IDs.

All responses use the Doctor-style envelope:
    { "success": True, "data": ... }                      OR
    { "success": False, "error": "<KEY>", "message": "<text>" }
"""
from __future__ import annotations

import logging
import os
import threading
import time
from collections import Counter, deque
from typing import Any, Deque, Dict

log = logging.getLogger("MEC.diagnostics_api")

# ---------------------------------------------------------------------
# Ring buffer: most-recent-N execution events
# ---------------------------------------------------------------------
_BUFFER_LIMIT = 500
_BUFFER_LOCK = threading.Lock()
_BUFFER: Deque[Dict[str, Any]] = deque(maxlen=_BUFFER_LIMIT)
_PATTERN_HITS: Counter[str] = Counter()
_CATEGORY_HITS: Counter[str] = Counter()
_FIRST_SEEN: Dict[str, float] = {}
_LAST_SEEN: Dict[str, float] = {}

# Background download jobs for Tier 2 GGUF models.
# job_id -> { id, repo, file, dest_dir, dest_path, bytes_done, total, status,
#             error, started_ts, finished_ts }
_DOWNLOAD_JOBS: Dict[str, Dict[str, Any]] = {}
_DOWNLOAD_LOCK = threading.Lock()


def record_event(event: Dict[str, Any]) -> None:
    """Insight executor wrapper calls this on every node_done / node_error.

    Also called by error_assistant.explain() so we can aggregate Tier-1 hits
    that happen in non-execution code paths (e.g. node validate).
    """
    ev = dict(event or {})
    ev.setdefault("ts", time.time())
    pid = ev.get("pattern_id")
    cat = ev.get("category")
    with _BUFFER_LOCK:
        _BUFFER.append(ev)
        if pid:
            _PATTERN_HITS[pid] += 1
            _LAST_SEEN[pid] = ev["ts"]
            _FIRST_SEEN.setdefault(pid, ev["ts"])
        if cat:
            _CATEGORY_HITS[cat] += 1


def _envelope_ok(data: Any) -> Dict[str, Any]:
    return {"success": True, "data": data}


def _envelope_err(error_key: str, message: str) -> Dict[str, Any]:
    return {"success": False, "error": error_key, "message": message}


# ---------------------------------------------------------------------
# Route registration
# ---------------------------------------------------------------------
class _DualRoutes:
    """Wraps aiohttp routes to register each /mec/* path also under /c2c/*.

    Back-compat: existing /mec/diagnostics/* URLs keep working forever.
    Forward: the C2C UI rebrand can call /c2c/diagnostics/* identically.
    """
    def __init__(self, inner):
        self._inner = inner

    def _decorator(self, method_name, path, **kwargs):
        inner_method = getattr(self._inner, method_name)
        primary = inner_method(path, **kwargs)
        alias = None
        if path.startswith("/mec/"):
            alias_path = "/c2c/" + path[len("/mec/"):]
            alias = inner_method(alias_path, **kwargs)

        def register(fn):
            primary(fn)
            if alias is not None:
                alias(fn)
            return fn
        return register

    def get(self, path, **kw):    return self._decorator("get", path, **kw)
    def post(self, path, **kw):   return self._decorator("post", path, **kw)
    def put(self, path, **kw):    return self._decorator("put", path, **kw)
    def delete(self, path, **kw): return self._decorator("delete", path, **kw)
    def patch(self, path, **kw):  return self._decorator("patch", path, **kw)

    def __getattr__(self, name):
        return getattr(self._inner, name)


def register_routes(server) -> None:
    try:
        from aiohttp import web
    except Exception as e:
        log.warning("[mec_diagnostics] aiohttp unavailable: %s", e)
        return
    routes = _DualRoutes(server.routes)

    @routes.get("/mec/diagnostics/health")
    async def _health(_req):  # noqa: ARG001
        with _BUFFER_LOCK:
            buf_len = len(_BUFFER)
            errors = sum(1 for e in _BUFFER if e.get("type") == "node_error"
                         or e.get("severity") in ("error", "warn"))
        return web.json_response(_envelope_ok({
            "buffer_len": buf_len,
            "buffer_limit": _BUFFER_LIMIT,
            "error_count": errors,
            "patterns_distinct": len(_PATTERN_HITS),
        }))

    @routes.get("/mec/diagnostics/recent")
    async def _recent(req):
        try:
            limit = max(1, min(_BUFFER_LIMIT, int(req.query.get("limit", "100"))))
        except Exception:
            limit = 100
        kind = req.query.get("kind")  # "error" | "all"
        with _BUFFER_LOCK:
            items = list(_BUFFER)
        if kind == "error":
            items = [e for e in items if e.get("type") == "node_error"
                     or e.get("severity") in ("error", "warn")]
        return web.json_response(_envelope_ok(items[-limit:][::-1]))

    @routes.get("/mec/diagnostics/statistics")
    async def _statistics(_req):  # noqa: ARG001
        with _BUFFER_LOCK:
            hits = _PATTERN_HITS.most_common(50)
            categories = _CATEGORY_HITS.most_common()
            patterns = [
                {
                    "pattern_id": pid,
                    "count": cnt,
                    "first_seen": _FIRST_SEEN.get(pid),
                    "last_seen": _LAST_SEEN.get(pid),
                }
                for pid, cnt in hits
            ]
        return web.json_response(_envelope_ok({
            "patterns": patterns,
            "categories": [{"category": c, "count": n} for c, n in categories],
            "total_events": sum(_PATTERN_HITS.values()),
        }))

    @routes.get("/mec/diagnostics/patterns")
    async def _patterns(_req):  # noqa: ARG001
        try:
            from . import error_assistant as _ea  # type: ignore
        except Exception:
            try:
                from nodes import error_assistant as _ea  # type: ignore
            except Exception as e:
                return web.json_response(_envelope_err(
                    "error_assistant_unavailable", str(e)), status=500)
        try:
            pats = _ea._get_patterns()
            data = [
                {
                    "id": p.name,
                    "category": p.category,
                    "priority": p.priority,
                    "confidence": p.confidence,
                    "source": p.source,
                    "exc_types": list(p.exc_types),
                }
                for p in pats
            ]
            return web.json_response(_envelope_ok({
                "count": len(data),
                "patterns": data,
            }))
        except Exception as e:
            log.exception("[mec_diagnostics] /patterns failed")
            return web.json_response(_envelope_err(
                "patterns_load_failed", f"{type(e).__name__}: {e}"), status=500)

    @routes.post("/mec/diagnostics/reload_patterns")
    async def _reload(_req):  # noqa: ARG001
        try:
            from . import error_assistant as _ea  # type: ignore
        except Exception:
            try:
                from nodes import error_assistant as _ea  # type: ignore
            except Exception as e:
                return web.json_response(_envelope_err(
                    "error_assistant_unavailable", str(e)), status=500)
        try:
            n = _ea.reload_patterns()
            return web.json_response(_envelope_ok({"count": n}))
        except Exception as e:
            log.exception("[mec_diagnostics] /reload_patterns failed")
            return web.json_response(_envelope_err(
                "reload_failed", f"{type(e).__name__}: {e}"), status=500)

    @routes.get("/mec/diagnostics/settings")
    async def _get_settings(_req):  # noqa: ARG001
        try:
            from . import error_assistant as _ea  # type: ignore
        except Exception:
            try:
                from nodes import error_assistant as _ea  # type: ignore
            except Exception as e:
                return web.json_response(_envelope_err(
                    "error_assistant_unavailable", str(e)), status=500)
        try:
            return web.json_response(_envelope_ok(_ea.load_settings()))
        except Exception as e:
            return web.json_response(_envelope_err(
                "settings_read_failed", str(e)), status=500)

    @routes.post("/mec/diagnostics/settings")
    async def _set_settings(req):
        try:
            payload = await req.json()
        except Exception as e:
            return web.json_response(_envelope_err(
                "bad_json", str(e)), status=400)
        try:
            from . import error_assistant as _ea  # type: ignore
        except Exception:
            try:
                from nodes import error_assistant as _ea  # type: ignore
            except Exception as e:
                return web.json_response(_envelope_err(
                    "error_assistant_unavailable", str(e)), status=500)
        try:
            _ea.save_settings(payload or {})
            return web.json_response(_envelope_ok(_ea.load_settings()))
        except Exception as e:
            return web.json_response(_envelope_err(
                "settings_write_failed", str(e)), status=500)

    @routes.post("/mec/diagnostics/clear")
    async def _clear(_req):  # noqa: ARG001
        with _BUFFER_LOCK:
            _BUFFER.clear()
            _PATTERN_HITS.clear()
            _CATEGORY_HITS.clear()
            _FIRST_SEEN.clear()
            _LAST_SEEN.clear()
        return web.json_response(_envelope_ok({"cleared": True}))

    # -----------------------------------------------------------------
    # Error Assistant — per-tier readiness + smoke test + secrets I/O
    # -----------------------------------------------------------------
    def _import_ea():
        try:
            from . import error_assistant as _ea  # type: ignore
            return _ea
        except Exception:
            try:
                from nodes import error_assistant as _ea  # type: ignore
                return _ea
            except Exception:
                return None

    def _import_secrets():
        try:
            from . import secrets_store as _ss  # type: ignore
            return _ss
        except Exception:
            try:
                from nodes import secrets_store as _ss  # type: ignore
                return _ss
            except Exception:
                return None

    def _import_local_llm():
        try:
            from . import local_llm as _ll  # type: ignore
            return _ll
        except Exception:
            try:
                from nodes import local_llm as _ll  # type: ignore
                return _ll
            except Exception:
                return None

    def _import_ollama_llm():
        try:
            from . import ollama_llm as _ol  # type: ignore
            return _ol
        except Exception:
            try:
                from nodes import ollama_llm as _ol  # type: ignore
                return _ol
            except Exception:
                return None

    @routes.get("/mec/diagnostics/error_assistant/status")
    async def _ea_status(_req):  # noqa: ARG001
        ea = _import_ea()
        ss = _import_secrets()
        ll = _import_local_llm()
        ol = _import_ollama_llm()
        s = ea.load_settings() if ea else {}
        # Tier 1
        try:
            t1_count = len(ea._get_patterns()) if ea else 0
            t1 = {"ready": t1_count > 0, "detail": f"{t1_count} pattern(s) loaded"}
        except Exception as e:
            t1 = {"ready": False, "detail": f"pattern load failed: {e}"}
        # Tier 2 — depends on selected backend (llamacpp | ollama).
        backend = (s.get("tier2_backend") or "llamacpp").strip().lower()
        t2_detail = []
        t2_ready = True
        if backend == "ollama":
            url = s.get("ollama_url") or "http://localhost:11434"
            t2_detail.append(f"backend: ollama @ {url}")
            if ol is None:
                t2_ready = False
                t2_detail.append("ollama_llm module unavailable")
            elif ol.is_available(url):
                models = ol.list_models(url)
                t2_detail.append(f"models: {len(models)}")
                want = s.get("ollama_model") or ""
                if want and want not in models:
                    t2_ready = False
                    t2_detail.append(f"model '{want}' NOT pulled — run "
                                     f"`ollama pull {want}`")
            else:
                t2_ready = False
                t2_detail.append("ollama daemon unreachable")
        else:
            t2_detail.append("backend: llama-cpp-python")
            if ll is not None:
                mid = s.get("local_model", "")
                try:
                    path = ll._resolve_model_path(mid) if hasattr(ll, "_resolve_model_path") else None
                except Exception:
                    path = None
                if path:
                    t2_detail.append(f"model: {os.path.basename(path)}")
                else:
                    t2_ready = False
                    t2_detail.append(f"model not found ({mid or 'unset'})")
                try:
                    import llama_cpp  # noqa: F401
                    t2_detail.append("llama-cpp-python ok")
                except Exception:
                    t2_ready = False
                    t2_detail.append("llama-cpp-python missing")
            else:
                t2_ready = False
                t2_detail.append("local_llm module unavailable")
        t2 = {"ready": t2_ready, "detail": "; ".join(t2_detail)}
        # Tier 3: API key for selected provider.
        prov = s.get("cloud_provider", "openai")
        if ss is not None:
            has_key = bool(ss.has_key_for(prov))
        else:
            has_key = False
        t3 = {
            "ready": has_key,
            "detail": (f"{prov}: API key set" if has_key
                       else f"{prov}: API key MISSING — set it in Tier 3 below"),
        }
        return web.json_response(_envelope_ok({
            "tier1": t1,
            "tier2": t2,
            "tier3": t3,
            "settings": s,
        }))

    @routes.post("/mec/diagnostics/error_assistant/test_tier")
    async def _ea_test_tier(req):
        try:
            payload = await req.json()
        except Exception as e:
            return web.json_response(_envelope_err("bad_json", str(e)), status=400)
        tier = int(payload.get("tier", 0))
        if tier not in (1, 2, 3):
            return web.json_response(_envelope_err(
                "bad_tier", "tier must be 1, 2, or 3"), status=400)
        ea = _import_ea()
        if ea is None:
            return web.json_response(_envelope_err(
                "error_assistant_unavailable", "module not importable"), status=500)
        # Construct a representative test exception. The pattern matcher will
        # match on the message; LLMs get the prompt regardless.
        try:
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
        except RuntimeError as test_exc:
            t0 = time.time()
            try:
                if tier == 1:
                    res = ea.explain(test_exc, mode="deterministic_only")
                elif tier == 2:
                    res = ea.explain(test_exc, mode="local_only")
                else:
                    res = ea.explain(test_exc, mode="cloud_only")
                elapsed_ms = int((time.time() - t0) * 1000)
                ok = res.get("tier") == tier
                return web.json_response(_envelope_ok({
                    "tier_requested": tier,
                    "tier_returned": res.get("tier"),
                    "ok": ok,
                    "elapsed_ms": elapsed_ms,
                    "headline": res.get("headline", "")[:200],
                    "preview": (res.get("cause") or "")[:240],
                }))
            except Exception as e:
                return web.json_response(_envelope_err(
                    "tier_test_failed", f"{type(e).__name__}: {e}"), status=500)

    @routes.get("/mec/diagnostics/error_assistant/secrets")
    async def _ea_get_secret(req):
        ss = _import_secrets()
        if ss is None:
            return web.json_response(_envelope_err(
                "secrets_unavailable", "secrets_store not importable"), status=500)
        prov = req.query.get("provider", "openai")
        try:
            key = ss.get_key(prov)
        except Exception:
            key = None
        if not key:
            return web.json_response(_envelope_ok({
                "provider": prov, "set": False, "preview": ""}))
        # Mask: show first 3 + last 4 chars only.
        if len(key) <= 8:
            preview = "*" * len(key)
        else:
            preview = key[:3] + "*" * max(4, len(key) - 7) + key[-4:]
        return web.json_response(_envelope_ok({
            "provider": prov, "set": True, "preview": preview,
        }))

    @routes.post("/mec/diagnostics/error_assistant/secrets")
    async def _ea_set_secret(req):
        try:
            payload = await req.json()
        except Exception as e:
            return web.json_response(_envelope_err("bad_json", str(e)), status=400)
        prov = (payload.get("provider") or "").strip()
        key = payload.get("api_key")
        if not prov:
            return web.json_response(_envelope_err(
                "bad_provider", "provider is required"), status=400)
        ss = _import_secrets()
        if ss is None:
            return web.json_response(_envelope_err(
                "secrets_unavailable", "secrets_store not importable"), status=500)
        try:
            if key is None or key == "":
                ss.delete_key(prov)
                return web.json_response(_envelope_ok({"provider": prov, "set": False}))
            ss.set_key(prov, str(key))
            return web.json_response(_envelope_ok({"provider": prov, "set": True}))
        except Exception as e:
            return web.json_response(_envelope_err(
                "secrets_write_failed", f"{type(e).__name__}: {e}"), status=500)

    # -----------------------------------------------------------------
    # Tier 2 — Ollama daemon helpers
    # -----------------------------------------------------------------
    @routes.get("/mec/diagnostics/ollama/tags")
    async def _ol_tags(req):
        ol = _import_ollama_llm()
        if ol is None:
            return web.json_response(_envelope_err(
                "ollama_unavailable", "ollama_llm module not importable"),
                status=500)
        url = req.query.get("url")
        if not url:
            ea = _import_ea()
            s = ea.load_settings() if ea else {}
            url = s.get("ollama_url") or "http://localhost:11434"
        try:
            avail = ol.is_available(url)
            models = ol.list_models(url) if avail else []
            return web.json_response(_envelope_ok({
                "url": url,
                "available": avail,
                "models": models,
            }))
        except Exception as e:
            return web.json_response(_envelope_err(
                "ollama_tags_failed", f"{type(e).__name__}: {e}"), status=500)

    # -----------------------------------------------------------------
    # Tier 2 — llama.cpp GGUF model scanner
    # -----------------------------------------------------------------
    @routes.get("/mec/diagnostics/local_llm/scan")
    async def _llm_scan(_req):  # noqa: ARG001
        ll = _import_local_llm()
        try:
            if ll is not None:
                dirs = ll._candidate_dirs()
            else:
                # Replicate _candidate_dirs logic when module is unavailable.
                here = os.path.dirname(os.path.abspath(__file__))
                pack_root = os.path.dirname(here)
                dirs = [os.path.join(pack_root, "user", "models")]
                try:
                    import folder_paths  # type: ignore  # noqa: F401
                    for _k in ("llm", "LLM", "language_models", "text_encoders", "clip"):
                        try:
                            dirs.extend(folder_paths.get_folder_paths(_k))
                        except Exception:
                            pass
                    try:
                        dirs.append(os.path.join(folder_paths.models_dir, "llm"))
                    except Exception:
                        pass
                except Exception:
                    pass
                dirs = [p for p in dirs if p and os.path.isdir(p)]
            installed = []
            for d in dirs:
                try:
                    for fn in os.listdir(d):
                        if fn.lower().endswith(".gguf"):
                            fp = os.path.join(d, fn)
                            try:
                                sz = round(os.path.getsize(fp) / (1024 * 1024), 1)
                            except Exception:
                                sz = 0
                            installed.append({
                                "filename": fn,
                                "path": fp,
                                "dir": d,
                                "size_mb": sz,
                            })
                except Exception:
                    continue
            return web.json_response(_envelope_ok({
                "dirs": dirs,
                "installed": installed,
            }))
        except Exception as e:
            return web.json_response(
                _envelope_err("scan_failed", f"{type(e).__name__}: {e}"),
                status=500)

    # -----------------------------------------------------------------
    # Tier 2 — GGUF download from HuggingFace
    # -----------------------------------------------------------------
    def _llm_dest_dir():
        """First writable directory for GGUFs."""
        ll = _import_local_llm()
        if ll is not None and hasattr(ll, "_candidate_dirs"):
            try:
                cands = ll._candidate_dirs()
            except Exception:
                cands = []
        else:
            cands = []
        try:
            import folder_paths  # type: ignore
            try:
                cands.append(os.path.join(folder_paths.models_dir, "llm"))
            except Exception:
                pass
        except Exception:
            pass
        for d in cands:
            if not d:
                continue
            try:
                os.makedirs(d, exist_ok=True)
                if os.access(d, os.W_OK):
                    return d
            except Exception:
                continue
        # Last resort: pack/user/models
        here = os.path.dirname(os.path.abspath(__file__))
        pack_root = os.path.dirname(here)
        fallback = os.path.join(pack_root, "user", "models")
        os.makedirs(fallback, exist_ok=True)
        return fallback

    def _do_download(job_id: str, repo: str, fname: str, dest_dir: str):
        import urllib.request
        import urllib.error
        rec = _DOWNLOAD_JOBS[job_id]
        dest_path = os.path.join(dest_dir, fname)
        rec["dest_path"] = dest_path
        # If file already exists and is non-empty, mark done.
        if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
            with _DOWNLOAD_LOCK:
                rec["status"] = "exists"
                rec["bytes_done"] = os.path.getsize(dest_path)
                rec["total"] = rec["bytes_done"]
                rec["finished_ts"] = time.time()
            return
        # Prefer huggingface_hub when present (handles mirrors, retries).
        try:
            from huggingface_hub import hf_hub_download  # type: ignore
            tmp = hf_hub_download(
                repo_id=repo,
                filename=fname,
                local_dir=dest_dir,
                local_dir_use_symlinks=False,
            )
            # hf_hub_download returns path. Move to canonical location if needed.
            if os.path.abspath(tmp) != os.path.abspath(dest_path):
                try:
                    os.replace(tmp, dest_path)
                except Exception:
                    dest_path = tmp
                    rec["dest_path"] = dest_path
            sz = os.path.getsize(dest_path)
            with _DOWNLOAD_LOCK:
                rec["bytes_done"] = sz
                rec["total"] = sz
                rec["status"] = "done"
                rec["finished_ts"] = time.time()
            return
        except ImportError:
            pass
        except Exception as e:
            # Fall through to urllib if HF lib path fails for any reason.
            log.warning("[mec_diagnostics] hf_hub_download failed (%s); fallback to urllib", e)
        # urllib fallback (streamed)
        url = f"https://huggingface.co/{repo}/resolve/main/{fname}"
        tmp_path = dest_path + ".part"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "MEC-Diagnostics/1.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                total = int(resp.headers.get("Content-Length", "0") or "0")
                with _DOWNLOAD_LOCK:
                    rec["total"] = total
                with open(tmp_path, "wb") as f:
                    chunk = 1024 * 256
                    while True:
                        buf = resp.read(chunk)
                        if not buf:
                            break
                        f.write(buf)
                        with _DOWNLOAD_LOCK:
                            rec["bytes_done"] += len(buf)
                            if rec.get("cancel"):
                                raise RuntimeError("cancelled")
            os.replace(tmp_path, dest_path)
            with _DOWNLOAD_LOCK:
                rec["status"] = "done"
                rec["finished_ts"] = time.time()
        except Exception as e:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass
            with _DOWNLOAD_LOCK:
                rec["status"] = "error"
                rec["error"] = f"{type(e).__name__}: {e}"
                rec["finished_ts"] = time.time()

    @routes.post("/mec/diagnostics/local_llm/download")
    async def _llm_download(req):
        import uuid
        try:
            payload = await req.json()
        except Exception as e:
            return web.json_response(_envelope_err("bad_json", str(e)), status=400)
        cid = (payload.get("id") or "").strip()
        repo = (payload.get("repo") or "").strip()
        fname = (payload.get("file") or "").strip()
        if not repo or not fname:
            return web.json_response(_envelope_err(
                "missing_repo_or_file",
                "Both 'repo' and 'file' are required (e.g. {repo:'bartowski/Qwen2.5-0.5B-Instruct-GGUF', file:'Qwen2.5-0.5B-Instruct-Q4_K_M.gguf'})."
            ), status=400)
        # Refuse anything that isn't .gguf.
        if not fname.lower().endswith(".gguf"):
            return web.json_response(_envelope_err(
                "not_gguf", "Only .gguf filenames are accepted."), status=400)
        # Refuse path traversal in filename.
        if "/" in fname or "\\" in fname or ".." in fname:
            return web.json_response(_envelope_err(
                "bad_filename", "Filename must be a bare name with no path components."),
                status=400)
        dest_dir = _llm_dest_dir()
        job_id = uuid.uuid4().hex[:12]
        rec = {
            "job_id": job_id,
            "id": cid,
            "repo": repo,
            "file": fname,
            "dest_dir": dest_dir,
            "dest_path": os.path.join(dest_dir, fname),
            "bytes_done": 0,
            "total": 0,
            "status": "running",
            "error": "",
            "started_ts": time.time(),
            "finished_ts": 0.0,
            "cancel": False,
        }
        with _DOWNLOAD_LOCK:
            _DOWNLOAD_JOBS[job_id] = rec
        th = threading.Thread(
            target=_do_download,
            args=(job_id, repo, fname, dest_dir),
            daemon=True,
            name=f"mec-dl-{job_id}",
        )
        th.start()
        return web.json_response(_envelope_ok({
            "job_id": job_id,
            "dest_dir": dest_dir,
            "dest_path": rec["dest_path"],
        }))

    @routes.get("/mec/diagnostics/local_llm/download_progress")
    async def _llm_download_progress(req):
        job_id = (req.query.get("job_id") or "").strip()
        if not job_id:
            return web.json_response(_envelope_err(
                "missing_job_id", "job_id query param required"), status=400)
        with _DOWNLOAD_LOCK:
            rec = _DOWNLOAD_JOBS.get(job_id)
            if rec is None:
                return web.json_response(_envelope_err(
                    "unknown_job", "no such job_id"), status=404)
            out = {k: v for k, v in rec.items() if k != "cancel"}
        # Add percent.
        total = out.get("total") or 0
        done = out.get("bytes_done") or 0
        out["percent"] = round((done / total * 100.0) if total > 0 else 0.0, 1)
        return web.json_response(_envelope_ok(out))

    @routes.post("/mec/diagnostics/local_llm/download_cancel")
    async def _llm_download_cancel(req):
        try:
            payload = await req.json()
        except Exception as e:
            return web.json_response(_envelope_err("bad_json", str(e)), status=400)
        job_id = (payload.get("job_id") or "").strip()
        with _DOWNLOAD_LOCK:
            rec = _DOWNLOAD_JOBS.get(job_id)
            if rec is None:
                return web.json_response(_envelope_err(
                    "unknown_job", "no such job_id"), status=404)
            rec["cancel"] = True
        return web.json_response(_envelope_ok({"job_id": job_id, "cancelling": True}))

    # -----------------------------------------------------------------
    # Tier 2 — HuggingFace GGUF discovery (live search + file listing)
    # -----------------------------------------------------------------
    # Server-side proxy to https://huggingface.co/api/models so the JS
    # avoids CORS hassles and we can cache + sanity-filter the response.
    _HF_SEARCH_CACHE: Dict[str, Any] = {}
    _HF_FILES_CACHE: Dict[str, Any] = {}
    _HF_CACHE_LOCK = threading.Lock()
    _HF_CACHE_TTL = 300.0  # 5 min

    def _hf_get_json(url: str, timeout: float = 12.0):
        import json as _json
        import urllib.request as _ur
        import urllib.error as _ue
        req = _ur.Request(url, headers={
            "User-Agent": "MEC-Diagnostics/1.0 (+huggingface-gguf-search)",
            "Accept": "application/json",
        })
        try:
            with _ur.urlopen(req, timeout=timeout) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
            return _json.loads(raw)
        except _ue.HTTPError as e:
            raise RuntimeError(f"HTTP {e.code} from {url}") from e
        except Exception as e:
            raise RuntimeError(f"{type(e).__name__}: {e}") from e

    @routes.get("/mec/diagnostics/hf_gguf_search")
    async def _hf_search(req):
        import asyncio
        q = (req.query.get("q") or "").strip()
        try:
            limit = max(1, min(50, int(req.query.get("limit") or "20")))
        except Exception:
            limit = 20
        cache_key = f"{q.lower()}|{limit}"
        now = time.time()
        with _HF_CACHE_LOCK:
            ent = _HF_SEARCH_CACHE.get(cache_key)
            if ent and (now - ent["ts"]) < _HF_CACHE_TTL:
                return web.json_response(_envelope_ok({**ent["data"], "cached": True}))
        params = [
            "filter=gguf",
            "sort=downloads",
            "direction=-1",
            f"limit={limit}",
            "full=false",
        ]
        if q:
            from urllib.parse import quote
            params.insert(0, f"search={quote(q)}")
        url = "https://huggingface.co/api/models?" + "&".join(params)
        try:
            raw = await asyncio.get_event_loop().run_in_executor(
                None, _hf_get_json, url)
        except Exception as e:
            return web.json_response(_envelope_err(
                "hf_search_failed", str(e)), status=502)
        if not isinstance(raw, list):
            return web.json_response(_envelope_err(
                "hf_search_bad_shape", "expected list response from HF Hub"),
                status=502)
        results = []
        for r in raw[:limit]:
            if not isinstance(r, dict):
                continue
            results.append({
                "id": r.get("id") or r.get("modelId") or "",
                "downloads": int(r.get("downloads") or 0),
                "likes": int(r.get("likes") or 0),
                "last_modified": r.get("lastModified") or "",
                "pipeline_tag": r.get("pipeline_tag") or "",
                "tags": [t for t in (r.get("tags") or []) if isinstance(t, str)][:20],
            })
        out = {"query": q, "count": len(results), "results": results, "cached": False}
        with _HF_CACHE_LOCK:
            _HF_SEARCH_CACHE[cache_key] = {"ts": now, "data": out}
        return web.json_response(_envelope_ok(out))

    @routes.get("/mec/diagnostics/hf_gguf_files")
    async def _hf_files(req):
        import asyncio
        repo = (req.query.get("repo") or "").strip()
        if not repo or "/" not in repo or ".." in repo:
            return web.json_response(_envelope_err(
                "bad_repo", "repo must look like 'owner/name'"), status=400)
        now = time.time()
        with _HF_CACHE_LOCK:
            ent = _HF_FILES_CACHE.get(repo)
            if ent and (now - ent["ts"]) < _HF_CACHE_TTL:
                return web.json_response(_envelope_ok({**ent["data"], "cached": True}))
        url = f"https://huggingface.co/api/models/{repo}"
        try:
            raw = await asyncio.get_event_loop().run_in_executor(
                None, _hf_get_json, url)
        except Exception as e:
            return web.json_response(_envelope_err(
                "hf_files_failed", str(e)), status=502)
        if not isinstance(raw, dict):
            return web.json_response(_envelope_err(
                "hf_files_bad_shape", "expected object response"), status=502)
        siblings = raw.get("siblings") or []
        files = []
        for s in siblings:
            if not isinstance(s, dict):
                continue
            fn = s.get("rfilename") or ""
            if not fn.lower().endswith(".gguf"):
                continue
            if "/" in fn or "\\" in fn:
                continue  # skip nested files; download POST refuses them anyway
            files.append({
                "file": fn,
                "size": int(s.get("size") or 0) if isinstance(s.get("size"), (int, float)) else 0,
            })
        files.sort(key=lambda r: (r["size"] or 1 << 62, r["file"]))
        out = {
            "repo": repo,
            "description": (raw.get("description") or "")[:400],
            "downloads": int(raw.get("downloads") or 0),
            "likes": int(raw.get("likes") or 0),
            "last_modified": raw.get("lastModified") or "",
            "files": files,
            "cached": False,
        }
        with _HF_CACHE_LOCK:
            _HF_FILES_CACHE[repo] = {"ts": now, "data": out}
        return web.json_response(_envelope_ok(out))

    # -----------------------------------------------------------------
    # Tier 1 — custom user patterns (add / list / remove / hot-reload)
    # -----------------------------------------------------------------
    def _user_patterns_path():
        here = os.path.dirname(os.path.abspath(__file__))
        pack_root = os.path.dirname(here)
        d = os.path.join(pack_root, "patterns", "user")
        os.makedirs(d, exist_ok=True)
        return os.path.join(d, "user_patterns.json")

    def _load_user_patterns():
        import json
        p = _user_patterns_path()
        if not os.path.exists(p):
            return {"patterns": []}
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict) or not isinstance(data.get("patterns"), list):
                return {"patterns": []}
            return data
        except Exception:
            return {"patterns": []}

    def _save_user_patterns(data):
        import json
        p = _user_patterns_path()
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @routes.get("/mec/diagnostics/patterns/custom")
    async def _custom_list(_req):  # noqa: ARG001
        return web.json_response(_envelope_ok(_load_user_patterns()))

    @routes.post("/mec/diagnostics/patterns/custom")
    async def _custom_add(req):
        import re as _re
        try:
            payload = await req.json()
        except Exception as e:
            return web.json_response(_envelope_err("bad_json", str(e)), status=400)
        pid = str(payload.get("id") or "").strip()
        regex = str(payload.get("regex") or "").strip()
        if not pid or not regex:
            return web.json_response(_envelope_err(
                "missing_field", "id and regex are required"), status=400)
        try:
            _re.compile(regex)
        except _re.error as e:
            return web.json_response(_envelope_err(
                "bad_regex", f"invalid regex: {e}"), status=400)
        entry = {
            "id": pid,
            "regex": regex,
            "cause": str(payload.get("cause") or ""),
            "fixes": list(payload.get("fixes") or []),
            "category": str(payload.get("category") or "user"),
            "exc_types": list(payload.get("exc_types") or []),
            "priority": int(payload.get("priority") or 50),
            "confidence": float(payload.get("confidence") or 0.8),
        }
        data = _load_user_patterns()
        # replace if id exists, else append
        out = [p for p in data["patterns"] if p.get("id") != pid]
        out.append(entry)
        data["patterns"] = out
        _save_user_patterns(data)
        # hot reload
        ea = _import_ea()
        n = ea.reload_patterns() if ea else 0
        return web.json_response(_envelope_ok({
            "added": entry, "total_patterns": n,
        }))

    @routes.delete("/mec/diagnostics/patterns/custom")
    async def _custom_remove(req):
        pid = req.query.get("id")
        if not pid:
            return web.json_response(_envelope_err(
                "missing_id", "query param `id` is required"), status=400)
        data = _load_user_patterns()
        before = len(data["patterns"])
        data["patterns"] = [p for p in data["patterns"] if p.get("id") != pid]
        removed = before - len(data["patterns"])
        _save_user_patterns(data)
        ea = _import_ea()
        n = ea.reload_patterns() if ea else 0
        return web.json_response(_envelope_ok({
            "removed": removed, "total_patterns": n,
        }))

    log.info("[mec_diagnostics] /mec/diagnostics/* routes registered")


# ---------------------------------------------------------------------
# Insight bridge: forward node_done / node_error events into our ring.
# ---------------------------------------------------------------------
def install_insight_bridge() -> bool:
    """Wraps `insight._emit` so every event is ALSO recorded for the sidebar.
    Idempotent."""
    try:
        from . import insight as _ins  # type: ignore
    except Exception:
        try:
            from nodes import insight as _ins  # type: ignore
        except Exception as e:
            log.debug("[mec_diagnostics] insight unavailable: %s", e)
            return False
    if getattr(_ins, "_MEC_DIAG_BRIDGED", False):
        return True
    orig_emit = _ins._emit

    def bridged_emit(event):
        try:
            record_event(event)
        except Exception:  # never break telemetry on observer failure
            pass
        try:
            return orig_emit(event)
        except Exception:
            pass

    _ins._emit = bridged_emit
    _ins._MEC_DIAG_BRIDGED = True
    log.info("[mec_diagnostics] insight bridge installed")
    return True
