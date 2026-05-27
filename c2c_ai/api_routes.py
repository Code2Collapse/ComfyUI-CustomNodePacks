"""HTTP routes exposed under ``/c2c/ai/*`` for the JS layer to consume.

    GET  /c2c/ai/status                 → router status (backends + policies + cost)
    GET  /c2c/ai/cost                   → cost snapshot
    POST /c2c/ai/cost/cap               → set daily cap          {"usd": 1.0}
    POST /c2c/ai/probe                  → force refresh of all health probes
    POST /c2c/ai/ask                    → non-streaming ask      {"feature","messages","..."}
    POST /c2c/ai/stream                 → SSE stream             same body
    GET  /c2c/ai/keys/list              → list canonical key names currently in keychain
    POST /c2c/ai/keys/set               → set one key            {"name","value"}
    POST /c2c/ai/keys/delete            → delete one key         {"name"}
    POST /c2c/ai/keys/import_txt        → import from txt file   {"path"}
    GET  /c2c/ai/local/detect           → probe known local servers
    GET  /c2c/ai/policy                 → list per-feature policy
    POST /c2c/ai/policy                 → override one           {"feature","policy"}
    GET  /c2c/ai/config                 → ai_config.json contents
    POST /c2c/ai/config                 → replace ai_config.json (also re-bootstraps)
    POST /c2c/ai/backends/test          → call .probe() on one backend, return result
    GET  /c2c/ai/prompts                → list registered prompt templates (name, version, sha256, vars)
    POST /c2c/ai/prompts/render         → render one             {"name","vars":{...}}
    GET  /c2c/ai/prompts/verify         → run golden-hash regression on every template
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path

log = logging.getLogger("c2c_ai.api")


def register_routes(server) -> None:
    try:
        from aiohttp import web
    except Exception as exc:
        log.error("aiohttp unavailable: %s", exc)
        return

    from . import bootstrap as bs
    from . import keychain as kc
    from . import policy as policy_mod
    from .cost_meter import get_meter
    from .router import get_router
    from .types import Message, Policy, Sensitivity, Capability
    from .backends.openai_compat import detect_local_servers

    routes = server.routes

    def _ok(data):
        return web.json_response({"success": True, "data": data})

    def _err(key: str, msg: str, code: int = 400):
        return web.json_response(
            {"success": False, "error": key, "message": msg}, status=code)

    # ---------------------------------------------------------- status
    @routes.get("/c2c/ai/status")
    async def _status(_req):
        return _ok(get_router().status())

    @routes.get("/c2c/ai/cost")
    async def _cost(_req):
        return _ok(get_meter().snapshot())

    @routes.post("/c2c/ai/cost/cap")
    async def _cost_cap(req):
        body = await req.json()
        usd = float(body.get("usd", 1.0))
        get_meter().set_daily_cap(usd)
        return _ok(get_meter().snapshot())

    @routes.post("/c2c/ai/probe")
    async def _probe(_req):
        loop = asyncio.get_event_loop()
        data = await loop.run_in_executor(None, get_router().probe_all)
        return _ok(data)

    # --------------------------------------------------------------- ask
    @routes.post("/c2c/ai/ask")
    async def _ask(req):
        body = await req.json()
        feature = body.get("feature") or "anonymous"
        msgs_raw = body.get("messages") or []
        if not msgs_raw:
            return _err("EMPTY_MESSAGES", "messages array is required")
        msgs = [Message(role=m["role"], content=m["content"]) for m in msgs_raw]

        kwargs = {}
        if "sensitivity" in body:
            kwargs["sensitivity"] = Sensitivity(body["sensitivity"])
        if "policy_override" in body and body["policy_override"]:
            kwargs["policy_override"] = Policy(body["policy_override"])
        if "max_tokens" in body:
            kwargs["max_tokens"] = int(body["max_tokens"])
        if "temperature" in body:
            kwargs["temperature"] = float(body["temperature"])
        if body.get("require_vision"):
            kwargs["required"] = {Capability.CHAT, Capability.VISION}

        loop = asyncio.get_event_loop()
        try:
            resp = await loop.run_in_executor(
                None, lambda: get_router().ask(feature, msgs, **kwargs))
        except Exception as exc:
            return _err("DISPATCH_FAILED", str(exc), code=500)
        return _ok({
            "text": resp.text,
            "backend_id": resp.backend_id,
            "model": resp.model,
            "input_tokens": resp.input_tokens,
            "output_tokens": resp.output_tokens,
            "cost_usd": resp.cost_usd,
            "latency_ms": resp.latency_ms,
            "redacted": resp.redacted,
        })

    # ----------------------------------------------------------- stream (SSE)
    @routes.post("/c2c/ai/stream")
    async def _stream(req):
        body = await req.json()
        feature = body.get("feature") or "anonymous"
        msgs_raw = body.get("messages") or []
        if not msgs_raw:
            return _err("EMPTY_MESSAGES", "messages array is required")
        msgs = [Message(role=m["role"], content=m["content"]) for m in msgs_raw]

        kwargs = {}
        if "sensitivity" in body:
            kwargs["sensitivity"] = Sensitivity(body["sensitivity"])
        if "policy_override" in body and body["policy_override"]:
            kwargs["policy_override"] = Policy(body["policy_override"])
        if "max_tokens" in body:
            kwargs["max_tokens"] = int(body["max_tokens"])

        resp = web.StreamResponse(status=200, reason="OK",
                                  headers={"Content-Type": "text/event-stream",
                                           "Cache-Control": "no-cache",
                                           "X-Accel-Buffering": "no"})
        await resp.prepare(req)
        loop = asyncio.get_event_loop()
        queue: asyncio.Queue = asyncio.Queue()
        SENTINEL = object()

        def worker():
            try:
                for chunk in get_router().stream(feature, msgs, **kwargs):
                    loop.call_soon_threadsafe(queue.put_nowait, chunk)
            except Exception as exc:
                loop.call_soon_threadsafe(queue.put_nowait,
                                          {"__error__": str(exc)})
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, SENTINEL)

        loop.run_in_executor(None, worker)
        while True:
            item = await queue.get()
            if item is SENTINEL:
                break
            if isinstance(item, dict) and "__error__" in item:
                await resp.write(("event: error\ndata: " +
                                  json.dumps(item) + "\n\n").encode("utf-8"))
                break
            await resp.write(("data: " + json.dumps({"chunk": item}) + "\n\n").encode("utf-8"))
        await resp.write(b"event: done\ndata: {}\n\n")
        await resp.write_eof()
        return resp

    # ------------------------------------------------------------- keys
    @routes.get("/c2c/ai/keys/list")
    async def _keys_list(_req):
        return _ok({"keys": kc.list_set_keys()})

    @routes.post("/c2c/ai/keys/set")
    async def _keys_set(req):
        body = await req.json()
        name = body.get("name") or ""
        value = body.get("value") or ""
        if name not in kc.ALL_KNOWN_KEYS:
            return _err("UNKNOWN_KEY", f"{name!r} is not a known key")
        if not value:
            return _err("EMPTY_VALUE", "value is required")
        try:
            kc.set_(name, value)
        except Exception as exc:
            return _err("KEYCHAIN_FAILED", str(exc), code=500)
        return _ok({"set": name})

    @routes.post("/c2c/ai/keys/delete")
    async def _keys_delete(req):
        body = await req.json()
        name = body.get("name") or ""
        ok = kc.delete(name)
        return _ok({"deleted": ok, "name": name})

    @routes.post("/c2c/ai/keys/import_txt")
    async def _keys_import_txt(req):
        body = await req.json()
        path = body.get("path") or ""
        if not path or not os.path.isfile(path):
            return _err("FILE_NOT_FOUND", f"no such file: {path}")
        try:
            written = kc.import_from_txt(path)
        except Exception as exc:
            return _err("IMPORT_FAILED", str(exc), code=500)
        return _ok({"imported": written, "source": path})

    # -------------------------------------------------------- local servers
    @routes.get("/c2c/ai/local/detect")
    async def _local_detect(_req):
        loop = asyncio.get_event_loop()
        found = await loop.run_in_executor(None, detect_local_servers)
        return _ok({"servers": found})

    # ------------------------------------------------------ text encoders
    # B1: list GGUF files under ComfyUI's text_encoders folder(s). The
    # Settings UI populates a combo from this so users can pick a local
    # in-process chat model without editing ai_config.json by hand.
    # Uses folder_paths so extra_model_paths.yaml entries also count.
    @routes.get("/c2c/ai/text_encoders/list")
    async def _text_encoders_list(_req):
        try:
            import folder_paths  # type: ignore
        except Exception as exc:
            return _err("NO_FOLDER_PATHS", f"folder_paths unavailable: {exc}",
                        code=500)
        try:
            files = folder_paths.get_filename_list("text_encoders") or []
        except Exception as exc:
            return _err("SCAN_FAILED", str(exc), code=500)
        ggufs = [f for f in files if isinstance(f, str)
                 and f.lower().endswith(".gguf")]
        # Surface absolute paths + file size so the UI can display them.
        items = []
        for name in ggufs:
            try:
                abs_p = folder_paths.get_full_path("text_encoders", name)
            except Exception:
                abs_p = None
            size = None
            if abs_p:
                try:
                    size = int(__import__("os").path.getsize(abs_p))
                except Exception:
                    size = None
            items.append({"name": name, "path": abs_p, "size_bytes": size})
        # Surface the configured roots so the UI can tell the user where
        # to drop new GGUFs.
        try:
            roots = list(folder_paths.get_folder_paths("text_encoders") or [])
        except Exception:
            roots = []
        # Detect whether llama-cpp-python is installed so the UI can warn
        # the user up-front instead of failing on first chat.
        try:
            import importlib.util
            llamacpp_available = importlib.util.find_spec("llama_cpp") is not None
        except Exception:
            llamacpp_available = False
        return _ok({
            "items": items,
            "roots": roots,
            "llamacpp_available": llamacpp_available,
        })

    # --------------------------------------------------------- policy
    @routes.get("/c2c/ai/policy")
    async def _policy_get(_req):
        return _ok([
            {"feature": e.feature,
             "default": e.default.value,
             "override": e.override.value if e.override else None,
             "effective": e.effective.value}
            for e in policy_mod.listing()
        ])

    @routes.post("/c2c/ai/policy")
    async def _policy_set(req):
        body = await req.json()
        feat = body.get("feature") or ""
        pol = body.get("policy")
        if not feat:
            return _err("MISSING_FEATURE", "feature is required")
        try:
            policy_mod.set_override(feat, Policy(pol) if pol else None)
        except ValueError:
            return _err("BAD_POLICY", f"unknown policy: {pol!r}")
        return _ok({"feature": feat, "policy": pol})

    # ---------------------------------------------------------- config
    @routes.get("/c2c/ai/config")
    async def _config_get(_req):
        return _ok(bs.load_config())

    @routes.post("/c2c/ai/config")
    async def _config_set(req):
        body = await req.json()
        if not isinstance(body, dict):
            return _err("BAD_BODY", "body must be a JSON object")
        bs.save_config(body)
        # Re-register backends from scratch
        router = get_router()
        for b in list(router.all_backends()):
            router.unregister(b.info.id)
        for entry in body.get("backends", []):
            try:
                router.register(bs.build_backend(entry))
            except Exception as exc:
                log.warning("config: skipping %s: %s", entry, exc)
        # kick a probe so status is fresh by the time the UI reads it
        asyncio.get_event_loop().run_in_executor(None, router.probe_all)
        return _ok({"saved": True, "backend_count": len(router.all_backends())})

    @routes.post("/c2c/ai/backends/test")
    async def _backend_test(req):
        body = await req.json()
        bid = body.get("id") or ""
        backend = get_router().get(bid)
        if not backend:
            return _err("UNKNOWN_BACKEND", f"no backend with id={bid!r}")
        loop = asyncio.get_event_loop()
        try:
            h = await loop.run_in_executor(None, backend.probe)
        except Exception as exc:
            return _err("PROBE_FAILED", str(exc), code=500)
        return _ok({
            "ok": h.ok,
            "last_rtt_ms": h.last_rtt_ms,
            "last_error": h.last_error,
            "last_probe_at": h.last_probe_at,
        })

    # ---------------------------------------------------------- prompts
    # Versioned Jinja2 system-prompt library (no inline prompt strings in JS).
    from . import prompts as prompt_lib

    @routes.get("/c2c/ai/prompts")
    async def _prompts_list(_req):
        return _ok({"templates": prompt_lib.list_templates()})

    @routes.post("/c2c/ai/prompts/render")
    async def _prompts_render(req):
        try:
            body = await req.json()
        except Exception as exc:
            return _err("BAD_BODY", f"invalid JSON: {exc}")
        if not isinstance(body, dict):
            return _err("BAD_BODY", "body must be a JSON object")
        name = body.get("name")
        if not isinstance(name, str) or not name:
            return _err("MISSING_NAME", "'name' (string) is required")
        vars_in = body.get("vars") or {}
        if not isinstance(vars_in, dict):
            return _err("BAD_VARS", "'vars' must be a JSON object")
        # All var values must be JSON-serialisable scalars/lists/dicts.
        if not prompt_lib.has(name):
            return _err("UNKNOWN_TEMPLATE", f"no prompt template named {name!r}", code=404)
        try:
            text = prompt_lib.render(name, **vars_in)
        except (KeyError, ValueError) as exc:
            return _err("RENDER_FAILED", str(exc))
        except Exception as exc:                       # pragma: no cover
            log.exception("prompt render failed: %s", name)
            return _err("RENDER_ERR", str(exc), code=500)
        # Look up version + golden so the JS client can pin against drift.
        meta = next((t for t in prompt_lib.list_templates() if t["name"] == name), {})
        return _ok({
            "name": name,
            "text": text,
            "version": meta.get("version", "0.0.0"),
            "golden_sha256": meta.get("golden_sha256", ""),
        })

    @routes.get("/c2c/ai/prompts/verify")
    async def _prompts_verify(_req):
        return _ok({"results": prompt_lib.verify_goldens()})

    log.info("c2c_ai routes registered under /c2c/ai/*")
