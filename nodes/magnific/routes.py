"""Local HTTP endpoints (on the ComfyUI server) driving the device-grant
sign-in from web/magnific.js. The browser only ever sees the verification URL;
tokens live server-side, next to the graphs that will use them — the moral
equivalent of the privilegedAuth boundary the desktop plugin hosts have.
"""

import asyncio
import functools
import threading
import time
from typing import Optional

from aiohttp import web

from . import api, auth, mcp, security, update_check

_signin_state: dict = {"state": "idle"}
_signin_lock = threading.Lock()
_signin_generation = 0


def _peer_ip(request: web.Request) -> Optional[str]:
    """The socket's real peer address, read from the transport rather than from
    request.remote — so a forwarded-header middleware, if one is ever mounted
    ahead of us, cannot make a network client look like loopback (TH-651)."""
    transport = request.transport
    peername = transport.get_extra_info("peername") if transport else None
    if isinstance(peername, (tuple, list)) and peername:
        return peername[0]
    return request.remote


def _require_local_ui(handler):
    """Refuse the request unless it comes from this process's own ComfyUI web UI
    (loopback peer + loopback Host + the per-process token). See security.py for
    why the upstream origin middleware cannot be relied on (TH-651)."""

    @functools.wraps(handler)
    async def guarded(request: web.Request) -> web.Response:
        if not security.is_authorized(request.headers, _peer_ip(request)):
            return web.json_response({"error": "forbidden"}, status=403)
        return await handler(request)

    return guarded


def _poll_worker(authorization: dict, generation: int) -> None:
    def stale() -> bool:
        return _signin_generation != generation

    try:
        auth.poll_for_tokens(authorization, should_stop=stale)
    except Exception as error:
        # Catch everything, not just AuthError: an unhandled URLError here
        # would kill the daemon thread and leave the UI polling forever.
        with _signin_lock:
            if not stale():
                _signin_state.update(state="error", error=str(error))
        return
    with _signin_lock:
        if stale():
            # A superseded worker must not touch the active session's state.
            return
        _signin_state.update(state="ok")
    mcp.client().reset()
    # Catalogs are per-account: drop anything a previous session cached, then
    # warm what the node combos read so the post-login refresh is instant.
    api.clear_catalog_caches()
    try:
        api.folder_choices(force=True)
        api.model_choices()
        api.library_choices("mine", force=True)
    except Exception:
        pass


def register(routes: web.RouteTableDef) -> None:
    # Hands the per-process token to the local UI. Guarded by Host +
    # fetch-metadata only — it cannot require the token it exists to hand out —
    # which is enough: a rebound or cross-origin caller fails that check, so it
    # never learns the token that every other route below demands.
    @routes.get("/magnific/csrf-token")
    async def csrf_token(request: web.Request) -> web.Response:
        if not security.is_local_ui_origin(request.headers, _peer_ip(request)):
            return web.json_response({"error": "forbidden"}, status=403)
        return web.json_response({"token": security.UI_TOKEN})

    @routes.get("/magnific/status")
    @_require_local_ui
    async def status(_request: web.Request) -> web.Response:
        return web.json_response({"signed_in": auth.is_signed_in()})

    @routes.get("/magnific/update")
    @_require_local_ui
    async def update(_request: web.Request) -> web.Response:
        return web.json_response(update_check.state())

    # Thumbnail feeds for the visual picker nodes (web/magnific.js galleries).
    @routes.post("/magnific/stock-search")
    @_require_local_ui
    async def stock_search_route(request: web.Request) -> web.Response:
        payload = await request.json()
        query = str(payload.get("query") or "").strip()
        if not query:
            return web.json_response({"items": [], "error": "type a search query first"}, status=422)
        raw_filters = payload.get("filters") if isinstance(payload.get("filters"), dict) else {}
        filters = {key: value for key, value in raw_filters.items() if value and value != "any"}
        try:
            response = await asyncio.get_running_loop().run_in_executor(
                None, lambda: api.stock_search(query, filters, page=max(1, int(payload.get("page") or 1)))
            )
            items = [
                {
                    "id": item.get("id"),
                    "type": item.get("type"),
                    "title": item.get("title"),
                    "previewUrl": item.get("previewUrl"),
                    "premium": bool(item.get("premium")),
                }
                for item in response.get("items", [])
                if isinstance(item, dict) and item.get("type") in api.STOCK_IMAGE_TYPES and item.get("previewUrl")
            ]
            return web.json_response({"items": items})
        except auth.NotSignedInError:
            return web.json_response({"items": [], "error": "not_signed_in"}, status=401)
        except Exception as error:
            message = str(error).split("(request_id")[0].strip()
            return web.json_response({"items": [], "error": message}, status=502)

    @routes.post("/magnific/folder-creations")
    @_require_local_ui
    async def folder_creations_route(request: web.Request) -> web.Response:
        payload = await request.json()
        project = str(payload.get("project") or "")
        folder = str(payload.get("folder") or api.PROJECT_ROOT)
        file_type = str(payload.get("content_type") or "image")
        if file_type not in ("image", "video", "audio"):
            file_type = "image"
        is_root = folder in ("", api.PROJECT_ROOT)
        label = project if is_root else f"{project} / {folder}"
        try:
            reference = await asyncio.get_running_loop().run_in_executor(
                None, lambda: api.folder_reference_for(label)
            )
            if not reference:
                return web.json_response({"items": [], "error": f"unknown folder '{label}'"}, status=422)
            items = await asyncio.get_running_loop().run_in_executor(
                None, lambda: api.folder_creations(reference, is_root, file_type)
            )
            return web.json_response({"items": items})
        except auth.NotSignedInError:
            return web.json_response({"items": [], "error": "not_signed_in"}, status=401)
        except Exception as error:
            message = str(error).split("(request_id")[0].strip()
            return web.json_response({"items": [], "error": message}, status=502)

    # Per-model capabilities — the video node's JS narrows its duration/
    # resolution/aspect_ratio combos per selected model from this.
    @routes.get("/magnific/video-models")
    @_require_local_ui
    async def video_models(_request: web.Request) -> web.Response:
        try:
            models = await asyncio.get_running_loop().run_in_executor(None, api.video_models)
            return web.json_response({"models": models})
        except auth.NotSignedInError:
            return web.json_response({"models": []})
        except Exception as error:
            message = str(error).split("(request_id")[0].strip()
            return web.json_response({"models": [], "error": message}, status=502)

    # Same, for the image node's resolution combo.
    @routes.get("/magnific/image-models")
    @_require_local_ui
    async def image_models(_request: web.Request) -> web.Response:
        try:
            models = await asyncio.get_running_loop().run_in_executor(None, api.image_models)
            return web.json_response({"models": models})
        except auth.NotSignedInError:
            return web.json_response({"models": []})
        except Exception as error:
            message = str(error).split("(request_id")[0].strip()
            return web.json_response({"models": [], "error": message}, status=502)

    # The flattened folder catalog ("Project / Folder / …" labels) — the Save
    # To node's JS narrows its folder combo per selected project from this.
    @routes.get("/magnific/folders")
    @_require_local_ui
    async def folders(_request: web.Request) -> web.Response:
        try:
            choices = await asyncio.get_running_loop().run_in_executor(None, api.folder_choices)
            return web.json_response({"choices": choices})
        except auth.NotSignedInError:
            return web.json_response({"choices": []})
        except Exception as error:
            message = str(error).split("(request_id")[0].strip()
            return web.json_response({"choices": [], "error": message}, status=502)

    # The Library catalog (characters, styles, elements, locations) as combo
    # labels — the Library Reference node's Refresh button re-lists its combo
    # from this without a node-definition reload. `?force=1` bypasses the
    # short server cache so an asset trained a second ago shows up.
    # `?scope=mine|public` picks the catalog (the user's own, or Magnific's);
    # each is fetched and cached on its own so one failing leaves the other up.
    @routes.get("/magnific/library")
    @_require_local_ui
    async def library(request: web.Request) -> web.Response:
        force = request.query.get("force") in ("1", "true")
        scope = request.query.get("scope", "mine")
        if scope not in api.LIBRARY_SCOPES:
            return web.json_response({"choices": [], "error": "unknown scope"}, status=400)
        try:
            choices = await asyncio.get_running_loop().run_in_executor(None, lambda: api.library_choices(scope, force=force))
            items = [
                {
                    "label": choice["label"],
                    "id": choice["entry"]["id"],
                    "name": choice["entry"]["name"],
                    "type": choice["entry"]["type"],
                    "source": choice["entry"]["source"],
                }
                for choice in choices
            ]
            return web.json_response({"scope": scope, "choices": items})
        except auth.NotSignedInError:
            return web.json_response({"choices": [], "error": "not_signed_in"}, status=401)
        except Exception as error:
            message = str(error).split("(request_id")[0].strip()
            return web.json_response({"choices": [], "error": message}, status=502)

    # Credit-cost preview for a node's current widget configuration, backed by
    # the simulate_cost tool. Args are built with the same builders the real
    # execution uses, so the estimate can't drift from what gets charged.
    @routes.post("/magnific/cost")
    @_require_local_ui
    async def cost(request: web.Request) -> web.Response:
        payload = await request.json()
        node = payload.get("node")
        inputs = payload.get("inputs") if isinstance(payload.get("inputs"), dict) else {}
        note = None
        try:
            if node == "MagnificGenerateImage":
                tool = "images_generate"
                # The prompt text never changes the price; a placeholder keeps
                # the simulate call valid before the user has typed one.
                args = api.generate_args(
                    prompt=str(inputs.get("prompt") or "cost estimate"),
                    mode=str(inputs.get("model") or "auto"),
                    count=max(1, int(inputs.get("count") or 1)),
                    aspect_ratio=inputs.get("aspect_ratio"),
                    # The tier is priced (resolution.affectsCost), so an estimate
                    # that ignored it would under-report a 4k run.
                    resolution=inputs.get("resolution"),
                )
            elif node == "MagnificUpscaleImage":
                tool = "images_upscale"
                args = api.upscale_args(
                    None,
                    None,
                    {
                        # The simulator only knows the 2x/4x tiers; clamp larger
                        # scales to 4x (real cost is dimension-dependent and
                        # reported as variable anyway) — same as the panel.
                        "scale": "2x" if inputs.get("scale", "2x") == "2x" else "4x",
                        "precision": inputs.get("precision"),
                        "presets": inputs.get("presets"),
                        "engine": api.upscale_engine_slug(inputs.get("engine")),
                        "mode": api.upscale_mode_slug(inputs.get("mode")),
                    },
                )
            elif node == "MagnificGenerateVideo":
                # The simulator requires a concrete model (auto resolves at run time).
                if inputs.get("model") in (None, "", "auto"):
                    return web.json_response({"error": "pick a model to estimate the cost"}, status=422)
                # Unlike the real clip-based call, the simulator takes flat args.
                tool = "video_generate"
                args = {"slug": str(inputs["model"]), "duration": max(1, int(inputs.get("duration") or 5))}
                resolution = str(inputs.get("resolution") or "").strip()
                if resolution and resolution != "auto":
                    args["resolution"] = resolution
                if inputs.get("aspect_ratio") not in (None, "", "auto"):
                    args["aspectRatio"] = inputs["aspect_ratio"]
                if inputs.get("sound_effects"):
                    args["withSoundEffects"] = True
            elif node == "MagnificUpscaleVideo":
                tool = "video_upscale"
                mode_value = str(inputs.get("mode") or "magnific")
                options = {"preview": bool(inputs.get("preview"))}
                if mode_value == "topaz":
                    options["upscaleFactor"] = int(inputs.get("upscale_factor") or 2)
                else:
                    options["targetResolution"] = int(inputs.get("target_resolution") or 1920)
                # The source clip's duration is only known at run time.
                args = {**api.upscale_video_args(None, mode_value, options), "duration": 5}
                note = "for a 5s clip"
            elif node == "MagnificGenerateMusic":
                tool = "audio_music_generate"
                args = api.music_args(
                    prompt="cost estimate",
                    model=api.music_model_slug(str(inputs.get("model") or "google-lyria")),
                    duration_seconds=int(inputs.get("duration_seconds") or 30),
                    instrumental=bool(inputs.get("instrumental")),
                    force_duration=True,  # required by the simulator for every model
                )
            elif node == "MagnificVoiceover":
                # The simulator prices by text length + model (eleven_v3 is the
                # server default the real model-less call resolves to).
                tool = "audio_tts"
                args = {"model": "eleven_v3", "textLength": max(1, len(str(inputs.get("text") or "")))}
            elif node == "MagnificSkinEnhancer":
                tool = "images_skin_enhancer"
                optimized = inputs.get("optimized_for")
                args = {
                    "version": inputs.get("version") or "faithful",
                    **({"optimizedFor": optimized} if optimized not in (None, "", "default") else {}),
                }
            elif node == "MagnificRemoveBackground":
                tool = "images_remove_background"
                args = {}
            else:
                return web.json_response({"error": f"unsupported node {node}"}, status=400)
            # The MCP client is sync urllib — keep it off the server's event loop.
            estimate = await asyncio.get_running_loop().run_in_executor(
                None, lambda: api.simulate_cost(tool, args)
            )
            if note:
                estimate = {**estimate, "note": note}
            return web.json_response(estimate)
        except auth.NotSignedInError:
            return web.json_response({"error": "not_signed_in"}, status=401)
        except Exception as error:
            # Tool errors end with correlation ids / agent instructions the
            # cost widget shouldn't display — keep the leading message only.
            message = str(error).split("(request_id")[0].strip()
            return web.json_response({"error": message or "cost unavailable"}, status=502)

    @routes.post("/magnific/signin")
    @_require_local_ui
    async def signin(_request: web.Request) -> web.Response:
        global _signin_generation
        # Highest-impact route (it can rebind the plugin to another account), so
        # it carries a second guard on top of the token: refuse to start a grant
        # while a session already exists. A rebind cannot be used to silently
        # overwrite the signed-in account — the UI must sign out first (TH-651).
        if auth.is_signed_in():
            return web.json_response({"error": "already_signed_in"}, status=409)
        try:
            authorization = auth.start_device_authorization()
        except auth.AuthError as error:
            return web.json_response({"error": str(error)}, status=502)
        with _signin_lock:
            _signin_generation += 1
            generation = _signin_generation
            _signin_state.clear()
            _signin_state.update(state="pending", started_at=time.time())
        threading.Thread(target=_poll_worker, args=(authorization, generation), daemon=True).start()
        return web.json_response(
            {
                "verification_uri_complete": authorization["verification_uri_complete"],
                "user_code": authorization["user_code"],
                "expires_at": authorization["expires_at"],
            }
        )

    @routes.get("/magnific/signin/status")
    @_require_local_ui
    async def signin_status(_request: web.Request) -> web.Response:
        with _signin_lock:
            return web.json_response(dict(_signin_state))

    @routes.post("/magnific/signout")
    @_require_local_ui
    async def signout(_request: web.Request) -> web.Response:
        global _signin_generation
        with _signin_lock:
            _signin_generation += 1  # cancels any in-flight poll thread
            _signin_state.clear()
            _signin_state.update(state="idle")
        auth.sign_out()
        mcp.client().reset()
        # The cached catalogs belong to the account that just left.
        api.clear_catalog_caches()
        return web.json_response({"signed_in": False})
