"""
_c2c_preview_guard.py — guarantee live sampling previews, resiliently.

Problem: if ComfyUI is launched with `--preview-method none` (or the preview
method is otherwise off), NO previews stream during sampling. That flag is read
by `latent_preview.get_previewer()` LIVE on every sampler callback, so a custom
pack can flip it back on after startup and previews resume — without touching the
sampler or any fragile internal.

What this does (defensive, update-proof):
  - On import, if the active preview method is "none", force it to Auto.
    Auto resolves to Latent2RGB, which needs NO model and cannot fail — the most
    resilient possible preview. (If TAESD decoders are present, switch to TAESD
    method only when the user opts in; Auto already falls back to Latent2RGB.)
  - Everything is wrapped so that ANY change in ComfyUI's preview API simply
    no-ops here instead of breaking the pack ("even with package updates all our
    nodes should work").
  - Opt out with env var C2C_NO_FORCE_PREVIEW=1.

No frontend overlap: this is BACKEND ONLY. It does not draw anything and does
not touch any node — it merely ensures ComfyUI's OWN previewer runs, so the
NATIVE in-node latent preview (latent_preview.py) displays during sampling.
There is deliberately no custom JS preview; ComfyUI core renders the preview
on the node. The get_previewer wrapper is purely additive (returns core's own
result untouched whenever core produces one) and fully guarded, so a bad
ComfyUI update can never break or be damaged by it.
"""
from __future__ import annotations

import logging
import os

log = logging.getLogger("c2c.preview")

# Recorded so the frontend can query what happened (via /object_info-independent log).
PREVIEW_GUARD_STATUS = "unknown"

# User preference set live from the frontend setting (js/c2c_preview_toggle.js)
# via POST /c2c/preview_method. None = no explicit choice (guard forces Auto so
# previews work by default). "off"/"none" = user disabled previews -> the
# get_previewer fallback below must NOT force them back on.
_USER_PREF = None


def ensure_previews_enabled() -> str:
    global PREVIEW_GUARD_STATUS
    if os.environ.get("C2C_NO_FORCE_PREVIEW") == "1":
        PREVIEW_GUARD_STATUS = "disabled_by_env"
        return PREVIEW_GUARD_STATUS
    try:
        from comfy.cli_args import args, LatentPreviewMethod
    except Exception as exc:  # ComfyUI internals moved — never break the pack
        PREVIEW_GUARD_STATUS = f"unavailable ({type(exc).__name__})"
        log.debug("[c2c.preview] cli_args unavailable: %s", exc)
        return PREVIEW_GUARD_STATUS
    try:
        cur = getattr(args, "preview_method", None)
        if cur == LatentPreviewMethod.NoPreviews:
            # Prefer the module helper if present (handles its own state).
            try:
                import latent_preview  # noqa: F401
                if hasattr(latent_preview, "set_preview_method"):
                    latent_preview.set_preview_method("auto")
                else:
                    args.preview_method = LatentPreviewMethod.Auto
            except Exception:
                args.preview_method = LatentPreviewMethod.Auto
            PREVIEW_GUARD_STATUS = "forced_auto (was none)"
            log.info("[c2c.preview] previews were OFF (--preview-method none) -> forced to Auto "
                     "(Latent2RGB; no model, cannot fail). Set C2C_NO_FORCE_PREVIEW=1 to opt out.")
        else:
            PREVIEW_GUARD_STATUS = f"already_on ({getattr(cur, 'value', cur)})"
            log.debug("[c2c.preview] previews already enabled: %s", cur)
    except Exception as exc:
        PREVIEW_GUARD_STATUS = f"error ({type(exc).__name__})"
        log.warning("[c2c.preview] could not ensure previews: %s", exc)
    return PREVIEW_GUARD_STATUS


def _install_previewer_fallback() -> None:
    """Bulletproof layer: patch latent_preview.get_previewer so it can NEVER
    return None. Core returns None only when previews are off — in that case we
    re-run core's own resolver with preview_method forced to Auto (Latent2RGB,
    no model, cannot fail). This survives the method being reset per-prompt or
    read differently across versions, so the live HUD always gets frames.
    Fully guarded: any API change just no-ops and leaves core untouched."""
    try:
        import latent_preview
        from comfy.cli_args import args, LatentPreviewMethod
    except Exception:
        return
    if getattr(latent_preview, "_c2c_previewer_patched", False):
        return
    orig = getattr(latent_preview, "get_previewer", None)
    if not callable(orig):
        return

    def _patched_get_previewer(device, latent_format):
        try:
            prev = orig(device, latent_format)
        except Exception:
            prev = None
        if prev is not None:
            return prev
        if _USER_PREF in ("off", "none"):
            return None  # user explicitly turned the sampler preview OFF
        # Core gave nothing (previews off) -> force Auto for one resolve.
        saved = getattr(args, "preview_method", None)
        try:
            args.preview_method = LatentPreviewMethod.Auto
            return orig(device, latent_format)
        except Exception:
            return prev
        finally:
            try:
                args.preview_method = saved
            except Exception:
                pass

    try:
        latent_preview.get_previewer = _patched_get_previewer
        latent_preview._c2c_previewer_patched = True
        log.info("[c2c.preview] installed get_previewer fallback (previews can never be None).")
    except Exception as exc:  # noqa: BLE001
        log.debug("[c2c.preview] previewer patch skipped: %s", exc)


def set_preview_method(method: str) -> dict:
    """Apply a preview-method choice live (called by the HTTP route below).

    method: "auto" | "latent2rgb" | "taesd" | "off"/"none".
    `get_previewer` reads args.preview_method live on every sampler callback,
    so this takes effect on the NEXT queue with no restart. Backend-only —
    drives ComfyUI's OWN native previewer; no overlay, no core damage.
    """
    global _USER_PREF
    method = str(method or "auto").lower()
    _USER_PREF = method
    try:
        from comfy.cli_args import args, LatentPreviewMethod
        import latent_preview
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"cli_args/latent_preview unavailable: {exc!r}"}
    try:
        if method in ("off", "none"):
            args.preview_method = LatentPreviewMethod.NoPreviews
        elif hasattr(latent_preview, "set_preview_method"):
            # Core's own setter understands "auto"/"latent2rgb"/"taesd".
            latent_preview.set_preview_method(method)
        else:
            args.preview_method = LatentPreviewMethod.Auto
        log.info("[c2c.preview] preview method set to %r by user.", method)
        return {"ok": True, "method": method}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": repr(exc)}


def _register_routes() -> None:
    """Expose POST /c2c/preview_method so the frontend toggle can enable/disable
    the native sampler preview. Fully guarded: if the server API changes this
    just no-ops and the pack is unaffected."""
    if getattr(_register_routes, "_done", False):
        return
    try:
        from server import PromptServer
        from aiohttp import web
        routes = PromptServer.instance.routes
    except Exception as exc:  # noqa: BLE001
        log.debug("[c2c.preview] route registration skipped: %s", exc)
        return

    @routes.post("/c2c/preview_method")
    async def _c2c_set_preview_method(request):  # noqa: ANN001
        try:
            data = await request.json()
        except Exception:  # noqa: BLE001
            data = {}
        result = set_preview_method(data.get("method", "auto"))
        return web.json_response(result, status=200 if result.get("ok") else 500)

    _register_routes._done = True
    log.info("[c2c.preview] registered POST /c2c/preview_method (enable/disable sampler preview).")


# Run at import (custom_nodes load after core, so latent_preview already exists).
ensure_previews_enabled()
_install_previewer_fallback()
_register_routes()
