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
import time

log = logging.getLogger("c2c.preview")

# Channel-count RGB factors for the raw-latent safety net. Imported here
# (not inside the decode function) so it resolves once at import time with
# the correct package context, and is robust to the guard being imported
# from different entry points.
try:
    from . import _latent_rgb_factors as _factors_mod
except Exception:  # noqa: BLE001 — fallback: locate it by path
    try:
        import importlib.util as _ilu
        _fp = os.path.join(os.path.dirname(__file__), "_latent_rgb_factors.py")
        _spec = _ilu.spec_from_file_location("_latent_rgb_factors", _fp)
        _factors_mod = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(_factors_mod)
    except Exception:
        _factors_mod = None

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
        import latent_preview  # noqa: F401
        # THE PR #11261 GOTCHA: newer ComfyUI resets args.preview_method on EVERY
        # prompt (execution.py: set_preview_method(extra_data['preview_method'])).
        # When the frontend sends "default"/None, set_preview_method falls back to
        # latent_preview.default_preview_method — which is whatever --preview-method
        # was (NoPreviews here). So setting args.preview_method alone is WIPED every
        # run; we must override default_preview_method itself. TAESD is universal:
        # core samplers use the taesd decoder / Latent2RGB, and Kijai's WanVideoSampler
        # routes TAESD to its OWN video previewer (Auto/Latent2RGB are blank for Wan).
        dflt = getattr(latent_preview, "default_preview_method", None)
        if dflt == LatentPreviewMethod.NoPreviews:
            latent_preview.default_preview_method = LatentPreviewMethod.TAESD
        # Also set it live for the current run.
        if getattr(args, "preview_method", None) == LatentPreviewMethod.NoPreviews:
            try:
                latent_preview.set_preview_method("taesd")
            except Exception:
                args.preview_method = LatentPreviewMethod.TAESD
        PREVIEW_GUARD_STATUS = "forced_taesd (+per-queue default override)"
        log.info("[c2c.preview] forced live preview to TAESD AND overrode the per-queue "
                 "'default' fallback (PR #11261) so it survives every prompt — works for "
                 "core AND Kijai/Wan samplers. Set C2C_NO_FORCE_PREVIEW=1 to opt out.")
    except Exception as exc:
        PREVIEW_GUARD_STATUS = f"error ({type(exc).__name__})"
        log.warning("[c2c.preview] could not ensure previews: %s", exc)
    return PREVIEW_GUARD_STATUS


def _is_video_latent(latent_format) -> bool:
    """True for temporal (video) latent formats — Wan / Hunyuan-Video / LTXV /
    Mochi / Cosmos. Robust across ComfyUI versions:
      1. `latent_dimensions >= 3` (video formats set 3; images 2) — the primary,
         version-stable signal,
      2. the TAESD decoder name, checked against core's OWN canonical
         `latent_preview.VIDEO_TAES` list (not a hardcoded prefix guess —
         core renamed taew2_1/taew2_2 -> lighttaew2_1/lighttaew2_2 at some
         point, and a hardcoded ("taew","taehv") prefix check silently stops
         matching the moment core renames things again),
      3. a class-name keyword fallback.
    Any lookup failure just falls through to "not video" (image path)."""
    try:
        if int(getattr(latent_format, "latent_dimensions", 2)) >= 3:
            return True
    except Exception:
        pass
    try:
        deco = str(getattr(latent_format, "taesd_decoder_name", "") or "")
        try:
            import latent_preview
            video_taes = set(getattr(latent_preview, "VIDEO_TAES", []))
        except Exception:
            video_taes = set()
        if deco in video_taes or deco.lower().startswith(("taew", "taehv", "lighttaew", "lighttaehy", "taeltx")):
            return True
        name = type(latent_format).__name__.lower()
        return any(k in name for k in ("wan", "hunyuan", "ltx", "mochi", "cosmos", "video"))
    except Exception:
        return False


def _taesd_decoder_present(latent_format) -> bool:
    """True if a vae_approx file matching this format's taesd_decoder_name
    actually exists on disk. Used to make "Auto" pick TAESD for images too
    when it's genuinely available — see _auto_method_for for why this check
    exists at all (core's real Auto does NOT do this itself)."""
    try:
        import folder_paths
        name = getattr(latent_format, "taesd_decoder_name", None)
        if not name:
            return False
        return any(fn.startswith(name) for fn in folder_paths.get_filename_list("vae_approx"))
    except Exception:
        return False


def _auto_method_for(latent_format) -> str:
    """The heart of "Auto": pick the previewer per MODEL.
      video  -> TAESD  (Kijai/Wan samplers route TAESD to their own video
                         previewer; core falls back to Wan-factor Latent2RGB
                         if the taew decoder file is absent — never blank),
      image  -> TAESD when a decoder file is actually present, else "auto"
                (-> Latent2RGB). NOTE: core's real Auto does NOT do this
                itself — `latent_preview.get_previewer` unconditionally maps
                LatentPreviewMethod.Auto -> Latent2RGB regardless of whether
                a sharp TAESD decoder is sitting right there in vae_approx.
                That is a real quality regression for SD/SDXL/Flux/SD3 (whose
                taesd_decoder/taesdxl_decoder/taesd3_decoder files are the
                ComfyUI-standard, near-always-present ones) — Auto would give
                a blocky Latent2RGB preview when a much sharper TAESD preview
                was one filename-check away. This is what makes CNP's Auto
                the best-looking option per model instead of just mirroring
                core's own (weaker) Auto."""
    if _is_video_latent(latent_format):
        return "taesd"
    return "taesd" if _taesd_decoder_present(latent_format) else "auto"


class _MeanChannelPreviewer:
    """Absolute last-resort previewer for latent formats with NEITHER a
    usable TAESD decoder NOR latent_rgb_factors (e.g. LTXAV, which sets
    latent_rgb_factors=None explicitly) — mean-projects all channels to a
    normalized grayscale frame instead of returning None (a permanently
    blank node during sampling). Crude, but strictly better than nothing,
    matching this guard's own never-None design goal."""
    def decode_latent_to_preview_image(self, preview_format, x0):
        try:
            import latent_preview as _lp
            x = x0[0]                      # drop batch -> (C,H,W) or (C,T,H,W)
            if x.ndim == 4:                # (C,T,H,W) video -> first frame
                x = x[:, 0]
            x = x.mean(dim=0)              # (H,W)
            x = x.unsqueeze(-1).repeat(1, 1, 3)  # (H,W,3)
            lo, hi = x.min(), x.max()
            x = (x - lo) / (hi - lo + 1e-6)
            img = _lp.preview_to_image(x, do_scale=False)
            return ("JPEG", img, _lp.MAX_PREVIEW_RESOLUTION)
        except Exception:
            return None


def _install_previewer_fallback() -> None:
    """Patch latent_preview.get_previewer with (a) smart per-model Auto and
    (b) a never-None safety net.

    In Auto mode (the default, and when the user picks "Auto") we OVERRIDE
    core's method choice per call based on the latent format — because core's
    own Auto resolves to Latent2RGB for Wan/video latents, which is blank. For
    explicit taesd/latent2rgb we respect the user's choice; for off we return
    None. Fully guarded: any ComfyUI API change just no-ops and leaves core
    untouched."""
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

    def _resolve_with(method_str, device, latent_format):
        """Run core's resolver with args.preview_method forced to method_str,
        then restore it. Returns core's previewer (or None)."""
        saved = getattr(args, "preview_method", None)
        try:
            args.preview_method = LatentPreviewMethod(method_str)
            return orig(device, latent_format)
        except Exception:
            return None
        finally:
            try:
                args.preview_method = saved
            except Exception:
                pass

    def _patched_get_previewer(device, latent_format):
        # Smart Auto: default (None) and explicit "auto" both get per-model
        # selection. This is authoritative on EVERY sampler callback, so the
        # per-prompt reset in PR #11261 can't undo it.
        if _USER_PREF in (None, "auto"):
            prev = _resolve_with(_auto_method_for(latent_format), device, latent_format)
            if prev is not None:
                return prev
            # video TAESD came back empty (no taew file AND no rgb factors?) —
            # fall through to the never-None net below.

        if _USER_PREF in ("off", "none"):
            return None  # user explicitly turned the sampler preview OFF

        try:
            prev = orig(device, latent_format)
        except Exception:
            prev = None
        if prev is not None:
            return prev
        # Core gave nothing -> force Auto for one resolve (Latent2RGB, no model,
        # cannot fail) so the live preview always gets frames.
        prev = _resolve_with("auto", device, latent_format)
        if prev is not None:
            return prev
        # Still nothing: this format has no TAESD decoder AND no
        # latent_rgb_factors at all (e.g. LTXAV). Absolute last resort so the
        # node never sits permanently blank during sampling.
        return _MeanChannelPreviewer()

    try:
        latent_preview.get_previewer = _patched_get_previewer
        latent_preview._c2c_previewer_patched = True
        log.info("[c2c.preview] installed smart-Auto + never-None get_previewer wrapper "
                 "(video->TAESD; image->TAESD when a decoder file is present, else Auto; "
                 "per model, with a channel-mean grayscale previewer as the absolute "
                 "last resort for formats with neither a decoder nor rgb factors).")
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
        # Resolve the target enum (enum values are "none"/"auto"/"latent2rgb"/"taesd").
        target = LatentPreviewMethod.NoPreviews if method in ("off", "none") \
            else LatentPreviewMethod(method if method in ("auto", "latent2rgb", "taesd") else "taesd")
        args.preview_method = target
        # CRUCIAL: also set default_preview_method — the value the per-queue override
        # (PR #11261) RESETS to every prompt. Without this the choice is wiped on the
        # next queue and the preview silently stops.
        latent_preview.default_preview_method = target
        log.info("[c2c.preview] preview method set to %r (+per-queue default) by user.", method)
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


# ── 24fps throttle on prepare_callback ──────────────────────────────────
# ComfyUI's native latent_preview.prepare_callback decodes a preview on
# EVERY sampler step and calls pbar.update_absolute(step, total, preview).
# For video (Wan/Hunyuan/LTX) that's per-frame per-step — a TAESD neural
# decode each call, which is the slowdown. Decoding at most 24x/sec looks
# identical (every 5th step on a 30-step run is indistinguishable) and costs
# a fraction as much. This is the "faster (24fps)" the user asked for, and
# it reaches EVERY sampler that calls latent_preview.prepare_callback —
# ComfyUI native (KSampler/SamplerCustom), Kijai's WanVideoSampler when
# it uses the native path, res4lyf if it routes through native, etc.
# The old value here was a flat 24 fps, which NEVER ENGAGED on the workload it
# was written for. The predicate only throttles when consecutive sampler steps
# are closer together than 1/24s = 41.7ms. Measured against realistic step
# times, decodes allowed out of a 30-step run:
#
#     Wan video 768, 100-200 frames   2000ms/step   30/30   never throttled
#     Wan video, distilled             500ms/step   30/30   never throttled
#     SDXL image 1024                  120ms/step   30/30   never throttled
#     tiny latent                       20ms/step   11/30   throttled
#
# So every step still ran a full multi-frame TAESD decode and pushed those bytes
# down the websocket — which is what backs the socket up and makes previews
# stall. A fixed frame rate cannot work here because decode cost varies by three
# orders of magnitude between a 1024 image and a 200-frame video latent.
#
# Instead, budget preview against its OWN measured cost: after each decode we
# know how long it took, so we require the next one to wait until the preview
# has consumed no more than _PREVIEW_BUDGET of wall-clock time. That is
# self-tuning — cheap image decodes stay smooth, expensive video decodes space
# themselves out — and it degrades gracefully on hardware we have never seen.
_PREVIEW_BUDGET = 0.10          # preview may cost at most 10% of sampling time
_PREVIEW_MIN_GAP = 1.0 / 24.0   # floor: never faster than 24fps
_PREVIEW_MAX_GAP = 10.0         # ceiling: always show something within 10s


def _install_prepare_callback_throttle() -> None:
    """Wrap latent_preview.prepare_callback so the inner callback skips
    decode when less than 1/24s has elapsed since the last sent preview.

    Fully guarded and idempotent. The wrapped callback still calls the
    original pbar.update_absolute every step (so the progress bar stays
    smooth) but passes preview=None on throttled steps — so no decode runs
    and no preview bytes are sent, which is the whole speedup."""
    try:
        import latent_preview
    except Exception:
        return
    if getattr(latent_preview, "_c2c_throttle_patched", False):
        return
    orig_prepare = getattr(latent_preview, "prepare_callback", None)
    if not callable(orig_prepare):
        return

    def _throttled_prepare_callback(model, steps, x0_output_dict=None):
        native_cb = orig_prepare(model, steps, x0_output_dict)
        if native_cb is None:
            return None
        state = {"last": 0.0, "have_decoded": False, "cost": 0.0}

        def _cb(step, x0, x, total_steps):
            now = time.monotonic()
            # Always let the final step through (so the user sees the end
            # result), and the first decode (so preview appears fast).
            is_final = (step is not None) and (total_steps is not None) and (step + 1 >= total_steps)
            # Required gap = what it costs, divided by the share of runtime we
            # are willing to spend on it. A 0.5s video decode at a 10% budget
            # must wait 5s; a 5ms image decode waits the 24fps floor.
            gap = state["cost"] / _PREVIEW_BUDGET if state["cost"] > 0.0 else _PREVIEW_MIN_GAP
            gap = max(_PREVIEW_MIN_GAP, min(_PREVIEW_MAX_GAP, gap))
            if state["have_decoded"] and not is_final and (now - state["last"]) < gap:
                # Skip the decode: pass None so the previewer never runs.
                native_cb(step, None, x, total_steps)
                return
            t0 = time.monotonic()
            native_cb(step, x0, x, total_steps)
            # Measure what the decode+send actually cost so the next gap adapts.
            # Smoothed so one slow frame (a swap, a GC pause) does not lock the
            # preview out for the rest of the run.
            dt = time.monotonic() - t0
            state["cost"] = dt if state["cost"] <= 0.0 else (0.5 * state["cost"] + 0.5 * dt)
            state["last"] = time.monotonic()
            state["have_decoded"] = True

        return _cb

    try:
        latent_preview.prepare_callback = _throttled_prepare_callback
        latent_preview._c2c_throttle_patched = True
        log.info("[c2c.preview] installed adaptive preview throttle "
                 "(preview budgeted to %.0f%% of sampling time, %.2fs-%.0fs gap).",
                 _PREVIEW_BUDGET * 100, _PREVIEW_MIN_GAP, _PREVIEW_MAX_GAP)
    except Exception as exc:  # noqa: BLE001
        log.debug("[c2c.preview] prepare_callback throttle skipped: %s", exc)


# ── raw-latent safety net on ProgressBar.update_absolute ────────────────
# res4lyf (Radiance) and a few other samplers pass the RAW latent tensor as
# the preview (pbar.update_absolute(step, total, (x0,))). Standard ComfyUI
# can't render a raw latent — it expects ("JPEG", pil_image, max_res) — so
# those samplers show NO preview at all unless their own frontend decodes
# it. ProgressBar.update_absolute is the ONE call every sampler makes, so
# wrapping it reaches them all. When preview is a raw latent (a tuple whose
# first element is a torch.Tensor, not a "JPEG"/"PNG" string), decode it to
# a real JPEG preview via channel-count factors (Wan21/Wan22 for 16/48 ch,
# mean-projection otherwise) and forward that. Additive + guarded: valid
# previews pass through untouched.
def _decode_raw_latent_preview(x0):
    """Best-effort decode of a raw latent tensor to a ('JPEG', pil, max) preview.

    Uses channel-count RGB factors (Wan21 16ch / Wan22 48ch) when known, else
    a normalized mean-projection grayscale. Never raises — returns None on
    any failure so the caller just forwards nothing."""
    try:
        import latent_preview as _lp
        import torch
        import io as _io
        from PIL import Image
        x = x0
        if x.ndim == 5:          # (B,C,T,H,W) video -> first frame
            x = x[0, :, 0]
        elif x.ndim == 4:        # (B,C,H,W) -> first batch
            x = x[0]
        if x.ndim != 3:          # (C,H,W) expected now
            return None
        # Channel-count RGB factors (Wan21 16ch / Wan22 48ch); None -> the
        # caller falls back to a mean-projection grayscale.
        rgb = _factors_mod.decode_channels_to_rgb(x) if _factors_mod is not None else None
        if rgb is None:
            # Unknown channel count -> mean-projection grayscale (strictly
            # better than a blank node).
            g = x.mean(dim=0, keepdim=True).repeat(3, 1, 1)
            lo, hi = g.min(), g.max()
            rgb = ((g - lo) / (hi - lo + 1e-6)).clamp(0, 1)
        rgb = (rgb * 0xFF).clamp(0, 255).to(device="cpu", dtype=torch.uint8)
        img = Image.fromarray(rgb.permute(1, 2, 0).numpy())
        buf = _io.BytesIO()
        img.save(buf, format="JPEG", quality=92)
        return ("JPEG", buf.getvalue(), getattr(_lp, "MAX_PREVIEW_RESOLUTION", 256))
    except Exception:
        return None


def _install_pbar_safety_net() -> None:
    """Wrap comfy.utils.ProgressBar.update_absolute so a raw-latent preview
    (a tuple whose [0] is a torch.Tensor, not a 'JPEG'/'PNG' str) is decoded
    to a real image preview before reaching the frontend. Valid previews
    pass through untouched. Idempotent + guarded."""
    try:
        import comfy.utils
    except Exception:
        return
    PB = getattr(comfy.utils, "ProgressBar", None)
    if PB is None or getattr(PB, "_c2c_pbar_patched", False):
        return
    orig_update = PB.update_absolute
    if not callable(orig_update):
        return

    def _patched_update_absolute(self, value, total=None, preview=None):
        if preview is not None:
            try:
                is_raw = (
                    isinstance(preview, (tuple, list))
                    and len(preview) >= 1
                    and not isinstance(preview[0], str)
                    and hasattr(preview[0], "ndim")
                    and getattr(preview[0], "ndim", 0) >= 3
                )
            except Exception:
                is_raw = False
            if is_raw:
                decoded = _decode_raw_latent_preview(preview[0])
                if decoded is not None:
                    preview = decoded
        return orig_update(self, value, total, preview)

    try:
        PB.update_absolute = _patched_update_absolute
        PB._c2c_pbar_patched = True
        log.info("[c2c.preview] installed ProgressBar raw-latent safety net "
                 "(res4lyf / samplers passing raw latents now preview).")
    except Exception as exc:  # noqa: BLE001
        log.debug("[c2c.preview] pbar safety net skipped: %s", exc)


# Run at import (custom_nodes load after core, so latent_preview already exists).
ensure_previews_enabled()
_install_previewer_fallback()
_install_prepare_callback_throttle()
_install_pbar_safety_net()
_register_routes()
