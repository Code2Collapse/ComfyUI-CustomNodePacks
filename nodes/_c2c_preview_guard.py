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
import threading
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


# ── fetching a missing TAE decoder ──────────────────────────────────────
# A TAESD preview is far sharper than the Latent2RGB fallback, and the only
# thing standing between them is a ~5 MB file sitting in models/vae_approx.
# Core logs a warning and gives up; this fetches it.
#
# EVERY URL BELOW WAS CHECKED before being written here, and four candidates
# were dropped because they did not resolve:
#   madebyollin/taehv        401, the repo is gated
#   the three Comfy-Org "lighttae*" paths   404, not at those paths
# So the video decoders are NOT auto-fetchable and this does not pretend
# otherwise - it names the exact file and folder and says to fetch it by hand.
# Guessing a URL here would produce a silent 404 on every sample, which is
# strictly worse than saying "I cannot get this one".
#
# Nothing is lost when a decoder is missing: the guard already falls through
# to Latent2RGB and then to the channel-mean previewer, so the node never goes
# blank. The fetch only ever UPGRADES the picture.
#
# Set C2C_NO_TAE_DOWNLOAD=1 to disable, or drop a JSON map of
# {"decoder_name": "https://..."} at models/vae_approx/_c2c_tae_sources.json
# to add your own without editing this file.
_TAE_SOURCES = {
    "taesd_decoder":   "https://github.com/madebyollin/taesd/raw/main/taesd_decoder.pth",
    "taesdxl_decoder": "https://github.com/madebyollin/taesd/raw/main/taesdxl_decoder.pth",
    "taesd3_decoder":  "https://github.com/madebyollin/taesd/raw/main/taesd3_decoder.pth",
    "taef1_decoder":   "https://github.com/madebyollin/taesd/raw/main/taef1_decoder.pth",
}

_TAE_MIN_BYTES = 1 << 20          # a 5 MB decoder; anything under 1 MB is an
                                  # error page that happened to return 200
_TAE_MAX_BYTES = 512 << 20        # refuse to stream something unbounded
_fetch_lock = threading.Lock()
_fetch_tried: set = set()         # one attempt per decoder per process


def _vae_approx_dir():
    try:
        import folder_paths
        dirs = folder_paths.get_folder_paths("vae_approx")
        return dirs[0] if dirs else None
    except Exception:
        return None


def _tae_sources() -> dict:
    """The built-in table, plus anything the user added alongside the models."""
    out = dict(_TAE_SOURCES)
    d = _vae_approx_dir()
    if not d:
        return out
    try:
        import json
        p = os.path.join(d, "_c2c_tae_sources.json")
        if os.path.isfile(p):
            with open(p, encoding="utf-8") as fh:
                extra = json.load(fh)
            if isinstance(extra, dict):
                out.update({str(k): str(v) for k, v in extra.items()})
    except Exception as exc:  # noqa: BLE001
        log.debug("[c2c.preview] could not read _c2c_tae_sources.json: %s", exc)
    return out


def _download_tae(name: str, url: str, dest_dir: str) -> None:
    """Fetch one decoder. Runs on a worker thread; never raises into a sampler.

    Writes to a temp name and renames, so a half-finished download can never
    be picked up as a model - an interrupted fetch would otherwise leave a
    truncated file that loads, fails deep inside the decoder, and looks like a
    corrupt install.
    """
    import urllib.request
    ext = ".safetensors" if url.endswith(".safetensors") else ".pth"
    final = os.path.join(dest_dir, name + ext)
    tmp = final + ".part"
    try:
        os.makedirs(dest_dir, exist_ok=True)
        log.info("[c2c.preview] fetching the %s preview decoder (~5 MB) from %s "
                 "- previews stay on the fallback until it lands", name, url)
        req = urllib.request.Request(url, headers={"User-Agent": "ComfyUI-CustomNodePacks"})
        total = 0
        with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as fh:
            while True:
                chunk = r.read(1 << 16)
                if not chunk:
                    break
                total += len(chunk)
                if total > _TAE_MAX_BYTES:
                    raise OSError(f"{name} exceeded {_TAE_MAX_BYTES} bytes - refusing")
                fh.write(chunk)
        if total < _TAE_MIN_BYTES:
            raise OSError(f"{name} came back as {total} bytes, too small to be a decoder")
        os.replace(tmp, final)
        log.info("[c2c.preview] %s ready (%.1f MB) - TAESD previews from the next "
                 "sample on", name, total / 1048576)
    except Exception as exc:  # noqa: BLE001
        log.warning("[c2c.preview] could not fetch %s: %s. Previews keep working on "
                    "the Latent2RGB fallback; drop the file in %s yourself to upgrade "
                    "them.", name, exc, dest_dir)
    finally:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass


def _ensure_decoder(latent_format) -> None:
    """Kick off a fetch if this model's decoder is missing. Returns at once.

    Deliberately non-blocking: a sampler callback must not wait on a network
    request. This sample previews on the fallback and the next one is sharp.
    """
    if os.environ.get("C2C_NO_TAE_DOWNLOAD"):
        return
    try:
        name = getattr(latent_format, "taesd_decoder_name", None)
        if not name or _taesd_decoder_present(latent_format):
            return
        with _fetch_lock:
            if name in _fetch_tried:
                return
            _fetch_tried.add(name)
        dest = _vae_approx_dir()
        if not dest:
            return
        url = _tae_sources().get(name)
        if not url:
            # Not guessing. A wrong URL is a silent 404 on every sample.
            log.info("[c2c.preview] no download source known for the '%s' preview "
                     "decoder (the video ones are gated or unpublished). Previews "
                     "use the Latent2RGB fallback, which works. To sharpen them, "
                     "put a file starting with '%s' in %s, or add "
                     '{"%s": "<url>"} to %s.',
                     name, name, dest, name,
                     os.path.join(dest, "_c2c_tae_sources.json"))
            return
        threading.Thread(target=_download_tae, args=(name, url, dest),
                         name=f"c2c-tae-{name}", daemon=True).start()
    except Exception as exc:  # noqa: BLE001
        log.debug("[c2c.preview] decoder fetch skipped: %s", exc)


# ── which frame of a video latent to show ───────────────────────────────
# Core shows frame 0, every time, for the whole sample. For a video model
# that is the WORST frame to pick: it is usually the one pinned to the
# conditioning image, so it looks correct from the first step whether or not
# the rest of the clip is working, and it never changes. You watch a still
# picture for two minutes and learn nothing.
#
#   sweep  (default) advance one frame per rendered preview, wrapping. Over a
#          20-step sample you see 20 different frames instead of the same one
#          20 times, so motion and temporal collapse are both visible.
#   middle a fixed frame from the centre of the clip - stable, and still far
#          more informative than frame 0.
#   first  core's behaviour, for anyone who wants it back.
#
# The frame is chosen by SLICING the latent before handing it to whichever
# previewer is active, so this works for TAESD, Latent2RGB and the
# channel-mean net alike - each of them takes "the first frame" of what it is
# given, and this changes what that is.
_VIDEO_FRAME_POLICY = os.environ.get("C2C_PREVIEW_VIDEO_FRAME", "sweep").lower()


class _FramePickingPreviewer:
    """Wraps a previewer so a video latent does not show frame 0 forever."""

    def __init__(self, inner, policy: str):
        self._inner = inner
        self._policy = policy
        self._n = 0

    def _pick(self, x0):
        # (B, C, T, H, W) only - an image latent is 4-D and passes straight
        # through untouched.
        if getattr(x0, "ndim", 0) != 5:
            return x0
        frames = int(x0.shape[2])
        if frames <= 1:
            return x0
        if self._policy == "middle":
            i = frames // 2
        else:
            i = self._n % frames
            self._n += 1
        return x0[:, :, i:i + 1]

    def decode_latent_to_preview_image(self, preview_format, x0):
        try:
            x0 = self._pick(x0)
        except Exception:
            pass  # a shape we did not expect: show whatever core would have
        return self._inner.decode_latent_to_preview_image(preview_format, x0)

    def decode_latent_to_preview(self, x0):
        try:
            x0 = self._pick(x0)
        except Exception:
            pass
        return self._inner.decode_latent_to_preview(x0)

    def __getattr__(self, item):
        return getattr(self._inner, item)


def _with_frame_policy(prev, latent_format):
    """Apply the frame policy to a video previewer, if one is wanted."""
    if prev is None or _VIDEO_FRAME_POLICY == "first":
        return prev
    try:
        if not _is_video_latent(latent_format):
            return prev
        return _FramePickingPreviewer(prev, _VIDEO_FRAME_POLICY)
    except Exception:
        return prev


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
        # If this model's TAE decoder is missing, start fetching it. Returns
        # immediately - this sample previews on the fallback, the next is
        # sharp. Placed here because it is the one function every sampler
        # reaches, core or third-party.
        _ensure_decoder(latent_format)

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
        # Wrap at the assignment, not at each return, so EVERY path out of
        # _patched_get_previewer is throttleable — a previewer that escapes
        # unwrapped would decode on steps we meant to skip.
        latent_preview.get_previewer = (
            lambda device, latent_format:
                _throttling_previewer(
                    _with_frame_policy(
                        _patched_get_previewer(device, latent_format),
                        latent_format))
        )
        latent_preview._c2c_previewer_patched = True
        log.info("[c2c.preview] installed smart-Auto + never-None get_previewer wrapper "
                 "(video->TAESD; image->TAESD when a decoder file is present, else Auto; "
                 "per model, with a channel-mean grayscale previewer as the absolute "
                 "last resort for formats with neither a decoder nor rgb factors). "
                 "Missing image decoders are fetched in the background; video frame "
                 "policy is %r.", _VIDEO_FRAME_POLICY)
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


class _SuppressFlag(threading.local):
    """Per-thread preview suppression. Thread-local because ComfyUI can run
    more than one sampler, and a global would let one sampler blank another's
    preview."""
    value = False


_SUPPRESS = _SuppressFlag()


def _throttling_previewer(prev):
    """Wrap a previewer so a throttled step yields NO preview bytes.

    This is where throttling belongs. The alternative — handing core a None
    x0 — is not a way to skip a decode: core assigns x0 into x0_output_dict
    (SamplerCustom's denoised latent) and passes it to the previewer
    regardless. Returning None here is exactly what core already treats as
    "no preview this step": `preview_bytes = None` then
    `pbar.update_absolute(step, total, preview_bytes)`.
    """
    if prev is None or getattr(prev, "_c2c_throttled", False):
        return prev
    orig_decode = prev.decode_latent_to_preview_image

    def _decode(preview_format, x0):
        if _SUPPRESS.value:
            return None
        return orig_decode(preview_format, x0)

    try:
        prev.decode_latent_to_preview_image = _decode
        prev._c2c_throttled = True
    except Exception:  # noqa: BLE001 — slotted/immutable previewer: leave as-is
        pass
    return prev


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
                # Suppress only the PREVIEW, never x0 itself. Core does
                #     x0_output_dict["x0"] = x0
                #     previewer.decode_latent_to_preview_image(fmt, x0)
                # with no None check, so passing x0=None both corrupts
                # SamplerCustom's denoised output and crashes the previewer
                # (TAESDPreviewerImpl does x0[:1] -> TypeError). The flag below
                # makes our previewer wrapper return no bytes for this call
                # while the real x0 flows through untouched.
                _SUPPRESS.value = True
                try:
                    native_cb(step, x0, x, total_steps)
                finally:
                    _SUPPRESS.value = False
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
        # getattr INSIDE the try. `import comfy.utils` can succeed while
        # `comfy.utils` is not set as an attribute on the `comfy` package
        # object - that is exactly what a partially-stubbed or partially-
        # initialised comfy looks like - and the AttributeError then escapes
        # this installer, propagates out of the module import, and takes the
        # whole pack out of /object_info over a preview nicety. The module
        # header promises "any change in ComfyUI's preview API simply no-ops
        # here"; this line is what makes that true.
        PB = getattr(comfy.utils, "ProgressBar", None)
    except Exception:
        return
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
#
# Each installer guards itself, and this belt-and-braces layer is here because
# the cost of being wrong is wildly asymmetric: every one of these is a
# NICETY - sharper previews, a smoother frame rate - while an exception
# escaping this module at import time removes EVERY node in the pack from
# /object_info. Nothing below is worth that trade, so a failure is logged and
# the rest still runs.
for _step in (ensure_previews_enabled, _install_previewer_fallback,
              _install_prepare_callback_throttle, _install_pbar_safety_net,
              _register_routes):
    try:
        _step()
    except Exception as _exc:  # noqa: BLE001
        log.warning("[c2c.preview] %s skipped: %s", _step.__name__, _exc)
