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

This pairs with js/c2c_live_preview.js, which renders a resilient HUD from the
`b_preview` / `progress` websocket events the server now emits.
"""
from __future__ import annotations

import logging
import os

log = logging.getLogger("c2c.preview")

# Recorded so the frontend can query what happened (via /object_info-independent log).
PREVIEW_GUARD_STATUS = "unknown"


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


# Run at import (custom_nodes load after core, so latent_preview already exists).
ensure_previews_enabled()
