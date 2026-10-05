"""Which of MaskOps' 68 widgets are worth showing right now.

THE PROBLEM. MaskOpsMEC exposes 68 parameters. Most of them belong to one
backend or one optional feature, so on any given setup the large majority do
nothing — but they all sit on the node looking like they do. That is worse
than clutter: a control that is visible and ignored teaches you the wrong
thing about what the node does, and then you spend an afternoon adjusting
`trimap_dilate` on a matter that has no trimap.

Measured at default settings, not estimated: of the node's 71 parameters,
SAM3.1 + ViTMatte puts 46 on screen and BiRefNet + ViTMatte 39, rising to 57
once the luma key, despill and a hard edge are switched on. Every optional
feature group stays folded until its own switch is on. Not a dozen — the core
set plus the feature switches is already about forty — but the ones left all
do something.

TWO KINDS OF RULE, because there are two reasons a widget is irrelevant:

  BACKEND    `trimap_dilate` exists for ViTMatte and nothing else. The
             backends already declare what they read in their own PARAMS, so
             this reads the registry rather than keeping a second list —
             a second list is how the folder incrementer's loader table went
             stale.

  TOGGLE     `luma_low` only matters while `enable_luma_key` is on. That is
             not backend knowledge, it is a feature switch, so it is declared
             here in one table.

WHAT IS NEVER HIDDEN. Anything carrying data — image, external_mask,
holdout_mask — and anything a hidden widget would silently change the meaning
of. Hiding a socket that is WIRED would leave the user with a connection they
cannot see; the rule is that sockets stay, only widgets fold away.
"""

from __future__ import annotations

from typing import Any, Dict, List

#: Widgets that are always on screen: the picture, the two model pickers, and
#: the controls that decide what the other rules do.
ALWAYS = (
    "image", "segmenter", "matter", "model", "matter_model",
    "edge_mode", "edge_radius", "subject_preset",
    "auto_quality", "post_refine", "despill",
    "enable_luma_key", "enable_advanced_trimap", "enable_diagnose",
    "robust_propagation", "auto_download", "seed",
    # which pipeline runs decides what everything else means
    "pipeline",
)

#: Pipeline-level widgets: they belong to the NODE, not to a backend, so no
#: segmenter will ever name them in PARAMS. They have to be listed somewhere
#: or they can never be shown at all — which is the bug this tuple fixes:
#: every one of these had a TOGGLES entry and no way into the candidate set,
#: so turning on `enable_luma_key` revealed nothing and the feature looked
#: broken. A TOGGLES entry RESTRICTS when a widget is shown; it never grants.
FEATURE = (
    # colour / compositing stage
    "despill_strength", "preserve_skin",
    "lightwrap_strength", "lightwrap_radius",
    "edge_band_radius", "premultiply",
    "edge_threshold", "edge_feather",
    # accuracy stage
    "tta_flip", "multiscale",
    "refine_radius", "refine_iterations",
    "quality_mode", "auto_disambiguate",
    # trimap stage
    "trimap_dilate", "trimap_erode",
    "trimap_inner_scale", "trimap_outer_scale",
    "trimap_smooth", "trimap_threshold",
    # luma-key pre-stage
    "luma_mode", "luma_low", "luma_high", "luma_gamma",
    "luma_falloff", "luma_invert", "luma_mix",
    # diagnostics
    "diag_ring_width", "diag_blur_threshold", "diag_brightness_threshold",
    # robust propagation
    "robust_confidence_threshold", "robust_reanchor_method",
    "robust_blend_alpha",
    # runtime
    "precision", "attention", "offload",
    # the ONYX pipeline's own controls (gated on `pipeline` in TOGGLES)
    "scene_prompts", "band_scale", "matte_tile", "temporal_stabilise",
    # ViTMatte cascade tiling (gated on matter + pipeline in tiling_visible)
    "matte_overlap", "matte_tile_batch",
)

#: pipeline="onyx" is ONE fixed pipeline - a SAM 3.1 video session, a trimap
#: whose band follows the resolution, tiled ViTMatte - so the cascade's own
#: machinery reads nothing there: the segmenter picker, the accuracy stages,
#: the luma pre-key, the pixel-sized trimap knobs, the re-anchoring. Showing
#: them teaches that they matter. The front-end drops these while ONYX is on.
ONYX_IGNORES = (
    "segmenter", "tta_flip", "multiscale",
    "auto_quality", "quality_mode", "auto_disambiguate",
    "robust_propagation", "robust_confidence_threshold",
    "robust_reanchor_method", "robust_blend_alpha",
    "enable_luma_key", "luma_mode", "luma_low", "luma_high", "luma_gamma",
    "luma_falloff", "luma_invert", "luma_mix",
    "enable_advanced_trimap", "trimap_inner_scale", "trimap_outer_scale",
    "trimap_smooth", "trimap_threshold",
    "trimap_dilate", "trimap_erode", "edge_radius", "subject_preset",
    "individual_objects", "object_id", "memory_size", "end_frame",
)
#: ...and what it DOES read beyond the core: the single-object shorthand
#: (legacy prompt sockets + the tracking window) used when scene_prompts is
#: empty.
ONYX_KEEPS = (
    "frame_annotation", "tracking_direction", "start_frame", "max_frames_to_track",
    "text_prompt", "positive_coords", "negative_coords",
)
_ONYX_ONLY = ("scene_prompts", "band_scale", "temporal_stabilise")

#: The simple trimap pair belongs to the MATTER that builds a trimap: ViTMatte names both in its PARAMS, RVM
#: and the salient matters name neither. Listing them in FEATURE (every pair) put them on screen for RVM,
#: visible and ignored - the exact failure this module exists to stop (tests/test_mask_visibility.py,
#: "trimap widget only appears for a matter that uses one"). They reach the candidate set through the
#: matter's PARAMS instead; the served spec omits them from "feature" so the front-end agrees.
TRIMAP_WIDGETS = frozenset({"trimap_dilate", "trimap_erode"})

#: ViTMatte tile controls — matte_tile on ONYX and cascade+vitmatte; overlap/batch
#: cascade+vitmatte only (ONYX tiling is internal to _onyx_pipeline).
_VITMATTE_TILE_WIDGETS = frozenset({"matte_tile", "matte_overlap", "matte_tile_batch"})
_CASCADE_VITMATTE_TILE = frozenset({"matte_overlap", "matte_tile_batch"})

#: Inputs that take data down a wire. Never folded away: a hidden socket that
#: is wired is a connection the user cannot see or undo.
SOCKETS = (
    "image", "external_mask", "external_trimap", "holdout_mask", "core_mask",
    "pos_bbox", "neg_bbox", "normal_bbox",
)

#: widget -> the switch that has to be on for it to matter.
#: `(widget, value)` means "visible while that widget is not this value".
TOGGLES: Dict[str, Any] = {
    "luma_mode": "enable_luma_key",
    "luma_low": "enable_luma_key",
    "luma_high": "enable_luma_key",
    "luma_gamma": "enable_luma_key",
    "luma_falloff": "enable_luma_key",
    "luma_invert": "enable_luma_key",
    "luma_mix": "enable_luma_key",

    "trimap_inner_scale": "enable_advanced_trimap",
    "trimap_outer_scale": "enable_advanced_trimap",
    "trimap_smooth": "enable_advanced_trimap",
    "trimap_threshold": "enable_advanced_trimap",

    "diag_ring_width": "enable_diagnose",
    "diag_blur_threshold": "enable_diagnose",
    "diag_brightness_threshold": "enable_diagnose",

    "robust_confidence_threshold": "robust_propagation",
    "robust_reanchor_method": "robust_propagation",
    "robust_blend_alpha": "robust_propagation",

    "quality_mode": "auto_quality",
    "auto_disambiguate": "auto_quality",

    "refine_radius": ("post_refine", "none"),
    "refine_iterations": ("post_refine", "none"),

    # The widget's off-value is "off", not "none" — gating on the wrong
    # string kept these on screen permanently, which looked like the rule
    # was not running at all.
    "despill_strength": ("despill", "off"),
    "preserve_skin": ("despill", "off"),

    # Light wrap is NOT part of despill: you can wrap without decontaminating
    # and frequently want to. It gates on its own strength being non-zero.
    "lightwrap_radius": "lightwrap_strength",

    # The hard/feather controls only mean something once a mode is chosen.
    "edge_threshold": ("edge_mode", "soft"),
    "edge_feather": ("edge_mode", "soft"),

    # ONYX's own controls read nothing on the legacy cascade.
    "scene_prompts": ("pipeline", "cascade (legacy)"),
    "band_scale": ("pipeline", "cascade (legacy)"),
    "temporal_stabilise": ("pipeline", "cascade (legacy)"),
}

#: `edge_band_radius` and `premultiply` are deliberately NOT toggled: the
#: edge/inside/outside outputs and the preview are produced on every run, so
#: both always do something.

#: Widgets a segmenter reads that its PARAMS may not name, because they are
#: about the CLIP rather than the prompt. Any backend that needs the whole
#: clip (NEEDS_VIDEO) gets the tracking window.
VIDEO_WINDOW = (
    "frame_annotation", "object_id", "memory_size", "max_frames_to_track",
    "start_frame", "end_frame", "tracking_direction", "individual_objects",
    "robust_propagation",
)

#: The prompt widgets, and which mode each belongs to. A backend that cannot
#: take text should not show a text box: typing in it and getting nothing is
#: the single most confusing thing this node can do.
BY_MODE = {
    "text": ("text_prompt",),
    "points": ("positive_coords", "negative_coords"),
    "bbox": ("pos_bbox", "neg_bbox"),
}


#: A backend's PARAMS names the kwargs ITS OWN `segment()` takes, which is not
#: the same vocabulary as MaskOps' widget names — `positive_points` is the
#: backend's word for what the node calls `pos_points`, and `bbox` covers three
#: separate widgets. Reusing PARAMS directly added names that matched no widget
#: and therefore did nothing, silently. This translates; PARAMS keeps its
#: existing meaning rather than every backend growing a second list to go
#: stale.
PARAM_TO_WIDGET: Dict[str, tuple] = {
    # Only the *_coords pair are real widgets; pos_points / neg_points are
    # legacy kwargs on the implementation and appear on no node.
    "positive_points": ("positive_coords",),
    "negative_points": ("negative_coords",),
    "bbox": ("pos_bbox", "neg_bbox", "normal_bbox"),
    "max_frames": ("max_frames_to_track",),
    "text": ("text_prompt",),
    "prompt": ("text_prompt",),
}

#: Backend params that are real, are used, and deliberately have NO widget —
#: they are computed or fixed by the node. Listed so that a NEW param arriving
#: from a new backend fails the parity test loudly instead of being dropped on
#: the floor, which is how `positive_points` went unnoticed.
NOT_EXPOSED = frozenset({
    "multimask",         # node always takes the best mask
    "matte_resolution",  # follows the image
    "downsample_ratio",  # RVM picks from the frame size
    "recurrent_state",   # carried internally between frames
    "warmup_frames",     # MatAnyone, fixed
    "box_threshold", "text_threshold",   # locate_anything, fixed
})


def _widgets_of(params) -> set:
    """Translate a backend's PARAMS into MaskOps widget names."""
    out = set()
    for name in params or ():
        if name in PARAM_TO_WIDGET:
            out.update(PARAM_TO_WIDGET[name])
        elif name not in NOT_EXPOSED:
            out.add(name)
    return out


def _seg_classes() -> Dict[str, type]:
    try:
        from .segmenters import all_segmenters
        return all_segmenters()
    except Exception:  # noqa: BLE001 - a registry that will not load is not fatal
        return {}


def _mat_classes() -> Dict[str, type]:
    try:
        from .matters import all_matters
        return all_matters()
    except Exception:  # noqa: BLE001
        return {}


def widgets_for(segmenter: str, matter: str) -> List[str]:
    """Every widget worth showing for this pair, ignoring toggles.

    PARAMS of None means the backend has not said, and the honest answer
    there is to show everything rather than hide a control that might matter.
    An empty tuple is a real declaration - "reads nothing extra" - and shows
    the core only. Collapsing the two would make silence look like a claim.
    """
    out = set(ALWAYS) | set(SOCKETS) | (set(FEATURE) - TRIMAP_WIDGETS)

    seg = _seg_classes().get(segmenter)
    if seg is None:
        return sorted(out)                       # unknown: show the core only
    declared = getattr(seg, "PARAMS", None)
    if declared is None:
        return ["*"]          # has not said: show everything, and say why
    out.update(_widgets_of(declared))
    for mode, names in BY_MODE.items():
        if mode in (getattr(seg, "SUPPORTS_MODES", set()) or set()):
            out.update(names)
    if getattr(seg, "NEEDS_VIDEO", False):
        out.update(VIDEO_WINDOW)

    mat = _mat_classes().get(matter)
    if mat is not None:
        mdec = getattr(mat, "PARAMS", None)
        if mdec is None:
            return ["*"]
        out.update(_widgets_of(mdec))
    return sorted(out)


def tiling_visible(name: str, matter: str, values: Dict[str, Any]) -> bool:
    """ViTMatte tile widgets: matte_tile on ONYX and cascade+vitmatte."""
    if name not in _VITMATTE_TILE_WIDGETS:
        return True
    if matter != "vitmatte":
        return False
    pipeline = str(values.get("pipeline", "cascade (legacy)"))
    if pipeline == "onyx":
        return name == "matte_tile"
    if pipeline == "cascade (legacy)":
        return True
    return False


def toggle_allows(name: str, values: Dict[str, Any]) -> bool:
    """Is this widget's feature switch currently on?

    `(widget, value)` means "visible while that widget is NOT this value";
    a bare name means "visible while that widget is truthy".
    """
    rule = TOGGLES.get(name)
    if rule is None:
        return True
    if isinstance(rule, tuple):
        return str(values.get(rule[0])) != str(rule[1])
    v = values.get(rule)
    return not (v is False or v == 0 or v is None or v == "")


def visible(segmenter: str, matter: str, values: Dict[str, Any]) -> List[str]:
    """What is actually ON SCREEN for this pair at these settings.

    `widgets_for` is the candidate set — everything this pair could ever read.
    This is the smaller number that the user sees, because a feature group
    stays folded until its own switch is on. The front-end applies the same
    two rules from the served spec; this is the reference they are tested
    against.
    """
    if str(values.get("pipeline", "")) == "onyx":
        # One fixed pipeline: the segmenter picker does not decide anything,
        # so the candidates are the core + ONYX's reads + the matter's own.
        cand = set(ALWAYS) | set(SOCKETS) | set(FEATURE) | set(ONYX_KEEPS)
        mat = _mat_classes().get(matter)
        if mat is not None:
            cand |= _widgets_of(getattr(mat, "PARAMS", ()) or ())
        cand -= set(ONYX_IGNORES)
        return sorted(
            n for n in cand
            if n in SOCKETS
            or (toggle_allows(n, values) and tiling_visible(n, matter, values))
        )
    cand = widgets_for(segmenter, matter)
    if cand == ["*"]:
        return cand
    return sorted(
        n for n in cand
        if n in SOCKETS
        or (toggle_allows(n, values) and tiling_visible(n, matter, values))
    )


def build_spec() -> Dict[str, Any]:
    """The whole table, for the front-end to apply without asking again.

    Sent as data rather than reimplemented in JS: the backends' PARAMS are the
    single source of truth, and a second copy in the front-end is how it goes
    stale the first time a backend is added.
    """
    segs = _seg_classes()
    mats = _mat_classes()
    return {
        "always": list(ALWAYS),
        "feature": [f for f in FEATURE if f not in TRIMAP_WIDGETS],
        "sockets": list(SOCKETS),
        "toggles": {k: (list(v) if isinstance(v, tuple) else v)
                    for k, v in TOGGLES.items()},
        "segmenters": {
            key: {
                "params": sorted(_widgets_of(getattr(cls, "PARAMS", ()) or ())),
                "modes": sorted(getattr(cls, "SUPPORTS_MODES", set()) or set()),
                "needs_video": bool(getattr(cls, "NEEDS_VIDEO", False)),
                "status": getattr(cls, "STATUS", "unknown"),
                "display": getattr(cls, "DISPLAY", key),
            }
            for key, cls in segs.items()
        },
        "matters": {
            key: {
                "params": sorted(_widgets_of(getattr(cls, "PARAMS", ()) or ())),
                "temporal": bool(getattr(cls, "TEMPORAL", False)),
                "status": getattr(cls, "STATUS", "unknown"),
                "display": getattr(cls, "DISPLAY", key),
            }
            for key, cls in mats.items()
        },
        "video_window": list(VIDEO_WINDOW),
        "by_mode": {k: list(v) for k, v in BY_MODE.items()},
        "onyx": {"ignores": list(ONYX_IGNORES), "keeps": list(ONYX_KEEPS),
                 "only": list(_ONYX_ONLY)},
        "vitmatte_tiling": {
            "widgets": sorted(_VITMATTE_TILE_WIDGETS),
            "cascade_only": sorted(_CASCADE_VITMATTE_TILE),
        },
    }


def describe(segmenter: str, matter: str, total: int) -> str:
    """What was hidden and why — so a missing control is never a mystery."""
    shown = widgets_for(segmenter, matter)
    if shown == ["*"]:
        return (f"{segmenter} has not declared which parameters it reads, so "
                "every control is shown. That is deliberate: hiding a control "
                "that might matter is worse than showing one that does not.")
    n = len(shown)
    lines = [
        f"{segmenter} + {matter}: {n} of {total} parameters apply; the rest "
        "are folded away because this combination does not read them.",
    ]
    seg = _seg_classes().get(segmenter)
    if seg is not None:
        modes = sorted(getattr(seg, "SUPPORTS_MODES", set()) or set())
        lines.append("Prompting: " + (", ".join(modes) if modes else "none"))
        if getattr(seg, "NEEDS_VIDEO", False):
            lines.append(
                "This backend tracks over time, so the tracking-window "
                "controls are shown and it wants the whole clip rather than a "
                "single frame.")
    mat = _mat_classes().get(matter)
    if mat is not None and not getattr(mat, "TEMPORAL", False):
        lines.append(
            f"NOTE: {matter} mattes each frame independently. On a clip that "
            "flickers — it is the commonest complaint about an otherwise good "
            "matte. RVM or MatAnyone carry state between frames.")
    return "\n".join(lines)


# ── serving the spec to the front-end ───────────────────────────────────────

def register_routes() -> bool:
    """Expose the spec at /c2c/mask/visibility.

    Served rather than baked into the JS because the answer depends on which
    backends this install actually has: a machine without SeC must not be
    shown SeC's controls. Never raises — a front-end that cannot reach the
    route falls back to showing everything, which is the same honest default
    an undeclared backend gets.
    """
    try:
        from server import PromptServer
        from aiohttp import web
    except Exception:  # noqa: BLE001 - no server here; nothing to register
        return False
    inst = getattr(PromptServer, "instance", None)
    routes = getattr(inst, "routes", None) if inst else None
    if routes is None or not hasattr(routes, "get"):
        return False

    @routes.get("/c2c/mask/visibility")
    async def _visibility(_request):           # noqa: ANN001
        try:
            return web.json_response(build_spec())
        except Exception as exc:  # noqa: BLE001
            return web.json_response(
                {"error": str(exc), "always": list(ALWAYS)}, status=200)

    return True
