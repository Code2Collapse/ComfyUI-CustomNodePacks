"""
ComfyUI-CustomNodePacks
=======================
A growing collection of custom nodes:
  - FolderIncrementer – auto-incrementing version strings
  - MaskEditControl  – pinpoint mask editing, SAM2/SAM3, per-axis erode/expand,
                       point editing, bbox tools, video mask propagation,
                       alpha matting (ViTMatte / MatAnyone2)
  - Universal Reroute – Nuke-style Dot node for clean wire management
  - Parameter Memory  – tracks every parameter change with history & defaults
"""

print("[MEC] Loading MaskEditControl node pack …")

# ── FolderIncrementer nodes ────────────────────────────────────────────
from .folder_incrementer import (
    NODE_CLASS_MAPPINGS as _FOLDER_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as _FOLDER_DISPLAY,
)

# ── Model Manager (shared cache / download) ───────────────────────────
from .nodes import model_manager as _model_manager  # noqa: F401

# ── MaskEditControl nodes ─────────────────────────────────────────────
# Unified composition wrappers — these replace 12 legacy node classes:
#   MaskEditMEC      replaces MaskTransformXY, MaskDrawFrame, DrawShapeMEC,
#                    PointsMaskEditor, BBoxSmooth
#   SplineMaskMEC    replaces SplineMaskEditorMEC, SplineMaskTrackerMEC,
#                    SplinePathFlowMaskMEC
#   MaskTrackerMEC   replaces MotionMaskTrackerMEC, MaskPropagateVideo,
#                    TemporalAnchorMEC, TemporalConsistencyCheckerMEC
# The original source files remain on disk as internal implementation
# modules; they are imported by the unified wrappers via composition.
# No legacy NODE_CLASS_MAPPINGS entries are registered.
from .nodes.mask_edit_mec import MaskEditMEC
from .nodes.spline_mask_mec import SplineMaskMEC
from .nodes.mask_tracker_mec import MaskTrackerMEC
from .nodes.parameter_memory import ParameterHistoryMEC
from .nodes.sec_matanyone_pipeline import SeCMatAnyonePipelineMEC
from .nodes.inpaint_suite import (
    InpaintCropProMEC,
    InpaintStitchProMEC,
    InpaintPasteBackMEC,
    InpaintCompositeMEC,
)
# VideoComparerC2C (renamed from VideoComparerMEC) replaces the deprecated
# ImageComparerMEC. The old image_comparer.py is retained on disk as an
# importable helper (ImageComparerMEC class), but only VideoComparerC2C is
# registered — with "VideoComparerMEC" kept as a back-compat alias key so
# saved workflows still load.
from .nodes.video_comparer import VideoComparerC2C
from .nodes.video_frame_player import VideoFramePlayerMEC
from .nodes.video_mask_editor import (
    VideoMaskEditorMEC,
    register_routes as _register_vme_routes,
)
from .nodes.vae_merge import VAEMergeMEC
from .nodes.vae_latent_inspector import VAELatentInspectorMEC
from .nodes.batch_version_manager import BatchVersionManagerMEC
from .nodes.model_metadata_extractor import ModelMetadataExtractorMEC
from .nodes.mask_failure_explainer import MaskFailureExplainerMEC

# ── MEC Paint Suite (Advanced Paint Canvas + Fixer + Refiner + Builder) ───
from .nodes.mec_paint_suite import (
    NODE_CLASS_MAPPINGS as _PAINT_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as _PAINT_DISPLAY,
)
# Face Fixer (auto YOLO11 detection + per-face KSampler + smart blend)
from .nodes.mec_face_fixer import (
    NODE_CLASS_MAPPINGS as _FACE_FIXER_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as _FACE_FIXER_DISPLAY,
)
# Mask + Matting (multi-backend: SAM2.1, SAM3 + ViTMatte, RVM, ...)
from .nodes.mask_matting import (
    NODE_CLASS_MAPPINGS as _MASKMATTE_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as _MASKMATTE_DISPLAY,
)
# Unified Segmentation (SAM2/SAM3/SeC, single node)
try:
    from .nodes.unified_segmentation import (
        NODE_CLASS_MAPPINGS as _USEG_MAPPINGS,
        NODE_DISPLAY_NAME_MAPPINGS as _USEG_DISPLAY,
    )
except Exception as _exc:  # pragma: no cover
    import logging
    logging.getLogger("MEC.UnifiedSeg").warning(
        "UnifiedSegmentation unavailable: %s", _exc
    )
    _USEG_MAPPINGS, _USEG_DISPLAY = {}, {}

# SAM Multi-Mask Picker — interactive 3-thumbnail mask chooser
# (user-recalled feature: pick best of N SAM candidates by score)
try:
    from .nodes.sam_multi_mask_picker import SamMultiMaskPickerMEC
    _SAMPICKER_MAPPINGS = {"SamMultiMaskPickerMEC": SamMultiMaskPickerMEC}
    _SAMPICKER_DISPLAY = {
        "SamMultiMaskPickerMEC": "SAM Multi-Mask Picker — 3 candidates + scores",
    }
except Exception as _exc:  # pragma: no cover
    import logging
    logging.getLogger("MEC.SamMultiPicker").warning(
        "SamMultiMaskPickerMEC unavailable: %s", _exc
    )
    _SAMPICKER_MAPPINGS, _SAMPICKER_DISPLAY = {}, {}

# Wan Director (P9) — multi-shot orchestration utilities
try:
    from .nodes.wan_director import (
        NODE_CLASS_MAPPINGS as _WANDIR_MAPPINGS,
        NODE_DISPLAY_NAME_MAPPINGS as _WANDIR_DISPLAY,
    )
except Exception as _exc:  # pragma: no cover
    import logging
    logging.getLogger("MEC.WanDirector").warning(
        "WanDirector unavailable: %s", _exc
    )
    _WANDIR_MAPPINGS, _WANDIR_DISPLAY = {}, {}

# C2C helpers (12 tiny utilities)
try:
    from .nodes.helpers import (
        NODE_CLASS_MAPPINGS as _HELPERS_MAPPINGS,
        NODE_DISPLAY_NAME_MAPPINGS as _HELPERS_DISPLAY,
    )
except Exception as _exc:  # pragma: no cover
    import logging
    logging.getLogger("C2C.helpers").warning(
        "C2C helpers unavailable: %s", _exc
    )
    _HELPERS_MAPPINGS, _HELPERS_DISPLAY = {}, {}

# Prompt Relay — refined port (native + Kijai + generic-fallback backends).
# Algorithm credit: Gordon Chen & contributors. See nodes/prompt_relay/NOTICE.md.
try:
    from .nodes.prompt_relay import (
        NODE_CLASS_MAPPINGS as _PROMPTRELAY_MAPPINGS,
        NODE_DISPLAY_NAME_MAPPINGS as _PROMPTRELAY_DISPLAY,
    )
except Exception as _exc:  # pragma: no cover
    import logging
    logging.getLogger("C2C.PromptRelay").warning(
        "PromptRelay unavailable: %s", _exc
    )
    _PROMPTRELAY_MAPPINGS, _PROMPTRELAY_DISPLAY = {}, {}

# ── NukeNodeMax suite (P0..F7) ────────────────────────────────────────
# ── ProPainter unified dispatcher (absorbs Temporal/Remove/Stitch/StitchRefine/FlowRefine) ──
# Helper source files are kept on disk as importable Python classes; only
# ProPainterMEC is registered here.
from .nodes.propainter_unified import ProPainterMEC
from .nodes.video_stabilizer_mec import (
    NODE_CLASS_MAPPINGS as _STABILIZER_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as _STABILIZER_DISPLAY,
)
# NOTE: The following node families now live EXCLUSIVELY in
# ComfyUI-NukeMaxNodes (May 2026 migration) to avoid duplicate functionality
# and divergent type semantics. The MEC copies have been moved to _deprecated/.
#   - Deep*    (DeepFromImage / DeepMerge / DeepHoldout / DeepComposite)
#   - Roto     (VectorRotoMEC      -> NukeMax_RotoSplineEditor + suite)
#   - Shuffle  (ShuffleMEC         -> NukeMax_ShuffleImage / NukeMax_ShuffleLatent)
#   - Flow     (OpticalFlowMEC     -> NukeMax_ComputeOpticalFlow + warps)
#   - Tcl/Nk   (TclSerialize/Parse -> NukeMax_NkScriptSerialize / Parse)
# FlowRefineMEC (post-flow inpainting prep) is kept here because it is part of
# the ProPainter pipeline, not a generic Nuke flow utility.
from .nodes.insight import InsightStatusMEC, install as _install_insight_hook
from .nodes.integrity_guard import (
    IntegrityStatusMEC,
    register_routes as _register_integrity_routes,
    start_background_scan as _start_integrity_scan,
)

_NUKEMAX_MAPPINGS = {
    "ProPainterMEC": ProPainterMEC,
    "InsightStatusMEC": InsightStatusMEC,
    "IntegrityStatusMEC": IntegrityStatusMEC,
}
_NUKEMAX_DISPLAY = {
    "ProPainterMEC": "ProPainter — Temporal / Remove / Stitch / Refine / Flow",
    "InsightStatusMEC": "Insight Status",
    "IntegrityStatusMEC": "Integrity Status",
}

# ── VFX nodes migrated to ComfyUI-NukeMaxNodes (Apr 2026) ─────────────
# (color_science, exr_io, render_pass, plate_tools, geometry_nodes,
#  metadata_nodes, exr_metadata_reader, universal_reroute)
# Model analysis stays here:
from .nodes.model_analysis import (
    NODE_CLASS_MAPPINGS as _MA_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as _MA_DISPLAY,
)
# Legacy mask refinement suite (DenseCRF / Guided / Thin / QualityScore / Trimap)
# is fully replaced by the single ``MaskRefineMEC`` node registered via
# ``mask_matting`` package — no separate registration here.

_MEC_MAPPINGS = {
    "MaskEditMEC": MaskEditMEC,
    "SplineMaskMEC": SplineMaskMEC,
    "MaskTrackerMEC": MaskTrackerMEC,
    "ParameterHistoryMEC": ParameterHistoryMEC,
    "SeCMatAnyonePipelineMEC": SeCMatAnyonePipelineMEC,
    "InpaintCropProMEC": InpaintCropProMEC,
    "InpaintCompositeMEC": InpaintCompositeMEC,
    "InpaintStitchProMEC": InpaintStitchProMEC,
    "InpaintPasteBackMEC": InpaintPasteBackMEC,
    "VideoComparerC2C": VideoComparerC2C,
    # NOTE: "VideoComparerMEC" was the legacy class key. To prevent it from
    # showing up as a separate (duplicate) entry in the node search palette,
    # it is NO LONGER registered here. Saved workflows that reference the old
    # type are migrated at graph-load time by js/video_comparer_c2c.js, which
    # rewrites `type: "VideoComparerMEC"` -> `"VideoComparerC2C"` before
    # LiteGraph instantiates the node.
    "VideoFramePlayerMEC": VideoFramePlayerMEC,
    "VideoMaskEditorMEC": VideoMaskEditorMEC,
    "VAEMergeMEC": VAEMergeMEC,
    "VAELatentInspectorMEC": VAELatentInspectorMEC,
    "BatchVersionManagerMEC": BatchVersionManagerMEC,
    "ModelMetadataExtractorMEC": ModelMetadataExtractorMEC,
    "MaskFailureExplainerMEC": MaskFailureExplainerMEC,
}

_MEC_DISPLAY = {
    "MaskEditMEC": "Mask Edit — Transform/Draw/Points/BBox",
    "SplineMaskMEC": "Spline Mask — Edit/Track/Flow-Path",
    "MaskTrackerMEC": "Mask Tracker — Motion/Propagate/Anchor/Consistency",
    "ParameterHistoryMEC": "Parameter History",
    "SeCMatAnyonePipelineMEC": "SeC + MatAnyone2 Pipeline",
    "InpaintCropProMEC": "Inpaint Crop Pro",
    "InpaintCompositeMEC": "Inpaint Composite",
    "InpaintStitchProMEC": "Inpaint Stitch Pro — legacy",
    "InpaintPasteBackMEC": "Inpaint Paste Back — legacy",
    "VideoComparerC2C": "Video Comparer — Wipe/Diff/Scopes/Audio",
    "VideoFramePlayerMEC": "Video Frame Player",
    "VideoMaskEditorMEC": "Video Mask Editor",
    "VAEMergeMEC": "VAE Merge",
    "VAELatentInspectorMEC": "VAE Latent Inspector",
    "BatchVersionManagerMEC": "Batch Version Manager",
    "ModelMetadataExtractorMEC": "Model Metadata Extractor",
    "MaskFailureExplainerMEC": "Mask Failure Explainer — Diagnostics",
}

# ── Merge all mappings ────────────────────────────────────────────────
NODE_CLASS_MAPPINGS = {
    **_FOLDER_MAPPINGS,
    **_MEC_MAPPINGS,
    **_MA_MAPPINGS,
    **_PAINT_MAPPINGS,
    **_FACE_FIXER_MAPPINGS,
    **_MASKMATTE_MAPPINGS,
    **_USEG_MAPPINGS,
    **_SAMPICKER_MAPPINGS,
    **_NUKEMAX_MAPPINGS,
    **_STABILIZER_MAPPINGS,
    **_WANDIR_MAPPINGS,
    **_HELPERS_MAPPINGS,
    **_PROMPTRELAY_MAPPINGS,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    **_FOLDER_DISPLAY,
    **_MEC_DISPLAY,
    **_MA_DISPLAY,
    **_PAINT_DISPLAY,
    **_FACE_FIXER_DISPLAY,
    **_MASKMATTE_DISPLAY,
    **_USEG_DISPLAY,
    **_SAMPICKER_DISPLAY,
    **_NUKEMAX_DISPLAY,
    **_STABILIZER_DISPLAY,
    **_WANDIR_DISPLAY,
    **_HELPERS_DISPLAY,
    **_PROMPTRELAY_DISPLAY,
}

WEB_DIRECTORY = "./js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

# ── Register server routes for Parameter Memory ──────────────────────
try:
    import server as _comfy_server
    from .nodes.parameter_memory import register_routes as _register_pm_routes
    _register_pm_routes(_comfy_server.PromptServer.instance)
    print("[MEC] Parameter Memory server route registered.")
except Exception:
    pass  # Server not available (e.g. during import-only testing)

# ── Register server routes for Video Mask Editor ─────────────────────
try:
    import server as _comfy_server_vme
    _register_vme_routes(_comfy_server_vme.PromptServer.instance)
except Exception:
    pass  # Server not available

# ── Register NukeNodeMax server-side hooks & routes ───────────────────
try:
    import server as _comfy_server  # noqa: F811
    _ps = _comfy_server.PromptServer.instance
    _register_integrity_routes(_ps)
    _start_integrity_scan()
    _install_insight_hook()
    try:
        from .nodes.mec_diagnostics_api import (
            register_routes as _register_mec_diag_routes,
            install_insight_bridge as _install_mec_diag_bridge,
        )
        _register_mec_diag_routes(_ps)
        _install_mec_diag_bridge()
        print("[MEC] mec_diagnostics sidebar API registered.")
    except Exception as _diag_e:
        print(f"[MEC] mec_diagnostics deferred: {_diag_e}")
    try:
        from .nodes.node_explain import register_routes as _register_node_explain_routes
        _register_node_explain_routes(_ps)
        print("[MEC] node_explain routes registered.")
    except Exception as _ne:
        print(f"[MEC] node_explain deferred: {_ne}")
    try:
        from .nodes.error_translator import register_routes as _register_error_translator_routes
        _register_error_translator_routes(_ps)
        print("[MEC] error_translator routes registered.")
    except Exception as _et:
        print(f"[MEC] error_translator deferred: {_et}")
    try:
        from .nodes.flamegraph import register_routes as _register_flamegraph_routes
        _register_flamegraph_routes(_ps)
        print("[MEC] flamegraph routes registered.")
    except Exception as _fg:
        print(f"[MEC] flamegraph deferred: {_fg}")
    try:
        from .nodes.tensor_inspector import register_routes as _register_tensor_inspector_routes
        _register_tensor_inspector_routes(_ps)
        print("[MEC] tensor_inspector routes registered.")
    except Exception as _ti:
        print(f"[MEC] tensor_inspector deferred: {_ti}")
    try:
        from .nodes.token_counter import register_routes as _register_token_counter_routes
        _register_token_counter_routes(_ps)
        print("[MEC] token_counter routes registered.")
    except Exception as _tc:
        print(f"[MEC] token_counter deferred: {_tc}")
    try:
        from .nodes.group_presets import register_routes as _register_group_presets_routes
        _register_group_presets_routes(_ps)
        print("[MEC] group_presets routes registered.")
    except Exception as _gp:
        print(f"[MEC] group_presets deferred: {_gp}")
    try:
        from .nodes.cost_estimator import register_routes as _register_cost_estimator_routes
        _register_cost_estimator_routes(_ps)
        print("[MEC] cost_estimator routes registered.")
    except Exception as _ce:
        print(f"[MEC] cost_estimator deferred: {_ce}")
    try:
        from .nodes.wizard import register_routes as _register_wizard_routes
        _register_wizard_routes(_ps)
        print("[MEC] wizard routes registered.")
    except Exception as _wz:
        print(f"[MEC] wizard deferred: {_wz}")
    try:
        from .nodes.workflow_doctor import register_routes as _register_workflow_doctor_routes
        _register_workflow_doctor_routes(_ps)
        print("[C2C] workflow_doctor routes registered.")
    except Exception as _wd:
        print(f"[C2C] workflow_doctor deferred: {_wd}")
    try:
        from .nodes.style_presets import register_routes as _register_style_presets_routes
        _register_style_presets_routes(_ps)
        print("[C2C] style_presets routes registered.")
    except Exception as _sp:
        print(f"[C2C] style_presets deferred: {_sp}")
    try:
        from .nodes.mask_matting.integrity_bridge import register_routes as _register_integrity_bridge_routes
        _register_integrity_bridge_routes(_ps)
        print("[C2C] mask_integrity bridge routes registered.")
    except Exception as _ib:
        print(f"[C2C] mask_integrity bridge deferred: {_ib}")
    # ── C2C AI spine (v2.0-dev) ────────────────────────────────────
    try:
        from .c2c_ai.api_routes import register_routes as _register_c2c_ai_routes
        from .c2c_ai.bootstrap import bootstrap as _c2c_ai_bootstrap
        _register_c2c_ai_routes(_ps)
        _c2c_ai_bootstrap()
        print("[C2C AI] spine registered (/c2c/ai/*).")
    except Exception as _ai:
        print(f"[C2C AI] spine deferred: {_ai}")
    print("[MEC] NukeNodeMax routes + hooks registered.")
except Exception as _e:
    print(f"[MEC] NukeNodeMax server hooks deferred: {_e}")

print(f"[MEC] Loaded {len(_MEC_MAPPINGS)} MaskEditControl nodes.")
