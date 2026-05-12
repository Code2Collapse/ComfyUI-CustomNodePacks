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
from .nodes.mask_transform_xy import MaskTransformXY
from .nodes.mask_draw_frame import MaskDrawFrame, DrawShapeMEC
from .nodes.mask_propagate_video import MaskPropagateVideo
from .nodes.points_mask_editor import PointsMaskEditor
# Legacy SAM* nodes and standalone BackgroundRemover / SemanticSegment are
# fully superseded by MaskMattingMEC (multi-backend pipeline). Their source
# files are kept on disk for reference but no longer registered.
from .nodes.bbox_nodes import BBoxSmooth
# Legacy standalone refiners (ViTMatteRefinerMEC, MaskRefinerTemporalMEC,
# MaskRefineCRF/Guided/ThinStructure/QualityScore/TrimapFromUncertainty) are
# fully superseded by the unified MaskRefineMEC inside Mask + Matting.
from .nodes.trimap_generator import TrimapGeneratorMEC
from .nodes.parameter_memory import ParameterHistoryMEC
from .nodes.sec_matanyone_pipeline import SeCMatAnyonePipelineMEC
from .nodes.luminance_keyer import LuminanceKeyerMEC
from .nodes.mask_failure_explainer import MaskFailureExplainerMEC
from .nodes.temporal_anchor import TemporalAnchorMEC
from .nodes.inpaint_suite import (
    InpaintCropProMEC,
    InpaintStitchProMEC,
    InpaintPasteBackMEC,
    InpaintCompositeMEC,
)
from .nodes.image_comparer import ImageComparerMEC
from .nodes.video_frame_player import VideoFramePlayerMEC
from .nodes.spline_mask_editor import SplineMaskEditorMEC
from .nodes.spline_path_flow_mask import SplinePathFlowMaskMEC
from .nodes.motion_mask_tracker import MotionMaskTrackerMEC
from .nodes.spline_mask_tracker import SplineMaskTrackerMEC
from .nodes.video_mask_editor import (
    VideoMaskEditorMEC,
    register_routes as _register_vme_routes,
)
from .nodes.vae_merge import VAEMergeMEC
from .nodes.vae_latent_inspector import VAELatentInspectorMEC
from .nodes.batch_version_manager import BatchVersionManagerMEC
from .nodes.temporal_consistency_checker import TemporalConsistencyCheckerMEC
from .nodes.model_metadata_extractor import ModelMetadataExtractorMEC

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

# ── NukeNodeMax suite (P0..F7) ────────────────────────────────────────
from .nodes.propainter_temporal_inpaint import ProPainterTemporalMEC
from .nodes.propainter_flow_refine import FlowRefineMEC
from .nodes.propainter_stitch_suite import (
    NODE_CLASS_MAPPINGS as _PROPAINTER_STITCH_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as _PROPAINTER_STITCH_DISPLAY,
)
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
    "ProPainterTemporalMEC": ProPainterTemporalMEC,
    "FlowRefineMEC": FlowRefineMEC,
    "InsightStatusMEC": InsightStatusMEC,
    "IntegrityStatusMEC": IntegrityStatusMEC,
}
_NUKEMAX_DISPLAY = {
    "ProPainterTemporalMEC": "ProPainter Temporal Inpaint (MEC)",
    "FlowRefineMEC": "Optical Flow Refine (MEC)",
    "InsightStatusMEC": "Insight Status (MEC)",
    "IntegrityStatusMEC": "Integrity Status (MEC)",
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
    "MaskTransformXY": MaskTransformXY,
    "MaskDrawFrame": MaskDrawFrame,
    "MaskPropagateVideo": MaskPropagateVideo,
    "PointsMaskEditor": PointsMaskEditor,
    "BBoxSmooth": BBoxSmooth,
    "TrimapGeneratorMEC": TrimapGeneratorMEC,
    "ParameterHistoryMEC": ParameterHistoryMEC,
    "SeCMatAnyonePipelineMEC": SeCMatAnyonePipelineMEC,
    "LuminanceKeyerMEC": LuminanceKeyerMEC,
    "MaskFailureExplainerMEC": MaskFailureExplainerMEC,
    "TemporalAnchorMEC": TemporalAnchorMEC,
    "InpaintCropProMEC": InpaintCropProMEC,
    "InpaintCompositeMEC": InpaintCompositeMEC,
    "InpaintStitchProMEC": InpaintStitchProMEC,
    "InpaintPasteBackMEC": InpaintPasteBackMEC,
    "ImageComparerMEC": ImageComparerMEC,
    "VideoFramePlayerMEC": VideoFramePlayerMEC,
    "SplineMaskEditorMEC": SplineMaskEditorMEC,
    "SplinePathFlowMaskMEC": SplinePathFlowMaskMEC,
    "MotionMaskTrackerMEC": MotionMaskTrackerMEC,
    "SplineMaskTrackerMEC": SplineMaskTrackerMEC,
    "VideoMaskEditorMEC": VideoMaskEditorMEC,
    "DrawShapeMEC": DrawShapeMEC,
    "VAEMergeMEC": VAEMergeMEC,
    "VAELatentInspectorMEC": VAELatentInspectorMEC,
    "BatchVersionManagerMEC": BatchVersionManagerMEC,
    "TemporalConsistencyCheckerMEC": TemporalConsistencyCheckerMEC,
    "ModelMetadataExtractorMEC": ModelMetadataExtractorMEC,
}

_MEC_DISPLAY = {
    "MaskTransformXY": "Mask Transform XY (MEC)",
    "MaskDrawFrame": "Mask Draw Frame (MEC)",
    "MaskPropagateVideo": "Mask Propagate Video (MEC)",
    "PointsMaskEditor": "Points Mask Editor (MEC)",
    "BBoxSmooth": "BBox Smooth Temporal (MEC)",
    "TrimapGeneratorMEC": "Trimap Generator (MEC)",
    "ParameterHistoryMEC": "Parameter History (MEC)",
    "SeCMatAnyonePipelineMEC": "SeC + MatAnyone2 Pipeline (MEC)",
    "LuminanceKeyerMEC": "Luminance Keyer (MEC)",
    "MaskFailureExplainerMEC": "Mask Failure Explainer (MEC)",
    "TemporalAnchorMEC": "Temporal Anchor System (MEC)",
    "InpaintCropProMEC": "Inpaint Crop Pro (MEC)",
    "InpaintCompositeMEC": "Inpaint Composite (MEC)",
    "InpaintStitchProMEC": "Inpaint Stitch Pro — legacy (MEC)",
    "InpaintPasteBackMEC": "Inpaint Paste Back — legacy (MEC)",
    "ImageComparerMEC": "Image Comparer (MEC)",
    "VideoFramePlayerMEC": "Video Frame Player (MEC)",
    "SplineMaskEditorMEC": "Spline Mask Editor (MEC)",
    "SplinePathFlowMaskMEC": "Spline Path Flow Mask (MEC)",
    "MotionMaskTrackerMEC": "Motion Mask Tracker (MEC)",
    "SplineMaskTrackerMEC": "Spline Mask Tracker (MEC)",
    "VideoMaskEditorMEC": "Video Mask Editor (MEC)",
    "DrawShapeMEC": "Draw Shape (MEC)",
    "VAEMergeMEC": "VAE Merge (MEC)",
    "VAELatentInspectorMEC": "VAE Latent Inspector (MEC)",
    "BatchVersionManagerMEC": "Batch Version Manager (MEC)",
    "TemporalConsistencyCheckerMEC": "Temporal Consistency Checker (MEC)",
    "ModelMetadataExtractorMEC": "Model Metadata Extractor (MEC)",
}

# ── Merge all mappings ────────────────────────────────────────────────
NODE_CLASS_MAPPINGS = {
    **_FOLDER_MAPPINGS,
    **_MEC_MAPPINGS,
    **_MA_MAPPINGS,
    **_PAINT_MAPPINGS,
    **_FACE_FIXER_MAPPINGS,
    **_MASKMATTE_MAPPINGS,
    **_NUKEMAX_MAPPINGS,
    **_PROPAINTER_STITCH_MAPPINGS,
    **_STABILIZER_MAPPINGS,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    **_FOLDER_DISPLAY,
    **_MEC_DISPLAY,
    **_MA_DISPLAY,
    **_PAINT_DISPLAY,
    **_FACE_FIXER_DISPLAY,
    **_MASKMATTE_DISPLAY,
    **_NUKEMAX_DISPLAY,
    **_PROPAINTER_STITCH_DISPLAY,
    **_STABILIZER_DISPLAY,
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
    print("[MEC] NukeNodeMax routes + hooks registered.")
except Exception as _e:
    print(f"[MEC] NukeNodeMax server hooks deferred: {_e}")

print(f"[MEC] Loaded {len(_MEC_MAPPINGS)} MaskEditControl nodes.")
