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
from .nodes.sam_model_loader import SAMModelLoaderMEC
from .nodes.sam_mask_generator import SAMMaskGeneratorMEC
from .nodes.bbox_nodes import BBoxSmooth
from .nodes.vitmatte_refiner import ViTMatteRefinerMEC
from .nodes.sam_vitmatte_pipeline import SAMViTMattePipelineMEC
from .nodes.trimap_generator import TrimapGeneratorMEC
from .nodes.parameter_memory import ParameterHistoryMEC
from .nodes.sec_matanyone_pipeline import SeCMatAnyonePipelineMEC
from .nodes.background_remover import BackgroundRemoverMEC
from .nodes.semantic_segment import SemanticSegmentMEC
from .nodes.luminance_keyer import LuminanceKeyerMEC
from .nodes.mask_failure_explainer import MaskFailureExplainerMEC
from .nodes.temporal_anchor import TemporalAnchorMEC
from .nodes.sam_multi_mask_picker import SamMultiMaskPickerMEC
from .nodes.inpaint_suite import InpaintCropProMEC, InpaintStitchProMEC, InpaintPasteBackMEC
from .nodes.image_comparer import ImageComparerMEC
from .nodes.spline_mask_editor import SplineMaskEditorMEC
from .nodes.motion_mask_tracker import MotionMaskTrackerMEC
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

# ── VFX nodes migrated to ComfyUI-NukeMaxNodes (Apr 2026) ─────────────
# (color_science, exr_io, render_pass, plate_tools, geometry_nodes,
#  metadata_nodes, exr_metadata_reader, universal_reroute)
# Model analysis stays here:
from .nodes.model_analysis import (
    NODE_CLASS_MAPPINGS as _MA_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as _MA_DISPLAY,
)

_MEC_MAPPINGS = {
    "MaskTransformXY": MaskTransformXY,
    "MaskDrawFrame": MaskDrawFrame,
    "MaskPropagateVideo": MaskPropagateVideo,
    "PointsMaskEditor": PointsMaskEditor,
    "SAMModelLoaderMEC": SAMModelLoaderMEC,
    "SAMMaskGeneratorMEC": SAMMaskGeneratorMEC,
    "BBoxSmooth": BBoxSmooth,
    "ViTMatteRefinerMEC": ViTMatteRefinerMEC,
    "SAMViTMattePipelineMEC": SAMViTMattePipelineMEC,
    "TrimapGeneratorMEC": TrimapGeneratorMEC,
    "ParameterHistoryMEC": ParameterHistoryMEC,
    "SeCMatAnyonePipelineMEC": SeCMatAnyonePipelineMEC,
    "BackgroundRemoverMEC": BackgroundRemoverMEC,
    "SemanticSegmentMEC": SemanticSegmentMEC,
    "LuminanceKeyerMEC": LuminanceKeyerMEC,
    "MaskFailureExplainerMEC": MaskFailureExplainerMEC,
    "TemporalAnchorMEC": TemporalAnchorMEC,
    "SamMultiMaskPickerMEC": SamMultiMaskPickerMEC,
    "InpaintCropProMEC": InpaintCropProMEC,
    "InpaintStitchProMEC": InpaintStitchProMEC,
    "InpaintPasteBackMEC": InpaintPasteBackMEC,
    "ImageComparerMEC": ImageComparerMEC,
    "SplineMaskEditorMEC": SplineMaskEditorMEC,
    "MotionMaskTrackerMEC": MotionMaskTrackerMEC,
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
    "SAMModelLoaderMEC": "SAM Model Loader (MEC)",
    "SAMMaskGeneratorMEC": "SAM Mask Generator (MEC)",
    "BBoxSmooth": "BBox Smooth Temporal (MEC)",
    "ViTMatteRefinerMEC": "ViTMatte Edge Refiner (MEC)",
    "SAMViTMattePipelineMEC": "SAM + ViTMatte Pipeline (MEC)",
    "TrimapGeneratorMEC": "Trimap Generator (MEC)",
    "ParameterHistoryMEC": "Parameter History (MEC)",
    "SeCMatAnyonePipelineMEC": "SeC + MatAnyone2 Pipeline (MEC)",
    "BackgroundRemoverMEC": "Background Remover (MEC)",
    "SemanticSegmentMEC": "Semantic Segment (MEC)",
    "LuminanceKeyerMEC": "Luminance Keyer (MEC)",
    "MaskFailureExplainerMEC": "Mask Failure Explainer (MEC)",
    "TemporalAnchorMEC": "Temporal Anchor System (MEC)",
    "SamMultiMaskPickerMEC": "SAM Multi-Mask Picker (MEC)",
    "InpaintCropProMEC": "Inpaint Crop Pro (MEC)",
    "InpaintStitchProMEC": "Inpaint Stitch Pro (MEC)",
    "InpaintPasteBackMEC": "Inpaint Paste Back (MEC)",
    "ImageComparerMEC": "Image Comparer (MEC)",
    "SplineMaskEditorMEC": "Spline Mask Editor (MEC)",
    "MotionMaskTrackerMEC": "Motion Mask Tracker (MEC)",
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
}
NODE_DISPLAY_NAME_MAPPINGS = {
    **_FOLDER_DISPLAY,
    **_MEC_DISPLAY,
    **_MA_DISPLAY,
    **_PAINT_DISPLAY,
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

print(f"[MEC] Loaded {len(_MEC_MAPPINGS)} MaskEditControl nodes.")
