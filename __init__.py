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

# ── Live-preview guard: force sampling previews on even if launched with
#    --preview-method none (Latent2RGB fallback, cannot fail). Best-effort. ──
try:
    from .nodes import _c2c_preview_guard as _c2c_preview_guard  # noqa: F401
except Exception:  # never block the pack on a preview tweak
    pass

# ── Memory guard: release pack-owned GPU caches when ComfyUI frees VRAM ──
try:
    from .nodes._c2c_memguard import install as _c2c_memguard_install
    _c2c_memguard_install()
except Exception:  # never block the pack on a memory hook
    pass

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
    InpaintMaskPrepareMEC,
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
# Face/Pose Delta Editor (anchor-relative landmark deltas, multi-keyframe)
try:
    from .nodes.face_pose_delta import (
        NODE_CLASS_MAPPINGS as _FPDELTA_MAPPINGS,
        NODE_DISPLAY_NAME_MAPPINGS as _FPDELTA_DISPLAY,
    )
except Exception as _fpd_exc:  # pragma: no cover
    _FPDELTA_MAPPINGS, _FPDELTA_DISPLAY = {}, {}
    try:
        from .nodes._c2c_registry import record_failure as _c2c_rec_fail_fpd
        _c2c_rec_fail_fpd(
            "face_pose_delta", _fpd_exc,
            hint="Face/Pose Delta Editor failed to import. "
                 "Ensure NumPy is installed and temporal_anchor.py is intact.",
            group="nodes",
        )
    except Exception:
        import logging as _lg
        _lg.getLogger("MEC").warning(
            "[MEC] face_pose_delta import failed: %s", _fpd_exc,
        )
# Image Mask Editor (C2C) — persistent native-resolution mask editor
try:
    from .nodes.image_mask_editor import (
        NODE_CLASS_MAPPINGS as _IMEMASK_MAPPINGS,
        NODE_DISPLAY_NAME_MAPPINGS as _IMEMASK_DISPLAY,
        register_routes as _register_imemask_routes,
    )
except Exception as _ime_exc:  # pragma: no cover
    _IMEMASK_MAPPINGS, _IMEMASK_DISPLAY = {}, {}
    _register_imemask_routes = None
    try:
        from .nodes._c2c_registry import record_failure as _c2c_rec_fail_ime
        _c2c_rec_fail_ime(
            "image_mask_editor", _ime_exc,
            hint="Image Mask Editor failed to import. Check nodes/image_mask_editor/.",
            group="nodes",
        )
    except Exception:
        import logging as _lg
        _lg.getLogger("MEC").warning(
            "[MEC] image_mask_editor import failed: %s", _ime_exc,
        )
# Mask + Matting (multi-backend: SAM2.1, SAM3 + ViTMatte, RVM, ...)
# Guarded: a failure inside mask_matting (e.g. a missing _reanchor.py helper
# on an out-of-sync install) must NOT abort this __init__ — that would drop
# every other node in the pack (NanoBanana, WanDirector, OmniPill, Wizard…).
try:
    from .nodes.mask_matting import (
        NODE_CLASS_MAPPINGS as _MASKMATTE_MAPPINGS,
        NODE_DISPLAY_NAME_MAPPINGS as _MASKMATTE_DISPLAY,
    )
except Exception as _mm_exc:  # pragma: no cover
    _MASKMATTE_MAPPINGS, _MASKMATTE_DISPLAY = {}, {}
    try:
        from .nodes._c2c_registry import record_failure as _c2c_rec_fail_mm
        _c2c_rec_fail_mm(
            "mask_matting", _mm_exc,
            hint="Mask+Matting pack failed to import. Ensure "
                 "nodes/mask_matting/_reanchor.py is present and "
                 "opencv-python / transformers are installed.",
            group="nodes",
        )
    except Exception:
        import logging as _lg
        _lg.getLogger("MEC").warning(
            "[MEC] mask_matting import failed: %s", _mm_exc,
        )
# Layer Effects (Photoshop-style — torch-native, batch-correct)
try:
    from .nodes.layer_effects import (
        NODE_CLASS_MAPPINGS as _LAYERFX_MAPPINGS,
        NODE_DISPLAY_NAME_MAPPINGS as _LAYERFX_DISPLAY,
    )
except Exception as _lfx_exc:  # pragma: no cover
    _LAYERFX_MAPPINGS, _LAYERFX_DISPLAY = {}, {}
    try:
        from .nodes._c2c_registry import record_failure as _c2c_rec_fail_lfx
        _c2c_rec_fail_lfx(
            "layer_effects", _lfx_exc,
            hint="Layer Effects failed to import. Check nodes/layer_effects/_blend.py and _ops.py.",
            group="nodes",
        )
    except Exception:
        import logging as _lg
        _lg.getLogger("MEC").warning(
            "[MEC] layer_effects import failed: %s", _lfx_exc,
        )
# ── Central failure registry (surface silent drops to the user) ───────
# Per ideas_summary.md §2.1: the #1 reason this pack "feels like stubs"
# is that optional sub-imports were swallowed by `except: pass` / quiet
# logger.debug() calls. From here on every optional-import block routes
# its failure through _c2c_registry.record_failure() so the user sees
# (a) a console line in the ComfyUI log, (b) a boot toast in the UI, and
# (c) a queryable /c2c/registry/status endpoint listing every miss with
# an actionable hint.
try:
    from .nodes._c2c_registry import (
        record_failure as _c2c_rec_fail,
        register_routes as _c2c_reg_register_routes,
    )
except Exception as _reg_exc:  # pragma: no cover
    import logging
    logging.getLogger("C2C").error(
        "c2c registry helper missing — falling back to plain logging: %s", _reg_exc
    )
    def _c2c_rec_fail(key, exc, *, hint=None, group="root", severity="warning"):  # type: ignore
        import logging
        logging.getLogger("C2C").warning("%s unavailable: %s (hint=%s)", key, exc, hint)
    def _c2c_reg_register_routes(_server):  # type: ignore
        return None

# Unified Segmentation — DEPRECATED (superseded by mask_matting + sam_model_loader
# + sam_mask_generator). The file is kept on disk for reference only.
_USEG_MAPPINGS, _USEG_DISPLAY = {}, {}

# SAM Model Loader + Mask Generator (standalone SAM2.1/SAM3 inference nodes)
try:
    from .nodes.sam_model_loader import SAMModelLoaderMEC
    from .nodes.sam_mask_generator import SAMMaskGeneratorMEC
    _SAM_MAPPINGS = {
        "SAMModelLoaderMEC": SAMModelLoaderMEC,
        "SAMMaskGeneratorMEC": SAMMaskGeneratorMEC,
    }
    _SAM_DISPLAY = {
        "SAMModelLoaderMEC": "SAM Model Loader \u2014 SAM2.1 / SAM3",
        "SAMMaskGeneratorMEC": "SAM Mask Generator \u2014 Points + BBox + Text",
    }
except Exception as _exc:  # pragma: no cover
    _c2c_rec_fail(
        "SAM Loader/Generator", _exc,
        hint="SAM nodes require the sam2 package and model weights under models/sam2/.",
        group="nodes",
    )
    _SAM_MAPPINGS, _SAM_DISPLAY = {}, {}

# Mask Placement — prompt/ref -> alpha -> quad placement -> propagate
# (Slice 1: static; Cutie tracking + landmark-lock staged. Design doc:
# MASK_PLACEMENT_NODE_DESIGN.md at the workspace root.)
try:
    from .nodes.mask_placement import MaskPlacementMEC
    _MASKPLACE_MAPPINGS = {"MaskPlacementMEC": MaskPlacementMEC}
    _MASKPLACE_DISPLAY = {
        "MaskPlacementMEC": "Mask Placement — Prompt/Ref → Place → Track",
    }
except Exception as _exc:  # pragma: no cover
    _c2c_rec_fail(
        "MaskPlacement", _exc,
        hint="Requires opencv-python; prompt segmentation additionally needs the SAM stack.",
        group="nodes",
    )
    _MASKPLACE_MAPPINGS, _MASKPLACE_DISPLAY = {}, {}

# SAM + ViTMatte combined pipeline (highest-quality mask in one node)
try:
    from .nodes.sam_vitmatte_pipeline import SAMViTMattePipelineMEC
    _SAMVIT_MAPPINGS = {"SAMViTMattePipelineMEC": SAMViTMattePipelineMEC}
    _SAMVIT_DISPLAY = {
        "SAMViTMattePipelineMEC": "SAM + ViTMatte Pipeline \u2014 Full Quality",
    }
except Exception as _exc:  # pragma: no cover
    _c2c_rec_fail(
        "SAMViTMattePipeline", _exc,
        hint="Requires sam2 package + transformers for ViTMatte refinement.",
        group="nodes",
    )
    _SAMVIT_MAPPINGS, _SAMVIT_DISPLAY = {}, {}

# Mask toolkit (LayerMask port — torch-native, batch-correct)
try:
    from .nodes.mask_toolkit import (
        NODE_CLASS_MAPPINGS as _MASKTOOLKIT_MAPPINGS,
        NODE_DISPLAY_NAME_MAPPINGS as _MASKTOOLKIT_DISPLAY,
    )
except Exception as _mtk_exc:  # pragma: no cover
    _MASKTOOLKIT_MAPPINGS, _MASKTOOLKIT_DISPLAY = {}, {}
    try:
        from .nodes._c2c_registry import record_failure as _c2c_rec_fail_mtk
        _c2c_rec_fail_mtk(
            "mask_toolkit", _mtk_exc,
            hint="Mask toolkit failed to import. Check nodes/mask_toolkit/_ops.py and nodes.py.",
            group="nodes",
        )
    except Exception:
        import logging as _lg
        _lg.getLogger("MEC").warning(
            "[MEC] mask_toolkit import failed: %s", _mtk_exc,
        )
# Frequency / Grain (NKD Basic Tools port — frequency separation + film grain)
try:
    from .nodes.frequency_grain import (
        NODE_CLASS_MAPPINGS as _FREQGRAIN_MAPPINGS,
        NODE_DISPLAY_NAME_MAPPINGS as _FREQGRAIN_DISPLAY,
    )
except Exception as _fg_exc:  # pragma: no cover
    _FREQGRAIN_MAPPINGS, _FREQGRAIN_DISPLAY = {}, {}
    try:
        from .nodes._c2c_registry import record_failure as _c2c_rec_fail_fg
        _c2c_rec_fail_fg(
            "frequency_grain", _fg_exc,
            hint="Frequency / Grain failed to import. Check nodes/frequency_grain/_ops.py and nodes.py.",
            group="nodes",
        )
    except Exception:
        import logging as _lg
        _lg.getLogger("MEC").warning(
            "[MEC] frequency_grain import failed: %s", _fg_exc,
        )
# AV Handles (ComfyUI-AV-Handles port, MIT — Add + Trim combined into one)
try:
    from .nodes.av_handles import (
        NODE_CLASS_MAPPINGS as _AVH_MAPPINGS,
        NODE_DISPLAY_NAME_MAPPINGS as _AVH_DISPLAY,
    )
except Exception as _avh_exc:  # pragma: no cover
    _AVH_MAPPINGS, _AVH_DISPLAY = {}, {}
    try:
        from .nodes._c2c_registry import record_failure as _c2c_rec_fail_avh
        _c2c_rec_fail_avh(
            "av_handles", _avh_exc,
            hint="AV Handles failed to import. Check nodes/av_handles.py.",
            group="nodes",
        )
    except Exception:
        import logging as _lg
        _lg.getLogger("MEC").warning("[MEC] av_handles import failed: %s", _avh_exc)
# Save Video (C2C) — streaming pro-format writer
try:
    from .nodes.save_video import (
        NODE_CLASS_MAPPINGS as _SAVEVIDEO_MAPPINGS,
        NODE_DISPLAY_NAME_MAPPINGS as _SAVEVIDEO_DISPLAY,
    )
except Exception as _sv_exc:  # pragma: no cover
    _SAVEVIDEO_MAPPINGS, _SAVEVIDEO_DISPLAY = {}, {}
    try:
        from .nodes._c2c_registry import record_failure as _c2c_rec_fail_sv
        _c2c_rec_fail_sv(
            "save_video", _sv_exc,
            hint="Save Video (C2C) failed to import. Check nodes/save_video.py.",
            group="nodes",
        )
    except Exception:
        import logging as _lg
        _lg.getLogger("MEC").warning("[MEC] save_video import failed: %s", _sv_exc)
# Tiled video refinement (tile plan / split / merge / identity lock)
try:
    from .nodes.tiling import (
        NODE_CLASS_MAPPINGS as _TILING_MAPPINGS,
        NODE_DISPLAY_NAME_MAPPINGS as _TILING_DISPLAY,
    )
except Exception as _tile_exc:  # pragma: no cover
    _TILING_MAPPINGS, _TILING_DISPLAY = {}, {}
    try:
        from .nodes._c2c_registry import record_failure as _c2c_rec_fail_tile
        _c2c_rec_fail_tile(
            "tiling", _tile_exc,
            hint="Tiled refinement failed to import. Check nodes/tiling/.",
            group="nodes",
        )
    except Exception:
        import logging as _lg
        _lg.getLogger("MEC").warning(
            "[MEC] tiling import failed: %s", _tile_exc,
        )
# Magnific (vendor pack ported in — hosted generation / upscale / stock)
#
# ComfyUI-OmniScale (the owner's API-key pack, D0.14) registers the same 15 Magnific* node ids and the same
# /magnific/* routes. Node ids go to the pack loaded LAST, routes to the one registered FIRST, so with both
# installed OmniScale's nodes talked to this copy's sign-in routes and asked for a key they could never receive
# (A9, L2.28). When OmniScale is installed and enabled, this copy stands down; C2C_MAGNIFIC_KEEP_CNP=1 keeps it.
def _c2c_omniscale_installed() -> bool:
    import os as _os
    try:
        parent = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
        for _name in _os.listdir(parent):
            low = _name.lower()
            if low.startswith("comfyui-omniscale") and not low.endswith(".disabled") \
                    and _os.path.isfile(_os.path.join(parent, _name, "__init__.py")):
                return True
    except OSError:
        pass
    return False


try:
    import os as _c2c_os
    if _c2c_omniscale_installed() and _c2c_os.environ.get("C2C_MAGNIFIC_KEEP_CNP") != "1":
        _MAGNIFIC_MAPPINGS, _MAGNIFIC_DISPLAY = {}, {}
        import logging as _lg
        _lg.getLogger("MEC").info(
            "[C2C] ComfyUI-OmniScale is installed: its API-key nodes provide Magnific*; CustomNodePacks' "
            "sign-in copy is not registered (set C2C_MAGNIFIC_KEEP_CNP=1 to keep it).")
    else:
        from .nodes.magnific import (
            NODE_CLASS_MAPPINGS as _MAGNIFIC_MAPPINGS,
            NODE_DISPLAY_NAME_MAPPINGS as _MAGNIFIC_DISPLAY,
        )
except Exception as _mag_exc:  # pragma: no cover
    _MAGNIFIC_MAPPINGS, _MAGNIFIC_DISPLAY = {}, {}
    try:
        from .nodes._c2c_registry import record_failure as _c2c_rec_fail_mag
        _c2c_rec_fail_mag(
            "magnific", _mag_exc,
            hint="Magnific failed to import. Check nodes/magnific/.",
            group="nodes",
        )
    except Exception:
        import logging as _lg
        _lg.getLogger("MEC").warning(
            "[MEC] magnific import failed: %s", _mag_exc,
        )
# VAE Clean (colour cast / oversaturation / decode artefacts after a decode)
try:
    from .nodes.vae_clean import (
        NODE_CLASS_MAPPINGS as _VAECLEAN_MAPPINGS,
        NODE_DISPLAY_NAME_MAPPINGS as _VAECLEAN_DISPLAY,
    )
except Exception as _vc_exc:  # pragma: no cover
    _VAECLEAN_MAPPINGS, _VAECLEAN_DISPLAY = {}, {}
    try:
        from .nodes._c2c_registry import record_failure as _c2c_rec_fail_vc
        _c2c_rec_fail_vc(
            "vae_clean", _vc_exc,
            hint="VAE Clean failed to import. Check nodes/vae_clean.py.",
            group="nodes",
        )
    except Exception:
        import logging as _lg
        _lg.getLogger("MEC").warning(
            "[MEC] vae_clean import failed: %s", _vc_exc,
        )
# Smart Image Crop / Stitch (Smart-Image-Crop-and-Stitch port — stills path)
try:
    from .nodes.smart_crop import (
        NODE_CLASS_MAPPINGS as _SMARTCROP_MAPPINGS,
        NODE_DISPLAY_NAME_MAPPINGS as _SMARTCROP_DISPLAY,
    )
except Exception as _sc_exc:  # pragma: no cover
    _SMARTCROP_MAPPINGS, _SMARTCROP_DISPLAY = {}, {}
    try:
        from .nodes._c2c_registry import record_failure as _c2c_rec_fail_sc
        _c2c_rec_fail_sc(
            "smart_crop", _sc_exc,
            hint="Smart Image Crop / Stitch failed to import. Check nodes/smart_crop.py.",
            group="nodes",
        )
    except Exception:
        import logging as _lg
        _lg.getLogger("MEC").warning(
            "[MEC] smart_crop import failed: %s", _sc_exc,
        )
# Luminance Keyer (Nuke-style luma key, pure tensor math)
try:
    from .nodes.luminance_keyer import LuminanceKeyerMEC
    _LUMAKEY_MAPPINGS = {"LuminanceKeyerMEC": LuminanceKeyerMEC}
    _LUMAKEY_DISPLAY = {
        "LuminanceKeyerMEC": "Luminance Keyer \u2014 Highlights / Shadows / Custom",
    }
except Exception as _exc:  # pragma: no cover
    _c2c_rec_fail(
        "LuminanceKeyer", _exc,
        hint="Luminance Keyer failed to import — check _interrupt_check / _progress helpers.",
        group="nodes",
    )
    _LUMAKEY_MAPPINGS, _LUMAKEY_DISPLAY = {}, {}

# Background Remover (RMBG-2.0 / BiRefNet one-click bg removal)
try:
    from .nodes.background_remover import BackgroundRemoverMEC
    _BGREMOVE_MAPPINGS = {"BackgroundRemoverMEC": BackgroundRemoverMEC}
    _BGREMOVE_DISPLAY = {
        "BackgroundRemoverMEC": "Background Remover \u2014 RMBG / BiRefNet",
    }
except Exception as _exc:  # pragma: no cover
    _c2c_rec_fail(
        "BackgroundRemover", _exc,
        hint="Requires transformers + model weights for RMBG or BiRefNet.",
        group="nodes",
    )
    _BGREMOVE_MAPPINGS, _BGREMOVE_DISPLAY = {}, {}

# Semantic Segment (SegFormer face / clothes parsing)
try:
    from .nodes.semantic_segment import SemanticSegmentMEC
    _SEMSEG_MAPPINGS = {"SemanticSegmentMEC": SemanticSegmentMEC}
    _SEMSEG_DISPLAY = {
        "SemanticSegmentMEC": "Semantic Segment \u2014 Face / Clothes Parsing",
    }
except Exception as _exc:  # pragma: no cover
    _c2c_rec_fail(
        "SemanticSegment", _exc,
        hint="Requires transformers + SegFormer model weights.",
        group="nodes",
    )
    _SEMSEG_MAPPINGS, _SEMSEG_DISPLAY = {}, {}

# SAM Multi-Mask Picker — interactive 3-thumbnail mask chooser
# (user-recalled feature: pick best of N SAM candidates by score)
try:
    from .nodes.sam_multi_mask_picker import SamMultiMaskPickerMEC
    _SAMPICKER_MAPPINGS = {"SamMultiMaskPickerMEC": SamMultiMaskPickerMEC}
    _SAMPICKER_DISPLAY = {
        "SamMultiMaskPickerMEC": "SAM Multi-Mask Picker \u2014 3 candidates + scores",
    }
except Exception as _exc:  # pragma: no cover
    _c2c_rec_fail(
        "SamMultiMaskPickerMEC", _exc,
        hint="Install SAM (sam2 or segment-anything) so candidate masks can be scored.",
        group="nodes",
    )
    _SAMPICKER_MAPPINGS, _SAMPICKER_DISPLAY = {}, {}

# Wan Director is owned by ComfyUI-WanNodeExperiments, which holds the whole
# director family (WanDirectorInspector, WanDirectorExtraArgs). The copy that
# lived here duplicated the WanDirectorC2C id and was resolved by unsorted
# os.listdir order; it is deleted (2026-08-29 dedup). Removed rather than left
# guarded, so the loader does not record a spurious import failure.
_WANDIR_MAPPINGS, _WANDIR_DISPLAY = {}, {}

# ── C2C Vault: password-locked subgraph ──────────────────────────────
# Guarded independently: the node stays registered even without
# `cryptography`, so a workflow containing a vault still LOADS and reports a
# clear install message instead of the node vanishing from the graph.
try:
    from .nodes.vault_node import (
        NODE_CLASS_MAPPINGS as _VAULT_MAPPINGS,
        NODE_DISPLAY_NAME_MAPPINGS as _VAULT_DISPLAY,
        register_routes as _register_vault_routes,
    )
except Exception as _exc:
    _c2c_rec_fail(
        "C2CVault", _exc,
        hint="Install `cryptography` for the password-locked subgraph node.",
        group="nodes",
    )
    _VAULT_MAPPINGS, _VAULT_DISPLAY, _register_vault_routes = {}, {}, None

# C2C helpers (12 tiny utilities)
try:
    from .nodes.helpers import (
        NODE_CLASS_MAPPINGS as _HELPERS_MAPPINGS,
        NODE_DISPLAY_NAME_MAPPINGS as _HELPERS_DISPLAY,
    )
except Exception as _exc:  # pragma: no cover
    _c2c_rec_fail(
        "C2C helpers", _exc,
        hint="Helper utilities should always load — report this traceback as a bug.",
        group="nodes", severity="error",
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
    _c2c_rec_fail(
        "PromptRelay", _exc,
        hint="Prompt Relay requires Wan 2.x or compatible video model nodes for full backends.",
        group="nodes",
    )
    _PROMPTRELAY_MAPPINGS, _PROMPTRELAY_DISPLAY = {}, {}

# AsymFlow sampler patch — Apache-2.0; algorithm by Lakonik/LakonLab.
try:
    from .nodes.asymflow_sampler import (
        NODE_CLASS_MAPPINGS as _ASYMFLOW_MAPPINGS,
        NODE_DISPLAY_NAME_MAPPINGS as _ASYMFLOW_DISPLAY,
    )
except Exception as _exc:  # pragma: no cover
    _c2c_rec_fail(
        "AsymFlow sampler", _exc,
        hint="AsymFlow patches ComfyUI's sampler module; a Comfy upgrade may have broken the patch surface.",
        group="nodes",
    )
    _ASYMFLOW_MAPPINGS, _ASYMFLOW_DISPLAY = {}, {}

# HDR Color Science nodes (ACES tonemap, VAE quality decode, color space)
try:
    from .nodes.hdr_color_science import (
        NODE_CLASS_MAPPINGS as _HDR_MAPPINGS,
        NODE_DISPLAY_NAME_MAPPINGS as _HDR_DISPLAY,
    )
except Exception as _exc:  # pragma: no cover
    _c2c_rec_fail(
        "HDR Color Science", _exc,
        hint="HDR color science nodes for ACES tonemap, VAE quality decode, and color space conversion.",
        group="nodes",
    )
    _HDR_MAPPINGS, _HDR_DISPLAY = {}, {}

# LocateAnything-3B grounding (open-vocabulary object detection → SAM prompts)
try:
    from .nodes.locate_anything import (
        NODE_CLASS_MAPPINGS as _LOCATE_MAPPINGS,
        NODE_DISPLAY_NAME_MAPPINGS as _LOCATE_DISPLAY,
    )
except Exception as _exc:  # pragma: no cover
    _c2c_rec_fail(
        "LocateAnything", _exc,
        hint="Requires transformers + nvidia/LocateAnything-3B weights for open-vocabulary grounding.",
        group="nodes",
    )
    _LOCATE_MAPPINGS, _LOCATE_DISPLAY = {}, {}

# Nano Banana (Google Gemini image API: gemini-3-pro-image + 2.5-flash-image)
try:
    from .nodes.nano_banana import (
        NODE_CLASS_MAPPINGS as _NANOBANANA_MAPPINGS,
        NODE_DISPLAY_NAME_MAPPINGS as _NANOBANANA_DISPLAY,
    )
except Exception as _exc:  # pragma: no cover
    _c2c_rec_fail(
        "NanoBanana", _exc,
        hint="Gemini image-generation node; needs only stdlib + PIL + a GEMINI_API_KEY.",
        group="nodes",
    )
    _NANOBANANA_MAPPINGS, _NANOBANANA_DISPLAY = {}, {}

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
    "ProPainterMEC": "ProPainter \u2014 Temporal / Remove / Stitch / Refine / Flow",
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
    "InpaintStitchProMEC": InpaintStitchProMEC,
    "InpaintPasteBackMEC": InpaintPasteBackMEC,
    "InpaintMaskPrepareMEC": InpaintMaskPrepareMEC,
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
    "MaskEditMEC": "Mask Edit \u2014 Transform/Draw/Points/BBox",
    "SplineMaskMEC": "Spline Mask \u2014 Edit/Track/Flow-Path",
    "MaskTrackerMEC": "Mask Tracker \u2014 Motion/Propagate/Anchor/Consistency",
    "ParameterHistoryMEC": "Parameter History",
    "SeCMatAnyonePipelineMEC": "SeC + MatAnyone2 Pipeline",
    "InpaintCropProMEC": "Inpaint Crop Pro",
    "InpaintStitchProMEC": "Inpaint Stitch Pro",
    "InpaintPasteBackMEC": "Inpaint Paste Back",
    "InpaintMaskPrepareMEC": "Inpaint Mask Prepare",
    "VideoComparerC2C": "Video Comparer \u2014 Player + Wipe/Diff/Scopes",
    "VideoFramePlayerMEC": "Video Frame Player",
    "VideoMaskEditorMEC": "Video Mask Editor",
    "VAEMergeMEC": "VAE Merge",
    "VAELatentInspectorMEC": "VAE Latent Inspector",
    "BatchVersionManagerMEC": "Batch Version Manager",
    "ModelMetadataExtractorMEC": "Model Metadata Extractor",
    "MaskFailureExplainerMEC": "Mask Failure Explainer \u2014 Diagnostics",
}

# ── Merge all mappings ────────────────────────────────────────────────
try:
    from .nodes.control_forge import (
        NODE_CLASS_MAPPINGS as _CONTROLFORGE_MAPPINGS,
        NODE_DISPLAY_NAME_MAPPINGS as _CONTROLFORGE_DISPLAY,
    )
except Exception as _cf_exc:  # pragma: no cover
    try:
        _c2c_rec_fail("Control AOV", _cf_exc,
                      hint="control_forge needs numpy + torch (OpenCV optional for Canny/motion).",
                      group="nodes")
    except Exception:
        pass
    _CONTROLFORGE_MAPPINGS, _CONTROLFORGE_DISPLAY = {}, {}

# Clipboard TCL — Nuke-style copy/paste of node graphs (F5). Restored 2026-06-20
# (was removed in a 2026-05 "deprecated cleanup"; it has no replacement).
try:
    from .nodes.clipboard_tcl import (
        NODE_CLASS_MAPPINGS as _TCL_MAPPINGS,
        NODE_DISPLAY_NAME_MAPPINGS as _TCL_DISPLAY,
        register_routes as _register_tcl_routes,
    )
except Exception as _exc:  # pragma: no cover
    _c2c_rec_fail(
        "ClipboardTCL", _exc,
        hint="Nuke-style TCL copy/paste of node graphs (F5); stdlib only.",
        group="nodes",
    )
    _TCL_MAPPINGS, _TCL_DISPLAY = {}, {}
    def _register_tcl_routes(_server):  # type: ignore
        return None


# ── Restored VFX / utility nodes (removed in a 2026-05 "deprecated cleanup";
#    verified they have NO replacement in the current pack, so brought back).
#    Each module is guarded independently so one failure never drops the rest.
_RESTORED_MAPPINGS, _RESTORED_DISPLAY = {}, {}
# NOTE (2026-08-29 dedup): color_science, exr_metadata_reader, geometry_nodes,
# metadata_nodes, plate_tools and render_pass were REMOVED from this list and
# deleted. They had been migrated to ComfyUI-NukeMaxNodes in Apr 2026, then
# re-registered here by a 2026-05 "restore" whose check ("no replacement in the
# current pack") only looked inside THIS pack - the replacement was in another
# one. That restore is what created 18 duplicate node ids, resolved by unsorted
# os.listdir order (ComfyUI/nodes.py:2295 last-write-wins, :2356 unsorted).
# exr_io STAYS here: its OpenImageIO backend is the only EXR path that works in
# this environment, and its SaveEXRMEC is the 6-input superset.
for _rmod in ("exr_io", "video_frame_extractor",
              "optical_flow", "roto", "shuffle", "pixel_aspect"):
    try:
        _rm = __import__(f"{__name__}.nodes.{_rmod}", fromlist=["NODE_CLASS_MAPPINGS"])
        _RESTORED_MAPPINGS.update(getattr(_rm, "NODE_CLASS_MAPPINGS", {}) or {})
        _RESTORED_DISPLAY.update(getattr(_rm, "NODE_DISPLAY_NAME_MAPPINGS", {}) or {})
    except Exception as _rexc:  # pragma: no cover
        _c2c_rec_fail(
            f"restored:{_rmod}", _rexc,
            hint="VFX/utility node restored from git history; may need cv2/OpenEXR.",
            group="nodes",
        )


# ── Fluid Shots & Audio FX (temporal normalizer pair + audio reverser) ──
_FLUID_MAPPINGS, _FLUID_DISPLAY = {}, {}
try:
    from .nodes import fluid_shots_audio as _fluid
    _FLUID_MAPPINGS.update(_fluid.NODE_CLASS_MAPPINGS)
    _FLUID_DISPLAY.update(_fluid.NODE_DISPLAY_NAME_MAPPINGS)
except Exception as _fexc:  # pragma: no cover
    _c2c_rec_fail(
        "fluid_shots_audio", _fexc,
        hint="Fluid Shot Encoder/Decoder + Audio Reverser; needs cv2 for optical flow.",
        group="nodes",
    )

# ── C2C Video loaders (S1a — lazy-handle decode nodes) ──
_C2CVIDEO_MAPPINGS, _C2CVIDEO_DISPLAY = {}, {}
try:
    from .nodes.c2c_video import nodes_load as _c2c_video_nodes
    _C2CVIDEO_MAPPINGS.update(_c2c_video_nodes.NODE_CLASS_MAPPINGS)
    _C2CVIDEO_DISPLAY.update(_c2c_video_nodes.NODE_DISPLAY_NAME_MAPPINGS)
except Exception as _cv_exc:  # pragma: no cover
    _c2c_rec_fail(
        "c2c_video_nodes", _cv_exc,
        hint="C2C Video loaders need PyAV, OpenImageIO, torch, and psutil.",
        group="nodes",
    )

# ── C2C Farm render-farm spooler (Tractor-style multi-cloud dispatch) ──
_FARM_MAPPINGS, _FARM_DISPLAY = {}, {}
try:
    from .nodes import render_farm as _farm
    _FARM_MAPPINGS.update(_farm.NODE_CLASS_MAPPINGS)
    _FARM_DISPLAY.update(_farm.NODE_DISPLAY_NAME_MAPPINGS)
except Exception as _farm_exc:  # pragma: no cover
    _c2c_rec_fail(
        "render_farm", _farm_exc,
        hint="C2C Farm spooler (C2C_Submit / ClusterStatus / JobHistory); "
             "check renderfarm/config/*.json is intact.",
        group="nodes",
    )

# ── Legacy nodes (deprecated — kept so old workflows load) ────────────
try:
    from .nodes.legacy_compat import (
        NODE_CLASS_MAPPINGS as _LEGACY_MAPPINGS,
        NODE_DISPLAY_NAME_MAPPINGS as _LEGACY_DISPLAY,
    )
except Exception as _leg_exc:  # pragma: no cover
    _LEGACY_MAPPINGS, _LEGACY_DISPLAY = {}, {}
    _c2c_rec_fail(
        "legacy_compat", _leg_exc,
        hint="Legacy workflow-compat nodes failed to import. Check nodes/legacy_compat.py.",
        group="nodes",
    )

_C2C_FAMILIES = [
    ("Vault", _VAULT_MAPPINGS, _VAULT_DISPLAY),
    ("Folder Incrementer", _FOLDER_MAPPINGS, _FOLDER_DISPLAY),
    ("Render farm", _FARM_MAPPINGS, _FARM_DISPLAY),
    ("Fluid Shots/Audio", _FLUID_MAPPINGS, _FLUID_DISPLAY),
    ("C2C video", _C2CVIDEO_MAPPINGS, _C2CVIDEO_DISPLAY),
    ("MaskEditControl", _MEC_MAPPINGS, _MEC_DISPLAY),
    ("Image Mask Editor", _IMEMASK_MAPPINGS, _IMEMASK_DISPLAY),
    ("Model analysis", _MA_MAPPINGS, _MA_DISPLAY),
    ("MEC Paint Suite", _PAINT_MAPPINGS, _PAINT_DISPLAY),
    ("Face Fixer", _FACE_FIXER_MAPPINGS, _FACE_FIXER_DISPLAY),
    ("Face/Pose Delta", _FPDELTA_MAPPINGS, _FPDELTA_DISPLAY),
    ("Mask + Matting", _MASKMATTE_MAPPINGS, _MASKMATTE_DISPLAY),
    ("Layer Effects", _LAYERFX_MAPPINGS, _LAYERFX_DISPLAY),
    ("Mask toolkit", _MASKTOOLKIT_MAPPINGS, _MASKTOOLKIT_DISPLAY),
    ("Frequency / Grain", _FREQGRAIN_MAPPINGS, _FREQGRAIN_DISPLAY),
    ("Smart Crop/Stitch", _SMARTCROP_MAPPINGS, _SMARTCROP_DISPLAY),
    ("Tiled refinement", _TILING_MAPPINGS, _TILING_DISPLAY),
    ("VAE Clean", _VAECLEAN_MAPPINGS, _VAECLEAN_DISPLAY),
    ("Magnific", _MAGNIFIC_MAPPINGS, _MAGNIFIC_DISPLAY),
    ("AV Handles", _AVH_MAPPINGS, _AVH_DISPLAY),
    ("Save Video", _SAVEVIDEO_MAPPINGS, _SAVEVIDEO_DISPLAY),
    ("Unified Segmentation", _USEG_MAPPINGS, _USEG_DISPLAY),
    ("SAM Multi-Mask Picker", _SAMPICKER_MAPPINGS, _SAMPICKER_DISPLAY),
    ("SAM Loader/Generator", _SAM_MAPPINGS, _SAM_DISPLAY),
    ("Mask Placement", _MASKPLACE_MAPPINGS, _MASKPLACE_DISPLAY),
    ("SAM + ViTMatte", _SAMVIT_MAPPINGS, _SAMVIT_DISPLAY),
    ("Luminance Keyer", _LUMAKEY_MAPPINGS, _LUMAKEY_DISPLAY),
    ("Background Remover", _BGREMOVE_MAPPINGS, _BGREMOVE_DISPLAY),
    ("Semantic Segment", _SEMSEG_MAPPINGS, _SEMSEG_DISPLAY),
    ("NukeNodeMax", _NUKEMAX_MAPPINGS, _NUKEMAX_DISPLAY),
    ("Video Stabilizer", _STABILIZER_MAPPINGS, _STABILIZER_DISPLAY),
    ("WanDirector", _WANDIR_MAPPINGS, _WANDIR_DISPLAY),
    ("C2C helpers", _HELPERS_MAPPINGS, _HELPERS_DISPLAY),
    ("Prompt Relay", _PROMPTRELAY_MAPPINGS, _PROMPTRELAY_DISPLAY),
    ("AsymFlow sampler", _ASYMFLOW_MAPPINGS, _ASYMFLOW_DISPLAY),
    ("HDR Color Science", _HDR_MAPPINGS, _HDR_DISPLAY),
    ("LocateAnything", _LOCATE_MAPPINGS, _LOCATE_DISPLAY),
    ("Nano Banana", _NANOBANANA_MAPPINGS, _NANOBANANA_DISPLAY),
    ("Control AOV", _CONTROLFORGE_MAPPINGS, _CONTROLFORGE_DISPLAY),
    ("Clipboard TCL", _TCL_MAPPINGS, _TCL_DISPLAY),
    ("Restored VFX", _RESTORED_MAPPINGS, _RESTORED_DISPLAY),
    ("Legacy (deprecated)", _LEGACY_MAPPINGS, _LEGACY_DISPLAY),
]

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
for _label, _maps, _disp in _C2C_FAMILIES:
    NODE_CLASS_MAPPINGS.update(_maps)
    NODE_DISPLAY_NAME_MAPPINGS.update(_disp)

C2C_LOAD_SUMMARY = {
    "pack": "ComfyUI-CustomNodePacks",
    "total": len(NODE_CLASS_MAPPINGS),
    "families": [[_lbl, len(_maps)] for _lbl, _maps, _ in _C2C_FAMILIES if len(_maps) > 0],
    "failed": [],
}
try:
    from .nodes._c2c_registry import summary as _c2c_reg_summary

    C2C_LOAD_SUMMARY["failed"] = [
        {
            "key": _rec["key"],
            "group": _rec["group"],
            "error": f"{_rec['exception_type']}: {_rec['message']}",
        }
        for _rec in _c2c_reg_summary()["failures"]
        if _rec.get("group") in ("nodes", "root")
    ]
except Exception:
    pass

WEB_DIRECTORY = "./js"

# ── One menu root for every Code2Collapse pack ─────────────────────────────
# Every node lands under "🐺 C2C/<pack>/<family>" in the Add Node menu and the
# node library (see _c2c_menu.py). Node ids are untouched, so saved workflows
# are unaffected. Guarded: a menu placement must never cost the pack its nodes.
try:
    from ._c2c_menu import rebrand_v1 as _c2c_menu_rebrand

    _c2c_menu_rebrand(
        NODE_CLASS_MAPPINGS, "\U0001F9F0 Core",
        strip=("C2C", "MEC", "MaskEditControl", "MaskEnhancedControl", "Code2Collapse",
               "ComfyUI-CustomNodePacks"),
        rename={
            "ModelAnalysis": "Model Analysis", "VideoMask": "Video Mask",
            "PromptRelay": "Prompt Relay", "LayerEffects": "Layer Effects",
            "RenderFarm": "Render Farm", "Masking": "Mask", "MaskMatting": "Matting",
            "Channels": "Mask", "Edit": "Mask", "Grounding": "Segmentation",
            "utils": "Utils",
        },
    )
except Exception as _c2c_menu_exc:  # noqa: BLE001
    import logging as _c2c_menu_log

    _c2c_menu_log.getLogger(__name__).warning("C2C menu root not applied: %s", _c2c_menu_exc)


__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

# ── Register server routes for Parameter Memory ──────────────────────
try:
    import server as _comfy_server
    from .nodes.parameter_memory import register_routes as _register_pm_routes
    _register_pm_routes(_comfy_server.PromptServer.instance)
    print("[MEC] Parameter Memory server route registered.")
except Exception:
    pass  # Server not available (e.g. during import-only testing)

# ── Autobatch: opt-in frame chunking (config reloads each prompt) ────
try:
    from .nodes._autobatch import register_prompt_hook as _c2c_autobatch_register
    _c2c_autobatch_register()
except Exception as _ab_exc:  # pragma: no cover
    try:
        _c2c_rec_fail(
            "autobatch", _ab_exc,
            hint="Automatic frame batching hook failed to register. "
                 "Set C2C_AUTOBATCH=0 to disable chunking.",
            group="nodes",
        )
    except Exception:
        import logging as _ab_log
        _ab_log.getLogger("C2C").warning("[C2C] autobatch hook failed: %s", _ab_exc)

# ── Register C2C Vault routes ────────────────────────────────────────
# The password only ever travels over these routes. It is never a widget and
# never enters the queued prompt, so it cannot be serialised into the workflow
# JSON that the vault exists to protect.
try:
    if _register_vault_routes is not None:
        import server as _comfy_server_vault
        _register_vault_routes(_comfy_server_vault.PromptServer.instance)
except Exception:
    pass  # Server not available

# ── Register server routes for Video Mask Editor ─────────────────────
try:
    import server as _comfy_server_vme
    _register_vme_routes(_comfy_server_vme.PromptServer.instance)
except Exception:
    pass  # Server not available

# ── Register server routes for Image Mask Editor (C2C) ───────────────
try:
    if _register_imemask_routes is not None:
        import server as _comfy_server_ime
        _register_imemask_routes(_comfy_server_ime.PromptServer.instance)
except Exception:
    pass  # Server not available

# ── Register Clipboard TCL (Nuke-style copy/paste) routes ────────────
try:
    import server as _comfy_server_tcl
    _register_tcl_routes(_comfy_server_tcl.PromptServer.instance)
    print("[MEC] Clipboard TCL (Nuke-style copy/paste) routes registered.")
except Exception:
    pass  # Server not available

# ── Register NukeNodeMax server-side hooks & routes ───────────────────
try:
    import server as _comfy_server  # noqa: F811
    _ps = _comfy_server.PromptServer.instance
    # Surface the central failure registry FIRST so it's queryable even if
    # everything below fails to register.
    try:
        _c2c_reg_register_routes(_ps)
    except Exception as _re:
        print(f"[C2C] registry routes deferred: {_re}")
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
        from .nodes.ai_workflow_builder import register_routes as _register_wf_builder_routes
        _register_wf_builder_routes(_ps)
        print("[MEC] ai_workflow_builder routes registered.")
    except Exception as _wb:
        print(f"[MEC] ai_workflow_builder deferred: {_wb}")
    try:
        from .nodes.ai_diagnose import register_routes as _register_ai_diagnose_routes
        _register_ai_diagnose_routes(_ps)
        print("[MEC] ai_diagnose routes registered.")
    except Exception as _ad:
        print(f"[MEC] ai_diagnose deferred: {_ad}")
    try:
        from .nodes.error_introspector import register_routes as _register_error_introspector_routes
        _register_error_introspector_routes(_ps)
        print("[MEC] error_introspector routes registered.")
    except Exception as _ei:
        print(f"[MEC] error_introspector deferred: {_ei}")
    try:
        # Model Browser backend (/c2c/models/search|download|dest_dirs). The JS
        # panel (c2c_model_browser.js) has shipped for a while but this
        # registration was missing — every search 404'd ("Error: check console").
        from .nodes.model_browser_routes import register_routes as _register_model_browser_routes
        _register_model_browser_routes(_ps)
        print("[MEC] model_browser routes registered.")
    except Exception as _mb:
        print(f"[MEC] model_browser deferred: {_mb}")
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
        from .nodes.c2c_int_aggregator import register_routes as _register_c2c_int_routes
        _register_c2c_int_routes(_ps)
        print("[C2C] int aggregator routes registered (/c2c/int/*).")
    except Exception as _int:
        print(f"[C2C] int aggregator deferred: {_int}")
    try:
        from .nodes.c2c_doctor import register_routes as _register_c2c_doctor_routes
        _register_c2c_doctor_routes(_ps)
        print("[C2C] doctor routes registered (/c2c/doctor/{pyenv,disk,scan_file}).")
    except Exception as _doc:
        print(f"[C2C] doctor deferred: {_doc}")
    try:
        from .nodes.c2c_workflow_library import register_routes as _register_c2c_library_routes
        _register_c2c_library_routes(_ps)
        print("[C2C] workflow library routes registered (/c2c/library/{locations,scan,load}).")
    except Exception as _lib:
        print(f"[C2C] workflow library deferred: {_lib}")
    try:
        from .nodes._c2c_autoconnect import register_routes as _register_c2c_autoconnect_routes
        _register_c2c_autoconnect_routes(_ps)
        print("[C2C] autoconnect routes registered (/c2c/autoconnect/*).")
    except Exception as _ac:
        print(f"[C2C] autoconnect deferred: {_ac}")
    try:
        from .nodes.c2c_sys_metrics import register_routes as _register_c2c_sys_metrics_routes
        _register_c2c_sys_metrics_routes(_ps)
        print("[C2C] sys metrics routes registered (/c2c/sys/metrics).")
    except Exception as _sm:
        print(f"[C2C] sys metrics deferred: {_sm}")
    try:
        from .nodes._c2c_vram_headroom import register_routes as _register_vram_headroom_routes
        _register_vram_headroom_routes()
        print("[C2C] browser VRAM headroom routes registered (/c2c/memory/browser_headroom).")
    except Exception as _vh:
        print(f"[C2C] browser VRAM headroom deferred: {_vh}")
    try:
        from .nodes.c2c_video.routes import register_routes as _register_c2c_video_routes
        _register_c2c_video_routes()
        print("[C2C] video loader routes registered (/c2c/video/probe, /c2c/video/preview).")
    except Exception as _cvr:
        print(f"[C2C] video loader routes deferred: {_cvr}")
    try:
        from .nodes.seedvr2_preview import register as _register_seedvr2_preview
        if _register_seedvr2_preview():
            print("[C2C] SeedVR2 preview ready (attaches to the numz SeedVR2 upscaler when it runs).")
    except Exception as _s2p:
        print(f"[C2C] SeedVR2 preview deferred: {_s2p}")
    try:
        from .nodes._c2c_secrets import register_routes as _register_c2c_secrets_routes, backend_name as _c2c_secrets_backend
        _register_c2c_secrets_routes(_ps)
        print(f"[C2C] secrets vault routes registered (/c2c/secrets/*) backend={_c2c_secrets_backend()}.")
    except Exception as _sec:
        print(f"[C2C] secrets vault deferred: {_sec}")
    try:
        from .nodes.style_presets import register_routes as _register_style_presets_routes
        _register_style_presets_routes(_ps)
        print("[C2C] style_presets routes registered.")
    except Exception as _sp:
        print(f"[C2C] style_presets deferred: {_sp}")
    # ── C2C Preset Hub (live aggregator: lexica/civitai/hf/openart/pdexter/issues) ─
    try:
        from .nodes._c2c_preset_hub import register_routes as _register_preset_hub_routes
        _register_preset_hub_routes(_ps)
        print("[C2C] preset hub routes registered (/c2c/presets/*).")
    except Exception as _ph:
        print(f"[C2C] preset hub deferred: {_ph}")
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
    # ── C2C prompt library (Gallery sources: lexica today; civitai/openart next) ─
    try:
        from .c2c_ai.prompt_library import register_routes as _register_c2c_prompts_routes
        _register_c2c_prompts_routes(_ps)
        print("[C2C AI] prompt library registered (/c2c/prompts/*).")
    except Exception as _pl:
        print(f"[C2C AI] prompt library deferred: {_pl}")
    # ── C2C dep-conflict checker (Manager integration) ─────────────
    try:
        from .nodes.dep_check_routes import register_routes as _register_depcheck_routes
        _register_depcheck_routes(_ps)
        print("[C2C] depcheck routes registered (/c2c/depcheck/*).")
    except Exception as _dc:
        print(f"[C2C] depcheck routes deferred: {_dc}")
    try:
        from .nodes._legacy_replacements import register as _register_legacy_replacements
        _n_legacy = _register_legacy_replacements(_ps)
        print(f"[C2C] {_n_legacy} legacy node replacements registered (old saved workflows -> unified nodes).")
    except Exception as _lr:
        print(f"[C2C] legacy replacements deferred: {_lr}")
    print("[MEC] NukeNodeMax routes + hooks registered.")
except Exception as _e:
    print(f"[MEC] NukeNodeMax server hooks deferred: {_e}")

_c2c_fam_part = ", ".join(f"{_lbl} {_n}" for _lbl, _n in C2C_LOAD_SUMMARY["families"])
_c2c_n_failed = len(C2C_LOAD_SUMMARY["failed"])
_c2c_line = (
    f"[C2C] CustomNodePacks: {C2C_LOAD_SUMMARY['total']} nodes loaded "
    f"({_c2c_fam_part}) - {_c2c_n_failed} failed"
)
if C2C_LOAD_SUMMARY["failed"]:
    _c2c_fail_part = "; ".join(
        f"{_f['key']} ({_f['error'][:80]})" for _f in C2C_LOAD_SUMMARY["failed"]
    )
    _c2c_line += f" : {_c2c_fail_part}"
print(_c2c_line)
if C2C_LOAD_SUMMARY["failed"]:
    print(
        "[C2C] CustomNodePacks: the [C2C registry] lines above say how to fix "
        "each failed family."
    )
