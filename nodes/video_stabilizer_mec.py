# FILE: nodes/video_stabilizer_mec.py
# FEATURE: MEC wrappers around the vendored ComfyUI-Video-Stabilizer (MIT,
# (c) 2025 ComfyUI Video Stabilizer Contributors, by nomadoor).
# INTEGRATES WITH:
#   third_party/ComfyUI-Video-Stabilizer/  (vendored upstream)
#   nodes/inpaint_suite.py                  (chain stabilization → inpaint crop)
"""
V1-style ComfyUI nodes that re-use the upstream Video-Stabilizer pipeline.

Why a wrapper?
  - Upstream uses the V3 `comfy_api.latest.ComfyExtension` registration,
    which lives in a different discovery path than our pack's V1
    `NODE_CLASS_MAPPINGS`. Wrapping lets us register inside our suite
    without modifying the upstream code, and it lets us:
      * default values tuned for the MEC inpaint pipeline,
      * "auto" preset that picks classic vs. flow based on clip length,
      * pass-through MASK input so we can chain stabilizer → InpaintCropPro,
      * standard MEC info string with timings.

Three nodes:
  - VideoStabilizerClassicMEC    : feature-tracking (CPU, fast)
  - VideoStabilizerFlowMEC       : RAFT dense flow (GPU, robust to texture-poor)
  - VideoStabilizerAutoMEC       : auto picks the backend based on clip length /
                                    available GPU memory.

License: upstream is MIT; see third_party/ComfyUI-Video-Stabilizer/LICENSE.
"""
from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

log = logging.getLogger("MEC.video_stabilizer")

# ---------------------------------------------------------------------
# Vendored import: load upstream modules from absolute paths to avoid the
# `nodes` package-name collision with our own pack.
# ---------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_VENDOR = _HERE.parent / "third_party" / "ComfyUI-Video-Stabilizer"

HAS_STABILIZER = False
HAS_FLOW = False
_IMPORT_ERROR: Optional[str] = None
_classic_normalize = _classic_reconstruct = _classic_stabilize = None  # type: ignore
_flow_normalize = _flow_reconstruct = _flow_stabilize = None           # type: ignore
_convert_masks_for_output = _parse_padding_color = None                # type: ignore


def _load_module_from_path(name: str, path: Path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


try:
    if not _VENDOR.is_dir():
        raise ImportError(f"vendored stabilizer not found at {_VENDOR}")
    # Load utils first — both backends import from it.
    _utils_path = _VENDOR / "nodes" / "stabilizer_utils.py"
    _classic_path = _VENDOR / "nodes" / "video_stabilizer_classic.py"
    _flow_path = _VENDOR / "nodes" / "video_stabilizer_flow.py"

    # Make `nodes.stabilizer_utils` importable for the inner relative imports.
    # The upstream files use `from .stabilizer_utils import ...` so we need to
    # set up a synthetic `mec_stab_pkg` package + register submodules under it.
    import importlib.util as _ilu
    pkg_name = "_mec_vendored_stabilizer"
    pkg_spec = _ilu.spec_from_loader(pkg_name, loader=None,
                                     is_package=True)
    pkg = _ilu.module_from_spec(pkg_spec)
    pkg.__path__ = [str(_VENDOR / "nodes")]                # type: ignore[attr-defined]
    sys.modules[pkg_name] = pkg

    # Load utils as submodule of the synthetic package.
    utils_spec = _ilu.spec_from_file_location(
        f"{pkg_name}.stabilizer_utils", str(_utils_path))
    utils_mod = _ilu.module_from_spec(utils_spec)              # type: ignore[arg-type]
    sys.modules[f"{pkg_name}.stabilizer_utils"] = utils_mod
    utils_spec.loader.exec_module(utils_mod)                   # type: ignore[union-attr]

    _convert_masks_for_output = utils_mod._convert_masks_for_output
    _parse_padding_color = utils_mod._parse_padding_color

    # Load classic backend.
    classic_spec = _ilu.spec_from_file_location(
        f"{pkg_name}.video_stabilizer_classic", str(_classic_path))
    classic_mod = _ilu.module_from_spec(classic_spec)         # type: ignore[arg-type]
    sys.modules[f"{pkg_name}.video_stabilizer_classic"] = classic_mod
    classic_spec.loader.exec_module(classic_mod)              # type: ignore[union-attr]

    _classic_normalize = classic_mod._normalize_video_input
    _classic_reconstruct = classic_mod._reconstruct_video
    _classic_stabilize = classic_mod._stabilize_frames
    HAS_STABILIZER = True

    # Load flow backend (best-effort — may need raft weights / GPU).
    try:
        flow_spec = _ilu.spec_from_file_location(
            f"{pkg_name}.video_stabilizer_flow", str(_flow_path))
        flow_mod = _ilu.module_from_spec(flow_spec)            # type: ignore[arg-type]
        sys.modules[f"{pkg_name}.video_stabilizer_flow"] = flow_mod
        flow_spec.loader.exec_module(flow_mod)                 # type: ignore[union-attr]
        _flow_normalize = flow_mod._normalize_video_input
        _flow_reconstruct = flow_mod._reconstruct_video
        _flow_stabilize = flow_mod._stabilize_frames
        HAS_FLOW = True
    except Exception as e:
        log.warning("[stabilizer] flow backend unavailable: %s", e)
        HAS_FLOW = False
except Exception as e:
    _IMPORT_ERROR = str(e)
    HAS_STABILIZER = False
    HAS_FLOW = False


def _require_stabilizer() -> None:
    if not HAS_STABILIZER:
        raise RuntimeError(
            "ComfyUI-Video-Stabilizer not vendored or import failed. "
            f"Expected at {_VENDOR}.  Underlying error: {_IMPORT_ERROR}"
        )


def _check_interrupt() -> None:
    try:
        import comfy.model_management as mm  # type: ignore
        mm.throw_exception_if_processing_interrupted()
    except ImportError:
        pass


def _masks_list_to_tensor(mask_list: List[Any], B: int, H: int, W: int) -> torch.Tensor:
    """Convert upstream mask output (list of numpy arrays) to (B,H,W) float tensor."""
    import numpy as np
    if not mask_list:
        return torch.zeros(B, H, W, dtype=torch.float32)
    arr = np.stack([np.asarray(m, dtype=np.float32) for m in mask_list], axis=0)
    if arr.ndim == 4:                # (B,H,W,1) -> (B,H,W)
        arr = arr[..., 0]
    if arr.shape[-2:] != (H, W):
        # Best-effort resize via torch.
        t = torch.from_numpy(arr).unsqueeze(1)
        t = torch.nn.functional.interpolate(t, size=(H, W),
                                            mode="nearest").squeeze(1)
        return t.clamp(0, 1)
    return torch.from_numpy(arr).clamp(0, 1)


def _stable_result_to_image(result, frames_in: torch.Tensor) -> torch.Tensor:
    """Convert StabilizationResult.frames (list of numpy float [0,1] HxWx3) to IMAGE."""
    import numpy as np
    arr = np.stack(result.frames, axis=0).astype(np.float32)
    if arr.shape[-1] != 3:
        # Ensure RGB.
        if arr.shape[-1] == 4:
            arr = arr[..., :3]
        elif arr.ndim == 4 and arr.shape[1] == 3:
            arr = arr.transpose(0, 2, 3, 1)
    return torch.from_numpy(arr).clamp(0, 1)


# =====================================================================
# A. VideoStabilizerClassicMEC — feature-tracking (CPU)
# =====================================================================
class VideoStabilizerClassicMEC:
    """Feature-tracking video stabilizer (sparse GFTT + LK optical flow).

    Best for: well-textured footage, short to medium clips, CPU-only.
    Outputs both stabilized frames AND a padding mask — the padding mask
    is essential to feed into InpaintCropProMEC if you want the inpaint
    pipeline to fill the introduced borders.

    Wraps upstream `VideoStabilizerClassic` (MIT, vendored).
    """
    VRAM_TIER = 1
    COLOR = "#7a4f9c"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames":         ("IMAGE",),
                "frame_rate":     ("FLOAT", {"default": 16.0, "min": 1.0, "step": 0.1}),
                "framing_mode":   (["crop", "crop_and_pad", "expand"],
                                   {"default": "crop_and_pad"}),
                "transform_mode": (["translation", "similarity", "perspective"],
                                   {"default": "similarity"}),
                "camera_lock":    ("BOOLEAN", {"default": False}),
                "strength":       ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05}),
                "smooth":         ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                "keep_fov":       ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.05}),
                "padding_color":  ("STRING", {"default": "127, 127, 127"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("stabilized_frames", "padding_mask", "info")
    FUNCTION = "stabilize"
    CATEGORY = "C2C/Stabilization"
    DESCRIPTION = "Feature-tracking video stabilizer (vendored MIT ComfyUI-Video-Stabilizer)."

    def stabilize(self, frames: torch.Tensor, frame_rate: float,
                  framing_mode: str, transform_mode: str, camera_lock: bool,
                  strength: float, smooth: float, keep_fov: float,
                  padding_color: str):
        _require_stabilizer()
        t0 = time.time()
        if frames.dim() != 4 or frames.shape[-1] != 3:
            raise ValueError(f"IMAGE shape (B,H,W,3) expected, got {tuple(frames.shape)}")
        B, H, W, _ = frames.shape
        try:
            context = _classic_normalize(frames)
            padding_rgb = _parse_padding_color(padding_color)
            result = _classic_stabilize(
                context=context,
                framing_mode=framing_mode,
                transform_mode=transform_mode,
                camera_lock=camera_lock,
                strength=strength,
                smooth=smooth,
                keep_fov=keep_fov,
                padding_rgb=padding_rgb,
                frame_rate=frame_rate,
            )
            out_frames = _stable_result_to_image(result, frames)
            out_masks = _masks_list_to_tensor(result.masks, B, H, W)
        finally:
            _check_interrupt()
        info = (f"backend=classic frames={B} HxW={H}x{W} "
                f"transform={transform_mode} framing={framing_mode} "
                f"strength={strength:.2f} smooth={smooth:.2f} "
                f"elapsed={time.time() - t0:.2f}s")
        return (out_frames, out_masks, info)


# =====================================================================
# B. VideoStabilizerFlowMEC — RAFT dense flow (GPU)
# =====================================================================
class VideoStabilizerFlowMEC:
    """Dense optical-flow video stabilizer using RAFT.

    Best for: texture-poor footage (smooth surfaces, blurred plates) where
    sparse feature tracking fails. Requires GPU. Heavier than classic but
    far more robust.

    Wraps upstream `VideoStabilizerFlow` (MIT, vendored).
    """
    VRAM_TIER = 4
    COLOR = "#5a3a8e"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames":         ("IMAGE",),
                "frame_rate":     ("FLOAT", {"default": 16.0, "min": 1.0, "step": 0.1}),
                "framing_mode":   (["crop", "crop_and_pad", "expand"],
                                   {"default": "crop_and_pad"}),
                "transform_mode": (["translation", "similarity", "perspective"],
                                   {"default": "similarity"}),
                "camera_lock":    ("BOOLEAN", {"default": False}),
                "strength":       ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05}),
                "smooth":         ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                "keep_fov":       ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.05}),
                "padding_color":  ("STRING", {"default": "127, 127, 127"}),
                "raft_iters":     ("INT", {"default": 12, "min": 4, "max": 32}),
                "use_half":       ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("stabilized_frames", "padding_mask", "info")
    FUNCTION = "stabilize"
    CATEGORY = "C2C/Stabilization"
    DESCRIPTION = "RAFT dense-flow video stabilizer (vendored MIT ComfyUI-Video-Stabilizer)."

    def stabilize(self, frames: torch.Tensor, frame_rate: float,
                  framing_mode: str, transform_mode: str, camera_lock: bool,
                  strength: float, smooth: float, keep_fov: float,
                  padding_color: str, raft_iters: int, use_half: bool):
        _require_stabilizer()
        if not HAS_FLOW:
            raise RuntimeError("Flow backend unavailable — falling back disabled. "
                               "Use VideoStabilizerClassicMEC or check upstream errors.")
        t0 = time.time()
        if frames.dim() != 4 or frames.shape[-1] != 3:
            raise ValueError(f"IMAGE shape (B,H,W,3) expected, got {tuple(frames.shape)}")
        B, H, W, _ = frames.shape
        try:
            context = _flow_normalize(frames)
            padding_rgb = _parse_padding_color(padding_color)
            # Try the upstream signature; some flow backends accept extra kwargs.
            try:
                result = _flow_stabilize(
                    context=context,
                    framing_mode=framing_mode,
                    transform_mode=transform_mode,
                    camera_lock=camera_lock,
                    strength=strength,
                    smooth=smooth,
                    keep_fov=keep_fov,
                    padding_rgb=padding_rgb,
                    frame_rate=frame_rate,
                    raft_iters=raft_iters,
                    use_half=use_half,
                )
            except TypeError:
                result = _flow_stabilize(
                    context=context,
                    framing_mode=framing_mode,
                    transform_mode=transform_mode,
                    camera_lock=camera_lock,
                    strength=strength,
                    smooth=smooth,
                    keep_fov=keep_fov,
                    padding_rgb=padding_rgb,
                    frame_rate=frame_rate,
                )
            out_frames = _stable_result_to_image(result, frames)
            out_masks = _masks_list_to_tensor(result.masks, B, H, W)
        finally:
            _check_interrupt()
        info = (f"backend=flow frames={B} HxW={H}x{W} "
                f"transform={transform_mode} framing={framing_mode} "
                f"strength={strength:.2f} smooth={smooth:.2f} "
                f"raft_iters={raft_iters} half={use_half} "
                f"elapsed={time.time() - t0:.2f}s")
        return (out_frames, out_masks, info)


# =====================================================================
# C. VideoStabilizerAutoMEC — auto-picks backend
# =====================================================================
class VideoStabilizerAutoMEC:
    """Auto-select stabilizer backend based on clip length and VRAM.

    Heuristic:
      - <= 24 frames OR no CUDA / <4 GB free  -> classic (CPU, fast)
      - else                                  -> flow (GPU, robust)
    Override with `force_backend`.
    """
    VRAM_TIER = 2
    COLOR = "#6a4d91"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames":         ("IMAGE",),
                "frame_rate":     ("FLOAT", {"default": 16.0, "min": 1.0, "step": 0.1}),
                "force_backend":  (["auto", "classic", "flow"], {"default": "auto"}),
                "preset":         (["handheld_light", "handheld_heavy",
                                    "vehicle", "tripod_lock"],
                                   {"default": "handheld_light"}),
                "padding_color":  ("STRING", {"default": "127, 127, 127"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("stabilized_frames", "padding_mask", "info")
    FUNCTION = "stabilize"
    CATEGORY = "C2C/Stabilization"
    DESCRIPTION = "Auto stabilizer that picks classic vs. flow backend based on clip length and VRAM."

    PRESETS = {
        "handheld_light": dict(transform_mode="similarity",  strength=0.7, smooth=0.5,
                               camera_lock=False, framing_mode="crop_and_pad", keep_fov=0.6),
        "handheld_heavy": dict(transform_mode="perspective", strength=0.9, smooth=0.7,
                               camera_lock=False, framing_mode="crop_and_pad", keep_fov=0.5),
        "vehicle":        dict(transform_mode="similarity",  strength=0.85, smooth=0.85,
                               camera_lock=False, framing_mode="expand",      keep_fov=0.5),
        "tripod_lock":    dict(transform_mode="translation", strength=1.0, smooth=0.95,
                               camera_lock=True,  framing_mode="crop_and_pad", keep_fov=0.8),
    }

    def stabilize(self, frames: torch.Tensor, frame_rate: float,
                  force_backend: str, preset: str, padding_color: str):
        _require_stabilizer()
        if frames.dim() != 4 or frames.shape[-1] != 3:
            raise ValueError(f"IMAGE shape (B,H,W,3) expected, got {tuple(frames.shape)}")
        B = frames.shape[0]

        # Decide backend.
        backend = force_backend
        if backend == "auto":
            backend = "classic"
            if HAS_FLOW and torch.cuda.is_available():
                try:
                    free, _total = torch.cuda.mem_get_info()
                    if free >= 4 * (1024 ** 3) and B > 24:
                        backend = "flow"
                except Exception:
                    pass
            if not HAS_FLOW:
                backend = "classic"

        cfg = self.PRESETS[preset]
        if backend == "flow":
            sub = VideoStabilizerFlowMEC()
            out, m, info = sub.stabilize(
                frames=frames, frame_rate=frame_rate,
                framing_mode=cfg["framing_mode"], transform_mode=cfg["transform_mode"],
                camera_lock=cfg["camera_lock"], strength=cfg["strength"],
                smooth=cfg["smooth"], keep_fov=cfg["keep_fov"],
                padding_color=padding_color, raft_iters=12, use_half=True,
            )
        else:
            sub = VideoStabilizerClassicMEC()
            out, m, info = sub.stabilize(
                frames=frames, frame_rate=frame_rate,
                framing_mode=cfg["framing_mode"], transform_mode=cfg["transform_mode"],
                camera_lock=cfg["camera_lock"], strength=cfg["strength"],
                smooth=cfg["smooth"], keep_fov=cfg["keep_fov"],
                padding_color=padding_color,
            )
        return (out, m, f"auto picked={backend} preset={preset} | {info}")


# =====================================================================
NODE_CLASS_MAPPINGS = {
    "VideoStabilizerClassicMEC": VideoStabilizerClassicMEC,
    "VideoStabilizerFlowMEC":    VideoStabilizerFlowMEC,
    "VideoStabilizerAutoMEC":    VideoStabilizerAutoMEC,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoStabilizerClassicMEC": "Video Stabilizer — Classic (C2C)",
    "VideoStabilizerFlowMEC":    "Video Stabilizer — RAFT Flow (C2C)",
    "VideoStabilizerAutoMEC":    "Video Stabilizer — Auto (C2C)",
}
