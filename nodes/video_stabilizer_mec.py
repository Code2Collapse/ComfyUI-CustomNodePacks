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

One node, VideoStabilizerMEC, method:
  - classic     : feature tracking (CPU, fast, well-textured footage)
  - raft_flow   : RAFT dense optical flow (torchvision, BSD-3; GPU or CPU) - EXPERIMENTAL, see below
  - dense_flow  : OpenCV DIS dense flow (CPU, no model)
  - auto        : dense_flow on a long clip with a free GPU, else classic (unchanged: what auto always ran)

Measured (docs/evidence/L9.39/stabilizer_ab.json, synthetic hand-shake with known shifts, ground truth from the
engine's applied warps, 3 seeds): residual shake textured classic 0.001 px, RAFT-large 0.03-0.05, DIS 0.18-0.22,
RAFT-small 0.15-0.20; near-featureless plate classic 0.21-0.23 px, DIS 0.69-0.86, RAFT 2.6-4.5. RAFT is therefore
not "robust on texture-poor plates" (the old label's claim) and auto does not pick it.
(VideoStabilizerClassic/Flow/AutoMEC are deprecated shims.)

L9.39: "raft_flow" used to run the vendored engine's OpenCV DIS flow - no RAFT anywhere, and raft_iters / use_half
were swallowed by a TypeError retry. RAFT now plugs into the engine's flow interface (_RaftFlow.calc), so the
engine's transform fitting, smoothing and framing are unchanged; the DIS path stays as method=dense_flow, and the
deprecated Flow / Auto shims keep using it so their saved workflows render as before.

License: upstream is MIT; see third_party/ComfyUI-Video-Stabilizer/LICENSE.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from ._is_changed_util import hash_args_and_kwargs

log = logging.getLogger("MEC.video_stabilizer")

# ---------------------------------------------------------------------
# Vendored import: load upstream modules from absolute paths to avoid the
# `nodes` package-name collision with our own pack.
# ---------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
# The engine (MIT, nomadoor/ComfyUI-Video-Stabilizer) lives IN the repo: it used to be read from the gitignored
# third_party/ folder, so on every fresh install the four Video Stabilizer nodes failed (L7.90, 2026-10-12).
_VENDOR = _HERE / "video_stabilizer"

HAS_STABILIZER = False
HAS_FLOW = False
_IMPORT_ERROR: Optional[str] = None
_classic_normalize = _classic_reconstruct = _classic_stabilize = None  # type: ignore
_flow_normalize = _flow_reconstruct = _flow_stabilize = None           # type: ignore
_convert_masks_for_output = _parse_padding_color = None                # type: ignore
_inverse_stabilize = None                                               # type: ignore
_FLOW_MOD = None


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
    _utils_path = _VENDOR / "stabilizer_utils.py"
    _classic_path = _VENDOR / "video_stabilizer_classic.py"
    _flow_path = _VENDOR / "video_stabilizer_flow.py"

    # Make `nodes.stabilizer_utils` importable for the inner relative imports.
    # The upstream files use `from .stabilizer_utils import ...` so we need to
    # set up a synthetic `mec_stab_pkg` package + register submodules under it.
    import importlib.util as _ilu
    pkg_name = "_mec_vendored_stabilizer"
    pkg_spec = _ilu.spec_from_loader(pkg_name, loader=None,
                                     is_package=True)
    pkg = _ilu.module_from_spec(pkg_spec)
    pkg.__path__ = [str(_VENDOR)]                # type: ignore[attr-defined]
    sys.modules[pkg_name] = pkg

    # Load utils as submodule of the synthetic package.
    utils_spec = _ilu.spec_from_file_location(
        f"{pkg_name}.stabilizer_utils", str(_utils_path))
    utils_mod = _ilu.module_from_spec(utils_spec)              # type: ignore[arg-type]
    sys.modules[f"{pkg_name}.stabilizer_utils"] = utils_mod
    utils_spec.loader.exec_module(utils_mod)                   # type: ignore[union-attr]

    _convert_masks_for_output = utils_mod._convert_masks_for_output
    _parse_padding_color = utils_mod._parse_padding_color
    _inverse_stabilize = getattr(utils_mod, "_apply_inverse_stabilization", None)

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
        _FLOW_MOD = flow_mod
        HAS_FLOW = True
    except Exception as e:
        log.warning("[stabilizer] flow backend unavailable: %s", e)
        HAS_FLOW = False
except Exception as e:
    _IMPORT_ERROR = str(e)
    HAS_STABILIZER = False
    HAS_FLOW = False


# ---------------------------------------------------------------------
# RAFT (torchvision, BSD-3) behind the engine's flow interface (L9.39)
# ---------------------------------------------------------------------
_RAFT_FILES = {"small": "raft_small_C_T_V2-01064c6d.pth", "large": "raft_large_C_T_SKHT_V2-ff5fadd5.pth"}
_RAFT_URL = "https://download.pytorch.org/models/"
_RAFT_CACHE: Dict[Tuple[str, str], Any] = {}


def _raft_dir() -> str:
    try:
        import folder_paths  # type: ignore
        return os.path.join(folder_paths.models_dir, "optical_flow")
    except Exception:  # noqa: BLE001 - outside ComfyUI
        return os.path.join(torch.hub.get_dir(), "checkpoints")


def _raft_weights(size: str) -> str:
    """The RAFT checkpoint: models/optical_flow/ or the torch hub cache; downloaded (hash-checked, into
    models/optical_flow/) only when the node's allow_download is ticked (L7.97)."""
    from ._download_policy import ModelNotDownloaded, downloads_allowed, note_found
    name = _RAFT_FILES[size]
    target = _raft_dir()
    for cand in (os.path.join(target, name), os.path.join(torch.hub.get_dir(), "checkpoints", name)):
        if os.path.isfile(cand):
            note_found(cand)
            return cand
    if not downloads_allowed():
        raise ModelNotDownloaded(f"RAFT-{size} (torchvision optical flow)", target=target,
                                 source=_RAFT_URL + name, file_name=name)
    os.makedirs(target, exist_ok=True)
    dst = os.path.join(target, name)
    torch.hub.download_url_to_file(_RAFT_URL + name, dst, hash_prefix=name.rsplit("-", 1)[1].split(".")[0],
                                   progress=False)
    return dst


def _raft_installed(size: str) -> bool:
    name = _RAFT_FILES[size]
    return any(os.path.isfile(os.path.join(d, name))
               for d in (_raft_dir(), os.path.join(torch.hub.get_dir(), "checkpoints")))


def _load_raft(size: str, device: str):
    key = (size, device)
    if key not in _RAFT_CACHE:
        from torchvision.models.optical_flow import raft_large, raft_small
        model = (raft_large if size == "large" else raft_small)(weights=None, progress=False)
        model.load_state_dict(torch.load(_raft_weights(size), map_location="cpu", weights_only=True))
        _RAFT_CACHE.clear()                    # one RAFT resident at a time
        _RAFT_CACHE[key] = model.to(device).eval()
    return _RAFT_CACHE[key]


def _release_raft() -> None:
    _RAFT_CACHE.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


try:
    from ._c2c_memguard import register as _c2c_mem_register
    _c2c_mem_register("cnp.stabilizer.raft", _release_raft, lambda: bool(_RAFT_CACHE))
except Exception:  # noqa: BLE001
    pass


class _RaftFlow:
    """`.calc(prev_gray, curr_gray, None) -> HxWx2` flow in pixels, as the engine expects from OpenCV DIS.
    RAFT-large on CUDA, RAFT-small on CPU - an installed copy of the other size beats a download; iters = RAFT's
    num_flow_updates; half = fp16 autocast (CUDA only)."""

    MIN_SIDE = 128          # RAFT's correlation pyramid needs feature maps >= 16 (input / 8)

    def __init__(self, iters: int = 12, half: bool = True, device: Optional[str] = None, size: Optional[str] = None):
        if device is None:
            try:                                   # ComfyUI's device: --cpu means CPU even with a GPU present
                import comfy.model_management as mm
                device = str(mm.get_torch_device())
            except Exception:  # noqa: BLE001 - outside ComfyUI
                device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        on_cuda = self.device.startswith("cuda")
        if size is None:
            size = "large" if on_cuda else "small"
            other = "small" if size == "large" else "large"
            if not _raft_installed(size) and _raft_installed(other):
                size = other
        self.size = size
        self.iters = int(max(1, iters))
        self.half = bool(half) and on_cuda
        self.model = _load_raft(self.size, self.device)
        self.name = f"RAFT-{self.size} ({self.device}{', fp16' if self.half else ''}, {self.iters} iters)"

    def _prep(self, gray: np.ndarray, H8: int, W8: int) -> torch.Tensor:
        x = torch.from_numpy(np.ascontiguousarray(gray)).to(self.device, torch.float32) / 255.0
        x = x[None, None].expand(1, 3, *x.shape)
        h, w = gray.shape[:2]
        if (H8, W8) != (h, w):                 # RAFT needs sides divisible by 8 and at least MIN_SIDE
            x = torch.nn.functional.pad(x, (0, W8 - w, 0, H8 - h), mode="replicate")
        return x * 2.0 - 1.0                   # the weights' own normalisation (OpticalFlow preset)

    def calc(self, prev: np.ndarray, curr: np.ndarray, _flow=None) -> np.ndarray:
        h, w = prev.shape[:2]
        H8, W8 = max(self.MIN_SIDE, -(-h // 8) * 8), max(self.MIN_SIDE, -(-w // 8) * 8)
        a, b = self._prep(prev, H8, W8), self._prep(curr, H8, W8)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=self.half):
            flow = self.model(a, b, num_flow_updates=self.iters)[-1]
        return flow[0, :, :h, :w].float().permute(1, 2, 0).cpu().numpy()


def _with_flow_backend(backend_obj, name: str, fn):
    """Run the engine with `backend_obj` as its flow estimator (the engine asks _select_flow_backend() once per
    clip; ComfyUI runs one prompt at a time, and the original is always restored)."""
    orig = _FLOW_MOD._select_flow_backend
    _FLOW_MOD._select_flow_backend = lambda: (name, backend_obj, None)
    try:
        return fn()
    finally:
        _FLOW_MOD._select_flow_backend = orig


def _json_default(o):
    """numpy scalars / arrays in the engine's meta, as JSON."""
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.generic):
        return o.item()
    return str(o)


def _require_stabilizer() -> None:
    if not HAS_STABILIZER:
        raise RuntimeError(
            "The video stabilizer engine failed to load "
            f"({_VENDOR}): {_IMPORT_ERROR}"
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
    # The engine hands back an ndarray (B,H,W[,1]) - `not mask_list` raised "truth value of an array is ambiguous",
    # so the unified node never ran against the real engine (L7.90, 2026-10-12). Lists, arrays and tensors all work.
    if mask_list is None or len(mask_list) == 0:
        return torch.zeros(B, H, W, dtype=torch.float32)
    if isinstance(mask_list, torch.Tensor):
        arr = mask_list.detach().float().cpu().numpy()
    elif isinstance(mask_list, np.ndarray):
        arr = mask_list.astype(np.float32)
    else:
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
# Unified VideoStabilizerMEC (replaces VideoStabilizerClassic/Flow/Auto)
# =====================================================================
class VideoStabilizerMEC:
    """Unified video stabilizer — pick a backend via the `method` dropdown.

    Methods:
      - auto       : dense_flow on a clip over 24 frames with 4 GB free on the GPU, else classic
      - classic    : feature-tracking (CPU, fast; most accurate in our measurements)
      - raft_flow  : RAFT dense optical flow (torchvision) - experimental
      - dense_flow : OpenCV DIS dense flow (CPU, no model; what raft_flow ran before L9.39)

    Outputs both stabilized frames AND a padding mask so the result can be
    chained directly into InpaintCropProMEC to fill the introduced borders.

    Wraps the vendored MIT ComfyUI-Video-Stabilizer (nomadoor et al., 2025).
    """
    VRAM_TIER = 2
    COLOR = "#6a4d91"

    PRESETS = {
        "handheld_light": dict(transform_mode="similarity",  strength=0.7,  smooth=0.5,
                               camera_lock=False, framing_mode="crop_and_pad", keep_fov=0.6),
        "handheld_heavy": dict(transform_mode="perspective", strength=0.9,  smooth=0.7,
                               camera_lock=False, framing_mode="crop_and_pad", keep_fov=0.5),
        "vehicle":        dict(transform_mode="similarity",  strength=0.85, smooth=0.85,
                               camera_lock=False, framing_mode="expand",      keep_fov=0.5),
        "tripod_lock":    dict(transform_mode="translation", strength=1.0,  smooth=0.95,
                               camera_lock=True,  framing_mode="crop_and_pad", keep_fov=0.8),
        "manual":         None,  # honour the per-widget values below
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames":         ("IMAGE",),
                "method":         (["auto", "classic", "raft_flow", "dense_flow"], {"default": "auto",
                                   "tooltip": "classic = feature tracking (fast; the most accurate in our "
                                              "tests). dense_flow = OpenCV DIS optical flow (no model). raft_flow = "
                                              "RAFT optical flow, experimental: in our tests it beat dense_flow on "
                                              "textured plates and did worst on near-featureless ones. auto = "
                                              "dense_flow on long clips with a free GPU, else classic."}),
                "preset":         (["handheld_light", "handheld_heavy", "vehicle",
                                    "tripod_lock", "manual"],
                                   {"default": "handheld_light",
                                    "tooltip": "Preset overrides the manual widgets below unless set to 'manual'."}),
                "frame_rate":     ("FLOAT", {"default": 16.0, "min": 1.0, "step": 0.1}),
                "padding_color":  ("STRING", {"default": "127, 127, 127"}),
                # Manual overrides (used when preset == 'manual')
                "framing_mode":   (["crop", "crop_and_pad", "expand"],
                                   {"default": "crop_and_pad"}),
                "transform_mode": (["translation", "similarity", "perspective"],
                                   {"default": "similarity"}),
                "camera_lock":    ("BOOLEAN", {"default": False}),
                "strength":       ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05}),
                "smooth":         ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                "keep_fov":       ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.05}),
                # RAFT-flow-only
                "raft_iters":     ("INT", {"default": 12, "min": 4, "max": 32,
                                            "tooltip": "RAFT refinement iterations (num_flow_updates). Used only by "
                                                       "method=raft_flow."}),
                "use_half":       ("BOOLEAN", {"default": True,
                                                "tooltip": "fp16 for RAFT on a CUDA GPU (ignored on CPU). Used only by "
                                                           "method=raft_flow."}),
                "allow_download": ("BOOLEAN", {"default": False,
                                               "tooltip": "Download the RAFT weights (21 MB large / 4 MB small, "
                                                          "torchvision, BSD-3) into models/optical_flow/ when they "
                                                          "are missing. Off: an installed copy is used, else "
                                                          "dense_flow runs and the info says so."}),
            },
            "optional": {
                # L9.39: the engine's exact inverse (upstream's "Video Stabilizer Inverse"), on this node
                "motion_meta": ("STRING", {"forceInput": True, "tooltip":
                    "Connect the motion_meta output of the Video Stabilizer that made these frames to PUT THE "
                    "ORIGINAL CAMERA MOTION BACK (after you painted / inpainted the steady plate). The frames must "
                    "be the stabilised size and count; the other settings are not used then. Pixels a crop framing "
                    "threw away cannot come back."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "STRING")
    RETURN_NAMES = ("stabilized_frames", "padding_mask", "info", "motion_meta")
    OUTPUT_TOOLTIPS = (
        "The stabilised clip (or, with motion_meta connected, the clip with its original camera motion back).",
        "1 where the frame does not cover the canvas: the borders to inpaint.",
        "What ran: method, flow engine, settings, time, and why a fallback happened.",
        "The exact per-frame warps as JSON. Wire it into another Video Stabilizer's motion_meta input to put the "
        "camera motion back after editing the steady plate.",
    )
    FUNCTION = "stabilize"
    CATEGORY = "C2C/Stabilization"
    DESCRIPTION = (
        "Video stabilizer: classic feature tracking, RAFT optical flow or OpenCV dense flow, with presets. "
        "Wraps the vendored MIT ComfyUI-Video-Stabilizer. Outputs frames + padding mask "
        "so it can chain straight into InpaintCropProMEC to fill borders."
    )

    @classmethod
    def IS_CHANGED(cls, frames, method, preset, frame_rate, padding_color,
                   framing_mode, transform_mode, camera_lock, strength, smooth,
                   keep_fov, raft_iters, use_half, **kwargs):
        return hash_args_and_kwargs(
            frames, method, preset, frame_rate, padding_color,
            framing_mode, transform_mode, camera_lock, strength, smooth,
            keep_fov, raft_iters, use_half, **kwargs,
        )

    # VideoStabilizerAutoMEC spelled its flow backend "flow" (its force_backend). Core's NodeReplace copies
    # values verbatim, so a graph migrated from it can carry "flow": accept it rather than fail validation.
    _METHOD_ALIASES = {"flow": "dense_flow"}   # VideoStabilizerAutoMEC's "flow" always ran DIS
    _METHODS = ("auto", "classic", "raft_flow", "dense_flow")

    @classmethod
    def VALIDATE_INPUTS(cls, method):
        # Naming only `method` keeps core's list/range checks for every other input.
        if method in cls._METHODS or method in cls._METHOD_ALIASES:
            return True
        return f"method must be one of {', '.join(cls._METHODS)}; got {method!r}"

    # --------------------------------------------------------------
    def _pick_backend(self, method: str, batch_size: int) -> str:
        if method != "auto":
            return method
        backend = "classic"
        if HAS_FLOW and torch.cuda.is_available():
            try:
                free, _total = torch.cuda.mem_get_info()
                if free >= 4 * (1024 ** 3) and batch_size > 24:
                    backend = "dense_flow"     # what auto always ran (it was labelled raft_flow); RAFT is opt-in
            except Exception:
                pass
        if not HAS_FLOW:
            backend = "classic"
        return backend

    def _resolve_params(self, preset, framing_mode, transform_mode, camera_lock,
                         strength, smooth, keep_fov):
        cfg = self.PRESETS.get(preset)
        if cfg is None:  # manual
            return dict(framing_mode=framing_mode, transform_mode=transform_mode,
                        camera_lock=camera_lock, strength=strength, smooth=smooth,
                        keep_fov=keep_fov)
        return dict(cfg)

    # --------------------------------------------------------------
    def stabilize(self, frames, method, preset, frame_rate, padding_color,
                  framing_mode, transform_mode, camera_lock, strength, smooth,
                  keep_fov, raft_iters, use_half, allow_download=False, motion_meta=None):
        method = self._METHOD_ALIASES.get(method, method)
        _require_stabilizer()
        if motion_meta:
            return self._restore_motion(frames, motion_meta, padding_color)
        if not isinstance(frames, torch.Tensor) or frames.dim() != 4 or frames.shape[-1] != 3:
            raise ValueError(
                f"VideoStabilizerMEC expects IMAGE tensor [B,H,W,3], got {tuple(getattr(frames, 'shape', ()))}"
            )
        from ._download_policy import allow_downloads
        with torch.no_grad(), allow_downloads(allow_download):
            return self._stabilize_impl(
                frames, method, preset, frame_rate, padding_color,
                framing_mode, transform_mode, camera_lock, strength, smooth,
                keep_fov, raft_iters, use_half,
            )

    def _restore_motion(self, frames, motion_meta, padding_color):
        """Inverse: stabilised (edited) frames + the meta that made them -> the original camera motion."""
        if _inverse_stabilize is None:
            raise RuntimeError("This engine build has no inverse stabilisation.")
        try:
            meta = json.loads(motion_meta) if isinstance(motion_meta, str) else dict(motion_meta)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"motion_meta is not the JSON a Video Stabilizer writes: {exc}") from exc
        t0 = time.time()
        with torch.no_grad():
            res = _inverse_stabilize(_classic_normalize(frames), meta, _parse_padding_color(padding_color))
        B, H, W, _ = frames.shape
        out = _stable_result_to_image(res, frames)
        masks = _masks_list_to_tensor(res.masks, out.shape[0], out.shape[1], out.shape[2])
        inv = res.meta.get("inverse_stabilization", {})
        info = (f"restored the original camera motion: frames={B} {W}x{H} -> "
                f"{out.shape[2]}x{out.shape[1]} framing={inv.get('framing_mode')} elapsed={time.time() - t0:.2f}s")
        return (out, masks, info, json.dumps(res.meta, default=_json_default))

    def _stabilize_impl(self, frames, method, preset, frame_rate, padding_color,
                        framing_mode, transform_mode, camera_lock, strength, smooth,
                        keep_fov, raft_iters, use_half):
        B, H, W, _ = frames.shape

        backend = self._pick_backend(method, B)
        notes: List[str] = []
        if backend in ("raft_flow", "dense_flow") and not HAS_FLOW:
            notes.append(f"{backend} unavailable (flow engine failed to load) - classic used")
            backend = "classic"
        raft = None
        if backend == "raft_flow":
            try:
                raft = _RaftFlow(raft_iters, use_half)
            except Exception as exc:  # noqa: BLE001 - weights missing / no permission / torchvision too old
                from ._download_policy import model_status
                notes.append(f"RAFT not used ({model_status(fallback_why=str(exc))}): {exc} - dense_flow used")
                backend = "dense_flow"

        p = self._resolve_params(preset, framing_mode, transform_mode,
                                  camera_lock, strength, smooth, keep_fov)

        t0 = time.time()
        flow_engine = ""
        try:
            if backend in ("raft_flow", "dense_flow"):
                context = _flow_normalize(frames)
                padding_rgb = _parse_padding_color(padding_color)

                def run():
                    return _flow_stabilize(
                        context=context, framing_mode=p["framing_mode"],
                        transform_mode=p["transform_mode"], camera_lock=p["camera_lock"],
                        strength=p["strength"], smooth=p["smooth"], keep_fov=p["keep_fov"],
                        padding_rgb=padding_rgb, frame_rate=frame_rate,
                    )
                result = _with_flow_backend(raft, raft.name, run) if raft is not None else run()
                meta = getattr(result, "meta", None) or {}
                flow_engine = str(meta.get("flow_backend") or (raft.name if raft else "DIS"))
                if meta.get("flow_fallback_reason"):
                    notes.append(str(meta["flow_fallback_reason"]))
            else:
                context = _classic_normalize(frames)
                padding_rgb = _parse_padding_color(padding_color)
                result = _classic_stabilize(
                    context=context, framing_mode=p["framing_mode"],
                    transform_mode=p["transform_mode"], camera_lock=p["camera_lock"],
                    strength=p["strength"], smooth=p["smooth"], keep_fov=p["keep_fov"],
                    padding_rgb=padding_rgb, frame_rate=frame_rate,
                )
            out_frames = _stable_result_to_image(result, frames)
            out_masks = _masks_list_to_tensor(result.masks, B, H, W)
        finally:
            _check_interrupt()

        info = (f"method={method} picked={backend}{f' flow={flow_engine}' if flow_engine else ''} preset={preset} "
                f"frames={B} HxW={H}x{W} transform={p['transform_mode']} "
                f"framing={p['framing_mode']} strength={p['strength']:.2f} "
                f"smooth={p['smooth']:.2f} elapsed={time.time() - t0:.2f}s")
        if notes:
            info += "\n" + "\n".join(notes)
        meta = getattr(result, "meta", None) or {}
        return (out_frames, out_masks, info, json.dumps(meta, default=_json_default))


# =====================================================================
# Deprecated alias shims — keep for ONE release (Batch 2 policy), then drop.
# Each emits a one-time warning then delegates to VideoStabilizerMEC.
# =====================================================================
class _DeprecatedStabilizerBase:
    _WARNED: set = set()

    @classmethod
    def _warn_once(cls, replacement_method: str):
        if cls.__name__ in _DeprecatedStabilizerBase._WARNED:
            return
        _DeprecatedStabilizerBase._WARNED.add(cls.__name__)
        log.warning(
            "[stabilizer] %s is deprecated and will be removed next release. "
            "Use VideoStabilizerMEC with method='%s'.",
            cls.__name__, replacement_method,
        )


class VideoStabilizerClassicMEC(_DeprecatedStabilizerBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames":         ("IMAGE",),
                "frame_rate":     ("FLOAT", {"default": 16.0, "min": 1.0, "step": 0.1}),
                "framing_mode":   (["crop", "crop_and_pad", "expand"], {"default": "crop_and_pad"}),
                "transform_mode": (["translation", "similarity", "perspective"], {"default": "similarity"}),
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
    DESCRIPTION = "DEPRECATED — use VideoStabilizerMEC (method=classic) instead."

    def stabilize(self, frames, frame_rate, framing_mode, transform_mode,
                  camera_lock, strength, smooth, keep_fov, padding_color):
        self._warn_once("classic")
        return VideoStabilizerMEC().stabilize(
            frames=frames, method="classic", preset="manual", frame_rate=frame_rate,
            padding_color=padding_color, framing_mode=framing_mode,
            transform_mode=transform_mode, camera_lock=camera_lock,
            strength=strength, smooth=smooth, keep_fov=keep_fov,
            raft_iters=12, use_half=True,
        )[:3]


class VideoStabilizerFlowMEC(_DeprecatedStabilizerBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames":         ("IMAGE",),
                "frame_rate":     ("FLOAT", {"default": 16.0, "min": 1.0, "step": 0.1}),
                "framing_mode":   (["crop", "crop_and_pad", "expand"], {"default": "crop_and_pad"}),
                "transform_mode": (["translation", "similarity", "perspective"], {"default": "similarity"}),
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
    DESCRIPTION = ("DEPRECATED — use VideoStabilizerMEC. This node always ran OpenCV DIS flow (method=dense_flow), "
                   "and still does, so saved workflows render as before; method=raft_flow is real RAFT.")

    def stabilize(self, frames, frame_rate, framing_mode, transform_mode,
                  camera_lock, strength, smooth, keep_fov, padding_color,
                  raft_iters, use_half):
        self._warn_once("dense_flow")
        return VideoStabilizerMEC().stabilize(
            frames=frames, method="dense_flow", preset="manual", frame_rate=frame_rate,
            padding_color=padding_color, framing_mode=framing_mode,
            transform_mode=transform_mode, camera_lock=camera_lock,
            strength=strength, smooth=smooth, keep_fov=keep_fov,
            raft_iters=raft_iters, use_half=use_half,
        )[:3]


class VideoStabilizerAutoMEC(_DeprecatedStabilizerBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames":         ("IMAGE",),
                "frame_rate":     ("FLOAT", {"default": 16.0, "min": 1.0, "step": 0.1}),
                "force_backend":  (["auto", "classic", "flow"], {"default": "auto"}),
                "preset":         (["handheld_light", "handheld_heavy", "vehicle", "tripod_lock"],
                                   {"default": "handheld_light"}),
                "padding_color":  ("STRING", {"default": "127, 127, 127"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("stabilized_frames", "padding_mask", "info")
    FUNCTION = "stabilize"
    CATEGORY = "C2C/Stabilization"
    DESCRIPTION = "DEPRECATED — use VideoStabilizerMEC (method=auto) instead."

    def stabilize(self, frames, frame_rate, force_backend, preset, padding_color):
        self._warn_once("auto")
        method = "dense_flow" if force_backend == "flow" else force_backend   # its "flow" was always DIS
        return VideoStabilizerMEC().stabilize(
            frames=frames, method=method, preset=preset, frame_rate=frame_rate,
            padding_color=padding_color,
            # preset drives these — manual fields are placeholders
            framing_mode="crop_and_pad", transform_mode="similarity",
            camera_lock=False, strength=0.7, smooth=0.5, keep_fov=0.6,
            raft_iters=12, use_half=True,
        )[:3]


# =====================================================================
NODE_CLASS_MAPPINGS = {
    "VideoStabilizerMEC":        VideoStabilizerMEC,
    # Deprecated aliases — remove next release.
    "VideoStabilizerClassicMEC": VideoStabilizerClassicMEC,
    "VideoStabilizerFlowMEC":    VideoStabilizerFlowMEC,
    "VideoStabilizerAutoMEC":    VideoStabilizerAutoMEC,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoStabilizerMEC":        "Video Stabilizer",
    "VideoStabilizerClassicMEC": "Video Stabilizer — Classic (deprecated)",
    "VideoStabilizerFlowMEC":    "Video Stabilizer — Dense Flow (deprecated)",
    "VideoStabilizerAutoMEC":    "Video Stabilizer — Auto (deprecated)",
}
