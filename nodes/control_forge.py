"""
control_forge.py — OmniControl Forge: a VFX-AOV-style control fusion node.

Design philosophy (from the VFX research): a Nuke comp never flattens render
passes into one RGB — it keeps depth / normals / motion vectors / ID mattes /
edges as SEPARATE AOVs and combines them mathematically, tuning each
independently. The diffusion equivalent of "maximum control / no pixel drift"
is the same: keep every control signal as its own pass and stack them
(multi-ControlNet at STAGGERED weights + a Tile ControlNet to lock layout),
rather than alpha-blending them into one image.

So this node emits the control signal in EVERY useful form, like an EXR with AOVs:
  - blended        : single convenience image (screen = least-destructive default)
  - channel_packed : depth=R, canny=G, pose=B (lossless per-signal packing; ideal
                     for union ControlNets — the EXR-AOV analogy)
  - depth/edge/pose/normal/motion/id : separate normalised AOV passes for
                     max-control multi-ControlNet stacking (the "no drift" path)
  - info           : present passes + recommended wiring

STAGING (this file = Stage 1):
  * VENDORED, zero extra deps, run internally NOW: Canny (OpenCV), Motion vectors
    (OpenCV Farneback optical flow → motion-vector pass).
  * ACCEPTED as inputs now (so the node is fully usable today): depth, pose,
    normal, id_matte — wire your existing DepthAnything/DWPose/NormalCrafter/SAM
    outputs in.
  * Stage 2 (separate change) vendors the heavy model loaders into this pack so
    they run internally too: DepthAnything V1/V2/V3, DWPose, ViTPose,
    DepthCrafter, NormalCrafter. Each lands as a real, tested backend — no stubs.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from ._is_changed_util import hash_args_and_kwargs

# Real LayerStyle (MIT) resize/scale/aspect code, vendored — gives ControlAOV the
# same width/height/scale/aspect/fit power as the dedicated resize nodes.
try:
    from ._vendored_resize import (
        resize_wh_or_scale as _resize_wh,
        SIZING_MODES as _RESIZE_MODES,
        FIT_WH as _RESIZE_FITS,
        RESAMPLE_METHODS as _RESIZE_METHODS,
    )
    _HAVE_RESIZE = True
except Exception:  # noqa: BLE001 - never let a resize-helper import break the node
    _resize_wh = None
    _RESIZE_MODES = ["off", "width/height", "scale"]
    _RESIZE_FITS = ["stretch", "pad", "crop"]
    _RESIZE_METHODS = ["lanczos", "bicubic", "hamming", "bilinear", "box", "nearest"]
    _HAVE_RESIZE = False

try:
    import cv2
    _HAVE_CV2 = True
except Exception:
    _HAVE_CV2 = False

_LUMA = (0.2126, 0.7152, 0.0722)
_BLEND_MODES = ("screen", "lighten_max", "linear_dodge", "multiply", "average", "weighted_avg", "overlay")


# ----------------------------------------------------------------- tensor utils
def _as_bhwc3(t):
    if t is None:
        return None
    if t.dim() == 3:
        t = t.unsqueeze(0)
    if t.shape[-1] == 1:
        t = t.repeat(1, 1, 1, 3)
    elif t.shape[-1] > 3:
        t = t[..., :3]
    return t.float().clamp(0.0, 1.0)


def _resize_to(t, h, w):
    if t.shape[1] == h and t.shape[2] == w:
        return t
    x = t.permute(0, 3, 1, 2)
    x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)
    return x.permute(0, 2, 3, 1).contiguous()


def _luma(t):
    return (t[..., 0] * _LUMA[0] + t[..., 1] * _LUMA[1] + t[..., 2] * _LUMA[2]).clamp(0, 1)


def _to_u8(frame_hwc):
    return (frame_hwc.detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)


# --------------------------------------------------------------- runtime delegation
# "Run everything internally" without re-downloading: call the preprocessor nodes
# already installed (comfyui_controlnet_aux etc.) via ComfyUI's global registry, so
# they use THEIR OWN models/paths — the same paths every other repo uses.
_DEPTH_MAP = {
    "depth_anything_v2": "DepthAnythingV2Preprocessor",
    "depth_anything_v1": "DepthAnythingPreprocessor",
    "depth_anything_v2_metric": "Metric_DepthAnythingV2Preprocessor",
    "zoe_depth_anything": "Zoe_DepthAnythingPreprocessor",
}
_POSE_MAP = {
    "dwpose": "DWPreprocessor",
    "openpose": "OpenposePreprocessor",
    "animal_pose": "AnimalPosePreprocessor",
    "densepose": "DensePosePreprocessor",
}
_EDGE_MAP = {
    "canny": "CannyEdgePreprocessor",   # controlnet_aux Canny (cleaner than internal OpenCV)
    "lineart": "LineArtPreprocessor",
    "anyline": "AnyLineArtPreprocessor_aux",
    "hed": "HEDPreprocessor",
    "pidinet": "PiDiNetPreprocessor",
    "teed": "TEEDPreprocessor",
}
# DepthAnything backbone checkpoints by size (S/M/L/G). The delegated preprocessor
# takes `ckpt_name`; a custom filename in its model dir overrides the size.
_DEPTH_CKPT = {
    "depth_anything_v2": {
        "small": "depth_anything_v2_vits.pth", "medium": "depth_anything_v2_vitb.pth",
        "large": "depth_anything_v2_vitl.pth", "giant": "depth_anything_v2_vitg.pth",
    },
    "depth_anything_v1": {
        "small": "depth_anything_vits14.pth", "medium": "depth_anything_vitb14.pth",
        "large": "depth_anything_vitl14.pth", "giant": "depth_anything_vitl14.pth",  # v1 has no giant
    },
}
_DEPTH_SIZES = ("small", "medium", "large", "giant")

# Vendored, self-contained model backends (run inside this pack; lazy + guarded).
try:
    from . import _control_backends as _cb
except Exception:  # pragma: no cover
    try:
        import _control_backends as _cb  # type: ignore
    except Exception:
        _cb = None
# depth_model values that route to the vendored runners (vs controlnet_aux delegation).
_VENDORED_DEPTH = ("da_v2", "da_v1", "midas", "da3", "depth_pro", "depthcrafter", "dvd")
_NORMAL_MODELS = ("off", "sobel_from_depth", "normalcrafter")


# ---- model-file pickers: surface the user's OWN installed models as dropdowns ----
# Read the actual model lists the delegated preprocessors expose, so the dropdown
# shows exactly what's on disk. Fully guarded: if a pack isn't installed the list
# is empty and the field falls back to a free-text STRING (never an empty combo).
def _aux_combo(class_name, *fields):
    try:
        import nodes as _cn  # ComfyUI global registry
        cls = _cn.NODE_CLASS_MAPPINGS.get(class_name)
        if cls is None:
            return []
        it = cls.INPUT_TYPES()
        allp = {**(it.get("required") or {}), **(it.get("optional") or {})}
        for f in fields:
            spec = allp.get(f)
            if isinstance(spec, (list, tuple)) and spec and isinstance(spec[0], (list, tuple)):
                return list(spec[0])
        return []
    except Exception:  # noqa: BLE001
        return []


def _dedupe(seq):
    seen, out = set(), []
    for x in seq:
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _depth_ckpt_options():
    opts = []
    opts += _aux_combo("DepthAnythingV2Preprocessor", "ckpt_name")
    opts += _aux_combo("DepthAnythingPreprocessor", "ckpt_name")
    opts += _aux_combo("Metric_DepthAnythingV2Preprocessor", "ckpt_name")
    for m in _DEPTH_CKPT.values():
        opts += list(m.values())
    return _dedupe(opts)


def _pose_ckpt_options():
    return _dedupe(_aux_combo("OnnxDetectionModelLoaderV2", "vitpose_model"))


def _ckpt_field(options, what):
    """A COMBO of ['(auto)'] + on-disk models, or a free-text STRING fallback if
    nothing was detected (so the field is never a broken empty dropdown)."""
    if options:
        return (["(auto)"] + options, {
            "default": "(auto)",
            "tooltip": f"Pick your own {what} model, or '(auto)' = use the backend/size default.",
        })
    return ("STRING", {
        "default": "",
        "tooltip": f"Custom {what} model filename (none auto-detected; type one or leave blank for default).",
    })


def _spec_default(spec):
    if not isinstance(spec, (list, tuple)) or not spec:
        return None
    t = spec[0]
    opts = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
    if isinstance(t, (list, tuple)):
        return t[0] if t else None
    if isinstance(opts, dict) and "default" in opts:
        return opts["default"]
    return {"INT": 0, "FLOAT": 0.0, "BOOLEAN": False, "STRING": ""}.get(t, None)


def _run_aux(class_name, image, target_res=512, overrides=None):
    """Resolve + call an installed preprocessor node. Returns (IMAGE_or_None, note)."""
    try:
        import nodes as _cn  # ComfyUI global registry
        cls = _cn.NODE_CLASS_MAPPINGS.get(class_name)
        if cls is None:
            return None, f"{class_name}=not-installed"
        params = {}
        it = cls.INPUT_TYPES()
        params.update(it.get("required", {}) or {})
        params.update(it.get("optional", {}) or {})
        ov = overrides or {}
        kwargs = {}
        for name, spec in params.items():
            if name == "image":
                kwargs[name] = image
            elif name in ov:
                kwargs[name] = ov[name]
            elif "resolution" in name:
                kwargs[name] = int(target_res)
            else:
                kwargs[name] = _spec_default(spec)
        out = getattr(cls(), cls.FUNCTION)(**kwargs)
        if isinstance(out, dict):  # NodeOutput-style
            out = out.get("result", ())
        if isinstance(out, (list, tuple)):
            for o in out:
                if torch.is_tensor(o) and o.dim() == 4 and o.shape[-1] in (1, 3):
                    return o, f"{class_name}=ok"
            if out and torch.is_tensor(out[0]):
                return out[0], f"{class_name}=ok"
        return None, f"{class_name}=no-image-out"
    except Exception as e:  # never crash the forge
        return None, f"{class_name}=err:{type(e).__name__}"


def _call_node(cls, fixed):
    """Run a registered node class, filling its INPUT_TYPES defaults and injecting
    `fixed` kwargs. Returns the node's output tuple."""
    it = cls.INPUT_TYPES()
    params = {}
    params.update(it.get("required", {}) or {})
    params.update(it.get("optional", {}) or {})
    kw = {}
    for name, spec in params.items():
        kw[name] = fixed[name] if name in fixed else _spec_default(spec)
    return getattr(cls(), cls.FUNCTION)(**kw)


def _run_vitpose(image, target_res=512, pose_ckpt=""):
    """ViTPose pose via WanV2's 3-node chain (loader -> detect -> draw) → skeleton IMAGE.
    Uses the ViTPose + YOLO .onnx files already in models/detection/. ``pose_ckpt``
    (if given and present in the list) picks a specific ViTPose .onnx; else auto."""
    try:
        import nodes as _cn
        Loader = _cn.NODE_CLASS_MAPPINGS.get("OnnxDetectionModelLoaderV2")
        Detect = _cn.NODE_CLASS_MAPPINGS.get("WanPoseDetectViTPoseV2")
        Draw = _cn.NODE_CLASS_MAPPINGS.get("DrawViTPoseV2")
        if not (Loader and Detect and Draw):
            return None, "vitpose=WanAnimatePreprocessV2 not installed"
        files = (Loader.INPUT_TYPES().get("required", {}).get("vitpose_model", [None]) or [None])[0] or []
        if not files:
            return None, "vitpose=no .onnx in models/detection/"
        onnx = [f for f in files if f.lower().endswith(".onnx")]
        lf = lambda s: s.lower()
        # ViTPose: prefer the HUMAN wholebody model; never an animal (apt36k) model
        # (those expect a different keypoint set → empty/garbage skeleton on people).
        vit = pose_ckpt if (pose_ckpt and pose_ckpt in files) else (
            next((f for f in onnx if "wholebody" in lf(f)), None)
            or next((f for f in onnx if "vitpose" in lf(f)
                     and "apt" not in lf(f) and "animal" not in lf(f)), None)
            or next((f for f in onnx if "vitpose" in lf(f)), None)
            or (onnx[0] if onnx else files[0]))
        # YOLO: a real PERSON detector ([N,C,640,640]); avoid pose/face/.pt files.
        yolo = (next((f for f in onnx if "yolov10" in lf(f)), None)
                or next((f for f in onnx if ("yolov8" in lf(f) or "yolox" in lf(f) or "yolo11" in lf(f))
                         and "face" not in lf(f) and "pose" not in lf(f)), None)
                or next((f for f in onnx if "yolo" in lf(f)
                         and "face" not in lf(f) and "pose" not in lf(f) and "vitpose" not in lf(f)), None))
        if yolo is None:
            return None, "vitpose=need a YOLO person-detector .onnx (e.g. yolov10m.onnx) in models/detection/"
        dev = "CUDAExecutionProvider" if _cuda() else "CPUExecutionProvider"
        model = _call_node(Loader, {"vitpose_model": vit, "yolo_model": yolo, "onnx_device": dev})[0]
        bundle = _call_node(Detect, {"images": image, "model": model})[0]
        H, W = int(image.shape[1]), int(image.shape[2])
        out = _call_node(Draw, {"pose_data": bundle, "width": W, "height": H})
        img = next((o for o in out if torch.is_tensor(o) and o.dim() == 4), out[0])
        return img, f"vitpose=ok [{vit}]"
    except Exception as e:
        return None, f"vitpose=err:{type(e).__name__}"


def _cuda():
    try:
        return torch.cuda.is_available()
    except Exception:
        return False


# ----------------------------------------------------------------- vendored passes
def _canny_pass(image_bhwc, low, high, aperture=3):
    """OpenCV Canny per frame → white edges on black, [B,H,W,3]."""
    if not _HAVE_CV2:
        return None
    ap = aperture if aperture in (3, 5, 7) else 3
    out = []
    for i in range(image_bhwc.shape[0]):
        g = cv2.cvtColor(_to_u8(image_bhwc[i]), cv2.COLOR_RGB2GRAY)
        e = cv2.Canny(g, int(low), int(high), apertureSize=ap)
        out.append(torch.from_numpy(e).float() / 255.0)
    e = torch.stack(out, 0).unsqueeze(-1).repeat(1, 1, 1, 3)
    return e.clamp(0, 1)


def _motion_pass(image_bhwc):
    """Farneback optical flow between consecutive frames → motion-vector AOV.
    flow_x→R, flow_y→G, magnitude→B. First frame is zero-motion."""
    if not _HAVE_CV2 or image_bhwc.shape[0] < 2:
        return torch.zeros(image_bhwc.shape[0], image_bhwc.shape[1], image_bhwc.shape[2], 3)
    grays = [cv2.cvtColor(_to_u8(image_bhwc[i]), cv2.COLOR_RGB2GRAY) for i in range(image_bhwc.shape[0])]
    H, W = grays[0].shape
    frames = [torch.zeros(H, W, 3)]
    for i in range(1, len(grays)):
        flow = cv2.calcOpticalFlowFarneback(grays[i - 1], grays[i], None,
                                            0.5, 3, 15, 3, 5, 1.2, 0)
        fx, fy = flow[..., 0], flow[..., 1]
        mag = np.sqrt(fx * fx + fy * fy)
        # normalise to 0..1 around 0.5 for direction, mag scaled by 95th pct
        scale = max(1e-3, float(np.percentile(np.abs(np.stack([fx, fy])), 95)))
        r = np.clip(fx / (2 * scale) + 0.5, 0, 1)
        g = np.clip(fy / (2 * scale) + 0.5, 0, 1)
        b = np.clip(mag / (scale * 2), 0, 1)
        frames.append(torch.from_numpy(np.stack([r, g, b], -1)).float())
    return torch.stack(frames, 0).clamp(0, 1)


# ----------------------------------------------------------------- the node
class ControlAOVC2C:
    CATEGORY = "C2C/Control"
    DESCRIPTION = ("VFX-AOV control fusion: emit depth/canny/pose/normal/motion/ID as separate passes, "
                   "a channel-packed image (depth=R/canny=G/pose=B), and a convenience blend. Runs Canny + "
                   "optical-flow motion internally; accepts depth/pose/normal/ID maps as inputs. Feeds any "
                   "ControlNet / union / control-video. Stack the separate passes for maximum spatial lock.")
    FUNCTION = "forge"
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("blended", "combined", "channel_packed", "depth", "canny", "pose", "normal", "motion", "id_matte", "info")
    OUTPUT_TOOLTIPS = (
        "Single image, all passes OVERLAID via the chosen blend mode.",
        "Single image, depth | pose | canny shown SIDE-BY-SIDE (or grid) — the clearest 'all 3 at once' view.",
        "depth=R, canny=G, pose=B — lossless packing for union ControlNets.",
        "Depth AOV (passthrough).", "Canny AOV (run internally if 'image' wired).",
        "Pose AOV (passthrough).", "Normal AOV (passthrough).",
        "Motion-vector AOV (optical flow, run internally on a frame batch).",
        "ID / segmentation matte AOV (passthrough).",
        "Per-pass STATUS + recommended max-control wiring.",
    )

    @classmethod
    def INPUT_TYPES(cls):
        wf = lambda d: ("FLOAT", {"default": d, "min": 0.0, "max": 2.0, "step": 0.05})
        return {
            "required": {
                "blend_mode": (_BLEND_MODES, {"default": "screen",
                               "tooltip": "How the 'blended' overlay combines passes. screen = least-destructive; "
                                          "linear_dodge clips; multiply darkens."}),
                "preview_layout": (("horizontal_3", "vertical_3", "grid_2x2"), {"default": "horizontal_3",
                               "tooltip": "Layout for the 'combined' output: horizontal_3 = depth | pose | canny; "
                                          "grid_2x2 = depth|pose // canny|original."}),
            },
            "optional": {
                "image": ("IMAGE", {"tooltip": "Source frames — preprocessors below run on this."}),
                "depth_model": (["off"] + list(_VENDORED_DEPTH) + list(_DEPTH_MAP.keys()), {"default": "off",
                                "tooltip": "Depth backend. VENDORED (run inside this pack): da_v2/da_v1/"
                                           "midas (transformers), da3 (Depth-Anything-3), depth_pro, depthcrafter (video), "
                                           "dvd (DVD — deterministic Wan-2.1 VIDEO depth, EnVision-Research; needs the "
                                           "DVD ckpt in models/DVD/, CC BY-NC). The depth_anything_* options delegate to "
                                           "comfyui_controlnet_aux. 'off' = wire an external depth map."}),
                "normal_model": (list(_NORMAL_MODELS), {"default": "off",
                                "tooltip": "Normal backend (vendored): sobel_from_depth (no model) or normalcrafter (video). "
                                           "'off' = wire an external normal map."}),
                "depth_size": (list(_DEPTH_SIZES), {"default": "small",
                               "tooltip": "DepthAnything backbone: small=ViT-S (fastest, test default) → giant=ViT-G "
                                          "(best). v1 has no giant (falls back to large). Ignored by metric/zoe."}),
                "depth_custom_ckpt": _ckpt_field(_depth_ckpt_options(), "depth"),
                "pose_model": (["off", "vitpose"] + list(_POSE_MAP.keys()), {"default": "off",
                               "tooltip": "Pose backend. vitpose = WanV2 ViTPose chain (uses models/detection/*.onnx). "
                                          "dwpose/openpose/... delegate to comfyui_controlnet_aux."}),
                "pose_ckpt": _ckpt_field(_pose_ckpt_options(), "ViTPose .onnx pose"),
                "id_matte_model": (["off", "sam_auto"], {"default": "off",
                               "tooltip": "ID/segmentation matte. sam_auto = automatic SAM segmentation (controlnet_aux "
                                          "SAMPreprocessor). 'off' = wire an external matte."}),
                "edge_model": (["internal_canny", "off"] + list(_EDGE_MAP.keys()), {"default": "internal_canny",
                               "tooltip": "internal_canny = OpenCV (no model). Others delegate to controlnet_aux."}),
                "run_canny": ("BOOLEAN", {"default": True, "tooltip": "Used only when edge_model = internal_canny."}),
                "canny_low": ("INT", {"default": 100, "min": 0, "max": 255}),
                "canny_high": ("INT", {"default": 200, "min": 0, "max": 255}),
                "canny_aperture": ([3, 5, 7], {"default": 3, "tooltip": "Sobel aperture for internal Canny (odd 3/5/7)."}),
                "depth_invert": ("BOOLEAN", {"default": False, "tooltip": "Invert depth (1 - depth). Use when source is 'far=bright' but the ControlNet expects 'near=bright'."}),
                "preproc_resolution": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 8,
                               "tooltip": "Resolution passed to delegated preprocessors."}),
                "run_motion": ("BOOLEAN", {"default": False,
                               "tooltip": "Optical-flow motion-vector pass (needs an image batch ≥ 2 frames)."}),
                "depth_weight": wf(1.0), "canny_weight": wf(1.0),
                "pose_weight": wf(1.0), "normal_weight": wf(0.0),
                "match_to": (("depth", "canny", "pose", "image", "largest"), {"default": "image",
                                "tooltip": "Internal: align every AOV pass to this source's size before resize/output."}),
                # ── In-node resize / scale / smart-resize (real LayerStyle code, MIT —
                #    nodes/_vendored_resize.py). One dropdown: width/height OR scale. ──
                "resize": (_RESIZE_MODES, {"default": "off",
                                "tooltip": "Resize ALL outputs in-node. off = keep source size; "
                                           "width/height = resize to an exact width x height; scale = multiply size by 'scale'."}),
                "width": ("INT", {"default": 1024, "min": 0, "max": 16384, "step": 8,
                                "tooltip": "Target width (resize=width/height). 0 = derive from height + aspect."}),
                "height": ("INT", {"default": 1024, "min": 0, "max": 16384, "step": 8,
                                "tooltip": "Target height (resize=width/height). 0 = derive from width + aspect."}),
                "scale": ("FLOAT", {"default": 1.0, "min": 0.05, "max": 8.0, "step": 0.05,
                                "tooltip": "Scale factor applied to source size (resize=scale)."}),
                "divisible_by": ("INT", {"default": 16, "min": 1, "max": 512, "step": 1,
                                "tooltip": "Snap output W/H to a multiple (VAE-safe: 16=Wan, 64=SD). 1 = no snapping."}),
                "fit": (_RESIZE_FITS, {"default": "crop",
                                "tooltip": "Aspect handling: stretch = distort to fit; pad = letterbox with pad_color; "
                                           "crop = centre-crop to fill."}),
                "resize_filter": (_RESIZE_METHODS, {"default": "lanczos",
                                "tooltip": "Resample filter. lanczos = sharpest; nearest = hard pixels."}),
                "pad_color": ("STRING", {"default": "#000000",
                                "tooltip": "Pad colour (hex), used only when fit=pad."}),
            },
        }

    @classmethod
    def IS_CHANGED(cls, blend_mode, preview_layout="horizontal_3", image=None, depth_model="off", normal_model="off", depth_size="small",
                   depth_custom_ckpt="", pose_model="off", id_matte_model="off", edge_model="internal_canny",
                   run_canny=True, canny_low=100, canny_high=200, canny_aperture=3, depth_invert=False,
                   preproc_resolution=512, run_motion=False, depth=None, canny=None, pose=None,
                   normal=None, id_matte=None, depth_weight=1.0, canny_weight=1.0, pose_weight=1.0,
                   normal_weight=0.0, match_to="largest", **kwargs):
        return hash_args_and_kwargs(
            blend_mode, preview_layout, image, depth_model, normal_model, depth_size,
            depth_custom_ckpt, pose_model, id_matte_model, edge_model, run_canny, canny_low,
            canny_high, canny_aperture, depth_invert, preproc_resolution, run_motion, depth,
            canny, pose, normal, id_matte, depth_weight, canny_weight, pose_weight,
            normal_weight, match_to, **kwargs,
        )

    def _target_size(self, present, match_to):
        if not present:
            return 64, 64
        if match_to in present and present[match_to] is not None:
            t = present[match_to]
            return t.shape[1], t.shape[2]
        cands = [t for t in present.values() if t is not None]
        if not cands:
            return 64, 64
        best = max(cands, key=lambda t: t.shape[1] * t.shape[2])
        return best.shape[1], best.shape[2]

    def forge(self, blend_mode, preview_layout="horizontal_3", image=None, depth_model="off", normal_model="off", depth_size="small",
              depth_custom_ckpt="", pose_model="off", pose_ckpt="(auto)", id_matte_model="off", edge_model="internal_canny",
              run_canny=True, canny_low=100, canny_high=200, canny_aperture=3, depth_invert=False,
              preproc_resolution=512, run_motion=False, depth=None, canny=None, pose=None,
              normal=None, id_matte=None, depth_weight=1.0, canny_weight=1.0, pose_weight=1.0,
              normal_weight=0.0, match_to="largest", **kwargs):
        image = _as_bhwc3(image)
        notes = []
        res = int(preproc_resolution)
        # Normalise the dropdown sentinel: "(auto)" means "use the backend default".
        if depth_custom_ckpt in ("(auto)", None):
            depth_custom_ckpt = ""
        _pose_ckpt = "" if pose_ckpt in ("(auto)", None) else str(pose_ckpt)

        # DEPTH: external input wins; else vendored backend, else controlnet_aux delegation.
        depth_t = _as_bhwc3(depth)
        if depth_t is None and image is not None and depth_model != "off":
            if depth_model in _VENDORED_DEPTH and _cb is not None:
                if depth_model == "da_v2":
                    depth_t, nt = _cb.run_hf_depth(image, "v2", depth_size, depth_custom_ckpt)
                elif depth_model == "da_v1":
                    depth_t, nt = _cb.run_hf_depth(image, "v1", depth_size, depth_custom_ckpt)
                elif depth_model == "midas":
                    depth_t, nt = _cb.run_hf_depth(image, "midas", depth_size, depth_custom_ckpt)
                elif depth_model == "da3":
                    depth_t, nt = _cb.run_da3(image, depth_size, depth_custom_ckpt)
                elif depth_model == "depth_pro":
                    depth_t, nt = _cb.run_depth_pro(image)
                elif depth_model == "dvd":
                    depth_t, nt = _cb.run_dvd(image)
                else:  # depthcrafter
                    depth_t, nt = _cb.run_depthcrafter(image)
                notes.append(nt)
            elif depth_model in _DEPTH_MAP:  # controlnet_aux delegation
                ov = {}
                if depth_model in _DEPTH_CKPT:
                    ckpt = depth_custom_ckpt.strip() or _DEPTH_CKPT[depth_model].get(depth_size,
                            _DEPTH_CKPT[depth_model]["large"])
                    ov["ckpt_name"] = ckpt
                depth_t, nt = _run_aux(_DEPTH_MAP[depth_model], image, res, overrides=ov)
                notes.append(nt + (f"[{ov['ckpt_name']}]" if ov else ""))
                if depth_t is None and _cb is not None:
                    # controlnet_aux unavailable → fall back to the in-repo
                    # transformers DepthAnything-V2 runner so depth still works
                    # (no silent black output).
                    depth_t, nt2 = _cb.run_hf_depth(image, "v2", depth_size, "")
                    notes.append(f"[fallback->da_v2: {nt2}]")
        if depth_invert and depth_t is not None:
            depth_t = (1.0 - _as_bhwc3(depth_t)).clamp(0, 1)
        # POSE
        pose_t = _as_bhwc3(pose)
        if pose_t is None and image is not None and pose_model != "off":
            if pose_model == "vitpose":
                pose_t, nt = _run_vitpose(image, res, _pose_ckpt)
            else:
                pose_t, nt = _run_aux(_POSE_MAP[pose_model], image, res)
                if pose_t is None:
                    # controlnet_aux unavailable → fall back to our own ViTPose
                    # detector (no external preprocessor dependency).
                    pose_t, nt2 = _run_vitpose(image, res, "")
                    nt = f"{nt} [fallback->vitpose: {nt2}]"
            notes.append(nt)
        # EDGE: external canny input wins; else internal Canny or a delegated edge model.
        canny_t = _as_bhwc3(canny)
        if canny_t is None and image is not None:
            if edge_model == "internal_canny" and run_canny:
                canny_t = _canny_pass(image, canny_low, canny_high, canny_aperture)
            elif edge_model in _EDGE_MAP:
                ov = ({"low_threshold": int(canny_low), "high_threshold": int(canny_high)}
                      if edge_model == "canny" else None)
                canny_t, nt = _run_aux(_EDGE_MAP[edge_model], image, res, overrides=ov)
                notes.append(nt)
                if canny_t is None:
                    # controlnet_aux unavailable → fall back to internal OpenCV Canny.
                    canny_t = _canny_pass(image, canny_low, canny_high, canny_aperture)
                    notes.append("[fallback->internal_canny]")
        motion_t = _motion_pass(image) if (run_motion and image is not None) else None

        # NORMAL: external wins; else vendored backend (sobel-from-depth or NormalCrafter).
        normal_t = _as_bhwc3(normal)
        if normal_t is None and normal_model != "off" and _cb is not None:
            if normal_model == "sobel_from_depth" and depth_t is not None:
                normal_t = _cb.normals_from_depth(depth_t)
            elif normal_model == "normalcrafter" and image is not None:
                normal_t, nt = _cb.run_normalcrafter(image)
                notes.append(nt)

        # ID-MATTE: external wins; else automatic SAM segmentation.
        id_t = _as_bhwc3(id_matte)
        if id_t is None and image is not None and id_matte_model == "sam_auto":
            id_t, nt = _run_aux("SAMPreprocessor", image, res)
            notes.append(nt)

        raw = {"depth": depth_t, "canny": canny_t, "pose": pose_t,
               "normal": normal_t, "motion": _as_bhwc3(motion_t),
               "id_matte": id_t, "image": image}
        h, w = self._target_size(raw, match_to)
        present = [v for v in raw.values() if v is not None]
        B = max((t.shape[0] for t in present), default=1)

        def fit(t):
            if t is None:
                return torch.zeros(B, h, w, 3)
            t = _resize_to(t, h, w)
            if t.shape[0] == 1 and B > 1:
                t = t.repeat(B, 1, 1, 1)
            elif t.shape[0] != B:
                t = t[torch.arange(B) % t.shape[0]]
            return t

        norm = {k: fit(v) for k, v in raw.items()}
        weights = {"depth": float(depth_weight), "canny": float(canny_weight),
                   "pose": float(pose_weight), "normal": float(normal_weight)}

        active = [(k, norm[k] * weights[k]) for k in ("depth", "canny", "pose", "normal")
                  if raw[k] is not None and weights[k] > 0]
        if not active:
            blended = torch.zeros(B, h, w, 3)
        elif blend_mode == "screen":
            acc = torch.zeros(B, h, w, 3)
            for _k, m in active:
                acc = 1.0 - (1.0 - acc) * (1.0 - m.clamp(0, 1))
            blended = acc
        elif blend_mode == "lighten_max":
            blended = active[0][1].clone()
            for _k, m in active[1:]:
                blended = torch.maximum(blended, m)
        elif blend_mode == "linear_dodge":
            blended = sum(m for _k, m in active)
        elif blend_mode == "multiply":
            blended = active[0][1].clone()
            for _k, m in active[1:]:
                blended = blended * m
        elif blend_mode == "weighted_avg":
            tw = sum(weights[k] for k, _ in active) or 1.0
            blended = sum(m for _k, m in active) / tw
        elif blend_mode == "overlay":
            blended = active[0][1].clone()
            for _k, m in active[1:]:
                lo = 2.0 * blended * m
                hi = 1.0 - 2.0 * (1.0 - blended) * (1.0 - m)
                blended = torch.where(blended < 0.5, lo, hi)
        else:  # average
            blended = sum(m for _k, m in active) / float(len(active))
        blended = blended.clamp(0, 1).contiguous()

        packed = torch.stack([_luma(norm["depth"]), _luma(norm["canny"]), _luma(norm["pose"])], dim=-1).clamp(0, 1).contiguous()

        # 'combined' panel — see depth + pose + canny in ONE image, side-by-side/grid.
        d3, p3, c3 = norm["depth"], norm["pose"], norm["canny"]
        orig3 = norm["image"]
        if preview_layout == "vertical_3":
            combined = torch.cat([d3, p3, c3], dim=1)
        elif preview_layout == "grid_2x2":
            top = torch.cat([d3, p3], dim=2)
            bot = torch.cat([c3, orig3], dim=2)
            combined = torch.cat([top, bot], dim=1)
        else:  # horizontal_3
            combined = torch.cat([d3, p3, c3], dim=2)
        combined = combined.clamp(0, 1).contiguous()

        # Per-pass health so you can SEE all 3 are working (not silently dropped).
        requested = {
            "depth": (depth_model != "off") or (depth is not None),
            "canny": (edge_model != "off") or (canny is not None),
            "pose": (pose_model != "off") or (pose is not None),
            "normal": (normal_model != "off") or (normal is not None),
        }

        def _stat(k):
            if raw[k] is None:
                return "FAIL" if requested.get(k) else "off"
            try:
                mx = float(norm[k].max())
            except Exception:
                mx = 0.0
            return "OK" if mx > 0.02 else "EMPTY"

        core = {k: _stat(k) for k in ("depth", "canny", "pose")}
        core_ok = all(v == "OK" for v in core.values())
        status = ("ALL 3 OK" if core_ok else "CHECK BELOW") + \
            " | " + " ".join(f"{k}={v}" for k, v in core.items()) + \
            f" | normal={_stat('normal')} id={_stat('id_matte') if raw['id_matte'] is not None else 'off'}"

        names = [k for k in ("depth", "canny", "pose", "normal", "motion", "id_matte") if raw[k] is not None]
        run_notes = " ".join(n for n in notes if n)
        info = (
            f"STATUS: {status}\n"
            f"Control AOV — passes: {', '.join(names) or 'none'} | blend: {blend_mode} | {w}x{h} x{B}\n"
            + (f"backends: {run_notes}\n" if run_notes else "")
            + "VENDORED depth (self-contained, weights download on first use): da_v2/da_v1/midas (transformers), "
              "da3 (Depth-Anything-3), depth_pro, depthcrafter (video). VENDORED normal: sobel_from_depth, normalcrafter. "
              "DELEGATED to comfyui_controlnet_aux: depth_anything_* depth, DWPose/OpenPose pose, Canny/LineArt/HED/etc. "
              "Wire as INPUTS: ViTPose (WanV2 detect->draw), ID-matte (SAM).\n"
              "MAX CONTROL / no-drift: don't rely on 'blended'. Stack the separate AOV passes into Apply-ControlNet "
              "(or a union ControlNet per-type) at STAGGERED weights (~0.6-0.9, not equal) with start/end-step "
              "scheduling, and add a Tile ControlNet to lock layout. 'channel_packed' suits union nets."
        )
        # ── Output sizing: resize EVERY AOV with the real LayerStyle scale code ──
        out_imgs = [blended, combined, packed, norm["depth"], norm["canny"], norm["pose"],
                    norm["normal"], norm["motion"], norm["id_matte"]]
        _resize_mode = str(kwargs.get("resize", "off"))
        if _resize_mode != "off" and _HAVE_RESIZE:
            try:
                rk = dict(
                    mode=_resize_mode,
                    width=int(kwargs.get("width", 1024)),
                    height=int(kwargs.get("height", 1024)),
                    scale=float(kwargs.get("scale", 1.0)),
                    divisible_by=int(kwargs.get("divisible_by", 16)),
                    fit=str(kwargs.get("fit", "crop")),
                    method=str(kwargs.get("resize_filter", "lanczos")),
                    pad_color=str(kwargs.get("pad_color", "#000000")),
                )
                resized, rtw, rth = [], None, None
                for t in out_imgs:
                    if t is None:
                        resized.append(None); continue
                    rt, rtw, rth = _resize_wh(t, **rk)
                    resized.append(rt)
                out_imgs = resized
                if rtw:
                    _by = f"scale x{rk['scale']}" if _resize_mode == "scale" else f"{rk['width']}x{rk['height']}"
                    info += f"\nOutput resized -> {rtw}x{rth} ({_by}, fit={rk['fit']}, /{rk['divisible_by']})."
            except Exception as _rz_exc:  # noqa: BLE001 - resize must never break the forge
                info += f"\n[output resize skipped: {_rz_exc}]"

        return (out_imgs[0], out_imgs[1], out_imgs[2], out_imgs[3], out_imgs[4],
                out_imgs[5], out_imgs[6], out_imgs[7], out_imgs[8], info)


NODE_CLASS_MAPPINGS = {"ControlAOVC2C": ControlAOVC2C}
NODE_DISPLAY_NAME_MAPPINGS = {"ControlAOVC2C": "Control AOV — Multi-Control Fusion (C2C)"}
