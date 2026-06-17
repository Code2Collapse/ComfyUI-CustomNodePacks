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

import hashlib

import numpy as np
import torch
import torch.nn.functional as F

try:
    import cv2
    _HAVE_CV2 = True
except Exception:
    _HAVE_CV2 = False

_LUMA = (0.2126, 0.7152, 0.0722)
_BLEND_MODES = ("screen", "lighten_max", "linear_dodge", "multiply", "average")


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
    "lineart": "LineArtPreprocessor",
    "anyline": "AnyLineArtPreprocessor_aux",
    "hed": "HEDPreprocessor",
    "pidinet": "PiDiNetPreprocessor",
    "teed": "TEEDPreprocessor",
}


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


# ----------------------------------------------------------------- vendored passes
def _canny_pass(image_bhwc, low, high):
    """OpenCV Canny per frame → white edges on black, [B,H,W,3]."""
    if not _HAVE_CV2:
        return None
    out = []
    for i in range(image_bhwc.shape[0]):
        g = cv2.cvtColor(_to_u8(image_bhwc[i]), cv2.COLOR_RGB2GRAY)
        e = cv2.Canny(g, int(low), int(high))
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
class OmniControlForgeMEC:
    CATEGORY = "C2C/Control"
    DESCRIPTION = ("VFX-AOV control fusion: emit depth/canny/pose/normal/motion/ID as separate passes, "
                   "a channel-packed image (depth=R/canny=G/pose=B), and a convenience blend. Runs Canny + "
                   "optical-flow motion internally; accepts depth/pose/normal/ID maps as inputs. Feeds any "
                   "ControlNet / union / control-video. Stack the separate passes for maximum spatial lock.")
    FUNCTION = "forge"
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("blended", "channel_packed", "depth", "canny", "pose", "normal", "motion", "id_matte", "info")
    OUTPUT_TOOLTIPS = (
        "Convenience single image (chosen blend mode).",
        "depth=R, canny=G, pose=B — lossless packing for union ControlNets.",
        "Depth AOV (passthrough).", "Canny AOV (run internally if 'image' wired).",
        "Pose AOV (passthrough).", "Normal AOV (passthrough).",
        "Motion-vector AOV (optical flow, run internally on a frame batch).",
        "ID / segmentation matte AOV (passthrough).",
        "Present passes + recommended max-control wiring.",
    )

    @classmethod
    def INPUT_TYPES(cls):
        wf = lambda d: ("FLOAT", {"default": d, "min": 0.0, "max": 2.0, "step": 0.05})
        return {
            "required": {
                "blend_mode": (_BLEND_MODES, {"default": "screen",
                               "tooltip": "screen = least-destructive default; linear_dodge clips; multiply darkens."}),
            },
            "optional": {
                "image": ("IMAGE", {"tooltip": "Source frames — preprocessors below run on this."}),
                "depth_model": (["off"] + list(_DEPTH_MAP.keys()), {"default": "off",
                                "tooltip": "Run a depth preprocessor internally (delegates to comfyui_controlnet_aux, "
                                           "using its own models/paths). 'off' = wire an external depth map instead."}),
                "pose_model": (["off"] + list(_POSE_MAP.keys()), {"default": "off",
                               "tooltip": "Run a pose preprocessor internally (DWPose/OpenPose/...)."}),
                "edge_model": (["internal_canny", "off"] + list(_EDGE_MAP.keys()), {"default": "internal_canny",
                               "tooltip": "internal_canny = OpenCV (no model). Others delegate to controlnet_aux."}),
                "run_canny": ("BOOLEAN", {"default": True, "tooltip": "Used only when edge_model = internal_canny."}),
                "canny_low": ("INT", {"default": 100, "min": 0, "max": 255}),
                "canny_high": ("INT", {"default": 200, "min": 0, "max": 255}),
                "preproc_resolution": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 8,
                               "tooltip": "Resolution passed to delegated preprocessors."}),
                "run_motion": ("BOOLEAN", {"default": False,
                               "tooltip": "Optical-flow motion-vector pass (needs an image batch ≥ 2 frames)."}),
                "depth": ("IMAGE", {"tooltip": "Depth map (DepthAnything/DepthCrafter/ZoeDepth)."}),
                "canny": ("IMAGE", {"tooltip": "External edge map; overrides internal Canny if provided."}),
                "pose": ("IMAGE", {"tooltip": "Pose render (DWPose/OpenPose/ViTPose)."}),
                "normal": ("IMAGE", {"tooltip": "Surface normals (NormalCrafter)."}),
                "id_matte": ("IMAGE", {"tooltip": "Segmentation/ID matte (SAM/cryptomatte-style)."}),
                "depth_weight": wf(1.0), "canny_weight": wf(1.0),
                "pose_weight": wf(1.0), "normal_weight": wf(0.0),
                "match_to": (("depth", "canny", "pose", "image", "largest"), {"default": "largest"}),
            },
        }

    @classmethod
    def IS_CHANGED(cls, blend_mode, image=None, depth_model="off", pose_model="off",
                   edge_model="internal_canny", run_canny=True, canny_low=100, canny_high=200,
                   preproc_resolution=512, run_motion=False, depth=None, canny=None, pose=None,
                   normal=None, id_matte=None, depth_weight=1.0, canny_weight=1.0,
                   pose_weight=1.0, normal_weight=0.0, match_to="largest", **_):
        h = hashlib.md5()
        h.update(repr((blend_mode, depth_model, pose_model, edge_model, run_canny, canny_low,
                       canny_high, preproc_resolution, run_motion, depth_weight, canny_weight,
                       pose_weight, normal_weight, match_to)).encode())
        for nm, t in (("i", image), ("d", depth), ("c", canny), ("p", pose), ("n", normal), ("m", id_matte)):
            h.update(nm.encode() if t is None else t.detach().cpu().numpy().tobytes())
        return h.hexdigest()

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

    def forge(self, blend_mode, image=None, depth_model="off", pose_model="off",
              edge_model="internal_canny", run_canny=True, canny_low=100, canny_high=200,
              preproc_resolution=512, run_motion=False, depth=None, canny=None, pose=None,
              normal=None, id_matte=None, depth_weight=1.0, canny_weight=1.0,
              pose_weight=1.0, normal_weight=0.0, match_to="largest"):
        image = _as_bhwc3(image)
        notes = []
        res = int(preproc_resolution)

        # DEPTH: external input wins; else delegate to the chosen preprocessor.
        depth_t = _as_bhwc3(depth)
        if depth_t is None and image is not None and depth_model != "off":
            depth_t, nt = _run_aux(_DEPTH_MAP[depth_model], image, res)
            notes.append(nt)
        # POSE
        pose_t = _as_bhwc3(pose)
        if pose_t is None and image is not None and pose_model != "off":
            pose_t, nt = _run_aux(_POSE_MAP[pose_model], image, res)
            notes.append(nt)
        # EDGE: external canny input wins; else internal Canny or a delegated edge model.
        canny_t = _as_bhwc3(canny)
        if canny_t is None and image is not None:
            if edge_model == "internal_canny" and run_canny:
                canny_t = _canny_pass(image, canny_low, canny_high)
            elif edge_model in _EDGE_MAP:
                canny_t, nt = _run_aux(_EDGE_MAP[edge_model], image, res)
                notes.append(nt)
        motion_t = _motion_pass(image) if (run_motion and image is not None) else None

        raw = {"depth": depth_t, "canny": canny_t, "pose": pose_t,
               "normal": _as_bhwc3(normal), "motion": _as_bhwc3(motion_t),
               "id_matte": _as_bhwc3(id_matte), "image": image}
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
        else:
            blended = sum(m for _k, m in active) / float(len(active))
        blended = blended.clamp(0, 1).contiguous()

        packed = torch.stack([_luma(norm["depth"]), _luma(norm["canny"]), _luma(norm["pose"])], dim=-1).clamp(0, 1).contiguous()

        names = [k for k in ("depth", "canny", "pose", "normal", "motion", "id_matte") if raw[k] is not None]
        run_notes = " ".join(n for n in notes if n)
        info = (
            f"OmniControl Forge — passes: {', '.join(names) or 'none'} | blend: {blend_mode} | {w}x{h} x{B}\n"
            + (f"backends: {run_notes}\n" if run_notes else "")
            + "Internal (delegated to comfyui_controlnet_aux, its own model paths): DepthAnything v1/v2/metric/zoe, "
              "DWPose/OpenPose/Animal/DensePose, Canny/LineArt/HED/PiDiNet/TEED. "
              "Wire as INPUTS (not in controlnet_aux / not installed): DepthAnything V3, ViTPose, DepthCrafter, "
              "NormalCrafter, ID-matte (SAM).\n"
              "MAX CONTROL / no-drift: don't rely on 'blended'. Stack the separate AOV passes into Apply-ControlNet "
              "(or a union ControlNet per-type) at STAGGERED weights (~0.6-0.9, not equal) with start/end-step "
              "scheduling, and add a Tile ControlNet to lock layout. 'channel_packed' suits union nets."
        )
        return (blended, packed, norm["depth"], norm["canny"], norm["pose"],
                norm["normal"], norm["motion"], norm["id_matte"], info)


NODE_CLASS_MAPPINGS = {"OmniControlForgeMEC": OmniControlForgeMEC}
NODE_DISPLAY_NAME_MAPPINGS = {"OmniControlForgeMEC": "OmniControl Forge — VFX Control Fusion (C2C)"}
