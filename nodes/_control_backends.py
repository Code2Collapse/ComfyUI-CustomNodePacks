"""
_control_backends.py — vendored depth/normal model runners for Control AOV.

Self-contained inference for the heavy backends, so Control AOV runs them itself
instead of depending on other ComfyUI node packs. Every backend is:
  - LAZY: nothing imports a model or downloads weights until its run_* fn is called.
  - GUARDED: a missing dependency / weight / repo returns (None, "note") instead of
    raising, so the node (and the whole pack) never breaks — even across ComfyUI or
    package updates.
  - CACHED: each model loads once and is reused.

Sources (cloned into ../third_party by request, or via transformers/HF):
  - DepthAnything V1/V2 + Midas/DPT : transformers `pipeline("depth-estimation")` (faithful, stable API).
  - DepthAnything V3                : third_party/Depth-Anything-3 (depth_anything_3.api.DepthAnything3).
  - Depth-Pro                       : third_party/ml-depth-pro (depth_pro.create_model_and_transforms / model.infer).
  - DepthCrafter (video depth)      : third_party/DepthCrafter (DepthCrafterPipeline, in-memory [b,c,h,w]).
  - NormalCrafter (video normals)   : third_party/NormalCrafter (NormalCrafterPipeline).

Tensors in/out are ComfyUI IMAGE convention: [B,H,W,3] float32 0..1.
NOTE: the repo-based runners (DA3/DepthPro/DepthCrafter/NormalCrafter) are written
faithfully to each repo's documented API but are pending a live GPU verification.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

_THIRD_PARTY = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "third_party"))
_CACHE = {}


# ----------------------------------------------------------------- helpers
def _ensure_path(rel):
    p = os.path.join(_THIRD_PARTY, rel)
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)
    return os.path.isdir(p)


def _bhwc(t):
    if t is None:
        return None
    if t.dim() == 3:
        t = t.unsqueeze(0)
    return t


def _to_pil_list(image_bhwc):
    from PIL import Image
    out = []
    for i in range(image_bhwc.shape[0]):
        a = (image_bhwc[i, ..., :3].detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        out.append(Image.fromarray(a))
    return out


def _gray_to_bhwc3(arr_or_t):
    """A single-channel depth (numpy or tensor, any range) → [1,H,W,3] 0..1, normalised."""
    t = torch.as_tensor(np.asarray(arr_or_t), dtype=torch.float32)
    while t.dim() > 2:
        t = t.squeeze(0) if t.shape[0] == 1 else t[..., 0]
    mn, mx = float(t.min()), float(t.max())
    if mx - mn > 1e-6:
        t = (t - mn) / (mx - mn)
    return t.unsqueeze(0).unsqueeze(-1).repeat(1, 1, 1, 3).clamp(0, 1)


# ----------------------------------------------------------------- transformers depth
_HF_DEPTH_MODELS = {
    ("v2", "small"): "depth-anything/Depth-Anything-V2-Small-hf",
    ("v2", "base"): "depth-anything/Depth-Anything-V2-Base-hf",
    ("v2", "large"): "depth-anything/Depth-Anything-V2-Large-hf",
    ("v1", "small"): "LiheYoung/depth-anything-small-hf",
    ("v1", "base"): "LiheYoung/depth-anything-base-hf",
    ("v1", "large"): "LiheYoung/depth-anything-large-hf",
    ("midas", "small"): "Intel/dpt-hybrid-midas",
    ("midas", "base"): "Intel/dpt-hybrid-midas",
    ("midas", "large"): "Intel/dpt-large",
}


def run_hf_depth(image_bhwc, version, size, custom_model=""):
    """DepthAnything v1/v2 or Midas/DPT via transformers. version in {v1,v2,midas}."""
    image_bhwc = _bhwc(image_bhwc)
    if image_bhwc is None:
        return None, "hf_depth: no image"
    model_id = custom_model.strip() or _HF_DEPTH_MODELS.get((version, size)) \
        or _HF_DEPTH_MODELS.get((version, "small"))
    if not model_id:
        return None, f"hf_depth: no model for {version}/{size}"
    key = "hf_depth::" + model_id
    try:
        if key not in _CACHE:
            from transformers import pipeline
            dev = 0 if torch.cuda.is_available() else -1
            _CACHE[key] = pipeline("depth-estimation", model=model_id, device=dev)
        pipe = _CACHE[key]
        frames = []
        for pil in _to_pil_list(image_bhwc):
            d = pipe(pil)["depth"]  # PIL grayscale
            frames.append(_gray_to_bhwc3(np.array(d)))
        return torch.cat(frames, 0), f"hf_depth ok [{model_id}]"
    except Exception as e:  # missing weights / transformers issue
        return None, f"hf_depth err [{model_id}]: {type(e).__name__}: {e}"


# ----------------------------------------------------------------- DepthAnything V3 (repo)
_DA3_MODELS = {
    "small": "depth-anything/DA3METRIC-LARGE",   # repo ships nested variants; LARGE is the common one
    "base": "depth-anything/DA3METRIC-LARGE",
    "large": "depth-anything/DA3NESTED-GIANT-LARGE",
    "giant": "depth-anything/DA3NESTED-GIANT-LARGE",
}


def run_da3(image_bhwc, size="large", custom_model=""):
    image_bhwc = _bhwc(image_bhwc)
    if image_bhwc is None:
        return None, "da3: no image"
    if not _ensure_path(os.path.join("Depth-Anything-3", "src")) and not _ensure_path("Depth-Anything-3"):
        return None, "da3: third_party/Depth-Anything-3 missing"
    model_id = custom_model.strip() or _DA3_MODELS.get(size, _DA3_MODELS["large"])
    key = "da3::" + model_id
    try:
        if key not in _CACHE:
            from depth_anything_3.api import DepthAnything3
            m = DepthAnything3.from_pretrained(model_id)
            if torch.cuda.is_available():
                try: m = m.to("cuda")
                except Exception: pass
            _CACHE[key] = m
        model = _CACHE[key]
        frames = []
        for pil in _to_pil_list(image_bhwc):
            pred = model.inference([pil])  # API takes a list of views
            depth = getattr(pred, "depth", None)
            if depth is None and isinstance(pred, dict):
                depth = pred.get("depth")
            frames.append(_gray_to_bhwc3(depth.detach().cpu().numpy() if torch.is_tensor(depth) else depth))
        return torch.cat(frames, 0), f"da3 ok [{model_id}]"
    except Exception as e:
        return None, f"da3 err [{model_id}]: {type(e).__name__}: {e}"


# ----------------------------------------------------------------- Depth-Pro (repo)
def run_depth_pro(image_bhwc):
    image_bhwc = _bhwc(image_bhwc)
    if image_bhwc is None:
        return None, "depth_pro: no image"
    if not _ensure_path(os.path.join("ml-depth-pro", "src")) and not _ensure_path("ml-depth-pro"):
        return None, "depth_pro: third_party/ml-depth-pro missing"
    try:
        if "depth_pro" not in _CACHE:
            import depth_pro
            model, transform = depth_pro.create_model_and_transforms()
            model.eval()
            if torch.cuda.is_available():
                try: model = model.to("cuda")
                except Exception: pass
            _CACHE["depth_pro"] = (model, transform)
        model, transform = _CACHE["depth_pro"]
        frames = []
        for pil in _to_pil_list(image_bhwc):
            img = transform(pil)
            if torch.cuda.is_available():
                try: img = img.to("cuda")
                except Exception: pass
            pred = model.infer(img)
            depth = pred["depth"] if isinstance(pred, dict) else pred
            frames.append(_gray_to_bhwc3(depth.detach().cpu().numpy()))
        return torch.cat(frames, 0), "depth_pro ok"
    except Exception as e:
        return None, f"depth_pro err: {type(e).__name__}: {e}"


# ----------------------------------------------------------------- DepthCrafter (video depth)
def run_depthcrafter(image_bhwc, num_inference_steps=5, guidance_scale=1.0,
                     unet_path="tencent/DepthCrafter", base="stabilityai/stable-video-diffusion-img2vid-xt"):
    image_bhwc = _bhwc(image_bhwc)
    if image_bhwc is None:
        return None, "depthcrafter: no image"
    if not _ensure_path("DepthCrafter"):
        return None, "depthcrafter: third_party/DepthCrafter missing"
    try:
        if "depthcrafter" not in _CACHE:
            from depthcrafter.depth_crafter_ppl import DepthCrafterPipeline
            from diffusers import UNetSpatioTemporalConditionModel
            unet = UNetSpatioTemporalConditionModel.from_pretrained(
                unet_path, low_cpu_mem_usage=True, torch_dtype=torch.float16)
            pipe = DepthCrafterPipeline.from_pretrained(
                base, unet=unet, torch_dtype=torch.float16, variant="fp16")
            if torch.cuda.is_available():
                pipe = pipe.to("cuda")
            _CACHE["depthcrafter"] = pipe
        pipe = _CACHE["depthcrafter"]
        # pipeline __call__ expects video [b,c,h,w] in [-1,1]
        vid = (image_bhwc[..., :3].permute(0, 3, 1, 2) * 2.0 - 1.0)
        if torch.cuda.is_available():
            vid = vid.to("cuda").half()
        res = pipe(vid, num_inference_steps=int(num_inference_steps),
                   guidance_scale=float(guidance_scale))
        depth = res.frames if hasattr(res, "frames") else res
        depth = torch.as_tensor(np.asarray(depth)).float()  # [B,H,W] or [B,H,W,C]
        if depth.dim() == 3:
            depth = depth.unsqueeze(-1).repeat(1, 1, 1, 3)
        return depth.clamp(0, 1), "depthcrafter ok"
    except Exception as e:
        return None, f"depthcrafter err: {type(e).__name__}: {e}"


# ----------------------------------------------------------------- NormalCrafter (video normals)
def run_normalcrafter(image_bhwc, unet_path="Yanrui95/NormalCrafter",
                      base="stabilityai/stable-video-diffusion-img2vid-xt"):
    image_bhwc = _bhwc(image_bhwc)
    if image_bhwc is None:
        return None, "normalcrafter: no image"
    if not _ensure_path("NormalCrafter"):
        return None, "normalcrafter: third_party/NormalCrafter missing"
    try:
        if "normalcrafter" not in _CACHE:
            from normalcrafter.normal_crafter_ppl import NormalCrafterPipeline
            from normalcrafter.unet import DiffusersUNetSpatioTemporalConditionModelNormalCrafter
            from diffusers import AutoencoderKLTemporalDecoder
            unet = DiffusersUNetSpatioTemporalConditionModelNormalCrafter.from_pretrained(
                unet_path, subfolder="unet", low_cpu_mem_usage=True, torch_dtype=torch.float16)
            vae = AutoencoderKLTemporalDecoder.from_pretrained(unet_path, subfolder="vae", torch_dtype=torch.float16)
            pipe = NormalCrafterPipeline.from_pretrained(base, unet=unet, vae=vae, torch_dtype=torch.float16, variant="fp16")
            if torch.cuda.is_available():
                pipe = pipe.to("cuda")
            _CACHE["normalcrafter"] = pipe
        pipe = _CACHE["normalcrafter"]
        frames = image_bhwc[..., :3].detach().cpu().numpy()  # [B,H,W,3] 0..1
        res = pipe.infer(frames, window_size=14, time_step_size=10)
        normals = torch.as_tensor(np.asarray(res)).float()   # [B,H,W,3] in [-1,1] or 0..1
        if normals.min() < -0.01:
            normals = (normals + 1.0) * 0.5
        return normals.clamp(0, 1), "normalcrafter ok"
    except Exception as e:
        return None, f"normalcrafter err: {type(e).__name__}: {e}"


# ----------------------------------------------------------------- normals from depth (no model)
def normals_from_depth(depth_bhwc):
    """Sobel-from-depth Lambertian normals — no model, always works."""
    d = _bhwc(depth_bhwc)
    if d is None:
        return None
    g = (d[..., 0] if d.shape[-1] >= 1 else d).unsqueeze(1)  # [B,1,H,W]
    import torch.nn.functional as F
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
    ky = kx.transpose(2, 3).contiguous()
    gx = F.conv2d(F.pad(g, (1, 1, 1, 1), mode="replicate"), kx)
    gy = F.conv2d(F.pad(g, (1, 1, 1, 1), mode="replicate"), ky)
    nz = torch.ones_like(gx)
    n = torch.cat([-gx, -gy, nz], dim=1)
    n = n / (n.norm(dim=1, keepdim=True) + 1e-6)
    n = (n * 0.5 + 0.5).permute(0, 2, 3, 1)  # [B,H,W,3] 0..1
    return n.clamp(0, 1).contiguous()
