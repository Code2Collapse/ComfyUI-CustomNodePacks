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
# The depthcrafter/ and normalcrafter/ inference packages live directly in this
# repo (nodes/depthcrafter, nodes/normalcrafter) so they work on every machine —
# third_party/ is gitignored and only exists on the dev box. Put this nodes/ dir
# on sys.path so `import depthcrafter` / `import normalcrafter` resolve in-repo.
_NODES_DIR = os.path.dirname(__file__)
if _NODES_DIR not in sys.path:
    sys.path.insert(0, _NODES_DIR)
# Repo-dir name (third_party) → in-repo package name, for runners shipped in-repo.
# DA3 / depth-pro are too large to ship in-repo → third_party clone only.
_IN_REPO = {"DepthCrafter": "depthcrafter", "NormalCrafter": "normalcrafter"}
_CACHE = {}


# ----------------------------------------------------------------- helpers
def _ensure_path(rel):
    # Prefer a third_party clone if present (dev machines).
    p = os.path.join(_THIRD_PARTY, rel)
    if os.path.isdir(p):
        if p not in sys.path:
            sys.path.insert(0, p)
        return True
    # Otherwise use the in-repo package (nodes/ is already on sys.path).
    pkg = _IN_REPO.get(rel)
    if pkg and os.path.isdir(os.path.join(_NODES_DIR, pkg)):
        return True
    return False


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


def _comfy_models_dir(sub):
    """Path to ComfyUI/models/<sub> if resolvable, else ''."""
    try:
        import folder_paths  # type: ignore
        return os.path.join(folder_paths.models_dir, sub)
    except Exception:  # noqa: BLE001
        return ""


def _write_mp4(image_bhwc, path, fps=16):
    """Write a [T,H,W,3] 0..1 batch to an mp4 (RGB→BGR). Used for video-model roundtrips."""
    import cv2
    arr = (image_bhwc[..., :3].detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    T, H, W = arr.shape[0], arr.shape[1], arr.shape[2]
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (W, H))
    if not vw.isOpened():
        raise RuntimeError("cv2.VideoWriter could not open mp4 writer")
    for i in range(T):
        vw.write(cv2.cvtColor(arr[i], cv2.COLOR_RGB2BGR))
    vw.release()


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


# ----------------------------------------------------------------- DVD (Wan2.1 deterministic video depth)
def run_dvd(image_bhwc, window_size=81, overlap=21, height=480, width=640, ckpt_dir=""):
    """DVD — EnVision-Research/DVD: deterministic single-pass *video* depth built on
    Wan 2.1 (DiffSynth). Relative depth, temporally stable.

    Code: third_party/DVD (Apache-2.0) — too large to ship in-repo (like DA3 /
    depth-pro), so it is a third_party clone that must be synced to the Linux box.
    Weights: HF ``FayeHongfeiZhang/DVD`` (model_config.yaml + model.safetensors,
    **CC BY-NC 4.0 — non-commercial only**). Place them in ComfyUI/models/DVD/ or
    set DVD_CKPT_DIR. Heavy: needs DiffSynth + the Wan 2.1 base model + a big GPU.
    """
    image_bhwc = _bhwc(image_bhwc)
    if image_bhwc is None:
        return None, "dvd: no image"
    dvd_root = os.path.join(_THIRD_PARTY, "DVD")
    if not os.path.isdir(dvd_root):
        return None, ("dvd: third_party/DVD missing — clone EnVision-Research/DVD into "
                      "ComfyUI-CustomNodePacks/third_party/DVD (too large to ship in-repo; sync to Linux).")
    # Resolve the checkpoint dir (must contain model_config.yaml + model.safetensors).
    cands = [c for c in [
        (ckpt_dir or "").strip(),
        os.environ.get("DVD_CKPT_DIR", ""),
        _comfy_models_dir("DVD"),
        os.path.join(dvd_root, "ckpt"),
    ] if c]
    def _has_ckpt(c):
        return (c and os.path.isfile(os.path.join(c, "model.safetensors"))
                and os.path.isfile(os.path.join(c, "model_config.yaml")))
    ckpt = next((c for c in cands if _has_ckpt(c)), None)
    if ckpt is None:
        # AUTO-DOWNLOAD the DVD checkpoint from HF (CC BY-NC 4.0 — non-commercial
        # use only). One-time, ~GBs. Lands in ComfyUI/models/DVD (or repo ckpt/).
        dest = _comfy_models_dir("DVD") or os.path.join(dvd_root, "ckpt")
        try:
            from huggingface_hub import snapshot_download
            os.makedirs(dest, exist_ok=True)
            print(f"[control_aov] DVD weights not found -> auto-downloading "
                  f"FayeHongfeiZhang/DVD into {dest} (one-time, large; CC BY-NC 4.0 "
                  f"non-commercial) ...", flush=True)
            snapshot_download(
                repo_id="FayeHongfeiZhang/DVD", local_dir=dest,
                allow_patterns=["model_config.yaml", "model.safetensors"],
            )
            ckpt = dest if _has_ckpt(dest) else None
        except Exception as exc:  # noqa: BLE001
            return None, (f"dvd: auto-download failed ({type(exc).__name__}: {exc}). "
                          f"Install huggingface_hub / check network, or manually place "
                          f"FayeHongfeiZhang/DVD weights in ComfyUI/models/DVD/.")
        if ckpt is None:
            return None, "dvd: weights still missing after auto-download attempt."
    try:
        # DVD imports `diffsynth`, `examples` and test_script/test_single_video —
        # put the repo root + test_script on sys.path.
        for p in (dvd_root, os.path.join(dvd_root, "test_script")):
            if p not in sys.path:
                sys.path.insert(0, p)
        from omegaconf import OmegaConf
        from argparse import Namespace
        import test_single_video as _dvd  # type: ignore
        if "dvd" not in _CACHE:
            yaml_args = OmegaConf.load(os.path.join(ckpt, "model_config.yaml"))
            _CACHE["dvd"] = _dvd.load_model(ckpt, yaml_args)
        model = _CACHE["dvd"]
        # Temp-video roundtrip so DVD's own read_video + resize preprocessing is used
        # verbatim (least error-prone, mirrors the depthcrafter pattern).
        import tempfile
        tmp = os.path.join(tempfile.gettempdir(), "c2c_dvd_in.mp4")
        _write_mp4(image_bhwc, tmp, fps=16)
        try:
            input_tensor, orig_size, _fps = _dvd.load_video_data(
                Namespace(input_video=tmp, height=int(height), width=int(width)))
            depth = _dvd.predict_depth(
                model, input_tensor, orig_size,
                Namespace(window_size=int(window_size), overlap=int(overlap)))
        finally:
            try:
                os.remove(tmp)
            except Exception:  # noqa: BLE001
                pass
        # depth: [T,H,W,1] float (unnormalised) → BHWC 3-channel, normalised 0..1.
        d = torch.as_tensor(np.asarray(depth)).float()
        if d.dim() == 4 and d.shape[-1] == 1:
            d = d[..., 0]
        while d.dim() > 3:
            d = d[0]
        mn, mx = float(d.min()), float(d.max())
        if mx - mn > 1e-6:
            d = (d - mn) / (mx - mn)
        d = d.unsqueeze(-1).repeat(1, 1, 1, 3).clamp(0, 1)
        return d, f"dvd ok [{os.path.basename(ckpt.rstrip(os.sep)) or ckpt}]"
    except Exception as e:  # noqa: BLE001
        return None, f"dvd err: {type(e).__name__}: {e}"


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
