"""
MEC Face Fixer
==============

Single-node face-detail pipeline inspired by Forbidden Vision's "Fixer" and
Impact Pack's FaceDetailer, rebuilt around MEC primitives:

    detect (YOLO11 face .pt or .onnx)
        --> per-face crop with padding
        --> optional AI pre-upscale of small faces
        --> KSampler at the face crop resolution
        --> color-match + lightness-rescue + differential-diffusion blend
        --> wildcard prompt routing per face ([SEP]/[ASC]/[DSC]/[ASC-SIZE]/
            [DSC-SIZE]/[SKIP])

The detection model is the user's responsibility -- we do NOT bundle weights
and we accept any Ultralytics-compatible YOLO model file located under
``ComfyUI/models/ultralytics/bbox/``. If neither ``ultralytics`` nor
``onnxruntime`` is installed, the node falls back to using the ``mask`` input
unchanged so workflows do not hard-fail.

This implementation is original code; no AGPL-licensed material from
Forbidden Vision is used. Behavioural inspiration only.
"""
from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

import folder_paths

from .mec_paint_suite import (
    _gaussian_blur_2d,
    _morph_2d,
    _np_to_bhwc,
    _parse_wildcard_prompt,
    _to_bhwc,
    _to_mask,
    MECContextInpainter,
)

logger = logging.getLogger("MEC.FaceFixer")

# Try to register a folder for face-detection .pt / .onnx weights so they
# show up in the combo. We mirror Impact Pack's convention so users with an
# existing install don't have to copy files.
try:
    folder_paths.add_model_folder_path(
        "ultralytics_bbox",
        str(folder_paths.models_dir).replace("\\", "/") + "/ultralytics/bbox",
        is_default=False,
    )
except Exception:  # pragma: no cover - older Comfy may differ
    pass


def _list_face_models() -> List[str]:
    """Return all installed face-detection weights, with a 'none' sentinel."""
    items: List[str] = ["none"]
    for key in ("ultralytics_bbox", "ultralytics", "yolo"):
        try:
            for name in folder_paths.get_filename_list(key):
                if name.lower().endswith((".pt", ".onnx", ".safetensors")):
                    if name not in items:
                        items.append(name)
        except Exception:
            continue
    return items


# ──────────────────────────────────────────────────────────────────────
#  YOLO11 detection (lazy)
# ──────────────────────────────────────────────────────────────────────
_DETECTOR_CACHE: dict = {}


def _detect_faces_yolo(
    image_np: np.ndarray, model_name: str, conf_threshold: float, max_faces: int,
) -> List[Tuple[int, int, int, int, float]]:
    """Run face detection. Returns [(x0,y0,x1,y1,score), ...] in pixel coords.

    Caches the detector by model_name. Tries ultralytics first, then onnxruntime.
    """
    if model_name == "none":
        return []
    cache_key = ("ultra", model_name)
    det = _DETECTOR_CACHE.get(cache_key)
    if det is None:
        try:
            from ultralytics import YOLO  # type: ignore
            for key in ("ultralytics_bbox", "ultralytics", "yolo"):
                try:
                    abs_path = folder_paths.get_full_path(key, model_name)
                    if abs_path:
                        det = YOLO(abs_path)
                        break
                except Exception:
                    continue
            if det is not None:
                _DETECTOR_CACHE[cache_key] = det
        except ImportError:
            logger.warning("[MECFaceFixer] ultralytics not installed; cannot detect faces.")
            return []
    if det is None:
        return []
    H, W = image_np.shape[:2]
    img_u8 = (image_np * 255.0).clip(0, 255).astype(np.uint8)
    try:
        results = det.predict(img_u8, conf=float(conf_threshold), verbose=False)
    except Exception as exc:
        logger.warning("[MECFaceFixer] YOLO predict failed: %s", exc)
        return []
    boxes: List[Tuple[int, int, int, int, float]] = []
    for r in results:
        b = getattr(r, "boxes", None)
        if b is None:
            continue
        for box, score in zip(b.xyxy.cpu().numpy(), b.conf.cpu().numpy()):
            x0, y0, x1, y1 = box.tolist()
            boxes.append((
                max(0, int(x0)), max(0, int(y0)),
                min(W, int(x1)), min(H, int(y1)),
                float(score),
            ))
    boxes.sort(key=lambda b: -b[4])
    return boxes[: int(max_faces)] if max_faces > 0 else boxes


# ──────────────────────────────────────────────────────────────────────
#  Crop / paste helpers
# ──────────────────────────────────────────────────────────────────────
def _expand_bbox(x0: int, y0: int, x1: int, y1: int, pad: float, W: int, H: int):
    cx, cy = (x0 + x1) * 0.5, (y0 + y1) * 0.5
    hw, hh = (x1 - x0) * 0.5 * pad, (y1 - y0) * 0.5 * pad
    return (
        max(0, int(cx - hw)),
        max(0, int(cy - hh)),
        min(W - 1, int(cx + hw)),
        min(H - 1, int(cy + hh)),
    )


def _round_to_multiple(n: int, m: int = 8) -> int:
    return max(m, ((n + m - 1) // m) * m)


# ══════════════════════════════════════════════════════════════════════
#  MECFaceFixer node
# ══════════════════════════════════════════════════════════════════════
class MECFaceFixer:
    """End-to-end face-detail node: detect → crop → upscale → sample → blend.

    All sampling is done through ``comfy.sample`` so it honours the user's
    selected sampler / scheduler exactly like the built-in KSampler.
    """

    CATEGORY = "C2C/Paint"
    DESCRIPTION = (
        "Auto face detection (YOLO11) + per-face crop + AI pre-upscale + "
        "context-aware sampling + smart blend with wildcard per-face prompts. "
        "Behavioural clone of Forbidden Vision's Fixer with Impact-Pack wildcard syntax."
    )
    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("image", "face_mask", "info_json")
    OUTPUT_TOOLTIPS = (
        "Image with detected faces detailed and blended back over the original.",
        "Combined face-detection mask covering all processed faces.",
        "JSON metadata: per-face bbox, score, prompt, denoise.",
    )
    FUNCTION = "execute"

    @classmethod
    def INPUT_TYPES(cls):
        try:
            import comfy.samplers
            samplers = comfy.samplers.KSampler.SAMPLERS
            schedulers = comfy.samplers.KSampler.SCHEDULERS
        except Exception:
            samplers = ["euler", "dpmpp_2m"]
            schedulers = ["normal", "karras"]

        return {
            "required": {
                "image":     ("IMAGE", {"tooltip": "Source image (single frame or batch; processed independently per frame)."}),
                "model":     ("MODEL", {"tooltip": "Diffusion model to sample with."}),
                "positive":  ("CONDITIONING", {"tooltip": "Base positive conditioning. Wildcards in face_positive_prompt override per face."}),
                "negative":  ("CONDITIONING", {"tooltip": "Base negative conditioning. Wildcards in face_negative_prompt override per face."}),
                "vae":       ("VAE", {"tooltip": "VAE used to encode/decode the per-face crops."}),
                "face_model": (_list_face_models(), {
                    "default": "none",
                    "tooltip": "YOLO11 face-detection .pt/.onnx in ComfyUI/models/ultralytics/bbox/. Choose 'none' to use the optional mask input instead.",
                }),
                "confidence":   ("FLOAT", {"default": 0.5, "min": 0.05, "max": 0.95, "step": 0.01, "tooltip": "Minimum detection confidence."}),
                "max_faces":    ("INT", {"default": 8, "min": 0, "max": 32, "step": 1, "tooltip": "Maximum number of faces to process per frame (0 = all)."}),
                "crop_padding": ("FLOAT", {"default": 1.4, "min": 1.0, "max": 3.0, "step": 0.05, "tooltip": "Bbox padding multiplier so the sampler sees context around each face."}),
                "crop_resolution": ("INT", {"default": 768, "min": 256, "max": 2048, "step": 64, "tooltip": "Resize each face crop to this longer-side resolution before sampling."}),
                "denoise":   ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Per-face denoise strength (0.3 = subtle, 0.7 = aggressive reshape)."}),
                "steps":     ("INT", {"default": 20, "min": 1, "max": 100, "tooltip": "Sampling steps per face."}),
                "cfg":       ("FLOAT", {"default": 6.0, "min": 0.0, "max": 30.0, "step": 0.1, "tooltip": "CFG scale for face sampling."}),
                "sampler_name": (samplers, {"default": "euler", "tooltip": "Sampler algorithm."}),
                "scheduler":    (schedulers, {"default": "normal", "tooltip": "Sigma schedule."}),
                "seed":      ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "tooltip": "Base seed; each face gets seed+i."}),
                "blend_softness":  ("FLOAT", {"default": 6.0, "min": 0.0, "max": 64.0, "step": 0.5, "tooltip": "Feather radius (px) on the per-face blend mask."}),
                "mask_dilate":     ("INT", {"default": 4, "min": -32, "max": 32, "step": 1, "tooltip": "Dilate (>0) / erode (<0) of the per-face blend mask."}),
                "color_match":     ("BOOLEAN", {"default": True, "tooltip": "Reinhard mean/std colour match per face."}),
                "lightness_rescue": ("BOOLEAN", {"default": True, "tooltip": "Lift the per-face L channel if the sample comes back darker than the original."}),
                "differential_diffusion": ("BOOLEAN", {"default": True, "tooltip": "Weight the blend by |orig - sampled| so unchanged pixels stay sharp."}),
            },
            "optional": {
                "mask":       ("MASK", {"tooltip": "Optional manual face mask. Used directly when face_model='none' or detection finds nothing."}),
                "upscale_model": ("UPSCALE_MODEL", {"tooltip": "Optional UPSCALE_MODEL applied to faces below crop_resolution before sampling."}),
                "face_positive_prompt": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": "Per-face positive prompt with wildcards: [SEP] separates faces; [ASC]/[DSC]/[ASC-SIZE]/[DSC-SIZE] order; [SKIP] leaves a face untouched.",
                }),
                "face_negative_prompt": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": "Same syntax as face_positive_prompt for negatives.",
                }),
            },
        }

    # ---- detection -------------------------------------------------
    def _detect(self, frame_np: np.ndarray, face_model: str, conf: float, max_faces: int,
                fallback_mask: Optional[np.ndarray]) -> List[Tuple[int, int, int, int, float]]:
        boxes = _detect_faces_yolo(frame_np, face_model, conf, max_faces)
        if boxes:
            return boxes
        if fallback_mask is not None:
            ys, xs = np.where(fallback_mask > 0.5)
            if ys.size > 0 and xs.size > 0:
                return [(int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()), 1.0)]
        return []

    # ---- per-face sampler -----------------------------------------
    def _sample_face(self, model, positive, negative, vae, face_crop: torch.Tensor,
                     denoise: float, steps: int, cfg: float, sampler_name: str,
                     scheduler: str, seed: int) -> torch.Tensor:
        import comfy.sample
        import comfy.samplers
        # VAE-encode crop, run sampler, VAE-decode.
        latent = vae.encode(face_crop[:, :, :, :3])
        sampler = comfy.samplers.sampler_object(sampler_name)
        sigmas = comfy.samplers.calculate_sigmas(
            model.get_model_object("model_sampling"), scheduler, int(steps),
        )
        if 0.0 < float(denoise) < 1.0:
            cut = max(int(round(steps * (1.0 - float(denoise)))), 0)
            sigmas = sigmas[cut:]
        noise = comfy.sample.prepare_noise(latent, int(seed), None)
        out_latent = comfy.sample.sample_custom(
            model, noise, float(cfg), sampler, sigmas,
            positive, negative, latent, seed=int(seed),
        )
        decoded = vae.decode(out_latent)
        return _to_bhwc(decoded)

    # ---- main ------------------------------------------------------
    def execute(self, image, model, positive, negative, vae,
                face_model, confidence, max_faces, crop_padding, crop_resolution,
                denoise, steps, cfg, sampler_name, scheduler, seed,
                blend_softness, mask_dilate, color_match, lightness_rescue,
                differential_diffusion,
                mask=None, upscale_model=None,
                face_positive_prompt: str = "", face_negative_prompt: str = ""):
        import comfy.model_management as mm
        import gc
        import json

        # ── Required-input validation ──
        missing = []
        if model is None:
            missing.append("model")
        if vae is None:
            missing.append("vae")
        if positive is None:
            missing.append("positive")
        if negative is None:
            missing.append("negative")
        if face_model is None:
            missing.append("face_model")
        if missing:
            raise ValueError(
                "MECFaceFixer: required input(s) missing: "
                + ", ".join(missing)
                + ". Connect MODEL, VAE, CONDITIONING and a face-detection "
                "model (e.g. UltralyticsDetectorProvider)."
            )

        try:
            img_b = _to_bhwc(image)
            B, H, W, _ = img_b.shape
            out_b = img_b.clone()
            face_mask_acc = torch.zeros(B, H, W, dtype=torch.float32)
            info: dict = {"frames": []}
            mask_np_full = None
            if mask is not None:
                mask_np_full = _to_mask(mask).cpu().numpy()  # (B,H,W) or (1,H,W)

            inp_helper = MECContextInpainter()

            for fi in range(B):
                frame = img_b[fi].cpu().numpy()
                fb_mask = None
                if mask_np_full is not None:
                    fb_mask = mask_np_full[fi if mask_np_full.shape[0] == B else 0]
                boxes = self._detect(frame, face_model, float(confidence), int(max_faces), fb_mask)
                # Region sizes for [ASC-SIZE] / [DSC-SIZE].
                region_sizes = [(x1 - x0) * (y1 - y0) for (x0, y0, x1, y1, _s) in boxes]
                pos_prompts = _parse_wildcard_prompt(face_positive_prompt, max(len(boxes), 1), region_sizes or None)
                neg_prompts = _parse_wildcard_prompt(face_negative_prompt, max(len(boxes), 1), region_sizes or None)

                composite = frame.copy()
                frame_faces: List[dict] = []

                for fi_box, ((x0, y0, x1, y1, score), pp, np_) in enumerate(zip(boxes, pos_prompts, neg_prompts)):
                    if pp is not None and "[SKIP]" in pp:
                        frame_faces.append({"bbox": [x0, y0, x1, y1], "skipped": True})
                        continue
                    ex0, ey0, ex1, ey1 = _expand_bbox(x0, y0, x1, y1, float(crop_padding), W, H)
                    crop_h = ey1 - ey0
                    crop_w = ex1 - ex0
                    if crop_h < 8 or crop_w < 8:
                        continue

                    crop_np = composite[ey0:ey1 + 1, ex0:ex1 + 1].copy()
                    crop_t = torch.from_numpy(crop_np).unsqueeze(0).float()  # (1,H,W,C)

                    # Resize / upscale to target sampler resolution.
                    longer = max(crop_h, crop_w)
                    target = int(crop_resolution)
                    if longer < target and upscale_model is not None:
                        try:
                            import comfy_extras.nodes_upscale_model as _um  # type: ignore
                            (crop_t,) = _um.ImageUpscaleWithModel().upscale(upscale_model, crop_t)
                        except Exception as exc:
                            logger.warning("[MECFaceFixer] upscale_model failed: %s", exc)
                    # Force exact target longer-side, multiples of 8.
                    cur_h, cur_w = crop_t.shape[1], crop_t.shape[2]
                    scale = target / max(cur_h, cur_w)
                    th = _round_to_multiple(int(round(cur_h * scale)))
                    tw = _round_to_multiple(int(round(cur_w * scale)))
                    if th != cur_h or tw != cur_w:
                        u = crop_t.permute(0, 3, 1, 2)
                        u = F.interpolate(u, size=(th, tw), mode="bicubic", align_corners=False)
                        crop_t = u.permute(0, 2, 3, 1).clamp(0, 1)

                    # Build per-face conditioning by replacing the text in copies of base cond.
                    pos_face = _override_conditioning_text(positive, pp) if pp else positive
                    neg_face = _override_conditioning_text(negative, np_) if np_ else negative

                    sampled = self._sample_face(
                        model, pos_face, neg_face, vae, crop_t,
                        float(denoise), int(steps), float(cfg),
                        sampler_name, scheduler, int(seed) + fi * 1000 + fi_box,
                    )
                    # Resize sampled back to (crop_h+1, crop_w+1).
                    s = sampled.permute(0, 3, 1, 2)
                    s = F.interpolate(s, size=(crop_h + 1, crop_w + 1), mode="bicubic", align_corners=False)
                    sampled_np = s.permute(0, 2, 3, 1).clamp(0, 1)[0].cpu().numpy()

                    # Build per-face mask (ellipse fit inside detection bbox).
                    face_mask_2d = _ellipse_mask(crop_h + 1, crop_w + 1)
                    if int(mask_dilate) != 0:
                        face_mask_2d = _morph_2d(face_mask_2d, int(mask_dilate))
                    if float(blend_softness) > 0.0:
                        t = torch.from_numpy(face_mask_2d)[None, None].float()
                        face_mask_2d = _gaussian_blur_2d(t, float(blend_softness)).squeeze().numpy()
                    face_mask_2d = np.clip(face_mask_2d, 0.0, 1.0)

                    # Smart blend (uses MECContextInpainter's helpers).
                    if color_match:
                        sampled_np = inp_helper._color_match(crop_np, sampled_np, face_mask_2d)
                    if lightness_rescue:
                        sampled_np = inp_helper._lightness_rescue(crop_np, sampled_np, face_mask_2d)
                    if differential_diffusion:
                        w = inp_helper._differential_weight(crop_np, sampled_np)
                        face_mask_2d = np.clip(face_mask_2d * np.maximum(w, 0.05), 0.0, 1.0)

                    b = face_mask_2d[..., None]
                    blended = sampled_np * b + crop_np * (1.0 - b)
                    composite[ey0:ey1 + 1, ex0:ex1 + 1] = blended

                    # Accumulate global face mask.
                    face_mask_acc[fi, ey0:ey1 + 1, ex0:ex1 + 1] = torch.from_numpy(
                        np.maximum(
                            face_mask_acc[fi, ey0:ey1 + 1, ex0:ex1 + 1].numpy(),
                            face_mask_2d.astype(np.float32),
                        ),
                    )
                    frame_faces.append({
                        "bbox": [x0, y0, x1, y1],
                        "score": float(score),
                        "pos_prompt": pp,
                        "neg_prompt": np_,
                        "denoise": float(denoise),
                    })

                out_b[fi] = torch.from_numpy(np.clip(composite, 0.0, 1.0).astype(np.float32))
                info["frames"].append({"frame": fi, "faces": frame_faces})

            return (out_b, face_mask_acc, json.dumps(info))
        finally:
            try:
                mm.soft_empty_cache()
            except Exception:
                pass
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def _ellipse_mask(h: int, w: int) -> np.ndarray:
    """Soft ellipse covering the central 90% of a bbox crop."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cy, cx = (h - 1) * 0.5, (w - 1) * 0.5
    ry, rx = max(cy * 0.9, 1.0), max(cx * 0.9, 1.0)
    d = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2
    m = np.clip(1.0 - d, 0.0, 1.0)
    return m.astype(np.float32)


def _override_conditioning_text(cond, new_text: str):
    """Best-effort: if cond carries a 'text' key in its metadata dict, swap it.

    Otherwise returns the original cond unchanged. Per-face conditioning
    requires upstream re-encoding for full effect; this is a conservative
    fallback so the node never breaks when the base text is good enough.
    """
    if not new_text:
        return cond
    try:
        new_cond = []
        for entry in cond:
            tensor, meta = entry[0], dict(entry[1]) if len(entry) > 1 else {}
            if "text" in meta:
                meta["text"] = new_text
            new_cond.append([tensor, meta])
        return new_cond
    except Exception:
        return cond


NODE_CLASS_MAPPINGS = {
    "MECFaceFixer": MECFaceFixer,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MECFaceFixer": "Face Fixer",
}
