"""Nano Banana (Google Gemini image) API node — C2C.

One node for the whole Nano Banana family:

  * ``gemini-3-pro-image-preview``            — Nano Banana **Pro** (1K/2K/4K,
    strongest text rendering, up to 14 reference images, thinking model)
  * ``gemini-2.5-flash-image``                — Nano Banana (GA, fast)
  * ``gemini-2.5-flash-image-preview``        — Nano Banana preview channel
  * ``gemini-2.0-flash-preview-image-generation`` — legacy image generation

Higgsfield-style feature set: text-to-image, image editing, multi-image
composition and style transfer, plus five prompting styles (raw, cinematic,
structured JSON brief, character-consistency, product shot). Reference
images are ComfyUI IMAGE inputs; batches are flattened in order (capped at
the API's 14-image limit).

The API key comes from the ``api_key`` widget or the ``GEMINI_API_KEY`` /
``GOOGLE_API_KEY`` environment variables. Errors are translated to plain
English — no raw tracebacks.
"""
from __future__ import annotations

import base64
import io
import json
import logging
import os
import time
import urllib.error
import urllib.request
from typing import Any, List, Optional, Tuple

import numpy as np
import torch

from ._is_changed_util import hash_args_and_kwargs

try:
    from comfy.utils import ProgressBar  # type: ignore
except Exception:  # pragma: no cover — offline import (tests)
    class ProgressBar:  # type: ignore
        def __init__(self, n): self.n = n
        def update_absolute(self, v): pass

log = logging.getLogger("C2C.NanoBanana")

_API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"

_MODELS = [
    "gemini-3-pro-image-preview",
    "gemini-2.5-flash-image",
    "gemini-2.5-flash-image-preview",
    "gemini-2.0-flash-preview-image-generation",
]

_MODES = ["text_to_image", "edit_image", "compose_images", "style_transfer"]

_PROMPT_STYLES = [
    "raw",
    "cinematic",
    "structured_json",
    "character_consistency",
    "product_shot",
]

_ASPECTS = ["auto", "1:1", "16:9", "9:16", "4:3", "3:4", "21:9", "3:2", "2:3", "5:4", "4:5"]
_RESOLUTIONS = ["auto", "1K", "2K", "4K"]

_MAX_REF_IMAGES = 14          # Nano Banana Pro limit; flash models use fewer
_REF_MAX_SIDE = 2048          # downscale refs to keep payloads sane


# ──────────────────────────────────────────────────────────────────────
# Prompt templates ("different types of promptings")
# ──────────────────────────────────────────────────────────────────────

def _style_prompt(prompt: str, style: str) -> str:
    p = prompt.strip()
    if style == "cinematic":
        return (
            "Cinematic photograph. " + p +
            "\nShot on a full-frame cinema camera, 35mm prime lens, shallow "
            "depth of field, motivated practical lighting, filmic colour "
            "grade, subtle grain, high dynamic range. Composition follows "
            "the rule of thirds; background falls off softly."
        )
    if style == "structured_json":
        # The model follows structured briefs very reliably; if the user
        # already wrote JSON, pass it through untouched.
        if p.startswith("{"):
            return "Follow this JSON image brief exactly:\n" + p
        return (
            "Follow this JSON image brief exactly:\n"
            + json.dumps({
                "scene": p,
                "style": "photorealistic, highly detailed",
                "lighting": "natural, soft key with gentle rim",
                "camera": {"angle": "eye level", "lens": "50mm", "depth_of_field": "moderate"},
                "constraints": ["no watermark", "no text unless requested",
                                "correct anatomy", "coherent perspective"],
            }, indent=2)
        )
    if style == "character_consistency":
        return (
            "Maintain the EXACT same character identity as in the reference "
            "image(s): same face geometry, skin tone, hair, eye colour and "
            "distinguishing marks. Do not beautify or restyle the person. " + p
        )
    if style == "product_shot":
        return (
            "Professional commercial product photograph. " + p +
            "\nStudio three-point softbox lighting, seamless background, "
            "crisp edge definition, controlled specular highlights, "
            "advertising-grade colour accuracy, tack-sharp focus on the product."
        )
    return p  # raw


def _mode_prompt(prompt: str, mode: str, n_refs: int) -> str:
    if mode == "edit_image":
        return ("Edit the provided reference image. Apply ONLY this change, "
                "preserving everything else (framing, identity, lighting): "
                + prompt)
    if mode == "compose_images":
        return (f"Compose a single coherent image using the {n_refs} provided "
                "reference images as sources. " + prompt)
    if mode == "style_transfer":
        return ("Re-render the FIRST reference image so it keeps its content "
                "and composition, but adopts the complete visual style "
                "(palette, brushwork/grain, lighting, mood) of the LAST "
                "reference image. " + prompt)
    return prompt


# ──────────────────────────────────────────────────────────────────────
# Tensor ↔ PNG helpers
# ──────────────────────────────────────────────────────────────────────

def _tensor_to_png_b64(img: torch.Tensor) -> str:
    """[H,W,3] float 0-1 → base64 PNG (longest side capped)."""
    from PIL import Image
    arr = (img.detach().cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
    pil = Image.fromarray(arr, "RGB")
    if max(pil.size) > _REF_MAX_SIDE:
        r = _REF_MAX_SIDE / max(pil.size)
        pil = pil.resize((max(1, int(pil.width * r)), max(1, int(pil.height * r))),
                         Image.LANCZOS)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _png_b64_to_tensor(b64: str) -> torch.Tensor:
    from PIL import Image
    raw = base64.b64decode(b64)
    pil = Image.open(io.BytesIO(raw)).convert("RGB")
    arr = np.asarray(pil, dtype=np.float32) / 255.0
    return torch.from_numpy(arr)  # [H,W,3]


def _collect_refs(*imgs: Optional[torch.Tensor]) -> List[torch.Tensor]:
    """Flatten optional IMAGE inputs (each possibly a batch) in order."""
    out: List[torch.Tensor] = []
    for t in imgs:
        if t is None:
            continue
        if t.ndim == 3:
            t = t.unsqueeze(0)
        for i in range(t.shape[0]):
            out.append(t[i])
            if len(out) >= _MAX_REF_IMAGES:
                return out
    return out


# ──────────────────────────────────────────────────────────────────────
# Friendly error translation
# ──────────────────────────────────────────────────────────────────────

def _humanise_http(status: int, body: str) -> str:
    try:
        msg = json.loads(body).get("error", {}).get("message", "")[:300]
    except Exception:
        msg = body[:200]
    if status in (401, 403):
        return ("Google rejected the API key. Check the `api_key` widget or the "
                "GEMINI_API_KEY environment variable (create a key at "
                "aistudio.google.com). Detail: " + msg)
    if status == 404:
        return ("This model is not available for your key/region. Try "
                "`gemini-2.5-flash-image`, or enable the preview model in "
                "Google AI Studio. Detail: " + msg)
    if status == 429:
        return ("Rate/quota limit hit on the Gemini API. Wait a minute or raise "
                "your quota in Google AI Studio. Detail: " + msg)
    if status == 400:
        return ("The request was rejected (often an unsupported aspect ratio / "
                "resolution for this model, or an over-long prompt). Detail: " + msg)
    if status >= 500:
        return "Google's image service is having a moment (HTTP %d). Retry shortly." % status
    return f"Gemini API error HTTP {status}: {msg}"


class NanoBananaC2C:
    """All Nano Banana / Gemini image models in one production node."""

    CATEGORY = "C2C/AI Image"
    FUNCTION = "generate"
    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("images", "text_response", "info")
    DESCRIPTION = (
        "Generate or edit images with Google's Nano Banana family "
        "(gemini-3-pro-image / gemini-2.5-flash-image). Modes: text-to-image, "
        "edit, multi-image compose, style transfer. Five prompting styles "
        "(raw / cinematic / structured JSON / character-consistency / product "
        "shot). Wire up to 4 IMAGE inputs as references (batches flatten, max "
        "14). Needs a Google AI Studio key in `api_key` or GEMINI_API_KEY."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "", "tooltip":
                           "What to generate / how to edit. With prompt_style="
                           "structured_json you may paste a full JSON brief."}),
                "model": (_MODELS, {"default": _MODELS[0], "tooltip":
                          "gemini-3-pro-image-preview = Nano Banana Pro (1K/2K/4K, "
                          "best text, 14 refs). gemini-2.5-flash-image = fast GA model."}),
                "mode": (_MODES, {"default": "text_to_image", "tooltip":
                         "edit_image needs ≥1 reference image; style_transfer needs ≥2 "
                         "(first = content, last = style)."}),
                "prompt_style": (_PROMPT_STYLES, {"default": "raw", "tooltip":
                                 "Wraps your prompt in a proven template: cinematic photo, "
                                 "structured JSON brief, character-consistency lock, or "
                                 "commercial product shot."}),
                "aspect_ratio": (_ASPECTS, {"default": "auto"}),
                "resolution": (_RESOLUTIONS, {"default": "auto", "tooltip":
                               "1K/2K/4K are Nano Banana **Pro** only; flash models "
                               "ignore this."}),
                "enhance_prompt": ("BOOLEAN", {"default": False, "tooltip":
                                   "Higgsfield-style enhancer: a fast Gemini text pass expands "
                                   "your prompt into a rich, detailed image brief before generation."}),
                "response_modalities": (["IMAGE+TEXT", "IMAGE"], {"default": "IMAGE+TEXT", "tooltip":
                                        "IMAGE = picture only; IMAGE+TEXT also returns the model's "
                                        "commentary on the text_response output."}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "batch_count": ("INT", {"default": 1, "min": 1, "max": 4, "tooltip":
                                "Sequential API calls; results are stacked into one batch."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0x7FFFFFFF, "tooltip":
                         "Gemini has no true seed — this controls ComfyUI caching: "
                         "same seed reuses the cached result, new seed regenerates."}),
            },
            "optional": {
                "api_key": ("STRING", {"default": "", "tooltip":
                            "Google AI Studio key. Empty = use GEMINI_API_KEY / "
                            "GOOGLE_API_KEY environment variable."}),
                "system_instruction": ("STRING", {"multiline": True, "default": "", "tooltip":
                                       "Optional system-level art direction applied to every "
                                       "generation (brand style, banned elements, palette rules)."}),
                "image_1": ("IMAGE", {"tooltip": "Reference image(s). Batches flatten in order."}),
                "image_2": ("IMAGE", {}),
                "image_3": ("IMAGE", {}),
                "image_4": ("IMAGE", {"tooltip": "For style_transfer the LAST image is the style."}),
            },
        }

    # ── caching: stable content hash (never NaN) ────────────────────
    @classmethod
    def IS_CHANGED(cls, prompt, model, mode, prompt_style, aspect_ratio,
                   resolution, temperature, batch_count, seed,
                   enhance_prompt=False, response_modalities="IMAGE+TEXT",
                   api_key="", system_instruction="", image_1=None,
                   image_2=None, image_3=None, image_4=None, **kwargs):
        return hash_args_and_kwargs(
            prompt, model, mode, prompt_style, aspect_ratio, resolution,
            temperature, batch_count, seed, enhance_prompt, response_modalities,
            system_instruction, image_1, image_2, image_3, image_4, **kwargs,
        )

    @classmethod
    def VALIDATE_INPUTS(cls, mode="text_to_image", **kw):
        return True  # key + ref-count checked at run with friendlier context

    # ── Higgsfield-style prompt enhancer (fast Gemini text pass) ─────
    @staticmethod
    def _enhance(key: str, prompt: str) -> str:
        body = json.dumps({
            "contents": [{"parts": [{"text":
                "Rewrite this image prompt into one richly detailed, concrete "
                "visual brief covering subject, setting, lighting, camera, "
                "style and mood. Keep the user's intent exactly. Return ONLY "
                "the rewritten prompt, no preamble:\n\n" + prompt}]}],
            "generationConfig": {"temperature": 0.9, "maxOutputTokens": 400},
        }).encode()
        req = urllib.request.Request(
            f"{_API_BASE}/gemini-2.5-flash:generateContent",
            data=body, method="POST",
            headers={"Content-Type": "application/json", "x-goog-api-key": key})
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode())
        out = ""
        for part in (data.get("candidates") or [{}])[0].get("content", {}).get("parts", []):
            if part.get("text"):
                out += part["text"]
        return out.strip() or prompt

    # ── execution ────────────────────────────────────────────────────
    def generate(self, prompt, model, mode, prompt_style, aspect_ratio,
                 resolution, temperature, batch_count, seed,
                 enhance_prompt=False, response_modalities="IMAGE+TEXT",
                 api_key="", system_instruction="", image_1=None,
                 image_2=None, image_3=None, image_4=None):

        key = (api_key or "").strip() or os.environ.get("GEMINI_API_KEY", "") \
            or os.environ.get("GOOGLE_API_KEY", "")
        if not key:
            raise ValueError(
                "No Gemini API key. Paste one into the `api_key` widget or set "
                "the GEMINI_API_KEY environment variable (keys: aistudio.google.com).")

        refs = _collect_refs(image_1, image_2, image_3, image_4)
        if mode == "edit_image" and not refs:
            raise ValueError("edit_image mode needs at least one reference image "
                             "wired into image_1.")
        if mode == "style_transfer" and len(refs) < 2:
            raise ValueError("style_transfer needs two images: image_1 = content, "
                             "last image = style.")
        if mode == "compose_images" and not refs:
            raise ValueError("compose_images needs at least one reference image.")

        enhanced = False
        if enhance_prompt and prompt.strip():
            try:
                prompt = self._enhance(key, prompt)
                enhanced = True
            except Exception as exc:                                    # noqa: BLE001
                log.warning("[NanoBanana] prompt enhancer failed (%s); using "
                            "the original prompt", exc)

        text = _mode_prompt(_style_prompt(prompt, prompt_style), mode, len(refs))
        if not text.strip():
            raise ValueError("The prompt is empty.")

        parts: List[dict] = [{"text": text}]
        for r in refs:
            parts.append({"inline_data": {"mime_type": "image/png",
                                          "data": _tensor_to_png_b64(r)}})

        gen_cfg: dict[str, Any] = {
            "responseModalities": (["TEXT", "IMAGE"] if response_modalities == "IMAGE+TEXT"
                                   else ["IMAGE"]),
            "temperature": float(temperature),
        }
        img_cfg: dict[str, Any] = {}
        if aspect_ratio != "auto":
            img_cfg["aspectRatio"] = aspect_ratio
        if resolution != "auto" and model.startswith("gemini-3"):
            img_cfg["imageSize"] = resolution
        if img_cfg:
            gen_cfg["imageConfig"] = img_cfg

        payload: dict[str, Any] = {"contents": [{"parts": parts}],
                                   "generationConfig": gen_cfg}
        if system_instruction.strip():
            payload["systemInstruction"] = {"parts": [{"text": system_instruction.strip()}]}
        url = f"{_API_BASE}/{model}:generateContent"

        out_imgs: List[torch.Tensor] = []
        texts: List[str] = []
        usage: dict = {}
        pbar = ProgressBar(int(batch_count))
        t0 = time.perf_counter()

        for b in range(int(batch_count)):
            body = json.dumps(payload).encode()
            req = urllib.request.Request(
                url, data=body, method="POST",
                headers={"Content-Type": "application/json",
                         "x-goog-api-key": key})
            try:
                with urllib.request.urlopen(req, timeout=420) as resp:
                    data = json.loads(resp.read().decode())
            except urllib.error.HTTPError as e:
                raise RuntimeError(_humanise_http(e.code, e.read().decode(errors="replace"))) from None
            except urllib.error.URLError as e:
                raise RuntimeError("Could not reach the Gemini API (network/DNS/"
                                   f"firewall): {getattr(e, 'reason', e)}") from None

            fb = data.get("promptFeedback") or {}
            if fb.get("blockReason"):
                raise RuntimeError("Gemini blocked this prompt (%s). Rephrase the "
                                   "request or remove restricted content."
                                   % fb["blockReason"])
            usage = data.get("usageMetadata", usage)

            cands = data.get("candidates") or []
            if not cands:
                raise RuntimeError("Gemini returned no result for this prompt. "
                                   "Try rephrasing or a different model.")
            fin = str(cands[0].get("finishReason", ""))
            got_img = False
            for part in (cands[0].get("content") or {}).get("parts", []):
                inline = part.get("inlineData") or part.get("inline_data")
                if inline and inline.get("data"):
                    out_imgs.append(_png_b64_to_tensor(inline["data"]))
                    got_img = True
                elif part.get("text"):
                    texts.append(part["text"])
            if not got_img:
                if fin in ("SAFETY", "IMAGE_SAFETY", "PROHIBITED_CONTENT", "RECITATION"):
                    raise RuntimeError("Gemini refused to render this image "
                                       f"({fin}). Adjust the prompt/references.")
                raise RuntimeError("Gemini answered with text only — no image. "
                                   "Response: " + (" ".join(texts)[:200] or fin or "empty"))
            pbar.update_absolute(b + 1)

        # Stack (resize any stragglers to the first image's size).
        h0, w0 = out_imgs[0].shape[0], out_imgs[0].shape[1]
        norm: List[torch.Tensor] = []
        for t in out_imgs:
            if t.shape[0] != h0 or t.shape[1] != w0:
                t = torch.nn.functional.interpolate(
                    t.permute(2, 0, 1).unsqueeze(0), size=(h0, w0),
                    mode="bilinear", align_corners=False)[0].permute(1, 2, 0)
            norm.append(t)
        batch = torch.stack(norm, 0)  # [B,H,W,3]

        info = json.dumps({
            "model": model, "mode": mode, "prompt_style": prompt_style,
            "aspect_ratio": aspect_ratio, "resolution": resolution,
            "prompt_enhanced": enhanced,
            "reference_images": len(refs), "images_out": batch.shape[0],
            "size": f"{w0}x{h0}", "latency_s": round(time.perf_counter() - t0, 2),
            "tokens": usage.get("totalTokenCount"),
        })
        log.info("[NanoBanana] %s", info)
        return (batch, "\n".join(texts).strip(), info)


NODE_CLASS_MAPPINGS = {"NanoBananaC2C": NanoBananaC2C}
NODE_DISPLAY_NAME_MAPPINGS = {"NanoBananaC2C": "Nano Banana · Gemini Image (C2C)"}
