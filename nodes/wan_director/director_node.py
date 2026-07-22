"""
Wan Director — visual multi-shot timeline node for Wan video models.

A single, model-aware Director node that lets the user lay out image,
text and audio clips on a visual timeline and emits the model-specific
bundle of (model, conditioning, latent, fps, audio, guide_data) needed
by Wan 2.1 / 2.2 / Fun / Animate. The actual timeline UI lives in the
companion ``js/wan_director_timeline.js`` extension; this Python class
holds the schema + dispatches per-variant output assembly.

Forked-from inspiration: WhatDreamsCost-ComfyUI / LTX Director (MIT).
See NOTICE.

Author: Code2Collapse, May 2026.  Apache-2.0.
"""
from __future__ import annotations

import base64
import io as _io
import json
import logging
import math
import os
from typing import Any

import numpy as np
import torch
from PIL import Image

from .._is_changed_util import hash_args_and_kwargs

import folder_paths  # type: ignore

log = logging.getLogger("MEC.WanDirector")

# ── Variant table ────────────────────────────────────────────────────
#
# Each entry encodes the per-variant differences that drive output
# assembly: latent shape, divisible_by, default fps, and which optional
# pathways (negative prompt, control video, reference image, dual cfg)
# are user-facing.

VARIANT_TABLE: dict[str, dict[str, Any]] = {
    "wan2.1_t2v": {
        "label":          "Wan 2.1 — Text → Video",
        "latent_channels": 16,
        "spatial_div":     8,    # Wan VAE spatial compression
        "temporal_div":    4,    # Wan VAE temporal compression
        "default_fps":     16.0,
        "needs_image":     False,
        "supports_neg":    True,
        "dual_cfg":        False,
        "ref_image":       False,
        "control_video":   False,
    },
    "wan2.1_i2v": {
        "label":          "Wan 2.1 — Image → Video",
        "latent_channels": 16,
        "spatial_div":     8,
        "temporal_div":    4,
        "default_fps":     16.0,
        "needs_image":     True,
        "supports_neg":    True,
        "dual_cfg":        False,
        "ref_image":       False,
        "control_video":   False,
    },
    "wan2.2_t2v": {
        "label":          "Wan 2.2 — Text → Video (dual-cfg)",
        "latent_channels": 16,
        "spatial_div":     8,
        "temporal_div":    4,
        "default_fps":     16.0,
        "needs_image":     False,
        "supports_neg":    True,
        "dual_cfg":        True,
        "ref_image":       False,
        "control_video":   False,
    },
    "wan2.2_i2v": {
        "label":          "Wan 2.2 — Image → Video (dual-cfg)",
        "latent_channels": 16,
        "spatial_div":     8,
        "temporal_div":    4,
        "default_fps":     16.0,
        "needs_image":     True,
        "supports_neg":    True,
        "dual_cfg":        True,
        "ref_image":       False,
        "control_video":   False,
    },
    "wan_fun_inp": {
        "label":          "Wan Fun — Inpaint (mask + control)",
        "latent_channels": 16,
        "spatial_div":     8,
        "temporal_div":    4,
        "default_fps":     16.0,
        "needs_image":     True,
        "supports_neg":    True,
        "dual_cfg":        False,
        "ref_image":       False,
        "control_video":   True,
    },
    "wan_fun_control": {
        "label":          "Wan Fun — Control Video (depth/pose/canny)",
        "latent_channels": 16,
        "spatial_div":     8,
        "temporal_div":    4,
        "default_fps":     16.0,
        "needs_image":     False,
        "supports_neg":    True,
        "dual_cfg":        False,
        "ref_image":       False,
        "control_video":   True,
    },
    "wan_animate": {
        "label":          "Wan Animate — Reference + Pose",
        "latent_channels": 16,
        "spatial_div":     8,
        "temporal_div":    4,
        "default_fps":     16.0,
        "needs_image":     False,
        "supports_neg":    True,
        "dual_cfg":        False,
        "ref_image":       True,
        "control_video":   True,
        "everanimate":    False,
    },
    # EverAnimate (vita-epfl/EverAnimate) — rank-32 LoRA on top of
    # Wan2.2-Animate-14B that adds Persistent Latent Propagation +
    # Restorative Flow Matching for minute-scale human animation. The
    # director only emits the LoRA descriptor + chunking program in
    # `tracks_program["everanimate"]`; an EverAnimate-runner node is
    # responsible for loading the LoRA and executing chunked inference.
    "wan2.2_animate_everanimate": {
        "label":          "EverAnimate — Long-horizon (Wan2.2-Animate + rank-32 LoRA)",
        "latent_channels": 16,
        "spatial_div":     8,
        "temporal_div":    4,
        "default_fps":     16.0,
        "needs_image":     False,
        "supports_neg":    True,
        "dual_cfg":        True,
        "ref_image":       True,
        "control_video":   True,
        "everanimate":    True,
    },
}

# EverAnimate-specific constants (used by the runner; the director just
# echoes them through tracks_program so different runners can implement
# the chunking + anchor logic consistently).
EVERANIMATE_STAGES = ("stage1_480p", "stage2_480p", "stage3_720p_beta")
EVERANIMATE_ANCHOR_STRATEGIES = ("auto", "first_only", "first_plus_random_3")

VARIANT_KEYS: list[str] = list(VARIANT_TABLE)


# ── Tensor helpers ───────────────────────────────────────────────────

def _snap(val: int, div: int) -> int:
    return max(div, (val // div) * div)


def _load_image_tensor(seg: dict) -> torch.Tensor:
    """Decode an image segment to a [1,H,W,3] float32 tensor in [0,1]."""
    if seg.get("imageFile"):
        file_path = os.path.join(folder_paths.get_input_directory(), seg["imageFile"])
        if os.path.exists(file_path):
            img = Image.open(file_path).convert("RGB")
            arr = np.asarray(img, dtype=np.float32) / 255.0
            return torch.from_numpy(arr).unsqueeze(0)

    b64_str = seg.get("imageB64", "") or ""
    if not b64_str or b64_str.startswith("/view?"):
        return torch.zeros((1, 512, 512, 3), dtype=torch.float32)
    if "," in b64_str:
        b64_str = b64_str.split(",", 1)[1]
    try:
        img_bytes = base64.b64decode(b64_str)
        img = Image.open(_io.BytesIO(img_bytes)).convert("RGB")
        arr = np.asarray(img, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).unsqueeze(0)
    except Exception as exc:                                    # noqa: BLE001
        log.warning("[WanDirector] image decode failed: %s", exc)
        return torch.zeros((1, 512, 512, 3), dtype=torch.float32)


def _resize_image(t: torch.Tensor, tw: int, th: int, method: str, div: int) -> torch.Tensor:
    """Resize [1,H,W,3] with method ∈ {stretch,fit,pad,crop} and snap to div."""
    from PIL import Image as _Pil  # local import keeps cold-start fast
    tw, th = _snap(tw, div), _snap(th, div)
    img_np = (t[0].cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    pil = _Pil.fromarray(img_np)
    sw, sh = pil.size
    if method == "stretch to fit":
        out = pil.resize((tw, th), _Pil.LANCZOS)
    elif method == "pad":
        r = min(tw / sw, th / sh)
        nw, nh = _snap(int(sw * r), div), _snap(int(sh * r), div)
        inner = pil.resize((nw, nh), _Pil.LANCZOS)
        out = _Pil.new("RGB", (tw, th), (0, 0, 0))
        out.paste(inner, ((tw - nw) // 2, (th - nh) // 2))
    elif method == "crop":
        r = max(tw / sw, th / sh)
        nw, nh = int(sw * r), int(sh * r)
        inner = pil.resize((nw, nh), _Pil.LANCZOS)
        l, top = (nw - tw) // 2, (nh - th) // 2
        out = inner.crop((l, top, l + tw, top + th))
    else:  # maintain aspect ratio
        r = min(tw / sw, th / sh)
        nw, nh = _snap(int(sw * r), div), _snap(int(sh * r), div)
        out = pil.resize((nw, nh), _Pil.LANCZOS)
    arr = np.asarray(out, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


# ── Audio mux (reused from LTX Director; PyAV) ───────────────────────

def _build_combined_audio(timeline_data_str: str, duration_frames: int,
                          frame_rate: float, target: str = "music_44k_stereo") -> dict:
    """Mix the timeline's audio clips into a single ComfyUI AUDIO dict.

    ``target`` chooses sample rate + channel count:
      * ``music_44k_stereo`` (default): 44.1 kHz stereo
      * ``speech_16k_mono``:            16 kHz mono   (for Wan-S2V style)
    """
    if target == "speech_16k_mono":
        sr, layout, channels = 16000, "mono", 1
    else:
        sr, layout, channels = 44100, "stereo", 2

    total = max(1, int(math.ceil(duration_frames / max(1e-3, frame_rate) * sr)))
    empty = {"waveform": torch.zeros((1, channels, total), dtype=torch.float32),
             "sample_rate": sr}

    if not timeline_data_str:
        return empty
    try:
        data = json.loads(timeline_data_str)
        audio_segs = data.get("audioSegments", []) or []
    except Exception:                                            # noqa: BLE001
        return empty
    if not audio_segs:
        return empty

    try:
        import av  # type: ignore
    except ImportError:
        log.warning("[WanDirector] PyAV missing; cannot mix audio. `pip install av`.")
        return empty

    out = torch.zeros((channels, total), dtype=torch.float32)

    for seg in audio_segs:
        buf = None
        if seg.get("audioFile"):
            fp = os.path.join(folder_paths.get_input_directory(), seg["audioFile"])
            if os.path.exists(fp):
                with open(fp, "rb") as f:
                    buf = _io.BytesIO(f.read())
        if buf is None and seg.get("audioB64"):
            b64 = seg["audioB64"]
            if "," in b64:
                b64 = b64.split(",", 1)[1]
            try:
                buf = _io.BytesIO(base64.b64decode(b64))
            except Exception:                                    # noqa: BLE001
                buf = None
        if buf is None:
            continue

        try:
            frames_t: list[torch.Tensor] = []
            with av.open(buf) as container:
                stream = container.streams.audio[0]
                resampler = av.AudioResampler(format="fltp", layout=layout, rate=sr)
                for fr in container.decode(stream):
                    for rf in resampler.resample(fr):
                        frames_t.append(torch.from_numpy(rf.to_ndarray()))
                for rf in resampler.resample(None):
                    frames_t.append(torch.from_numpy(rf.to_ndarray()))
            if not frames_t:
                continue
            wave = torch.cat(frames_t, dim=1)
            if wave.shape[0] != channels:                        # mono ↔ stereo coerce
                wave = wave.mean(dim=0, keepdim=True) if channels == 1 else wave.expand(channels, -1)

            trim_start = float(seg.get("trimStart", 0))
            length = float(seg.get("length", 1))
            start = float(seg.get("start", 0))
            s0 = int(trim_start / max(1e-3, frame_rate) * sr)
            n = max(0, int(length / max(1e-3, frame_rate) * sr))
            s1 = min(wave.shape[1], s0 + n)
            if s1 <= s0:
                continue
            clip = wave[:, s0:s1]
            d0 = int(start / max(1e-3, frame_rate) * sr)
            if d0 >= out.shape[1]:
                continue
            d1 = min(out.shape[1], d0 + clip.shape[1])
            out[:, d0:d1] += clip[:, : d1 - d0]
        except Exception as exc:                                 # noqa: BLE001
            log.warning("[WanDirector] audio segment error (%s): %s", seg.get("fileName"), exc)
            continue

    return {"waveform": out.unsqueeze(0), "sample_rate": sr}


# ── Variant-aware latent factory ─────────────────────────────────────

def _empty_wan_latent(variant: str, w: int, h: int, frames: int) -> dict:
    """Build an empty Wan latent of the right shape.

    Wan VAE: 16 channels, /8 spatial, /4 temporal.
    The temporal formula mirrors Wan's reference code: ``(F - 1) // 4 + 1``
    latent frames for ``F`` pixel frames.
    """
    cfg = VARIANT_TABLE[variant]
    ch = int(cfg["latent_channels"])
    sd = int(cfg["spatial_div"])
    td = int(cfg["temporal_div"])
    lw, lh = max(1, w // sd), max(1, h // sd)
    lt = max(1, (frames - 1) // td + 1)
    try:
        import comfy.model_management as mm  # type: ignore
        device = mm.intermediate_device()
    except Exception:                                            # noqa: BLE001
        device = torch.device("cpu")
    samples = torch.zeros((1, ch, lt, lh, lw), device=device, dtype=torch.float32)
    return {"samples": samples}


# ── PromptRelay helpers (universal: native + Kijai + generic fallback) ──
#
# These wrap `nodes.prompt_relay` so the Director can embed PromptRelay
# without duplicating logic. The PromptRelay module already handles
# any open-source video model via its generic-introspection fallback.

def _apply_prompt_relay_native(model, clip, latent, global_prompt, local_list,
                               segment_lengths_str, epsilon):
    """Apply PromptRelay to a native ComfyUI MODEL. Returns (patched_model, conditioning)."""
    from ..prompt_relay._nodes import _encode_native as _pr_encode_native
    local_prompts_str = " | ".join(local_list)
    return _pr_encode_native(
        model, clip, latent,
        global_prompt or local_list[0],
        local_prompts_str,
        segment_lengths_str or "",
        float(epsilon),
        None,
    )


def _t5gemma_encode(t5gemma, text, max_tokens=512):
    """Encode `text` with a loaded T5Gemma encoder → ComfyUI CONDITIONING
    (``[[hidden, {pooled_output, attention_mask}]]``), matching the shape
    WanT5GemmaTextEncode emits so downstream metadata attaches cleanly.

    R&D only: T5Gemma states are a different space/width from Wan's UMT5-XXL;
    usable on a Wan model adapted/finetuned for T5Gemma, not stock Wan.
    """
    model, tok = t5gemma["model"], t5gemma["tokenizer"]
    dev = next(model.parameters()).device
    enc = tok(text or "", return_tensors="pt", padding=True, truncation=True,
              max_length=int(max_tokens))
    enc = {k: v.to(dev) for k, v in enc.items()}
    with torch.inference_mode():
        hidden = model(**enc).last_hidden_state
    mask = enc.get("attention_mask")
    if mask is not None:
        pooled = (hidden * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
    else:
        pooled = hidden.mean(1)
    return [[hidden.float(), {"pooled_output": pooled.float(), "attention_mask": mask}]]


def _apply_kijai_branch(wan_model, wan_t5, pos_text, neg_text, local_list,
                        segment_lengths_str, duration_frames, epsilon,
                        enable_prompt_relay):
    """Drive Kijai WanVideoWrapper: encode prompts via T5, optionally PromptRelay-patch."""
    from ..prompt_relay._core import (
        build_segments, convert_pixel_to_latent_lengths, create_mask_fn,
        distribute_segment_lengths, map_token_indices,
    )
    from ..prompt_relay._patches import patch_kijai

    relay_used = False
    relay_note = ""

    encoder = wan_t5["model"]
    tokenizer = getattr(encoder, "tokenizer", None)
    if tokenizer is None:
        raise RuntimeError(
            "Kijai T5 encoder is missing .tokenizer. "
            "Update ComfyUI-WanVideoWrapper to a recent version."
        )

    try:
        import comfy.model_management as mm  # type: ignore
        device_to = mm.get_torch_device()
    except Exception:                                                # noqa: BLE001
        device_to = torch.device("cuda")

    if enable_prompt_relay and len(local_list) >= 2:
        latent_frames = max(1, (duration_frames - 1) // 4 + 1)

        class _TokAdapter:
            add_eos = True
            def __call__(self_inner, text):
                ids, _mask = tokenizer([text], return_mask=True, add_special_tokens=True)
                return {"input_ids": ids}

        full_prompt, token_ranges = map_token_indices(_TokAdapter(), pos_text, local_list)

        parsed_lengths = None
        if segment_lengths_str.strip():
            pixel_lengths = [int(x.strip()) for x in segment_lengths_str.split(",") if x.strip()]
            parsed_lengths = convert_pixel_to_latent_lengths(pixel_lengths, 4, latent_frames)
        effective_lengths = distribute_segment_lengths(len(local_list), latent_frames, parsed_lengths)

        transformer = getattr(getattr(wan_model, "model", wan_model), "diffusion_model", None)
        patch_size = tuple(getattr(transformer, "patch_size", (1, 2, 2))) if transformer is not None else (1, 2, 2)
        fallback_tpf = max(1, 64 * 64 // (patch_size[1] * patch_size[2]))

        q_token_idx = build_segments(token_ranges, effective_lengths, epsilon, None)
        mask_fn = create_mask_fn(q_token_idx, fallback_tpf, latent_frames)

        n = patch_kijai(wan_model, mask_fn)
        relay_used = True
        relay_note = f"Kijai PromptRelay applied to {n} cross-attn blocks ({len(local_list)} segments)"
        encode_prompt = full_prompt
    elif enable_prompt_relay:
        relay_note = "skipped (need 2+ local prompts)"
        encode_prompt = pos_text
    else:
        encode_prompt = pos_text

    with torch.no_grad():
        try:
            context = encoder([encode_prompt], device_to)
            context_null = encoder([neg_text or ""], device_to)
        finally:
            try:
                import comfy.model_management as mm  # type: ignore
                mm.soft_empty_cache()
            except Exception:                                        # noqa: BLE001
                pass

    text_embeds = {
        "prompt_embeds": context,
        "negative_prompt_embeds": context_null,
        "echoshot": False,
    }
    return wan_model, text_embeds, relay_used, relay_note


# ── Timeline schema v2 ───────────────────────────────────────────────
#
# v1 (legacy): {"segments": [...], "audioSegments": [...]}
#              — no schema_version, only image/text/audio.
#
# v2 (current): adds four optional track arrays for downstream consumption.
#               Image/text/audio handling is unchanged so v1 workflows keep
#               working without any user action; the migration is automatic.
#
#   schema_version : 2
#   segments       : image + text clips (UNCHANGED)
#   audioSegments  : audio clips        (UNCHANGED)
#   loraSegments   : [{name, strength, start, length, [easing]}]
#   cameraSegments : [{type, start, length, params, [easing]}]
#                    type ∈ {static, pan, zoom, orbit, dolly}
#                    params is a free-form dict whose meaningful keys depend
#                    on type (e.g. pan→dx/dy, zoom→from/to, orbit→radius/deg).
#   seedSegments   : [{seed, start, length, mode}]
#                    mode ∈ {fixed, increment, random_per_frame}
#   poseSegments   : [{poseFile|poseB64, start, length, strength,
#                      [interpolation]}]
#                    interpolation ∈ {nearest, linear}
#
# The director itself does not *apply* LoRAs / cameras / seeds / poses (that
# is the job of downstream applier nodes); it parses and validates the
# program, then emits a single compact JSON STRING (`tracks_program`)
# alongside the existing 10 outputs. Validation issues are surfaced via the
# `info` JSON's `track_warnings` array and via Python warnings.

_SCHEMA_VERSION = 2
_TRACK_KEYS = ("loraSegments", "cameraSegments", "seedSegments", "poseSegments")
_CAMERA_TYPES = frozenset(("static", "pan", "zoom", "orbit", "dolly"))
_SEED_MODES = frozenset(("fixed", "increment", "random_per_frame"))
_POSE_INTERP = frozenset(("nearest", "linear"))


def _migrate_timeline(tdata: dict) -> tuple[dict, list[str]]:
    """Upgrade a timeline dict to schema v2 in-place; return (dict, notes).

    A v1 document (no ``schema_version`` field) is upgraded by adding the
    version tag and ensuring every v2 track array exists (empty). No data
    is lost. Already-v2 documents pass through.
    """
    notes: list[str] = []
    if not isinstance(tdata, dict):
        return ({"schema_version": _SCHEMA_VERSION,
                 "segments": [], "audioSegments": [],
                 "loraSegments": [], "cameraSegments": [],
                 "seedSegments": [], "poseSegments": []},
                ["timeline_data was not a JSON object; replaced with empty v2."])
    version = int(tdata.get("schema_version", 1) or 1)
    if version < 1:
        version = 1
    if version > _SCHEMA_VERSION:
        notes.append(
            f"timeline_data schema_version={version} is newer than this "
            f"director (v{_SCHEMA_VERSION}); unknown fields will be ignored."
        )
    if version < 2:
        notes.append(f"timeline_data migrated v{version} → v{_SCHEMA_VERSION}.")
    tdata.setdefault("segments", [])
    tdata.setdefault("audioSegments", [])
    for k in _TRACK_KEYS:
        tdata.setdefault(k, [])
        if not isinstance(tdata[k], list):
            notes.append(f"timeline_data.{k} was not a list; reset to [].")
            tdata[k] = []
    tdata["schema_version"] = _SCHEMA_VERSION
    return tdata, notes


def _clip_frames(start: Any, length: Any, total: int) -> tuple[int, int] | None:
    """Coerce + clamp a (start, length) pair into [0, total). Returns None
    if the segment lies entirely outside the timeline."""
    try:
        s = int(start)
        n = int(length)
    except (TypeError, ValueError):
        return None
    if n <= 0:
        return None
    s = max(0, s)
    n = min(n, total - s)
    if n <= 0:
        return None
    return s, n


def _compile_lora_program(segs: list, total: int) -> tuple[list[dict], list[str]]:
    """Validate LoRA segments and emit a normalised list.

    Output entry: {"name": str, "strength": float, "start": int, "length": int}.
    Out-of-range and malformed entries are dropped with a warning each.
    Overlapping LoRAs are allowed (downstream stacker decides composition).
    """
    out: list[dict] = []
    warns: list[str] = []
    for idx, seg in enumerate(segs or []):
        if not isinstance(seg, dict):
            warns.append(f"loraSegments[{idx}] is not an object; skipped.")
            continue
        name = str(seg.get("name", "")).strip()
        if not name:
            warns.append(f"loraSegments[{idx}] missing 'name'; skipped.")
            continue
        clipped = _clip_frames(seg.get("start", 0), seg.get("length", 0), total)
        if clipped is None:
            warns.append(f"loraSegments[{idx}] ({name}) outside timeline; skipped.")
            continue
        s, n = clipped
        try:
            strength = float(seg.get("strength", 1.0))
        except (TypeError, ValueError):
            warns.append(f"loraSegments[{idx}] ({name}) bad strength; defaulting to 1.0.")
            strength = 1.0
        strength = max(-4.0, min(4.0, strength))
        out.append({"name": name, "strength": strength, "start": s, "length": n})
    return out, warns


def _compile_camera_program(segs: list, total: int,
                            frame_rate: float) -> tuple[list[dict], list[str]]:
    """Validate camera segments. Emits normalised entries with the same
    type+params the user supplied (the applier interprets params)."""
    out: list[dict] = []
    warns: list[str] = []
    fps = max(1e-3, float(frame_rate))
    for idx, seg in enumerate(segs or []):
        if not isinstance(seg, dict):
            warns.append(f"cameraSegments[{idx}] is not an object; skipped.")
            continue
        ctype = str(seg.get("type", "static")).strip().lower()
        if ctype not in _CAMERA_TYPES:
            warns.append(
                f"cameraSegments[{idx}] type='{ctype}' unknown; falling back to 'static'."
            )
            ctype = "static"
        clipped = _clip_frames(seg.get("start", 0), seg.get("length", 0), total)
        if clipped is None:
            warns.append(f"cameraSegments[{idx}] ({ctype}) outside timeline; skipped.")
            continue
        s, n = clipped
        params = seg.get("params", {})
        if not isinstance(params, dict):
            warns.append(f"cameraSegments[{idx}] ({ctype}) params not an object; reset to {{}}.")
            params = {}
        out.append({
            "type": ctype, "start": s, "length": n,
            "duration_s": round(n / fps, 4),
            "params": params,
            "easing": str(seg.get("easing", "linear")),
        })
    return out, warns


def _compile_seed_program(segs: list, total: int) -> tuple[list[dict], list[str]]:
    """Validate seed segments. Emits normalised entries."""
    out: list[dict] = []
    warns: list[str] = []
    for idx, seg in enumerate(segs or []):
        if not isinstance(seg, dict):
            warns.append(f"seedSegments[{idx}] is not an object; skipped.")
            continue
        mode = str(seg.get("mode", "fixed")).strip().lower()
        if mode not in _SEED_MODES:
            warns.append(
                f"seedSegments[{idx}] mode='{mode}' unknown; using 'fixed'."
            )
            mode = "fixed"
        clipped = _clip_frames(seg.get("start", 0), seg.get("length", 0), total)
        if clipped is None:
            warns.append(f"seedSegments[{idx}] outside timeline; skipped.")
            continue
        s, n = clipped
        try:
            seed = int(seg.get("seed", 0))
        except (TypeError, ValueError):
            warns.append(f"seedSegments[{idx}] non-int seed; using 0.")
            seed = 0
        # Mask to 64-bit signed range (downstream samplers).
        seed = seed & 0x7FFFFFFFFFFFFFFF
        out.append({"seed": seed, "start": s, "length": n, "mode": mode})
    return out, warns


def _compile_pose_program(segs: list, total: int) -> tuple[list[dict], list[str]]:
    """Validate pose segments. Either ``poseFile`` (input-dir basename) or
    ``poseB64`` (data URL or raw base64) must be present; both is OK
    (file wins downstream)."""
    out: list[dict] = []
    warns: list[str] = []
    for idx, seg in enumerate(segs or []):
        if not isinstance(seg, dict):
            warns.append(f"poseSegments[{idx}] is not an object; skipped.")
            continue
        has_src = bool(seg.get("poseFile")) or bool(seg.get("poseB64"))
        if not has_src:
            warns.append(f"poseSegments[{idx}] missing poseFile/poseB64; skipped.")
            continue
        clipped = _clip_frames(seg.get("start", 0), seg.get("length", 0), total)
        if clipped is None:
            warns.append(f"poseSegments[{idx}] outside timeline; skipped.")
            continue
        s, n = clipped
        interp = str(seg.get("interpolation", "linear")).strip().lower()
        if interp not in _POSE_INTERP:
            warns.append(f"poseSegments[{idx}] interpolation='{interp}' unknown; using 'linear'.")
            interp = "linear"
        try:
            strength = float(seg.get("strength", 1.0))
        except (TypeError, ValueError):
            warns.append(f"poseSegments[{idx}] bad strength; defaulting to 1.0.")
            strength = 1.0
        strength = max(0.0, min(2.0, strength))
        entry = {"start": s, "length": n, "strength": strength,
                 "interpolation": interp}
        if seg.get("poseFile"):
            entry["poseFile"] = str(seg["poseFile"])
        if seg.get("poseB64"):
            entry["poseB64"] = str(seg["poseB64"])
        out.append(entry)
    return out, warns


# Control-video kinds the timeline's Control track can carry. Mirrors the
# CONTROL_TYPES list in js/wan_director_timeline.js. Any of these routes the
# clip to a matching ControlNet / Uni3C camera-embed path downstream.
_CONTROL_TYPES = {
    "uni3c_camera",  # Uni3C camera-embed video
    "pose",          # DWPose / OpenPose
    "depth",         # depth map sequence
    "canny",         # canny / lineart / edges
    "normal",        # normal map
    "tile",          # tile / reference
    "raw",           # feed the frames through unmodified
}


def _compile_control_program(segs: list, total: int) -> tuple[list[dict], list[str]]:
    """Validate the Control-Video track (``motionSegments`` in the timeline).

    Each entry only carries file references + metadata (no decode here), so
    the program stays cheap on RAM/VRAM — the downstream runner opens the
    clip lazily and applies the ControlNet indicated by ``controlType``.
    """
    out: list[dict] = []
    warns: list[str] = []
    for idx, seg in enumerate(segs or []):
        if not isinstance(seg, dict):
            warns.append(f"motionSegments[{idx}] is not an object; skipped.")
            continue
        vfile = str(seg.get("videoFile", "")).strip()
        if not vfile:
            warns.append(f"motionSegments[{idx}] missing videoFile; skipped.")
            continue
        clipped = _clip_frames(seg.get("start", 0), seg.get("length", 0), total)
        if clipped is None:
            warns.append(f"motionSegments[{idx}] ({vfile}) outside timeline; skipped.")
            continue
        s, n = clipped
        ctype = str(seg.get("controlType", "uni3c_camera")).strip().lower()
        if ctype not in _CONTROL_TYPES:
            warns.append(f"motionSegments[{idx}] controlType='{ctype}' unknown; using 'uni3c_camera'.")
            ctype = "uni3c_camera"
        try:
            trim = max(0, int(seg.get("trimStart", 0)))
        except (TypeError, ValueError):
            trim = 0
        out.append({
            "videoFile": vfile,
            "fileName": str(seg.get("fileName", vfile)),
            "controlType": ctype,
            "start": s, "length": n, "trimStart": trim,
            "srcDurationFrames": int(seg.get("srcDurationFrames", 0) or 0),
        })
    return out, warns


# ── The node ─────────────────────────────────────────────────────────

class WanDirectorC2C:
    """Wan Director — visual multi-shot timeline (single node, all Wan variants).

    The timeline UI is rendered by ``js/wan_director_timeline.js``. The
    hidden string widgets ``timeline_data`` / ``local_prompts`` /
    ``segment_lengths`` / ``guide_strength`` / ``negative_prompts`` are
    serialised by the JS extension and parsed here on execute.
    """

    CATEGORY = "C2C/Wan_Director"
    FUNCTION = "execute"

    DESCRIPTION = (
        "Visual timeline director for Wan 2.1 / 2.2 / Fun / Animate. "
        "Drag image, text and audio clips onto the timeline, choose a "
        "model_variant, and the node emits the matching CONDITIONING, "
        "LATENT, FPS and AUDIO bundle. Inspired by WhatDreamsCost / "
        "LTX Director (MIT), redesigned for the Wan VAE shape (16ch, /8 "
        "spatial, /4 temporal) and Wan-specific options (dual-CFG for "
        "2.2, reference image for Animate, control track for Fun)."
    )

    RETURN_TYPES = (
        "MODEL",
        "CONDITIONING",
        "CONDITIONING",
        "LATENT",
        "FLOAT",
        "AUDIO",
        "IMAGE",
        "STRING",
        "WANVIDEOMODEL",
        "WANVIDEOTEXTEMBEDS",
        "STRING",
        "IMAGE",
        "MASK",
        "STRING",
        "STRING",
    )
    RETURN_NAMES = (
        "model",
        "positive",
        "negative",
        "video_latent",
        "frame_rate",
        "combined_audio",
        "reference_image",
        "info",
        "wan_model",
        "wan_text_embeds",
        "tracks_program",
        "control_video",
        "control_mask",
        "guide_data",
        "quality_recipe",
    )
    OUTPUT_TOOLTIPS = (
        "Native MODEL (patched with PromptRelay + NAG + PAG + Dynamic CFG + AsymFlow as enabled). When backend='kijai', passthrough of input `model` if connected.",
        "Positive CONDITIONING (native branch) with guide_strength embedded. Empty list when backend='kijai'.",
        "Negative CONDITIONING (native branch). Empty list when backend='kijai'.",
        "Wan latent: VAE-encoded reference for i2v (if VAE connected) or empty latent. Channels=16, /8 spatial, /4 temporal.",
        "Frame rate echoed for downstream sampler/saver nodes.",
        "Audio waveform mixed from the timeline's audio segments.",
        "Reference image (first image clip) — used by Wan I2V/Animate as the start/reference frame. Black image if none.",
        "JSON: resolved backend, variant, latent shape, segment count, audio sample rate, prompt-relay status, quality stack status, warnings.",
        "Kijai WANVIDEOMODEL (only populated when backend='kijai'; PromptRelay-patched in place if enabled).",
        "Kijai WANVIDEOTEXTEMBEDS dict (only populated when backend='kijai'). Feed directly into WanVideoSampler.",
        "JSON: timeline schema_version + normalised lora/camera/seed/pose tracks for downstream applier nodes.",
        "Control video passthrough (IMAGE) for Wan Fun/Animate. None if not connected.",
        "Control mask passthrough (MASK) for Wan Fun Inpaint. None if not connected.",
        "JSON: per-segment guide_strength values for downstream guide applier nodes.",
        "JSON: quality recipe config (SLG, FETA, RIFLEx, cache, FreeInit, phase-shift) for downstream sampler.",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "backend": (["native", "kijai"], {
                    "default": "native",
                    "tooltip": (
                        "Which video-model stack to drive.\n\n"
                        "  native — ComfyUI's built-in Wan implementation. Connect `model` + `clip`.\n"
                        "  kijai  — Kijai's ComfyUI-WanVideoWrapper. Connect `wan_model` + `wan_t5` (optional sockets).\n\n"
                        "PromptRelay (if enabled) is applied to whichever backbone is active and "
                        "falls back to the generic-introspection patcher for any third-party model."
                    ),
                }),
                "model_variant": (VARIANT_KEYS, {
                    "default": "wan2.1_i2v",
                    "tooltip": "Which Wan family / mode this timeline targets. "
                               "Changes which optional sliders are visible and how "
                               "the latent + conditioning are assembled.",
                }),
                "duration_frames": ("INT", {
                    "default": 81, "min": 1, "max": 10000, "step": 1,
                    "tooltip": "Total timeline length in pixel-space frames. Wan 2.x "
                               "defaults to 81 frames (≈ 5 s @ 16 fps).",
                }),
                "duration_seconds": ("FLOAT", {
                    "default": 5.0, "min": 0.1, "max": 1000.0, "step": 0.01,
                    "tooltip": "Total timeline duration in seconds (synced from frames by the UI).",
                }),
                "frame_rate": ("FLOAT", {
                    "default": 16.0, "min": 1.0, "max": 240.0, "step": 1.0,
                    "tooltip": "FPS. Wan 2.x is trained at 16 fps; raise for slow-motion-like output.",
                }),
                "global_prompt": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": "Persistent context prepended to every per-clip prompt "
                               "(characters, lighting, style anchors).",
                }),
                # Timeline-managed hidden strings (the JS extension writes these).
                "timeline_data":     ("STRING", {"multiline": True, "default": ""}),
                "local_prompts":     ("STRING", {"multiline": True, "default": ""}),
                "negative_prompts":  ("STRING", {"multiline": True, "default": ""}),
                "segment_lengths":   ("STRING", {"default": ""}),
                "guide_strength":    ("STRING", {"default": ""}),
                "display_mode":      (["seconds", "frames"], {"default": "seconds"}),
                # Resolution + resize policy
                "custom_width":  ("INT", {"default": 832, "min": 0, "max": 8192, "step": 8,
                                          "tooltip": "Target width. 0 = inherit from first image clip."}),
                "custom_height": ("INT", {"default": 480, "min": 0, "max": 8192, "step": 8,
                                          "tooltip": "Target height. 0 = inherit from first image clip."}),
                "resize_method": (["maintain aspect ratio", "stretch to fit", "pad", "crop"],
                                  {"default": "maintain aspect ratio"}),
                # Wan 2.2 dual-CFG (only meaningful when variant is wan2.2_*)
                "cfg_high_noise": ("FLOAT", {
                    "default": 3.5, "min": 0.0, "max": 20.0, "step": 0.1,
                    "tooltip": "Wan 2.2 high-noise expert CFG. Ignored for non-2.2 variants.",
                }),
                "cfg_low_noise":  ("FLOAT", {
                    "default": 3.5, "min": 0.0, "max": 20.0, "step": 0.1,
                    "tooltip": "Wan 2.2 low-noise expert CFG. Ignored for non-2.2 variants.",
                }),
                # Wan Animate
                "ref_strength":  ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "Wan Animate reference-image influence. Ignored for other variants.",
                }),
                # EverAnimate (only meaningful when variant is wan2.2_animate_everanimate)
                "everanimate_stage": (list(EVERANIMATE_STAGES), {
                    "default": "stage2_480p",
                    "tooltip": (
                        "Which EverAnimate LoRA checkpoint to apply on top of Wan2.2-Animate-14B:\n"
                        "  stage1_480p     — base motion fidelity (480p training).\n"
                        "  stage2_480p     — Restorative Flow Matching, sharper temporal coherence (recommended).\n"
                        "  stage3_720p_beta — 720p beta with higher detail; needs more VRAM.\n"
                        "Ignored for non-EverAnimate variants."
                    ),
                }),
                "everanimate_num_chunks": ("INT", {
                    "default": 1, "min": 1, "max": 50, "step": 1,
                    "tooltip": (
                        "Long-horizon chunk count. 1 = single ~5 s clip (standard Wan2.2-Animate). "
                        "≥2 enables EverAnimate's Persistent Latent Propagation across anchor frames "
                        "for minute-scale animation. Ignored for non-EverAnimate variants."
                    ),
                }),
                "everanimate_overlap_frames": ("INT", {
                    "default": 4, "min": 0, "max": 16, "step": 1,
                    "tooltip": (
                        "Frames of latent overlap between consecutive chunks (anchor padding). "
                        "Higher = smoother seams but slower. Ignored if num_chunks=1 or non-EverAnimate variant."
                    ),
                }),
                "everanimate_lora_strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "EverAnimate rank-32 LoRA strength. 1.0 = paper default. Ignored for non-EverAnimate variants.",
                }),
                "everanimate_anchor_strategy": (list(EVERANIMATE_ANCHOR_STRATEGIES), {
                    "default": "auto",
                    "tooltip": (
                        "Anchor-frame selection for chunks 2+:\n"
                        "  auto                 — first chunk uses first frame only, later chunks use first + 3 random.\n"
                        "  first_only           — always 1 anchor (faster, slight quality loss).\n"
                        "  first_plus_random_3  — always 4 anchors (paper-default; best quality).\n"
                        "Ignored for non-EverAnimate variants."
                    ),
                }),
                # Audio target
                "audio_target": (["music_44k_stereo", "speech_16k_mono"], {
                    "default": "music_44k_stereo",
                    "tooltip": "Output AUDIO format. Use `speech_16k_mono` if you intend "
                               "to feed Wan-S2V or any speech-driven pipeline downstream.",
                }),
                # Embedded PromptRelay (universal — works for native + Kijai +
                # any third-party diffusion backbone via the generic-fallback patcher).
                "enable_prompt_relay": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "Internal PromptRelay: bias each backbone cross-attention block so the "
                        "timeline's per-clip prompts only steer their own frame span. On by default — "
                        "the whole point of the timeline. Works on native ComfyUI MODEL, Kijai "
                        "WANVIDEOMODEL, and arbitrary video-diffusion models (auto-falls back to "
                        "generic introspection). With 2+ text/image clips the per-clip prompts become "
                        "the local prompts and `global_prompt` is the anchor; with 0 or 1 clip it is a "
                        "no-op. Turn off to encode one flat prompt for the whole clip."
                    ),
                }),
                "prompt_relay_epsilon": ("FLOAT", {
                    "default": 1e-3, "min": 1e-6, "max": 0.99, "step": 1e-4,
                    "tooltip": "PromptRelay penalty decay. <0.1 = sharp boundaries; ≥0.5 softer.",
                }),
                # ── HDR / Advanced Guidance (Radiance-style) ──────────
                "enable_dynamic_cfg": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "Cosine-ramped dynamic CFG across denoising steps. "
                        "Early steps get 1.2× CFG (stronger structure), late steps get 0.7× "
                        "(softer detail). Prevents oversaturation and improves quality."
                    ),
                }),
                "guidance_rescale_phi": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": (
                        "Guidance rescale (phi). Rescales guided output to match conditional "
                        "std-deviation, preventing color oversaturation at high CFG. "
                        "0=off, 0.7=recommended for Wan 2.2. Requires enable_dynamic_cfg=True."
                    ),
                }),
                "pag_scale": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 5.0, "step": 0.1,
                    "tooltip": (
                        "Perturbed Attention Guidance scale. Improves prompt adherence by "
                        "guiding away from identity-attention outputs. 0=off, 1.0–3.0 typical."
                    ),
                }),
                "enable_phase_shift": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "Phase-shift sampling: Euler for early steps (structure), "
                        "DPM++ 2M for late steps (detail). Uses smooth sigma crossfade."
                    ),
                }),
                "phase_shift_pct": ("FLOAT", {
                    "default": 0.70, "min": 0.3, "max": 0.95, "step": 0.05,
                    "tooltip": "Step fraction where phase-shift transitions from Euler to DPM++.",
                }),
                "vae_fp32_decode": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "Force VAE decode in fp32 for maximum quality. Wan VAE produces "
                        "significantly better results in fp32 (recommended by HuggingFace). "
                        "Uses more VRAM during decode only."
                    ),
                }),
                "enable_multi_clip": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "Multi-slot CLIP conditioning. Split prompts into structure (early) "
                        "and detail (late) phases for finer control over generation."
                    ),
                }),
                "structure_prompt": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": (
                        "Structure prompt (active during early denoising, 0–35%). "
                        "Focus on composition, layout, camera angles, scene description. "
                        "Only used when enable_multi_clip=True."
                    ),
                }),
                "detail_prompt": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": (
                        "Detail prompt (active during late denoising, 55–100%). "
                        "Focus on textures, materials, lighting, color grading. "
                        "Only used when enable_multi_clip=True."
                    ),
                }),
                # ── Quality Stack (model patches applied before sampling) ──
                "enable_nag": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Normalized Attention Guidance: boosts prompt adherence via attention-space CFG.",
                }),
                "nag_scale": ("FLOAT", {
                    "default": 11.0, "min": 0.0, "max": 30.0, "step": 0.5,
                    "tooltip": "NAG guidance scale. Higher = stronger guidance.",
                }),
                "enable_asymflow": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "AsymFlow time-shift for improved temporal consistency.",
                }),
                "asymflow_shift": ("FLOAT", {
                    "default": 3.0, "min": 0.1, "max": 20.0, "step": 0.1,
                    "tooltip": "AsymFlow shift parameter.",
                }),
                # ── Quality recipe (config passed to downstream sampler) ──
                "cache_type": (["none", "teacache", "magcache", "easycache"], {
                    "default": "none",
                    "tooltip": "Inference caching strategy. Speeds up generation by skipping redundant transformer passes.",
                }),
                "cache_threshold": ("FLOAT", {
                    "default": 0.10, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Cache skip threshold. Lower = more aggressive caching (faster but less accurate).",
                }),
                "enable_slg": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Skip-Layer Guidance: run a second pass with layers removed for quality boost.",
                }),
                "slg_layers": ("STRING", {
                    "default": "",
                    "tooltip": "Comma-separated layer indices to skip (e.g. '7,8,9'). Empty = auto-select.",
                }),
                "slg_scale": ("FLOAT", {
                    "default": 0.7, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "SLG guidance scale.",
                }),
                "enable_feta": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Frequency-Enhanced Temporal Attention for better frame coherence.",
                }),
                "feta_scale": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "FETA scale. 0 = off.",
                }),
                "enable_riflex": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "RIFLEx RoPE rescaling for length extrapolation beyond training length.",
                }),
                "riflex_k": ("INT", {
                    "default": 2, "min": 1, "max": 8, "step": 1,
                    "tooltip": "Number of lowest RoPE frequencies to rescale.",
                }),
            },
            "optional": {
                "model":            ("MODEL", {"tooltip": "Native Wan MODEL (2.1 / 2.2 / Fun / Animate). Required when backend='native'."}),
                "clip":             ("CLIP",  {"tooltip": "Text encoder paired with the Wan model (UMT5 for 2.x). Required when backend='native'."}),
                "vae":              ("VAE",   {"tooltip": "Wan VAE. If connected, the Director encodes reference images for i2v and forces fp32 when vae_fp32_decode=True."}),
                "clip_vision":      ("CLIP_VISION", {"tooltip": "CLIP Vision model for image embeddings (Kijai i2v/Animate). If not connected, image_embeds output is None."}),
                "optional_latent":  ("LATENT", {"tooltip": "Override the auto-built empty latent."}),
                "control_video":    ("IMAGE",  {"tooltip": "For Wan Fun / Animate: control sequence (depth/pose/canny). Passed through to control_video output."}),
                "control_mask":     ("MASK",   {"tooltip": "For Wan Fun Inpaint: per-frame mask track. Passed through to control_mask output."}),
                "wan_model":        ("WANVIDEOMODEL",    {"tooltip": "Kijai WanVideoWrapper model patcher (required when backend='kijai')."}),
                "wan_t5":           ("WANTEXTENCODER",   {"tooltip": "Kijai T5 text encoder (required when backend='kijai')."}),
                "t5gemma":          ("T5GEMMA_ENCODER",  {"tooltip": "R&D: connect a 'T5Gemma Encoder Loader (WNE)' to encode the Director's composed "
                                                                     "prompt through T5Gemma instead of CLIP (backend='native'). NOTE: T5Gemma hidden states "
                                                                     "are a different space/width from Wan's UMT5-XXL — this only works on a Wan model "
                                                                     "finetuned/adapted for T5Gemma; a stock Wan checkpoint will error or produce garbage."}),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return hash_args_and_kwargs(**kwargs)

    # ── Execute ──────────────────────────────────────────────────────

    def execute(self,
                backend, model_variant, duration_frames, duration_seconds,
                frame_rate, global_prompt, timeline_data, local_prompts,
                negative_prompts, segment_lengths, guide_strength, display_mode,
                custom_width, custom_height, resize_method,
                cfg_high_noise, cfg_low_noise, ref_strength,
                everanimate_stage, everanimate_num_chunks,
                everanimate_overlap_frames, everanimate_lora_strength,
                everanimate_anchor_strategy,
                audio_target,
                enable_prompt_relay, prompt_relay_epsilon,
                enable_dynamic_cfg=False, guidance_rescale_phi=0.0,
                pag_scale=0.0, enable_phase_shift=False, phase_shift_pct=0.70,
                vae_fp32_decode=True, enable_multi_clip=False,
                structure_prompt="", detail_prompt="",
                enable_nag=False, nag_scale=11.0,
                enable_asymflow=False, asymflow_shift=3.0,
                cache_type="none", cache_threshold=0.10,
                enable_slg=False, slg_layers="", slg_scale=0.7,
                enable_feta=False, feta_scale=0.5,
                enable_riflex=False, riflex_k=2,
                model=None, clip=None,
                vae=None, clip_vision=None,
                optional_latent=None, control_video=None, control_mask=None,
                wan_model=None, wan_t5=None, t5gemma=None):

        # Per-backend input validation — clearer than ComfyUI's generic
        # "Required input is missing" because the user genuinely doesn't
        # need both pairs at once.
        if backend == "native":
            # T5Gemma (when connected) provides the text encoder, so CLIP is
            # optional in that R&D path; the Wan MODEL is always required.
            _need = [("model", model)] if t5gemma is not None else [("model", model), ("clip", clip)]
            missing = [n for n, v in _need if v is None]
            if missing:
                raise ValueError(
                    f"WanDirector: backend='native' requires {', '.join(missing)}. "
                    "Either wire a native Wan MODEL + CLIP (UMT5) pair (or a T5Gemma encoder in "
                    "place of CLIP), or switch backend='kijai' and wire wan_model + wan_t5 instead."
                )
        elif backend == "kijai":
            missing = [n for n, v in (("wan_model", wan_model), ("wan_t5", wan_t5)) if v is None]
            if missing:
                raise ValueError(
                    f"WanDirector: backend='kijai' requires {', '.join(missing)}. "
                    "Wire Kijai's WanVideoModelLoader + WanVideoTextEncode (or switch "
                    "backend='native')."
                )

        if model_variant not in VARIANT_TABLE:
            raise ValueError(f"Unknown model_variant '{model_variant}'. "
                             f"Choices: {VARIANT_KEYS}")
        vcfg = VARIANT_TABLE[model_variant]
        warnings: list[str] = []

        # ── Parse timeline for image clips → reference frame ─────────
        try:
            tdata = json.loads(timeline_data) if timeline_data else {}
        except json.JSONDecodeError as exc:
            warnings.append(f"timeline_data JSON parse failed: {exc}; treating as empty.")
            tdata = {}

        # Schema v1 → v2 migration (idempotent). After this, tdata has
        # schema_version, segments, audioSegments, and all four v2 track
        # arrays guaranteed to exist (possibly empty).
        tdata, migration_notes = _migrate_timeline(tdata)
        warnings.extend(migration_notes)

        # Track mutes (timeline UI left-gutter dots): a muted track is
        # excluded from the build entirely — same semantics as LTX Director's
        # toggleable tracks. Keys mirror the JS: image/audio/video + v2 keys.
        _muted = tdata.get("trackMuted") or {}
        if isinstance(_muted, dict) and any(_muted.values()):
            _mute_map = {"image": "segments", "audio": "audioSegments", "video": "motionSegments"}
            for _mk, _arr_key in _mute_map.items():
                if _muted.get(_mk):
                    tdata[_arr_key] = []
                    warnings.append(f"track '{_mk}' muted in timeline; skipped.")
            for _tk in _TRACK_KEYS:
                if _muted.get(_tk):
                    tdata[_tk] = []
                    warnings.append(f"track '{_tk}' muted in timeline; skipped.")

        # Compile + validate the four v2 tracks. Each compiler returns
        # (normalised_list, per_track_warnings) and never raises.
        _total_frames = max(1, int(duration_frames))
        lora_program,   _w_lora   = _compile_lora_program(tdata.get("loraSegments", []),   _total_frames)
        camera_program, _w_cam    = _compile_camera_program(tdata.get("cameraSegments", []), _total_frames, float(frame_rate))
        seed_program,   _w_seed   = _compile_seed_program(tdata.get("seedSegments", []),   _total_frames)
        pose_program,   _w_pose   = _compile_pose_program(tdata.get("poseSegments", []),   _total_frames)
        control_program, _w_ctrl  = _compile_control_program(tdata.get("motionSegments", []), _total_frames)
        track_warnings = _w_lora + _w_cam + _w_seed + _w_pose + _w_ctrl
        warnings.extend(track_warnings)

        img_segs = sorted(
            [s for s in tdata.get("segments", [])
             if s.get("type", "image") == "image"
             and (s.get("imageFile") or s.get("imageB64"))
             and int(s.get("start", 0)) < duration_frames],
            key=lambda s: s.get("start", 0),
        )

        # Resolve output dimensions: prefer user override; else first image.
        out_w, out_h = int(custom_width), int(custom_height)
        ref_image = torch.zeros((1, max(64, out_h or 480), max(64, out_w or 832), 3),
                                dtype=torch.float32)
        if img_segs:
            first = _load_image_tensor(img_segs[0])
            if out_w <= 0 or out_h <= 0:
                out_h, out_w = first.shape[1], first.shape[2]
            ref_image = _resize_image(first, out_w, out_h, resize_method, vcfg["spatial_div"])
            out_h, out_w = ref_image.shape[1], ref_image.shape[2]
        else:
            out_w = _snap(out_w or 832, vcfg["spatial_div"])
            out_h = _snap(out_h or 480, vcfg["spatial_div"])
            ref_image = torch.zeros((1, out_h, out_w, 3), dtype=torch.float32)
            if vcfg["needs_image"]:
                warnings.append(f"Variant '{model_variant}' expects an image clip; "
                                "using a black reference frame.")

        # ── Build positive / negative prompt text ────────────────────
        local_list = [p.strip() for p in (local_prompts or "").split("|") if p.strip()]
        neg_list   = [p.strip() for p in (negative_prompts or "").split("|") if p.strip()]

        pos_text = (global_prompt.strip() + ". " if global_prompt.strip() else "") + \
                   (" ".join(local_list) if local_list else global_prompt.strip())
        if not pos_text.strip():
            pos_text = "a cinematic shot"
            warnings.append("Empty positive prompt; defaulted to 'a cinematic shot'.")

        neg_text = " ".join(neg_list) if neg_list else \
                   "low quality, blurry, distorted, watermark, text, jpeg artifacts"

        # ── Latent ───────────────────────────────────────────────────
        if optional_latent is not None:
            latent = optional_latent
        else:
            latent = _empty_wan_latent(model_variant, out_w, out_h, int(duration_frames))

        # ── Audio mix ────────────────────────────────────────────────
        audio_out = _build_combined_audio(
            timeline_data, int(duration_frames), float(frame_rate), audio_target,
        )

        # ── VAE: encode reference image for i2v ─────────────────────
        vae_encoded = False
        if vae is not None and vcfg["needs_image"] and optional_latent is None and img_segs:
            try:
                if vae_fp32_decode:
                    from .features._local_vae_hdr import force_fp32_vae
                    force_fp32_vae(vae)
                    warnings.append("VAE forced to fp32 for encode/decode quality.")

                # ComfyUI VAE.encode expects (B,H,W,C) float32 [0,1] and does
                # its own movedim(-1,1) + temporal handling internally — do NOT
                # pre-permute or unsqueeze (a 5D input corrupts the encode).
                # ref_image is already [B,H,W,3] from _load_image_tensor.
                encoded = vae.encode(ref_image[:, :, :, :3])
                if isinstance(encoded, dict):
                    encoded_samples = encoded.get("samples", encoded.get("latent", None))
                else:
                    encoded_samples = encoded
                if encoded_samples is not None:
                    # Merge encoded reference frame(s) into the empty latent.
                    # Wan latent is 5D (B,C,T,H/8,W/8); the encode of a single
                    # frame may come back 5D (B,C,T',H/8,W/8) or 4D (B,C,H/8,W/8).
                    empty_s = latent["samples"]
                    if encoded_samples.ndim == 5 and empty_s.ndim == 5:
                        nt = min(encoded_samples.shape[2], empty_s.shape[2])
                        empty_s[:, :, :nt] = encoded_samples[:, :, :nt]
                    elif encoded_samples.ndim == 4 and empty_s.ndim == 5:
                        empty_s[:, :, 0] = encoded_samples
                    elif encoded_samples.shape == empty_s.shape:
                        empty_s[:] = encoded_samples
                    latent = {"samples": empty_s}
                    vae_encoded = True
                    warnings.append("VAE-encoded reference image into latent frame 0 for i2v.")
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"VAE encode failed ({exc}); using empty latent.")
                log.warning("VAE encode failed: %s", exc)

        # ── Parse guide_strength per-segment ──────────────────────────
        guide_data_parsed = []
        if guide_strength and guide_strength.strip():
            try:
                gs_values = [float(x.strip()) for x in guide_strength.split(",") if x.strip()]
                guide_data_parsed = gs_values
            except (ValueError, TypeError) as exc:
                warnings.append(f"guide_strength parse failed ({exc}); using defaults.")

        guide_data_json = json.dumps({
            "per_segment": guide_data_parsed,
            "n_segments": len(guide_data_parsed),
            "duration_frames": int(duration_frames),
        })

        # ── Backend dispatch ─────────────────────────────────────────
        out_model_native = model
        out_pos = []
        out_neg = []
        out_wan_model = wan_model
        out_text_embeds = None

        relay_used = False
        relay_note = ""

        def _clone_model_once(mdl, original):
            """Clone model if not already cloned."""
            return mdl.clone() if mdl is original else mdl

        if backend == "native":
            if t5gemma is not None:
                # R&D path: encode the composed prompt through T5Gemma instead of
                # CLIP. Different embedding space/width from UMT5 — only valid on a
                # Wan model adapted for T5Gemma. Warned loudly; CLIP is ignored.
                pos_cond = _t5gemma_encode(t5gemma, pos_text, 512)
                neg_cond = _t5gemma_encode(t5gemma, neg_text, 512)
                warnings.append(
                    f"T5Gemma text encoder active ('{t5gemma.get('name', '?')}'): prompts encoded "
                    "via T5Gemma, CLIP ignored. R&D — requires a Wan model adapted for T5Gemma; "
                    "a stock UMT5-trained checkpoint will error or produce garbage.")
            # Multi-CLIP slot conditioning
            elif enable_multi_clip and clip and (structure_prompt.strip() or detail_prompt.strip()):
                try:
                    from .features._local_multi_clip import build_multi_slot_conditioning
                    pos_cond = build_multi_slot_conditioning(
                        clip,
                        structure_prompt=structure_prompt or pos_text,
                        detail_prompt=detail_prompt or pos_text,
                        style_prompt=global_prompt,
                    )
                    if not pos_cond:
                        pos_cond = clip.encode_from_tokens_scheduled(clip.tokenize(pos_text))
                        warnings.append("Multi-CLIP produced empty conditioning; fell back to single prompt.")
                    else:
                        warnings.append(f"Multi-CLIP: {len(pos_cond)} conditioning slots active.")
                except Exception as exc:
                    pos_cond = clip.encode_from_tokens_scheduled(clip.tokenize(pos_text))
                    warnings.append(f"Multi-CLIP failed ({exc}); fell back to single prompt.")
                neg_cond = clip.encode_from_tokens_scheduled(clip.tokenize(neg_text))
            else:
                pos_cond = clip.encode_from_tokens_scheduled(clip.tokenize(pos_text))
                neg_cond = clip.encode_from_tokens_scheduled(clip.tokenize(neg_text))

            if vcfg["dual_cfg"]:
                for c in pos_cond:
                    c[1]["wan_cfg_high_noise"] = float(cfg_high_noise)
                    c[1]["wan_cfg_low_noise"]  = float(cfg_low_noise)
                for c in neg_cond:
                    c[1]["wan_cfg_high_noise"] = float(cfg_high_noise)
                    c[1]["wan_cfg_low_noise"]  = float(cfg_low_noise)
            if vcfg["ref_image"]:
                for c in pos_cond:
                    c[1]["wan_ref_strength"] = float(ref_strength)

            # Embed guide_strength into conditioning metadata
            if guide_data_parsed:
                for c in pos_cond:
                    c[1]["wan_guide_strength"] = guide_data_parsed

            # ── Model patching stack (applied in order) ──────────────
            out_model_native = model

            # 1. Dynamic CFG + guidance rescale
            if enable_dynamic_cfg and out_model_native is not None:
                try:
                    from .features._local_dynamic_guidance import build_dynamic_cfg_patch
                    base_cfg = float(cfg_high_noise if vcfg["dual_cfg"] else 6.0)
                    patch_fn = build_dynamic_cfg_patch(
                        base_cfg=base_cfg, total_steps=0, denoise=1.0,
                        phi=float(guidance_rescale_phi),
                    )
                    out_model_native = _clone_model_once(out_model_native, model)
                    out_model_native.set_model_sampler_cfg_function(patch_fn)
                    warnings.append(
                        f"Dynamic CFG active (phi={guidance_rescale_phi:.2f}). "
                        "CFG ramps adaptively based on sampler step count."
                    )
                except Exception as exc:
                    warnings.append(f"Dynamic CFG failed ({exc}); using static CFG.")
                    log.warning("Dynamic CFG setup failed: %s", exc)

            # 2. PAG (Perturbed Attention Guidance)
            if pag_scale > 0.0 and out_model_native is not None:
                try:
                    from .features._local_pag import apply_pag_to_model
                    out_model_native = apply_pag_to_model(out_model_native, float(pag_scale))
                    warnings.append(f"PAG active (scale={pag_scale:.1f}).")
                except Exception as exc:
                    warnings.append(f"PAG failed ({exc}); skipped.")
                    log.warning("PAG setup failed: %s", exc)

            # 3. NAG (Normalized Attention Guidance)
            if enable_nag and out_model_native is not None:
                try:
                    from ._kijai_adapter import apply_nag
                    out_model_native = _clone_model_once(out_model_native, model)
                    out_model_native = apply_nag(out_model_native, scale=float(nag_scale))
                    warnings.append(f"NAG active (scale={nag_scale:.1f}).")
                except Exception as exc:
                    warnings.append(f"NAG failed ({exc}); skipped.")
                    log.warning("NAG setup failed: %s", exc)

            # 4. AsymFlow time-shift
            if enable_asymflow and out_model_native is not None:
                try:
                    from ._kijai_adapter import apply_asymflow
                    out_model_native = _clone_model_once(out_model_native, model)
                    out_model_native = apply_asymflow(
                        out_model_native, shift=float(asymflow_shift),
                    )
                    warnings.append(f"AsymFlow active (shift={asymflow_shift:.1f}).")
                except Exception as exc:
                    warnings.append(f"AsymFlow failed ({exc}); skipped.")
                    log.warning("AsymFlow setup failed: %s", exc)

            # 5. SLG (Selective Layer Guidance)
            if enable_slg and out_model_native is not None:
                try:
                    from .features._local_slg import apply_slg
                    _slg_layers = [int(x.strip()) for x in slg_layers.split(",") if x.strip()] if slg_layers.strip() else []
                    out_model_native = _clone_model_once(out_model_native, model)
                    out_model_native = apply_slg(
                        out_model_native, layers=_slg_layers,
                        scale=float(slg_scale), start_percent=0.0, end_percent=1.0,
                    )
                    warnings.append(f"SLG active (layers={_slg_layers or 'auto'}, scale={slg_scale:.2f}).")
                except Exception as exc:
                    warnings.append(f"SLG failed ({exc}); config emitted in quality_recipe.")
                    log.warning("SLG apply failed: %s", exc)

            # 6. FETA (Frequency-Enhanced Temporal Attention)
            if enable_feta and feta_scale > 0 and out_model_native is not None:
                try:
                    from .features._local_feta import apply_feta
                    out_model_native = _clone_model_once(out_model_native, model)
                    out_model_native = apply_feta(
                        out_model_native, feta_scale=float(feta_scale),
                    )
                    warnings.append(f"FETA active (scale={feta_scale:.2f}).")
                except Exception as exc:
                    warnings.append(f"FETA failed ({exc}); config emitted in quality_recipe.")
                    log.warning("FETA apply failed: %s", exc)

            # 7. PromptRelay (internal — on by default, driven by the timeline)
            relay_cond = None
            if enable_prompt_relay and t5gemma is not None:
                relay_note = "skipped (PromptRelay needs the CLIP tokenizer; T5Gemma encode is active)"
            elif enable_prompt_relay and clip is None:
                relay_note = "skipped (no CLIP connected)"
            elif enable_prompt_relay and len(local_list) >= 2:
                try:
                    out_model_native, relay_cond = _apply_prompt_relay_native(
                        out_model_native, clip, latent, global_prompt, local_list,
                        segment_lengths, float(prompt_relay_epsilon),
                    )
                    relay_used = True
                    relay_note = f"native PromptRelay applied ({len(local_list)} segments)"
                except Exception as exc:                            # noqa: BLE001
                    relay_cond = None
                    warnings.append(f"PromptRelay (native) skipped: {exc}")
                    log.warning("PromptRelay native failed: %s", exc)
            elif enable_prompt_relay:
                relay_note = "skipped (need 2+ local prompts)"

            # Use the relay-produced segment-aware conditioning when it was
            # built (previously discarded — the relay only half-worked). Carry
            # over the metadata we attached to pos_cond (dual-CFG, ref strength,
            # guide strength) so switching to relay_cond doesn't lose it.
            if relay_cond:
                meta_src = pos_cond[0][1] if (pos_cond and len(pos_cond[0]) > 1
                                              and isinstance(pos_cond[0][1], dict)) else {}
                carry = {k: meta_src[k] for k in (
                    "wan_cfg_high_noise", "wan_cfg_low_noise",
                    "wan_ref_strength", "wan_guide_strength") if k in meta_src}
                if carry:
                    for c in relay_cond:
                        if len(c) > 1 and isinstance(c[1], dict):
                            c[1].update(carry)
                out_pos = relay_cond
            else:
                out_pos = pos_cond
            out_neg = neg_cond

        elif backend == "kijai":
            if wan_model is None or wan_t5 is None:
                raise ValueError(
                    "backend='kijai' requires both `wan_model` (WANVIDEOMODEL) and "
                    "`wan_t5` (WANTEXTENCODER) optional sockets to be connected."
                )
            out_wan_model, out_text_embeds, relay_used, relay_note = \
                _apply_kijai_branch(
                    wan_model, wan_t5, pos_text, neg_text,
                    local_list, segment_lengths, int(duration_frames),
                    float(prompt_relay_epsilon), enable_prompt_relay,
                )
            out_pos = []
            out_neg = []
        else:
            raise ValueError(f"Unknown backend '{backend}'. Choices: native, kijai.")

        # EverAnimate descriptor — only populated when the variant opts in.
        # Validated/clamped here so the runner can trust the values.
        ea_active = bool(vcfg.get("everanimate", False))
        ea_stage = str(everanimate_stage)
        if ea_stage not in EVERANIMATE_STAGES:
            warnings.append(
                f"everanimate_stage='{ea_stage}' not in {EVERANIMATE_STAGES}; "
                f"falling back to 'stage2_480p'."
            )
            ea_stage = "stage2_480p"
        ea_strategy = str(everanimate_anchor_strategy)
        if ea_strategy not in EVERANIMATE_ANCHOR_STRATEGIES:
            warnings.append(
                f"everanimate_anchor_strategy='{ea_strategy}' not in "
                f"{EVERANIMATE_ANCHOR_STRATEGIES}; falling back to 'auto'."
            )
            ea_strategy = "auto"
        ea_num_chunks = max(1, min(50, int(everanimate_num_chunks)))
        ea_overlap = max(0, min(16, int(everanimate_overlap_frames)))
        ea_strength = max(0.0, min(2.0, float(everanimate_lora_strength)))
        everanimate_program = {
            "active":           ea_active,
            "stage":            ea_stage,
            "num_chunks":       ea_num_chunks,
            "overlap_frames":   ea_overlap,
            "lora_strength":    ea_strength,
            "anchor_strategy":  ea_strategy,
            # Hint to the runner where the LoRA file lives. Kept loose so
            # users can drop the file in any folder_paths("loras") root.
            "lora_filename":    f"everanimate_{ea_stage}.safetensors",
            "base_model":       "Wan-AI/Wan2.2-Animate-14B",
        }

        # ── Build quality_recipe for downstream sampler ─────────────
        slg_layer_list = []
        if enable_slg and slg_layers.strip():
            try:
                slg_layer_list = [int(x.strip()) for x in slg_layers.split(",") if x.strip()]
            except (ValueError, TypeError):
                warnings.append("slg_layers parse failed; SLG will auto-select layers.")

        quality_recipe = {
            "nag": {"enabled": bool(enable_nag), "scale": float(nag_scale), "tau": 2.5, "alpha": 0.25},
            "slg": {
                "enabled": bool(enable_slg),
                "skip_layers": slg_layer_list,
                "scale": float(slg_scale),
                "start_pct": 0.0,
                "end_pct": 1.0,
            },
            "feta": {"enabled": bool(enable_feta), "scale": float(feta_scale), "freq_center": 0.20, "freq_bandwidth": 0.15},
            "riflex": {"enabled": bool(enable_riflex), "k": int(riflex_k), "source_len": 81, "target_len": int(duration_frames)},
            "cache": {"type": str(cache_type), "threshold": float(cache_threshold), "max_skips": 5},
            "freeinit": {"enabled": False, "filter_type": "butterworth", "d_s": 1.0, "d_t": 1.0},
            "phase_shift": {
                "enabled": bool(enable_phase_shift),
                "split_pct": float(phase_shift_pct),
                "scheduler_early": "simple",
                "scheduler_late": "simple",
                "shift": 8.0,
            },
            "vae": {
                "fp32_decode": bool(vae_fp32_decode),
                "spatial_tiling": True,
                "tile_size": 256,
                "overlap": 32,
            },
            "dynamic_cfg": {
                "enabled": bool(enable_dynamic_cfg),
                "phi": float(guidance_rescale_phi),
            },
            "pag": {"enabled": pag_scale > 0.0, "scale": float(pag_scale)},
            "asymflow": {"enabled": bool(enable_asymflow), "shift": float(asymflow_shift)},
            "multi_clip": {"enabled": bool(enable_multi_clip)},
        }
        quality_recipe_json = json.dumps(quality_recipe, separators=(",", ":"))

        # Count active quality features
        active_features = [k for k, v in quality_recipe.items()
                           if isinstance(v, dict) and v.get("enabled")]

        info = json.dumps({
            "backend":      backend,
            "variant":      model_variant,
            "label":        vcfg["label"],
            "out_size":     [out_w, out_h],
            "frames":       int(duration_frames),
            "fps":          float(frame_rate),
            "latent_shape": list(latent["samples"].shape),
            "vae_encoded":  vae_encoded,
            "n_text":       len(local_list),
            "n_neg":        len(neg_list),
            "n_image":      len(img_segs),
            "n_lora":       len(lora_program),
            "n_camera":     len(camera_program),
            "n_seed":       len(seed_program),
            "n_pose":       len(pose_program),
            "guide_strengths": guide_data_parsed,
            "schema_version": _SCHEMA_VERSION,
            "audio_sr":     audio_out["sample_rate"],
            "prompt_relay": {
                "enabled":   bool(enable_prompt_relay),
                "applied":   bool(relay_used),
                "note":      relay_note,
                "epsilon":   float(prompt_relay_epsilon),
            },
            "everanimate":  everanimate_program if ea_active else {"active": False},
            "model_patches": {
                "dynamic_cfg": bool(enable_dynamic_cfg),
                "guidance_rescale_phi": float(guidance_rescale_phi),
                "pag": pag_scale > 0.0,
                "pag_scale": float(pag_scale),
                "nag": bool(enable_nag),
                "nag_scale": float(nag_scale),
                "asymflow": bool(enable_asymflow),
                "asymflow_shift": float(asymflow_shift),
            },
            "quality_features": active_features,
            "quality_recipe": quality_recipe,
            "control_video_connected": control_video is not None,
            "control_clips": [
                {"controlType": c["controlType"], "file": c["fileName"],
                 "start": c["start"], "length": c["length"]}
                for c in control_program
            ],
            "control_mask_connected": control_mask is not None,
            "track_warnings": track_warnings,
            "warnings":     warnings,
        }, indent=2)

        tracks_program = json.dumps({
            "schema_version": _SCHEMA_VERSION,
            "duration_frames": int(duration_frames),
            "frame_rate": float(frame_rate),
            "lora":   lora_program,
            "camera": camera_program,
            "seed":   seed_program,
            "pose":   pose_program,
            "control": control_program,
            "everanimate": everanimate_program,
        }, separators=(",", ":"))

        return (out_model_native, out_pos, out_neg, latent, float(frame_rate),
                audio_out, ref_image, info, out_wan_model, out_text_embeds,
                tracks_program, control_video, control_mask,
                guide_data_json, quality_recipe_json)
