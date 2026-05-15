# Prompt Relay — node classes (refined port).
#
# Four nodes registered:
#   - PromptRelayEncodeC2C        : native MODEL+CLIP path
#   - PromptRelayEncodeSmartC2C   : native, smart-prompt syntax
#   - PromptRelayEncodeKijaiC2C   : Kijai WANVIDEOMODEL + WANTEXTENCODER path
#   - PromptRelayAdvancedOptionsC2C : RELAY_OPTIONS bundle (optional)
#
# Display names follow ComfyUI-CustomNodePacks convention (no _C2C suffix in UI).
# Algorithm credit: Gordon Chen & contributors — see NOTICE.md.

from __future__ import annotations

import logging
from typing import Optional

from ._core import (
    build_segments,
    convert_pixel_to_latent_lengths,
    create_mask_fn,
    distribute_segment_lengths,
    get_raw_tokenizer_native,
    map_token_indices,
)
from ._parser import parse_smart_prompt
from ._patches import (
    detect_native_arch,
    patch_generic,
    patch_kijai,
    patch_native,
)

log = logging.getLogger("C2C.PromptRelay")

_CATEGORY = "ComfyUI-CustomNodePacks/PromptRelay"


# ────────────────────────────────────────────────────────────────────────
# Shared encoder driver (native MODEL+CLIP path)
# ────────────────────────────────────────────────────────────────────────

def _encode_native(
    model,
    clip,
    latent,
    global_prompt: str,
    local_prompts: str,
    segment_lengths: str,
    epsilon: float,
    relay_options: Optional[dict] = None,
):
    for name, val in (
        ("global_prompt", global_prompt),
        ("local_prompts", local_prompts),
        ("segment_lengths", segment_lengths),
    ):
        if val is None:
            raise ValueError(
                f"PromptRelay: {name!r} arrived as None. Likely a stale workflow "
                "or a disconnected upstream input. Use an empty string instead."
            )

    locals_list = [p.strip() for p in local_prompts.split("|") if p.strip()]
    if not locals_list:
        raise ValueError("At least one local prompt is required (separate with |).")

    # Try native; if the model object doesn't match either native arch, fall
    # back to generic introspection.
    arch_info = None
    try:
        arch, patch_size, temporal_stride = detect_native_arch(model)
        arch_info = (arch, patch_size, temporal_stride)
    except ValueError:
        pass

    samples = latent["samples"]
    latent_frames = samples.shape[2]

    if arch_info is not None:
        _arch, patch_size, temporal_stride = arch_info
        tokens_per_frame = (samples.shape[3] // patch_size[1]) * (samples.shape[4] // patch_size[2])
    else:
        # Generic fallback: assume stride=4 (Wan-like) and 1×1 patches for tpf accounting;
        # the mask_fn ignores tpf when grid_sizes is provided by transformer_options.
        temporal_stride = 4
        tokens_per_frame = max(1, samples.shape[3] * samples.shape[4])

    parsed_lengths = None
    if segment_lengths.strip():
        pixel_lengths = [int(x.strip()) for x in segment_lengths.split(",") if x.strip()]
        parsed_lengths = convert_pixel_to_latent_lengths(pixel_lengths, temporal_stride, latent_frames)

    raw_tokenizer = get_raw_tokenizer_native(clip)
    full_prompt, token_ranges = map_token_indices(raw_tokenizer, global_prompt, locals_list)

    log.info("Global tokens [0:%d]", token_ranges[0][0])
    for i, (s, e) in enumerate(token_ranges):
        log.info("Segment %d: tokens [%d:%d] (%d tokens)", i, s, e, e - s)

    conditioning = clip.encode_from_tokens_scheduled(clip.tokenize(full_prompt))

    effective_lengths = distribute_segment_lengths(len(locals_list), latent_frames, parsed_lengths)
    log.info("Latent frames=%d tpf=%d segments=%s", latent_frames, tokens_per_frame, effective_lengths)

    q_token_idx = build_segments(token_ranges, effective_lengths, epsilon, relay_options)
    mask_fn = create_mask_fn(q_token_idx, tokens_per_frame, latent_frames)

    patched = model.clone()
    if arch_info is not None:
        patch_native(patched, arch_info[0], mask_fn)
    else:
        count = patch_generic(patched, mask_fn)
        if count == 0:
            raise ValueError(
                "PromptRelay generic-fallback: could not locate cross-attention "
                "modules on this model. Use the Kijai variant if this is a "
                "WANVIDEOMODEL, or open an issue with the model type."
            )

    return patched, conditioning


# ────────────────────────────────────────────────────────────────────────
# Node 1 — native encoder (manual segment lengths)
# ────────────────────────────────────────────────────────────────────────

class PromptRelayEncodeC2C:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "latent": ("LATENT", {"tooltip": "Empty latent video — dimensions are read from its shape."}),
                "global_prompt": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": "Conditions the entire video — persistent characters, style, lighting.",
                }),
                "local_prompts": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": "Per-segment prompts separated by '|'. One pipe per boundary.",
                }),
                "segment_lengths": ("STRING", {
                    "default": "",
                    "tooltip": "Comma-separated pixel-space frame counts. Empty = equal distribution.",
                }),
                "epsilon": ("FLOAT", {
                    "default": 1e-3, "min": 1e-6, "max": 0.99, "step": 1e-4,
                    "tooltip": "Penalty decay parameter. Below ~0.1 = sharp boundaries; ≥0.5 softer.",
                }),
            },
            "optional": {
                "relay_options": ("RELAY_OPTIONS", {
                    "tooltip": "Optional Prompt Relay Advanced Options bundle.",
                }),
            },
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING")
    RETURN_NAMES = ("model", "positive")
    FUNCTION = "execute"
    CATEGORY = _CATEGORY
    DESCRIPTION = (
        "Native ComfyUI MODEL/CLIP path. Encodes a global prompt with temporal "
        "local prompts and patches the model's cross-attention via "
        "ModelPatcher.add_object_patch (compatible with every native sampler)."
    )

    def execute(self, model, clip, latent, global_prompt, local_prompts,
                segment_lengths, epsilon, relay_options=None):
        patched, conditioning = _encode_native(
            model, clip, latent, global_prompt, local_prompts,
            segment_lengths, epsilon, relay_options,
        )
        return (patched, conditioning)


# ────────────────────────────────────────────────────────────────────────
# Node 2 — native smart-prompt encoder
# ────────────────────────────────────────────────────────────────────────

class PromptRelayEncodeSmartC2C:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "latent": ("LATENT",),
                "global_prompt": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": "Leave empty to auto-use the first parsed segment as the global anchor.",
                }),
                "smart_prompt": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": (
                        "Smart syntax:\n"
                        "  Inline: 'text one [0-50] | text two [50-150]'\n"
                        "  Block:  'Scene 1:\\ntext one\\nScene 2:\\ntext two'\n"
                        "Auto-detected; tags are stripped before encoding."
                    ),
                }),
                "normalize_by_tokens": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Scale each segment's weight by its CLIP token count.",
                }),
                "epsilon": ("FLOAT", {
                    "default": 1e-3, "min": 1e-6, "max": 0.99, "step": 1e-4,
                }),
            },
            "optional": {
                "relay_options": ("RELAY_OPTIONS",),
            },
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING")
    RETURN_NAMES = ("model", "positive")
    FUNCTION = "execute"
    CATEGORY = _CATEGORY
    DESCRIPTION = "Smart-syntax variant: '|' or 'Scene N:' headers auto-parsed into segments."

    def execute(self, model, clip, latent, global_prompt, smart_prompt,
                normalize_by_tokens, epsilon, relay_options=None):
        parsed = parse_smart_prompt(smart_prompt)
        valid = [s for s in parsed if s["text"].strip()]
        if not valid:
            valid = [{"text": " ", "weight": 1.0}]

        raw_tokenizer = get_raw_tokenizer_native(clip) if normalize_by_tokens else None

        locals_list = []
        weights = []
        for seg in valid:
            text = seg["text"]
            w = seg["weight"]
            if normalize_by_tokens and raw_tokenizer is not None:
                try:
                    ids = raw_tokenizer(text)["input_ids"]
                    n = ids.shape[-1] if hasattr(ids, "shape") and len(ids.shape) >= 2 else len(ids)
                    n -= 1 if getattr(raw_tokenizer, "add_eos", False) else 0
                    n = max(1, int(n))
                    w *= n
                except Exception as exc:
                    log.warning("Token counting failed for %r: %s", text, exc)
            locals_list.append(text)
            weights.append(w)

        local_prompts_str = " | ".join(locals_list)
        scale = 100000.0
        segment_lengths_str = ", ".join(str(int(round(w * scale))) for w in weights)

        global_prompt_str = global_prompt.strip()
        if not global_prompt_str:
            global_prompt_str = valid[0]["text"]

        patched, conditioning = _encode_native(
            model, clip, latent, global_prompt_str, local_prompts_str,
            segment_lengths_str, epsilon, relay_options,
        )
        return (patched, conditioning)


# ────────────────────────────────────────────────────────────────────────
# Node 3 — Kijai WanVideoWrapper path
# ────────────────────────────────────────────────────────────────────────

class PromptRelayEncodeKijaiC2C:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("WANVIDEOMODEL", {
                    "tooltip": "Kijai WanVideoWrapper model patcher (from WanVideoModelLoader).",
                }),
                "t5": ("WANTEXTENCODER", {
                    "tooltip": "Kijai T5 text encoder (from LoadWanVideoT5TextEncoder).",
                }),
                "latent_frames": ("INT", {
                    "default": 81, "min": 1, "max": 10000, "step": 1,
                    "tooltip": (
                        "Latent frame count Kijai's sampler will produce. For Wan: "
                        "(pixel_frames - 1) // 4 + 1. Must match what you'll pass to "
                        "WanVideoSampler."
                    ),
                }),
                "global_prompt": ("STRING", {"multiline": True, "default": ""}),
                "local_prompts": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": "Per-segment prompts separated by '|'.",
                }),
                "segment_lengths": ("STRING", {
                    "default": "",
                    "tooltip": "Comma-separated pixel-space frame counts. Empty = equal distribution.",
                }),
                "negative_prompt": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": "Sent through Kijai's T5 once; produces 'negative_prompt_embeds'.",
                }),
                "epsilon": ("FLOAT", {
                    "default": 1e-3, "min": 1e-6, "max": 0.99, "step": 1e-4,
                }),
                "encode_device": (["gpu", "cpu"], {"default": "gpu"}),
            },
            "optional": {
                "relay_options": ("RELAY_OPTIONS",),
            },
        }

    RETURN_TYPES = ("WANVIDEOMODEL", "WANVIDEOTEXTEMBEDS")
    RETURN_NAMES = ("model", "text_embeds")
    FUNCTION = "execute"
    CATEGORY = _CATEGORY
    DESCRIPTION = (
        "Kijai WanVideoWrapper path. Patches the live WanModel in place (Kijai's "
        "sampler bypasses ModelPatcher.object_patches), and produces "
        "WANVIDEOTEXTEMBEDS by encoding the concatenated prompt through Kijai's T5. "
        "Restore with the 'Prompt Relay Restore (Kijai)' node if you want to revert."
    )

    def execute(self, model, t5, latent_frames, global_prompt, local_prompts,
                segment_lengths, negative_prompt, epsilon, encode_device,
                relay_options=None):
        import torch
        try:
            import comfy.model_management as mm  # type: ignore
        except Exception:  # pragma: no cover
            mm = None

        locals_list = [p.strip() for p in local_prompts.split("|") if p.strip()]
        if not locals_list:
            raise ValueError("At least one local prompt is required (separate with |).")

        # ── token ranges via Kijai's T5 tokenizer ──
        encoder = t5["model"]
        tokenizer = getattr(encoder, "tokenizer", None)
        if tokenizer is None:
            raise RuntimeError(
                "Kijai T5 encoder is missing .tokenizer attribute. "
                "Update ComfyUI-WanVideoWrapper to a recent version."
            )

        # ── segment scheduling (Wan stride = 4) ──
        parsed_lengths = None
        if segment_lengths.strip():
            pixel_lengths = [int(x.strip()) for x in segment_lengths.split(",") if x.strip()]
            parsed_lengths = convert_pixel_to_latent_lengths(pixel_lengths, 4, latent_frames)

        # Adapt Kijai tokenizer to the (callable returning dict) interface map_token_indices expects.
        class _TokAdapter:
            add_eos = True  # Kijai's HuggingfaceTokenizer adds </s> by default.
            def __call__(self_inner, text):
                ids, _mask = tokenizer([text], return_mask=True, add_special_tokens=True)
                return {"input_ids": ids}

        full_prompt, token_ranges = map_token_indices(_TokAdapter(), global_prompt, locals_list)
        log.info("Kijai path: full_prompt token-ranges = %s", token_ranges)

        effective_lengths = distribute_segment_lengths(len(locals_list), latent_frames, parsed_lengths)
        log.info("Kijai path: latent_frames=%d effective_lengths=%s", latent_frames, effective_lengths)

        # tokens_per_frame for Wan with patch_size=(1,2,2): need diffusion model's patch_size.
        transformer = getattr(getattr(model, "model", model), "diffusion_model", None)
        if transformer is None:
            raise RuntimeError("Kijai model patcher is missing .model.diffusion_model.")
        patch_size = tuple(getattr(transformer, "patch_size", (1, 2, 2)))
        # Without grid_sizes from transformer_options the mask_fn uses fallback tpf;
        # Kijai's WanModel passes grid_sizes via transformer_options, so this is mostly a hint.
        fallback_tpf = max(1, 64 * 64 // (patch_size[1] * patch_size[2]))

        q_token_idx = build_segments(token_ranges, effective_lengths, epsilon, relay_options)
        mask_fn = create_mask_fn(q_token_idx, fallback_tpf, latent_frames)

        n_patched = patch_kijai(model, mask_fn)
        log.info("Kijai path: patched %d cross_attn blocks", n_patched)

        # ── encode positive (concatenated) + negative via Kijai T5 ──
        device_to = (mm.get_torch_device() if mm is not None else torch.device("cuda")) \
            if encode_device == "gpu" else torch.device("cpu")
        with torch.no_grad():
            try:
                context = encoder([full_prompt], device_to)
                context_null = encoder([negative_prompt or ""], device_to)
            finally:
                if mm is not None:
                    try:
                        mm.soft_empty_cache()
                    except Exception:
                        pass

        text_embeds = {
            "prompt_embeds": context,
            "negative_prompt_embeds": context_null,
            "echoshot": False,
        }
        return (model, text_embeds)


class PromptRelayRestoreKijaiC2C:
    """Restore the original cross_attn.forward methods on a Kijai WanVideoModel."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("WANVIDEOMODEL",),
            }
        }

    RETURN_TYPES = ("WANVIDEOMODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "execute"
    CATEGORY = _CATEGORY
    DESCRIPTION = "Undo a Prompt Relay (Kijai) patch — restores the original cross_attn.forward."

    def execute(self, model):
        from ._patches import restore_kijai
        restore_kijai(model)
        return (model,)


# ────────────────────────────────────────────────────────────────────────
# Node 4 — RELAY_OPTIONS bundle
# ────────────────────────────────────────────────────────────────────────

class PromptRelayAdvancedOptionsC2C:
    @classmethod
    def INPUT_TYPES(cls):
        tt_strength = (
            "Multiplier on the temporal penalty. 0 disables segmentation. Most "
            "useful in 0–1 to soften boundaries; >1 saturates quickly at the "
            "default epsilon — raise epsilon to ~0.1 to make >1 meaningful."
        )
        tt_window = (
            "Scales the flat anchor zone (default L/2 - 2 frames). <1 starts "
            "falloff sooner; >1 widens the rigid zone."
        )
        return {
            "required": {
                "video_strength":    ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.05, "tooltip": tt_strength}),
                "video_window_scale":("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0,  "step": 0.05, "tooltip": tt_window}),
                "audio_epsilon":     ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.99, "step": 1e-4,
                                                 "tooltip": "LTX audio stream epsilon. 0 = inherit from the encoder."}),
                "audio_strength":    ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.05, "tooltip": tt_strength}),
                "audio_window_scale":("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0,  "step": 0.05, "tooltip": tt_window}),
            }
        }

    RETURN_TYPES = ("RELAY_OPTIONS",)
    RETURN_NAMES = ("relay_options",)
    FUNCTION = "execute"
    CATEGORY = _CATEGORY
    DESCRIPTION = "Optional per-stream tuning for the Prompt Relay encoders."

    def execute(self, video_strength, video_window_scale,
                audio_epsilon, audio_strength, audio_window_scale):
        opts = {
            "video_strength": video_strength,
            "video_window_scale": video_window_scale,
            "audio_epsilon": audio_epsilon if audio_epsilon > 0 else None,
            "audio_strength": audio_strength,
            "audio_window_scale": audio_window_scale,
        }
        return (opts,)
