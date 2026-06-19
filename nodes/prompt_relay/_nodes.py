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
# Unified Prompt Relay Encoder (Batch 2 — 2026-05-18)
#
# Replaces:
#   - PromptRelayEncodeC2C       (native)        -> backend="native"
#   - PromptRelayEncodeSmartC2C  (smart syntax)  -> backend="smart"
#   - PromptRelayEncodeKijaiC2C  (Kijai Wan path) -> backend="kijai"
#
# One node, dynamic sockets. The JS extension (web/extensions/c2c/
# prompt_relay_dyn.js) hides the inputs/outputs that don't apply to the
# currently-selected backend. All sockets are declared as `optional` here so
# the prompt validator accepts the node regardless of which inputs are wired.
# ────────────────────────────────────────────────────────────────────────

def _encode_kijai(model, t5, latent_frames, global_prompt, local_prompts,
                  segment_lengths, negative_prompt, epsilon, encode_device,
                  relay_options=None):
    """Kijai WanVideoWrapper encode + cross-attn patch (extracted body)."""
    import torch
    try:
        import comfy.model_management as mm  # type: ignore
    except Exception:
        mm = None

    locals_list = [p.strip() for p in (local_prompts or "").split("|") if p.strip()]
    if not locals_list:
        raise ValueError("At least one local prompt is required (separate with |).")

    encoder = t5["model"]
    tokenizer = getattr(encoder, "tokenizer", None)
    if tokenizer is None:
        raise RuntimeError(
            "Kijai T5 encoder is missing .tokenizer attribute. "
            "Update ComfyUI-WanVideoWrapper to a recent version."
        )

    parsed_lengths = None
    if (segment_lengths or "").strip():
        pixel_lengths = [int(x.strip()) for x in segment_lengths.split(",") if x.strip()]
        parsed_lengths = convert_pixel_to_latent_lengths(pixel_lengths, 4, latent_frames)

    class _TokAdapter:
        add_eos = True
        def __call__(self_inner, text):
            ids, _mask = tokenizer([text], return_mask=True, add_special_tokens=True)
            return {"input_ids": ids}

    full_prompt, token_ranges = map_token_indices(_TokAdapter(), global_prompt, locals_list)
    log.info("Kijai path: token-ranges = %s", token_ranges)

    effective_lengths = distribute_segment_lengths(len(locals_list), latent_frames, parsed_lengths)

    transformer = getattr(getattr(model, "model", model), "diffusion_model", None)
    if transformer is None:
        raise RuntimeError("Kijai model patcher is missing .model.diffusion_model.")
    patch_size = tuple(getattr(transformer, "patch_size", (1, 2, 2)))
    fallback_tpf = max(1, 64 * 64 // (patch_size[1] * patch_size[2]))

    q_token_idx = build_segments(token_ranges, effective_lengths, epsilon, relay_options)
    mask_fn = create_mask_fn(q_token_idx, fallback_tpf, latent_frames)

    n_patched = patch_kijai(model, mask_fn)
    log.info("Kijai path: patched %d cross_attn blocks", n_patched)

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
    return model, text_embeds


def _encode_smart(model, clip, latent, global_prompt, smart_prompt,
                  normalize_by_tokens, epsilon, relay_options=None):
    """Smart-syntax parser + native encode (extracted body)."""
    parsed = parse_smart_prompt(smart_prompt or "")
    valid = [s for s in parsed if s["text"].strip()]
    if not valid:
        valid = [{"text": " ", "weight": 1.0}]

    raw_tokenizer = get_raw_tokenizer_native(clip) if normalize_by_tokens else None

    locals_list, weights = [], []
    for seg in valid:
        text, w = seg["text"], seg["weight"]
        if normalize_by_tokens and raw_tokenizer is not None:
            try:
                ids = raw_tokenizer(text)["input_ids"]
                n = ids.shape[-1] if hasattr(ids, "shape") and len(ids.shape) >= 2 else len(ids)
                n -= 1 if getattr(raw_tokenizer, "add_eos", False) else 0
                w *= max(1, int(n))
            except Exception as exc:
                log.warning("Token counting failed for %r: %s", text, exc)
        locals_list.append(text)
        weights.append(w)

    local_prompts_str = " | ".join(locals_list)
    scale = 100000.0
    segment_lengths_str = ", ".join(str(int(round(w * scale))) for w in weights)
    global_prompt_str = (global_prompt or "").strip() or valid[0]["text"]

    return _encode_native(
        model, clip, latent, global_prompt_str, local_prompts_str,
        segment_lengths_str, epsilon, relay_options,
    )


class PromptRelayEncodeC2C:
    """Unified Prompt Relay encoder. Pick a backend via dropdown.

    Backend modes:
      - native  : ComfyUI MODEL + CLIP (manual `segment_lengths`).
      - smart   : ComfyUI MODEL + CLIP, segments parsed from `smart_prompt`.
      - kijai   : Kijai WANVIDEOMODEL + WANTEXTENCODER for the WanVideoWrapper.

    Outputs are declared once but only two are populated per call. The JS
    extension hides the slots that don't apply to the current backend so the
    UI stays clean. Algorithm credit: Gordon Chen — see NOTICE.md.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "backend": (["native", "smart", "kijai"], {
                    "default": "native",
                    "tooltip": "native = MODEL+CLIP; smart = MODEL+CLIP with auto-segmented prompt; kijai = WanVideoWrapper.",
                }),
                "global_prompt": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": "Persistent prompt across the whole video.",
                }),
                "epsilon": ("FLOAT", {
                    "default": 1e-3, "min": 1e-6, "max": 0.99, "step": 1e-4,
                    "tooltip": "Temporal penalty decay (sharpness of segment boundaries).",
                }),
            },
            "optional": {
                # MODEL+CLIP path (native / smart)
                "model":  ("MODEL",  {"tooltip": "native/smart only."}),
                "clip":   ("CLIP",   {"tooltip": "native/smart only."}),
                "latent": ("LATENT", {"tooltip": "native/smart only — frame count read from shape."}),
                # Kijai path
                "wan_model": ("WANVIDEOMODEL", {"tooltip": "kijai only — from WanVideoModelLoader."}),
                "wan_t5":    ("WANTEXTENCODER", {"tooltip": "kijai only — from LoadWanVideoT5TextEncoder."}),
                # Shared widgets
                "local_prompts":   ("STRING", {"multiline": True, "default": "",
                                                "tooltip": "Per-segment prompts separated by '|' (native/kijai)."}),
                "segment_lengths": ("STRING", {"default": "",
                                                "tooltip": "Comma-separated pixel-space frame counts. Empty = equal (native/kijai)."}),
                # Smart-only
                "smart_prompt": ("STRING", {"multiline": True, "default": "",
                                            "tooltip": "smart only — auto-parsed (`|` or `Scene N:`)."}),
                "normalize_by_tokens": ("BOOLEAN", {"default": False,
                                                    "tooltip": "smart only — scale segment weight by token count."}),
                # Kijai-only
                "latent_frames": ("INT", {"default": 81, "min": 1, "max": 10000, "step": 1,
                                          "tooltip": "kijai only — (pixel_frames-1)//4 + 1."}),
                "negative_prompt": ("STRING", {"multiline": True, "default": "",
                                                "tooltip": "kijai only — encoded once for negative_prompt_embeds."}),
                "encode_device": (["gpu", "cpu"], {"default": "gpu",
                                                    "tooltip": "kijai only — device for T5 encode."}),
                # All backends
                "relay_options": ("RELAY_OPTIONS", {
                    "tooltip": "Optional Prompt Relay Advanced Options bundle.",
                }),
            },
        }

    # Four outputs: native/smart populate (model, positive); kijai populates
    # (wan_model, wan_text_embeds). The JS extension hides the unused pair.
    RETURN_TYPES = ("MODEL", "CONDITIONING", "WANVIDEOMODEL", "WANVIDEOTEXTEMBEDS")
    RETURN_NAMES = ("model", "positive", "wan_model", "wan_text_embeds")
    FUNCTION = "execute"
    CATEGORY = _CATEGORY
    DESCRIPTION = (
        "Unified Prompt Relay encoder (native / smart / kijai). Dynamic "
        "sockets via the JS extension."
    )

    def execute(self, backend, global_prompt, epsilon,
                model=None, clip=None, latent=None,
                wan_model=None, wan_t5=None,
                local_prompts="", segment_lengths="",
                smart_prompt="", normalize_by_tokens=False,
                latent_frames=81, negative_prompt="", encode_device="gpu",
                relay_options=None):

        if backend == "kijai":
            if wan_model is None or wan_t5 is None:
                raise ValueError(
                    "Prompt Relay Encode (kijai): connect `wan_model` (WANVIDEOMODEL) "
                    "and `wan_t5` (WANTEXTENCODER) inputs."
                )
            patched_model, text_embeds = _encode_kijai(
                wan_model, wan_t5, latent_frames, global_prompt, local_prompts,
                segment_lengths, negative_prompt, epsilon, encode_device, relay_options,
            )
            return (None, None, patched_model, text_embeds)

        if model is None or clip is None or latent is None:
            raise ValueError(
                f"Prompt Relay Encode ({backend}): connect `model` (MODEL), "
                "`clip` (CLIP), and `latent` (LATENT) inputs."
            )

        if backend == "smart":
            patched, conditioning = _encode_smart(
                model, clip, latent, global_prompt, smart_prompt,
                normalize_by_tokens, epsilon, relay_options,
            )
        else:  # native
            patched, conditioning = _encode_native(
                model, clip, latent, global_prompt, local_prompts,
                segment_lengths, epsilon, relay_options,
            )
        return (patched, conditioning, None, None)


# ────────────────────────────────────────────────────────────────────────
# Deprecated alias shims — keep for ONE release, then drop.
# ────────────────────────────────────────────────────────────────────────
class _DeprecatedPRBase:
    _WARNED: set = set()

    @classmethod
    def _warn_once(cls, replacement_backend: str):
        if cls.__name__ in _DeprecatedPRBase._WARNED:
            return
        _DeprecatedPRBase._WARNED.add(cls.__name__)
        log.warning(
            "[prompt_relay] %s is deprecated and will be removed next release. "
            "Use PromptRelayEncodeC2C with backend='%s'.",
            cls.__name__, replacement_backend,
        )


class PromptRelayEncodeSmartC2C(_DeprecatedPRBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "latent": ("LATENT",),
                "global_prompt": ("STRING", {"multiline": True, "default": ""}),
                "smart_prompt": ("STRING", {"multiline": True, "default": ""}),
                "normalize_by_tokens": ("BOOLEAN", {"default": False}),
                "epsilon": ("FLOAT", {"default": 1e-3, "min": 1e-6, "max": 0.99, "step": 1e-4}),
            },
            "optional": {"relay_options": ("RELAY_OPTIONS",)},
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING")
    RETURN_NAMES = ("model", "positive")
    FUNCTION = "execute"
    CATEGORY = _CATEGORY
    DESCRIPTION = "DEPRECATED — use PromptRelayEncodeC2C with backend='smart'."

    def execute(self, model, clip, latent, global_prompt, smart_prompt,
                normalize_by_tokens, epsilon, relay_options=None):
        self._warn_once("smart")
        patched, conditioning = _encode_smart(
            model, clip, latent, global_prompt, smart_prompt,
            normalize_by_tokens, epsilon, relay_options,
        )
        return (patched, conditioning)


class PromptRelayEncodeKijaiC2C(_DeprecatedPRBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("WANVIDEOMODEL",),
                "t5": ("WANTEXTENCODER",),
                "latent_frames": ("INT", {"default": 81, "min": 1, "max": 10000, "step": 1}),
                "global_prompt": ("STRING", {"multiline": True, "default": ""}),
                "local_prompts": ("STRING", {"multiline": True, "default": ""}),
                "segment_lengths": ("STRING", {"default": ""}),
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "epsilon": ("FLOAT", {"default": 1e-3, "min": 1e-6, "max": 0.99, "step": 1e-4}),
                "encode_device": (["gpu", "cpu"], {"default": "gpu"}),
            },
            "optional": {"relay_options": ("RELAY_OPTIONS",)},
        }

    RETURN_TYPES = ("WANVIDEOMODEL", "WANVIDEOTEXTEMBEDS")
    RETURN_NAMES = ("model", "text_embeds")
    FUNCTION = "execute"
    CATEGORY = _CATEGORY
    DESCRIPTION = "DEPRECATED — use PromptRelayEncodeC2C with backend='kijai'."

    def execute(self, model, t5, latent_frames, global_prompt, local_prompts,
                segment_lengths, negative_prompt, epsilon, encode_device,
                relay_options=None):
        self._warn_once("kijai")
        patched_model, text_embeds = _encode_kijai(
            model, t5, latent_frames, global_prompt, local_prompts,
            segment_lengths, negative_prompt, epsilon, encode_device, relay_options,
        )
        return (patched_model, text_embeds)


# ────────────────────────────────────────────────────────────────────────
# Restore (Kijai) — unchanged
# ────────────────────────────────────────────────────────────────────────
class PromptRelayRestoreKijaiC2C:
    """Restore the original cross_attn.forward methods on a Kijai WanVideoModel."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("WANVIDEOMODEL",)}}

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
# Advanced Options bundle — unchanged
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

