# Prompt Relay — node classes (refined port).
#
# Nodes registered:
#   - PromptRelayEncodeC2C        : unified encoder (native / smart / kijai backends)
#   - PromptRelayRestoreKijaiC2C  : undo Kijai patch
#   - PromptRelayAdvancedOptionsC2C : RELAY_OPTIONS bundle (optional)
#
# Display names follow ComfyUI-CustomNodePacks convention (no _C2C suffix in UI).
# Algorithm credit: Gordon Chen & contributors — see NOTICE.md.

from __future__ import annotations

import contextvars
import json
import logging
import os
import tempfile
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
    cross_attention_count,
    detect_native_arch,
    patch_generic,
    patch_kijai,
    patch_native,
    restore_kijai_checked,
)
try:                                                       # A28 layout declarations (display only)
    from .._layout_decl import with_layout
except ImportError:  # loaded as a plain file (tests): the same module by path, never an absolute import
    try:
        import importlib.util as _c2c_ilu
        import os as _c2c_os
        _c2c_spec = _c2c_ilu.spec_from_file_location("c2c_layout_by_path", _c2c_os.path.join(
            _c2c_os.path.dirname(_c2c_os.path.abspath(__file__)), '..', '_layout_decl.py'))
        _c2c_mod = _c2c_ilu.module_from_spec(_c2c_spec)
        _c2c_spec.loader.exec_module(_c2c_mod)
        with_layout = _c2c_mod.with_layout
    except Exception:  # pragma: no cover - layout keys are display only: never take the node down
        def with_layout(spec, layout):
            return spec

log = logging.getLogger("C2C.PromptRelay")

# L9.174: the segment inspector's collector. A context variable, so the encode helpers keep their exact call
# signature (tests/test_legacy_replacements.py pins the arguments the node passes them).
_COLLECT: "contextvars.ContextVar[Optional[dict]]" = contextvars.ContextVar("c2c_prompt_relay_collect", default=None)

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
    collect: Optional[dict] = None,
):
    if collect is None:
        collect = _COLLECT.get()
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
    if arch_info is None and cross_attention_count(model) == 0:
        # Refuse before touching the text encoder (audit 2026-11-27: MiniMax H3 crashed deep in the patcher).
        dm = getattr(getattr(model, "model", None), "diffusion_model", None)
        raise ValueError(
            f"Prompt Relay does not support this model ({type(dm).__name__ if dm is not None else 'unknown'}): it "
            "works by masking the text cross-attention of Wan or LTX video models, and this model has no "
            "cross-attention to mask (MiniMax H3, for example, uses joint self-attention over text and video). "
            "Remove the Prompt Relay node for this model, or use a Wan / LTX model.")

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
    if collect is not None:
        collect.update(arch=arch_info[0] if arch_info is not None else "generic", full_prompt=full_prompt,
                       token_ranges=token_ranges, segment_lengths=effective_lengths, latent_frames=latent_frames,
                       tokens_per_frame=tokens_per_frame, temporal_stride=temporal_stride, locals=locals_list)

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
#   - PromptRelayEncodeSmartC2C  (removed L7.65) -> backend="smart" (auto-migrate on load)
#   - PromptRelayEncodeKijaiC2C  (removed L7.65) -> backend="kijai" (auto-migrate on load)
#
# One node, dynamic sockets. The JS extension (web/extensions/c2c/
# prompt_relay_dyn.js) hides the inputs/outputs that don't apply to the
# currently-selected backend. All sockets are declared as `optional` here so
# the prompt validator accepts the node regardless of which inputs are wired.
# ────────────────────────────────────────────────────────────────────────

def _encode_kijai(model, t5, latent_frames, global_prompt, local_prompts,
                  segment_lengths, negative_prompt, epsilon, encode_device,
                  relay_options=None, collect=None):
    """Kijai WanVideoWrapper encode + cross-attn patch (extracted body)."""
    if collect is None:
        collect = _COLLECT.get()
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
    if collect is not None:
        collect.update(arch="kijai", full_prompt=full_prompt, token_ranges=token_ranges,
                       segment_lengths=effective_lengths, latent_frames=latent_frames, tokens_per_frame=fallback_tpf,
                       temporal_stride=4, locals=locals_list)

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
                  normalize_by_tokens, epsilon, relay_options=None, collect=None):
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
    if collect is None:
        collect = _COLLECT.get()
    if collect is not None:
        collect["weights"] = weights

    return _encode_native(
        model, clip, latent, global_prompt_str, local_prompts_str,
        segment_lengths_str, epsilon, relay_options, collect,
    )


# ── L9.174 (A25): multi-role prompt organisation + segment inspector ───────────────────────────────────────────────
ROLE_KEYS = ("subject", "action", "camera", "environment", "style")


def parse_roles(text: str) -> dict:
    """'subject: a fox\ncamera: slow dolly' -> {role: text}. Unknown keys / lines without 'key:' are errors."""
    roles = {}
    for n, raw in enumerate((text or "").splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        key, sep, val = line.partition(":")
        key = key.strip().lower()
        if not sep or key not in ROLE_KEYS:
            raise ValueError(f"Prompt Relay Encode: role_sections line {n} ({line[:40]!r}) must start with one of "
                             f"{', '.join(k + ':' for k in ROLE_KEYS)}")
        val = val.strip()
        if val:
            roles[key] = (roles[key] + " " + val) if key in roles else val
    return roles


def compose_global(global_prompt: str, roles: dict) -> str:
    """global_prompt followed by the role sections in canonical order, as sentences."""
    parts = [p.strip().rstrip(".") for p in [global_prompt or ""] + [roles[k] for k in ROLE_KEYS if k in roles]]
    parts = [p for p in parts if p]
    return ". ".join(parts) + ("." if roles and parts else "")


def segment_report(backend: str, info: dict, epsilon: float, roles: dict, global_prompt: str) -> str:
    """How the text was divided and where each segment sits in tokens and latent frames, as JSON."""
    ranges = info.get("token_ranges") or []
    lengths = info.get("segment_lengths") or []
    stride = int(info.get("temporal_stride") or 4)
    weights = info.get("weights")
    segs, start = [], 0
    for i, text in enumerate(info.get("locals") or []):
        tr = list(ranges[i]) if i < len(ranges) else None
        n = int(lengths[i]) if i < len(lengths) else 0
        segs.append({"index": i, "text": text, "token_range": tr,
                     "token_count": (tr[1] - tr[0]) if tr else None,
                     "latent_frame_range": [start, start + n], "approx_pixel_frames": n * stride,
                     **({"weight": weights[i]} if weights and i < len(weights) else {})})
        start += n
    rep = {"backend": backend, "arch": info.get("arch"), "global_prompt": global_prompt, "roles": roles,
           "full_prompt": info.get("full_prompt"),
           "global_token_range": [0, ranges[0][0]] if ranges else None, "segments": segs,
           "latent_frames": info.get("latent_frames"), "tokens_per_frame": info.get("tokens_per_frame"),
           "temporal_stride": stride, "epsilon": epsilon}
    return json.dumps(rep, indent=2, default=str)


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
                # L9.174 (A25), appended: empty = the behaviour of every saved workflow
                "role_sections": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": "Structured global prompt, one section per line: 'subject: ...', 'action: ...', "
                               "'camera: ...', 'environment: ...', 'style: ...'. The sections are added to "
                               "global_prompt in that order (all backends). Empty = global_prompt only.",
                }),
            },
        }

    # Four outputs: native/smart populate (model, positive); kijai populates
    # (wan_model, wan_text_embeds). The JS extension hides the unused pair.
    RETURN_TYPES = ("MODEL", "CONDITIONING", "WANVIDEOMODEL", "WANVIDEOTEXTEMBEDS", "STRING")
    RETURN_NAMES = ("model", "positive", "wan_model", "wan_text_embeds", "segment_report")
    OUTPUT_TOOLTIPS = ("native / smart: the MODEL with the segment masks patched in - wire it to the sampler.",
                       "native / smart: the prompt conditioning for the sampler's positive input.",
                       "kijai: the WanVideo model with the segment masks patched in (they stay until Prompt Relay Restore or a reload).",
                       "kijai: text embeds for the WanVideo sampler.",
                       "Every backend: how the text was divided - the full prompt, the global token range and per "
                       "segment its text, token range, latent / pixel frame range (and weight for smart), as JSON.")
    FUNCTION = "execute"
    CATEGORY = _CATEGORY
    DESCRIPTION = (
        "Unified Prompt Relay encoder (native / smart / kijai). Dynamic "
        "sockets via the JS extension. kijai backend: the segment masks are "
        "patched INTO the loaded Kijai model (his sampler gives no per-run hook), "
        "so they stay after you delete this node - run 'Prompt Relay Restore "
        "(Kijai)' on the model, or reload it, to remove them."
    )

    def execute(self, backend, global_prompt, epsilon,
                model=None, clip=None, latent=None,
                wan_model=None, wan_t5=None,
                local_prompts="", segment_lengths="",
                smart_prompt="", normalize_by_tokens=False,
                latent_frames=81, negative_prompt="", encode_device="gpu",
                relay_options=None, role_sections=""):

        roles = parse_roles(role_sections)
        if roles:
            global_prompt = compose_global(global_prompt, roles)
        info = {}
        token = _COLLECT.set(info)
        try:
            return self._execute(backend, global_prompt, epsilon, model, clip, latent, wan_model, wan_t5,
                                 local_prompts, segment_lengths, smart_prompt, normalize_by_tokens, latent_frames,
                                 negative_prompt, encode_device, relay_options, roles, info)
        finally:
            _COLLECT.reset(token)

    def _execute(self, backend, global_prompt, epsilon, model, clip, latent, wan_model, wan_t5, local_prompts,
                 segment_lengths, smart_prompt, normalize_by_tokens, latent_frames, negative_prompt, encode_device,
                 relay_options, roles, info):
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
            return (None, None, patched_model, text_embeds,
                    segment_report(backend, info, epsilon, roles, global_prompt))

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
        return (patched, conditioning, None, None, segment_report(backend, info, epsilon, roles, global_prompt))


# ────────────────────────────────────────────────────────────────────────
# Restore (Kijai) — unchanged
# ────────────────────────────────────────────────────────────────────────
FOREIGN_POLICIES = ("restore anyway", "keep foreign patch", "error")


class PromptRelayRestoreKijaiC2C:
    """Restore the original cross_attn.forward methods on a Kijai WanVideoModel."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("WANVIDEOMODEL", {"tooltip": "A WanVideo model a Prompt Relay (kijai) encode patched."})},
                "optional": {
                    # L9.174 (A25), appended: "restore anyway" = the behaviour of every saved workflow
                    "on_foreign_patch": (list(FOREIGN_POLICIES), {
                        "default": FOREIGN_POLICIES[0],
                        "tooltip": "When another node patched a block's cross-attention AFTER Prompt Relay: restore "
                                   "anyway = put back the original saved by Prompt Relay (drops the other patch, as "
                                   "before). keep foreign patch = leave that block's current patch and only remove "
                                   "Prompt Relay's bookkeeping. error = stop before changing anything."})}}

    RETURN_TYPES = ("WANVIDEOMODEL", "STRING")
    RETURN_NAMES = ("model", "patch_state")
    OUTPUT_TOOLTIPS = ("The same model with its original cross-attention restored.",
                       "Patch state before and after, as JSON: blocks found, patched, foreign patches, restored, "
                       "anything left behind and a verdict (clean / was not patched / restore incomplete / foreign "
                       "patch kept).")
    FUNCTION = "execute"
    CATEGORY = _CATEGORY
    DESCRIPTION = "Undo a Prompt Relay (Kijai) patch — restores the original cross_attn.forward."

    def execute(self, model, on_foreign_patch=FOREIGN_POLICIES[0]):
        state = restore_kijai_checked(model, on_foreign_patch)
        return (model, json.dumps(state, indent=2))


# ────────────────────────────────────────────────────────────────────────
# Advanced Options bundle — unchanged
# ────────────────────────────────────────────────────────────────────────
# A28 (docs/audit/A28/PARAM_CNP_B.md, PromptRelayAdvancedOptionsC2C): the LTX audio-stream controls sit in a
# section. Display only.
_RELAY_OPTIONS_LAYOUT = {k: {"c2c_section": "Audio stream"}
                         for k in ("audio_epsilon", "audio_strength", "audio_window_scale")}
_RELAY_OPTIONS_LAYOUT.update({k: {"c2c_section": "Profiles"} for k in ("profile_action", "profile_name",
                                                                        "profile_target")})

# ── L9.174 (A25): option profiles + capability report ─────────────────────────────────────────────────────────────
_OPTION_FIELDS = ("video_strength", "video_window_scale", "audio_epsilon", "audio_strength", "audio_window_scale")
_OPTION_RANGES = {"video_strength": (0.0, 10.0), "video_window_scale": (0.0, 4.0), "audio_epsilon": (0.0, 0.99),
                  "audio_strength": (0.0, 10.0), "audio_window_scale": (0.0, 4.0)}
PROFILE_TARGETS = ("any", "wan (native)", "ltx (native)", "kijai wan", "generic")


def _profiles_path() -> str:
    try:
        import folder_paths
        base = folder_paths.get_user_directory()
    except Exception:  # noqa: BLE001 - outside ComfyUI
        base = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "c2c")
        return os.path.join(base, "prompt_relay_profiles.json")
    return os.path.join(base, "c2c", "prompt_relay_profiles.json")


def _validate_options(values: dict, where: str) -> dict:
    out = {}
    for f in _OPTION_FIELDS:
        if f not in values:
            raise ValueError(f"Prompt Relay options ({where}): {f} is missing")
        v = values[f]
        lo, hi = _OPTION_RANGES[f]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not lo <= float(v) <= hi:
            raise ValueError(f"Prompt Relay options ({where}): {f}={v!r} must be a number from {lo} to {hi}")
        out[f] = float(v)
    return out


def _read_profiles() -> dict:
    p = _profiles_path()
    if not os.path.isfile(p):
        return {}
    try:
        with open(p, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Prompt Relay options: the profile file {p} is unreadable ({exc}); fix or delete it")
    if not isinstance(data, dict):
        raise ValueError(f"Prompt Relay options: the profile file {p} is not a JSON object; fix or delete it")
    return data


def save_profile(name: str, target: str, values: dict) -> dict:
    vals = _validate_options(values, f"profile {name!r}")
    data = _read_profiles()
    data[name] = {"target": target, "values": vals}
    p = _profiles_path()
    os.makedirs(os.path.dirname(p), exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".prompt_relay_profiles.", dir=os.path.dirname(p))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
        os.replace(tmp, p)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return data[name]


def load_profile(name: str) -> dict:
    data = _read_profiles()
    if name not in data:
        raise ValueError(f"Prompt Relay options: no profile named {name!r}. Saved profiles: "
                         f"{', '.join(sorted(data)) or 'none'}")
    entry = data[name]
    if not isinstance(entry, dict) or not isinstance(entry.get("values"), dict):
        raise ValueError(f"Prompt Relay options: profile {name!r} is malformed in {_profiles_path()}")
    return {"target": entry.get("target", "any"), "values": _validate_options(entry["values"], f"profile {name!r}")}


# Which controls each path reads (from the code, L9.174):
#  - video_strength / video_window_scale: _core.build_segments -> segment "strength" / "window" -> the video temporal
#    cost (_core._build_temporal_cost) used by create_mask_fn for every backend (native Wan / LTX, kijai, generic).
#  - audio_epsilon / audio_strength / audio_window_scale: segment "sigma_audio" / "strength_audio" /
#    "window_audio" -> only _core._build_temporal_cost_scaled, the LTX audio cross-attention path (_patches
#    _ltx_forward). Wan (native / kijai) and the generic patcher have no audio stream: create_mask_fn uses the scaled
#    cost only when a query's length is not latent_frames x tokens_per_frame, so they ignore them on the video path.
_READS = {
    "wan (native)": {"video_strength", "video_window_scale"},
    "ltx (native)": set(_OPTION_FIELDS),
    "kijai wan": {"video_strength", "video_window_scale"},
    "generic": {"video_strength", "video_window_scale"},
}
_WHERE = {"video_strength": "_core.build_segments -> video temporal cost (create_mask_fn)",
          "video_window_scale": "_core.build_segments -> video anchor window",
          "audio_epsilon": "_core.build_segments sigma_audio -> LTX audio cross-attention only",
          "audio_strength": "_core._build_temporal_cost_scaled -> LTX audio cross-attention only",
          "audio_window_scale": "_core.build_segments window_audio -> LTX audio cross-attention only"}


def capability_report(values: dict, target: str = "any") -> str:
    targets = [t for t in PROFILE_TARGETS if t != "any"] if target == "any" else [target]
    if target not in PROFILE_TARGETS:
        raise ValueError(f"Prompt Relay options: profile_target must be one of {', '.join(PROFILE_TARGETS)}")
    lines = []
    for t in targets:
        used = [f for f in _OPTION_FIELDS if f in _READS[t]]
        ignored = [f for f in _OPTION_FIELDS if f not in _READS[t]]
        lines.append(f"[{t}] reads: {', '.join(used)}")
        if ignored:
            lines.append(f"[{t}] ignores: {', '.join(ignored)} (no audio stream: they are read only by the scaled "
                         f"cost, which runs for a query whose length is not the video token grid)")
    lines.append("where: " + "; ".join(f"{f} = {_WHERE[f]}" for f in _OPTION_FIELDS))
    warn = []
    if float(values["video_strength"]) == 0.0:
        warn.append("video_strength 0 turns segmentation OFF (every segment sees the whole video)")
    for f, neutral in (("video_strength", 1.0), ("video_window_scale", 1.0), ("audio_strength", 1.0),
                       ("audio_window_scale", 1.0)):
        if float(values[f]) == neutral:
            warn.append(f"{f} {neutral} = neutral (no change from the encoder's default)")
    if not values.get("audio_epsilon"):
        warn.append("audio_epsilon 0 = inherit the encoder's epsilon")
    lines += [f"note: {w}" for w in warn]
    return "\n".join(lines)


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
        return with_layout({
            "required": {
                "video_strength":    ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.05, "tooltip": tt_strength}),
                "video_window_scale":("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0,  "step": 0.05, "tooltip": tt_window}),
                "audio_epsilon":     ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.99, "step": 1e-4,
                                                 "tooltip": "LTX audio stream epsilon. 0 = inherit from the encoder."}),
                "audio_strength":    ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.05, "tooltip": tt_strength}),
                "audio_window_scale":("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0,  "step": 0.05, "tooltip": tt_window}),
            },
            "optional": {
                # L9.174 (A25), appended: "none" = the behaviour of every saved workflow
                "profile_action": (["none", "save", "load"], {"default": "none", "tooltip": "save: store the five "
                                   "values above under profile_name (range-checked) in user/c2c/"
                                   "prompt_relay_profiles.json. load: use the saved profile's values instead of the "
                                   "widgets. none: the widgets as they are."}),
                "profile_name": ("STRING", {"default": "", "tooltip": "Name of the profile to save or load."}),
                "profile_target": (list(PROFILE_TARGETS), {"default": "any", "tooltip": "The encoder path these "
                                   "options are for: stored with a saved profile and used by capability_report to "
                                   "list which controls that path reads and which it ignores."}),
            },
        }, _RELAY_OPTIONS_LAYOUT)

    RETURN_TYPES = ("RELAY_OPTIONS", "STRING")
    RETURN_NAMES = ("relay_options", "capability_report")
    OUTPUT_TOOLTIPS = ("Wire to Prompt Relay Encode's relay_options input.",
                       "Which of the five controls the selected encoder path reads and which it ignores (from the "
                       "code), plus notes on values that have no effect, and the profile used.")
    FUNCTION = "execute"
    CATEGORY = _CATEGORY
    DESCRIPTION = "Optional per-stream tuning for the Prompt Relay encoders."

    def execute(self, video_strength, video_window_scale,
                audio_epsilon, audio_strength, audio_window_scale,
                profile_action="none", profile_name="", profile_target="any"):
        if profile_action not in ("none", "save", "load"):
            raise ValueError(f"Prompt Relay options: profile_action must be none, save or load, got {profile_action!r}")
        name = (profile_name or "").strip()
        if profile_action != "none" and not name:
            raise ValueError(f"Prompt Relay options: profile_action={profile_action} needs a profile_name")
        values = {"video_strength": video_strength, "video_window_scale": video_window_scale,
                  "audio_epsilon": audio_epsilon, "audio_strength": audio_strength,
                  "audio_window_scale": audio_window_scale}
        head = ""
        if profile_action == "save":
            save_profile(name, profile_target, values)
            head = f"profile {name!r} saved ({profile_target})\n"
        elif profile_action == "load":
            prof = load_profile(name)
            values = prof["values"]
            video_strength, video_window_scale = values["video_strength"], values["video_window_scale"]
            audio_epsilon, audio_strength = values["audio_epsilon"], values["audio_strength"]
            audio_window_scale = values["audio_window_scale"]
            head = f"profile {name!r} loaded (saved for {prof['target']})\n"
        report = head + capability_report(values, profile_target)
        opts = {
            "video_strength": video_strength,
            "video_window_scale": video_window_scale,
            "audio_epsilon": audio_epsilon if audio_epsilon > 0 else None,
            "audio_strength": audio_strength,
            "audio_window_scale": audio_window_scale,
        }
        return (opts, report)

