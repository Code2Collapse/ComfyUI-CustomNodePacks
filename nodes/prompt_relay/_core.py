# Prompt Relay — core algorithm (refined port).
#
# Refined port of Gordon Chen's Prompt Relay reference implementation. The math
# (Gaussian temporal cost, segment scheduler, token-range mapping) is unchanged
# from upstream; this file reorganises it into a backend-agnostic core that
# both native and Kijai patchers consume.
# See ``NOTICE.md`` for full attribution.

from __future__ import annotations

import logging
import math
from typing import Callable, List, Optional, Sequence, Tuple

import torch

log = logging.getLogger("C2C.PromptRelay")


# ---------- temporal cost matrices ----------

def _build_temporal_cost(
    q_token_idx: List[dict],
    Lq: int,
    Lk: int,
    device: torch.device,
    dtype: torch.dtype,
    tokens_per_frame: int,
) -> torch.Tensor:
    """Gaussian penalty matrix [Lq, Lk] — integer-frame query indexing (Wan, video LTX)."""
    offset = torch.zeros(Lq, Lk, device=device, dtype=dtype)
    query_frames = torch.arange(Lq, device=device, dtype=torch.long) // tokens_per_frame
    for seg in q_token_idx:
        local = seg["local_token_idx"].to(device=device)
        d = (query_frames.float()[:, None] - seg["midpoint"]).abs()
        strength = seg.get("strength", 1.0)
        cost = strength * (torch.relu(d - seg["window"]) ** 2) / (2 * seg["sigma"] ** 2)
        offset[:, local] = cost.to(offset.dtype)
    return offset


def _build_temporal_cost_scaled(
    q_token_idx: List[dict],
    Lq: int,
    Lk: int,
    device: torch.device,
    dtype: torch.dtype,
    latent_frames: int,
) -> torch.Tensor:
    """Gaussian penalty matrix — fractional-frame indexing (LTX audio cross-attn)."""
    offset = torch.zeros(Lq, Lk, device=device, dtype=dtype)
    query_frames = torch.arange(Lq, device=device, dtype=torch.float32) * latent_frames / Lq
    for seg in q_token_idx:
        local = seg["local_token_idx"].to(device=device)
        d = (query_frames[:, None] - seg["midpoint"]).abs()
        sigma_a = seg.get("sigma_audio", seg["sigma"])
        window_a = seg.get("window_audio", seg["window"])
        strength_a = seg.get("strength_audio", 1.0)
        cost = strength_a * (torch.relu(d - window_a) ** 2) / (2 * sigma_a ** 2)
        offset[:, local] = cost.to(offset.dtype)
    return offset


def create_mask_fn(
    q_token_idx: List[dict],
    fallback_tokens_per_frame: int,
    latent_frames: int,
) -> Callable:
    """Build the cross-attention mask closure used by every backend.

    Returns ``mask_fn(q, k, transformer_options) -> Tensor | None``. None means
    "let optimized_attention handle it" (e.g. self-attention, unconditional pass,
    or cross-modal attention slots that aren't text↔video).
    """
    cache: dict = {}
    max_token_idx = max(int(seg["local_token_idx"].max().item()) for seg in q_token_idx) + 1

    def mask_fn(q, k, transformer_options):
        Lq, Lk = q.shape[1], k.shape[1]
        if Lq == Lk:
            return None

        # Conditional pass only — never penalise the negative prompt.
        cond_or_uncond = transformer_options.get("cond_or_uncond", []) if transformer_options else []
        if 1 in cond_or_uncond and 0 not in cond_or_uncond:
            return None

        grid_sizes = transformer_options.get("grid_sizes", None) if transformer_options else None
        video_tpf = (
            int(grid_sizes[1]) * int(grid_sizes[2])
            if grid_sizes is not None
            else fallback_tokens_per_frame
        )
        video_lq = latent_frames * video_tpf

        # Skip cross-modal attention that shouldn't be temporally masked.
        if Lk == video_lq or Lk < max_token_idx:
            return None

        mode = "video" if Lq == video_lq else "scaled"
        key = (Lq, Lk, mode, q.device)
        if key not in cache:
            if mode == "video":
                cost = _build_temporal_cost(q_token_idx, Lq, Lk, q.device, q.dtype, video_tpf)
            else:
                cost = _build_temporal_cost_scaled(q_token_idx, Lq, Lk, q.device, q.dtype, latent_frames)
            log.info(
                "Built penalty matrix (%s): Lq=%d Lk=%d nonzero=%d/%d",
                mode, Lq, Lk, int((cost > 0).sum().item()), cost.numel(),
            )
            cache[key] = -cost
        return cache[key].to(q.dtype)

    return mask_fn


# ---------- segment scheduling ----------

def build_segments(
    token_ranges: Sequence[Tuple[int, int]],
    segment_lengths: Sequence[int],
    epsilon: float = 1e-3,
    relay_options: Optional[dict] = None,
) -> List[dict]:
    """Per-segment metadata (midpoint, window, sigma, strength) for the penalty.

    ``relay_options`` (optional) overrides per-stream knobs:
        ``video_strength``, ``video_window_scale``,
        ``audio_epsilon``, ``audio_strength``, ``audio_window_scale``.
    """
    sigma = 1.0 / math.log(1.0 / epsilon) if 0 < epsilon < 1 else 0.1448
    opts = relay_options or {}
    v_strength = float(opts.get("video_strength", 1.0))
    v_window_scale = float(opts.get("video_window_scale", 1.0))
    a_epsilon = opts.get("audio_epsilon")
    a_strength = float(opts.get("audio_strength", 1.0))
    a_window_scale = float(opts.get("audio_window_scale", 1.0))

    if a_epsilon is not None and 0 < float(a_epsilon) < 1:
        sigma_audio = 1.0 / math.log(1.0 / float(a_epsilon))
    else:
        sigma_audio = sigma

    if relay_options:
        log.info(
            "Advanced options — video: strength=%.3f window_scale=%.3f | "
            "audio: epsilon=%s strength=%.3f window_scale=%.3f",
            v_strength, v_window_scale,
            f"{float(a_epsilon):.4f}" if a_epsilon is not None else "inherit",
            a_strength, a_window_scale,
        )

    q_token_idx: List[dict] = []
    frame_cursor = 0
    for (tok_start, tok_end), L in zip(token_ranges, segment_lengths):
        if L <= 0:
            frame_cursor += L
            continue
        midpoint = (2 * frame_cursor + L) // 2
        base_window = max(L // 2 - 2, 0)
        q_token_idx.append({
            "local_token_idx": torch.arange(tok_start, tok_end),
            "midpoint": midpoint,
            "window": max(base_window * v_window_scale, 0.0),
            "sigma": sigma,
            "strength": v_strength,
            "window_audio": max(base_window * a_window_scale, 0.0),
            "sigma_audio": sigma_audio,
            "strength_audio": a_strength,
        })
        frame_cursor += L
    return q_token_idx


def distribute_segment_lengths(
    num_segments: int,
    latent_frames: int,
    specified_lengths: Optional[Sequence[int]] = None,
) -> List[int]:
    """Validate/auto-distribute per-segment frame counts, capped to ``latent_frames``."""
    if specified_lengths:
        if len(specified_lengths) != num_segments:
            raise ValueError(
                f"Number of segment_lengths ({len(specified_lengths)}) "
                f"must match number of local prompts ({num_segments})"
            )
        lengths = list(specified_lengths)
    else:
        step = -(-latent_frames // num_segments)  # ceil division
        lengths = [step] * num_segments

    effective: List[int] = []
    cursor = 0
    for L in lengths:
        end = min(cursor + L, latent_frames)
        effective.append(max(end - cursor, 0))
        cursor = end
    return effective


def convert_pixel_to_latent_lengths(
    pixel_lengths: Sequence[int],
    temporal_stride: int,
    latent_frames: int,
) -> List[int]:
    """Largest-remainder conversion of pixel-space frame counts to latent frames."""
    if not pixel_lengths:
        return []
    total_pixel = sum(pixel_lengths)
    if total_pixel <= 0:
        return [1] * len(pixel_lengths)
    naive_total = max(1, round(total_pixel / temporal_stride))
    target_total = min(latent_frames, naive_total)
    if target_total >= latent_frames - 1:
        target_total = latent_frames

    exact = [p * target_total / total_pixel for p in pixel_lengths]
    result = [int(e) for e in exact]
    diff = target_total - sum(result)
    if diff > 0:
        order = sorted(range(len(exact)), key=lambda i: -(exact[i] - int(exact[i])))
        for k in range(diff):
            result[order[k % len(order)]] += 1
    for i in range(len(result)):
        if result[i] < 1:
            max_idx = max(range(len(result)), key=lambda j: result[j])
            if result[max_idx] > 1:
                result[max_idx] -= 1
                result[i] = 1
    return result


# ---------- tokenization (native ComfyUI CLIP) ----------

def get_raw_tokenizer_native(clip):
    """Extract the underlying SentencePiece/HF tokenizer from a ComfyUI CLIP object."""
    tokenizer_wrapper = clip.tokenizer
    for attr_name in dir(tokenizer_wrapper):
        if attr_name.startswith("_"):
            continue
        inner = getattr(tokenizer_wrapper, attr_name, None)
        if inner is not None and hasattr(inner, "tokenizer"):
            return inner.tokenizer
    raise RuntimeError(
        "Could not find raw tokenizer on CLIP object. "
        f"Known attributes: {[a for a in dir(tokenizer_wrapper) if not a.startswith('_')]}"
    )


def map_token_indices(
    raw_tokenizer,
    global_prompt: str,
    local_prompts: Sequence[str],
) -> Tuple[str, List[Tuple[int, int]]]:
    """Tokenize ``global + ' '.join(locals)``; return the joined prompt and per-local token ranges.

    Works with any ``raw_tokenizer`` exposing ``tokenizer(text)["input_ids"]`` —
    covers both ComfyUI's HF wrappers and Kijai's ``HuggingfaceTokenizer``.
    """
    prefixed = [" " + lp for lp in local_prompts]
    full_prompt = global_prompt + "".join(prefixed)
    has_eos = bool(getattr(raw_tokenizer, "add_eos", False))
    eos_adj = 1 if has_eos else 0

    def _token_count(text: str) -> int:
        ids = raw_tokenizer(text)["input_ids"]
        # HuggingfaceTokenizer in Kijai returns a 2D tensor [batch, seq]; flatten to first batch.
        if hasattr(ids, "shape") and len(getattr(ids, "shape", ())) >= 2:
            return int(ids.shape[-1])
        return len(ids)

    prev_len = _token_count(global_prompt) - eos_adj
    token_ranges: List[Tuple[int, int]] = []
    built = global_prompt
    for plp in prefixed:
        built += plp
        cur_len = _token_count(built) - eos_adj
        if cur_len <= prev_len:
            raise ValueError(f"Local prompt produced no tokens: {plp.strip()!r}")
        token_ranges.append((prev_len, cur_len))
        prev_len = cur_len
    return full_prompt, token_ranges
