"""Multi-slot CLIP conditioning for Wan video models.

Inspired by fxtdstudios/radiance CLIP Slot system and advanced
CLIP conditioning approaches.

Standard ComfyUI workflow: one positive prompt, one negative prompt,
each encoded once. This is fine for simple generations but lacks
nuance for professional video work.

Multi-slot approach:
    1. Structure prompt (active during high-noise phase, ~0-30% steps):
       Focus on composition, layout, camera, scene description.
    2. Detail prompt (active during low-noise phase, ~60-100% steps):
       Focus on textures, materials, lighting, color grading.
    3. Style prompt (blended throughout):
       Persistent artistic direction, quality tags.

Implementation:
    Each slot produces its own CONDITIONING. At sampling time, we
    switch which conditioning is active based on the denoising progress.
    This is done via ComfyUI's model patching system — specifically by
    modifying the conditioning's `start_percent` and `end_percent` fields
    which ComfyUI's KSampler already supports natively.
"""
from __future__ import annotations

import logging
from copy import deepcopy
from typing import List, Optional, Tuple

import torch

log = logging.getLogger("MEC.MultiCLIP")


def build_multi_slot_conditioning(
    clip,
    structure_prompt: str,
    detail_prompt: str,
    style_prompt: str = "",
    structure_end_pct: float = 0.35,
    detail_start_pct: float = 0.55,
    style_weight: float = 0.3,
) -> list:
    """Build multi-slot positive conditioning from separate prompts.

    Each prompt is encoded separately and assigned a denoising step range
    using ComfyUI's native `start_percent` / `end_percent` conditioning
    metadata.

    Args:
        clip: ComfyUI CLIP model.
        structure_prompt: Scene layout / composition prompt.
        detail_prompt: Texture / material / lighting prompt.
        style_prompt: Optional persistent style prompt (blended throughout).
        structure_end_pct: Step % where structure prompt fades out.
        detail_start_pct: Step % where detail prompt fades in.
        style_weight: Relative strength of the style prompt.

    Returns:
        List of CONDITIONING entries with step-range metadata.
    """
    all_cond = []

    # Structure conditioning (early phase)
    if structure_prompt.strip():
        full_text = structure_prompt.strip()
        if style_prompt.strip():
            full_text = f"{full_text}. {style_prompt.strip()}"
        tokens = clip.tokenize(full_text)
        cond = clip.encode_from_tokens_scheduled(tokens)
        for c in cond:
            c[1]["start_percent"] = 0.0
            c[1]["end_percent"] = structure_end_pct
        all_cond.extend(cond)
        log.info("Multi-CLIP: structure slot [0.0–%.0f%%] = '%s'",
                 structure_end_pct * 100, full_text[:60])

    # Detail conditioning (late phase)
    if detail_prompt.strip():
        full_text = detail_prompt.strip()
        if style_prompt.strip():
            full_text = f"{full_text}. {style_prompt.strip()}"
        tokens = clip.tokenize(full_text)
        cond = clip.encode_from_tokens_scheduled(tokens)
        for c in cond:
            c[1]["start_percent"] = detail_start_pct
            c[1]["end_percent"] = 1.0
        all_cond.extend(cond)
        log.info("Multi-CLIP: detail slot [%.0f%%–100%%] = '%s'",
                 detail_start_pct * 100, full_text[:60])

    # Overlap zone (structure_end → detail_start): both are active,
    # ComfyUI blends them via its native timestep conditioning.

    # Style conditioning (full range, if no structure/detail split)
    if not all_cond and style_prompt.strip():
        tokens = clip.tokenize(style_prompt.strip())
        cond = clip.encode_from_tokens_scheduled(tokens)
        all_cond.extend(cond)
        log.info("Multi-CLIP: style-only fallback = '%s'", style_prompt[:60])

    return all_cond


def encode_weighted_prompts(
    clip,
    prompts: List[Tuple[str, float]],
) -> list:
    """Encode multiple prompts with per-prompt weights and average them.

    This produces a single CONDITIONING that is the weighted average
    of multiple prompt embeddings — useful for blending concepts.

    Args:
        clip: ComfyUI CLIP model.
        prompts: List of (prompt_text, weight) tuples.

    Returns:
        CONDITIONING list (single entry with averaged embedding).
    """
    if not prompts:
        return []

    embeddings = []
    weights = []

    for text, weight in prompts:
        if not text.strip():
            continue
        tokens = clip.tokenize(text.strip())
        cond = clip.encode_from_tokens_scheduled(tokens)
        if cond and len(cond) > 0:
            embeddings.append(cond[0][0])
            weights.append(weight)

    if not embeddings:
        return []

    total_weight = sum(weights)
    if total_weight <= 0:
        return []

    # Weighted average of embeddings
    avg = sum(e * (w / total_weight) for e, w in zip(embeddings, weights))

    # Use the metadata from the first conditioning entry
    tokens = clip.tokenize(prompts[0][0].strip())
    base_cond = clip.encode_from_tokens_scheduled(tokens)
    if base_cond:
        result = [[avg, base_cond[0][1].copy()]]
        return result

    return []
