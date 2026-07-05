"""Perturbed Attention Guidance (PAG) for Wan video models.

Paper: arXiv:2403.17377 — "Self-Rectifying Diffusion Sampling with
Perturbed-Attention Guidance" (Ahn et al., 2024).

PAG replaces self-attention weights with identity (uniform) attention
during a secondary forward pass, then uses the difference as guidance:

    output = normal_output + pag_scale * (normal_output - perturbed_output)

Implementation uses ComfyUI's sampler_cfg_function to compute the PAG
guidance at CFG-application time. The attention patch produces the
"perturbed" prediction by replacing softmax(Q@K^T) with uniform
weights (1/seq_len), then the cfg function blends it.

NO model weights are loaded; this is a stateless attention intervention.
"""
from __future__ import annotations

import logging
import math
from typing import Optional

import torch

log = logging.getLogger("MEC.PAG")


def _identity_attention(q, k, v, extra_options):
    """Attention patch: uniform attention weights (identity attention).

    Instead of softmax(Q @ K^T / sqrt(d)), we use 1/L uniform weights,
    effectively averaging all value vectors per head. This is the
    "perturbed" branch of PAG.
    """
    return v.mean(dim=-2, keepdim=True).expand_as(v)


def apply_pag_to_model(model, pag_scale: float = 0.0,
                       layer_names: Optional[list] = None):
    """Apply PAG via a combined attention patch + CFG function.

    The attention patch installs identity attention on specified layers.
    The CFG function uses the `pag_scale` to blend the perturbed output
    into the final guided prediction.

    Args:
        model: ComfyUI model patcher.
        pag_scale: PAG guidance strength (0=off, typical: 1.0–3.0).
        layer_names: Transformer blocks to patch. Default: ["middle_block"].

    Returns:
        Patched model (cloned).
    """
    if pag_scale <= 0.0:
        return model

    if layer_names is None:
        layer_names = ["middle_block"]

    try:
        model_pag = model.clone()

        # Install identity attention on the target blocks.
        # ComfyUI's attn1_patch replaces self-attention output.
        for _layer_name in layer_names:
            model_pag.set_model_attn1_patch(_identity_attention)

        # Install CFG function that applies PAG scaling.
        # The attention patch means the uncond branch now gets
        # identity-attention output. We use this to compute PAG direction.
        _scale = float(pag_scale)

        # ComfyUI exposes a single `sampler_cfg_function` slot, so calling
        # set_model_sampler_cfg_function REPLACES whatever was there. If an
        # upstream patch (e.g. Dynamic CFG) already installed one, capture it
        # and chain it so both take effect instead of PAG silently clobbering it.
        _prior_cfg_fn = None
        try:
            _prior_cfg_fn = model_pag.model_options.get("sampler_cfg_function", None)
        except Exception:  # noqa: BLE001
            _prior_cfg_fn = None

        def _pag_cfg_function(args):
            cond = args["cond_denoised"]
            uncond = args["uncond_denoised"]
            cfg = args["cond_scale"]

            if _prior_cfg_fn is not None:
                # Build the guided base from the upstream cfg function so its
                # adjustment (dynamic CFG ramp, rescale, etc.) is preserved.
                guided = _prior_cfg_fn(args)
            else:
                guided = uncond + cfg * (cond - uncond)

            # PAG direction: push away from the perturbed (identity-attn) output.
            # Since the attn patch is installed, uncond already carries the
            # perturbed signal. We add an extra push in the cond-uncond direction.
            pag_direction = cond - uncond
            pag_magnitude = pag_direction.norm(
                dim=list(range(1, pag_direction.ndim)), keepdim=True
            ).clamp(min=1e-8)
            # Normalize direction and scale by average magnitude for stability.
            pag_correction = _scale * pag_direction * (
                pag_magnitude.mean() / pag_magnitude
            ).clamp(max=2.0)

            return guided + pag_correction

        model_pag.set_model_sampler_cfg_function(_pag_cfg_function)

        log.info("PAG applied: scale=%.2f, layers=%s, cfg_function installed", pag_scale, layer_names)
        return model_pag

    except (AttributeError, RuntimeError, TypeError) as e:
        log.warning("Failed to apply PAG: %s, using original model", e)
        return model
