"""Dynamic CFG and guidance rescale for Wan video models.

Inspired by fxtdstudios/radiance Sampler Pro's dynamic guidance system.
Clean-room implementation using the same mathematical principles:

Dynamic CFG:
    Rather than a flat CFG value across all denoising steps, we ramp
    the CFG using a 3-phase cosine schedule:
        - Early phase (0%–15%):  CFG * 1.2 (stronger structure guidance)
        - Mid phase (15%–85%):   CFG * 1.0 (nominal)
        - Late phase (85%–100%): CFG * 0.7 (softer detail refinement)
    Transitions between phases use smooth cosine blending to avoid
    artifacts from abrupt CFG changes.

Guidance Rescale (phi):
    After computing guided = uncond + cfg * (cond - uncond), we rescale
    the guided output to match the standard deviation of the conditional
    output. This prevents color oversaturation and brightness drift that
    occurs at high CFG values. The phi parameter controls the blend
    between rescaled and raw guided output:
        result = phi * rescaled + (1 - phi) * guided

Both are pure-tensor operations with no model weights.
"""
from __future__ import annotations

import math
from typing import Optional

import torch

# Dynamic CFG constants
DYNAMIC_CFG_EARLY_MULTIPLIER = 1.2
DYNAMIC_CFG_LATE_MULTIPLIER = 0.7
DYNAMIC_CFG_EARLY_THRESHOLD = 0.15
DYNAMIC_CFG_LATE_THRESHOLD = 0.85
DYNAMIC_CFG_RAMP_WIDTH = 0.05

# Dynamic Guidance (for embedding-guided models like Flux, but also
# usable as a secondary guidance modulation for Wan's text conditioning)
DYNAMIC_GUIDANCE_EARLY_MULTIPLIER = 0.6
DYNAMIC_GUIDANCE_LATE_MULTIPLIER = 0.95
DYNAMIC_GUIDANCE_EARLY_THRESHOLD = 0.20
DYNAMIC_GUIDANCE_LATE_THRESHOLD = 0.90
DYNAMIC_GUIDANCE_RAMP_WIDTH = 0.05


def _cosine_blend(progress: float, threshold: float, ramp: float,
                  val_before: float, val_after: float) -> float:
    """Smooth cosine transition centered at `threshold` over width `2*ramp`."""
    if progress < threshold - ramp:
        return val_before
    if progress > threshold + ramp:
        return val_after
    t = (progress - (threshold - ramp)) / (2 * ramp)
    blend = 0.5 * (1.0 - math.cos(math.pi * t))
    return val_before + (val_after - val_before) * blend


def compute_dynamic_cfg(
    base_cfg: float,
    step: int,
    total_steps: int,
    denoise: float = 1.0,
) -> float:
    """Compute step-dependent CFG using a 3-phase cosine schedule.

    Args:
        base_cfg: The user's nominal CFG value.
        step: Current step index (0-based).
        total_steps: Total number of sampling steps.
        denoise: Denoise strength (affects effective step range).

    Returns:
        Modulated CFG value for this step.
    """
    if total_steps <= 0 or base_cfg <= 1.0:
        return base_cfg

    denoising_steps = max(1, int(total_steps * denoise) if denoise < 1.0 else total_steps)
    denoising_start = total_steps - denoising_steps
    progress = max(0.0, min(1.0, (step - denoising_start) / denoising_steps))

    cfg_early = base_cfg * DYNAMIC_CFG_EARLY_MULTIPLIER
    cfg_late = base_cfg * DYNAMIC_CFG_LATE_MULTIPLIER

    # Phase 1: early → nominal
    val = _cosine_blend(progress, DYNAMIC_CFG_EARLY_THRESHOLD,
                        DYNAMIC_CFG_RAMP_WIDTH, cfg_early, base_cfg)
    # Phase 2: nominal → late
    val = _cosine_blend(progress, DYNAMIC_CFG_LATE_THRESHOLD,
                        DYNAMIC_CFG_RAMP_WIDTH, val, cfg_late)
    return val


def guidance_rescale(
    cond_output: torch.Tensor,
    uncond_output: torch.Tensor,
    cfg: float,
    phi: float = 0.7,
) -> torch.Tensor:
    """Apply guidance with std-deviation rescaling to prevent oversaturation.

    Standard CFG: guided = uncond + cfg * (cond - uncond)
    This often oversaturates colors at high CFG. The fix: rescale the
    guided output so its per-sample std matches the conditional output,
    then blend with the raw guided output using phi.

    Args:
        cond_output: Conditional model output.
        uncond_output: Unconditional model output.
        cfg: Classifier-free guidance scale.
        phi: Rescale blend factor (0=raw guided, 1=fully rescaled).

    Returns:
        Guided output with optional rescaling applied.
    """
    if phi <= 0.0 or cfg <= 1.0:
        return uncond_output + cfg * (cond_output - uncond_output)

    guided = uncond_output + cfg * (cond_output - uncond_output)

    dims = list(range(1, guided.ndim))
    guided_std = guided.std(dim=dims, keepdim=True).clamp(min=1e-6)
    cond_std = cond_output.std(dim=dims, keepdim=True).clamp(min=1e-6)

    rescaled = guided * (cond_std / guided_std)

    return rescaled * phi + guided * (1.0 - phi)


def build_dynamic_cfg_patch(
    base_cfg: float,
    total_steps: int = 0,
    denoise: float = 1.0,
    phi: float = 0.0,
):
    """Build a model sampler_cfg_function patch for dynamic CFG + rescale.

    Install on a model via:
        model = model.clone()
        model.set_model_sampler_cfg_function(patch_fn)

    Args:
        base_cfg: Nominal CFG value.
        total_steps: Total sampling steps. 0 = auto-detect from
            ``model_options["sampler_cfg_function"]`` call count or
            ``sigmas`` length if available.
        denoise: Denoise strength.
        phi: Guidance rescale phi (0=off).

    Returns:
        Callable suitable for set_model_sampler_cfg_function.
    """
    state = {"step": 0, "total": total_steps}

    def cfg_function(args):
        cond = args["cond_denoised"]
        uncond = args["uncond_denoised"]

        # Auto-detect total steps from the sigma schedule if not provided.
        if state["total"] <= 0:
            sigma = args.get("sigma", None)
            model_options = args.get("model_options", {})
            sigmas = model_options.get("sigmas", None)
            if sigmas is not None and hasattr(sigmas, "__len__"):
                state["total"] = max(1, len(sigmas) - 1)
            elif sigma is not None:
                state["total"] = 30  # last-resort fallback

        step = state["step"]
        state["step"] += 1

        effective_cfg = compute_dynamic_cfg(base_cfg, step, state["total"], denoise)

        if phi > 0.0:
            return guidance_rescale(cond, uncond, effective_cfg, phi)
        else:
            return uncond + effective_cfg * (cond - uncond)

    return cfg_function
