"""Phase-shift sampling for Wan video models.

Inspired by fxtdstudios/radiance Sampler Pro's phase-shift mode.

The idea: different samplers excel at different denoising phases.
Euler is excellent at the early high-noise regime (layout, structure),
while DPM++ variants are better at the late low-noise refinement
(textures, fine detail). Phase-shift combines both by switching
sampler mid-schedule with a smooth sigma crossfade.

Supported modes:
    Phase-Shift (Euler→DPM):   Euler for early steps, DPM++ 2M for late
    Phase-Shift (Euler→SGM):   Euler for early steps, SGM Uniform for late

The transition point defaults to 70% of steps but is configurable.
A 3-step cosine sigma blend ensures the switch doesn't introduce
artifacts.
"""
from __future__ import annotations

import logging
import math
from typing import Tuple

import torch

log = logging.getLogger("MEC.PhaseShift")


def compute_phase_shift_split(
    total_steps: int,
    split_pct: float = 0.70,
    denoise: float = 1.0,
) -> int:
    """Compute the step index where the sampler transition occurs."""
    effective_steps = max(1, int(total_steps * denoise) if denoise < 1.0 else total_steps)
    start = total_steps - effective_steps
    split = start + max(1, int(effective_steps * split_pct))
    return min(split, total_steps - 1)


def gradual_sigma_blend(
    sigmas_a: torch.Tensor,
    sigmas_b: torch.Tensor,
    blend_steps: int = 3,
) -> torch.Tensor:
    """Smooth cosine blend between two sigma schedules at the transition.

    Args:
        sigmas_a: Sigma schedule from the first sampler phase.
        sigmas_b: Sigma schedule for the second sampler phase.
        blend_steps: Number of steps over which to blend.

    Returns:
        Blended sigma schedule.
    """
    if blend_steps <= 0 or len(sigmas_a) == 0 or len(sigmas_b) == 0:
        return sigmas_b

    result = sigmas_b.clone()
    last_sigma_a = sigmas_a[-1].item()
    blend_steps = min(blend_steps, len(result) - 1)

    result[0] = last_sigma_a

    for i in range(1, blend_steps):
        t = i / blend_steps
        blend_factor = 0.5 * (1.0 - math.cos(math.pi * t))
        result[i] = last_sigma_a * (1.0 - blend_factor) + sigmas_b[i].item() * blend_factor

    log.debug(
        "Sigma blend: %d steps, transition %.4f → %.4f",
        blend_steps, last_sigma_a, sigmas_b[blend_steps - 1].item()
    )
    return result


def build_phase_shift_schedule(
    model,
    total_steps: int,
    scheduler_early: str = "simple",
    scheduler_late: str = "simple",
    split_pct: float = 0.70,
    denoise: float = 1.0,
    shift: float = 8.0,
) -> Tuple[torch.Tensor, int]:
    """Build a composite sigma schedule for phase-shift sampling.

    Args:
        model: ComfyUI model (for sigma computation).
        total_steps: Total sampling steps.
        scheduler_early: Scheduler for the early phase.
        scheduler_late: Scheduler for the late phase.
        split_pct: Fraction of steps for the early phase.
        denoise: Denoise strength.
        shift: Sigma shift value (Wan default: 8.0).

    Returns:
        Tuple of (blended_sigmas, split_step_index).
    """
    import comfy.samplers

    split_step = compute_phase_shift_split(total_steps, split_pct, denoise)

    # Generate sigma schedules for both phases
    sigmas_full_early = comfy.samplers.calculate_sigmas(
        model.get_model_object("model_sampling"),
        scheduler_early, total_steps
    )
    sigmas_full_late = comfy.samplers.calculate_sigmas(
        model.get_model_object("model_sampling"),
        scheduler_late, total_steps
    )

    # Splice: early sigmas up to split, late sigmas from split onward
    sigmas_early = sigmas_full_early[:split_step + 1]
    sigmas_late = sigmas_full_late[split_step:]

    # Blend at the transition
    sigmas_late = gradual_sigma_blend(sigmas_early, sigmas_late, blend_steps=3)

    # Concatenate
    result = torch.cat([sigmas_early[:-1], sigmas_late])

    log.info(
        "Phase-shift schedule: %s (steps 0–%d) → %s (steps %d–%d), "
        "sigma range [%.4f → %.4f]",
        scheduler_early, split_step, scheduler_late, split_step, total_steps,
        result[0].item(), result[-1].item()
    )

    return result, split_step
