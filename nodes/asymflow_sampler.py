# SPDX-License-Identifier: Apache-2.0
"""AsymFlow sampler patch for ComfyUI.

Adapts the AsymFlow shifted-signal-ratio schedule from
Lakonik/LakonLab ``lakonlab/models/diffusions/asymflow.py`` to ComfyUI's
flow-matching ``ModelSamplingDiscreteFlow`` family. AsymFlow defines

    signal_ratio(sigma, shift) = alpha^2 / (alpha^2 + (shift * sigma)^2),
    where alpha = 1 - sigma.

Inverting ``signal_ratio = 1 - t`` for ``sigma`` given a uniform timestep
``t in [0, 1]`` yields::

    r     = sqrt(t / (1 - t))
    sigma = r / (shift + r)

That mapping is plugged into a custom ``ModelSamplingDiscreteFlow`` so the
sampler steps are re-distributed without retraining: low-rank / high-frequency
detail gets more attention at high noise when ``shift > 1`` (AsymFlow's
asymmetric velocity regime), and vice-versa for ``shift < 1``.

The node is intentionally a thin model patch — no sampling loop changes, no
inference-time backprop. It produces a new ``MODEL`` whose model_sampling is
swapped to the AsymFlow schedule. Compatible with any flow-matching backbone
that already uses ``ModelSamplingDiscreteFlow`` (SD3, Flux dev/schnell, etc.).
"""

from __future__ import annotations

import math
from typing import Any

import torch

from ._is_changed_util import hash_args_and_kwargs

try:
    from comfy import model_sampling as _ms  # type: ignore
    from comfy.model_patcher import ModelPatcher  # type: ignore  # noqa: F401
except Exception:  # pragma: no cover - allow import-only static analysis
    _ms = None  # type: ignore


def asymflow_time_shift(shift: float, t: torch.Tensor | float) -> torch.Tensor | float:
    """Map a uniform timestep ``t`` to a sigma under the AsymFlow schedule.

    sigma(t; shift) = r / (shift + r), with r = sqrt(t / (1 - t)).
    Returns the input type (tensor or python float).
    """
    if isinstance(t, torch.Tensor):
        t_c = t.clamp(min=1e-7, max=1.0 - 1e-7)
        r = (t_c / (1.0 - t_c)).sqrt()
        return r / (shift + r)
    t_f = max(min(float(t), 1.0 - 1e-7), 1e-7)
    r = math.sqrt(t_f / (1.0 - t_f))
    return r / (shift + r)


def _make_asymflow_sampling_cls():
    """Build a ``ModelSamplingDiscreteFlow`` subclass using AsymFlow shift.

    Done lazily so import of this module does not require ``comfy`` to be
    available (e.g. during static checks).
    """
    if _ms is None:
        raise RuntimeError("comfy.model_sampling not available")

    base = _ms.ModelSamplingDiscreteFlow

    class ModelSamplingAsymFlow(base):  # type: ignore[misc, valid-type]
        """ModelSampling for AsymFlow-shifted flow matching."""

        def set_parameters(self, shift: float = 1.0, timesteps: int = 1000, multiplier: int = 1000) -> None:  # type: ignore[override]
            self.shift = float(shift)
            self.multiplier = int(multiplier)
            ts = self.sigma(
                (torch.arange(1, timesteps + 1, 1) / timesteps) * multiplier
            )
            self.register_buffer("sigmas", ts)

        def sigma(self, timestep):  # type: ignore[override]
            return asymflow_time_shift(self.shift, timestep / self.multiplier)

        def percent_to_sigma(self, percent: float):  # type: ignore[override]
            if percent <= 0.0:
                return 1.0
            if percent >= 1.0:
                return 0.0
            return float(asymflow_time_shift(self.shift, 1.0 - percent))

    return ModelSamplingAsymFlow


class AsymFlowSamplerPatch:
    """ComfyUI node that patches a MODEL's sampling to the AsymFlow schedule."""

    CATEGORY = "Code2Collapse/Sampling"
    DESCRIPTION = (
        "Replace the model's flow-matching schedule with AsymFlow's "
        "shifted-signal-ratio mapping (sigma = r/(shift+r), r=sqrt(t/(1-t))). "
        "Inference-only; no retraining required. Best on flow models "
        "(SD3, Flux, AsymFlow-trained checkpoints)."
    )
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "patch"

    @classmethod
    def INPUT_TYPES(cls):  # noqa: N802 - ComfyUI convention
        return {
            "required": {
                "model": ("MODEL",),
                "shift": (
                    "FLOAT",
                    {
                        "default": 3.0,
                        "min": 0.05,
                        "max": 20.0,
                        "step": 0.05,
                        "tooltip": (
                            "AsymFlow signal-shift. shift=1 -> linear flow. "
                            ">1 spends more steps at high noise (recommended "
                            "for high-resolution / Flux-like models)."
                        ),
                    },
                ),
                "multiplier": (
                    "INT",
                    {
                        "default": 1000,
                        "min": 1,
                        "max": 10000,
                        "step": 1,
                        "tooltip": "Discretization multiplier (timesteps scale).",
                    },
                ),
            }
        }

    @classmethod
    def IS_CHANGED(cls, model, shift, multiplier, **kwargs):
        return hash_args_and_kwargs(model, shift, multiplier, **kwargs)

    def patch(self, model: Any, shift: float, multiplier: int):
        if _ms is None:
            raise RuntimeError(
                "comfy.model_sampling is unavailable; AsymFlowSamplerPatch "
                "requires a ComfyUI runtime."
            )
        asym_cls = _make_asymflow_sampling_cls()

        patched = model.clone()
        # Build a fresh model_sampling instance using the current model_config
        # if available, so sampling_settings (e.g. existing shift defaults)
        # propagate sensibly.
        mc = getattr(patched.model, "model_config", None)
        ms_inst = asym_cls(model_config=mc) if mc is not None else asym_cls()
        ms_inst.set_parameters(shift=float(shift), multiplier=int(multiplier))

        # Preserve prediction-type mixin behaviour (EPS / CONST / etc.) by
        # copying it from the existing model_sampling, mirroring how
        # ModelSamplingFlux/SD3 nodes do their patch.
        existing = getattr(patched.model, "model_sampling", None)
        if existing is not None:
            for attr in ("calculate_denoised", "calculate_input", "noise_scaling", "inverse_noise_scaling"):
                fn = getattr(existing, attr, None)
                if fn is not None and not hasattr(ms_inst, attr + "_overridden"):
                    setattr(ms_inst, attr, fn)

        patched.add_object_patch("model_sampling", ms_inst)
        return (patched,)


NODE_CLASS_MAPPINGS = {
    "AsymFlowSamplerPatch": AsymFlowSamplerPatch,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "AsymFlowSamplerPatch": "AsymFlow Sampler Patch (Lakonik signal-shift)",
}
