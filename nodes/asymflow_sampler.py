# SPDX-License-Identifier: Apache-2.0
"""AsymFlow sampler patch for ComfyUI.

Adapts the AsymFlow shifted-signal-ratio schedule from
Lakonik/LakonLab ``lakonlab/models/diffusions/asymflow.py`` to ComfyUI's
flow-matching ``ModelSamplingDiscreteFlow`` family. AsymFlow defines

    signal_ratio(sigma, shift) = alpha^2 / (alpha^2 + (shift * sigma)^2),
    where alpha = 1 - sigma.

A pretrained flow model's sigma keeps its own meaning (x = (1 - sigma) * x0 + sigma * noise), so the
patch maps a uniform timestep ``t`` to the sigma whose UNSHIFTED signal ratio equals the shifted signal
ratio of ``t``::

    signal_ratio(sigma, 1) = signal_ratio(t, shift)
    =>  sigma / (1 - sigma) = shift * t / (1 - t)
    =>  sigma = shift * t / (1 + (shift - 1) * t)

(the same SNR shift ComfyUI's ``time_snr_shift`` / ModelSamplingSD3 use: AsymFlow's signal ratio is
SNR / (SNR + shift^2), an SNR scaled by 1/shift^2). shift = 1 is the linear flow schedule, and
shift > 1 spends more steps at high noise (shift = 3: 75 % of the schedule above sigma 0.5).

Audit 2026-11-27 (L-aimodels): the earlier code solved ``signal_ratio(sigma, shift) = 1 - t``
(sigma = r / (shift + r), r = sqrt(t / (1 - t))). That is not linear at shift = 1 and runs the OTHER
way: shift = 3 put 90 % of the steps below sigma 0.5 and shift = 12 put 99 % there, the opposite of
this node's own tooltip. It also dropped the CONST prediction class, the MiniMax H3 audio_scale and a
Flux model's timestep convention; all three are now kept from the patched model.

The node is intentionally a thin model patch — no sampling loop changes, no
inference-time backprop. It produces a new ``MODEL`` whose model_sampling is
swapped to the AsymFlow schedule. Compatible with any flow-matching backbone
that already uses ``ModelSamplingDiscreteFlow`` (SD3, Flux dev/schnell, etc.).
"""

from __future__ import annotations

from typing import Any

import torch

from ._is_changed_util import hash_args_and_kwargs

try:
    from comfy import model_sampling as _ms  # type: ignore
    from comfy.model_patcher import ModelPatcher  # type: ignore  # noqa: F401
except Exception:  # pragma: no cover - allow import-only static analysis
    _ms = None  # type: ignore


def asymflow_time_shift(shift: float, t: torch.Tensor | float) -> torch.Tensor | float:
    """Map a uniform timestep ``t`` to a sigma under the AsymFlow signal-shift schedule.

    sigma(t; shift) = shift * t / (1 + (shift - 1) * t), i.e. the sigma whose unshifted signal ratio
    equals AsymFlow's shifted signal ratio of ``t`` (see the module docstring). Returns the input type.
    """
    shift = float(shift)
    if isinstance(t, torch.Tensor):
        return shift * t / (1.0 + (shift - 1.0) * t)
    t_f = float(t)
    return shift * t_f / (1.0 + (shift - 1.0) * t_f)


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

        # Packed audio-video flow models (MiniMax H3, core ModelSamplingAV) carry the audio stream at
        # shift / audio_shift; the patch copies audio_shift from the replaced sampling (None = no audio).
        audio_shift = None

        @property
        def audio_scale(self):
            if self.audio_shift is None:
                return 1.0
            return self.shift / self.audio_shift

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
        "shifted-signal-ratio mapping (sigma = shift*t/(1+(shift-1)*t)). "
        "Inference-only; no retraining required. Best on flow models "
        "(SD3, Flux, AsymFlow-trained checkpoints)."
    )
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    OUTPUT_TOOLTIPS = ("The same model with the AsymFlow schedule - wire it to the sampler.",)
    FUNCTION = "patch"

    @classmethod
    def INPUT_TYPES(cls):  # noqa: N802 - ComfyUI convention
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "A flow-matching model (SD3, Flux, AsymFlow-trained) whose noise schedule is replaced."}),
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
                            "for high-resolution / Flux-like models). It replaces "
                            "any earlier shift node (e.g. MiniMax H3 Sigma Shift): use one or the other."
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
        # L9.264 (CNP-NODES-2 audit): the signal-shift schedule is for FLOW models (CONST prediction: Flux, SD3, Wan,
        # H3 ...). On an EPS / v-prediction model it silently produced wrong output; refuse instead.
        existing0 = getattr(getattr(model, "model", None), "model_sampling", None)
        flow_types = tuple(t for t in (getattr(_ms, "CONST", None), getattr(_ms, "COSMOS_RFLOW", None)) if t)
        if existing0 is not None and flow_types and not isinstance(existing0, flow_types):
            raise ValueError(
                "AsymFlow Sampler Patch only works with flow models (Flux, SD3, Wan, H3 and other rectified-flow "
                f"models). This model predicts with {type(existing0).__name__}, which this schedule would get wrong. "
                "Remove the node for this model.")
        asym_cls = _make_asymflow_sampling_cls()
        # Keep the model's prediction class (CONST, IMG_TO_IMG_FLOW, COSMOS_RFLOW ...): the k-diffusion samplers
        # test isinstance(model_sampling, CONST), so a copied method alone is not enough (audit 2026-11-27).
        mixin = None
        if existing0 is not None and flow_types:
            for klass in type(existing0).__mro__:
                if issubclass(klass, flow_types) and not issubclass(klass, torch.nn.Module):
                    mixin = klass
                    break
        if mixin is not None and not issubclass(asym_cls, mixin):
            asym_cls = type("ModelSamplingAsymFlow", (asym_cls, mixin), {})

        # The model's own timestep convention wins over the widget: Flux feeds sigma itself (x1), Wan / H3 / SD3
        # feed sigma * 1000. A different multiplier would give a pretrained model timesteps it never saw.
        mult = int(multiplier)
        model_mult = getattr(existing0, "multiplier", None)
        if model_mult is None and existing0 is not None and callable(getattr(existing0, "timestep", None)):
            try:
                model_mult = float(existing0.timestep(torch.tensor(1.0)))
            except Exception:
                model_mult = None
        if model_mult is not None and float(model_mult) > 0 and abs(float(model_mult) - mult) > 1e-6:
            print(f"[AsymFlow] multiplier {mult} ignored: this model uses timestep = sigma x {float(model_mult):g}.")
            mult = int(round(float(model_mult))) or 1

        patched = model.clone()
        # Build a fresh model_sampling instance using the current model_config
        # if available, so sampling_settings (e.g. existing shift defaults)
        # propagate sensibly.
        mc = getattr(patched.model, "model_config", None)
        ms_inst = asym_cls(model_config=mc) if mc is not None else asym_cls()
        ms_inst.set_parameters(shift=float(shift), multiplier=mult)
        if getattr(existing0, "audio_shift", None) is not None:
            ms_inst.audio_shift = existing0.audio_shift

        # No known prediction class found: fall back to borrowing the existing methods.
        existing = getattr(patched.model, "model_sampling", None)
        if existing is not None and mixin is None:
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
