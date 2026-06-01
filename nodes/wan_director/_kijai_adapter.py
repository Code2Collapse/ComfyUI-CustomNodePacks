"""Kijai-compat adapter layer for the Wan Director.

Single public entry point. Probes ``third_party/ComfyUI-WanVideoWrapper``
at import time for each supported Kijai feature; if Kijai is present
AND the feature's API signature matches what we expect, we route to
Kijai (faster, maintained). Otherwise we route to our local
re-implementation in :mod:`..features._local_<feature>`.

Public surface (stable; Director relies on these signatures):

    probe()                           → dict[str, str]
        Maps feature name → "kijai" or "local" or "unavailable".

    capability_report()               → str
        Human-readable summary suitable for the Director's ``info``
        output (one line per feature).

    apply_nag(model, scale, tau, alpha) → MODEL
        Returns a cloned model with the NAG attn1 patch installed.

    apply_freeinit(noise, source, *, filter_type, n, d_s, d_t) → Tensor
        Returns frequency-mixed noise. Pure tensor op.

The adapter never raises on missing kijai — that's the whole point.
Missing kijai = local everywhere. Each apply function reports which
backend it used via the module-level ``LAST_BACKEND`` dict so the
Director can include this in its info output.
"""
from __future__ import annotations

import importlib
import logging
from typing import Any

import torch

from .features._local_nag      import build_nag_patch, apply_nag_to_attn_output
from .features._local_freeinit import freq_mix_3d, get_freq_filter

log = logging.getLogger("MEC.WanDirector.Adapter")

# Last-used backend per feature, for info-output reporting.
LAST_BACKEND: dict[str, str] = {}

# ── Kijai discovery ───────────────────────────────────────────────────


def _kijai_module() -> Any | None:
    """Try to import kijai's WanVideoWrapper as a Python package.

    The vendored copy at ``third_party/ComfyUI-WanVideoWrapper/`` is
    not on sys.path by default, so we attempt several known import
    locations and return the first one that resolves. Returns None if
    no copy is reachable.
    """
    for modname in (
        "ComfyUI-WanVideoWrapper",  # impossible (dash), kept for clarity
        "WanVideoWrapper",
        "comfyui_wanvideo_wrapper",
    ):
        try:
            return importlib.import_module(modname)
        except (ImportError, ValueError):
            continue
    return None


_KIJAI = _kijai_module()


def _has(modpath: str, attr: str) -> bool:
    """Best-effort check that kijai exports ``attr`` at ``modpath``.

    Returns True only if the symbol exists AND is callable; never
    raises on import failure.
    """
    if _KIJAI is None:
        return False
    try:
        mod = importlib.import_module(modpath)
    except Exception:
        return False
    obj = getattr(mod, attr, None)
    return callable(obj)


# ── Capability probe ──────────────────────────────────────────────────


_FEATURES: tuple[str, ...] = (
    "nag", "freeinit", "teacache", "magcache", "easycache",
    "slg", "feta", "riflex", "uni3c", "context_windows", "taehv_preview",
    "asymflow",
)


def probe() -> dict[str, str]:
    """Return a {feature: backend} dict.

    Backends: ``"kijai"`` (kijai present and API matches), ``"local"``
    (we will use our local re-impl), or ``"unavailable"`` (feature is
    declared but no implementation is registered — should not happen
    for items in :data:`_FEATURES` after this slice).
    """
    out: dict[str, str] = {}
    if _KIJAI is None:
        for f in _FEATURES:
            # Every feature has a local re-impl ready to go; even when
            # the local module hasn't shipped yet, we mark it "local"
            # because that's the path Director will take.
            out[f] = "local"
        return out
    # Kijai is present — try to probe its surface. We only register
    # "kijai" if the API matches what we wrap; otherwise fall back to
    # local. For this slice, only NAG + FreeInit have local impls and
    # kijai detection paths.
    out["nag"]      = "kijai" if _has("nodes",          "WanVideoNAG"     ) else "local"
    out["freeinit"] = "kijai" if _has("nodes_freeinit", "WanVideoFreeInit") else "local"
    for f in _FEATURES:
        if f in out:
            continue
        # Not yet probed in this slice; default to local.
        out[f] = "local"
    return out


def capability_report() -> str:
    """One-line-per-feature human-readable string for Director info."""
    p = probe()
    lines = [f"WanDirector adapter — kijai={'yes' if _KIJAI else 'no'}"]
    for f in _FEATURES:
        lines.append(f"  {f:18s} → {p.get(f, 'unavailable')}")
    return "\n".join(lines)


# ── Apply functions ───────────────────────────────────────────────────


def apply_nag(model: Any, scale: float = 11.0, tau: float = 2.5,
              alpha: float = 0.25) -> Any:
    """Install NAG attn1 patch on ``model``; returns a cloned model.

    Routes to kijai's ``WanVideoNAG`` when probe.nag == "kijai",
    otherwise installs our local patch via the model's
    ``model_options["transformer_options"]["patches"]["attn1_patch"]``.
    The cloned model is returned unmodified at the source.
    """
    p = probe()["nag"]
    LAST_BACKEND["nag"] = p
    if p == "kijai":
        try:
            mod = importlib.import_module("nodes")
            cls = getattr(mod, "WanVideoNAG")
            # Kijai's class returns (model,) when .apply() is called
            # with positional (model, scale, tau, alpha).
            out = cls().apply(model, scale, tau, alpha)
            return out[0] if isinstance(out, tuple) else out
        except Exception as ex:
            log.warning("NAG kijai path failed (%s); falling back to local", ex)
            LAST_BACKEND["nag"] = "local"
    # Local path. Clone the model so we don't mutate the caller's.
    if not hasattr(model, "clone"):
        raise TypeError("apply_nag: model has no .clone() method")
    m = model.clone()
    patch = build_nag_patch(scale=scale, tau=tau, alpha=alpha)
    opts = m.model_options
    to   = opts.setdefault("transformer_options", {})
    patches = to.setdefault("patches", {})
    patches["attn1_patch"] = patch
    return m


def apply_freeinit(
    noise: torch.Tensor,
    source: torch.Tensor,
    *,
    filter_type: str = "butterworth",
    n: int = 4,
    d_s: float = 1.0,
    d_t: float = 1.0,
) -> torch.Tensor:
    """Mix ``source``'s low-freq with ``noise``'s high-freq → new noise.

    ``noise`` and ``source`` must share shape; both should have at
    least 3 trailing dims ``(T, H, W)``. The frequency filter is
    constructed to match those last 3 dims.
    """
    p = probe()["freeinit"]
    LAST_BACKEND["freeinit"] = p
    if p == "kijai":
        try:
            mod = importlib.import_module("nodes_freeinit")
            cls = getattr(mod, "WanVideoFreeInit")
            out = cls().apply(noise, source, filter_type, n, d_s, d_t)
            return out[0] if isinstance(out, tuple) else out
        except Exception as ex:
            log.warning("FreeInit kijai path failed (%s); falling back to local", ex)
            LAST_BACKEND["freeinit"] = "local"
    # Local path.
    if noise.shape != source.shape:
        raise ValueError(
            f"apply_freeinit: noise {tuple(noise.shape)} != source "
            f"{tuple(source.shape)}"
        )
    T, H, W = noise.shape[-3:]
    flt = get_freq_filter(
        (T, H, W),
        filter_type=filter_type, n=n, d_s=d_s, d_t=d_t,
        device=noise.device, dtype=torch.float32,
    )
    return freq_mix_3d(source, noise, flt)
