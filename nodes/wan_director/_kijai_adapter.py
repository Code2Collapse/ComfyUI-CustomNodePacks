"""Kijai-compat adapter layer for the Wan Director.

Probes ``third_party/ComfyUI-WanVideoWrapper`` at import time for each
supported Kijai feature; when Kijai is present AND the feature's API
signature matches what we wrap, the adapter routes to Kijai (faster,
maintained upstream). Otherwise it routes to our in-tree
re-implementation in :mod:`..features._local_<feature>`. Every feature
has a local implementation — the adapter never raises on missing kijai.

Public surface (stable; Director relies on these signatures):

    probe()                           → dict[str, str]
        Maps feature name → ``"kijai"`` or ``"local"``.

    capability_report()               → str
        Human-readable per-feature summary for the Director's ``info``
        output.

    apply_nag(model, scale, tau, alpha) → MODEL
    apply_freeinit(noise, source, ...) → Tensor
    make_cache(kind, **kwargs)         → TeaCache | MagCache | EasyCache | None
    build_slg(...) / apply_slg(...)
    apply_feta(attn_logits, ...) → Tensor
    apply_riflex(rope_freqs, ...) → Tensor
    apply_uni3c(extrinsics, ...) → Tensor
    apply_context_windows(...) / splice_context_windows(...)
    apply_latent_preview(latent) → RGB Tensor
    apply_asymflow(model, shift, multiplier) → MODEL

Each apply function records the backend it used in the module-level
``LAST_BACKEND`` dict so the Director can include it in info output.
"""
from __future__ import annotations

import importlib
import logging
from typing import Any

import torch

from .features._local_nag       import build_nag_patch, apply_nag_to_attn_output
from .features._local_freeinit  import freq_mix_3d, get_freq_filter
from .features._local_teacache  import TeaCache
from .features._local_magcache  import MagCache
from .features._local_easycache import EasyCache
from .features._local_slg       import SLGConfig, should_apply_slg, \
                                       make_layer_skip_predicate, combine_slg
from .features._local_feta            import feta_attention_bias
from .features._local_riflex          import rescale_rope_freqs, \
                                             riflex_extrapolation_active
from .features._local_uni3c           import encode_camera_poses
from .features._local_context_windows import plan_windows, \
                                             blend_window_weights, \
                                             splice_windows
from .features._local_latent_preview  import latent_to_rgb_preview, \
                                             supported_channel_counts, \
                                             latent_model_for_channels

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
    "slg", "feta", "riflex", "uni3c", "context_windows", "latent_preview",
    "asymflow",
)


def probe() -> dict[str, str]:
    """Return a {feature: backend} dict.

    Backends: ``"kijai"`` (kijai present and its API for that feature
    matches our wrapper signature) or ``"local"`` (route to our
    in-tree re-implementation in :mod:`..features._local_<feature>`).
    Every feature in :data:`_FEATURES` has a local implementation, so
    the value is always one of those two — never ``"unavailable"``.
    """
    out: dict[str, str] = {}
    if _KIJAI is None:
        return {f: "local" for f in _FEATURES}
    out["nag"]       = "kijai" if _has("nodes",          "WanVideoNAG"            ) else "local"
    out["freeinit"]  = "kijai" if _has("nodes_freeinit", "WanVideoFreeInit"       ) else "local"
    out["teacache"]  = "kijai" if _has("nodes",          "WanVideoTeaCache"       ) else "local"
    out["magcache"]  = "kijai" if _has("nodes",          "WanVideoMagCache"       ) else "local"
    out["easycache"] = "kijai" if _has("nodes",          "WanVideoEasyCache"      ) else "local"
    out["slg"]       = "kijai" if _has("nodes",          "WanVideoSLG"            ) else "local"
    out["feta"]      = "kijai" if _has("nodes",          "WanVideoFETA"           ) else "local"
    out["riflex"]    = "kijai" if _has("nodes",          "WanVideoRIFLEx"         ) else "local"
    out["uni3c"]     = "kijai" if _has("nodes",          "WanVideoUni3CController") else "local"
    out["context_windows"] = "kijai" if _has("nodes",    "WanVideoContextWindows" ) else "local"
    out["latent_preview"]  = "kijai" if _has("nodes",    "WanVideoTAEHVPreview"   ) else "local"
    out["asymflow"]        = "local"  # Code2Collapse-only contribution
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
    # Use ComfyUI's proper API to install attn1 patch.
    try:
        m.set_model_attn1_patch(patch)
    except AttributeError:
        # Fallback: inject into model_options directly.
        opts = m.model_options
        to   = opts.setdefault("transformer_options", {})
        patches_d = to.setdefault("patches", {})
        patches_d["attn1_patch"] = patch
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


# ── Caches (TeaCache / MagCache / EasyCache) ──────────────────────────


def make_cache(kind: str, **kwargs):
    """Build a cache gate of the requested kind.

    Args:
        kind: one of ``"teacache"``, ``"magcache"``, ``"easycache"``,
            or ``"none"`` (returns ``None`` — caller should bypass).
        **kwargs: forwarded to the cache class constructor; unknown
            keys are silently ignored so callers can pass a single
            superset dict.

    Returns:
        A cache instance (or ``None`` for ``"none"``). Each instance
        has ``should_skip``, ``record``, ``cached_residual``, ``reset``,
        and ``report`` per the local cache classes.
    """
    k = (kind or "none").lower()
    LAST_BACKEND[k] = probe().get(k, "local") if k != "none" else "none"
    if k == "none":
        return None
    if k == "teacache":
        return TeaCache(
            rel_l1_thresh=float(kwargs.get("rel_l1_thresh", 0.10)),
            max_skips=int(kwargs.get("max_skips", 5)),
        )
    if k == "magcache":
        return MagCache(
            mag_thresh=float(kwargs.get("mag_thresh", 1.2)),
            ema_alpha=float(kwargs.get("ema_alpha", 0.3)),
            max_skips=int(kwargs.get("max_skips", 5)),
        )
    if k == "easycache":
        return EasyCache(
            l1_thresh=float(kwargs.get("l1_thresh", 0.02)),
            max_skips=int(kwargs.get("max_skips", 4)),
        )
    raise ValueError(f"make_cache: unknown kind {kind!r}")


# ── SLG (Skip Layer Guidance) ─────────────────────────────────────────


def build_slg(
    skip_layers: list[int] | tuple[int, ...] = (),
    *,
    slg_scale: float = 0.7,
    start_pct: float = 0.0,
    end_pct:   float = 1.0,
) -> SLGConfig:
    """Build a validated SLGConfig; records backend in LAST_BACKEND."""
    LAST_BACKEND["slg"] = probe().get("slg", "local")
    return SLGConfig(
        skip_layers=tuple(skip_layers),
        slg_scale=slg_scale,
        start_pct=start_pct,
        end_pct=end_pct,
    )


def apply_slg(
    eps_pos:  torch.Tensor,
    eps_skip: torch.Tensor,
    cfg: SLGConfig,
    *,
    step: int = 0,
    n_steps: int = 1,
) -> torch.Tensor:
    """Combine a full + skip-layer prediction into the SLG-guided eps.

    If the current ``step`` falls outside the cfg's step window
    (``start_pct``..``end_pct``) or ``skip_layers`` is empty, the
    function returns ``eps_pos`` unchanged.
    """
    if not should_apply_slg(step, n_steps, cfg):
        return eps_pos
    return combine_slg(eps_pos, eps_skip, cfg.slg_scale)


# ── FETA / RIFLEx / Uni3C / Context Windows / Latent preview ──────────


def apply_feta(
    attn_logits: torch.Tensor,
    *,
    feta_scale:     float = 0.5,
    freq_center:    float = 0.20,
    freq_bandwidth: float = 0.15,
) -> torch.Tensor:
    """Apply FETA frequency-domain bias to attention logits."""
    LAST_BACKEND["feta"] = probe().get("feta", "local")
    return feta_attention_bias(
        attn_logits,
        feta_scale=feta_scale,
        freq_center=freq_center,
        freq_bandwidth=freq_bandwidth,
    )


def apply_riflex(
    rope_freqs: torch.Tensor,
    *,
    source_len: int,
    target_len: int,
    k: int = 2,
) -> torch.Tensor:
    """Rescale the lowest-k RoPE freqs for length extrapolation.

    If ``target_len`` is not greater than ``source_len`` this returns
    a clone unchanged.
    """
    LAST_BACKEND["riflex"] = probe().get("riflex", "local")
    if not riflex_extrapolation_active(target_len, source_len):
        return rope_freqs.clone()
    return rescale_rope_freqs(
        rope_freqs, source_len=source_len, target_len=target_len, k=k,
    )


def apply_uni3c(
    extrinsics: torch.Tensor,
    *,
    embed_dim: int | None = None,
) -> torch.Tensor:
    """Encode camera extrinsics into a per-frame embedding sequence."""
    LAST_BACKEND["uni3c"] = probe().get("uni3c", "local")
    return encode_camera_poses(extrinsics, embed_dim=embed_dim)


def apply_context_windows(
    n_frames: int,
    *,
    window: int,
    overlap: int,
) -> list:
    """Plan context windows for long-video sampling.

    Returns a list of ``range`` objects. Use :func:`splice_context_windows`
    to blend per-window outputs back together.
    """
    LAST_BACKEND["context_windows"] = probe().get("context_windows", "local")
    return plan_windows(n_frames, window=window, overlap=overlap)


def splice_context_windows(
    window_outputs,
    plan,
    full_len: int,
    *,
    overlap: int = 0,
) -> torch.Tensor:
    """Blend per-window outputs into a single ``(full_len, ...)`` tensor."""
    return splice_windows(window_outputs, plan, full_len, overlap=overlap)


def apply_latent_preview(latent: torch.Tensor) -> torch.Tensor:
    """Render an RGB preview from a Wan latent tensor.

    Uses ComfyUI's published Wan21 / Wan22 ``latent_rgb_factors`` —
    auto-routed by channel count (16 → Wan2.1, 48 → Wan2.2). The
    output matches ComfyUI's built-in fast preview pixel for pixel.
    """
    LAST_BACKEND["latent_preview"] = probe().get("latent_preview", "local")
    return latent_to_rgb_preview(latent)


# ── AsymFlow (Lakonik signal-shift) ───────────────────────────────────


def asymflow_time_shift(shift: float, t):
    """Pure-tensor AsymFlow time→sigma mapping.

    Re-exports the pure function from ``nodes.asymflow_sampler`` so
    the Director can use it without importing the ComfyUI runtime
    side of the node module.
    """
    # Late import to keep the adapter import-light and avoid pulling
    # in ``comfy`` when this module is used in unit tests.
    from ..asymflow_sampler import asymflow_time_shift as _f
    return _f(shift, t)


def apply_asymflow(model, *, shift: float = 3.0, multiplier: int = 1000):
    """Install the AsymFlow shifted-flow schedule on a model.

    Wraps :class:`nodes.asymflow_sampler.AsymFlowSamplerPatch` so the
    Director sees a single interface. AsymFlow is a Code2Collapse
    contribution (not vendored kijai), so this is always "local".
    """
    LAST_BACKEND["asymflow"] = "local"
    from ..asymflow_sampler import AsymFlowSamplerPatch
    out = AsymFlowSamplerPatch().patch(model, shift, multiplier)
    return out[0] if isinstance(out, tuple) else out


# Re-exported so the Director / sampler wrapper can use the predicate
# inside its transformer-block iteration without re-importing.
__all__ = [
    "probe", "capability_report", "LAST_BACKEND",
    "apply_nag", "apply_freeinit",
    "make_cache",
    "build_slg", "apply_slg", "make_layer_skip_predicate",
    "apply_feta", "apply_riflex", "apply_uni3c",
    "apply_context_windows", "splice_context_windows",
    "apply_latent_preview",
    "apply_asymflow", "asymflow_time_shift",
    "supported_channel_counts", "latent_model_for_channels",
]
