"""OCIO colour transforms for Save Video (C2C)."""

from __future__ import annotations

import os
from typing import Any

import numpy as np

_FALLBACK_SPACES = ("sRGB - Display", "Linear Rec.709 (sRGB)", "ACEScg")

_OCIO = None
_OCIO_SPACES: list[str] | None = None
_OCIO_ERR: str | None = None


def _piecewise_srgb_to_linear(x: np.ndarray) -> np.ndarray:
    x = np.clip(x.astype(np.float32), 0.0, 1.0)
    low = x / 12.92
    high = ((x + 0.055) / 1.055) ** 2.4
    return np.where(x <= 0.04045, low, high).astype(np.float32)


def _piecewise_linear_to_srgb(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    low = x * 12.92
    high = 1.055 * np.clip(x, 0.0, None) ** (1.0 / 2.4) - 0.055
    return np.where(x <= 0.0031308, low, high).astype(np.float32)


_CFG = None
_PROCS: dict = {}


def _space_names(cfg) -> list:
    """Colour-space names on OCIO 2.x (getNumColorSpaces / ColorProcessor are OCIO 1 API and do not exist in 2.5)."""
    try:
        return list(cfg.getColorSpaceNames())
    except Exception:
        return [cs.getName() for cs in cfg.getColorSpaces()]


def _load_ocio():
    global _OCIO, _OCIO_SPACES, _OCIO_ERR, _CFG
    if _OCIO is not None or _OCIO_ERR is not None:
        return _OCIO
    try:
        import PyOpenColorIO as ocio  # type: ignore[import-not-found]
        path = os.environ.get("OCIO", "").strip()
        if path:
            cfg = ocio.Config.CreateFromFile(path)
        else:
            # Without $OCIO, CreateFromEnv gives the one-space "raw" config: use the built-in ACES studio config.
            cfg = ocio.Config.CreateFromEnv()
            if len(_space_names(cfg)) <= 1:
                cfg = ocio.Config.CreateFromFile("ocio://studio-config-latest")
        _CFG = cfg
        _OCIO = ocio
        _OCIO_SPACES = sorted(set(_space_names(cfg)))
        return _OCIO
    except Exception as exc:
        _OCIO_ERR = str(exc)
        _OCIO = False
        return False


def _cpu_processor(src: str, dst: str):
    key = (src, dst)
    proc = _PROCS.get(key)
    if proc is None:
        proc = _PROCS[key] = _CFG.getProcessor(src, dst).getDefaultCPUProcessor()
    return proc


def colorspace_choices() -> list[str]:
    _load_ocio()
    if _OCIO_SPACES:
        return list(_OCIO_SPACES)
    return list(_FALLBACK_SPACES)


def default_colorspace_in() -> str:
    choices = colorspace_choices()
    for cand in ("sRGB - Display", "sRGB", "Output - sRGB"):
        if cand in choices:
            return cand
    return choices[0]


def default_colorspace_out_exr() -> str:
    choices = colorspace_choices()
    for cand in ("ACEScg", "Linear Rec.709 (sRGB)", "scene_linear", "Linear"):
        if cand in choices:
            return cand
    return choices[-1]


def _same_space(a: str, b: str) -> bool:
    return (a or "").strip() == (b or "").strip() or b in ("same as input", "same", "")


def transform_chunk(
    chunk: np.ndarray,
    colorspace_in: str,
    colorspace_out: str,
) -> np.ndarray:
    """RGB float32 [N,H,W,3] in → out. Alpha channel unchanged if present."""
    if chunk.shape[-1] >= 4:
        rgb = chunk[..., :3]
        alpha = chunk[..., 3:4]
    else:
        rgb = chunk[..., :3]
        alpha = None

    if _same_space(colorspace_in, colorspace_out):
        out_rgb = rgb.astype(np.float32, copy=False)
    elif _load_ocio() and _OCIO_SPACES:
        if colorspace_in not in _OCIO_SPACES or (
            not _same_space(colorspace_out, colorspace_in) and colorspace_out not in _OCIO_SPACES
        ):
            raise ValueError(
                f"Colour space not in OCIO config: {colorspace_in!r} → {colorspace_out!r}. "
                "Install PyOpenColorIO and set $OCIO, or use sRGB / Linear only."
            )
        flat = np.ascontiguousarray(rgb.astype(np.float32).reshape(-1, 3))
        _cpu_processor(colorspace_in, colorspace_out).applyRGB(flat)   # in place, float32
        out_rgb = flat.reshape(rgb.shape)
    else:
        out_rgb = _fallback_transform(rgb, colorspace_in, colorspace_out)

    if alpha is not None:
        return np.concatenate([out_rgb, alpha], axis=-1)
    return out_rgb


def _fallback_transform(rgb: np.ndarray, src: str, dst: str) -> np.ndarray:
    s = (src or "").lower()
    d = (dst or "").lower()
    linear_names = ("linear rec.709", "linear", "acescg", "scene_linear")
    srgb_names = ("srgb", "display")

    def is_linear(name: str) -> bool:
        return any(k in name for k in linear_names)

    def is_srgb(name: str) -> bool:
        return any(k in name for k in srgb_names)

    if is_srgb(s) and is_linear(d):
        return _piecewise_srgb_to_linear(rgb)
    if is_linear(s) and is_srgb(d):
        return _piecewise_linear_to_srgb(rgb)
    if s == d or not dst:
        return rgb.astype(np.float32, copy=False)
    raise ValueError(
        f"PyOpenColorIO is not available ({_OCIO_ERR or 'not installed'}). "
        f"Only sRGB ↔ Linear transforms are supported without it; got {src!r} → {dst!r}."
    )


def ocio_available() -> bool:
    return bool(_load_ocio() and _OCIO_SPACES)
