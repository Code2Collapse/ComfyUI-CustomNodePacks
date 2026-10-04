"""C2C memory guard — release pack-owned model caches when ComfyUI frees VRAM.

ComfyUI's ``free_memory`` only evicts ``ModelPatcher`` entries in
``current_loaded_models``. Our packs also hold raw ``nn.Module`` / pipeline
objects in module-level caches that are invisible to that path. This module
hooks ``free_memory`` once per process and registers cheap release callbacks
so preprocessing models (depth, pose, gaze, matting, segmentation) do not
squat VRAM while ComfyUI loads sampling models.

The registry lives on ``comfy.model_management`` so byte-identical copies of
this file in different packs share one process-global table without
cross-pack imports.

Opt out is not exposed: release is always safe — a model in use by a running
node survives through its local reference and is freed when that node returns.
"""
from __future__ import annotations

import gc
import logging
from typing import Callable

log = logging.getLogger("c2c.memguard")

_REGISTRY_ATTR = "_c2c_memguard_registry"
_INSTALLED_ATTR = "_c2c_memguard_installed"
_ORIG_ATTR = "_c2c_memguard_orig_free_memory"

ReleaseFn = Callable[[], None]
LoadedFn = Callable[[], bool]


def _mm():
    try:
        import comfy.model_management as mm
        return mm
    except Exception:
        return None


def _get_registry() -> dict[str, tuple[ReleaseFn, LoadedFn]] | None:
    mm = _mm()
    if mm is None:
        return None
    reg = getattr(mm, _REGISTRY_ATTR, None)
    if reg is None:
        reg = {}
        setattr(mm, _REGISTRY_ATTR, reg)
    return reg


def register(name: str, release: ReleaseFn, is_loaded: LoadedFn) -> None:
    """Register a named cache release. ``is_loaded`` must be cheap — it runs
    on every ``free_memory`` call for a GPU device."""
    reg = _get_registry()
    if reg is None:
        return
    reg[name] = (release, is_loaded)
    install()


def release_all(reason: str) -> int:
    """Release every cache that currently holds a model. Returns how many
    ``release`` callbacks actually ran."""
    reg = _get_registry()
    if not reg:
        return 0

    released = 0
    for name, (release, is_loaded) in list(reg.items()):
        loaded = True
        try:
            loaded = bool(is_loaded())
        except Exception as exc:
            log.debug("[c2c.memguard] is_loaded(%s) raised: %s", name, exc)
        if not loaded:
            continue
        try:
            release()
            released += 1
        except Exception as exc:
            log.debug("[c2c.memguard] release(%s) failed: %s", name, exc)

    if released > 0:
        gc.collect()
        mm = _mm()
        if mm is not None:
            soft = getattr(mm, "soft_empty_cache", None)
            if callable(soft):
                try:
                    soft()
                except Exception as exc:
                    log.debug("[c2c.memguard] soft_empty_cache failed: %s", exc)

    if released > 0:
        log.debug("[c2c.memguard] released %d caches (%s)", released, reason)
    return released


def _is_gpu_device(device) -> bool:
    if device is None:
        return False
    dev_type = getattr(device, "type", None)
    if dev_type is not None:
        return dev_type != "cpu"
    return str(device).lower() not in ("cpu", "meta")


def install() -> bool:
    """Wrap ``comfy.model_management.free_memory``. Idempotent."""
    mm = _mm()
    if mm is None or getattr(mm, _INSTALLED_ATTR, False):
        return False
    orig = getattr(mm, "free_memory", None)
    if not callable(orig):
        return False

    def _wrapped_free_memory(memory_required, device, *args, **kwargs):
        reg = _get_registry()
        if reg and _is_gpu_device(device):
            release_all("ComfyUI is loading or freeing models")
        return orig(memory_required, device, *args, **kwargs)

    setattr(mm, _ORIG_ATTR, orig)
    mm.free_memory = _wrapped_free_memory
    setattr(mm, _INSTALLED_ATTR, True)
    log.debug("[c2c.memguard] installed free_memory wrapper")
    return True
