"""Stage cache for viewer-driven Mask + Matting runs (L7.114, owner answer 2026-10-15: "inside the node + stage cache").

A Mask + Matting run is three stages: TRACK (SAM / the segmenter finds the subject on every frame), EDGE (trimap +
matter: ViTMatte, guided filter, ...), FINISH (holdout / core, edge mode, despill, light wrap, split masks,
diagnostics). ComfyUI caches the node's outputs as a whole, so changing despill re-ran SAM. With the viewer's
selection set (`picks`), the first two stages are kept here, keyed by what they read:

    track key = (stage, plate fingerprint, selection, segmenter settings, which weights are installed)
    edge key  = (track key, matter settings, trimap settings, which weights are installed)

so a late setting reruns only the late stages (tests/test_matte_stage_cache.py counts the calls).

Bounded in BYTES, not entries: least recently used entries go first, an entry bigger than the whole budget is not
kept. Budget: env C2C_MATTE_CACHE_MB (default 1024 MB; 0 turns the cache off). Tensors are kept on the CPU (never
VRAM). The plate fingerprint is SHA-1 over every byte of the plate (1.6 GB/s measured on this machine,
2026-10-15, 30 x 1080p float32 frames): a sampled hash could miss a small upstream edit and hand back a stale matte.
Plain runs (picks empty) never touch this module.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from collections import OrderedDict
from typing import Any, Dict, Hashable, Optional, Tuple

logger = logging.getLogger("MEC.MaskOps.StageCache")

DEFAULT_MB = 1024
ENV = "C2C_MATTE_CACHE_MB"


def budget_bytes() -> int:
    raw = os.environ.get(ENV, "").strip()
    if not raw:
        return DEFAULT_MB * 1024 * 1024
    try:
        mb = float(raw)
    except ValueError:
        logger.warning("[MaskOps] %s=%r is not a number - using %d MB", ENV, raw, DEFAULT_MB)
        return DEFAULT_MB * 1024 * 1024
    return max(0, int(mb * 1024 * 1024))


def nbytes(value: Any) -> int:
    """Bytes held by the tensors / arrays inside ``value`` (dicts, lists, tuples are walked)."""
    try:
        import torch
    except Exception:  # pragma: no cover
        torch = None  # type: ignore
    if torch is not None and isinstance(value, torch.Tensor):
        return int(value.numel()) * int(value.element_size())
    if hasattr(value, "nbytes") and not isinstance(value, (str, bytes)):
        try:
            return int(value.nbytes)
        except Exception:  # noqa: BLE001
            return 0
    if isinstance(value, dict):
        return sum(nbytes(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(nbytes(v) for v in value)
    return 0


def to_cpu(value: Any) -> Any:
    """The same structure with every tensor detached and on the CPU (the cache never holds VRAM)."""
    try:
        import torch
    except Exception:  # pragma: no cover
        return value
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: to_cpu(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_cpu(v) for v in value]
    if isinstance(value, tuple):
        return tuple(to_cpu(v) for v in value)
    return value


def fingerprint(t: Any) -> str:
    """Exact content fingerprint of a tensor (shape, dtype, every byte); "none" for None."""
    if t is None:
        return "none"
    try:
        import torch
    except Exception:  # pragma: no cover
        return f"obj:{id(t)}"
    if not isinstance(t, torch.Tensor):
        return f"{type(t).__name__}:{json.dumps(t, sort_keys=True, default=str)}"
    a = t.detach()
    if a.device.type != "cpu":
        a = a.cpu()
    a = a.contiguous()
    h = hashlib.sha1()
    h.update(f"{tuple(a.shape)}|{a.dtype}|".encode())
    try:
        h.update(memoryview(a.numpy()).cast("B"))
    except Exception:  # noqa: BLE001 - bf16 and friends have no numpy view
        h.update(memoryview(a.view(torch.uint8).numpy()).cast("B") if a.numel() else b"")
    return h.hexdigest()


def value_key(v: Any) -> Any:
    """One input as a key part: tensors by content, containers as JSON, objects by type and identity."""
    try:
        import torch
        if isinstance(v, torch.Tensor):
            return fingerprint(v)
    except Exception:  # pragma: no cover
        pass
    if isinstance(v, (str, int, float, bool, type(None))):
        return v
    if isinstance(v, (list, tuple, dict)):
        try:
            return json.dumps(v, sort_keys=True)
        except (TypeError, ValueError):
            return repr(v)
    return f"{type(v).__name__}:{id(v)}"


def make_key(*parts: Any) -> Tuple[Hashable, ...]:
    """A hashable key from plain values (dicts / lists are frozen as sorted JSON)."""
    out = []
    for p in parts:
        if isinstance(p, (dict, list)):
            out.append(json.dumps(p, sort_keys=True, default=str))
        else:
            out.append(p if isinstance(p, (str, int, float, bool, type(None), tuple)) else str(p))
    return tuple(out)


class StageCache:
    """Least-recently-used cache bounded by the bytes of the tensors it holds."""

    def __init__(self, budget: int):
        self.budget = int(budget)
        self._data: "OrderedDict[Hashable, Tuple[Any, int]]" = OrderedDict()
        self._bytes = 0
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    @property
    def bytes(self) -> int:
        return self._bytes

    def __len__(self) -> int:
        return len(self._data)

    def get(self, key: Hashable) -> Optional[Any]:
        if key is None:
            return None
        with self._lock:
            hit = self._data.get(key)
            if hit is None:
                self.misses += 1
                return None
            self._data.move_to_end(key)
            self.hits += 1
            return hit[0]

    def put(self, key: Hashable, value: Any) -> bool:
        """Keep ``value`` (moved to the CPU). False when it is bigger than the whole budget."""
        if key is None or self.budget <= 0:
            return False
        value = to_cpu(value)
        size = nbytes(value)
        if size > self.budget:
            logger.info("[MaskOps] stage result of %.1f MB is bigger than the %.1f MB cache - not kept",
                        size / 2**20, self.budget / 2**20)
            return False
        with self._lock:
            old = self._data.pop(key, None)
            if old is not None:
                self._bytes -= old[1]
            self._data[key] = (value, size)
            self._bytes += size
            self._evict()
        return True

    def _evict(self) -> None:
        while self._bytes > self.budget and self._data:
            _k, (_v, s) = self._data.popitem(last=False)
            self._bytes -= s

    def set_budget(self, budget: int) -> None:
        with self._lock:
            self.budget = int(budget)
            self._evict()

    def clear(self) -> None:
        with self._lock:
            self._data.clear()
            self._bytes = 0

    def stats(self) -> Dict[str, Any]:
        return {"entries": len(self._data), "mb": round(self._bytes / 2**20, 1),
                "budget_mb": round(self.budget / 2**20, 1), "hits": self.hits, "misses": self.misses}


_CACHE: Optional[StageCache] = None


def get_cache() -> Optional[StageCache]:
    """The process-wide cache, or None when C2C_MATTE_CACHE_MB=0. The budget follows the env on every call."""
    global _CACHE
    b = budget_bytes()
    if b <= 0:
        if _CACHE is not None:
            _CACHE.clear()
        return None
    if _CACHE is None:
        _CACHE = StageCache(b)
    elif _CACHE.budget != b:
        _CACHE.set_budget(b)
    return _CACHE


def installed(key: str) -> Tuple[str, ...]:
    """Which weight files a backend folder holds now: part of every key, so placing a model invalidates the
    model-free result that ran without it."""
    if not key:
        return ()
    try:
        from .utils import list_backend_files
        return tuple(sorted(list_backend_files(key)))
    except Exception:  # noqa: BLE001
        return ()
