"""Shared IS_CHANGED helpers — content hashes without float('nan')."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any


def _update_value(h: hashlib._Hash, v: Any) -> None:
    # O(1) per scalar; O(n) for tensor payload sampling.
    if v is None:
        return
    if hasattr(v, "detach"):
        try:
            import torch
            t = v.detach()
            if t.numel() > 65536:
                # Downsample large tensors for stable, cheap fingerprints.
                step = max(1, int((t.numel() / 8192) ** 0.5))
                t = t[..., ::step, ::step] if t.ndim >= 2 else t
            h.update(t.cpu().numpy().tobytes())
        except Exception:
            h.update(str(getattr(v, "shape", v)).encode())
        return
    if isinstance(v, (str, bytes)):
        h.update(v if isinstance(v, bytes) else v.encode())
        return
    h.update(str(v).encode())


def hash_kwargs(**kwargs: Any) -> str:
    """Stable MD5 hex digest of keyword arguments. O(n) over kwargs."""
    h = hashlib.md5()
    for k in sorted(kwargs):
        h.update(k.encode())
        _update_value(h, kwargs[k])
    return h.hexdigest()


def hash_args_and_kwargs(*args: Any, **kwargs: Any) -> str:
    """Stable MD5 hex digest of positional + keyword arguments."""
    h = hashlib.md5()
    for v in args:
        _update_value(h, v)
    for k in sorted(kwargs):
        h.update(k.encode())
        _update_value(h, kwargs[k])
    return h.hexdigest()


def dir_version_fingerprint(scan_dir: os.PathLike[str] | str, prefix: str, padding: int) -> str:
    """Filesystem fingerprint for versioned output dirs. O(n) over entries."""
    scan_path = Path(scan_dir)
    if not scan_path.is_dir():
        return "empty"
    parts: list[str] = []
    for entry in sorted(scan_path.iterdir(), key=lambda p: p.name):
        if entry.is_dir() and entry.name.startswith(prefix):
            try:
                parts.append(f"{entry.name}:{entry.stat().st_mtime_ns}")
            except OSError:
                parts.append(entry.name)
    return "|".join(parts) if parts else "empty"
