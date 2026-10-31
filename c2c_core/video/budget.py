"""RAM budget helpers for chunked video decode."""

from __future__ import annotations

import numpy as np


class BudgetError(Exception):
    """Raised when a requested frame batch exceeds available RAM."""


def bytes_per_frame(width: int, height: int, channels: int, dtype: np.dtype | str) -> int:
    dt = np.dtype(dtype)
    return int(width) * int(height) * int(channels) * int(dt.itemsize)


def frames_that_fit(
    width: int,
    height: int,
    channels: int,
    dtype: np.dtype | str,
    *,
    fraction: float = 0.25,
    available: int | None = None,
) -> int:
    if available is None:
        import psutil

        available = int(psutil.virtual_memory().available)
    bpf = bytes_per_frame(width, height, channels, dtype)
    if bpf <= 0:
        return 0
    budget = int(available * float(fraction))
    return max(0, budget // bpf)


def budget_error_message(
    *,
    asked: int,
    fits: int,
    width: int,
    height: int,
    channels: int,
    dtype: str,
) -> str:
    bpf = bytes_per_frame(width, height, channels, dtype)
    need = asked * bpf
    have = fits * bpf
    return (
        f"You asked for {asked} frames ({width}x{height}x{channels} {dtype}, "
        f"about {need / (1024 ** 2):.1f} MB), but only about {fits} frames "
        f"({have / (1024 ** 2):.1f} MB) fit in the allowed RAM budget. "
        f"Lower the frame count or use iter_chunks() to process the clip in smaller batches."
    )
