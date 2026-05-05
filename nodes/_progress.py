# -*- coding: utf-8 -*-
"""Combined ComfyUI ProgressBar + tqdm + interrupt-poll wrapper.

Usage::

    from . import _progress as _PB

    for x in _PB.track(iterable, total, "MyNode"):
        ...

* Drives the green progress fill on the node UI via ``comfy.utils.ProgressBar``.
* Prints a terminal progress line with ETA via ``tqdm``.
* Raises ``InterruptProcessingException`` when the user clicks Stop.

All three integrations degrade gracefully when ComfyUI / tqdm are absent
(e.g. unit tests), so the helper is safe to import from any module.
"""
from __future__ import annotations

from typing import Iterable, Iterator, Optional, TypeVar

T = TypeVar("T")

try:
    from comfy.utils import ProgressBar as _ComfyPB  # type: ignore
except Exception:  # noqa: BLE001
    _ComfyPB = None  # type: ignore

try:
    from comfy.model_management import (  # type: ignore
        throw_exception_if_processing_interrupted as _throw,
    )
except Exception:  # noqa: BLE001
    def _throw() -> None:  # type: ignore[no-redef]
        return None

try:
    from tqdm import tqdm as _tqdm  # type: ignore
except Exception:  # noqa: BLE001
    _tqdm = None  # type: ignore


def track(
    iterable: Iterable[T],
    total: Optional[int] = None,
    desc: str = "",
) -> Iterator[T]:
    """Yield items from ``iterable`` while updating progress + interrupt poll."""
    if total is None:
        try:
            total = len(iterable)  # type: ignore[arg-type]
        except Exception:
            total = None

    pbar = None
    if _ComfyPB is not None and total:
        try:
            pbar = _ComfyPB(int(total))
        except Exception:
            pbar = None

    if _tqdm is not None:
        it = _tqdm(iterable, total=total, desc=desc or None, leave=False)
    else:
        it = iterable

    i = 0
    try:
        for item in it:
            _throw()
            yield item
            i += 1
            if pbar is not None:
                try:
                    pbar.update_absolute(i)
                except Exception:
                    pass
    finally:
        if _tqdm is not None and hasattr(it, "close"):
            try:
                it.close()  # type: ignore[union-attr]
            except Exception:
                pass
