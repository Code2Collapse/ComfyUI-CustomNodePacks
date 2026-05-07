# -*- coding: utf-8 -*-
"""Combined ComfyUI ProgressBar + tqdm + interrupt-poll wrapper.

Two usage modes::

    # Simple (legacy) — each track() call drives its own 0→100% bar.
    from . import _progress as _PB
    for x in _PB.track(iterable, total, "MyNode"):
        ...

    # Session (preferred) — ONE 0→100% bar spans the whole node call.
    # Nested track() calls are pass-through (interrupt poll only). The bar
    # NEVER resets between phases — it only marches forward.
    with _PB.session("MyNode"):
        for x in _PB.track(it_a, total_a, "phase 1"):
            ...
        for x in _PB.track(it_b, total_b, "phase 2"):
            ...

* Drives the green progress fill on the node UI via ``comfy.utils.ProgressBar``.
* Prints a single terminal tqdm bar (its description updates per phase).
* Raises ``InterruptProcessingException`` when the user clicks Stop.

All three integrations degrade gracefully when ComfyUI / tqdm are absent
(e.g. unit tests), so the helper is safe to import from any module.
"""
from __future__ import annotations

import contextlib
import functools
import threading
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


# ══════════════════════════════════════════════════════════════════════
#  Session — single 0..100% bar across all phases of a node execution.
# ══════════════════════════════════════════════════════════════════════

_local = threading.local()


def _active_session() -> Optional["_Session"]:
    return getattr(_local, "session", None)


class _Session:
    """One ProgressBar(100) + one tqdm(total=100) shared across phases.

    Phase allocation is *adaptive*: each top-level ``track()`` claims half
    of the remaining bar (50%, 25%, 12.5%, ...). This guarantees the bar
    only ever moves FORWARD and never overshoots 100%, regardless of how
    many phases the node ends up running. ``close()`` snaps the bar to
    100% on exit so the user sees a clean finish.
    """

    def __init__(self, name: str):
        self.name = name
        self._consumed = 0.0           # in [0, 100]
        self._depth = 0                 # nested-track counter
        self._pbar = None
        self._tqdm = None
        if _ComfyPB is not None:
            try:
                self._pbar = _ComfyPB(100)
            except Exception:  # noqa: BLE001
                self._pbar = None
        if _tqdm is not None:
            try:
                # Single tqdm bar per node call.  ``leave=False`` keeps the
                # terminal clean; ``ncols=100`` stops bars from wrapping on
                # wide terminals; ``dynamic_ncols=False`` prevents resize
                # flicker mid-run.
                self._tqdm = _tqdm(
                    total=100, desc=name, leave=False, unit="%",
                    ncols=100, dynamic_ncols=False,
                )
            except Exception:  # noqa: BLE001
                self._tqdm = None

    # ----- internal helpers -----

    def _push_ui(self, value: float) -> None:
        v = max(0.0, min(100.0, value))
        if self._pbar is not None:
            try:
                self._pbar.update_absolute(int(round(v)), 100)
            except Exception:  # noqa: BLE001
                pass
        if self._tqdm is not None:
            try:
                # Round to integer so tqdm shows "97/100" not "96.9328…".
                v_int = int(round(v))
                delta = v_int - int(round(self._tqdm.n))
                if delta > 0:
                    self._tqdm.update(delta)
            except Exception:  # noqa: BLE001
                pass

    def _set_desc(self, desc: str) -> None:
        if self._tqdm is not None and desc:
            try:
                # Avoid double-prefix: if the caller already included the
                # session name (e.g. "InpaintCropProMEC: per-frame crop"),
                # don't prepend it again.
                if desc.startswith(self.name):
                    label = desc
                else:
                    label = f"{self.name}: {desc}"
                self._tqdm.set_description(label)
            except Exception:  # noqa: BLE001
                pass

    def _phase_weight(self) -> float:
        """Adaptive allocation: half the remaining bar (capped 50%, floor 1%).

        Asymptotic: 50, 25, 12.5, 6.25 ... — never overshoots, always
        progresses by at least 1% so the user sees motion every phase.
        """
        remaining = max(0.0, 100.0 - self._consumed)
        return max(1.0, min(50.0, remaining * 0.5))

    # ----- public iterator -----

    def track(self, iterable: Iterable[T], total: Optional[int],
              desc: str) -> Iterator[T]:
        self._depth += 1
        try:
            if self._depth > 1:
                # Nested call: pass items through with interrupt poll only.
                # Do NOT touch the progress bar (would reset / double-count).
                for item in iterable:
                    _throw()
                    yield item
                return

            # Top-level phase.
            if total is None:
                try:
                    total = len(iterable)  # type: ignore[arg-type]
                except Exception:  # noqa: BLE001
                    total = None

            weight = self._phase_weight()
            base = self._consumed
            self._set_desc(desc)

            if not total or int(total) <= 0:
                # Unknown / empty: just iterate, advance once at the end.
                for item in iterable:
                    _throw()
                    yield item
                self._consumed = base + weight
                self._push_ui(self._consumed)
                return

            tot = int(total)
            n = 0
            for item in iterable:
                _throw()
                yield item
                n += 1
                self._consumed = base + weight * (n / tot)
                self._push_ui(self._consumed)
            # Snap to end of phase even if iteration short-circuited.
            self._consumed = base + weight
            self._push_ui(self._consumed)
        finally:
            self._depth -= 1

    def close(self) -> None:
        # Force final 100% so the user sees a clean finish.
        self._consumed = 100.0
        self._push_ui(100.0)
        if self._tqdm is not None:
            try:
                self._tqdm.close()
            except Exception:  # noqa: BLE001
                pass
            self._tqdm = None


@contextlib.contextmanager
def session(name: str):
    """Open a node-wide progress session.

    All ``track()`` calls inside the ``with`` block share a single 0..100%
    bar (UI + terminal). Nested ``track()`` calls are pass-through. Safe
    to nest sessions (the inner one becomes a no-op view of the outer).
    """
    prev = _active_session()
    if prev is not None:
        # Already inside a session: don't create a new one. Just yield the
        # outer so callers can ``with _PB.session(...) as s:`` uniformly.
        yield prev
        return
    s = _Session(name)
    _local.session = s
    try:
        yield s
    finally:
        try:
            s.close()
        finally:
            _local.session = prev


# ══════════════════════════════════════════════════════════════════════
#  track() — session-aware. Falls back to per-call bar when no session.
# ══════════════════════════════════════════════════════════════════════

def with_session(name: str):
    """Decorator: wrap a node method so its whole call shares one bar.

    Usage::

        @_PB.with_session("BgRemover")
        def remove_bg(self, ...):
            for i in _PB.track(range(B), B, "phase 1"):
                ...
    """
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            with session(name):
                return fn(*args, **kwargs)
        return wrapper
    return deco

def track(
    iterable: Iterable[T],
    total: Optional[int] = None,
    desc: str = "",
) -> Iterator[T]:
    """Yield items from ``iterable`` while updating progress + interrupt poll.

    If a ``_PB.session(...)`` is open on the current thread, this delegates
    to the session (one bar across all phases). Otherwise it falls back to
    the legacy per-call bar.
    """
    s = _active_session()
    if s is not None:
        return s.track(iterable, total, desc)
    return _legacy_track(iterable, total, desc)


def _legacy_track(
    iterable: Iterable[T],
    total: Optional[int],
    desc: str,
) -> Iterator[T]:
    if total is None:
        try:
            total = len(iterable)  # type: ignore[arg-type]
        except Exception:  # noqa: BLE001
            total = None

    pbar = None
    if _ComfyPB is not None and total:
        try:
            pbar = _ComfyPB(int(total))
        except Exception:  # noqa: BLE001
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
                except Exception:  # noqa: BLE001
                    pass
    finally:
        if _tqdm is not None and hasattr(it, "close"):
            try:
                it.close()  # type: ignore[union-attr]
            except Exception:  # noqa: BLE001
                pass
