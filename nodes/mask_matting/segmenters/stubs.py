"""Stub segmenters — registered so the combo lists them, but report
``experimental`` until wired. Each follows the same shape so wiring later
is a drop-in.
"""
from __future__ import annotations

from . import BaseSegmenter, register


def _stub(key: str, display: str, models_key: str, modes: set, deps_check=lambda: False):
    cls = type(
        f"Stub_{key.replace('-', '_').replace('.', '_')}",
        (BaseSegmenter,),
        {
            "KEY": key,
            "DISPLAY": display,
            "MODELS_KEY": models_key,
            "SUPPORTS_MODES": modes,
            "STATUS": "ready" if deps_check() else "experimental",
            "load": lambda self: (_ for _ in ()).throw(
                NotImplementedError(f"{key} backend not yet wired (experimental).")
            ),
            "segment": lambda self, *a, **k: (_ for _ in ()).throw(
                NotImplementedError(f"{key} backend not yet wired (experimental).")
            ),
        },
    )
    register(cls)
    return cls


# Remaining stubs awaiting real wiring (Tier-1 candidates only).
# Removed 2026-05-12:
#   * SeC / VideoMaMa / Cutie / XMem — multi-day video integrations,
#     stay out of the combo list until backed by a real implementation.
#   * grounding-dino / dis / person-mask — promoted to real backends
#     in ``experimental_backend.py`` (registered there, overrides the
#     stub registration order; stubs no longer needed here).
# birefnet / rmbg / inspyrenet are real backends in ``salient_backend.py``.
# This file is intentionally empty of stub registrations now; kept so
# the package import wiring remains stable.
