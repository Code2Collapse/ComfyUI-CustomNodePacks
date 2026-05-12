"""Stub matter backends. Registered so combos enumerate them, but error on
``load`` until wired."""
from __future__ import annotations

from . import BaseMatter, register


def _stub(key, display, models_key, needs_trimap=False):
    cls = type(
        f"Stub_{key.replace('-', '_').replace('.', '_')}",
        (BaseMatter,),
        {
            "KEY": key,
            "DISPLAY": display,
            "MODELS_KEY": models_key,
            "NEEDS_TRIMAP": needs_trimap,
            "STATUS": "experimental",
            "load": lambda self: (_ for _ in ()).throw(
                NotImplementedError(f"{key} matter not yet wired (experimental).")
            ),
            "matte": lambda self, *a, **k: (_ for _ in ()).throw(
                NotImplementedError(f"{key} matter not yet wired (experimental).")
            ),
        },
    )
    register(cls)
    return cls


# Removed 2026-05-12:
#   * MatAnyone — multi-day video matting integration, stays out until real.
#   * bgmattingv2 — promoted to real backend in ``bgmattingv2_backend.py``.
# birefnet / rmbg are kept as "matter view" stubs because the segmenter
# side already implements them.
_stub("birefnet",    "BiRefNet (matter view)",    "birefnet")
_stub("rmbg",        "RMBG-2.0 (matter view)",    "rmbg")
