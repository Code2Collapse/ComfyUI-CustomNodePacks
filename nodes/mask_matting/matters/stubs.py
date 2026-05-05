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


_stub("matanyone",   "MatAnyone (video matting)", "matanyone", needs_trimap=True)
_stub("bgmattingv2", "BackgroundMattingV2",       "bgmattingv2")
_stub("birefnet",    "BiRefNet (matter view)",    "birefnet")
_stub("rmbg",        "RMBG-2.0 (matter view)",    "rmbg")
