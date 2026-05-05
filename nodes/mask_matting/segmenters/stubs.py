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


# All remaining backends from the master list.
_stub("sec",            "SeC (Segment-and-Track)",         "sec",            {"points", "bbox", "video", "auto"})
_stub("grounding-dino", "Grounding DINO (text → bbox)",    "grounding-dino", {"text"})
_stub("birefnet",       "BiRefNet (BG removal)",           "birefnet",       {"auto"})
_stub("rmbg",           "RMBG-2.0 (BG removal)",           "rmbg",           {"auto"})
_stub("videomama",      "VideoMaMa (text-video)",          "videomama",      {"text", "video"})
_stub("inspyrenet",     "InSPyReNet (BG removal)",         "inspyrenet",     {"auto"})
_stub("cutie",          "Cutie (video object track)",      "cutie",          {"points", "bbox", "video"})
_stub("dis",            "DIS-IS-Net (high-res salient)",   "dis",            {"auto"})
_stub("xmem",           "XMem++ (long-form video)",        "xmem",           {"points", "bbox", "video"})
_stub("person-mask",    "Impact PersonMask",               "ultralytics_bbox", {"auto"})
