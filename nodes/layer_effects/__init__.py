"""MEC Layer Effects — guarded registration."""
from __future__ import annotations

try:
    from .._c2c_registry import record_failure as _c2c_rec
except Exception:  # pragma: no cover
    _c2c_rec = None

try:
    from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
except Exception as _exc:  # pragma: no cover
    if _c2c_rec is not None:
        _c2c_rec(
            "layer_effects",
            _exc,
            hint="Layer Effects failed to import. Check nodes/layer_effects/_blend.py and _ops.py.",
            group="layer_effects",
        )
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
