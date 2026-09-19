"""MEC Mask toolkit — guarded registration."""
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
            "mask_toolkit",
            _exc,
            hint="Mask toolkit failed to import. Check nodes/mask_toolkit/_ops.py and nodes.py.",
            group="mask_toolkit",
        )
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
