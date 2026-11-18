"""C2C Distributed: multi-GPU for ComfyUI (design: docs/c2c-distributed.md).

Kill switch: environment variable C2C_DISTRIBUTED=off - every node refuses with that reason and the model patches
fall back to a normal single-GPU call. Importing this package starts nothing.
"""
from __future__ import annotations

import os


def enabled() -> bool:
    """L3.01: C2C setting 'C2C Distributed (multi-GPU)'; C2C_DISTRIBUTED still wins."""
    try:
        try:
            from ..nodes import _c2c_settings as _S
        except ImportError:                     # imported as a top-level package (tests, tools)
            from nodes import _c2c_settings as _S
        return bool(_S.get("c2c.distributed.enabled"))
    except Exception:  # noqa: BLE001
        return os.environ.get("C2C_DISTRIBUTED", "").strip().lower() not in ("off", "0", "false", "no")


def kill_switch_reason() -> str:
    return ("C2C Distributed is switched off (Settings > C2C > Multi-GPU > C2C Distributed, or the environment "
            "variable C2C_DISTRIBUTED=off).")


__all__ = ["enabled", "kill_switch_reason"]
