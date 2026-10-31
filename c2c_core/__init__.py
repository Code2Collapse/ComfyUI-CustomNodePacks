"""c2c_core - the C2C shared engine (L9.73, owner A20: "vendored shared core").

One canonical source here in ComfyUI-CustomNodePacks; byte-identical copies ship inside other packs (the MiniMax H3
Suite: mmx_vendor/c2c_core/), kept equal by tools/sync_vendored.py and the pre-push gate. Design:
docs/evidence/A20/L9.73_vendored_core_design.md in the workspace.

Rules for everything under this package:
- import only the standard library, torch / numpy, optional libraries lazily, ComfyUI core (comfy.*, folder_paths)
  and this package (relative imports) - never a pack module, so every copy works without the other packs;
- register no nodes, routes or settings - each pack wraps the engine in its own nodes;
- process-wide state (model caches, single-slot holders) goes through shared_state(), so a process with two copies
  does not hold two of everything;
- objects that cross packs carry a contract name + version and are checked with is_contract(), never isinstance
  against one copy's class.
"""
from __future__ import annotations

import sys
import threading
import types

CORE_VERSION = "1"

_REGISTRY = "c2c_core_shared_state"
_LOCK = threading.Lock()


def shared_state(name: str, factory):
    """The process-wide object called `name`, created once by factory() across every copy of the core.

    Copies of a different CORE_VERSION keep their own entry (one copy cannot read another's layout).
    """
    with _LOCK:
        reg = sys.modules.get(_REGISTRY)
        if reg is None:
            reg = types.ModuleType(_REGISTRY)
            reg.entries = {}
            sys.modules[_REGISTRY] = reg
        key = (CORE_VERSION, name)
        if key not in reg.entries:
            reg.entries[key] = factory()
        return reg.entries[key]


def is_contract(obj, name: str, major: int | None = None) -> bool:
    """True when obj declares contract `name` (attribute c2c_contract) and, if given, that major version
    (attribute c2c_contract_version, "major.minor" or an int)."""
    if getattr(obj, "c2c_contract", None) != name:
        return False
    if major is None:
        return True
    v = str(getattr(obj, "c2c_contract_version", ""))
    return v.split(".")[0] == str(major)


__all__ = ["CORE_VERSION", "is_contract", "shared_state"]
