"""C2C_EXR_PASSES - every layer of a loaded EXR source, lazily (owner A21, Rev 3 Part 3.7 / 3.11; ledger L9.89).

The loader's IMAGE carries one layer. This bundle carries ALL of them without decoding any: it holds the loaded handle
(selection, crop and size included) and the layer table of the first frame, and reads a layer for given positions on
request. NukeMax's NUKE_PASSES is an eager single-frame dict (every AOV of a 100-frame 4K sequence would be tens of GB),
hence a separate, lazy type. Consumers check the protocol (``c2c_type == "C2C_EXR_PASSES"``), never the class, so a
vendored copy of this module in another pack is the same type.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Sequence

import numpy as np

from .handle import C2CVideo

TYPE = "C2C_EXR_PASSES"


class ExrPasses:
    c2c_type = TYPE

    def __init__(self, handle: C2CVideo | None, layers=(), *, layer_used: str = "") -> None:
        self._handle = handle
        self._layers = {lay.key: lay for lay in layers}
        self.layer_used = layer_used

    @classmethod
    def empty(cls) -> ExrPasses:
        return cls(None, ())

    def __len__(self) -> int:
        return len(self._handle) if self._handle is not None else 0

    def __bool__(self) -> bool:                # an empty bundle is a valid value, not a missing one
        return True

    def layers(self) -> list[str]:
        return list(self._layers)

    def kind(self, key: str) -> str:
        return self._get(key).kind

    def _get(self, key: str):
        lay = self._layers.get(key)
        if lay is None:
            low = {k.lower(): v for k, v in self._layers.items()}.get(str(key).lower())
            if low is None:
                have = ", ".join(self._layers) or "none (not an EXR source)"
                raise KeyError(f"layer {key!r} is not in this source; its layers: {have}")
            lay = low
        return lay

    def handle_for(self, key: str) -> C2CVideo:
        """The loaded handle reading ``key`` instead of the IMAGE's layer (same frames, crop and size)."""
        if self._handle is None:
            raise KeyError("an empty pass bundle (the source is not an EXR)")
        lay = self._get(key)
        src = self._handle.source
        if getattr(src, "kind", "") != "sequence":             # by kind: a vendored copy has its own class
            raise KeyError("only image sources have passes")
        new_src = replace(src, part=lay.part, channels=tuple(lay.channels), alpha_channel=lay.alpha, layer=lay.key)
        return replace(self._handle, source=new_src, has_alpha=bool(lay.alpha))

    def read(self, key: str, positions: Sequence[int] | None = None) -> np.ndarray:
        """[N, H, W, 3 or 4] float32 of layer ``key`` at the given positions (default: all). Raw values."""
        from .reader import read

        h = self.handle_for(key)
        return read(h, range(len(h)) if positions is None else positions, fmt="float32")

    def to_dict(self) -> dict:
        return {"type": TYPE, "frames": len(self), "layer_used": self.layer_used,
                "layers": [lay.to_dict() for lay in self._layers.values()]}

    def __repr__(self) -> str:
        return f"<C2C_EXR_PASSES {len(self)} frames: {', '.join(self._layers) or 'empty'}>"


def is_passes(obj) -> bool:
    return getattr(obj, "c2c_type", None) == TYPE
