"""Tiled video refinement that does not move anything.

Three layers, each usable alone:

  _core      geometry: one tile plan per clip, complementary windows that sum
             to exactly 1, float32 merge. Model-agnostic and layout-agnostic -
             pixels or latents, one frame or a hundred.
  _identity  the promise: keep the plate's structure, take the model's detail.
  _nodes     the ComfyUI surface.

The loop is turned inside out on purpose: Tile Split emits tiles as a BATCH,
you wire any refiner you like, Tile Merge puts them back. There is no sampler
code here, so nothing breaks when the sampler API moves and "works with any
model" is not a list of the four it was tested against.
"""

from ._nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
