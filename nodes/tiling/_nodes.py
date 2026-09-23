"""Tiled video refinement, as nodes.

THE SHAPE OF THIS, and why it is not one big node.

The obvious design is a single "tiled upscale" node that takes a MODEL and a
sampler and loops over tiles internally. Every pack that does this is welded
to one sampler, one ComfyUI version, and one model family, and "works with any
model set" becomes a list of the four it was tested against.

So the loop is turned inside out. `TileSplitMEC` emits the tiles AS A BATCH,
the user wires whatever refiner they like - any sampler, any model, an upscale
model, a LUT, three nodes in a row - and `TileMergeMEC` puts the batch back.
ComfyUI already knows how to run a batch through anything, so there is no
sampler code here at all and nothing to break when the sampler API moves.

    IMAGE ──> TileSplit ──> [ anything that takes and returns IMAGE ] ──┐
      │           │                                                     │
      │        TILE_PLAN ────────────────────────────────────────────> TileMerge
      │                                                                 │
      └────────────────> IdentityLock <─────────────────────────────────┘

BATCH ORDER is tile-major: every frame of tile 0, then every frame of tile 1.
Stated because it matters - a video refiner reading the batch as a sequence
sees one tile's whole time range contiguously, which is what lets it be
temporally consistent. Frame-major would hand it a batch that jumps around the
frame, and it would deflicker across tiles rather than across time.
"""

from __future__ import annotations

import torch

from ._core import (
    TilePlan,
    TilingError,
    build_plan,
    crop,
    describe_plan,
    merge,
    plan_for_tile_size,
)
from ._identity import (
    IdentityLockError,
    describe as describe_lock,
    drift,
    drift_map,
    identity_lock,
)

CATEGORY = "C2C/Tiling"


def _spatial(x: torch.Tensor) -> tuple[int, int]:
    from ._core import _spatial_dims
    return _spatial_dims(x)


class TilePlanMEC:
    """Work out the tiles once, for the whole clip."""

    VRAM_TIER = 0

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Any frame at the size you will "
                                               "be tiling. Only its dimensions "
                                               "are read."}),
                "mode": (["tile size", "tile count"], {"default": "tile size"}),
                "tile_size": ("INT", {
                    "default": 1024, "min": 128, "max": 8192, "step": 64,
                    "tooltip": "A MAXIMUM, not a target. Tiles are equal-sized "
                               "and must cover the frame exactly, so the size "
                               "chosen is the largest at or below this that "
                               "does. The report says what it picked."}),
                "rows": ("INT", {"default": 2, "min": 1, "max": 16}),
                "cols": ("INT", {"default": 2, "min": 1, "max": 16}),
                "overlap": ("INT", {
                    "default": 96, "min": 0, "max": 512, "step": 8,
                    "tooltip": "How far neighbouring tiles share pixels. The "
                               "blend has to hide any difference between two "
                               "tiles inside this band, so too narrow shows a "
                               "seam. 64-128px is usual."}),
                "blend": (["cosine", "linear", "none"], {
                    "default": "cosine",
                    "tooltip": "cosine is the right answer: the fade-in and "
                               "fade-out sum to exactly 1 across the overlap, "
                               "so there is no seam and no brightness dip. "
                               "'none' is for checking where the tiles are."}),
            }
        }

    RETURN_TYPES = ("TILE_PLAN", "STRING")
    RETURN_NAMES = ("tile_plan", "info")
    FUNCTION = "plan"
    CATEGORY = CATEGORY
    DESCRIPTION = ("Decide the tiles once for a whole clip. Reusing one plan "
                   "for every frame is what stops seams crawling.")

    @classmethod
    def IS_CHANGED(cls, image, **kw):
        h, w = _spatial(image)
        return f"{w}x{h}:" + repr(sorted(kw.items()))

    def plan(self, image, mode, tile_size, rows, cols, overlap, blend):
        h, w = _spatial(image)
        if mode == "tile size":
            plan = plan_for_tile_size(h, w, tile=int(tile_size),
                                      overlap=int(overlap), blend=blend)
        else:
            plan = build_plan(h, w, rows=int(rows), cols=int(cols),
                              overlap=int(overlap), blend=blend)
        return (plan, describe_plan(plan))


class TileSplitMEC:
    """Every tile of every frame, as one batch."""

    VRAM_TIER = 1

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "tile_plan": ("TILE_PLAN",),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "STRING")
    RETURN_NAMES = ("tiles", "tile_count", "info")
    FUNCTION = "split"
    CATEGORY = CATEGORY
    DESCRIPTION = ("Split a clip into overlapping tiles, emitted as a batch so "
                   "any refiner can process them. Wire the result through "
                   "whatever you like, then Tile Merge.")

    @classmethod
    def IS_CHANGED(cls, image, tile_plan):
        import hashlib
        h = hashlib.md5()
        h.update(image.detach().cpu().numpy().tobytes())
        h.update(repr(tile_plan.tiles if isinstance(tile_plan, TilePlan)
                      else tile_plan).encode())
        return h.hexdigest()

    def split(self, image, tile_plan):
        if not isinstance(tile_plan, TilePlan):
            raise TilingError(
                "The tile_plan input is not a tile plan. Wire it from Tile "
                "Plan's own output.")
        h, w = _spatial(image)
        if (h, w) != (tile_plan.height, tile_plan.width):
            raise TilingError(
                f"This plan is for {tile_plan.width}x{tile_plan.height} but the "
                f"image is {w}x{h}. Build the plan from the same footage you "
                "are tiling, or every tile lands on the wrong part of the frame.")

        frames = image.shape[0]
        # Tile-major: all frames of tile 0, then all frames of tile 1. A video
        # refiner reading this batch as a sequence sees one tile's whole time
        # range contiguously, which is what lets it be temporally consistent.
        pieces = [crop(image, t) for t in tile_plan.tiles]
        batch = torch.cat(pieces, dim=0)

        info = [
            f"{len(tile_plan.tiles)} tiles x {frames} frame(s) = "
            f"{batch.shape[0]} images out, each "
            f"{batch.shape[2]}x{batch.shape[1]}.",
            "Batch order is TILE-MAJOR: every frame of tile 1, then every "
            "frame of tile 2, and so on. A video refiner therefore sees each "
            "tile's whole time range in one run, which is what it needs to be "
            "temporally consistent.",
            "Whatever you wire next must return the SAME number of images in "
            "the SAME order. It may change their size, as long as every tile "
            "changes by the same factor.",
        ]
        if batch.shape[0] > 64:
            info.append(
                f"WARNING: {batch.shape[0]} images is a large batch. Most "
                "samplers will try to do it in one go and run out of memory - "
                "consider fewer tiles, or a node that batches internally.")
        return (batch, len(tile_plan.tiles), "\n".join(info))


class TileMergeMEC:
    """Put the tiles back, weighted, with no seam."""

    VRAM_TIER = 1

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tiles": ("IMAGE",),
                "tile_plan": ("TILE_PLAN",),
            },
            "optional": {
                "scale": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 8.0, "step": 0.05,
                    "tooltip": "How much the refiner enlarged each tile. 0 "
                               "means work it out from the tiles themselves, "
                               "which is right when every tile scaled by the "
                               "same clean factor. Set it explicitly if the "
                               "ratio is not a round number."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "info")
    FUNCTION = "merge"
    CATEGORY = CATEGORY
    DESCRIPTION = "Blend the tiles back into one clip, in float32, with no seam."

    @classmethod
    def IS_CHANGED(cls, tiles, tile_plan, scale=0.0):
        import hashlib
        h = hashlib.md5()
        h.update(tiles.detach().cpu().numpy().tobytes())
        h.update(repr(getattr(tile_plan, "tiles", tile_plan)).encode())
        h.update(repr(scale).encode())
        return h.hexdigest()

    def merge(self, tiles, tile_plan, scale=0.0):
        if not isinstance(tile_plan, TilePlan):
            raise TilingError(
                "The tile_plan input is not a tile plan. It must be the SAME "
                "plan Tile Split used - a different one puts the tiles back in "
                "the wrong places.")

        n_tiles = len(tile_plan.tiles)
        total = tiles.shape[0]
        if total % n_tiles:
            raise TilingError(
                f"{total} images came back but the plan has {n_tiles} tiles, "
                f"which does not divide evenly. Whatever refined the tiles "
                "changed how many there are - it must return the same count, "
                "in the same order.")
        frames = total // n_tiles

        th, tw = _spatial(tiles)
        src = tile_plan.tiles[0]
        if scale <= 0:
            ratio_h = th / float(src.height)
            ratio_w = tw / float(src.width)
            if abs(ratio_h - ratio_w) > 0.01:
                raise TilingError(
                    f"The tiles came back {tw}x{th} from {src.width}x"
                    f"{src.height} - that is {ratio_w:.3f}x wide by "
                    f"{ratio_h:.3f}x tall, which is not a uniform scale. Set "
                    "the scale input explicitly, or make the refiner preserve "
                    "the aspect.")
            scale = ratio_h

        # Regroup from tile-major back to per-tile clips.
        per_tile = [tiles[i * frames:(i + 1) * frames] for i in range(n_tiles)]
        out = merge(per_tile, tile_plan, scale=float(scale))

        info = [
            f"Merged {n_tiles} tiles x {frames} frame(s) into "
            f"{out.shape[2]}x{out.shape[1]}, scale {scale:.3f}x.",
            "Blended in float32 and divided by the accumulated weight, so the "
            "frame edges - where the windows cannot sum to 1 because there is "
            "no neighbour - come out at the right level rather than dark.",
        ]
        return (out, "\n".join(info))


class IdentityLockMEC:
    """Take the refinement's detail; keep the plate's structure."""

    VRAM_TIER = 1

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original": ("IMAGE", {
                    "tooltip": "The plate, at the SAME size as the refined "
                               "image. If the refinement upscaled, upscale "
                               "this to match first."}),
                "refined": ("IMAGE", {"tooltip": "What came back from the model."}),
                "radius": ("FLOAT", {
                    "default": 8.0, "min": 0.0, "max": 128.0, "step": 0.5,
                    "tooltip": "The split frequency. SMALLER locks harder and "
                               "lets less refinement through; LARGER passes "
                               "more detail and more movement with it. If a "
                               "face is still moving, go SMALLER."}),
                "strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "0 = the raw refinement, 1 = fully locked. Set "
                               "it to 0 WITH a protect mask to lock only the "
                               "face and leave the background completely free."}),
                "corrections": ("INT", {
                    "default": 2, "min": 0, "max": 8,
                    "tooltip": "Extra passes that push the result's structure "
                               "back onto the plate's. A Gaussian is not a "
                               "brick-wall filter, so one pass alone holds "
                               "only about 70% of a face shift; two hold about "
                               "94%, at a 2% cost in detail."}),
            },
            "optional": {
                "protect": ("MASK", {
                    "tooltip": "Where the lock is at FULL strength regardless "
                               "of the strength above - a face. Everywhere "
                               "else gets whatever strength says."}),
                "want_drift_map": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Also output a map of where structure moved. "
                               "Bright means it moved. Use it to find what the "
                               "drift number is telling you about."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "FLOAT", "STRING")
    RETURN_NAMES = ("image", "drift_map", "drift", "info")
    FUNCTION = "lock"
    CATEGORY = CATEGORY
    DESCRIPTION = ("Refine every pixel without letting anything move. Keeps "
                   "the plate's low frequencies - where composition and face "
                   "geometry live - and takes only the model's high "
                   "frequencies. The face cannot move, because the only thing "
                   "taken from the model is the band that does not encode "
                   "where anything is.")

    @classmethod
    def IS_CHANGED(cls, original, refined, **kw):
        import hashlib
        h = hashlib.md5()
        h.update(original.detach().cpu().numpy().tobytes())
        h.update(refined.detach().cpu().numpy().tobytes())
        h.update(repr(sorted((k, str(v)) for k, v in kw.items())).encode())
        return h.hexdigest()

    def lock(self, original, refined, radius, strength, corrections,
             protect=None, want_drift_map=False):
        if original.shape != refined.shape:
            raise IdentityLockError(
                f"The plate is {original.shape[2]}x{original.shape[1]} and the "
                f"refined image is {refined.shape[2]}x{refined.shape[1]}. They "
                "must match - upscale the plate to the refined size first, or "
                "the lock would compare different pixels.")

        out = identity_lock(original, refined, radius=float(radius),
                            strength=float(strength), protect=protect,
                            corrections=int(corrections))
        moved = drift(original, out, radius=float(radius))
        info = describe_lock(original, out, radius=float(radius),
                             strength=float(strength),
                             protected=protect is not None,
                             corrections=int(corrections))

        if want_drift_map:
            dmap = drift_map(original, out, radius=float(radius))
        else:
            # An all-zero map rather than None: a MASK output that is sometimes
            # absent breaks any graph that wires it.
            dmap = torch.zeros(original.shape[0], original.shape[1],
                               original.shape[2], dtype=torch.float32)
        return (out, dmap, float(moved), info)


NODE_CLASS_MAPPINGS = {
    "TilePlanMEC": TilePlanMEC,
    "TileSplitMEC": TileSplitMEC,
    "TileMergeMEC": TileMergeMEC,
    "IdentityLockMEC": IdentityLockMEC,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TilePlanMEC": "Tile Plan",
    "TileSplitMEC": "Tile Split",
    "TileMergeMEC": "Tile Merge",
    "IdentityLockMEC": "Identity Lock — refine without moving",
}
