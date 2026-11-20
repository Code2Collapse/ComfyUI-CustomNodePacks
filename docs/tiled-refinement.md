# Tiled refinement

Four nodes for refining a large frame, or a whole clip, in tiles. Tile Plan decides the tiles once. Tile Split cuts
every frame into those tiles and hands them on as one IMAGE batch. You run that batch through any refiner you like
(a KSampler chain, an upscale model, a LUT, several nodes in a row). Tile Merge puts the tiles back and blends the
overlaps. Identity Lock is optional: it keeps the original plate's shapes and takes only the fine detail from the
refined result.

There is no sampler inside these nodes. Whatever refines the tiles is up to you.

| Node | Use it to |
|---|---|
| [Tile Plan](#tile-plan-tileplanmec) | Decide the tile grid once for a clip |
| [Tile Split](#tile-split-tilesplitmec) | Cut every frame into tiles, as one batch |
| [Tile Merge](#tile-merge-tilemergemec) | Blend the refined tiles back into frames |
| [Identity Lock — refine without moving](#identity-lock--refine-without-moving-identitylockmec) | Keep the plate's shapes, take only the refined detail |

## The chain

```
                    ┌──────────► Tile Plan ──── tile_plan ────────────┐
                    │                │                                │
Load Video ── image ┴──► Tile Split ─┘ tiles ──► [ your refiner ] ──► Tile Merge ──► image
    │                                          (IMAGE in, IMAGE out)     │
    │                                                                    ▼
    └── (upscaled to the merged size, if the refiner upscaled) ──► Identity Lock ──► Save
                                                    original            refined
```

Wire the same `tile_plan` into Tile Split and Tile Merge. Identity Lock goes after Tile Merge. It can also be used on
its own, after any refiner, without tiles.

---

## Tile Plan (`TilePlanMEC`)
**Menu:** 🐺 C2C/🧰 Core/Tiling · **Parameters:** [NODE_REFERENCE](../NODE_REFERENCE.md#tileplanmec)

### What it is for
It works out the tiles for one frame size. Only the size of `image` is read, not its pixels. You build the plan once
and every frame of the clip uses it. If the tile boundaries were worked out again for each frame, rounding could move
them slightly from frame to frame, and a seam that moves is easier to see than one that stays put.

### Key controls
- `mode`:
  - `tile size` makes the largest equal tiles that are no bigger than `tile_size`.
  - `tile count` makes a fixed `rows` x `cols` grid.
- `tile_size` (default 1024) is a maximum, not a target. All tiles are the same size and must cover the frame
  exactly, so the node picks the largest size at or below this value that works. Example: a 1920x1080 frame at
  `tile_size` 1024 with `overlap` 96 gives 4 tiles (2x2) of 1008x588.
- `overlap` (default 96 px) is how much neighbouring tiles share. The blend hides tile differences only inside this
  band. The tooltip says 64–128 px is usual.
- `blend`:
  - `cosine` (default): the fade-out of one tile and the fade-in of the next always add up to exactly 1.
  - `linear`: also adds up to 1, with a straight ramp.
  - `none`: no fade at all. Use it to see where the tiles are.

### Reading the outputs
- `tile_plan` goes to Tile Split and Tile Merge.
- `info` gives the tile count, the grid, the tile size, the overlap and the blend. It also gives how much extra area
  is processed because of the overlap. For the example above that is "14% more area".
- `info` adds a warning when the overlap is under 32 px, or when more than twice the frame area is processed. In the
  second case, fewer and larger tiles do the same work.

### Common mistakes
- **Overlap as wide as the frame, or `tile_size` not bigger than `overlap`.** The plan is refused with an error.
  Use a smaller overlap or a larger tile.

---

## Tile Split (`TileSplitMEC`)
**Menu:** 🐺 C2C/🧰 Core/Tiling · **Parameters:** [NODE_REFERENCE](../NODE_REFERENCE.md#tilesplitmec)

### What it is for
It cuts every frame of `image` into the plan's tiles and outputs them all as one IMAGE batch.

**Batch order is tile by tile:** first every frame of tile 1, then every frame of tile 2, and so on. A video refiner
that reads the batch as a sequence therefore gets one tile's whole time range in one piece. (The output tooltip says
"frame by frame". The code and the `info` text both use the tile-by-tile order described here.)

### Minimal wiring
```
image ──► Tile Split ◄── tile_plan (from Tile Plan)
              │
           tiles ──► your refiner ──► Tile Merge
```

### Reading the outputs
- `tiles`: the batch. Its size is (tiles per frame) x (frames). A 3-frame clip with a 2x2 plan gives 12 images.
- `tile_count`: the number of tiles per frame. This is the number of tiles in the plan, not the batch size.
- `info`: the tile count and frame count, the tile size, the batch order, and the rule for the refiner (see below).

### Common mistakes
- **The plan was made for another size.** Tile Split stops with "This plan is for WxH but the image is WxH". Build
  the plan from the same footage you split.
- **Something other than Tile Plan wired into `tile_plan`.** Tile Split refuses it.
- **Batch too large for the refiner.** Above 64 images, `info` warns that most samplers try to run the whole batch
  at once and may run out of memory. Use fewer tiles, or a refiner node that splits the batch itself.

---

## Tile Merge (`TileMergeMEC`)
**Menu:** 🐺 C2C/🧰 Core/Tiling · **Parameters:** [NODE_REFERENCE](../NODE_REFERENCE.md#tilemergemec)

### What it is for
It puts the refined tiles back into frames. Each tile is weighted by the plan's blend over the overlap. The sum is
built in float32 and divided by the total weight, so the frame edges (where a tile has no neighbour) keep the right
brightness. The result has the input tiles' dtype.

### The rule for your refiner
Whatever sits between Split and Merge must return **the same number of images, in the same order**. It may change
their size, as long as every tile is scaled by the same factor (for example, a 2x upscale model).

### Key controls
- `scale` (default 0) is how much the refiner enlarged each tile. At 0 the node works it out from the first tile's
  returned size. Set it by hand when the factor is not a clean number.

### Reading the outputs
- `image`: the merged clip, at (plan size x scale).
- `info`: tiles x frames merged, the output size and the scale used.

### Common mistakes
- **The refiner changed the image count.** "N images came back but the plan has M tiles, which does not divide
  evenly." The refiner must return one image per input image.
- **Width and height scaled differently.** If the returned tiles' width and height ratios differ by more than 1%,
  Tile Merge stops and asks you to set `scale` or keep the aspect. Note: Tile Plan does not round tile sizes to a
  multiple of 8 or 16 (the example above gives 1008x588 tiles). If a step in your refiner crops or pads tiles to
  such a multiple, check that the tiles come back at the size you expect.
- **A different plan from the one Tile Split used.** Use the same `tile_plan` output for both nodes.

---

## Identity Lock — refine without moving (`IdentityLockMEC`)
**Menu:** 🐺 C2C/🧰 Core/Tiling · **Parameters:** [NODE_REFERENCE](../NODE_REFERENCE.md#identitylockmec)

### What it is for
A refiner that adds detail can also redraw shapes slightly: a jaw shifts a few pixels, an eye changes. Over a clip
that reads as the face "swimming". Identity Lock splits both images with a Gaussian blur of size `radius`:

```
out = blur(original) + (refined − blur(refined))
```

The coarse part (layout, face geometry, colour) comes from the original. Only the fine part (pores, hair, grain, edge
detail) comes from the refined image.

The trade-off: anything the refiner fixed in the shapes does not come through either. A warped ear stays warped. Use
it for a finishing pass on a shot whose shapes are already right, not for a repair pass.

### When to pick it
- After tiled refinement or any img2img / upscale pass where the faces or the layout must not move.
- For a pass that is supposed to change shapes, leave it out, or lower `strength`.
- For a manual split into detail and base layers, see `FrequencySeparateMEC` / `FrequencyCombineMEC` in
  [NODE_REFERENCE](../NODE_REFERENCE.md#frequencyseparatemec).

### Minimal wiring
```
plate ──(upscale to the refined size, if needed)──► original ─┐
refined (e.g. Tile Merge image) ────────────────────► refined ─┤
face mask (optional) ───────────────────────────────► protect ─┴► Identity Lock ──► image
```

### Key controls
- `radius` (default 8 px) is where coarse ends and fine begins.
  - **Smaller** locks harder and lets less refinement through.
  - **Larger** lets more detail through, and more movement with it.
  - If a face still moves, go smaller.
- `strength` (default 1): 0 = the raw refined image, 1 = fully locked.
- `protect` (optional mask): inside it the lock is always at full strength, whatever `strength` says. Use
  `strength` 0 with a face mask to lock only the face and leave the background free.
- `corrections` (default 2): extra passes that push the result's coarse part back onto the original's. 0 is the plain
  split. The tooltip says one pass holds about 70% of a face shift and two hold about 94%, at a 2% cost in detail.
  That figure comes from a 6 px synthetic shift at radius 12 (docstring of `nodes/tiling/_identity.py`).
- `want_drift_map` (default off): also output a map of where the shapes moved.

### Reading the outputs
- `image`: the locked result.
- `drift` (FLOAT): how far the coarse part moved, as a fraction of the original's value range, at `radius`. Lower is
  closer to the plate.
- `drift_map` (MASK): bright where the shapes moved, scaled so the largest movement is 1. It is all black unless
  `want_drift_map` is on.
- `info`:
  - the settings, and the movement in % at `radius` and again at a coarser scale (4 x `radius`, at least 16 px);
  - a WARNING when `strength` is at full and the coarse-scale movement is above 2%. Use a smaller `radius` then;
  - "Structure is effectively unchanged" when the coarse-scale movement is 0.5% or less.

Measured on synthetic test data only: in the node tests, a test refiner added detail and shifted the picture by 4, 12
and 24 px. Through the whole chain the lock removed 94%, 98% and 99% of that drift (radius 6, two corrections; commit
f248a4d). This was not measured on real footage with a real model.

### Common mistakes
- **Original and refined have different sizes.** The node stops: "upscale the plate to the refined size first". After
  a 2x tiled upscale, upscale the plate 2x before `original`.
- **Expecting it to fix a bad hand or ear.** It cannot: the shapes come from the original by design.
