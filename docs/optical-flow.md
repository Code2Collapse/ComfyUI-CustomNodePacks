# Optical Flow Re-Vector

## Optical Flow Re-Vector (MEC) (`OpticalFlowMEC`)
**Menu:** 🐺 C2C/🧰 Core/VFX · **Parameters:** [NODE_REFERENCE](../NODE_REFERENCE.md#opticalflowmec)
**Optional:** RAFT needs a ProPainter checkout and its `raft-things.pth` weight, which you install yourself (see
below). Without them the node uses a built-in Lucas-Kanade flow, which needs no model.

### What it is for
The node measures dense optical flow between `frame_a` and `frame_b`. It then moves `frame_a`'s pixels along that
flow, toward where they are in `frame_b`. With a `mask`, only the masked area moves and the rest of `frame_a` stays
untouched. It also outputs the flow as a picture and a map of where the flow is reliable.

It does not blur to hide holes. Where `frame_a` has no pixel to move in (for example, something newly uncovered),
the result is simply wrong there. The `consistency` output shows those places.

### When to pick it
- Pick it to push one frame, or one masked area, toward the next frame. Examples: re-aligning a patch, or moving a
  paint fix along with the motion.
- For flow inside video inpainting, or RAFT flow with warp and consistency maps as part of ProPainter, use
  `ProPainterMEC` with `mode` = `flow` (see [ProPainter](propainter.md)).

### Minimal wiring
```
frame A (to move) ───► frame_a ─┐
frame B (target) ────► frame_b ─┤
mask (optional) ─────► mask ────┴► Optical Flow Re-Vector ──► re_vectored ──► …
                                                         ├──► flow_rgb    (preview)
                                                         └──► consistency (MASK)
```
`frame_a` and `frame_b` must have the same size and the same frame count. A batch of A frames is matched one to one
with a batch of B frames.

### Flow direction (changed 2026-11-11, commit 6853f48)
`re_vectored` moves `frame_a` **toward** `frame_b`. Before that commit it moved the content the other way, against the
motion. In the commit's test (a texture shifted 4 px), the old result was further from `frame_b` than `frame_a`
itself.

If a workflow saved before then used `scale` = -1 to get the right direction, set `scale` back to 1. The `flow_rgb`
output did not change: it was and is the forward flow, from A to B.

### Key controls
- `scale` (default 1) multiplies the motion. 1 = all the way to `frame_b`, 0.5 = half way, 0 = no move, -1 = the
  opposite way.
- `iters` (default 20) is the number of RAFT refinement passes. More passes give a finer flow but take longer. The
  Lucas-Kanade fallback ignores it and always uses its own 5.
- `consistency_thr` (default 1.5 px): where the forward and backward flows disagree by more than this, `consistency`
  is 0.

### Reading the outputs
- `re_vectored`: `frame_a` moved along the flow. With a mask, only inside the mask.
- `flow_rgb`: the A-to-B flow as a picture, scaled per frame by the largest motion.
  - red = horizontal motion and green = vertical motion; 0.5 means still;
  - blue = speed.
- `consistency` (MASK): 1 where the forward and backward flows agree, 0 where they do not. Expect 0 at occlusions,
  at newly uncovered areas and at flow errors. Use it to find where `re_vectored` cannot be trusted.

### RAFT or Lucas-Kanade
- **RAFT** is used when two things are both present:
  - a ProPainter checkout, in `C2C_PROPAINTER_DIR` or `ComfyUI-CustomNodePacks/third_party/ProPainter`;
  - `raft-things.pth` in `models/propainter/`.
- The pack never downloads either one. ProPainter and its weights are under the S-Lab License 1.0, which allows
  non-commercial use only. Installing them is your decision; [ProPainter](propainter.md) explains where they go.
- **Lucas-Kanade** (pure PyTorch, no model) is used otherwise. When the RAFT weight is missing, the console logs
  "RAFT weights unavailable … using LK fallback". When ProPainter is missing, it logs "ProPainter unavailable — using
  LK fallback". The node itself does not report which method ran, so check the console.
- The flow runs on ComfyUI's torch device (the GPU when there is one). The move itself is done on the CPU.

### Common mistakes
- **`frame_a` and `frame_b` of different sizes or frame counts.** The node stops with "frame_a/b shape mismatch".
- **Expecting RAFT without installing it.** Without the checkout or the weight, the node quietly uses Lucas-Kanade.
  Check the console.
- **Trusting `re_vectored` everywhere.** Where `consistency` is 0, the moved pixels are a guess.
