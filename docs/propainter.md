# ProPainter video inpaint

One node, `ProPainterMEC`, fills a masked region across the frames of a clip and blends inpainted crops back into
video. It wraps ProPainter (RAFT flow + recurrent flow completion + InpaintGenerator), which you install yourself, and
since 2026-11-19 has a model-free OpenCV fallback for its two filling modes.

| Node | Use it to |
|---|---|
| [ProPainter](#propainter-propaintermec) (`ProPainterMEC`) | Remove an object, wire or rig across a clip; fill an inpaint crop over time; hide the seam of a stitched inpaint; look at the optical flow between two frames |

## Installing ProPainter (your decision)

ProPainter is licensed under the **S-Lab License 1.0 - non-commercial use only**
(<https://github.com/sczhou/ProPainter/blob/main/LICENSE>). The weights carry the same licence. This pack does not
ship, download or install the code or the weights. The node uses ProPainter only when you already have it:

1. Get the code yourself, for example `git clone https://github.com/sczhou/ProPainter.git`. Then either set the
   environment variable `C2C_PROPAINTER_DIR` to that folder, or put it at
   `ComfyUI-CustomNodePacks/third_party/ProPainter`.
2. Put the three weights from the official release
   (<https://github.com/sczhou/ProPainter/releases/tag/v0.1.0>: `raft-things.pth`, `recurrent_flow_completion.pth`,
   `ProPainter.pth`) in `models/propainter/`.
3. Restart ComfyUI.

Until 2026-10-12 the pack fetched the code from GitHub on its first start; it no longer does. For commercial work,
use `backend` = `auto_no_propainter` or `opencv_flow` (below), which never use ProPainter.

## ProPainter (`ProPainterMEC`)

**Menu:** 🐺 C2C/🧰 Core/Inpaint, shown as "ProPainter — Temporal / Remove / Stitch / Refine / Flow" ·
**Parameters:** [NODE_REFERENCE](../NODE_REFERENCE.md#propaintermec)
`temporal`, `remove`, `stitch` and `stitch_refine` need ProPainter (code + the three weights) unless `backend`
picks the OpenCV fallback, which exists for `temporal` and `remove` only. The OpenCV fallback is APPROXIMATE and
was tested on a synthetic clip only; there is no real-footage A/B against ProPainter (commit b9ec253).

### What it is for

Pick a `mode`; the node shows only the controls that mode reads.

| `mode` | What it does | Wire |
|---|---|---|
| `remove` (default) | Fills the masked region of every frame from what other frames show: object, wire or rig removal. No diffusion model | `images`, `masks` |
| `temporal` | The same fill for a crop from Inpaint Crop Pro, with a blend at the crop boundary | `images`, `masks`, optional `stitch_data` |
| `stitch` | Pastes a generative inpaint back into the frames and repaints a band around the seam with ProPainter, so the seam stays steady over time | `stitch_data`, `inpainted_image` |
| `stitch_refine` | Repaints only a thin ring along the mask edge of an already-stitched clip | `stitched_image`, `stitch_data` or `mask_override` |
| `flow` | Optical flow between two frames, with a warp and a forward / backward consistency mask | `frame_a`, `frame_b` |

The five old nodes are these modes. Saved workflows with `ProPainterTemporalMEC` load as this node in mode
`temporal`. The other four (`ProPainterRemoveMEC`, `ProPainterStitchMEC`, `ProPainterStitchRefineMEC`,
`FlowRefineMEC`) have no automatic migration: replace them by hand with this node in the matching mode.

### When to pick it

- To remove something and the background is visible in other frames: `remove`. Nothing generative is involved, so
  it cannot invent what no frame shows.
- To generate new content in a region: run a diffusion inpaint first (see [inpaint-suite.md](inpaint-suite.md)),
  then blend it back with Inpaint Stitch Pro, or with `stitch` here when the seam flickers over time.
- For a single image: use the [Inpaint Suite](inpaint-suite.md); this node works across frames.
- For a flow-based re-vectoring of frames: Optical Flow Re-Vector (`OpticalFlowMEC`, see
  [optical-flow.md](optical-flow.md)). The `flow` mode here is for
  looking at the flow and its consistency.

### Minimal wiring

```
Remove an object:
Load Video ──────────▶ images ──┐
(a mask per frame) ──▶ masks ───┤ ProPainter (mode remove, backend auto) ─┬─ image_out ─▶ Save
                                                                          └─ info ─▶ (which engine ran)

Blend a generative inpaint back:
Inpaint Crop Pro ─┬─ stitcher ──────────────────────────────▶ stitch_data ─────┐
                  └─ cropped_image ─▶ your inpaint ─▶ image ─▶ inpainted_image ─┤ ProPainter (mode stitch) ─ image_out ─▶ Save
```

### Key controls

- `backend` (`temporal` / `remove` only; added 2026-11-19, commit b9ec253, the last input):
  - `propainter` (default, and what every older save runs): ProPainter only. Fails with the install hint when
    ProPainter is missing.
  - `auto`: ProPainter when you installed the code **and** all three weights, otherwise the OpenCV fallback. Nothing
    is downloaded.
  - `auto_no_propainter`: never ProPainter (licence-safe for commercial work).
  - `opencv_flow`: always the OpenCV fallback.

  The OpenCV fallback is model-free: Farneback optical flow at up to 512 px, with the masked object erased from
  both frames first; the flow inside the hole is filled in from its surroundings; the background is pulled from the
  nearest frames forwards and backwards in time; OpenCV Telea inpainting fills what no frame shows. Pixels outside
  the mask come back unchanged. It has no learned prior, so it cannot invent structure the way ProPainter can.
- `remove_quality` (`remove`): a preset that overrides `raft_iter`, `neighbor_stride`, `ref_stride` and
  `subvideo_length`:

  | Preset | RAFT iters | neighbor_stride | ref_stride | subvideo_length |
  |---|---:|---:|---:|---:|
  | `fast` | 8 | 10 | 20 | 8 |
  | `balanced` | 12 | 5 | 10 | 8 |
  | `quality` | 20 | 3 | 6 | 12 |

- `remove_dilate_pixels` (`remove`): grows the mask before filling, to cover anti-aliased edges.
- `color_match_mode`: per-frame colour match of the fill to its surroundings. `off` and `none` mean no match;
  `lab` and `lab_transfer` are the same.
- `subvideo_length` and `raft_chunk` (Windowing section): frames per InpaintGenerator window and frame pairs per RAFT
  pass. Lower them if you run out of VRAM. The tooltip suggests 8-30 frames on 8 GB cards, 40-60 on 12 GB, 80+ on
  24 GB.
- `use_half`: fp16 on CUDA, ignored on the CPU.
- `blend_boundary` (`temporal`): blends the fill with the original at the crop boundary, using the blend mask in
  `stitch_data`.
- `boundary_band_pixels`, `preserve_inpaint_center`, `upscale_method` (`stitch`): the width of the seam band
  ProPainter repaints (0 = none), whether the generative inpaint stays untouched in the centre, and how the crop is
  scaled to its place.
- `ring_pixels` (`stitch_refine`): half-width of the repainted ring.
- `flow_consistency_thr` (`flow`): forward / backward agreement threshold in pixels.

In `temporal` mode the node unloads the other models in ComfyUI (checkpoint, VAE, encoders) before it loads
ProPainter.

### Reading the outputs

The node always has the same five outputs, so links survive a mode change:

| Output | `temporal` / `remove` / `stitch` / `stitch_refine` | `flow` |
|---|---|---|
| `image_out` | The filled / stitched / refined frames | `frame_b` pulled onto `frame_a`'s pixel grid along the flow |
| `mask_out` | The region filled (`remove`: the dilated mask), the repainted band, or the ring | The consistency mask: 1 where forward and backward flow agree |
| `aux_image` | QC: \|output − input\| × 4 | The flow field as colour (R = U, G = V, B = magnitude) |
| `aux_mask` | QC: pixels the node changed (by more than 1.5/255 in any channel) | Where the flow is unreliable (1 − consistency) |
| `info` | `mode=...`, the engine's settings and timings, then `qc: changed X% of pixels, Y% outside mask_out` | `mode=flow` with the iterations and threshold |

- With a `backend` other than `propainter`, `info` adds `backend=<engine> (requested <value>: <reason>)`. With the
  OpenCV fallback it starts with `APPROXIMATE model-free fallback ...` and gives the flow resolution and how much of
  the hole was filled from other frames versus Telea (a spatial guess).
- A high "outside mask_out" share means the node changed pixels outside the region: look for spill or seams in
  `aux_image`.
- The QC outputs compare against the mode's input (`images`, the stitch canvas, or `stitched_image`). When the sizes
  differ they stay at zero and `info` says `qc: off (...)`.
- The `flow` mode uses ProPainter's RAFT when ProPainter and its weights are installed, otherwise a multi-scale
  Lucas-Kanade flow. Only the console says which one ran.

### Common mistakes

| Message or symptom | Why | What to do |
|---|---|---|
| `ProPainter is not installed. This node uses it only when you already have it.` | `backend` is `propainter` (the default), or a stitch mode, without a ProPainter checkout | Install it as above, or for `temporal` / `remove` set `backend` to `auto` or `opencv_flow` |
| `ProPainter weight '...' is not in .../models/propainter` | A weight file is missing | Put the three official `.pth` files in `models/propainter/` |
| `mode='remove' requires 'images' and 'masks'.` (and the same for the other modes' inputs) | The mode's inputs are not wired | Wire the inputs listed in the mode table |
| `the STITCHER's crop size (width) changes from frame to frame` | The stitch modes need one crop size for the whole clip | Use a fixed crop size in Inpaint Crop Pro |
| `the STITCHER's canvases differ in size from frame to frame` | Same cause, for the canvas | Same fix |
| `No inpaint backend is available` / `The OpenCV inpaint backend is not available` | OpenCV (`opencv-python`) is missing | Reinstall the pack requirements |
| Out of memory | Windows too large | Lower `subvideo_length`, then `raft_chunk` |
| `info` says `NOOP_empty_mask` (or `NOOP_empty_ring`, `NOOP_empty_repaint_mask`) | The mask, ring or band is empty | Check the mask input |

## Credits

ProPainter by sczhou and contributors, <https://github.com/sczhou/ProPainter>, S-Lab License 1.0
(non-commercial). Not included in this pack.
