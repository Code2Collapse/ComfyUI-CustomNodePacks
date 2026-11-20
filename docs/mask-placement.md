# Mask Placement

Mask Placement puts one cut-out (a mask, and its colour when you give one) onto a still or a video with a
four-corner perspective quad, then keeps it in place as the shot moves.

| Node | Use it to |
|---|---|
| [Mask Placement](#mask-placement--promptref--place--track-maskplacementmec) (`MaskPlacementMEC`) | Place a mask with a perspective quad and make it follow the video |

## Mask Placement — Prompt/Ref → Place → Track (`MaskPlacementMEC`)

**Menu:** 🐺 C2C/🧰 Core/Mask · **Parameters:** [NODE_REFERENCE](../NODE_REFERENCE.md#maskplacementmec)

Needs `opencv-python`. The prompt path needs a SAM model (SAM Model Loader) and a GroundingDINO model; with a wired
`source_mask` no model is used. `object_track` and `static` use no model; `landmark_lock` needs a landmark stream.

### What it is for

Putting a shape where it belongs in a shot: a logo on a wall, a patch on a car door, a mask over a mouth or a cheek
for inpainting. You give it the shape, drag a quad onto the frame, and the node writes one placed mask per frame.

### When to pick it

- You have a shape and want to put it somewhere else in the frame, in perspective: this node.
- You want to cut the subject out where it already is: [Mask + Matting](mask-matting-pipeline.md).
- You want to follow an existing mask through a clip or smooth its box: Mask Track
  ([video-temporal-bbox.md](video-temporal-bbox.md)).
- You want to draw or paint the mask yourself: Image Mask Editor ([image-mask-editor.md](image-mask-editor.md)).

### Minimal wiring

With your own mask:

```
Load Video ─────────────images──► Mask Placement ──masks───────────► Inpaint / composite
Load Image ──► (any mask) ─source_mask──►   │     ──placed_rgb──────► composite with masks
Load Image ─────────────source_image──►     │     ──overlay_preview─► Preview Image (check this first)
                                            └─────info
```

With a text prompt instead of a mask:

```
SAM Model Loader ──sam_model──► Mask Placement   prompt = "dog", grounding_model = groundingdino_...
Load Image (optional) ─ref_image─►               (segments the dog on ref_image, else on the anchor frame)
```

For a face or body part, add landmarks:

```
Face Landmarks (Facepoints) / DWPose ──landmarks──► Mask Placement   track_mode = landmark_lock
```

### Step by step

1. **Give it something to place.** Wire `source_mask` (its first frame is the shape; add `source_image` for the
   colour, same size as the mask). Or type a `prompt` (`dog`, `mouth`) with `sam_model` wired: the node segments the
   object on `ref_image` when wired, else on the frame `anchor_frame`. While `source_mask` is wired, `prompt`,
   `grounding_model` and `allow_download` are hidden. The shape is cropped to its own bounds, so the quad maps the
   object, not the empty space around it.
2. **Queue once.** The editor on the node then shows the anchor frame as its backdrop.
3. **Place it.** Drag:
   - a **corner**: free perspective (Shift + corner scales around the centre);
   - an **edge middle**: one side;
   - **inside** the quad: move;
   - the **handle above the top edge**: rotate.
   **R** resets to a centred quad, **F** fits the quad to the frame. The quad is saved in `placement_json` (hidden,
   anchor-frame pixels). Empty means a centred quad of 60 % of the real frame.
4. **Choose how it follows** (`track_mode`), then queue again.

### Key controls

| `track_mode` | What follows | Use it for | When it cannot follow |
|---|---|---|---|
| `static` (default) | Nothing: the same place on every frame | Stills, locked-off shots | - |
| `object_track` | The picture inside the quad: corners tracked frame to frame (Lucas-Kanade, forward-backward checked), one perspective transform per frame; no model | A sign on a wall, a car door, a tabletop: anything with visible texture under the quad | The frame keeps the last placement and its `frame_status` says `held` (flat colour, blur, the area leaves the frame) |
| `landmark_lock` | The wired `landmarks` (POSE_KEYPOINT) nearest the quad, of the person nearest the quad | A face or body part (mouth, cheek, forehead), even with little skin texture | `held` when fewer than 3 landmarks are shared |
| `auto` | `landmark_lock` when `landmarks` is wired, else `static` | - | - |

- Both trackers walk out from `anchor_frame` in both directions, so you can place the quad on the clearest frame in
  the middle of the shot. Tracking runs only when `images` has more than one frame.
- A landmark stream shorter than the clip holds its last frame.
- `feather_px` (default 6) softens the placed edge; a `feather` inside `placement_json` wins.
- `allow_download` (off by default) lets the node download the GroundingDINO model for a text prompt.

All inputs with tooltips: [NODE_REFERENCE](../NODE_REFERENCE.md#maskplacementmec).

### Reading the outputs

| Output | What it is |
|---|---|
| `masks` | One placed mask per frame |
| `placed_rgb` | The placed colour over black (black when no colour was given); use it with `masks` to composite |
| `overlay_preview` | The frames with the placed mask tinted green and the quad drawn: check this first |
| `info` | JSON report |

`info` fields:
- `source_mode`: `byo_mask` (a wired mask) or `prompt_segmentation`;
- `model_status`: `loaded - found at <path>`, `missing - model-free result` / `needs permission - model-free result`
  (the SAM Model Loader ran without a model, so the cut-out is approximate), or `not used (source_mask wired)`;
- `source_bbox`, `corners`, `feather_px`, `frames`;
- `track_mode_requested` and `track_mode_effective`: what you asked for and what ran;
- `frame_status`: one entry per frame when a tracker ran: `anchor`, `tracked (homography, N pts)` /
  `tracked (affine, N pts)` (object_track), `locked (affine, N pts)` / `locked (similarity, N pts)` (landmark_lock), or
  `held (...)`. Empty for `static`;
- `note`: how many frames held, or why a mode fell back to `static`.

### Common mistakes

| What you see | Why | What to do |
|---|---|---|
| "a text prompt needs the 'sam_model' input wired" | `prompt` is set but no SAM model | Wire SAM Model Loader, or wire `source_mask` instead |
| "has no mask source" | Neither `source_mask` nor `prompt` | Give one of them |
| "the source mask is empty — nothing to place" | The mask or the segmentation found nothing | Check the mask, or try another prompt |
| `landmark_lock` behaves like `static`; `note` says it "needs the `landmarks` input wired" | No landmarks | Wire Face Landmarks (Facepoints) or a DWPose estimator |
| Many frames `held` with `object_track`; the anchor-frame status says "no trackable detail inside the quad" | The area under the quad is flat or blurred | Place the quad over textured content (a printed label, an edge), or use `landmark_lock` for a face |
| `placed_rgb` is black | No colour source, or `source_image` is not the size of `source_mask` (the mask is then placed without colour) | Wire a `source_image` of the mask's size |
| Only the first frame of a mask batch is placed | The node uses the first frame of `source_mask` as the shape | Feed the single shape you want |
| "MaskPlacementMEC needs opencv-python (cv2) installed" | OpenCV is missing | Install `opencv-python` |

For faces, prefer `landmark_lock`: `object_track` follows the skin texture under the quad, which changes with the
expression, while landmarks follow the face. This is a usage tip, not a measured comparison.
