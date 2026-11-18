# Mask Placement

**Mask Placement — Prompt/Ref → Place → Track** (`MaskPlacementMEC`) puts one cut-out (a mask, and its colour when
you give one) onto a video with a four-corner perspective quad, then keeps it in place as the shot moves.

## 1. Give it something to place

One of:

- **`source_mask`** wired: that mask is the shape. Add `source_image` to carry its colour too.
- **`prompt`** (for example `dog`, `mouth`) plus `sam_model` and a `grounding_model`: the node segments the object
  itself, on `ref_image` when wired, else on the anchor frame. Without a SAM model it uses a GrabCut stand-in and
  says so in `info` (`allow_download` decides whether a model may be fetched).

The source is cropped to its own bounds, so the quad maps the object, not the empty space around it.

## 2. Place it

Queue once to see the anchor frame in the editor, then drag:

- the four **corners** (free perspective; Shift = scale around the centre),
- the **edge middles** (one side), the **body** (move) and the **handle above the top edge** (rotate),
- **R** resets, **F** fits.

The quad is saved in `placement_json`, in anchor-frame pixels. `anchor_frame` is the frame you placed it on.
`feather_px` softens the edge when the placement has no feather of its own.

## 3. Choose how it follows the video (`track_mode`)

| Mode | What follows | Use it for | When it cannot follow |
|---|---|---|---|
| `static` | nothing: same place on every frame | stills, locked-off shots | - |
| `object_track` | the picture inside the quad: corners tracked frame to frame (Lucas-Kanade, forward-backward checked), one perspective transform per frame from the anchor frame; no model needed | a sign on a wall, a car door, a tabletop - anything with visible texture under the quad | the frame keeps the last placement and `frame_status` says `held` (flat colour, blur, the area leaves the frame) |
| `landmark_lock` | the wired `landmarks` (POSE_KEYPOINT, e.g. Face Landmarks (Facepoints) or DWPose) nearest the quad | a face or body part (mouth, cheek, forehead), even when the skin has little texture | held when too few shared landmarks |
| `auto` | `landmark_lock` when `landmarks` is wired, else `static` | - | - |

Both trackers walk out from the anchor frame in both directions, so you can place the quad on the clearest frame
in the middle of the shot.

## Outputs

`masks` (one per frame), `placed_rgb` (the placed colour), `overlay_preview` (frames with the mask tinted and the quad
drawn: check this first) and `info` (JSON: source, corners, `track_mode_effective`, one `frame_status` per frame,
and a `note` when something fell back).

## Tips

- `object_track` follows what is *under the quad* on the anchor frame. Place it over textured content; if the
  object itself is plain (a white mug), widen the quad a little onto its printed label or edge detail.
- Big occlusions (a hand passing in front) can pull the track: place the anchor after the occlusion or split the
  clip.
- For faces, `landmark_lock` is steadier than `object_track`: skin moves with expression, landmarks follow the face.
