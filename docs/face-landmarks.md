# Face Landmarks (Facepoints, 620 + 68/98) (`FacepointsLandmarksC2C`)

**Menu:** 🐺 C2C/🧰 Core/Face · **Parameters:** [NODE_REFERENCE](../NODE_REFERENCE.md#facepointslandmarksc2c)

**Needs model files** (not shipped), both in `models/facepoints/`:
- myned-ai/facepoints: `model.safetensors` (416 MB) + `config.json`. The weights are **CC BY-NC 4.0: non-commercial
  use only**. Always needed.
- YuNet (`face_detection_yunet_2023mar.onnx`, 0.2 MB, MIT, OpenCV Zoo). Needed only for `face_finder` =
  `yunet (finds faces)`.

They are downloaded only when you tick `allow_download` (the facepoints files come from the pinned commit 6d51ab4
and their SHA-256 is checked) and the C2C setting **Never download models** is off (`C2C_NO_DOWNLOADS=1` also
blocks it). The facepoints download uses the `huggingface_hub` package. The node draws with OpenCV (`opencv-python`). A
`yolo:` face finder also needs `ultralytics`. Tested with the real weights on CPU (677af0d, 90b1484); a GPU run is
NOT VERIFIED in the commit history.

## What it is for

It finds each face and places 620 points on it (Google's GNM Head mesh), each with the model's own error estimate.
It also gives the standard 68-point (iBUG) and 98-point (WFLW) sets and both pupils. The results come out in
standard formats: an OpenPose `POSE_KEYPOINT`, a dot control image, an overlay, a face mask and a JSON with every
set. It does not change the picture. The engine is a port of myned-ai/facepoints (code Apache-2.0); on a real face
its points differ from upstream's by at most 0.0002 px (677af0d).

## When to pick it

- **Face Landmarks**: you need face points or a face-shaped mask for another node, on a still or a clip.
- **[Mask Placement](mask-placement.md)** (`MaskPlacementMEC`): its `track_mode` = `landmark_lock` follows a
  `POSE_KEYPOINT` stream - wire `pose_keypoint` from this node into its `landmarks` input.
- **[Face Fixer](face-fixer.md)**: you want the faces re-rendered with your model. It finds faces with a YOLO box
  only, no landmarks.

## Minimal wiring

```
Load Image / a video loader ─ image ─► Face Landmarks ─ pose_keypoint ──► Mask Placement (landmarks) or any
                                                      │                    OpenPose face consumer
                                                      ├ landmark_image ─► ControlNet / Preview Image
                                                      ├ overlay ────────► Preview Image
                                                      ├ face_mask ──────► mask / inpaint nodes
                                                      └ landmarks_json ─► a text viewer
```

First run: tick `allow_download` (shown in the **Model setup** section only while a model file is missing), run
once, untick it.

## Key controls

- `face_finder`: `yunet (finds faces)` (default), `whole image (already a face crop)`, or `yolo: <file>` for each
  YOLO face model in `models/ultralytics/bbox/`. A wired `face_mask` replaces the finder: its box is used, one face.
  While `face_mask` is wired, `face_finder`, `max_faces`, `finder_confidence` and `track_video` are hidden.
- `max_faces` (1 by default): faces per frame, left to right. `finder_confidence` (0.6): the finder's minimum score.
- `passes` (3, the released default) and `track_video` (on: each frame's box starts from the face's points in the
  previous frame) sit in the **Tracking** section.
- `landmark_set` (**Output** section): which points are drawn into `landmark_image` - `68 + iris (OpenPose face)`,
  `98 (WFLW)` or `620 dense (GNM Head)`. `pose_keypoint` is always the 68 points + 2 pupils.
- `confident_pct` (**Confidence** section, 2.5): a point counts as sure when its calibrated error estimate is below
  this % of the face size; its confidence `c` is exactly 0.5 there.

47 more controls sit in collapsible sections, one per name prefix. With every one at its default, the points, the
pose and the mask are what the node gave before these controls existed (bit-identical tests reported in 90b1484,
59d91e6 and 5935c7f); the default `screening_robust_detection` = auto only acts on frames where no face was found:
- **Appearance** (`appearance_`): colours as `#RRGGBB`. They apply only with `appearance_scheme` = `custom` and never
  change the pose numbers. `appearance_line_width` sets the contour line width.
- **Strength** (`strength_`): overlay opacity, dot intensity, how strongly unsure points are drawn, the confidence
  below which a point counts as unsure, a multiplier for the `c` values in `pose_keypoint`, and
  `strength_connection_lines` (0 = off: contour lines for the 68 and 98 sets).
- **Resolution** (`resolution_`): `resolution_processing` finds and crops faces on a smaller frame (points still come
  back in source pixels); `resolution_output` sets the size of the images, the mask and the coordinates, so
  everything stays aligned.
- **Screening** (`screening_`): minimum face confidence, minimum point confidence (pose `c` = 0, x / y kept), how
  unsure 620-set points are drawn (`dim (default)`, `fade`, `hide`, `keep`), rejected faces in the JSON, and
  `screening_robust_detection`: `auto (when a frame finds no face)` (default) retries a frame without a face on
  enhanced copies (light, colour, deblur, a 1.5x look), then with a nearby frame's box; `always` detects every frame
  on the enhanced copy; `off`. The copies are only shown to the detector; every output keeps the original footage.
- **Mask** (`mask_`): `mask_faces` (`all faces`, `largest face`, `face index` + `mask_face_index`), `mask_part`
  (`whole face (default)`, eyes, one eye, eyelids, brows, nose, lips, inner mouth / teeth, around the mouth, cheeks,
  chin + jaw, and the ESTIMATED `forehead`, `ears`, `hair`, `head + hair`, or a `custom list` typed in
  `mask_parts_custom`), `mask_head_refine` (fits the estimated parts to the picture with GrabCut), grow, shrink,
  feather, edge softness, threshold, hull smoothing, `mask_invert`, `mask_extend_up / down / left / right` (% of the
  face height, along the head's tilt when `mask_extend_follow_tilt` is on) with `mask_extend_falloff`, and
  `mask_preview` (tint or outline on the overlay; the MASK output is not changed).
- **Temporal** (`temporal_`): landmark smoothing for video, separate from `track_video`. A still point is averaged
  over about 1 / (1 − `temporal_strength`) frames (0.5: about 2, 0.8: about 5); faster moves are followed at once.
  Also: confidence-aware smoothing, a lag limit (`temporal_max_offset`), a scene-cut restart (`temporal_scene_cut`),
  a restart after a lost face, and `temporal_stable_order` (faces keep their place by identity). With smoothing or
  stable order on, a scene cut also drops the previous shot's tracking boxes. With identity tracking on,
  `mask_faces` = `largest face` stays on the face it picked first while that face is in view.

Measured with the real model on CPU (90b1484; evidence `docs/evidence/A27/face/` in the workspace): mean landmark
jitter 1.47 → 0.29 px on a shaky clip with `temporal_strength` 0.8; resized outputs aligned within 0.07 px.

## Reading the outputs

- `pose_keypoint`: one OpenPose entry per frame (`people`, `canvas_width`, `canvas_height`). Each person has
  `face_keypoints_2d` = 70 x `[x, y, c]` in output pixels: the 68 iBUG points, then the pupil of points 36-41 and the
  pupil of points 42-47 (zeros when the pupils were not found). The body and hand lists are empty.
- `landmark_image`: the chosen set as dots, white on black by default. Black wherever no face was accepted.
- `overlay`: the input with the points drawn. The 620 points are coloured by kind: `lmk` blue, `skin` orange, `lid`
  green, `teeth_u` magenta, `teeth_l` pink, `iris` yellow; unsure points are dim by default.
- `face_mask`: the filled hull of the face points (or the chosen parts), 0..1.
- `landmarks_json`:
  - top level: `format` (`c2c-facepoints-1`), `model`, `weights_licence`, `status` (whether the model was found and
    where), `image_size`, `classes_620`, `std_note`;
  - `frames[]`: per frame `faces` (each with `box`, `median_std_px`, `face_size_px`, `confidence`, `points_620` as
    `[x, y, std]`, `set_68`, `set_98`, `iris`; plus `identity`, `screened_pose_points` and `raw` (unsmoothed points)
    when those controls are used), `diagnostics` (`box_source`, `tracked_boxes`, `finder_boxes`, `faces_inferred`,
    `faces_accepted`, `faces_rejected`, `rejected_reasons`, `points_finite_in_frame`, `points_drawn_on_overlay`, and
    `robust` when enhanced detection ran), `rejected_faces` (with `screening_keep_rejected_faces`) and `scene_cut`;
  - `diagnostics`: each image output's `shape`, `min`, `max`, `lit_pixels`, `lit_fraction`, plus
    `frames_without_face`, `class_warnings` and `frames_found_by_enhanced_detection`;
  - `notes`: plain-language warnings (see below);
  - when any control differs from its default: `controls` (the changed values), `source_size`, `processing_size`,
    `output_scale` and `coordinates` (output pixels; source pixels = output / `output_scale`).

## Common mistakes

- **Model not installed and `allow_download` off** (or **Never download models** on): the run stops with "Model
  'myned-ai/facepoints (...)' is not installed and downloading is off. Expected file 'model.safetensors' in folder:
  ...". Put the files there, or tick `allow_download`.
- **A download fails its SHA-256 check**: the partial file is removed; run again to retry.
- **Black `landmark_image`**: no face was accepted in that frame. `notes` says "no face in N of M frame(s) (first:
  ...: the face finder found no face)" or "every face was rejected: ...". Try `face_finder` = `whole image (already
  a face crop)` for a face crop, a lower `finder_confidence`, or a larger processing resolution. A small face on a
  large frame gives a mostly black but correct image: check `lit_fraction`.
- **Expecting 620 points in `pose_keypoint`**: it always holds the 68 + 2 pupils. `landmark_set` changes the drawing
  only; the full 620 set is in `landmarks_json`.
- **Connection lines with the 620 set**: none are drawn (the dense points have no published contour order);
  `notes` says so.
- **`mask_face_index` past the number of faces**: `face_mask` is empty in those frames and `notes` names them.
- **A colour that is not `#RRGGBB`** (with `appearance_scheme` = `custom`): "'...' is not a colour - write it as
  #RRGGBB (e.g. #FF8C00)".
- **Commercial work**: the facepoints weights are CC BY-NC 4.0, non-commercial use only.

Memory: the loaded model (about 416 MB of weights) is moved back to the CPU after each run and is released when
ComfyUI frees memory for another model (18899f3).
