# Changelog

All notable changes to ComfyUI-CustomNodePacks are documented here.

## Unreleased – 2026-10-09 (owner list A9: settings, link snapping, auto-connect, Vault, image batching)

Evidence: `docs/evidence/L2.30`, `L2.38` and `L2.40` in the work area (ComfyUI 0.36.0 / frontend 1.52.7 testbed,
classic canvas and Nodes 2.0, with the owner's own C2C setting values).

### Fixed

- **VAE Quality Decode works on the GPU with its defaults.** With "force fp32" on (the default) it changed the shared VAE's
  weights in place. ComfyUI 0.36 pins those weights in memory, so the decode failed with "CUDA error: invalid argument".
  It now decodes with its own fp32 copy of the VAE and leaves yours untouched.
- **Big image batches no longer need 10x their size in RAM.** The C2C tensor inspector, which records min / max /
  mean for every node output, built a huge index while doing it. A 2K x 24 frame batch (0.6 GB) briefly took 6 GB more
  after every node, and runs on 16 GB machines could run out of memory. Measured on a 4K chain load → grade → blur →
  save: 9.0 GB → 3.4 GB peak and 15.7 s → 9.9 s. The statistics are the same.
- **Load Video (C2C) and Save Video (C2C) hold far less in RAM.** They staged 16 to 64 frames at a time, which is up to
  1.6 GB at 4K. They now stage about 64 MB at a time. Measured server chain load → save: 2.6x → 1.7x the batch, at 2K
  and at 4K. The frames are identical, and the speed is unchanged.
- **Clear workflow no longer leaks old widget values into new nodes** (a ComfyUI bug, fixed on our side). After
  "Clear workflow", a node added in the same spot of the id sequence took a deleted node's value for every widget with
  the same name, for example a new sampler node came up with the old sampler. Seen in ComfyUI 1.52.7 with every C2C
  front-end switched off, in both renderers. Setting: Settings › C2C › Canvas › Clear workflow (Off = ComfyUI's own
  behaviour).
- **No more "inputEl is deprecated" warnings** from the Folder Incrementer (it now uses the widget's element).
- **C2C settings no longer disappear.** Lite mode (it switches on by itself when the browser draws without the GPU)
  hid 23 settings, including every overlay switch, because they were registered when an extension started up and
  Lite skips that. Every C2C setting is now declared up front: 144 in Full and in Lite.
- **A wire released anywhere on a node connects to the matching slot again**, with the snap preview while you
  hover. ComfyUI 1.52 turned every widget row into an input socket: over a widget of another type the wire got no
  preview and the release did nothing. This was most of the body of our widget-heavy nodes. Measured on all 315 C2C
  node types with an IMAGE input: 8 failed on the classic canvas, now 0. Setting: Settings › C2C › Canvas › Link
  snapping. A drop on a slot that fits behaves exactly as before.
- **Nodes 2.0: a wire released over the C2C wolf mark** (the empty-state logo in charts and editors) connects. An SVG
  is not an HTML element, and ComfyUI's Nodes 2.0 drop ignored it.
- **Auto-connect wires a new node to the nearest node on its left**, by exact type: every free input whose type has
  exactly one matching output there. It used to wire the first input to whichever node was clicked last. Nothing
  is wired when two nodes are about equally near, when the node arrives with a paste, or when ComfyUI already wired
  it (a node created from a dragged wire). One Ctrl+Z removes the wires.
- **C2C Vault asks for its password on the node.** A Locked vault shows a password field, Unlock, and "Lock again";
  Enter unlocks. It shows the server's real session state, also after a reload. The password is never saved in
  the workflow or the queued prompt, and the field clears after unlocking. A Sealed vault shows the field for
  "Open for editing". The vault id is hidden, because it means nothing to a person. Double-click opens an
  unlocked vault for editing.
- **C2C Vault protects a whole workflow** (Settings › C2C › Vault › Scope: Selected nodes / Whole workflow).
  - Locked: "C2C Vault: Save workflow encrypted" writes the file as ciphertext. Opening it, by drag-drop or from
    the workflow list, asks for the password, and without it nothing loads. Saving again keeps it encrypted. While
    it is open, Export is refused, and the browser's tab-restore keeps only the ciphertext.
  - Sealed: "C2C Vault: Seal whole workflow" puts every node except the outputs into one sealed vault. The workflow
    runs without a password; the Preview / Save nodes stay outside so the results still show.
- **Load Video (C2C) and OmniScale Load Video preview ProRes, DNxHR, HEVC 10-bit and FFV1.** They played the original
  file in the browser, which cannot decode those formats. The preview is now transcoded on the server.
- **Load Video: "select every nth"** keeps the audio for the whole clip and reports the real frame rate (fps / n),
  as VHS does. The audio used to stop halfway.
- **Nodes 2.0: double-clicking a slot** (Get/Set, Suggest) works; it did nothing there.
- **C2C video reads that stop part-way through a file release their decoder cleanly.** A frame cap, every-nth or
  random read left about 17 frames in flight in the decoder's threads, and freeing it like that hung once in the
  test suite (inside FFmpeg). The decoder is now drained first; this costs nothing measurable on 1080p reads.
- **Mask Ops with ViTMatte on the CPU uses a third of the memory.** Tiles now run one at a time on the CPU. Measured
  with the real ViTMatte weights on a 1536×1152 hair plate at the node's defaults: peak RAM 7.7 GB → 2.5 GB,
  9.3 s → 5.8 s, the same matte. The GPU keeps the "tiles per forward pass" setting. The same measurement shows the
  tiled matte is as good as one full-frame pass (within 3 %), and downscaling the frame instead is far worse.
- **VAE Quality Decode: "apply ACES" no longer washes the picture out.** It tone-mapped the decoded (sRGB-encoded)
  image as if it were linear light and then encoded it again, so mid-grey came out at 0.79 instead of 0.55. It now
  matches C2C ACES Tonemap with source sRGB. Workflows that had it on will look darker in the mid-tones, as intended.
- **Image batching is findable.** The setting is called "Image batching" (Settings › C2C › Image batching, also
  found by searching the Settings dialog): Off / Internal (default) / Universal. Per node type: right-click a
  node › Image batching for this node type.

### Added

- **Save Video (C2C) stream save: wire a video latent and its VAE instead of images.** The clip is decoded in order,
  keeping the VAE's causal state, and each frame goes straight into the file, so the full frame batch is never held in
  RAM. The frames are identical to decoding first and then saving (checked on Wan 2.1, Wan 2.2, MiniMax H3 and LTX).
  Measured with Wan 2.1 at 960x544 x 81 frames: 0.48 GB peak RAM instead of 1.57 GB, and the peak stays flat as clips
  get longer. At 2K on an 8 GB card, tile mode Auto (default) / Off / Manual decodes 512 px tiles, each over the whole
  clip and blended on disk: 2048x1080 x 81 frames used 0.79 GB of RAM instead of 4.62 GB, with an identical file,
  in 250 s instead of 201 s. A VAE that cannot be streamed exactly is decoded whole, and the node says so.
- **VAE Quality Decode tiles by itself when the GPU is too small: tile mode Auto (default) / Off / Manual.** On an
  8 GB card a 2K Wan decode took 502 s untiled, because Windows moved the overflow into system RAM. Tiled it took
  37 s with the same picture. Video is tiled in space only: splitting a video VAE in time cost 21 dB. Saved workflows
  that set a tile size open as Manual.
- **A notice when a run spills GPU memory into system RAM** (Windows' NVIDIA driver does this instead of reporting
  out-of-memory). It tells you the driver setting that fixes it. Shown once per session; Settings › C2C ›
  Performance › GPU memory.
- **SeedVR2 preview.** The SeedVR2 Video Upscaler node (numz pack, unchanged) shows each decoded batch while it runs, with
  a before/after wipe. When it finishes, a frame slider steps through every frame, and Difference and Alpha views show
  what changed. Settings › C2C › Video › SeedVR2 preview.
- **Save Video (C2C)** (`🐺 C2C/…/Video`), the pro writer next to VHS Video Combine. It writes MP4 H.264 / H.265
  10-bit, MOV ProRes 422 / HQ / 4444 / 4444 XQ and DNxHR HQ / HQX / 444, lossless MKV FFV1, WebM VP9, GIF,
  animated WebP, and 16-bit PNG / half and float EXR sequences. Alpha is kept where the format has it, audio is
  muxed in (a WAV next to a sequence), and the frames stream to the encoder in chunks sized to free RAM.
  - Optional OCIO colour-space conversion on write (for example sRGB → ACEScg for an EXR plate), from `$OCIO` or the
    built-in ACES studio config.
  - The node plays what it saved, with a scrub timeline and format chips. Formats a browser cannot play (ProRes,
    DNxHR, FFV1, sequences) get a small H.264 preview.
  - Naming: ComfyUI counter (`name_00001`, never overwrites) or Folder Version (wire `subfolder` from the Folder
    Version Incrementer; an existing version is never overwritten), per node or from Settings › C2C › Video ›
    Save Video.
  - Measured round trip per format (PSNR, alpha, audio length): `docs/evidence/L7.56` in the work area.
    For example, ProRes 4444 gives 65.8 dB with alpha, and FFV1, 16-bit PNG and float EXR are exact.

## Unreleased – 2026-10-08 (owner list A9: graph-damaging bugs, settings, Vault)

Evidence for every item: `docs/evidence/L2.15`, `L2.23` and `L2.24` in the work area (Playwright runs on a
ComfyUI 0.36.0 / frontend 1.52.7 testbed, classic canvas and Nodes 2.0).

### Changed defaults (each one is still a setting)

- **Smart guides while dragging: off.** Settings › C2C › Canvas › Align turns them back on.
- **Floating ports: off.** With them on, a wire to a node above or below ends at a dot on that node's edge,
  which read as a stale line. Settings › C2C › Canvas › Floating ports turns them back on.
- **Ctrl+R reloads the page again.** Resetting nodes to their defaults with Ctrl+R is now opt-in
  (Settings › C2C › Reset to Defaults). Right-click a node › "Reset to ORIGINAL" still works.
- **Double-click a slot** is one setting: Nothing, Get/Set variable (default), or Suggest (experimental).

### Fixed

- **Ctrl+Z inside a C2C editor no longer undoes the whole workflow.** ComfyUI catches Ctrl+Z before any editor
  sees it, so one press undid the editor stroke and reloaded the entire graph. This affected the points/box
  editor, the spline editor and tracker, the video mask editor and Face Controller 3D. Ctrl+Z on the canvas
  still undoes the graph.
- **Tidy layout** runs only when you ask: on the selected nodes, or on the whole graph after a confirmation. It
  keeps the graph where it was, moves each group as one block, never moves pinned nodes, and is one undo step.
  It used to send every node to the origin and stack the graph in a single column.
- **Alt-drag duplicate** no longer drags the original along.
- **Dragging a group** no longer picks up nodes that merely overlap its new position.
- **Insert Reroute** (canvas menu on a link) works again.
- **Auto-connect** never wires nodes while a workflow loads, an undo replays, a paste runs or an Alt-drag copies.
- **The command palette** closes on Escape.
- **Get/Set by slot double-click** works at every zoom level and inside subgraphs. Its module had failed to load.
- **Reset to defaults:**
  - Settings › C2C › Reset to Defaults has a button that resets the C2C settings themselves. API keys are kept.
  - Resetting all nodes asks first and is one undo step.
- **C2C Vault works.**
  - Lock and Seal used to hang after the password.
  - A Locked vault could never run, and Unlock always failed.
  - Wires went into the wrong inputs.
  - Core V3 nodes returned the wrong value inside a vault.
  - The creator's session now opens on lock. "Lock session" takes effect on the next queue. The node shows only
    its real sockets.

## [1.14.1] – 2026-05-04

### Fixed

- **Hidden multiline-string widgets no longer leak as giant textareas on top
  of editor nodes.** On modern ComfyUI, multiline `STRING` widgets are
  backed by a real `<textarea>` DOM element parented to a
  `div.dom-widget` wrapper. Setting `widget.type = "hidden"` was no
  longer enough to hide them — the DOM textarea kept rendering on top of
  the canvas and made `PointsMaskEditor`, `SplineMaskEditorMEC`,
  `MECAdvancedPaintCanvas`, and `InpaintCompositeMEC` look like a wall of
  empty text fields. The hide path now also sets `element.style.display
  = "none"` and collapses the `dom-widget` wrapper so the visible UI
  matches what those widgets are supposed to be: invisible internal state.
  Affected widgets:
  - `PointsMaskEditor.editor_data`
  - `SplineMaskEditorMEC.spline_data` / `mask_color` / `mask_opacity`
  - `MECAdvancedPaintCanvas.canvas_data`
  - `InpaintCompositeMEC` mode-conditional widgets (`blend_mode_override`,
    `color_match`, `upscale_method`, `feather_edges`, `feather_radius`)
- **Verified end‐to‐end via Playwright**: created PointsMaskEditor and
  SplineMaskEditorMEC, captured pre/post screenshots, and exercised the
  interactive editors. Points editor accepts left/right click for
  positive/negative points and Ctrl-drag for bboxes — `editor_data`
  payload contains real coordinates with distinct `label:1`/`label:0`
  values (no hardcoded zeros). Spline editor accepts 4 control points
  and renders a smooth catmull-rom curve closed loop, not a polygon.

## [1.14.0] – 2026-05-04

### Fixed

- **SAM Multi-Mask Picker — render real per-mask thumbnails**: the JS
  widget previously called an empty `buildMaskThumbnails()` stub and then
  drew a flat red `fillRect` over all three thumbnails identically, so
  every candidate looked the same regardless of what SAM produced. The
  Python node now serializes the three candidate masks as grayscale PNGs
  to ComfyUI's temp directory (stable input-hash filenames) and ships
  them on the UI payload as `mask_thumbs`. The JS now loads each PNG via
  `/view?...&type=temp` into `this.maskImages[i]` and composites the
  real mask onto each thumbnail using an offscreen canvas with
  `globalCompositeOperation="source-in"` plus a per-thumb tint.
- **SAM Multi-Mask Picker — widget no longer renders ~600 px below the
  node body**: the `addCustomWidget` registration was forwarding
  `node.pos[0]` and `node.pos[1] + widgetY` into the inner draw/mouse
  handlers, but LiteGraph already translates the canvas context to the
  node origin before invoking widget callbacks — the result was a
  doubled offset that pushed the picker off the bottom of the node.
  Registration now passes widget-local coordinates (`0, widgetY` for
  draw; `pos[0], pos[1]` for mouse), so the picker lives inside the
  node body where it belongs.
- **SAM Multi-Mask Picker — stop force-rerunning SAM on every queue**:
  removed `IS_CHANGED → NaN`, which was unconditionally invalidating
  the cache and re-running SAM on identical inputs. Standard cache
  semantics now apply.

## [1.13.1] – 2026-05-04

### Fixed

- **Latest-ComfyUI compatibility**: removed deprecated `disable_noise=False`
  kwarg from two `comfy.sample.sample_custom(…)` call sites in
  `MECBuilderSampler` (mec_paint_suite.py). The argument was dropped from
  `comfy.sample.sample_custom`'s signature in recent ComfyUI builds and
  was raising `TypeError: unexpected keyword argument 'disable_noise'`
  at sample time.
- Audited all 41 nodes against the current ComfyUI runtime: full pack
  imports cleanly, no other deprecated APIs in use
  (`prepare_noise`, `sampler_object`, `calculate_sigmas`,
  `OUTPUT_NODE`, `IS_CHANGED` all match current signatures).

### Notes

- ComfyUI's V3 node API (`io.ComfyNode` / `define_schema` /
  `MatchType` / `Autogrow` / `DynamicCombo`) is **opt-in and additive**
  — the V1 API used throughout this pack is fully supported and not
  scheduled for removal. No mass migration required.

## [1.12.0] – 2026-05-04

### Added

#### Inpaint Composite (MEC) — unified Stitch + Paste Back

- New `InpaintCompositeMEC` node merges the previous two-node split
  (`InpaintStitchProMEC` + `InpaintPasteBackMEC`) behind a single `mode`
  dropdown:
  - `stitch_pro` — advanced blend pipeline (edge-aware, Laplacian
    pyramid, frequency, gaussian) with optional Reinhard color match.
  - `paste_back` — clean resize + paste with optional Gaussian-feathered
    edges. Faster, deterministic, no blend pipeline.
- Unified output signature `(IMAGE, MASK, STRING)`. In `paste_back` mode
  the MASK is the paste rectangle (feathered if enabled) so downstream
  nodes always receive a usable mask.
- New `js/inpaint_composite.js` extension hides parameters that don't
  apply to the selected mode (clean UI without losing widget state).
- Internally delegates to the existing Stitch and PasteBack classes — no
  duplicated logic, no behavioural drift.

### Changed

- Legacy `InpaintStitchProMEC` and `InpaintPasteBackMEC` are still
  registered for backward compatibility, but their display labels now
  read "— legacy (MEC)" to steer new workflows toward the unified node.
- README updated with a new "Inpaint Composite" section, mode-vs-mode
  guidance, and an updated reference table.

## [1.11.0] – 2026-05-08

### Added

#### Ronin-gimbal cinema stabilization (Inpaint Crop / Stitch / Paste-Back)

- New `video_stab_strength` (0–1) widget on `InpaintCropProMEC` activates a
  two-stage filter cascade on the per-frame mask centroid trajectory:
  1. **EMA low-pass** at `4 Hz × strength + 0.05` cutoff to kill HF jitter
     while preserving subject drift.
  2. **Critically-damped spring** (`k = 8·strength + 0.5`,
     `damping = 2·√k`) integrated with semi-implicit Euler for inertia
     and a tiny natural overshoot.
  - **Pre-warm pass**: the spring is run once on the first 12 frames, the
    final `(pos, vel)` state is captured, and the real pass is initialised
    from it — eliminating the snap-to-first-frame artefact that plagues
    naive spring filters.
- **99th-percentile center-displacement sizing** (not full envelope): crop
  width = P99(detected widths) + (P99(cx) − P1(cx)) + 2·padding. Matches
  cinema operator framing — tight on the subject, ignoring outliers.
- New `video_stab_fps` (FPS for filter time-base) and `video_stab_padding`
  widgets.
- `stitch_data` schema bumped to **v3** with new `frame_offsets` field
  (per-frame `(x, y)` paste positions in canvas-space). v2 remains fully
  supported by `InpaintStitchProMEC` and `InpaintPasteBackMEC`.
- `InpaintStitchProMEC._stitch_v2` and `InpaintPasteBackMEC.paste_back`
  rewritten with a per-frame paste loop that consumes `frame_offsets`
  when present (v3) and falls back to the original single-position paste
  for v2 round-trips.
- New helpers in `nodes/stabilization_utils.py`: `_median_filter_1d`,
  `_ema_lowpass`, `_spring_filter`, `ronin_filter_1d`,
  `compute_ronin_bbox_trajectory`.

### Changed

- `InpaintCropProMEC`: `crop_for_inpaint` signature gains
  `video_stab_strength`, `video_stab_fps`, `video_stab_padding` (all
  defaulted; backward compatible).
- `Step 8.6 crop_mask` now emits a per-frame rectangle stack when Ronin
  is active.
- `Step 10` info string reports the active stabilization mode.

### Validated

- Numerical: 11.13 px raw frame-to-frame jitter → 1.34 / 1.64 / 1.76 px at
  strength 0.3 / 0.7 / 1.0 (≥85% reduction); init-snap ≤ 1.2 px (warmup
  works).
- Round-trip: identity inpaint + paste-back is bit-perfect (mean abs
  diff = 0.0000) on 12-frame jittery video.
- Static-subject test: 0 px center spread on stationary subject with
  ±1 px detector noise.

## [1.9.0] – 2026-04-29

### Added

#### MEC Paint Suite (`MEC/Paint`)

- **MECAdvancedPaintCanvas** – interactive paint canvas (DOM widget) with
  Nuke-style procedural mask math: raw alpha → `mask_hardness` core threshold
  → `mask_expansion` morphological dilate/erode → Gaussian
  `mask_blur_radius` lerped against the hard mask by `mask_blur_strength`.
  Outputs the alpha-composited painted image plus the processed mask.
- **MECContextInpainter** – smart blend-back of an inpainted image. Per-channel
  Reinhard colour match, CIE LAB lightness rescue when the inpainted region
  is >5% darker, optional differential-diffusion preservation weight from
  `|orig − inpaint|`, plus `[SEP]` / `[SKIP]` / `[ASC]` / `[DSC]` wildcard
  parsing across connected mask regions.
- **MECToneRefiner** – percentile-based black/white-point tone curve
  (smoothstep), gray-world colour balance, optional bicubic upscale and
  centre-focus fake DOF driven by `ai_dof_focus_depth`.
- **MECBuilderSampler** – KSampler with adaptive CFG (`Constant` / `Linear` /
  `Ease Down` via `set_model_sampler_cfg_function` keyed on per-step sigma)
  and an optional 2-step self-correction polish pass at low denoise.
- **`js/mec_advanced_paint.js`** – RGBA canvas widget with L-paint / R-erase,
  linear-interpolated stamping (no dotted strokes on fast drags), brush
  cursor mirroring size + hardness, mouse-wheel brush sizing, base64-PNG
  serialisation into a hidden `canvas_data` widget so the Python node can
  decode the user's drawing.

## [1.8.0] – 2026-04-29

### Fixed

- **FolderIncrementer (JS)** – removed the `isInputLoader` block that was
  preventing filenames from `LoadImage` / `LoadVideo` / `VHS_LoadVideo`
  from being read. Loaders are now exactly the nodes whose filenames
  feed the versioning folder name.
- **FolderIncrementer (JS)** – added `source_choice` routing
  (`auto`/`image`/`video`) and `trigger_image` / `trigger_video` inputs.
  In `auto` mode video is preferred over image when both are connected.
  Re-syncs on disconnect and on `source_choice` change.
- **Points & BBox Mask Editor** – fixed canvas auto-zoom-in/out jitter
  on first image load and on graph execution. The double-pass `fitView`
  (rAF + 300 ms) was measuring a stale bounding rect from a pre-reflow
  layout, then snapping to the correct rect. Now a single deferred fit
  (two animation frames) runs after LiteGraph reflow has settled.
- **Points & BBox Mask Editor** – `_hasAutoFitted` is no longer reset on
  reference-image disconnect, so brief upstream disconnects keep the
  user's zoom/pan instead of crazy-refitting on reconnect.
- **Spline Mask Editor** – wheel-zoom is hard-clamped (80 px ↔ 8 000 px)
  to prevent the preview bounds from collapsing or exploding after a
  few scroll events. Auto-fit only re-runs when the underlying image
  dimensions change, not on every preview update.

### Added

#### VAE & latent diagnostics

- **VAEMergeMEC** – merge two (or three) VAEs with 8 algorithms
  (weighted_sum, add_difference, tensor_sum, triple_sum, slerp, dare_ties,
  block_swap, clamp_interp). Per-block alpha overrides via JSON or
  comma list, brightness/contrast post-tuning on `decoder.conv_out`,
  CPU-side merge in float32, architecture detection (sd1x / sdxl / flux /
  mochi). Returns the merged VAE plus a JSON info report.
- **VAELatentInspectorMEC** – per-channel min/max/mean/std/abs_mean
  statistics, NaN/Inf counts, and a `verdict` of corrupt / saturated /
  low_contrast / healthy. Optional `fail_on_corrupt` to raise hard.
- **VAESimilarityAnalyserMEC** – cosine similarity between two VAEs
  globally and per block, with optional per-tensor breakdown.
- **VAEBlockInspectorMEC** – per-block weight statistics for any VAE
  (mean / std / abs_mean / count).

#### Color science (`MaskEditControl/Color`)

- **ColorSpaceConvertMEC** – sRGB ↔ Linear ↔ Rec.709 ↔ ACEScg via
  built-in matrices and OETFs. No PyOpenColorIO required.
- **LUTApplyMEC** – Adobe `.cube` LUT loader (1D and 3D) with trilinear
  interpolation in pure torch and a strength blend.
- **ExposureGradeMEC** – exposure (stops), white-balance (temp / tint),
  and contrast around a configurable mid-grey pivot. Operates in linear
  by default with sRGB encode/decode round-trip.

#### EXR I/O & metadata (`MaskEditControl/IO` / `Metadata`)

- **LoadEXRMEC**, **SaveEXRMEC** – EXR read/write with OpenEXR-first,
  imageio-fallback, and final 16-bit TIFF fallback. Half-float by default.
- **EXRMetadataReaderMEC** – pure-python EXR header parser when OpenEXR
  isn't installed; surfaces width/height/channels/compression.
- **MetadataWriterMEC** – write/merge JSON sidecars next to image batches.
- **ShotMetadataNodeMEC** – read shot.json descriptor (show / shot / task /
  frame_in / frame_out / fps).
- **FrameRangeRouterMEC** – slice IMAGE/MASK batches by `[start:end:step]`
  with negative-index support.
- **BatchVersionManagerMEC** – `<root>/<show>/<shot>/<task>/v###/` layout
  with safe sanitization and atomic version reservation via lockfile.

#### Render passes & geometry (`MaskEditControl/Render` / `Geometry`)

- **MergeRenderPassesMEC** – composite beauty + diffuse / specular /
  emission / AO with independent gains.
- **DepthOfFieldMaskMEC** – depth → CoC mask with focus_distance and
  aperture controls; emits both CoC and in-focus masks.
- **DepthWarpMEC** – horizontal parallax warp driven by a depth pass
  (cheap stereo synthesis).
- **NormalToCurvatureMEC** – curvature mask from a tangent-space normal
  pass via central-difference divergence.
- **PositionPassSplitterMEC** – split a world-position pass into per-axis
  X / Y / Z masks (auto- or manually-ranged).

#### Plate tools (`MaskEditControl/PlateTools`)

- **GrainMatchMEC** – extract grain (reference − denoise(reference)) and
  re-apply it to a target plate, with seeded per-frame sampling.
- **PlateStabilizerMEC** – ORB+RANSAC affine stabilization (cv2) with
  FFT-translation fallback when cv2 isn't available.
- **CleanPlateExtractorMEC** – pixelwise median across a batch with
  optional mask exclusion of foreground samples.
- **DifferenceMatteMEC** – L1/L2 difference matte with threshold and
  softness.

#### Diagnostics

- **TemporalConsistencyCheckerMEC** – flicker / instability score across
  a video batch using mask_iou, pixel_diff, or Farneback flow_warp.
- **ModelMetadataExtractorMEC** – inspect safetensors / checkpoint files
  without ever unpickling. SHA256 fingerprint on size + first/last 1MB,
  bounded `pickletools.dis` for legacy pickles.

### Changed

- **FolderIncrementerMEC** – added `folder_name_override` STRING and
  `reserve_version` BOOLEAN. Names are sanitized against Windows reserved
  names and illegal characters. Outputs use POSIX-style paths.
- **SAMModelLoaderMEC** – added `original_device` tracking and module
  helpers (`move_to_inference_device`, `restore_device`) for safe
  CPU↔GPU shuffling on offload-enabled wrappers.
- **MattingNodeMEC** – `auto_download` switch + four-step fallback chain:
  vitmatte_base → vitmatte_small → cv2 guided filter → torch Gaussian.
- **UnifiedSegmentationNode** – added `force_mode` (auto / image / video).

## [1.7.0] – 2026-04-02

### Added

- **DrawShapeMEC** – unified 12-shape drawing node with a single dropdown.
  All parameters exposed as named inputs with descriptive tooltips — no
  more raw JSON editing. Replaces the 5 legacy per-shape wrapper nodes.
- **SplineMaskEditorMEC (JS rewrite)** – complete rewrite of the spline
  editor following Olm SplineMask patterns:
  - Normalized [0,1] coordinates (resolution-independent)
  - Segment insertion via Ctrl+click near curve
  - Close path by clicking first point (highlighted orange)
  - Right-click context menu (Delete, Open/Close, Smooth/Sharp)
  - Zoom-relative point sizes and fonts
  - Property-based persistence with backward-compatible deserialization
  - Status bar with keyboard hints

### Changed

- **DrawCircleMEC, DrawRectangleMEC, DrawEllipseMEC, DrawPolygonMEC,
  DrawLineMEC** – deprecated, now thin wrappers around DrawShapeMEC.
  Kept for backward compatibility with existing workflows.
- Node count updated: 38 → 47 (44 MEC + 3 FolderIncrementer).
- README, docs, and project structure updated for all new nodes.

## [1.6.0] – 2026-03-19

### Added

- **SplineMaskEditorMEC** – interactive spline mask drawing with
  Catmull-Rom, Bezier, and polyline modes. Outputs mask, SAM-compatible
  coords, and SPLINE_DATA for downstream chaining.
- **MotionMaskTrackerMEC** – per-frame motion detection with 4 methods
  (pixel diff, optical flow, background subtraction, histogram diff),
  camera stabilization (homography/affine/translation), and post-processing.
- **BBoxSmooth** – smooth bounding-box sequences across video frames
  using moving-average or exponential smoothing.
- **stabilization_utils.py** – shared camera stabilization helpers for
  motion tracker and inpaint suite.

### Changed

- **InpaintSuite** – uses stabilization_utils for camera-stable cropping.
- **BBoxNodes** – added BBoxSmooth to the 5 existing BBox nodes.

## [1.5.0] – 2025-07-17

### Added

- **InpaintCropProMEC – Canvas Expansion** – crops near image edges now
  extend beyond bounds via `F.pad(mode="replicate")`, producing perfectly
  centered crops everywhere. Ported from lquesada/ComfyUI-Inpaint-CropAndStitch.
- **InpaintCropProMEC – optional_context_mask** – new optional input: union
  of context mask bbox with crop bbox to include surrounding context.
- **InpaintCropProMEC – Iterative Hole-Filling** – `_fill_mask_holes` rewritten
  with 14-threshold iterative approach using scipy `binary_closing` +
  `binary_fill_holes` (with pure-torch fallback). Preserves gradient values
  in soft masks.
- **Stitch Data v2 Format** – new coordinate system: `ctc_x/y/w/h`
  (crop-to-canvas) + `cto_x/y/w/h` (canvas-to-original) for perfect
  reversal of canvas expansion in stitching.

### Changed

- **InpaintStitchProMEC** – v1/v2 dispatch: v2 uses canvas coordinates
  for seamless compositing of expanded crops; v1 fallback preserved for
  backward compatibility with existing workflows.
- **InpaintPasteBackMEC** – updated for v2 stitch format with canvas
  coordinate handling; v1 fallback preserved.
- **Points Editor (JS)** – `computeSize` override prevents LiteGraph
  relayout jitter; widget height computed from actual other-widget heights
  instead of magic number; locked dimensions after image load.
- **Image Comparer (JS)** – `computeSize` override prevents resize jitter
  after image comparison loads.

### Fixed

- **unified_segmentation.py** – marked as DEPRECATED (dead module superseded
  by unified_segmentation_node.py + model_manager.py). Prevents confusion
  from divergent MODEL_REGISTRY.

## [1.4.0] – 2025-07-16

### Added

- **InpaintCropProMEC** – 3 new mask preprocessing inputs from original
  InpaintCropAndStitch: `mask_invert` (flip inpaint/keep regions),
  `mask_fill_holes` (flood-fill enclosed gaps), `mask_hipass_filter`
  (threshold out near-transparent noise).
- **MaskBatchManager** – 2 new temporal operations for video workflows:
  `smooth_temporal` (Gaussian blur along time axis to reduce flicker) and
  `reduce_flicker` (median filter across frames to remove outliers).
- **BBoxSmooth (MEC)** – new node to smooth bounding-box sequences across
  video frames using moving-average or exponential smoothing, eliminating
  jitter in tracked crops.

### Changed

- **InpaintCropProMEC** – crop centering rewrite: bbox now expands
  symmetrically around the mask center, then clamps to image bounds. Fixes
  asymmetric off-center crops when the mask is near image edges.
- **Points Mask Editor (JS)** – complete toolbar UI overhaul: brighter
  button colors, 11 px font, 36 px toolbar height, `textBaseline="middle"`
  vertical centering, 120 ms active-state flash on click, wider separators,
  higher-opacity pills.
- **Points Mask Editor (Python)** – bbox rendering rewritten to use soft
  Gaussian-edged brushes: positive bboxes use `torch.max(mask, brush)`
  (additive, preserves soft point edges), negative bboxes use multiplicative
  erase. Replaces hard `mask = 1.0` overwrite that destroyed soft blending.

### Fixed

- **utils.py** – `multi_scale_guided_refine` and `color_aware_refine` now
  handle 2D grayscale input arrays (was IndexError when `img_np.ndim == 2`).
- **model_manager.py** – float8 dtype comparison guarded with
  `hasattr(torch, "float8_e4m3fn")` to prevent AttributeError on
  PyTorch < 2.1.
- **Points Editor (JS)** – fixed 4 memory leaks: `requestAnimationFrame` now
  cancelled in `onRemoved`, resize debounce timer cleared, keyboard event
  listener properly removed, complete cleanup chain on node deletion.
- **InpaintCropProMEC** – `_resize_lanczos` fix for single-channel masks:
  squeeze channel dim to 2D for PIL, restore after (was crashing with
  `Cannot handle this data type: (1, 1, 1), |u1`).

## [1.3.1] – 2025-07-15

### Added

- **InpaintPasteBackMEC** – new lightweight node to paste inpainted crop back
  onto the original image using stitch_data, with optional Gaussian-feathered
  alpha blending at the crop boundary.
- **InpaintCropProMEC** – `downscale_method` and `upscale_method` inputs
  with 5 interpolation modes: lanczos (PIL-based, highest quality), bicubic,
  bilinear, nearest-exact, area. Upscale method is stored in stitch_data and
  automatically used by InpaintStitchProMEC.

### Changed

- **InpaintCropProMEC** – `padding_multiple` now enforces step=2, min=2 to
  guarantee even-valued padding (required by many tiled/diffusion models).
- **Image Comparer (JS)** – mode labels now use Unicode icons
  (◧ Compare, ⊕ Overlay, ≠ Diff); labels fade during drag; divider grip
  uses a dot-grid pattern; overlay scrubber upgraded to rounded bar + circular
  handle.

### Fixed

- **InpaintStitchProMEC** – now reads `upscale_method` from stitch_data
  (defaulting to lanczos) instead of always using bilinear.

## [1.3.0] – 2025-07-14

### Added

- **ImageComparerMEC** – new interactive before/after comparison node with three
  view modes: drag-slider, adjustable-opacity overlay, and amplified difference
  heatmap. Rendered entirely in-node via a DOM canvas widget.
- **MaskDrawFrame** – 7 new shapes: triangle, star, diamond, cross,
  rounded_rectangle, heart, arrow (total: 12 shapes).
- **MaskDrawFrame** – `rotation` parameter (−360 ° to +360 °) applies to every
  shape including the original five.
- **SAMMaskGeneratorMEC** – `negative_text_prompt` input: describe what to
  exclude; GroundingDINO detects the region and injects negative points into SAM
  inference.
- **InpaintCropProMEC** – `downscale_factor` (0.25–1.0) to shrink the crop
  before sending to the inpainting model and automatically upscale on stitch.
- **InpaintCropProMEC** – `mask_blur` / `mask_grow` pre-processing: blur and
  morphologically dilate/erode the mask before computing the crop region.
- **InpaintCropProMEC** – `aspect_ratio` selector with 12 presets (1:1, 4:3,
  16:9, …) plus custom ratio; crop region is adjusted to match the chosen AR.
- **InpaintCropProMEC** – 6th output `crop_mask`: binary mask showing the
  cropped region in original image space.

### Fixed

- **Points/BBox Editor (JS)** – eliminated canvas jitter caused by
  `updateEditorSize()` re-entrantly calling `node.setSize()` in a
  ResizeObserver loop. Added 50 ms debounce, integer-snap for pixel positions,
  and a re-entrance guard.
- **Points/BBox Editor (JS)** – toolbar separators and pill buttons now render
  at integer pixel positions, removing sub-pixel aliasing artefacts.

### Changed

- **Model Manager** – `.safetensors` format is now preferred over `.pt`/`.pth`/
  `.bin` when a safetensors variant exists locally. SAM3 (`sam3.pt`) is
  explicitly excluded from this mapping.
- **InpaintCropProMEC** – crop dimensions are now snapped up to the nearest
  `padding_multiple` to avoid size mismatches with tiled models.

## [1.2.6] – 2025-07-13

### Fixed

- GitHub Actions publish workflow: added `FORCE_JAVASCRIPT_ACTIONS_TO_NODE24`
  env var and fork guard.
- Added missing `MIT-License` file required by PyPI metadata.

### Changed

- README overview table expanded from 2 to 4 packs.
