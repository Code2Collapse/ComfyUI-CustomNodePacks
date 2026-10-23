# Save Video (C2C) v2, built-in handles, Video Comparer v2: design (ledger L9.40)

Written before the code (A7). Owner request and answers: `docs/evidence/L9.40/owner_answers_2026-10-23.md`
(workspace). Everything here extends the existing nodes; no node is added or renamed.

## Goals

1. Save Video gets Load Video's player for the written file, an export range set in that player, a preview run before
   the real encode, and an A/B view of the saved file against the frames that came in.
2. Format is chosen with five controls - container, codec, profile, bit depth, chroma - each filtered by the ones
   before it, plus colour tags (primaries, transfer, range) that are written into the file AND used for the conversion.
3. naming = Folder Version Incrementer or default fills the version path by itself: from a Folder Version Incrementer
   node in the graph when there is one (no wire), otherwise with the same incrementer logic built in. The node shows
   the path it will write before you queue.
4. Handles live inside the video nodes: Load Video adds them (fit / add, model frame grid, audio kept in step), the
   count travels with `video_info`, Save Video trims exactly that count; Save also trims manually or not at all.
5. Video Comparer gets a real synced A/B player, zoom / pan / loupe, per-frame metrics on the timeline, A/B/C/D,
   flicker, more wipe shapes, diff gain, and previews of image tensors without a re-queue.

## Constraints

- Saved workflows keep working (standing order 7). ComfyUI restores widget values BY POSITION, so:
  - every new input is appended at the end of `optional` (the generic tail-pad in `js/widget_migration.js` fills
    missing values with defaults);
  - the old `format` combo stays where it is, now meaning a legacy label or `custom` (use the five controls). A saved
    workflow whose `format` is a legacy label keeps encoding exactly as before; the front-end converts it to the
    five controls when the node is loaded and sets `format` to `custom`;
  - API prompts that only send the old inputs still validate: every new input is optional with a default.
- Front-end: both renderers, inside subgraphs, no polling while idle, only the node's own DOM and widgets.
- Never two copies of a thing: the Save player IS `js/c2c_video_player.js` (Load Video's); the handle arithmetic IS
  `nodes/av_handles.py` (+ `_frame_grid.py`); the version logic IS `folder_incrementer.py`.

## Format matrix (`nodes/c2c_video/save_matrix.py`)

`build_spec(container, codec, profile, bit_depth, chroma, alpha) -> FormatSpec` replaces the fixed table for new
choices; the old 21 labels map onto it (`legacy_choice(label)`), so both paths produce the same `FormatSpec` the
encoders already consume (PyAV path, ffmpeg path, OIIO sequences). The matrix lists only encoders this machine has
(probed once, PyAV + the ffmpeg CLI): unavailable choices are shown with the reason, never silently replaced.

| container | codecs |
|---|---|
| MP4 | H.264, H.265, AV1 (SVT), H.264 / H.265 / AV1 NVENC |
| MOV | ProRes, DNxHR, CineForm, H.264, H.265, PNG (alpha), QuickTime Animation (alpha), Uncompressed 10-bit 4:2:2 |
| MKV | FFV1, UT Video, H.264, H.265, AV1 (SVT), ProRes |
| WebM | VP9, AV1 (SVT) |
| GIF / WebP | animated GIF / WebP |
| Image sequence | PNG, TIFF, DPX, EXR, JPEG |

Profiles follow the codec (H.264 baseline/main/high/high10/high422/high444; H.265 main/main10/main12/main422-10/
main444-8/main444-10; ProRes proxy/LT/422/HQ/4444/4444 XQ; DNxHR LB/SQ/HQ/HQX/444; VP9 0-3; EXR half/float ×
compression; DPX 10/12/16 ...). Bit depth and chroma lists are what that profile allows; a choice that the next
control no longer allows snaps to the nearest allowed value and the node says so.

Colour tags: `primaries` (auto, BT.709, BT.2020, Display P3, BT.601), `transfer` (auto, BT.709, sRGB, linear, PQ,
HLG, gamma 2.2), `range` (auto, limited, full). auto = today's behaviour (derived from colorspace_out). The YUV matrix
follows the primaries (BT.709 / BT.2020 NCL / BT.601) and the range drives the conversion, so the file says exactly
what was done to it. Tags are refused with a sentence where they cannot apply (RGB codecs have no matrix).

## Naming (`nodes/c2c_video/save_naming.py`)

- `naming`: default, ComfyUI counter, Folder Version Incrementer (the label "Folder Version" of old workflows maps
  to it). default = the C2C setting `c2c.saveVideo.naming`, whose default becomes Folder Version Incrementer.
- Auto-detect: the hidden PROMPT holds every node of the graph, also unconnected ones. The first Folder Version
  Incrementer found (a wired `subfolder` / `filename_prefix` still wins) gives its settings; its own counter files
  are used, so the node and Save Video never disagree about the version. None in the graph: the incrementer runs
  with the global defaults.
- One version per save: the counter is advanced only when the file is actually written (not on a preview run, not
  on a cached run), and a write that fails does not consume a version.
- `GET /c2c/save_video/next_path` returns the path the next write would use; the node shows it under the naming
  control and refreshes it when a naming input changes (no polling).

## Handles (`nodes/c2c_video/handles_io.py`)

- Load Video (C2C): appended inputs `handles` (off / fit / add), `handle_frames`, `handle_side`, `handle_grid`
  (the AV Handles grids: H3 17n+5, Wan 4n+1, LTX 8n+1, custom step / plus) and the custom step / plus. The existing
  `format` (model preset) already picks a frame grid: when it names a model, the handles use that grid and the
  handle grid control shows it read-only - one grid, not two. Audio is shifted with the frames (av_handles.
  shift_audio_ends). `video_info["c2c_handles"] = {head, tail, fps, grid, source_frames}`.
- Save Video (C2C): appended `handles_trim` (auto / manual / off) + `trim_head` / `trim_tail`. auto takes
  `video_info["c2c_handles"]`, else AV Handles' remembered record, else nothing - and the report says which.
  Trimming happens before the export range; the range is in the trimmed clip's frames.
- AV Handles stays as it is (saved workflows, and the separate-node case).

## Save Video player, range, preview run, A/B

- The written file plays in `c2c_video_player.js` (the Load player). Its in / out handles write the appended
  `range_in` / `range_out` widgets (0 / 0 = whole clip), so the next run writes that range.
- `write_mode`: `write` (default; today's behaviour) or `preview first`. preview first: the run writes a fast proxy
  (H.264, at most 1280 px, temp dir) of the frames that would be written (handles trimmed, range applied) and shows
  it; the player's "Write file" button switches the run to write once and queues: upstream nodes are cached (same
  inputs), so only the encode runs.
- A/B: every run also keeps the incoming frames' proxy (temp, at most 1280 px, written while encoding). The player's
  A/B toggle shows proxy (source) against the saved file with a wipe, on the same timeline.

## Video Comparer v2

- Player: the shared transport (play / pause, J / K / L, step, loop, in / out, timecode) drives A and B together,
  with an offset in frames between them.
- View: zoom (wheel), pan (drag with the hand tool / middle button), 1:1, fit, a loupe with pixel values (RGB 0-1
  and 8-bit), the same transform on every source.
- Metrics: per-frame PSNR, SSIM (on luma) and mean colour difference (ΔE2000 on a subsample), computed on the server
  when queued (the proxies are cached so a re-queue with the same inputs recomputes nothing), drawn as a strip under
  the scrubber; click the worst frame to go there.
- Inputs and modes: image_c / image_d (+ file_c / file_d), grid 2x2, flicker (toggle A/B at a rate), vertical /
  horizontal / diagonal wipe, diff gain. Image tensors are written to cached proxies on the first run and play
  without a re-queue after that.

## Failure modes

| Failure | Behaviour |
|---|---|
| Codec missing on this machine | listed as unavailable with the install hint; choosing it refuses with a sentence |
| A control no longer allowed after an earlier change | snaps to the nearest allowed value, the node says so |
| Odd frame size for 4:2:0 / 4:2:2 | padded by one pixel (today's behaviour), reported |
| Trim more frames than exist | refused with the numbers (AV Handles rule) |
| Range outside the clip | clamped and reported |
| No incrementer node, no settings | built-in defaults; the path is still shown |
| Counter write races (two saves in one prompt) | one lock; each write takes its own version |
| Preview run, then the inputs change | the Write button is disabled until a new preview ran |

## Budgets

- No new work while idle: the next-path request runs on naming changes only; players poll nothing while paused.
- The source proxy costs one extra H.264 encode at <= 1280 px; it is skipped for clips over 20 000 frames.
- Metrics: PSNR / SSIM on luma at <= 960 px, ΔE on a 64 k-pixel subsample per frame.

## Kill switches

`C2C_SAVE_V2_MATRIX=0` hides the five controls and uses the old table; `C2C_SAVE_SOURCE_PROXY=0` stops the source
proxy (A/B then shows the saved file only); `C2C_VIDEO_HANDLES=0` disables the built-in handles (inputs ignored,
reported).

## Slices (each tested, screenshotted in both renderers and in a subgraph before the next)

1. Format matrix + colour tags + legacy mapping (backend) - encode a clip in every available combination, probe the
   file's codec / profile / pix_fmt / colour tags with ffprobe.
2. Naming: auto-detect + built-in + next-path route + one version per write.
3. Handles in Load and Save (backend): add on load, trim on save, audio in step, every source of the count.
4. Save Video front-end: sectioned controls, cascading lists, player, range, preview-then-write, A/B.
5. Video Comparer v2.

## As built (2026-10-23)

Changes made while building and testing, all covered by tests and the live check:

- The player and the A/B source show the **whole trimmed clip**: In / Out are set over it, and the export range is
  applied only when writing. The range is not part of the preview fingerprint, so moving In / Out after a preview
  still writes on the same Write file press.
- Save's `auto` trim reads only `video_info` (never AV Handles' remembered count): an implicit trim must not act on a
  record from an unrelated earlier run.
- Load Video's handles reuse AV Handles' planner on a stand-in batch of the right length, then repeat edge-frame
  indices on the video handle. IMAGE, mask and the VAE path get the handles without decoding the clip first.
- Measured: PyAV 17's `VideoFrame.reformat` knows BT.709, FCC, BT.601 and SMPTE-240M only, and converts BT.2020 with
  BT.601 while the file would say BT.2020. The PyAV path refuses BT.2020 with a message; the ffmpeg path converts it
  (tested by decoding with the right and the wrong matrix).
- ffmpeg 8.1's `av1_nvenc` refuses `profile=main`, so none is passed. ffprobe reports ProRes 4444 decoding as 12-bit
  and VP9 alpha as a side channel (`alpha_mode=1`); both files are correct.
- Video Comparer: previews of image batches use the full-length proxy (the old 600-frame cap would have put the
  metrics strip and the picture out of step). `b_offset` is a widget, so the player and the metrics use the same
  number.
