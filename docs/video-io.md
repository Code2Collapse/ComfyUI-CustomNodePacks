# Loading and saving video (C2C)

These nodes bring clips into a graph and write them out again. Load Video (C2C) reads a movie, an image, an image
sequence or an EXR sequence. Save Video (C2C) writes a movie or an image sequence. The two are linked: Save finds the
Load its frames came from: it trims that Load's handles, and can reverse its colour conversion (`auto (from Load)`)
and reuse its frame rate and name.
AV Handles does the handle part as a separate node, for frames that do not come from Load Video (C2C).

| Node | Use it to |
|---|---|
| [Load Video (C2C)](#load-video-c2c-loadvideoc2c) | Load a movie, image, image sequence or EXR sequence as frames, audio, mask, or a LATENT |
| [Load Images Path (C2C)](#load-images-path-c2c-loadimagespathc2c) | Load every image of one folder, in name order, with a cap / skip / every-Nth |
| [Video Info (C2C)](#video-info-c2c-videoinfoc2c) | Turn a loader's `video_info` into numbers (fps, frame count, size) |
| [Save Video (C2C)](#save-video-c2c-savevideoc2c) | Write H.264 / H.265 / ProRes / DNxHR / FFV1 / WebM / GIF / WebP or an image sequence |
| [AV Handles — Add / Trim](#av-handles--add--trim-avhandlesmec) | Pad a clip (and its audio) to a video model's frame grid, then trim exactly that back off |

Other video guides: [Video Frame Player](video-frame-player.md) (scrub, trim, crop a frame batch),
[Video Comparer](video-comparer.md) (A/B two or more clips). Design records, written before the code:
[save-video-v2.md](save-video-v2.md), [LOAD_SAVE_VIDEO_CM_DESIGN.md](LOAD_SAVE_VIDEO_CM_DESIGN.md).

## Load Video or Load Images Path?

| You have | Pick |
|---|---|
| A movie file (mp4, mov, mkv, webm ...) | Load Video (C2C) |
| A numbered sequence where the frame NUMBERS matter (shot.1001.exr ...), with gaps to fill, a frame range like `1001-1100`, an EXR layer, or a colour space to convert | Load Video (C2C), `location` = `server path` (or pick the sequence pattern from the input folder) |
| A folder you want read as "every image, in name order", no frame numbers, no audio, no resize | Load Images Path (C2C) |
| Frames you want to resize, retime, crop, put handles on, or encode with a VAE while loading | Load Video (C2C) - Load Images Path has none of these controls |

Load Images Path does not look at frame numbers: a gap in the numbering is not filled, the next file simply follows.
Load Video (C2C) works by frame number and stops on a gap by default (`missing_frames` = `error`).

---

## Load Video (C2C) (`LoadVideoC2C`)

**Menu:** 🐺 C2C/🧰 Core/Video · **Parameters:** [NODE_REFERENCE](../NODE_REFERENCE.md#loadvideoc2c)

Movies are read with PyAV (`av`). EXR, DPX and HDR files need the OpenImageIO Python package; without it the file
list only offers PNG / JPEG / TIFF / BMP / WebP images. The Wan VAE stream encode (below) is tested on the CPU with
tiny random weights only: GPU speed / VRAM and tiled quality with real weights are NOT VERIFIED.

### What it is for

Load a clip from the ComfyUI input folder, or any path on the server, as IMAGE frames, a MASK (from alpha) and AUDIO.
The frames are read in chunks, never the whole file at once. You can retime, resize, crop, cap and skip while loading,
add handles for a video model, convert colour with OCIO, and get a map from each output frame back to its source
frame number. Load Video Path (C2C) was merged into this node (`location` = `server path`); saved workflows load as
Load Video (C2C).

### Minimal wiring

```
Load Video (C2C) ──IMAGE──────► your graph ──► Save Video (C2C)
                 ──audio──────────────────────► Save Video (C2C).audio
                 (video_info is optional: Save finds this Load on its own)

With a VAE:
VAE Loader ──vae──► Load Video (C2C) ──IMAGE (a LATENT now)──► sampler
```

### Key controls

- `video`: the file from the input folder. Upload with the node's own bar or drop files on the node. Sequences are
  listed once, as a pattern (`plates/shot.####.exr`).
- `location` = `server path` and `path`: read any clip, image, folder or pattern (`shot.####.exr`, `shot.%04d.exr`,
  `shot.$F4.exr`, `shot.[1001-1100].exr`). One frame of a sequence finds the whole sequence.
- `force_rate`, `frame_load_cap`, `skip_first_frames`, `select_every_nth`, `seconds_cap`: what gets loaded.
  `frame_range` takes source frame numbers instead (`1001-1100`, `1001-1100x2`).
- `resolution` (longest side, e.g. `1080p` = 1920) or `custom_width` / `custom_height`, rounded by `divisible_by`.
  Crop with the Crop tool in the node's player (`crop_x` ... `crop_h`).
- `format`: a model preset. It trims the frame count DOWN to the model's grid (Wan 4n+1, LTXV 8n+1, MiniMax H3
  17n+5 ...) and sets a size multiple. It does not change the frame rate.
- `handles` = `fit` or `add`: extra frames for a video model to settle on (repeated edge frames, audio moved with
  them). Save Video (C2C) trims them off again. `extra_handles` adds whole grid steps on top.
- `colorspace_in` / `working_space`: OCIO colour management. `off (as before)` is the default (no conversion);
  `auto (detect)` reads the file's own tags. Save Video (C2C) on `auto (from Load)` reverses exactly this.
- `decoder`: `auto` / `fast` / `exact`. See [Settings](#settings-that-affect-these-nodes) for what `auto` may use.

### VAE: loading straight to a LATENT

Connect a `vae` and the first output carries a LATENT instead of IMAGE (the `mask` output is then empty).

- **Wan 2.1 / Wan 2.2 video VAE** (2026-11-19, 0a0579c): the frames are fed to the encoder while they stream in,
  in the same steps core uses, with one causal cache. Untiled, the latent is identical to core's own VAE Encode
  (`torch.equal`, tested on the CPU with tiny random weights). When one step would not fit the device, the node
  splits the frame into spatial tiles: it writes each tile's pixels to the temp folder, encodes each tile over the
  whole clip, and blends the tiles. A tiled latent approximates the untiled encode; it is not identical.
- Wan encoders use `1 + 4 × ((frames - 1) // 4)` frames: a 50-frame load encodes 49. Set `format` = `Wan` so the
  count is already on the grid. The console logs the method and "N of M frames used".
- **Image VAEs** (SD, SDXL, Flux ...): encoded per chunk, as before.
- **Other video VAEs** (that compress time) are refused: use their own encode node on the IMAGE output.

### Reading the outputs

- `frame_count`, `audio`, `mask` (1 - alpha), `video` (a lazy handle other C2C video nodes read in windows).
- `video_info`: VHS-compatible numbers (`source_*` and `loaded_*`). It also carries the handles, the colour record
  and the source frame numbers / timecode that Save Video (C2C) uses.
- `info` (JSON): `file`; `source` (width, height, fps, frames, duration, pix_fmt, bit_depth, has_alpha, rotation,
  has_audio); `loaded` (size, fps, frames, first / last frame, start / end seconds); `format`; `audio`; `decoder`
  (`asked`, `used`, `note`); `media` (what the source is, frame numbers, missing frames, notes, a RAM estimate);
  `handles` and `colour` when those are on.
- `frame_map` (JSON, h3.frame_ledger v1.1): each output frame -> its source frame and frame number (-1 = a black
  frame). `frame_offset` shifts the reported numbers only.
- `exr_passes`: every EXR layer, read lazily. `metadata` (JSON): the header or container tags.

IMAGE and MASK are decoded only when something reads them. With only `video`, `info` or `frame_count` connected,
nothing is decoded.

### Common mistakes

- "server path mode - enter a path in 'path'": `location` is `server path` but `path` is empty.
- A path outside ComfyUI's input / output / temp folders is refused only when ComfyUI listens on an address other
  machines can reach (`--listen 0.0.0.0`). Extra folders are allowed through the `C2C_MEDIA_ROOTS` environment
  variable or `c2c_media_roots.txt` in the ComfyUI folder.
- "this selection is N frames of WxH, X GB as an IMAGE, and only Y GB of RAM is free": the IMAGE would not fit in
  half the free RAM. Lower `frame_load_cap`, raise `select_every_nth`, or load smaller. (A VAE-encoded load does not
  build the IMAGE.)
- `Format "...": need at least K frames after trimming`: the preset's grid needs more frames than are left.
- A sequence with a missing file stops with the missing numbers (`missing_frames` = `error`). Pick `black`,
  `hold previous` or `nearest` to fill; `skip (as before)` drops the frame, so later frames play early.
- Tiled Wan encode: "needs X GB of temp disk space" (free space or load fewer frames); "not a multiple of the VAE's
  scale" (set `divisible_by` / `format` so the size fits).

---

## Load Images Path (C2C) (`LoadImagesPathC2C`)

**Menu:** 🐺 C2C/🧰 Core/Video · **Parameters:** [NODE_REFERENCE](../NODE_REFERENCE.md#loadimagespathc2c)
· Reads through OpenImageIO.

### What it is for

Read every image in one folder (`.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff`, `.exr`, `.hdr`, `.dpx`), sorted by name
(natural order: 2 before 10), as an IMAGE batch. `skip_first_images`, `select_every_nth` and `image_load_cap` pick
which ones, in that order.

### When to pick it

For a plain "folder of frames" with nothing else to set. Anything with frame numbers, frame rate, resize, audio,
colour or handles: use [Load Video (C2C)](#load-video-c2c-loadvideoc2c) with `location` = `server path` and the
folder as `path`.

### Minimal wiring

```
Load Images Path (C2C) ──IMAGE──► your graph
   directory = D:/renders/shot010/
```

### Reading the outputs

`IMAGE`, `MASK` (1 - alpha when the images have alpha), `frame_count`, and `video` (the lazy handle). The folder
is decoded only when IMAGE or MASK is connected.

### Common mistakes

- The same remote-access rule as Load Video: on a server other machines can reach, only the input / output / temp
  folders and the configured media roots are read.
- The RAM check is the same as Load Video's; lower `image_load_cap` or raise `select_every_nth`.
- The node does not care about numbering: missing numbers are not filled, other images in the folder are included.

---

## Video Info (C2C) (`VideoInfoC2C`)

**Menu:** 🐺 C2C/🧰 Core/Video · **Parameters:** [NODE_REFERENCE](../NODE_REFERENCE.md#videoinfoc2c)

Splits a `video_info` (from Load Video (C2C) or a VHS loader) into ten numbers: `source_fps`,
`source_frame_count`, `source_duration`, `source_width`, `source_height`, and the same five as `loaded_*` (after
skip, cap, retime and resize). Use it to drive a frame count or fps widget elsewhere.

```
Load Video (C2C) ──video_info──► Video Info (C2C) ──loaded_fps──► (any FLOAT fps input)
```

---

## Save Video (C2C) (`SaveVideoC2C`)

**Menu:** 🐺 C2C/🧰 Core/Video · **Parameters:** [NODE_REFERENCE](../NODE_REFERENCE.md#savevideoc2c)

Movies are encoded with an installed FFmpeg (5.1 or newer) for the codecs it supports (setting *Video > Save >
Encode with FFmpeg*), otherwise with PyAV; image sequences (EXR, DPX, PNG, TIFF, JPEG) go through OpenImageIO. The
format lists offer only what this machine can write. NVENC codecs need an NVIDIA GPU. Stream save of Wan 2.2 latents with real weights, and its GPU speed, are
NOT VERIFIED (4a394e2).

### What it is for

Write an IMAGE batch, or a video latent decoded on the fly, to a file in the output folder, with audio, colour tags,
an OCIO conversion, handles trimmed, an export range, and a player on the node that shows the result next to the
frames that came in.

### When to pick it

Use it at the end of a graph that started at [Load Video (C2C)](#load-video-c2c-loadvideoc2c): it reads that Load's
frame rate, handles, colour and name without any wire. It also works on any IMAGE batch.

### Minimal wiring

```
… ──IMAGE──► Save Video (C2C)
Load Video (C2C).audio ──► Save Video (C2C).audio      (optional)

Stream save (no full IMAGE batch in RAM):
sampler ──LATENT──► Save Video (C2C).latent
VAE ──────────────► Save Video (C2C).vae
```

### Key controls

- Format: `format_family` (Video / Image Sequence / Custom) -> `subformat` (H.264, H.265, ProRes 422 HQ, EXR ...).
  Only the chosen subformat's own options show. `Custom` gives the five controls `container` -> `codec` ->
  `profile` -> `bit_depth` -> `chroma`. An old workflow's single-list `format` label still encodes as before.
- `fps`: left at 24 with a Load found, the Load's loaded frame rate is used.
- `filename_prefix`: may contain folders. `auto (from Load)` names the file after the followed Load (its
  `name_prefix` + clip name + `name_suffix`). `naming`: `default` (the C2C setting, Folder Version Incrementer
  unless changed), `ComfyUI counter`, or `Folder Version Incrementer`. The node shows the next path before you queue.
- `handles_trim` = `auto`: trims exactly the handles Load Video (C2C) added. `manual` uses `trim_frames` /
  `trim_side`; `off` writes every frame.
- `range_in` / `range_out`: export only part of the clip; set them with the In / Out handles of the node's player.
- `write_mode` = `preview first`: a run shows what would be written and writes nothing; press **Write file** in the
  player, and only the encode runs again.
- `pingpong` (2026-11-19, 9a20c0c; ported from ComfyUI-AusBoss, MIT): forward then backward without repeating the
  two turn-around frames (n frames -> 2n - 2). The audio bounces with the picture. Applied after the handle trim and
  the range.
- `write_manifest` (2026-11-19, 9a20c0c): also writes `<name>.manifest.json` (`<folder>.manifest.json` for a
  sequence), then reads the written file back and checks it. See below.
- `review_copy` + `burn_in`: a half-size H.264 `<name>_review.mp4` in the display space, with name / version /
  frame / timecode / colour / date in the corners. The master gets no burn-in.
- `save_metadata`: writes `<name>.png` (the first frame with the workflow inside) next to the file. The video itself
  never carries the workflow.
- Colour: `colorspace_in` / `colorspace_out` (`auto (from Load)` reverses Load's conversion), and the tags
  `color_primaries` / `color_transfer` / `color_range`.
- Stream save: `tile_mode` = `auto` tiles a Wan 2.1 / 2.2 latent spatially only when a frame would not fit the GPU;
  `manual` uses `tile_size`. Each tile is decoded over the whole clip in one causal pass and blended on disk.

### Reading the outputs

- `filenames`: the full path of each written file, one per line (empty in `preview first`).
- `preview`: the saved frames; only the last frame for a stream save or a preview-first run.
- The node shows a report after each run: format, frames, size, colour, which Load it followed, the handles it
  trimmed and where that count came from, warnings, the version folder, and for a stream save the method used.
- The manifest (`write_manifest`) holds: `outputs`; `format` (container, codec, profile, bit depth, pix_fmt,
  alpha, quality, encoder preset / gop / bitrate); `frames`, `fps`, `width`, `height`; `source` (handles trimmed,
  range, pingpong, `frame_runs` = which incoming frames were written as `[start, stop, step]` runs, source frame
  numbers, timecode, streamed); `audio` (sample rate, channels, samples); `colour` (spaces, tags, OCIO plan);
  `warnings`; and `validation` (`ok`, what was `checked`, `problems`: frame count, size, frame rate, audio stream).
  A failed check is also shown as a warning on the node; the file is still written.

### Common mistakes

- "Wire images, OR latent + vae - not both." / "Stream save needs both latent and vae wired."
- `pingpong` with a stream save is refused: a streamed latent is read once, front to back. Decode to images first.
  With pingpong on, source frame numbers and timecode are not written into the file. A preview-first run shows
  the clip without the bounce.
- "H.264 (NVENC) encodes at most 4096 px a side" (2026-11-19, f6f6f07): the node stops before FFmpeg starts.
  Pick H.265 (NVENC) or AV1 (NVENC) (up to 8192 px), H.264 on the CPU, or scale down. It never switches codec on
  its own.
- "DNxHR needs frames of at least ...": pick ProRes or FFV1 for small frames.
- BT.2020 primaries through PyAV are refused ("no BT.2020 matrix"): install FFmpeg 5.1 or newer, or use BT.709.
- "Format ... is not available on this machine": the message names the fix; pick another format.
- "Folder Version: ... already exists — will not overwrite."
- A stream save keeps no source copy, so the player's A/B shows the saved file only, and `review_copy` is skipped.
  Stream save with tiles works for Wan 2.1 / 2.2 VAEs only; other VAEs decode untiled (a warning says so), and a VAE
  without a streaming decode is decoded whole (not bounded in RAM, also a warning).
- "Trimming H + T frames off an N-frame clip would leave nothing": the handles in `video_info` do not belong to
  these frames.

---

## AV Handles — Add / Trim (`AVHandlesMEC`)

**Menu:** 🐺 C2C/🧰 Core/Video · **Parameters:** [NODE_REFERENCE](../NODE_REFERENCE.md#avhandlesmec)

### What it is for

A video model wants a frame count on its grid (Wan 4n+1, LTX 8n+1, MiniMax H3 17n+5 ...) and settles over the first
frames. This node pads a clip with repeated edge frames (and the audio with silence) before the model, and trims
exactly that amount off afterwards. The trim reads the count the add recorded, so the two ends cannot drift apart.

### When to pick it

Load Video (C2C) has the same handles built in (`handles`), and Save Video (C2C) trims them (`handles_trim` =
`auto`). Use AV Handles when the frames do not come from Load Video (C2C), for audio only, or when the trim must
happen before the save.

### Minimal wiring

```
frames ──► AV Handles (mode = fit) ──images──► video model ──► AV Handles (mode = trim) ──► Save
                                   ──handles─────────────────────► handles   (optional, exact)
audio ───► .audio ──audio──────────────────────────────────────► .audio
```

### Key controls

- `mode`: `fit` (ask for `target_frames` out; it pads up to the model grid and trims back to the target), `add`
  (put `handle_frames` on), `trim` (take them off).
- `model`: the frame grid. `side`: `head`, `tail` or `both`.
- `trim_amount` = `auto` uses the wired `handles`, else the remembered count; `manual` uses `handle_frames`.
- `memo_key`: give each clip a name when two clips are handled in one workflow.
- Added 2026-11-19 (9a20c0c); the defaults behave as before:
  - `handle_align` (add): rounds each end's handles up to whole grid steps, so a clip that was on the grid is on it
    again after the trim.
  - `audio_sync` = `exact samples`: the add records the exact audio samples it inserted, and the trim removes
    exactly those. The commit reports a bit-identical audio round trip at 23.976 fps, also across workflows. Set it
    on the add AND the trim.
  - `audio_fade_ms`: short linear fades at the audio edges the node creates, to avoid clicks. 0 = off.

### Reading the outputs

`images`, `audio`, `total_frames`, `handles` (the exact count, head / tail, fps, target; wire it to the trim node)
and `info`. `info` (also shown on the node) says what was done and WHERE the trim count came from: "the `handles`
wire", "remembered from the add node earlier in THIS run", or "... in an EARLIER run" (worth a check). With
`audio_sync` = `exact samples`, a line says whether the exact samples were used or why it fell back to frame duration.

### Common mistakes

- "fit mode needs images": for audio alone use `add` with a count.
- "This clip is already N frames but the target is T. fit only pads": trim first, or raise the target.
- "Trimming N frames off an M-frame clip would leave nothing": the remembered count belongs to another clip. Wire
  `handles`, or use `memo_key`.
- A trim with nothing to go on trims 0 and says so in `info` and the console.
- `manual_fps` = 0 measures the fps from the audio length. With no audio, set it: a guessed rate shifts the audio.
- `exact samples` falls back to frame duration when the trim is `manual`, the add did not record samples, the trim
  is not the recorded add, or the sample rate differs; `info` says which.

---

## Settings that affect these nodes

In ComfyUI's settings under **C2C**. Where a setting has an environment variable, the variable wins when it is set.

| Setting | Variable | Effect |
|---|---|---|
| Video > Load > Automatic decoder may use (`exact` default / `fast` / `gpu`) | `C2C_VIDEO_AUTO` | What Load Video's `decoder` = `auto` may pick. `exact` = always the frame-accurate PyAV reader (measured as fast on the CPU). `fast` / `gpu` allow the FFmpeg path (gpu: NVDEC) for large 8-bit loads after a self-check. |
| Video > Upload > Largest upload (MB) | `C2C_UPLOAD_MAX_MB` | Largest file the Load node's upload bar accepts. |
| Video > Save > Save Video file naming | - | What `naming` = `default` means. |
| Video > Save > Encode with FFmpeg | `C2C_SAVE_FFMPEG` | Off: always the PyAV encoder. |
| Video > Save > Write a source proxy | `C2C_SAVE_SOURCE_PROXY` | Save Video writes a small H.264 copy of the incoming frames for its player's A/B view. Off: the A/B shows the saved file only. |
| Video > Colour Management > OCIO config file / Built-in OCIO config / Display view / Processing | - | The OCIO config and view Load and Save use; Processing = Auto / Exact CPU / OCIO GPU. |

Environment-only switches: `C2C_VIDEO_FAST` (`0` turns the fast decoder off), `C2C_VIDEO_HANDLES=0` (built-in
handles off), `C2C_VIDEO_CM=off` (no colour management), `C2C_SAVE_V2_MATRIX=0` (Save shows the old single format
list), `C2C_MEDIA_LIST=videos` (the Load list shows movies only).
