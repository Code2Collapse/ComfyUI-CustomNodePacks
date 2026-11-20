# Video Comparer

One node for comparing clips or images: a synced A/B player, overlay views (wipe, onion, difference, false colour),
scopes, audio views, per-frame metrics, and a change report that finds and boxes what changed. Image Comparer
(`ImageComparerMEC`) is no longer registered; this node replaces it and compares images too. Workflows saved with
the old `VideoComparerMEC` id are converted to `VideoComparerC2C` when they load.

Related guides: [Loading and saving video](video-io.md) (Save Video (C2C) has its own A/B view of the written file),
[Video Frame Player](video-frame-player.md) (scrub, trim and crop one clip).

## Video Comparer — Player + Wipe/Diff/Scopes (`VideoComparerC2C`)

**Menu:** 🐺 C2C/🧰 Core/Preview · **Parameters:** [NODE_REFERENCE](../NODE_REFERENCE.md#videocomparerc2c)

`audio_spectro` needs the `librosa` package and `audio_loudness` needs `pyloudnorm`; without them those modes stop
with a message naming the package. No model weights are needed.

### What it is for

Put two versions of a shot side by side and see where and when they differ: before / after a sampler change, two
upscalers, a plate against its render. A is the reference. B, and optionally C and D, are compared against it. Each
source is either an IMAGE input (`image_a` ...) or a file from the input folder (`file_a` ...; video, image or EXR),
used when the IMAGE socket is not wired.

### When to pick it

- To compare a written file against the frames that came in, Save Video (C2C)'s own player already has an A/B view
  ([video-io.md](video-io.md#save-video-c2c-savevideoc2c)).
- To scrub, trim or crop a single clip, use the [Video Frame Player](video-frame-player.md).
- Use this node for everything else: two to four clips or images, with numbers.

### Minimal wiring

```
version 1 ──IMAGE──► image_a ┐
version 2 ──IMAGE──► image_b ├─ Video Comparer ──annotated──► Preview Image
                             │                  ──change_report──► (Show / Save Text)
(optional) ─────────► image_c / image_d
```

Queue once: wired IMAGE batches reach the browser only after a run. Files picked in `file_a` / `file_b` play without
a run.

### Key controls

- `mode` (default `synced_player`):
  - shown live in the browser, no queue needed: `synced_player`, `wipe`, `onion`, `diff`, `side_by_side`,
    `per_channel`, `false_color`, `bit_depth_crush`;
  - computed when the node runs: `waveform_scope`, `parade_scope`, `vectorscope`, `histogram_scope`,
    `audio_waveform`, `audio_spectro`, `audio_loudness`.
- The player (in `synced_player`): Load Video's transport (play / pause, backwards, J / K / L, frame step, loop,
  in / out, timecode, full screen) drives A; the other sources follow it frame by frame. Views: A, B, wipe
  (vertical / horizontal / diagonal), difference with gain, onion, flicker, side by side, and a 2 x 2 grid of
  A / B / C / D. Zoom, pan, 1:1 and a loupe with pixel values.
- `b_offset`: B starts this many frames after A, so A frame i is compared with B frame i - `b_offset`. The player,
  the metrics and the change report all use it. C and D always start with A.
- `metrics` (on): per-frame PSNR, SSIM (on luma) and colour difference (CIEDE2000) of A against B, drawn as a strip
  under the scrubber. Hover to read a frame, click to go there; the worst frame of each is marked.
- `diff_threshold`: in `diff`, smaller differences count as none. The change report uses it too (0 means 0.02,
  about 5 / 255).
- `frame_index`: the frame the queue-time modes, `preview`, `diff_mask` and `annotated` use.
- `preview_fps`: playback rate of the browser preview of wired IMAGE batches (tensors carry no frame rate). The
  preview is at most 1280 px and keeps every frame; the comparison outputs use the full frames.

### Reading the outputs

- `preview`: the picture of the current mode at `frame_index` (in `synced_player`, A's frame).
- `diff_mask`: the per-pixel difference (mean over channels, before gain). It is filled in the `diff`,
  `per_channel`, `false_color` and `bit_depth_crush` modes only; in the other modes it is all zero.
- `scope`: a histogram of A at `frame_index`.
- `info`: one text line - mode, bit depth, what A and B are, their shapes, PSNR and the percentage of differing
  pixels (above `diff_threshold`). Audio modes write a shorter line.
- `annotated` (2026-11-19, f6d8917): A at `frame_index` with the difference to its B partner painted as a heat map,
  a red box around each changed region and "frame N: X% changed". A alone when there is no B.
- `change_report` (JSON, 2026-11-19, f6d8917), computed on every run at up to 256 px on the long side:
  - `reference`, `threshold`, `significant_share` (0.001: a frame with 0.1 % or more changed pixels is
    significant), `offset`, `pairs`, `first`;
  - `changed_share` and `mean_abs_diff`: one value per A frame (`null` where B has no partner frame);
  - `significant_frames`, `significant_ranges` (`[first, last]` runs), `worst_frame`;
  - `regions`: for the 10 worst frames and `frame_index`, up to 8 boxes each as `box_xywh` in full-size A pixels,
    with their `share` (regions under 0.05 % of the frame are not boxed);
  - `sources`: B, C and D against A (PSNR / SSIM / colour difference summaries, changed share, worst frames), and
    `ranking`: their keys by mean PSNR, highest first;
  - `note` when something is missing ("no B frames", "do not overlap at this offset", "B resized to A's size").
  If the report itself fails, it holds `error` and the other outputs are still produced.

Preview memory (2026-11-19, 6178bc9): the preview writer encodes 8 frames at a time instead of copying the whole
batch first. Measured in that commit: 60 frames of 720p, 7.61 s / +489 MB before, 2.40 s / +342 MB after.

### Common mistakes

- Wired IMAGE inputs show nothing in the player until the node has run once.
- `diff_mask` is empty in `wipe`, `onion`, `side_by_side` and the scope modes: switch to `diff` for a mask, or
  threshold `annotated` / use `change_report` instead.
- The audio modes read `audio_a` / `audio_b` only together with `image_a` / `image_b`, or from a file that has
  audio. An AUDIO input on its own is ignored.
- A and B of different sizes: the metrics and the change report scale B to A; `info`'s PSNR is `nan` when the shapes
  differ.
- "image_a must be IMAGE tensor [B,H,W,C]": something other than an image batch is wired.
