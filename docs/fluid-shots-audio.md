# Fluid Shots and Audio Reverser

Video models follow motion in small steps. A speed-ramped clip, slow in one part and fast in another, moves too far
between some frames. Fluid Shot Encoder adds in-between frames where the motion is fast, so every step is small; you
run your video model on that longer clip; Fluid Shot Decoder then picks the frames that stand for the original ones,
so the result has the original frame count and pacing again. Audio Reverser plays audio backwards and describes the
time change for syncing. All three run on the CPU with no model.

| Node | Use it to |
|---|---|
| [Fluid Shot Encoder](#fluid-shot-encoder-fluidshotencodermec) (`FluidShotEncoderMEC`) | Add flow-warped in-betweens on fast sections, and write the TIMEMAP |
| [Fluid Shot Decoder](#fluid-shot-decoder-fluidshotdecodermec) (`FluidShotDecoderMEC`) | Pick the processed frames that match the original frames, retime audio, check the pick |
| [Audio Reverser](#audio-reverser-audioreversermec) (`AudioReverserMEC`) | Reverse a clip's audio, or a time range of it, or only the order of short grains |

## The round trip and the TIMEMAP

```
Load Video ─▶ Fluid Shot Encoder ─┬─ normalized_images ─▶ your video model ─▶ images ───┐
                                  └─ time_map ─────────────────────────────▶ time_map ──┤ Fluid Shot Decoder
                                                         (optional) audio from the model ┘   ├─ images ─▶ Save
                                                                                             ├─ audio
                                                                                             └─ integrity
```

The TIMEMAP is a small dictionary that travels from the Encoder to the Decoder on its own wire type (`TIMEMAP`):
- `kind`: `"fluid_shot"`. The Decoder refuses anything else.
- `src_count` / `out_count`: frames in and frames out of the Encoder.
- `original_positions`: for each original frame, its index in the longer batch. Original frames are copied into the
  longer batch unchanged.
- `src_of_output`: for each frame of the longer batch, the source time it stands for, in frames (for example `3.5` =
  halfway between frames 3 and 4).
- Only when a 2026-11-19 option is on: `motion_metric`, `pair_motion` (the measured motion of each frame pair, in
  pixels) and `scene_cuts` (the pairs treated as cuts).

## Fluid Shot Encoder (`FluidShotEncoderMEC`)

**Menu:** 🐺 C2C/🧰 Core/Temporal, shown as "Fluid Shot Encoder (Temporal Normalizer)" ·
**Parameters:** [NODE_REFERENCE](../NODE_REFERENCE.md#fluidshotencodermec)
Needs OpenCV (`opencv-python`).

### What it is for

Making motion per frame roughly even before a video model sees the clip. For each pair of neighbouring frames it
measures the motion with Farneback optical flow (at a long side of at most 384 px, scaled back to full-resolution
pixels). If the motion is above `max_px_per_frame`, it inserts `ceil(motion / max_px_per_frame) - 1` in-betweens,
made by warping both frames along the flow and cross-fading them. If the warp fails it blends the two frames, and as
a last resort repeats the first one.

### When to pick it

When the source has fast sections or speed ramps and you want the model to see small steps. It is not a frame
interpolator for playback: always pair it with Fluid Shot Decoder to get back to the original frames. A clip of one
frame passes through unchanged.

### Key controls

- `max_px_per_frame` (default 5.0): the largest motion allowed between two output frames, in pixels. Lower = more
  in-betweens.
- `scene_cut_threshold` (added 2026-11-19, commit 6985d30; 0 = off): a frame pair whose gray-level histograms differ
  by at least this much (Bhattacharyya distance, 0 = same, 1 = unrelated) counts as a scene cut and gets no
  in-betweens, so two shots are never blended. The tooltip notes that hard cuts are usually above 0.4.
- `motion_metric` (added 2026-11-19, commit 6985d30): `global p95` (default, as before) uses the 95th percentile of
  the whole frame's flow. `local tiles` cuts the frame into a 4×4 grid and uses the fastest tile, so a small fast
  subject on a still background (a hand, a ball) also gets in-betweens.

Limits: at most 16 in-betweens per frame pair, and the whole clip may grow to at most 4 × its frame count + 16.

### Reading the outputs

| Output | What it holds |
|---|---|
| `normalized_images` | The longer clip: the original frames with in-betweens on fast sections |
| `time_map` | The TIMEMAP described above. Wire it to Fluid Shot Decoder |

### Common mistakes

| Message | What to do |
|---|---|
| `Fluid Shot Encoder needs OpenCV (opencv-python)` | Install `opencv-python` in ComfyUI's Python |
| `Fluid Shot Encoder would make more than N frames from M` | Raise `max_px_per_frame`, or split the clip at the fast cut |
| `unknown motion_metric` | Use `global p95` or `local tiles` |

## Fluid Shot Decoder (`FluidShotDecoderMEC`)

**Menu:** 🐺 C2C/🧰 Core/Temporal, shown as "Fluid Shot Decoder (Restore Timing)" ·
**Parameters:** [NODE_REFERENCE](../NODE_REFERENCE.md#fluidshotdecodermec)

### What it is for

Taking the processed longer clip and keeping only the frames at `original_positions`, so the result has the
original frame count and pacing. If the model kept the frame count, the pick is exact.

### When to pick it

Always after Fluid Shot Encoder, with the same `time_map`.

### Key controls

- `allow_approximate` (default on): when the model changed the frame count (for example a model that trims to
  4n+1 frames), the original positions are scaled proportionally and rounded; `info` says the largest rounding error
  in frames. Off: the node stops with an error instead.
- `audio` (optional, added 2026-11-19, commit 6985d30): audio made on the longer timeline, for example by an
  audio-video model. It is retimed onto the original frames with the same map as the pictures: linear between the
  original frames' positions, at speed 1 after the last one. The audio is resampled, not time-stretched, so in
  stretches that got in-betweens it plays faster and higher.

### Reading the outputs

| Output | What it holds |
|---|---|
| `images` | The processed frames at the original count |
| `info` | `exact: N original frames selected from M.`, or `APPROXIMATE: batch has M frames, the encoder made K; ... (rounding error up to X frame).` |
| `audio` | The wired audio retimed onto the original frames; nothing when no audio is wired |
| `frame_map` | JSON: `src_count`, `batch`, `expected`, `exact`, and `frames`, one entry per original frame with `src` (original index), `out` (the batch frame picked) and `pos` (its position on the longer timeline) |
| `integrity` | JSON check of the pick: `ok`, `count_mismatch`, `missing` (positions outside the batch), `repeated` (the same frame picked twice in a row), `non_monotonic` (out of order) and `content_duplicates` (two consecutive picks with identical pixels). `ok` is true only when all are empty |

### Common mistakes

| Message or symptom | What to do |
|---|---|
| `time_map is not a Fluid Shot TIMEMAP (wire it from Fluid Shot Encoder)` | Wire the Encoder's `time_map`. The Audio Reverser's TIMEMAP does not fit here |
| `the batch has M frames but the encoder made K` | The model changed the frame count and `allow_approximate` is off: turn it on, or keep the frame count unchanged |
| `info` starts with `APPROXIMATE` | Timing is only close; check `integrity` for repeated frames |
| `audio must be an AUDIO dict with a 'waveform' tensor` | Wire a ComfyUI AUDIO output |

## Audio Reverser (`AudioReverserMEC`)

**Menu:** 🐺 C2C/🧰 Core/Audio, shown as "Audio Reverser" ·
**Parameters:** [NODE_REFERENCE](../NODE_REFERENCE.md#audioreversermec)

### What it is for

Playing a ComfyUI AUDIO backwards, keeping its sample rate and channels, for example to match a video you reversed.

### Key controls

- `start_seconds` / `end_seconds` (added 2026-11-19, commit 6985d30): reverse only this range; audio outside it plays
  unchanged and the clip keeps its length. `0` and `-1` (the defaults) = the whole clip, as before.
- `reverse_mode` (added 2026-11-19): `reverse` turns every sample backwards (as before; notes swell instead of
  decaying). `grain_reverse (keeps note attacks)` reverses the order of short grains while each grain plays forwards,
  so notes keep their pitch and attack and only the order of events runs backwards.
- `grain_ms` (default 80): grain length for `grain_reverse`, with 50 % overlap. A range shorter than two grains is
  reversed plainly.

### Reading the outputs

| Output | What it holds |
|---|---|
| `audio` | The reversed audio |
| `time_map` | A TIMEMAP of kind `audio_reverse`: `sample_rate`, `num_samples`, `duration_s`, `formula` (`t_new = duration_s - t_orig`) and `map_endpoints`. With a range: `range_start_s`, `range_end_s` and `segments` (each with `src`, `dst` and `reversed`), and a piecewise formula. With `grain_reverse`: `reverse_mode`, `grain_ms` and a note that the map holds to ± `grain_ms` / 2 |

No node in this pack reads the `audio_reverse` TIMEMAP; it is there for your own syncing.

### Common mistakes

| Message | What to do |
|---|---|
| `the range ... holds no audio` | `end_seconds` must be after `start_seconds` and inside the clip (or -1) |
| `expected AUDIO dict with a 'waveform' tensor` | Wire a ComfyUI AUDIO output |
