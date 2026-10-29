# Mask + Matting on ComfyUI's own SAM 3 (L7.125)

Owner answer 2026-10-29: "Yes, use core SAM 3". The sam3 / sam3.1 / ONYX tracking runs on ComfyUI's own SAM 3
(core 0.39+), with no extra library. CNP's own path stays only as a fallback for an older ComfyUI, and each backend is
measured before it ships. The viewer's Detect already runs on core SAM 3 (L7.114, F7.114 in docs/C2C_DECISIONS.md of
the workspace).

## 1. Today

| Path | Engine | What it needs | State on a fresh install |
|---|---|---|---|
| cascade segmenter `sam3.1` (`segmenters/sam31_backend.py`) | the vendored `third_party/sam3_lib`. **Per frame**: the same prompts on every frame, no tracking | sam3_lib (not shipped) | `[missing-deps]` |
| cascade segmenter `sam3` (`segmenters/sam3_backend.py`) | the `sam3` pip package, per frame | the package (not in requirements) | `[missing-deps]` |
| ONYX (`_onyx_pipeline.py`) | sam3_lib's video predictor: `init_state`, `add_prompt` (points; box = corner points with labels 2 / 3), `propagate_in_video` forward / backward, `reset_state` | sam3_lib | falls back to the model-free tracker (`_classic.ClassicVideoTracker`, same protocol), labelled approximate |

## 2. Core facts used (core v0.39.0-43, read 2026-10-29)

- One Comfy-Org checkpoint holds the model and the text encoder:
  `comfy.sd.load_checkpoint_guess_config(path, output_vae=False)` returns (MODEL, CLIP).
- One frame: the detector head for words. The SAM decoder `forward_segment(frame_1008, point_inputs, box_inputs,
  mask_inputs)` takes clicks and boxes in 1008 x 1008 space and returns logits.
- Video: `net.forward_video(images [N,3,H,W] CPU, initial_masks [O,1,H,W], text_prompts=None, ...)`.
  - It tracks **forward from the first frame** only.
  - SAM 3.1 (multiplex) returns `{"packed_masks": [N, O, Hm, Wm/8] uint8 bits, "n_frames", "scores"}`.
  - SAM 3 returns `[N, O, 1008, 1008]` logits.
  - `comfy.ldm.sam3.tracker.unpack_masks` unpacks the bits.

## 3. Design

One shared module `segmenters/_core_sam3.py`. It is called at run time; nothing of core is copied (core is GPL-3.0,
CNP Apache-2.0).

- `available()`: core ships SAM 3 and the kill switch is not set.
- `load(path)`: one model in memory, reused, released by the memory guard.
- `encode(clip, text)`
- `detect_words(...)` and `segment_prompts(...)`: the logic the viewer's Detect has used since L7.114, moved here.
  Detect calls it.
- `track(handle, frames, init_masks)`: forward only, both output forms, masks back at plate size.
- `CoreVideoTracker`: ONYX's tracker protocol on top of `track`.
  - `add_prompt` keeps each object's keyframes. A new `add_mask` takes a seed mask directly; ONYX's text and mask
    keyframes use it instead of reducing the mask to points.
  - On `propagate_in_video`, each object's mask is built on its keyframes (SAM decoder from points / box, or the given
    mask). Each segment runs from a keyframe to the next keyframe of that object, or to the range end. Objects with the
    same segment run in one `forward_video` call (SAM 3.1 tracks up to 16 at once).
  - Backward = the same on the reversed frames.
  - Yields `(frame, {"out_obj_ids", "out_binary_masks", "out_probs"})` like sam3_lib.

Where it plugs in:

| Path | With sam3_lib / the sam3 package | Without them, core SAM 3 present |
|---|---|---|
| cascade `sam3.1` / `sam3` | unchanged | core, per frame (the same semantics as today) - see section 5 for tracking |
| ONYX | unchanged | `CoreVideoTracker` (instead of the model-free fallback) |
| STATUS badge | ready | ready (was `[missing-deps]`; VALIDATE_INPUTS already strips a saved badge, L2.14) |

## 4. Failure modes and limits

- No checkpoint: the existing "checkpoint is not installed: put <file> in <folder>" sentence.
- A file that is not a SAM 3 checkpoint: Detect's sentence (name the Comfy-Org file).
- No core SAM 3 (ComfyUI < 0.39) and no sam3_lib: unchanged (`[missing-deps]`, ONYX falls back to model-free).
- Memory: CPU fp32 is 2.7 GB resident after load (measured, F7.114).
  - `forward_video` keeps the plate on the CPU and moves one frame at a time.
  - Masks come back bit-packed (SAM 3.1).
- Interrupt: `forward_video` runs core's progress bar, so ComfyUI's cancel stops between frames.
- Kill switch: `C2C_SAM_CORE=0` turns the core path off everywhere in Mask + Matting, which then behaves as before.
  The viewer's Detect keeps its own path (L7.114).

## 5. Measured before shipping (owner condition)

Plate: the first 20 frames of ComfyUI-MatAnyone's `test-sample1.mp4` at 640 x 360. References: the same repo's
`joker_00000.png` and `joker_00019.png` (that clip's subject masks at frames 0 and 19; SAM played no part in making
them). Prompts: one click and the reference box on frame 0. Measured per backend:

- the IoU at frame 19;
- the seconds;
- the peak RAM.

Cases: cascade `sam3.1` per frame (core), cascade `sam3.1` tracked (core), ONYX (core tracker), and ONYX model-free (the
fallback today). Per-frame stays the cascade's default unless tracking measures better; the numbers go in
`docs/evidence/L7.125/`.

Measured 2026-10-29 (CPU, 20 frames; workspace `docs/evidence/L7.125/core_tracking.md`):

| Case | IoU f0 | IoU f19 | s |
|---|---:|---:|---:|
| per frame | 0.979 | 0.983 | 650 |
| tracked | 0.979 | 0.986 | 570 |
| ONYX core tracker | 0.979 | 0.985 | 576 |
| ONYX model-free | 0.952 | 0.778 | 6 |

As built:
- ONYX tracks with core when the library is absent.
- The cascade stays per frame: tracking was only marginally better on a clip where the subject barely moves, and
  tracking would change what an existing workflow does.
