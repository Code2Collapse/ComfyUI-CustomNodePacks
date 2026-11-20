# Control AOV

## Control AOV — Multi-Control Fusion (C2C) (`ControlAOVC2C`)
**Menu:** 🐺 C2C/🧰 Core/Control · **Parameters:** [NODE_REFERENCE](../NODE_REFERENCE.md#controlaovc2c)
**Needs:** OpenCV for the built-in edge and motion passes. Every model pass (depth, pose, normals, ID matte) needs
an extra package, weights or another node pack (see [What each pass needs](#what-each-pass-needs)). DA3, Depth Pro
and DVD have NOT been VERIFIED with real weights (commit 3a6053a). DepthCrafter and NormalCrafter are marked "pending
a live GPU verification" in the code.

### What it is for
The node makes ControlNet control images and keeps each kind as its own pass, like render passes (AOVs) in
compositing: depth, edges, pose, normals, motion and an ID matte. It also gives three combined forms:
- `blended`: all passes overlaid in one image;
- `channel_packed`: depth in red, edges in green, pose in blue;
- `combined`: depth, pose and edges side by side, for a quick look.

Each pass can come from a model the node runs, or from a map you wire in.

### When to pick it
- Pick it when you want several control signals from one source, kept as separate passes, at one size.
- If you only need one map (for example just a depth map), a single preprocessor node does the same with fewer
  controls.

### Minimal wiring
```
Load Image / Load Video ──► image ─┐
(optional) your own depth ► depth ─┼► Control AOV ──► depth ──► Apply ControlNet (depth)
                                   │              ├──► canny ──► Apply ControlNet (canny)
                                   │              ├──► pose  ──► Apply ControlNet (pose)
                                   │              ├──► channel_packed ──► union ControlNet
                                   │              └──► info  ──► Preview Any
```
With only `image` wired and nothing else changed, the node runs the built-in Canny edge pass (`edge_model` =
`internal_canny`). Every other pass is off by default.

### What each pass needs

| Pass | Setting | What runs | What it needs |
|---|---|---|---|
| Edges | `edge_model` = `internal_canny` (default) | OpenCV Canny in the node | `opencv-python`; no model |
| Edges | `canny`, `lineart`, `anyline`, `hed`, `pidinet`, `teed` | the comfyui_controlnet_aux node of that name | comfyui_controlnet_aux installed; if it fails, falls back to the built-in Canny |
| Motion | `run_motion` on | OpenCV Farneback flow between neighbouring frames | `opencv-python`; a batch of 2+ frames (the first frame has zero motion) |
| Depth | `da_v2`, `da_v1`, `midas` | Hugging Face `transformers` depth pipeline in the node | `transformers`; the model in the Hugging Face cache, or `allow_download` on |
| Depth | `da3` | Depth Anything 3 | the `depth_anything_3` package (`pip install git+https://github.com/ByteDance-Seed/Depth-Anything-3`) and its weights |
| Depth | `depth_pro` | Apple Depth Pro: the `depth_pro` package if installed, else `transformers` with `apple/DepthPro-hf` | weights; Apple research licence, non-commercial |
| Depth | `depthcrafter` (video) | DepthCrafter, code shipped in this pack | `diffusers`; `tencent/DepthCrafter` plus the Stable Video Diffusion base `stabilityai/stable-video-diffusion-img2vid-xt` (about 10 GB, commit 3a6053a) |
| Depth | `dvd` (video) | DVD (Wan 2.1 based) | the DVD code (clone into `C2C_DVD_DIR` or `third_party/DVD`); weights `FayeHongfeiZhang/DVD` in `models/DVD/` (CC BY-NC 4.0); DiffSynth, the Wan 2.1 base model and a large GPU |
| Depth | `depth_anything_v2`, `depth_anything_v1`, `depth_anything_v2_metric`, `zoe_depth_anything` | the comfyui_controlnet_aux node | comfyui_controlnet_aux and its model; if it fails, falls back to the in-node `da_v2` |
| Pose | `vitpose` | the ViTPose chain of ComfyUI-WanAnimatePreprocessV2 | that pack, a ViTPose `.onnx` and a YOLO person-detector `.onnx` in `models/detection/` |
| Pose | `dwpose`, `openpose`, `animal_pose`, `densepose` | the comfyui_controlnet_aux node | comfyui_controlnet_aux; if it fails, falls back to `vitpose` |
| Normals | `sobel_from_depth` | normals worked out from the depth pass | no model; a depth pass (made by the node or wired in) |
| Normals | `normalcrafter` (video) | NormalCrafter, code shipped in this pack | `diffusers`; `Yanrui95/NormalCrafter` plus the same Stable Video Diffusion base |
| ID matte | `id_matte_model` = `sam_auto` | the comfyui_controlnet_aux SAMPreprocessor node | comfyui_controlnet_aux and its SAM model |

Licences named in the code's tooltips:
- **Non-commercial:** Depth-Anything-V2 base / large, DA3 large / giant, DVD, Depth Pro.
- **Apache-2.0:** Depth-Anything-V2 small.

**Without any extra package or model** you still get:
- the built-in Canny edges and motion vectors (with OpenCV);
- `sobel_from_depth` normals from a depth map you wire in;
- every pass you wire into the `depth`, `canny`, `pose`, `normal` and `id_matte` sockets;
- `blended`, `channel_packed` and `combined` built from those passes, and the in-node resize.

### Downloads
`allow_download` is off by default. While it is off, the in-node backends use only copies that are already installed
(a local folder or the Hugging Face cache). When something is missing, the backend says so in its report. Turn
`allow_download` on to let them fetch missing weights from Hugging Face; some are many GB.

The pack-wide never-download switch (environment variable `C2C_NO_DOWNLOADS=1`) wins over `allow_download`.
`allow_download` has no effect on the comfyui_controlnet_aux and WanAnimatePreprocessV2 nodes: they handle their own
models.

Note: the last part of the `info` text still says the in-node depth models "download on first use". Since commit
3a6053a that happens only with `allow_download` on.

### Key controls
- **A wired map wins.** A map in the `depth`, `canny`, `pose`, `normal` or `id_matte` socket is used instead of the
  model for that pass. `depth_invert` and `depth_weight` still apply to it.
- `depth_size` picks the backbone size for the Depth Anything family.
  - For `da_v2`, `da_v1` and `midas`, only `small` and `large` pick different models. `medium` and `giant` are not
    in the code's model table for these backends and use the small model.
  - For `da3`, `small` uses `DA3METRIC-LARGE` and every other size uses `DA3NESTED-GIANT-LARGE`.
- `depth_weight`, `canny_weight`, `pose_weight` and `normal_weight` change only the `blended` output. 0 leaves that
  pass out. `normal_weight` defaults to 0.
- `blend_mode` (default `screen`) sets how `blended` combines the passes.
- `match_to` (default `image`) is the source whose size every pass is resized to. After that, `resize` can resize
  all outputs (`width/height` or `scale`, snapped to `divisible_by`, with `fit` stretch / pad / crop).
- `preproc_resolution` (default 512) is passed to the delegated preprocessor nodes.

### Reading the outputs
- `depth`, `canny`, `pose`, `normal`, `motion`, `id_matte`: the separate passes. A pass that is off comes out as a
  black image of the same size, so the socket can stay wired.
- `channel_packed`: the brightness of depth, edges and pose in R, G and B.
- `blended`: the weighted overlay. `combined`: depth | pose | edges (or the `grid_2x2` layout, which adds the
  original).
- `info`, first line: `STATUS: ALL 3 OK` or `STATUS: CHECK BELOW`, then `depth=… canny=… pose=…`, `normal=…` and
  `id=…`, where:
  - `OK` = the pass has content;
  - `EMPTY` = the pass came out (almost) black;
  - `FAIL` = it was requested and produced nothing;
  - `off` = not requested.

  `ALL 3 OK` needs depth, edges and pose all `OK`. With one of them off, the status reads `CHECK BELOW` even when
  nothing is wrong.
- `info`, second line: the passes present, the blend mode and the size. A `backends:` line follows with each
  backend's own report (for example `hf_depth ok [depth-anything/Depth-Anything-V2-Small-hf]`, or what is missing).

### Common mistakes
- **A model pass is selected but its backend is missing.** The run stops with "Control AOV (C2C): these passes were
  requested but produced nothing: … Backend report: …". Install what the report names, wire your own map into that
  socket, or set that model to `off`. Before 2026-11-04 such a pass silently became a black map.
- **No OpenCV.** The built-in edge pass and the motion pass stop the run with "needs opencv-python".
- **`sobel_from_depth` without depth.** It needs a depth pass. The error says "needs a depth pass".
- **Motion on a single image.** The motion pass needs a batch of at least 2 frames.
- **`blended` as the only control.** The node's own `info` advises the separate passes, each into its own Apply
  ControlNet at different weights, rather than relying on `blended`.
