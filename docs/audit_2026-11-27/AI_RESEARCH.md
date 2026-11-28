# AI_RESEARCH — CNP (ComfyUI-CustomNodePacks, Apache-2.0): segmentation, matting, tracking, inpainting, depth/flow

Date: 2026-11-27. Scope: brief `D:\PROJECT\Custom_Nodes\new_tas.md` Phase 4 plus the matting / roto / tracking
questions of Phase 5. Output only; no repo file was edited, no weights were downloaded.

**Skills applied**
- `deep-dive` (C:\Users\varig\.claude\skills\deep-dive\SKILL.md): the plan was split into sub-questions with an order:
  baseline, then licences, then published numbers, then synthesis, followed by one round of gap filling. Two parts of
  the skill were not used. The research ran inline, with no parallel subagents, because this run was already the
  delegated agent and the global policy says "don't spawn unless asked". The report was saved here rather than in a
  project `docs/deep-dive/`, because the owner's brief fixed the output path and forbade repo edits.
- `karpathy-guidelines` (C:\Users\varig\.claude\skills\karpathy-guidelines\SKILL.md): assumptions are stated, each
  recommendation is the smallest change that solves the problem, every item carries a verifiable success criterion,
  and no code was changed.

**Evidence rules used in this report**
- **PUBLISHED** means a number copied from the cited paper or README table. Nothing in this report was **measured
  locally**: no model was run, no benchmark was executed and no VRAM was profiled on the owner's hardware.
- **VERIFIED** next to a licence means it was read on 2026-11-27 from the actual source: the GitHub `license` API plus
  the decoded LICENSE text, or the Hugging Face model API `license:` tag and model card. "HF tag" means the Hub
  metadata tag only.
- All 37 arXiv IDs cited were checked against the arXiv API (titles match). Repo existence was checked through the
  GitHub API.
- I am not a lawyer. Licence conclusions that matter for shipping are marked **NEEDS_OWNER** (standing order 9).

---

## 0. Executive summary

1. **CNP's segmentation surface is already broad.** It has SAM 3 and SAM 3.1 (core path plus a vendored path),
   BiRefNet, RMBG, DIS, Grounding-DINO, LocateAnything, Cutie, XMem, SeC, MatAnyone, MatAnyone 2, RVM, ViTMatte,
   BGMv2, guided filter and SAMRefiner-style refinement. The biggest gaps are not "more models". They are:
   - (a) correct **negative semantics**: soft subtraction, negative text, negative exemplars;
   - (b) **foreground-colour estimation**: real edge decontamination, not only despill;
   - (c) an **evaluation harness** (IoU / boundary-F / SAD / Grad / Conn / dtSSD / MESSDdt), so that "better" can be
     measured;
   - (d) **licence hygiene**: several defaults point at non-commercial weights or code.
2. **Core ComfyUI 0.46.3 already provides several pieces CNP should reuse, not duplicate:**
   - SAM 3 / 3.1 (`SAM3_Detect`, `SAM3_VideoTrack`, `SAM3_TrackPreview`, `SAM3_TrackToMask`);
   - BiRefNet background removal (`LoadBackgroundRemovalModel` / `RemoveBackground`);
   - Depth Anything 3 (`LoadDA3Model`, `DA3Inference`, and others);
   - RAFT-large optical flow (`OpticalFlowLoader`, torchvision BSD weights);
   - VOID object-and-interaction removal, `VideoDeflicker` and RT-DETR detection.
3. **Licence-clean, high-value options (VERIFIED):**
   - SAM 2.1 (Apache-2.0 code and weights); SAM 3 / 3.1 (SAM License, permissive with conditions);
   - SAMRefiner code (MIT, now verified; CNP's file says "unchecked"); HQ-SAM (Apache-2.0);
   - ViTMatte (MIT code, Apache-2.0 HF weights); BiRefNet (MIT); Cutie and XMem (MIT); SEA-RAFT (BSD-3, BSD weights);
   - Video-Depth-Anything-**Small** (Apache-2.0); DA3-SMALL / BASE / METRIC-LARGE / MONO-LARGE (Apache-2.0);
   - VACE-Wan2.1 (Apache-2.0); VOID (Apache-2.0 code and weights, but a CogVideoX-Fun base with its own licence);
   - DiffuEraser (Apache-2.0, but it **requires ProPainter weights**, which are S-Lab non-commercial).
4. **Non-commercial or research-only (VERIFIED). These must stay opt-in, labelled, and never be defaults:**
   - MatAnyone, MatAnyone 2 and ProPainter (S-Lab License 1.0, non-commercial);
   - RMBG-2.0 (CC BY-NC 4.0); DEVA (CC BY-NC-SA 4.0); MaGGIe (CC BY-NC 4.0); ZIM (CC BY-NC 4.0);
   - VideoMaMa (CC BY-NC 4.0); MiniMax-Remover weights (CC BY-NC 4.0 HF tag);
   - Depth Pro weights (Apple ML Research licence, research only);
   - DA3-GIANT / LARGE / NESTED, VDA-Base / Large, DA-V2-Base / Large (CC BY-NC 4.0);
   - DepthCrafter (Tencent "academic, research and education only", **code and weights**);
   - RVM (GPL-3.0: fine to call at run time, never port into Apache).
5. **Licence and code findings in CNP itself:**
   - (i) `nodes/depthcrafter/` **commits Tencent DepthCrafter code whose licence forbids commercial or production
     use** inside the Apache-2.0 repo. It has its own LICENSE file, so it is attributed, but it is not
     Apache-compatible. **NEEDS_OWNER.**
   - (ii) `segmenters/video_backend.py` describes **VideoMaMa** as "Mamba-based VOS" and points at
     `yyk-wew/VideoMamba`. That repo returns 404 on the GitHub API. The real VideoMaMa (arXiv 2601.14255,
     `cvlab-kaist/VideoMaMa`) is a **mask-guided video matting** model under CC BY-NC 4.0. The backend is mis-specified.
   - (iii) `_control_backends._DA3_MODELS` maps size "small" and "base" to `DA3METRIC-LARGE` although
     Apache-2.0 `DA3-SMALL` / `DA3-BASE` exist. Its default "large" maps to the NC `DA3NESTED-GIANT-LARGE`, although a
     tooltip does warn about this.
   - (iv) `segment_refine.py` says the SAMRefiner repo licence is "unchecked". It is MIT (VERIFIED), so the
     context-aware elastic box (CEBox) can now be ported with attribution instead of being skipped.

---

## 1. Baseline: what CNP and core already do (read-only inspection)

### 1.1 CNP (`D:\PROJECT\Custom_Nodes\ComfyUI-CustomNodePacks\nodes`)

| Area | Implementation (file) | Notes |
|---|---|---|
| SAM 3 / 3.1 via core | `mask_matting/segmenters/_core_sam3.py` (CoreSAM3: detect, segment_union, mask_from_prompts, track; CoreVideoTracker) | Calls core at run time; core code is not copied (core is GPL-3.0). Kill switch `C2C_SAM_CORE=0`. Recommends `sam3.1_multiplex_fp16.safetensors` (Comfy-Org). |
| SAM 3.1 vendored | `mask_matting/segmenters/sam31_backend.py` (uses `third_party/sam3_lib`, git-ignored, licence copy at `THIRD_PARTY_LICENSES/SAM3-LICENSE.txt`) | **Native negative exemplar boxes** (`add_multiple_box_prompts(..., labels False)`), text, points. |
| SAM 3 pip package | `mask_matting/segmenters/sam3_backend.py` | Falls back to core. |
| Negative box on core path | `_core_sam3.segment_union` lines ~203-240 | Core's geometry encoder hard-codes `box_labels = ones` (`comfy/ldm/sam3/detector.py` ~line 301). CNP's workaround: with text, it **drops** detections centred in the negative box; without text, it **subtracts a binary SAM mask** of the negative box. This is a hard subtraction, not soft. |
| Negative text prompt | none found (grep for `negative_text`, `exclude_text` and similar) | Phase-5 "Positive: person / Negative: necklace, microphone" is not implemented as a text path. |
| SAMRefiner-style refine | `segment_refine.py` (opt-in, EXPERIMENTAL), wired in `segment_c2c.py` slice 3 | Distance-guided points and a Gaussian mask prompt, written from the paper. **CEBox omitted** (λ=0). |
| SAM 2 (vendored inside SeC) | `sec_engine/sam2/...` (Apache-2.0, notices kept) | Used by the SeC engine. |
| Video object segmentation | `mask_matting/segmenters/video_backend.py`: Cutie, XMem, SeC, "VideoMaMa" | Cutie and XMem need user-installed packages ("missing-deps" otherwise). VideoMaMa is mis-specified (see 0.5 ii). |
| Matting | `mask_matting/matters/`: vitmatte (tiled, 709 lines), matanyone, rvm (torch.hub at run time), bgmattingv2, birefnet/rmbg soft alpha, guided_filter | Trimap from `utils.mask_to_trimap` uses a **fixed erode/dilate** (default 8/8 px), with no uncertainty or edge awareness. |
| ONYX pipeline | `mask_matting/_onyx_pipeline.py` | SAM 3.1 video session plus tiled ViTMatte. |
| SeC + MatAnyone 2 | `sec_matanyone_pipeline.py` | SeC-4B (HF tag Apache-2.0) plus MatAnyone 2 (S-Lab NC). |
| Edge handling | `mask_matting/node.py` (despill on a named backing colour, `preserve_skin`, lightwrap, premultiplied output), `_mask_edge_refine.py` | **No foreground-colour estimation** (multi-level / blur-fusion). Despill is not decontamination of a natural background. |
| Inpainting | `propainter_bridge.py`, `propainter_temporal_inpaint.py`, `propainter_flow_refine.py`, `propainter_unified.py`, `propainter_stitch_suite.py` | ProPainter is **not vendored**: an existing checkout goes on `sys.path` and weights come from ProPainter releases. The licence copy is in `THIRD_PARTY_LICENSES/ProPainter-LICENSE.txt`. |
| Optical flow | `optical_flow.py` (OpticalFlowMEC), using `propainter_flow_refine.compute_flow` | RAFT comes from the **ProPainter checkout** (`raft-things.pth`), with a Lucas-Kanade pyramid as fallback. |
| Depth / normals | `_control_backends.py` + `control_forge.py`: DA V1/V2 (transformers), DA3 (pip package), Depth Pro (HF `apple/DepthPro-hf`), DepthCrafter (**vendored** `nodes/depthcrafter/`), NormalCrafter (vendored `nodes/normalcrafter/`, MIT code), "DVD" | The file header says the repo-based runners are "pending a live GPU verification". |
| Metrics | none found in `nodes/` (grep for dtSSD, MESSDdt, boundary F, IoU helpers) | There is no quality harness, so no matting choice is measured. |

### 1.2 Core ComfyUI 0.46.3 (`D:\PROJECT\ComfyUI_windows_portable\ComfyUI`, `comfyui_version.py` = 0.46.3)

| Core node(s) | File | What CNP should do |
|---|---|---|
| `SAM3_Detect` (text conditioning, bboxes, positive/negative point JSON, threshold, refine_iterations, individual_masks), `SAM3_VideoTrack` (initial_mask, text conditioning, detect_interval, max_objects ≤64), `SAM3_TrackPreview`, `SAM3_TrackToMask` | `comfy_extras/nodes_sam3.py`, `comfy/ldm/sam3/` | Keep calling core (as `_core_sam3.py` does). Do not re-ship a second SAM 3 loader except for features core lacks: negative exemplar boxes, presence score, per-object soft logits. |
| `LoadBackgroundRemovalModel` / `RemoveBackground` (BiRefNet architecture) | `comfy_extras/nodes_bg_removal.py`, `comfy/bg_removal_model.py` | CNP's BiRefNet segmenter could call core's loader instead of its own path. |
| `LoadDA3Model`, `DA3Inference`, `DA3Render`, `DA3GeometryToMesh`, `DA3GeometryToPointCloud` | `comfy_extras/nodes_depth_anything_3.py` | Control AOV's DA3 runner duplicates this. Prefer core when present. |
| `OpticalFlowLoader` (torchvision `raft_large.pth`, BSD) + VOID nodes (`VOIDQuadmaskPreprocess`, `VOIDInpaintConditioning`, `VOIDWarpedNoise`, `VOIDSampler`) | `comfy_extras/nodes_void.py`, `comfy/ldm/raft/model.py` | CNP's RAFT dependency on a ProPainter checkout can be replaced by core's RAFT. VOID is a core object-removal path; CNP should not wrap it again. |
| `VideoDeflicker` | `comfy_extras/nodes_video_deflicker.py` | Relevant to matte flicker. Check its contract before building a matte deflicker. |
| `RTDETR_detect` | `comfy_extras/nodes_rtdetr.py` | A closed-set detector already in core. |

---

## 2. Licence matrix (VERIFIED on 2026-11-27 unless marked)

| Project | Code licence | Weight licence | Commercial-clean? | Source of check |
|---|---|---|---|---|
| SAM 3 / SAM 3.1 (`facebookresearch/sam3`) | "SAM License" (Nov 19 2025): royalty-free; use, modify and redistribute allowed; redistribution only under the same agreement; no ITAR or military use; no reverse-engineering clause | Same SAM License. `facebook/sam3` and `facebook/sam3.1` are **gated (manual)**; `Comfy-Org/sam3.1` has HF tag `other` and is not gated | Yes, with conditions (attach the agreement, follow trade-control terms). NOT Apache: do not relicense derived code as Apache | GitHub LICENSE text; HF API |
| SAM 2 / 2.1 | Apache-2.0 | `facebook/sam2.1-hiera-large` Apache-2.0 | Yes | GitHub API, HF API |
| HQ-SAM (`SysCV/sam-hq`) | Apache-2.0 | `syscv-community/sam-hq-vit-huge` Apache-2.0 | Yes | GitHub, HF |
| SAMRefiner (`linyq2117/SAMRefiner`) | **MIT** | Uses the SAM ViT-H (Apache-2.0) checkpoint; no weights of its own | Yes | GitHub API |
| Grounding DINO (`IDEA-Research/GroundingDINO`) | Apache-2.0 | `grounding-dino-tiny/base` Apache-2.0 | Yes | GitHub, HF |
| Grounded-SAM-2 | Apache-2.0 | n/a | Yes | GitHub |
| Florence-2 | n/a (HF) | `microsoft/Florence-2-base/large` MIT | Yes | HF |
| LocateAnything-3B (NVIDIA) | n/a | HF tag `other` | **Unknown: NEEDS_OWNER** | HF |
| BiRefNet (`ZhengPeng7/BiRefNet`) | MIT | `BiRefNet`, `_HR`, `-matting`, `_dynamic`: MIT | Yes | GitHub, HF |
| RMBG-2.0 (`briaai/RMBG-2.0`) | n/a | **CC BY-NC 4.0**; "Commercial use is subject to a commercial agreement with BRIA"; BiRefNet architecture | **No** | HF model card |
| ViTMatte (`hustvl/ViTMatte`) | MIT | `vitmatte-small/base-composition-1k`, `-distinctions-646`: Apache-2.0 | Yes (note: trained on Adobe Composition-1k, a research dataset) | GitHub, HF |
| MatAnyone (`pq-yang/MatAnyone`) | **S-Lab License 1.0 (non-commercial)** | HF card has no licence tag; the code licence governs | **No** | GitHub LICENSE text |
| MatAnyone 2 (`pq-yang/MatAnyone2`) | **S-Lab License 1.0 (non-commercial)** | `PeiqingYang/MatAnyone2` has no tag | **No** | GitHub LICENSE text |
| Robust Video Matting | **GPL-3.0** | Distributed with the repo | Run-time use only; never port into Apache CNP | GitHub API |
| BackgroundMattingV2 | MIT | (repo) | Yes | GitHub API |
| Matte-Anything (`hustvl/Matte-Anything`) | MIT | Composite of SAM + GroundingDINO + ViTMatte | Yes (if its components are) | GitHub API |
| MaGGIe (`hmchuong/MaGGIe`) | **CC BY-NC 4.0** (LICENSE text) | HF video card shows cc-by-4.0 (web search, not verified) | **No** (code) | GitHub LICENSE text |
| ZIM (`naver-ai/ZIM`) | **CC BY-NC 4.0** | n/a | **No** | GitHub LICENSE text |
| VideoMaMa (`cvlab-kaist/VideoMaMa`) | **CC BY-NC 4.0** | n/a | **No** | GitHub LICENSE text |
| GVM (`aim-uofa/GVM`) | **No licence file** (GitHub API returns none) = all rights reserved | n/a | **No** | GitHub API |
| Cutie | MIT | Repo releases (no separate weight licence found: verify) | Yes (code) | GitHub API |
| XMem | MIT | Repo releases (same caveat) | Yes (code) | GitHub API |
| DEVA | **CC BY-NC-SA 4.0** | n/a | **No** | GitHub LICENSE text |
| SeC (`OpenIXCLab/SeC`) | Apache-2.0 | `OpenIXCLab/SeC-4B` Apache-2.0 | Yes | GitHub, HF |
| ProPainter | **S-Lab License 1.0 (non-commercial)** | Same | **No** | GitHub LICENSE text |
| DiffuEraser | Apache-2.0 "except for the third-party components" | `lixiaowen/diffuEraser` Apache-2.0, but it **needs ProPainter weights as its prior** ("Users must comply with Propainter's license") plus SD-1.5 | **No as shipped** (the ProPainter prior is NC). The README says "you can use other model to replace it" | GitHub README, HF |
| MiniMax-Remover (`zibojia/MiniMax-Remover`) | **No LICENSE file** in the repo (GitHub API returns none) | `zibojia/minimax-remover` **CC BY-NC 4.0** (HF tag) | **No** | GitHub API, HF |
| FloED (arXiv 2412.00857) | Code release not confirmed | n/a | Unknown | arXiv, web search |
| VACE (`ali-vilab/VACE`) | Apache-2.0 | Wan2.1-VACE-1.3B / 14B Apache-2.0; VACE-LTX-Video-0.9 RAIL-M | Yes (Wan variants) | GitHub README, HF |
| VOID (`netflix/void-model`) | Apache-2.0 | `netflix/void-model` Apache-2.0; base `alibaba-pai/CogVideoX-Fun-V1.5-5b-InP` HF tag `other` | Conditional: **NEEDS_OWNER** for the CogVideoX base licence | GitHub, HF |
| Depth Anything 3 | Apache-2.0 | DA3-SMALL, DA3-BASE, DA3METRIC-LARGE, DA3MONO-LARGE: **Apache-2.0**. DA3-LARGE, DA3-GIANT, DA3NESTED-GIANT-LARGE: **CC BY-NC 4.0** | Only the Apache ones | GitHub README, HF |
| Depth Anything V2 (HF) | n/a | Small Apache-2.0; Base and Large **CC BY-NC 4.0** | Small only | HF |
| Video Depth Anything | Apache-2.0 | Small **Apache-2.0**; Base / Large / Metric-Large **CC BY-NC 4.0** | Small only | GitHub README, HF |
| Depth Pro (`apple/ml-depth-pro`) | Apple sample-code licence (permissive) | `apple/DepthPro` and `-hf`: **apple-amlr: "exclusively for Research Purposes ... does not include any commercial exploitation"** | **No** (weights) | GitHub LICENSE, HF LICENSE |
| DepthCrafter (`Tencent/DepthCrafter`) | **"only for academic, research and education purposes ... refrain from using it for any commercial or production purposes"**. The definition of "Software" covers inference code **and weights** | Same (HF `license: other`, links LICENSE) | **No** | GitHub LICENSE text, HF README |
| NormalCrafter (`Binyr/NormalCrafter`) | MIT | `Yanrui95/NormalCrafter` Apache-2.0 tag; fine-tuned from Stable Video Diffusion (SVD HF tag `other`) | Probably yes for the NormalCrafter delta; **NEEDS_OWNER** on SVD inheritance | GitHub, HF |
| RAFT | BSD-3-Clause | Core uses torchvision `raft_large` (BSD) | Yes | GitHub API |
| SEA-RAFT | BSD-3-Clause | `MemorySlices/Tartan-C-T-TSKH-spring540x960-M` BSD-3-Clause | Yes | GitHub, HF |
| MemFlow (`DQiaole/MemFlow`) | Apache-2.0 | Repo-hosted | Yes (code) | GitHub API |
| pymatting (foreground estimation) | MIT | n/a (no weights) | Yes | GitHub API |
| FBA Matting | MIT | (repo) | Yes (code) | GitHub API |

---

## 3. Segmentation

### 3.1 SAM 3 / SAM 3.1 (concept segmentation, exemplars, video)

- **Problem it solves:** open-vocabulary "all instances of a concept", with text and/or **image exemplars**
  (positive or negative boxes), plus a SAM 2-style tracker for video.
- **CNP today:** described in 1.1. Exemplar boxes are native only on the vendored path. The core path emulates a
  negative box by dropping detections or subtracting a hard mask. There is no negative-text path.
- **What it changes, PUBLISHED (SAM 3 paper, arXiv 2511.16719 v2):**
  - Exemplars: "a bounding box and an associated binary label (positive or negative)". "False positive and false
    negative objects can be removed or added respectively using image exemplars." Constraint: "All prompts must be
    consistent in their category definition, or the model's behavior is undefined". So a negative *exemplar* removes
    other instances **of the same concept**. It is not a way to carve a necklace out of a person.
  - Table 5, VOS (SAM 2.1 L → SAM 3): MOSEv1 val 77.9→78.4; DAVIS17 val 90.7→92.2; LVOSv2 79.6→88.5; SA-V val
    77.9→83.5; SA-V test 78.4→84.4; YTVOS19 89.3→89.7; MOSEv2 47.9→60.3. The metric symbol in the HTML header was
    garbled; VOS convention is J&F.
  - Table 6, interactive SA-37 mIoU at 1/3/5 clicks: SAM 2.1 L 66.4/80.3/84.3 at 93.0 FPS; SAM 3 66.1/81.3/85.1 at
    43.5 FPS. **SAM 3 is about 2x slower per click than SAM 2.1** for point-only work.
  - Speed: "On an H200 GPU, SAM 3 runs in 30 ms for a single image with 100+ detected objects". Video latency "scales
    with the number of objects ... near real-time ... ∼5 concurrent objects". 848M parameters (README).
  - Presence head ablation, Table 8a: cgF1 50.7→52.2.
- **SAM 3.1 (RELEASE_SAM3p1.md, 2026-03-27, PUBLISHED):** Object Multiplex (shared-memory joint multi-object tracking).
  "~7x speedup" at 128 objects on one H100. Accuracy is mixed: SA-V cgF1 30.3→30.5, YT-Temporal-1B 50.8→52.9, MOSEv2
  VOS 60.3→62.3, LVVIS mAP 36.3→34.3, BURST HOTA 44.5→43.3.
- **Runtime and VRAM:** not published as a VRAM figure. Requirements: Python 3.12+, PyTorch 2.7+, CUDA 12.6+.
- **Dependencies:** none beyond core when the core path is used.
- **Licence:** SAM License (section 2).
- **Integration complexity:** low on the core path (already done). Exposing **negative exemplar boxes** on the core
  path needs the box labels passed to `_encode_boxes`, which core hard-codes to 1. Options: (a) an upstream core PR
  (preferred, no monkey-patching); (b) CNP calls `geometry_encoder._encode_boxes` directly with its own labels.
  Option (b) is brittle across core updates, so guard it and fall back.
- **Limitations:** a negative exemplar is category-level, not part-level. SAM 3 is weaker than SAM 2.1 on 1-click.
  Gated Meta checkpoints.
- **Worth it:**
  - YES: negative exemplars on the core path, and a "presence score" readout.
  - NO: a third loader.
  - YES: keep the SAM 2.1 point path for pure click work, because it is faster per click (Table 6).

### 3.2 SAM 2.1

- **Problem it solves:** a fast promptable image/video segmenter with an Apache-2.0 licence.
- **CNP today:** vendored only inside SeC (`sec_engine/sam2`). There is no first-class SAM 2.1 segmenter key in Mask
  + Matting.
- **PUBLISHED:** SAM 3 paper Table 6: SAM 2.1 L reaches 93.0 FPS with 1-click mIoU 66.4, against 43.5 FPS for SAM 3.
- **Licence:** Apache-2.0 code and weights.
- **Integration complexity:** low, because the vendored copy already exists.
- **Worth it:** LATER. Only useful as a *fast interactive* fallback; core SAM 3.1 covers quality.

### 3.3 HQ-SAM

- **Problem it solves:** SAM's blobby boundaries on thin structures.
- **PUBLISHED (arXiv 2306.01567, Table 2, ViT-L):** average over DIS/COIFT/HRSOD/ThinObject, mIoU 79.5→89.1 and mBIoU
  71.1→81.8. DIS 62.0→78.6 and ThinObject 73.6→89.5. Cost: +5.1M learnable parameters; 4.8 vs 5.0 FPS; both use 7.6G
  memory.
- **CNP today:** "sam_hq" is mentioned in `model_manager.py` and `sam_multi_mask_picker.py`. It is not a Mask +
  Matting backend key.
- **Licence:** Apache-2.0.
- **Limitations:** SAM-1 based, with no text and no video. Its gains overlap with what a matting stage provides.
- **Worth it:** LATER. A matting stage (ViTMatte) after SAM gives most of the thin-structure gain for the soft alpha.
  HQ-SAM helps only the *binary* mask (useful for roto shapes and for trimap seeding).

### 3.4 SAMRefiner (ICLR 2025, arXiv 2502.06756; code MIT)

- **Problem it solves:** coarse masks (from any source) refined by SAM, using prompts mined from the mask itself.
- **CNP today:** `segment_refine.py` implements distance-guided points, the tight box and a Gaussian mask prompt. It
  omits CEBox and the IoU adaption (SAMRefiner++).
- **PUBLISHED (Table 1, DAVIS-585, SAMRefiner / ++):**

  | Prompt | IoU | boundary IoU |
  |---|---|---|
  | Coarse mask | 81.4 | 71.4 |
  | Box + Mask | 84.6 / 85.1 | 74.2 / 75.4 |
  | All | 86.9 / 87.1 | 75.1 / 75.4 |

  - Table 2: point sampling Random 37.8, Box Center 22.2, **Distance-Guided 53.7** IoU. CEBox APbox 36.7→38.2 (small).
  - Table 5 (DAVIS-585 IoU): CRF 81.0, CascadePSP 81.9, CRM 82.9, SegRefiner 80.3, SAMRefiner 87.1. Refining COCO
    train5K takes 0.6 h total against 1.0–3.4 h for the others.
  - Limitation (paper): "still fails to work when the initial masks are extremely noisy". SAMRefiner++ "is
    dataset-dependent".
- **What the alternative changes:**
  - (a) Port **CEBox** from the MIT code (attribute it). It needs image-embedding similarity, which the core SAM 3
    wrapper can reach via `forward_segment` internals or by running the backbone once.
  - (b) Add the paper's selection rule: keep the refined mask only if its IoU with the coarse mask is above a
    threshold, else revert. This matches the paper's caution about noisy inputs.
- **Runtime and VRAM:** one SAM decode per iteration; negligible next to the backbone.
- **Worth it:** YES for (b), which is cheap and protects quality. LATER for CEBox, whose published gain is small
  (+1.5 APbox).

### 3.5 Open-vocabulary detectors (for "Negative: necklace, microphone")

- **Problem it solves:** text → boxes for things SAM 3's concept head may not separate as *parts* of a person.
- **CNP today:**
  - Grounding-DINO: `experimental_backend.py`, text → bbox mask.
  - LocateAnything-3B: `locate_anything_backend.py`, licence `other`, unknown.
  - Florence-2: only mentioned in `mask_failure_explainer.py`.
- **Options:**
  - Grounding DINO (arXiv 2303.05499, Apache-2.0 code and weights): proven, good for accessory nouns.
  - Florence-2 (arXiv 2311.06242, MIT weights): it also does phrase grounding and referring-expression segmentation.
  - Core RT-DETR is closed-set, so it cannot handle "microphone"-style free text reliably.
- **PUBLISHED numbers:** not extracted in this pass. Do not quote any until the tables are read.
- **Worth it:** YES, as the **negative-text** front end. Text → detector boxes → SAM box-prompt masks → soft negative
  matte N (section 6.1).

### 3.6 BiRefNet / RMBG-2.0

- **Problem it solves:** high-resolution salient or background-removal masks with soft edges.
- **CNP today:** `salient_backend.py` / `salient_matter_backend.py` provide both BiRefNet and RMBG-2.0. Core ships a
  BiRefNet loader.
- **Licence:** BiRefNet MIT (code and all listed weights). **RMBG-2.0 is CC BY-NC 4.0** and uses the same BiRefNet
  architecture.
- **PUBLISHED numbers:** not extracted in this pass.
- **Worth it:**
  - YES: make **BiRefNet (or `BiRefNet-matting`, MIT) the default**, label RMBG-2.0 "non-commercial", and route
    BiRefNet through core's `LoadBackgroundRemovalModel` when available (so there is one model in memory).

---

## 4. Alpha matting

| Method (licence) | Input | PUBLISHED quality | Speed / VRAM (PUBLISHED) | CNP status | Verdict |
|---|---|---|---|---|---|
| **ViTMatte** (MIT / Apache-2.0 weights; arXiv 2305.15272) | Image + **trimap** | Composition-1k (Table 1) SAD/MSE/Grad/Conn: ViTMatte-S 21.46/3.3/7.24/16.21; **ViTMatte-B 20.33/3.0/6.74/14.78**; MatteFormer 23.80/4.03/8.68/18.90. Distinctions-646 (Table 2): ViTMatte-B 17.05/1.5/7.03/12.95 | ViTMatte-S 25.8M params; VRAM not published | Tiled backend + ONYX | Keep it as the commercial-clean default. The improvement is the **trimap** (below), not the model. |
| **MatAnyone** (S-Lab NC; arXiv 2501.14677) | Video + **first-frame mask** | VideoMatte 1920×1080 (Table 1) MAD/MSE/Grad/dtSSD: 4.24/0.33/4.00/1.19. RVM: 6.57/1.93/10.55/1.90 | Not stated | Backend present | Opt-in, NC label. |
| **MatAnyone 2** (S-Lab NC; arXiv 2512.11782, CVPR 2026) | Video + first-frame mask | VideoMatte 1920×1080: **4.10/0.28/3.45/1.15**. YouTubeMatte 1920×1080: 1.61/0.50/7.14/1.53. Real-world CRGNN (Table 2) MAD/MSE/Grad/dtSSD: **4.24/2.00/11.74/4.54** vs MatAnyone 5.76/3.04/15.55/5.44 and RVM 5.98/2.79/13.68/5.36 | Not stated (training only: 8×A800) | Used in the SeC + MatAnyone 2 pipeline | Best published video matte. **NC only.** Keep it opt-in. |
| **RVM** (GPL-3.0; arXiv 2108.11515) | Video, **no guidance** | VideoMatte 1920×1080 (from the MatAnyone 2 Table 1): 6.57/1.93/10.55/1.90 | 3.749M params; 104.2 FPS at 1080p, 76.5 FPS at 4K on a GTX 1080Ti (Table 3/4) | torch.hub at run time | Keep as the "fast preview" matter. Paper limitation: "prefers videos with clear target subjects"; people in the background confuse it. |
| **MaGGIe** (CC BY-NC 4.0; arXiv 2404.16035) | Video + **mask on every frame** | VideoMatte 1920×1080: 4.42/0.40/4.03/1.31. CRGNN real: 9.50/6.11/16.51/6.02 | Not stated | Absent | NO (NC, and it needs per-frame masks). |
| **GVM** (no licence; arXiv 2508.07905) | Video, diffusion | VideoMatte 1920×1080: 6.33/2.08/8.04/1.59 | Diffusion (slow) | Absent | NO (no licence). |
| **VideoMaMa** (CC BY-NC 4.0; arXiv 2601.14255, CVPR 2026) | Video + coarse masks, video-diffusion prior | Not extracted | Diffusion | **Mis-specified** in CNP as Mamba VOS | Fix the description; NC; LATER. |
| **ZIM** (CC BY-NC 4.0; arXiv 2411.00626) | Image, SAM-style prompts → matte | Not extracted | — | Absent | NO (NC). |
| **Matte-Anything** (MIT; arXiv 2306.04121) | Image: SAM + GroundingDINO + ViTMatte with an *automatic trimap* (including transparency detection) | Not extracted | — | CNP already has every component | Port the **idea** (trimap from SAM plus open-vocabulary "transparent object" detection). Code is MIT, so a port is OK with attribution. |
| BackgroundMattingV2 (MIT) | Image + clean plate | — | — | Present | Keep for clean-plate shots. |

### 4.1 Hair, motion blur, transparency and temporal stability (published claims only)

- **MatAnyone 2:**
  - Claims fine hair is preserved and that MatAnyone has "segmentation-like edges" (qualitative, Fig. 6).
  - **No quantitative motion-blur evaluation** (paper).
  - Temporal stability is measured by dtSSD (table above).
- **MatAnyone:** "region-adaptive memory fusion". A boundary-change predictor blends the current memory readout with
  the previous frame's values, so the core stays stable and only the edges update. This is the mechanism to copy *as
  an idea* in a commercial-clean way: temporal smoothing of alpha **only inside the unknown band**, using
  flow-warped previous alpha. It is a clean-room algorithm, no code is taken.
- **ViTMatte:** single-image, so it has **no temporal model**. Tiled per-frame ViTMatte flickers in the unknown band.
  This is the main weakness of CNP's ONYX pipeline (SAM 3.1 + tiled ViTMatte), and it is unmeasured here.
- **Transparency:**
  - None of these models is trained for glass or smoke specifically.
  - Matte-Anything addresses it by detecting transparent objects with an open-vocabulary detector and widening the
    trimap's unknown region there.
  - Production advice: transparent and refractive objects need a hand-made or keyed alpha. Never present SAM output
    as alpha (brief Phase 5: "Never treat a binary segmentation mask as equivalent to a production-ready alpha matte").

### 4.2 Trimap generation (the cheapest quality lever for ViTMatte)

CNP's `utils.mask_to_trimap` uses a constant erode/dilate of 8 px. Published practice and the options, in order of
cost:

1. **Scale-aware band:** band width proportional to the object's size (or to the distance transform). A fixed 8 px is
   too thin for hair at 4K and too thick for hard edges at 512 px.
2. **Uncertainty-driven band:** widen the unknown band where the SAM logits are near 0 (|logit| < τ) or where SAM 3's
   three tracker masks disagree. SAM 3 "predict[s] three output masks for every tracked object", which is a free
   disagreement signal. CNP already has `_uncertainty.py`; wire it into the trimap.
3. **Semantic widening (Matte-Anything idea):** a text detector for "hair", "fur", "glass", "veil" widens the unknown
   region there.
4. **Temporal trimap:** warp the previous frame's unknown band with flow and union it, which reduces boundary popping.

Success criterion (karpathy): on a matting test set with ground truth, lower SAD/Grad than the fixed 8 px band, at
equal or lower runtime. **Not measured yet.**

### 4.3 Foreground colour estimation / edge decontamination (missing in CNP)

- **Problem it solves:** with only a correct alpha, composited edges still carry the old background colour (a halo).
  Despill only fixes a *known backing colour*.
- **Options:**
  - **Multi-level foreground estimation** (Germer et al., arXiv 2006.14970; implemented in `pymatting`, MIT).
  - **Blur-fusion approximate foreground** (Forte & Pitié, "Approximate Fast Foreground Colour Estimation", ICIP 2021;
    no arXiv ID found, so implement it clean-room from the paper).
  - Both are model-free, run on CPU or GPU in torch, and need no weights.
- **Integration:** a new stage after alpha, before premultiply, in `mask_matting/node.py`. The output is
  `foreground_rgb`, then premult = FG × α.
- **Worth it:** **YES, top priority.** It is the single biggest visible VFX gain per line of code, with no licence
  risk.

---

## 5. Video object tracking

| Method (licence) | PUBLISHED J&F | Memory / speed (PUBLISHED) | CNP status | Verdict |
|---|---|---|---|---|
| **SAM 3 / 3.1 tracker** (SAM License) | Table 5 above. SAM 3.1 MOSEv2 62.3 | 3.1 is "~7x" faster at 128 objects on an H100 | Core path (`CoreVideoTracker`), ONYX | Primary tracker. |
| **Cutie** (MIT; arXiv 2310.12982) | Table 1, Cutie-base with MOSE: MOSE val 68.3, DAVIS-17 val 88.8, test-dev 85.3, YTVOS19 86.5. LVOS (Table S3): val 63.5, test 63.6 | Table 2, max GPU memory: Cutie-small 1.35G (FIFO) / 2.28G (LT), Cutie-base 2.36G (LT), XMem 3.03–3.34G. Small model 45.5 FPS on a V100 (ablation) | Backend, "missing-deps" unless installed | **Second tracker**: low VRAM, MIT, mask-seeded. Good for re-seeding after drift and for long shots on small GPUs. |
| **XMem** (MIT; arXiv 2207.07115) | Cutie Table 1, XMem with MOSE: MOSE 59.6, DAVIS-17 val 86.0 | 3.03–3.34G | Backend | Superseded by Cutie. Keep only for compatibility. |
| **DEVA** (CC BY-NC-SA 4.0; arXiv 2309.03903) | Cutie Table 1, with MOSE: MOSE 66.0, DAVIS-17 87.0 | — | Absent | NO (NC-SA). Its *idea* (decoupled detection + propagation with periodic re-detection) is already what core `SAM3_VideoTrack.detect_interval` does. |
| **SeC** (Apache-2.0; SeC-4B Apache) | Not extracted | 4B multimodal LLM (heavy) | Vendored engine | Keep for concept-heavy, appearance-change shots. |

### 5.1 Drift, occlusion and re-identification (design, from published mechanisms)

- **Bidirectional propagation:** run the tracker forward from keyframe k and backward from keyframe k+1, then blend by
  temporal distance. Disagreement means low confidence, which flags frames for correction. The cost is 2x inference.
- **Keyframe correction:** a user fix on frame f re-seeds the memory at f (SAM 2/3 `add_mask` on a frame, Cutie
  `step(mask=...)`). CNP's `CoreVideoTracker` exposes `add_mask`, so the work left is the UI and the re-run range.
- **Scene-cut detection:** reset the memory at cuts. Use histogram or SSIM jumps between frames; no model needed.
  The SAM 3 presence score dropping to near 0 is a second signal.
- **Re-identification:** SAM 3's detector re-detects the concept every `detect_interval` frames. Matching detections
  to tracks by IoU and appearance is what core does. CNP should **not** re-implement it.
- **Boundary flicker:**
  - Temporal smoothing of alpha **in the unknown band only**, with flow-warped previous alpha (section 4.1).
  - Core's `VideoDeflicker` acts on images, not mattes, so check its contract first.
  - Measure with dtSSD / MESSDdt (section 8).

---

## 6. Phase-5 design notes (positive/negative mattes; processing order)

### 6.1 Positive / negative system (proposed, not implemented)

- **P** (soft, 0..1) is built from positive inputs: text → SAM 3, points, boxes, drawn masks, positive exemplar
  boxes. Union = `1 − Π(1 − p_i)` (probabilistic OR, which keeps softness), not `max` and not a hard union.
- **N** (soft) is built from negative inputs:
  - negative text → Grounding DINO / Florence-2 boxes → SAM box masks (section 3.5);
  - negative points → SAM masks of the clicked object;
  - negative boxes; painted exclusion; polygon and spline exclusion.
  - Optional choke or spread of N, then feather.
- **Output:** `A = clamp(P × (1 − N), 0, 1)`, as specified in the brief. Keep P and N as **separate outputs**
  (non-destructive), so the user can re-grade either.
- **Ordering rule (critical):** the matting stage must not restore excluded regions.
  - (a) Build the trimap from P only, then **force the trimap to background where N > τ_hard**.
  - (b) Run the matter.
  - (c) Multiply by (1 − N) **after** matting as well, because the matter can bleed.
  - (d) Then do foreground estimation (section 4.3) on the final alpha.
- **Where SAM 3 is native:** negative *exemplar* boxes remove same-category false positives (section 3.1). Use them
  for "not that other person". For "person but not microphone", use the separate-pass subtraction above, as the brief
  already anticipates.

### 6.2 Technically correct order for matte operations (soft alpha preserved)

1. Garbage and holdout mattes (hard limits).
2. Core matte (eroded, solid).
3. Edge matte from the matter (soft).
4. Combine: `max(core, edge) × (1 − holdout) × garbage`.
5. Choke / spread on the soft alpha. Use grey-scale morphology or a distance-based shift, not thresholding.
6. Feather / edge blur (in that order).
7. Density / grade (gamma, levels).
8. Foreground estimation.
9. Premultiply.
10. Motion blur last, on premultiplied RGBA.
11. Unpremultiply only for colour ops between those steps.

This avoids binarising and avoids grading dark halos into the edge.

---

## 7. Video inpainting / restoration

| Method | Licence (VERIFIED) | PUBLISHED quality | Runtime / VRAM (PUBLISHED) | CNP / core status | Verdict |
|---|---|---|---|---|---|
| **ProPainter** (arXiv 2309.03897) | **S-Lab NC** | Table 1: YouTube-VOS PSNR/SSIM/VFID/E_warp 34.43/0.9735/0.042/0.974; DAVIS 34.47/0.9776/0.098/1.187 (better than E2FGVI and FGT) | 0.083 s/frame (Table 1); 480p 0.249 s/frame (App. Table 4); no VRAM table ("other methods ... exceeding 32G") | Bridged (not vendored), CNP's main inpainter | Keep it **opt-in with an NC label**. It must not be a default in an Apache pack marketed for production. |
| **DiffuEraser** (arXiv 2501.10018) | Apache-2.0, but it **requires ProPainter weights** as its prior, plus SD-1.5 | Not extracted | README (L20 GPU, 250 frames): 1280×720 33G / 314 s; 960×540 20G / 175 s; 640×360 12G / 92 s | Absent | LATER, and only if the prior is swapped for a permissive one. Otherwise it is NC by inheritance. |
| **MiniMax-Remover** (arXiv 2505.24873) | Repo: no LICENSE. Weights **CC BY-NC 4.0** | Paper says "as few as 6 sampling steps", no CFG; tables not retrieved (arXiv HTML returned 404) | Example 480×832, 81 frames; Wan-family VAE (`AutoencoderKLWan`); VRAM not stated | Absent | NO for shipping (NC). |
| **FloED** (arXiv 2412.00857) | Code release not confirmed | Claims to beat prior diffusion methods (not quantified here) | — | Absent | NO until code and licence exist. |
| **VACE** (arXiv 2503.07598) | **Apache-2.0** code; Wan2.1-VACE-1.3B / 14B **Apache-2.0** | Not extracted | 1.3B at ~81×480×832; 14B at ~81×720×1280 (the 8-GPU example); VRAM not stated | WNE / core Wan stack | **Best permissive generative inpainter.** MV2V masked inpainting. CNP should feed it masks (a mask → VACE control-video helper), not re-implement it. |
| **VOID** (arXiv 2604.02296) | Apache-2.0 code and weights; CogVideoX-Fun base `other` | No benchmark on the README | "40GB+ VRAM" | **Already in core** (VOID nodes) | Do not duplicate. CNP's value is producing the **quadmask** from its tracker (0 = remove, 63 = overlap, 127 = affected, 255 = keep). |

**Permissive non-generative alternative:**
- Flow-guided propagation (the ProPainter *idea*: flow completion → pixel propagation) built clean-room on core
  RAFT / SEA-RAFT (BSD), with VACE or an image inpainter for never-seen pixels.
- Integration complexity is high, but it removes the NC dependency from the default path.

---

## 8. Depth, normals and optical flow

| Method | Licence (VERIFIED) | PUBLISHED | Runtime / VRAM (PUBLISHED) | CNP / core status | Verdict |
|---|---|---|---|---|---|
| **Depth Anything 3** (arXiv 2511.10647) | Code Apache-2.0. **SMALL (0.08B), BASE (0.12B), METRIC-LARGE (0.35B), MONO-LARGE (0.35B) Apache-2.0**. LARGE, GIANT, NESTED CC BY-NC | Ray-pose AUC3 table (README) | DA3-Streaming "less than 12GB" for ultra-long video | **Core ships DA3 nodes**. CNP `_control_backends.run_da3` maps small and base to METRIC-LARGE and large to NESTED (NC) | Fix the mapping to the real Apache checkpoints. Default to an Apache model. Prefer core's loader. |
| **Video Depth Anything** (arXiv 2501.12375) | Small Apache-2.0; Base / Large / Metric CC BY-NC | Metric (README): KITTI δ1 0.877/0.887/0.910 (S/B/L); ScanNet TAE 1.48/1.26/1.09 | A100, 32×518×518: Small FP16 7.5 ms, 6.8 GB; Large FP16 14 ms, 23.6 GB | Absent | **YES: VDA-Small** as the commercial-clean, temporally stable video depth (replaces DepthCrafter as default). |
| **DepthCrafter** (arXiv 2409.02095) | **Academic / research / education only (code and weights)** | Table 1, AbsRel/δ1: Sintel 0.270/0.697, ScanNet 0.123/0.856, KITTI 0.104/0.896, Bonn 0.071/0.972 (beats DA-V2) | 465.84 ms/frame at 1024×576 (A100); ~24 GB for 110-frame windows (~12 GB at 40) | **Code committed** in `nodes/depthcrafter/` | **NEEDS_OWNER**: remove from the Apache tree or move it to an NC-labelled optional download. |
| **Depth Pro** (arXiv 2410.02073) | Code: Apple sample-code licence. **Weights research-only** (apple-amlr) | VDA README table: KITTI δ1 0.822, NYUv2 0.953 | — | Runner present (HF `apple/DepthPro-hf`) | Label it research-only. Not a default. |
| **NormalCrafter** (arXiv 2504.11427) | MIT code; weights Apache-2.0 tag; SVD base | Not extracted | SVD-class (heavy) | Vendored `nodes/normalcrafter/` (MIT, OK) | Keep. NEEDS_OWNER on SVD-base inheritance. Cheap fallback: `normals_from_depth`. |
| **RAFT** (arXiv 2003.12039) | BSD-3 | MemFlow Table 1 (C+T): Sintel clean/final 1.43/2.71, KITTI Fl-all 17.4 | — | **Core `OpticalFlowLoader`** (torchvision raft_large). CNP uses ProPainter's `raft-things.pth` | Switch CNP to core's RAFT (no ProPainter checkout needed). |
| **SEA-RAFT** (arXiv 2405.14793) | **BSD-3 code and weights** | Spring EPE 3.69, 1px 0.36 (README); "at least 2.3x faster than existing methods" | — | Absent | **YES** as the fast, accurate flow for trimap warping, MESSDdt and propagation. |
| **MemFlow** (arXiv 2404.04808) | Apache-2.0 | Table 1 (C+T): Sintel 0.93/2.08, KITTI 13.7; 5.6 FPS at 1024×436 on an A100 | — | Absent | LATER (multi-frame; SEA-RAFT is simpler). |

---

## 9. Evaluation metrics to adopt (definitions; no results here)

| Metric | Use | Definition / source |
|---|---|---|
| IoU (J), Dice | Binary masks | IoU = \|A∩B\|/\|A∪B\|; Dice = 2\|A∩B\|/(\|A\|+\|B\|) |
| Boundary F (F) and J&F | VOS / roto | DAVIS boundary F-measure (precision/recall of contour pixels within a tolerance); J&F = mean of the two (DAVIS 2017, arXiv 1704.00675) |
| Boundary IoU / mBIoU | Thin structures | As used by HQ-SAM (Table 2) and SAMRefiner (Table 1) |
| SAD, MSE, Grad, Conn | Alpha | Composition-1k convention, computed on the trimap unknown region (ViTMatte Table 1) or on the whole image (D646). MAD is used by the video papers |
| dtSSD | Temporal alpha | "Mean squared difference of direct temporal gradients" (TCVOM, arXiv 2105.11427, after Erofeev et al., BMVC 2015) |
| MESSDdt | Temporal alpha, motion-compensated | "Mean squared difference between the warped temporal gradient" using optical flow (same sources). The TCVOM authors note "severe motion noise at translucent pixels" |
| Flicker (no ground truth) | Production clips | Mean \|α_t − warp(α_{t−1})\| inside the unknown band, plus the share of pixels flipping across 0.5. Report it per shot |

**Recommendation:** a CPU-runnable `tools/` metric module plus a tiny fixed test set (a few public-licence clips and
images with ground truth). Every matting or trimap change must then report before/after numbers. Without this, the
Phase-5 items cannot reach DONE under standing order 3.

---

## 10. Ranked shortlist: 10 most worthwhile, licence-clean improvements for CNP

Ranked by (quality gain per effort) × licence safety. All items stay within existing nodes (owner rule: no new nodes
without asking). Every quality expectation is from published mechanisms. **Nothing was measured locally.**

1. **Foreground-colour estimation stage** (multi-level, arXiv 2006.14970 / pymatting MIT, and/or blur-fusion
   clean-room).
   - Where: Mask + Matting, after alpha and before premultiply.
   - Gain: removes background halos on every matter.
   - Cost: no weights, low complexity.
   - Verify: composite onto contrasting backgrounds; edge colour error against ground-truth FG.
2. **Soft positive/negative matte system** (section 6.1): P, N, `A = P(1−N)`, N forced into the trimap, a
   post-matting re-multiply, separate P and N outputs.
   - Gain: fixes the current hard drop/subtract of `_core_sam3.segment_union`.
   - Cost: model-free core, low to medium complexity.
   - Verify: unit tests that excluded pixels stay 0 after any matter.
3. **Negative-text path** via Grounding DINO (Apache) or Florence-2 (MIT) → SAM box masks → N. CNP already has the
   Grounding-DINO backend. Verify on a "person minus microphone" set (IoU of the excluded part).
4. **Evaluation harness** (section 9): IoU / boundary-F / SAD / Grad / Conn / dtSSD / MESSDdt / flicker, CPU,
   permissive test clips. This makes every other item provable (standing order 3).
5. **Adaptive trimap** (section 4.2): size-scaled band, SAM-logit and 3-mask uncertainty widening (reuse
   `_uncertainty.py`), optional flow-warped temporal union. Gain: ViTMatte (MIT / Apache) quality is trimap-bound.
   Verify SAD/Grad against the fixed 8/8 band.
6. **Temporal alpha stabilisation in the unknown band only** for ONYX (SAM 3.1 + tiled ViTMatte). Use flow-warped
   previous alpha with confidence blending: the MatAnyone *idea*, clean-room, no code. Flow from core RAFT or
   SEA-RAFT (BSD). Gain: lower dtSSD without NC MatAnyone. Verify with dtSSD / MESSDdt.
7. **Licence-safe defaults and labels:**
   - (a) DA3 mapping → real `DA3-SMALL` / `DA3-BASE` / `DA3METRIC-LARGE` / `DA3MONO-LARGE` (Apache), with the NC ones
     opt-in;
   - (b) BiRefNet default, RMBG-2.0 labelled NC;
   - (c) MatAnyone / MatAnyone 2 / ProPainter / Depth Pro / VDA-B/L labelled NC or research-only in combos and docs;
   - (d) **NEEDS_OWNER:** the vendored DepthCrafter code in an Apache repo;
   - (e) fix the VideoMaMa backend description and URL (`cvlab-kaist/VideoMaMa`, CC BY-NC 4.0, a *matting* model).
8. **Flow on core RAFT, plus SEA-RAFT as an option** (BSD code and weights). Replace `optical_flow.py`'s dependency on
   a ProPainter checkout with core `comfy.ldm.raft` (already installed). Add SEA-RAFT-M for speed and accuracy. Gain:
   OpticalFlowMEC works on a clean install, with no NC dependency.
9. **Video-Depth-Anything-Small** (Apache-2.0) as the default temporally consistent video depth in Control AOV.
   PUBLISHED: 7.5 ms and 6.8 GB FP16 on an A100 for 32×518². It replaces DepthCrafter (NC) as the default.
10. **SAM 3 negative exemplar boxes on the core path, plus the SAMRefiner IoU-guard.**
    - Pass exemplar labels to core's geometry encoder: preferably an upstream core PR, otherwise a guarded CNP call
      with fallback.
    - Add a keep-if-IoU-with-coarse ≥ τ rule to `segment_refine.py`.
    - Port CEBox from the now-verified MIT SAMRefiner code (attribute it, `# PORTED FROM:`) as a later sub-step.

**Explicitly not recommended now:**
- MatAnyone 2, MiniMax-Remover, DEVA, MaGGIe, ZIM, VideoMaMa and GVM as defaults: NC, NC-SA or unlicensed.
- A second VOID or SAM 3 wrapper: core already ships them.
- DiffuEraser: it is NC through its ProPainter prior.
- **Cutie** is licence-clean (MIT) and worth keeping as the low-VRAM second tracker (Table 2: 1.35–2.36 GB). It sits
  just outside the top 10 because it is already present as a backend; the remaining work is installation and UX.

---

## 11. Open questions / not verified

- **Not retrieved here:** Grounding DINO, Florence-2 and BiRefNet benchmark tables; the MiniMax-Remover tables (the
  arXiv HTML returned 404); NormalCrafter numbers; SAM 3 Table 5 metric symbol (assumed J&F). Do not quote these until
  read.
- **Weight licences not separately stated:** Cutie and XMem GitHub-release weights; MatAnyone / MatAnyone 2 HF cards
  (no tag, so the code licence is assumed); LocateAnything-3B (`other`).
- **Inherited base-model licences:** SVD (NormalCrafter, DepthCrafter) and CogVideoX-Fun (VOID) carry their own
  licences. NEEDS_OWNER.
- **Not measured locally:** no runtime or VRAM figure was measured on the owner's hardware. All the numbers above are
  vendor or paper numbers on A100, H100, H200, V100, L20 or 1080Ti GPUs.

## 12. Sources (all opened or queried on 2026-11-27)

**Repos (GitHub API licence and LICENSE text)**
- https://github.com/facebookresearch/sam3 (LICENSE, RELEASE_SAM3p1.md)
- https://github.com/facebookresearch/sam2
- https://github.com/SysCV/sam-hq
- https://github.com/linyq2117/SAMRefiner
- https://github.com/IDEA-Research/GroundingDINO
- https://github.com/IDEA-Research/Grounded-SAM-2
- https://github.com/ZhengPeng7/BiRefNet
- https://github.com/hustvl/ViTMatte
- https://github.com/pq-yang/MatAnyone
- https://github.com/pq-yang/MatAnyone2
- https://github.com/PeterL1n/RobustVideoMatting
- https://github.com/PeterL1n/BackgroundMattingV2
- https://github.com/hustvl/Matte-Anything
- https://github.com/hmchuong/MaGGIe
- https://github.com/naver-ai/ZIM
- https://github.com/cvlab-kaist/VideoMaMa
- https://github.com/aim-uofa/GVM
- https://github.com/hkchengrex/Cutie
- https://github.com/hkchengrex/XMem
- https://github.com/hkchengrex/Tracking-Anything-with-DEVA
- https://github.com/OpenIXCLab/SeC
- https://github.com/sczhou/ProPainter
- https://github.com/lixiaowen-xw/DiffuEraser
- https://github.com/zibojia/MiniMax-Remover
- https://github.com/ali-vilab/VACE
- https://github.com/netflix/void-model
- https://github.com/ByteDance-Seed/Depth-Anything-3
- https://github.com/DepthAnything/Video-Depth-Anything
- https://github.com/apple/ml-depth-pro
- https://github.com/Tencent/DepthCrafter
- https://github.com/Binyr/NormalCrafter
- https://github.com/princeton-vl/RAFT
- https://github.com/princeton-vl/SEA-RAFT
- https://github.com/DQiaole/MemFlow
- https://github.com/pymatting/pymatting
- https://github.com/MarcoForte/FBA_Matting

**Hugging Face model API / cards**
- briaai/RMBG-2.0 (card: CC BY-NC 4.0)
- facebook/sam3, facebook/sam3.1, Comfy-Org/sam3.1
- facebook/sam2.1-hiera-large
- syscv-community/sam-hq-vit-huge
- hustvl/vitmatte-*
- ZhengPeng7/BiRefNet*
- microsoft/Florence-2-*
- IDEA-Research/grounding-dino-*
- PeiqingYang/MatAnyone2
- depth-anything/DA3-*, depth-anything/Video-Depth-Anything-*, depth-anything/Depth-Anything-V2-*-hf
- apple/DepthPro (LICENSE) and apple/DepthPro-hf
- tencent/DepthCrafter
- Yanrui95/NormalCrafter
- zibojia/minimax-remover
- lixiaowen/diffuEraser
- Wan-AI/Wan2.1-VACE-1.3B, Wan-AI/Wan2.1-VACE-14B
- netflix/void-model
- alibaba-pai/CogVideoX-Fun-V1.5-5b-InP
- MemorySlices/Tartan-C-T-TSKH-spring540x960-M
- OpenIXCLab/SeC-4B
- nvidia/LocateAnything-3B
- stabilityai/stable-video-diffusion-img2vid-xt

**Papers (arXiv IDs verified via the arXiv API; tables read from arXiv HTML where cited)**
- 2511.16719 SAM 3 · 2408.00714 SAM 2 · 2306.01567 HQ-SAM · 2502.06756 SAMRefiner
- 2303.05499 Grounding DINO · 2311.06242 Florence-2 · 2401.03407 BiRefNet
- 2305.15272 ViTMatte · 2501.14677 MatAnyone · 2512.11782 MatAnyone 2 · 2108.11515 RVM
- 2306.04121 Matte Anything · 2404.16035 MaGGIe · 2411.00626 ZIM · 2601.14255 VideoMaMa · 2610.26857 VideoMaMa++
- 2508.07905 GVM
- 2310.12982 Cutie · 2207.07115 XMem · 2309.03903 DEVA
- 2309.03897 ProPainter · 2501.10018 DiffuEraser · 2505.24873 MiniMax-Remover · 2412.00857 FloED · 2503.07598 VACE
- 2604.02296 VOID
- 2511.10647 Depth Anything 3 · 2501.12375 Video Depth Anything · 2410.02073 Depth Pro · 2409.02095 DepthCrafter
- 2504.11427 NormalCrafter
- 2003.12039 RAFT · 2405.14793 SEA-RAFT · 2404.04808 MemFlow
- 2105.11427 TCVOM (dtSSD / MESSDdt) · 1704.00675 DAVIS 2017 · 2006.14970 Fast Multi-Level Foreground Estimation
- Not on arXiv: Erofeev et al., BMVC 2015 (dtSSD / MESSDdt origin); Forte & Pitié, ICIP 2021 (blur-fusion)

**Local (read-only)**
- CNP `nodes/` files named in section 1.1
- Core `comfy_extras/nodes_sam3.py`, `nodes_bg_removal.py`, `nodes_depth_anything_3.py`, `nodes_void.py`,
  `nodes_video_deflicker.py`, `nodes_rtdetr.py`
- Core `comfy/ldm/sam3/detector.py`, `comfy/ldm/raft/model.py`, `comfy/bg_removal_model.py`
