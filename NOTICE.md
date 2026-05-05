# NOTICE — ComfyUI-CustomNodePacks

Copyright (c) 2025-2026 Code2Collapse  
License: MIT (see MIT-License)

This project integrates code and pretrained models from multiple third-party
open-source projects. Their copyrights and licenses are listed below.

---

> ⚠️  **NON-COMMERCIAL MODEL WARNING**
>
> Several AI models that can be loaded through this pack carry **non-commercial
> use restrictions** that apply to **the model weights only** (not this code):
>
> | Model | Owner | Restriction |
> |-------|-------|-------------|
> | RMBG-2.0 | Bria AI | [BRIA RMBG License](https://huggingface.co/briaai/RMBG-2.0) — non-commercial & research only |
> | SAM-HQ weights (lkeab/hq-sam) | IIAI | Apache-2.0 ✅ |
> | MatAnyone2 (pq-yang/MatAnyone2) | NTU/Peng-Qi Yang | [Check HuggingFace page](https://huggingface.co/pq-yang/MatAnyone2) for current license |
> | ProPainter (sczhou/ProPainter) | S-Lab NTU | [S-Lab License 1.0](https://github.com/sczhou/ProPainter/blob/main/LICENSE) — non-commercial research only |
>
> **If you use RMBG-2.0 model weights for commercial purposes you must obtain
> a separate commercial licence from Bria AI.**  
> All other models listed in this NOTICE are Apache-2.0 or MIT licensed.

---

## 1. SAM-HQ (Segment Anything in High Quality)

Vendored in `third_party/sam-hq/`. LICENSE file is preserved at
`third_party/sam-hq/LICENSE`.

**Repository**: <https://github.com/SysCV/sam-hq>  
**Copyright**: Copyright (c) 2023 ETH Zurich, IIAI  
**License**: Apache License 2.0  
<https://github.com/SysCV/sam-hq/blob/main/LICENSE>

The `sam-hq` directory also contains:

### 1a. SAM (Segment Anything Model) — Meta AI

**Repository**: <https://github.com/facebookresearch/segment-anything>  
**Copyright**: Copyright (c) Meta Platforms, Inc. and affiliates.  
**License**: Apache License 2.0  
<https://github.com/facebookresearch/segment-anything/blob/main/LICENSE>

### 1b. SAM2 / SAM2.1 — Meta AI

Used at runtime via `pip install git+https://github.com/facebookresearch/sam2.git`

**Repository**: <https://github.com/facebookresearch/sam2>  
**Copyright**: Copyright (c) Meta Platforms, Inc. and affiliates.  
**License**: Apache License 2.0  
<https://github.com/facebookresearch/sam2/blob/main/LICENSE>

### 1c. GroundingDINO — IDEA Research

Vendored in `third_party/sam-hq/seginw/GroundingDINO/`. LICENSE file is
preserved at `third_party/sam-hq/seginw/GroundingDINO/LICENSE`.

**Repository**: <https://github.com/IDEA-Research/GroundingDINO>  
**Copyright**: Copyright (c) 2023 IDEA Research  
**License**: Apache License 2.0  

---

## 2. ViTMatte

Vendored in `third_party/ViTMatte/`. LICENSE file is preserved at
`third_party/ViTMatte/LICENSE`.

**Repository**: <https://github.com/hustvl/ViTMatte>  
**Copyright**: Copyright (c) 2023 Hust Vision Lab  
**License**: MIT License  
<https://github.com/hustvl/ViTMatte/blob/main/LICENSE>

---

## 3. RMBG-2.0 — Bria AI

Model weights downloaded from `briaai/RMBG-2.0` on Hugging Face.  
**This model has a non-commercial license.** See the warning box at the top
of this file.

**HuggingFace**: <https://huggingface.co/briaai/RMBG-2.0>  
**License**: BRIA AI RMBG License Agreement (non-commercial / research)

---

## 4. BiRefNet — ZhengPeng7

Model weights downloaded from `ZhengPeng7/BiRefNet` and
`ZhengPeng7/BiRefNet-portrait` on Hugging Face.

**Repository**: <https://github.com/ZhengPeng7/BiRefNet>  
**Copyright**: Copyright (c) 2023 ZhengPeng7  
**License**: MIT License  
<https://github.com/ZhengPeng7/BiRefNet/blob/main/LICENSE>

---

## 5. SeC (Segment and Caption) — OpenIXCLab

Model weights downloaded from `OpenIXCLab/SeC-4B` on Hugging Face.

**Repository**: <https://github.com/OpenIXCLab/SeC>  
**Copyright**: Copyright (c) 2024 OpenIXCLab  
**License**: Apache License 2.0

---

## 6. MatAnyone2 — Peng-Qi Yang / NTU S-Lab

Model weights downloaded from `pq-yang/MatAnyone2` on Hugging Face.

**Repository**: <https://github.com/pq-yang/MatAnyone>  
**Copyright**: Copyright (c) 2024 Peng-Qi Yang, NTU S-Lab  
**License**: Check <https://huggingface.co/pq-yang/MatAnyone2> for current terms.  
The original MatAnyone paper describes an S-Lab research model; verify
commercial use terms before deployment.

---

## 7. HQ-SAM Weights (lkeab/hq-sam)

Model weights downloaded from `lkeab/hq-sam` on Hugging Face.

**Copyright**: Copyright (c) 2023 ETH Zurich, IIAI  
**License**: Apache License 2.0

---

## 8. SAM3 (apozz/sam3-safetensors)

Model weights downloaded from `apozz/sam3-safetensors` on Hugging Face.
SAM3 is the successor to SAM2 by Meta AI.

**License**: Apache License 2.0 (per Meta AI SAM2 upstream)

---

## 9. ViTMatte Models (hustvl/vitmatte)

Model weights downloaded from `hustvl/vitmatte-small-distinctions-646` and
`hustvl/vitmatte-base-distinctions-646` on Hugging Face.

**Copyright**: Copyright (c) 2023 Hust Vision Lab  
**License**: MIT License

---

## 10. PyTorch

**Repository**: <https://github.com/pytorch/pytorch>  
**Copyright**: Copyright (c) 2016-2024 Facebook, Inc. and its affiliates  
**License**: BSD-style license — <https://github.com/pytorch/pytorch/blob/main/LICENSE>

---

## 11. OpenCV

**Repository**: <https://github.com/opencv/opencv>  
**License**: Apache License 2.0 (since OpenCV 4.5)

---

## 12. SciPy

**Repository**: <https://github.com/scipy/scipy>  
**Copyright**: Copyright (c) 2001-2024 SciPy Developers  
**License**: BSD 3-Clause

---

## 13. safetensors (Hugging Face)

**Repository**: <https://github.com/huggingface/safetensors>  
**Copyright**: Copyright (c) 2022 The HuggingFace Team  
**License**: Apache License 2.0

---

## 14. ProPainter — sczhou / S-Lab NTU

Vendored in `third_party/ProPainter/`. The upstream repository is shallow-cloned
and **not modified**. Used by `nodes/propainter_temporal_inpaint.py` and
`nodes/propainter_stitch_suite.py` for flow-aware temporal video inpainting
and seam refinement.

**Repository**: <https://github.com/sczhou/ProPainter>  
**Paper**: ProPainter: Improving Propagation and Transformer for Video Inpainting (ICCV 2023)  
**Copyright**: Copyright (c) 2023 Shangchen Zhou, S-Lab, Nanyang Technological University  
**License**: S-Lab License 1.0 (non-commercial research-only — see
`third_party/ProPainter/LICENSE`)  
<https://github.com/sczhou/ProPainter/blob/main/LICENSE>

> ⚠️ **NON-COMMERCIAL**: ProPainter weights and code carry an academic /
> research-only license. The MEC node code (`nodes/propainter_*.py`) is MIT,
> but commercial use of ProPainter requires a separate licence from the
> S-Lab authors.

The ProPainter weights (`raft-things.pth`,
`recurrent_flow_completion.pth`, `ProPainter.pth`) are downloaded from the
upstream GitHub release and stored in `ComfyUI/models/propainter/`.

---

## 15. ComfyUI-Video-Stabilizer — nomadoor

Vendored in `third_party/ComfyUI-Video-Stabilizer/`. The upstream repository
is **not modified**. Used by `nodes/video_stabilizer_mec.py` for the three
`VideoStabilizer*MEC` wrapper nodes (Classic / Flow / Auto).

**Repository**: <https://github.com/nomadoor/ComfyUI-Video-Stabilizer>  
**Copyright**: Copyright (c) 2025 ComfyUI Video Stabilizer Contributors  
**License**: MIT License  
<https://github.com/nomadoor/ComfyUI-Video-Stabilizer/blob/main/LICENSE>

The full upstream LICENSE file is preserved at
`third_party/ComfyUI-Video-Stabilizer/LICENSE`. The wrapper nodes in
`nodes/video_stabilizer_mec.py` import upstream helper functions
(`_stabilize_frames`, `_normalize_video_input`, etc.) without modification.

---

## Code Attribution

All Python source files in `nodes/` are original work by Code2Collapse, written
as ComfyUI node wrappers that call into the above upstream models and
libraries. No source code was copied verbatim from any of the above projects.

---

## UX / Design Inspirations (no code copied)

The following projects inspired the **interaction design** of certain widgets
in this pack. **No source code, assets, or literal expression was copied**
from either project. The implementations in `nodes/video_frame_player.py` and
`js/video_frame_player.js` are original, clean-room code written from scratch
using common HTML5 canvas patterns (8-handle drag-resize, aspect-locked snap,
dim-overlay, rule-of-thirds guides). Listed here purely to credit the
upstream UX ideas that prompted the feature requests.

### A. Olm DragCrop — Olli Sorjonen (`@o-l-l-i`)

- Repository: <https://github.com/o-l-l-i/ComfyUI-Olm-DragCrop>
- Author: Olli Sorjonen
- License: **Source-available, NOT open-source.** Per the upstream README:
  > "Redistribution, resale, rebranding, or claiming authorship of this code
  > or extension is strictly prohibited without explicit written permission."
- **Compatibility note:** Olm DragCrop's licence prohibits redistribution
  without written permission, so its code **cannot** be vendored or copied
  into this MIT-licensed pack. The drag-rectangle UX in
  `VideoFramePlayerMEC` was implemented independently. Credit goes to Olli
  Sorjonen for popularising the in-node drag-crop UX pattern in the ComfyUI
  ecosystem.

### B. WhatDreamsCost-ComfyUI — Jonathan Watkins (`@WhatDreamsCost`)

- Repository: <https://github.com/WhatDreamsCost/WhatDreamsCost-ComfyUI>
- Author: Jonathan Watkins
- License: **GPL-3.0** (strong copyleft).
- **Compatibility note:** GPL-3.0 is **incompatible** with this project's
  MIT licence for the purposes of code incorporation — including any GPL
  source into this repo would force the entire repo to GPL-3.0. The
  "Load Video UI" node's *layout idea* (timeline + trim handles + resize
  method combo + custom width/height + frame-rate widgets) inspired the
  widget layout of `VideoFramePlayerMEC`, but the code is original. No
  GPL-3.0 source was copied, vendored, or executed during development.

If you wish to use Olm DragCrop or WhatDreamsCost-ComfyUI directly, please
install them as separate ComfyUI custom-node packs from their official
repositories above, where their respective licences apply.
