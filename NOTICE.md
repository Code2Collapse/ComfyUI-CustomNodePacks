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

## Code Attribution

All Python source files in `nodes/` are original work by Code2Collapse, written
as ComfyUI node wrappers that call into the above upstream models and
libraries. No source code was copied verbatim from any of the above projects.
