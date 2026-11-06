# SeC inference engine (ported, L7.90)

Source: <https://github.com/9nate-drake/Comfyui-SecNodes> at commit `af39e6260bc414873fbb2dc8df355904d1609754`:
`inference/` -> this folder, `model_config/` -> `model_config/`. Licences: Apache-2.0 (Comfyui-SecNodes, SeC by
OpenIXCLab, InternLM, SAM 2 by Meta) and MIT (InternVL files). Each file keeps its original notice; a
`# PORTED FROM:` line was added on top. `LICENSE` is the upstream Apache-2.0 text.

Why: Mask Ops' "SeC + MatAnyone2" engine loaded this code from the pack's `third_party/` folder, which git ignores,
so it failed on every fresh install (and here too: the config path it used, `model_config/SeC-4B`, never existed).

Changes (all mechanical; the model code is unchanged):

1. `sam2/__init__.py`: hydra removed. The builder reads the YAML itself, so `init_sam2_hydra()` is now a no-op.
2. `sam2_video_predictor.py`: omegaconf replaced by plain YAML + dicts (`_yaml_cfg.py`, new). PyYAML reads `1e-6`
   as a string; the loader adds the YAML 1.2 float rule so values load as OmegaConf loaded them. The configs folder is
   `sam2/configs` (the upstream builder pointed at the duplicate top-level `configs/`; the files are identical).
3. `sam2/modeling/backbones/hieradet.py`: iopath's `g_pathmgr.open` replaced by `open` (local paths only).
4. `sam2/utils/transforms.py`: `from inference.sam2.utils.misc` made relative.
5. `modeling_sec.py`: `peft` is imported only when a LoRA wrap is asked for (SeC-4B: none).
6. `model_config/tokenizer.json` (7 MB) is not shipped: transformers rebuilds the same tokenizer from `vocab.json` +
   `merges.txt` (ids checked equal, tests/test_l790_sec_engine.py).
7. Not shipped: `sam2/csrc/connected_components.cu` (an optional CUDA extension; the Python fallback is used).
8. `@torch.inference_mode()` -> `@torch.no_grad()` (9 sites): inference-mode tensors cannot be edited in place by the
   next node, which this pack forbids (tests/test_no_inference_tensors.py). Gradients stay off either way.

hydra-core, omegaconf and iopath are therefore not needed; ComfyUI's Python here has none of them.
