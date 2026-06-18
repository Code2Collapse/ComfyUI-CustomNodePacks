# Prompt Relay — Attribution

This module is a refined, multi-backend port of the **Prompt Relay** algorithm and
its ComfyUI implementation.

## Original work

- **Algorithm / paper / reference implementation**
  Gordon Chen and contributors —
  <https://gordonchen19.github.io/Prompt-Relay/>

- **ComfyUI reference nodes (`gordonchen19/Prompt-Relay`)**
  Repository: <https://github.com/GordonChen19/Prompt-Relay>
  All design credit for the temporal-cost matrix, segment scheduler, smart-prompt
  parser, and timeline UI belongs to the upstream authors.

## What this refined port adds

1. **Native ComfyUI MODEL/CLIP path** — unchanged algorithm, repackaged under the
   legacy `INPUT_TYPES` schema used by the rest of `ComfyUI-CustomNodePacks` so
   it composes with every native sampler (KSampler, KSamplerAdvanced, custom
   sigma schedulers, SamplerCustomAdvanced, etc.) via `ModelPatcher.add_object_patch`.

2. **Kijai `WanVideoWrapper` backend** — Kijai's `WanVideoSampler` bypasses
   ComfyUI's `ModelPatcher.object_patches` dict and calls the transformer
   directly, so the standard patch path is silently ignored. This module adds
   an idempotent in-place rebind of `block.cross_attn.forward` on the live
   Kijai `WanModel`, storing the original method for clean restore.
   Tokenizes via Kijai's own `HuggingfaceTokenizer` so segment token-ranges line
   up with the embeddings the sampler actually sees.

3. **Generic-fallback patcher** — for any third-party model whose diffusion
   backbone is neither a native `comfy.ldm.wan.model.WanModel`, a native
   `comfy.ldm.lightricks.model` transformer, nor a Kijai `WanModel`: walks the
   block list, detects modules whose name contains `cross_attn` and whose
   `forward` signature matches the expected `(x, context, ...)` shape, and
   wires up the same Gaussian-penalty mask. Falls back gracefully (no patch,
   logged warning) if the structure is unrecognised.

4. **Safety**:
   * Detects collision with other cross-attn patchers (e.g. KJNodes NAG) and
     fails loud rather than silently overriding.
   * Kijai in-place rebind is reference-counted and restorable.
   * All three patch paths share one `mask_fn` builder, so the temporal
     semantics are identical across backends.

## Refinement details vs. upstream

- Renamed to avoid namespace collision with the upstream registry. Display
  names follow `ComfyUI-CustomNodePacks` convention (no `_C2C` suffix in UI).
- Schema switched to the legacy `INPUT_TYPES` API used throughout this pack.
- Algorithm code reorganised into `_core.py` (math + tokenization), `_parser.py`
  (smart-prompt syntax), `_patches.py` (three backends). No algorithmic change.
- Added `LICENSE`-pending note. The upstream repository ships without a LICENSE
  file as of May 2026 but its author has publicly stated the project is
  open-source and free for use; this notice preserves that authorial intent
  while explicitly crediting all contributors.

## License

Per the upstream author's stated intent (open / free for use), this port is
distributed under the same Apache-2.0 license that covers the rest of
`ComfyUI-CustomNodePacks`. If the upstream maintainers add a specific license
that conflicts, this module will be relicensed to match.
