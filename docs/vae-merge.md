# VAE Tools (MEC)

Four VAE-focused nodes covering merging, latent diagnostics, and model
introspection.

## VAEMergeMEC

Merge two (or three) VAEs using one of eight algorithms:

- `weighted_sum` — `out = (1-α)·A + α·B`
- `add_difference` — `out = A + α·(B - C)` (requires `vae_c`)
- `tensor_sum` — element-wise mean of A and B
- `triple_sum` — equal mean of A, B, C (requires `vae_c`)
- `slerp` — spherical linear interpolation of flattened parameters
- `dare_ties` — DARE/TIES with sparsity drop and sign-resolution
- `block_swap` — replace whole blocks of A with B according to per-block
  weights
- `clamp_interp` — like `weighted_sum` but bounds the result to the per-tensor
  range of A and B

### Per-block alpha

Pass either a JSON object or a comma-separated list to override the
global `alpha` for individual blocks. Recognised names follow the
SD/SDXL VAE block layout (`block_conv_in`, `block_0..3`, `block_mid`,
`block_norm_out`, `block_conv_out`).

### Brightness / contrast

After the merge, two scalar tweaks can be applied to the
`decoder.conv_out` weights only — useful for nudging output luminance
without retraining.

The merged VAE is returned as a fresh `deepcopy`; the inputs are never
modified. The merge runs on CPU in float32 and is cast back to the
source dtype before being installed into the wrapper.

## Probe (C2C) — latent

Wire a **LATENT** into **Probe (C2C)** (`ImageStatsProbeMEC` under C2C/Helpers).
Read `info_json`, `verdict`, `nan_count`, and `inf_count` (outputs 10–13); the
latent passes through on output `latent`. Use `fail_on_corrupt=True` to stop
the run when NaN or Inf are present. **VAELatentInspectorMEC** was merged into
this node; saved workflows migrate on load (see `docs/MIGRATION.md`).

## VAE Inspect — compare mode

Wire two VAEs into **VAE Inspect** (`VAEBlockInspectorMEC`): `vae` (first) and
`vae_b` (second), set `mode` to **compare**, and optionally
`include_per_tensor` for verbose JSON. Read `report_json`, `global_cosine`, and
`most_divergent_blocks`. **VAESimilarityAnalyserMEC** was merged into this
node; saved workflows migrate on load (see `docs/MIGRATION.md`).

## VAE Inspect — inspect mode

Per-block weight statistics (mean / std / abs_mean / count) for a single
VAE on **VAE Inspect** with `mode` **inspect**. Useful to spot blocks dominated
by NaN/zeros, or to compare fine-tunes against a reference.
