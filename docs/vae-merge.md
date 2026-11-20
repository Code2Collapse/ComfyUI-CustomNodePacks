# VAE Merge and VAE Inspect

Two nodes for working on VAE weights: **VAE Merge** blends two or three VAEs into a new one, and **VAE Inspect**
looks inside one VAE or compares two. Neither node changes the VAEs you connect.

| Node | Use it to |
|---|---|
| [VAE Merge](#vae-merge-vaemergemec) | Blend 2 or 3 VAEs of the same architecture into a new VAE |
| [VAE Inspect](#vae-inspect-vaeblockinspectormec) | Check one VAE for broken or odd weights, compare two VAEs, or run a reconstruction test |

To read statistics of a **latent** (NaN / Inf, range), wire it into **Probe (C2C)** (`ImageStatsProbeMEC`), described
in [utility-nodes.md](utility-nodes.md). The old VAE Latent Inspector was merged into that node; saved workflows load
as Probe (C2C).

The block heatmap, the reconstruction benchmark, the per-weight overrides and the merge validation were added on
2026-11-19 (commit 4ebaa55). That commit tested them on the CPU with stand-in VAEs: behaviour with real VAE weights on
a GPU and the live UI are NOT VERIFIED.

## VAE Merge (`VAEMergeMEC`)

**Menu:** 🐺 C2C/🧰 Core/VAE · **Parameters:** [NODE_REFERENCE](../NODE_REFERENCE.md#vaemergemec)

### What it is for

It combines the weights of `vae_a` and `vae_b` (and `vae_c` for three-VAE modes) tensor by tensor, and returns a new
VAE built on a copy of `vae_a`. Use it to mix two fine-tunes of the same VAE, or to move part of one VAE's behaviour
into another.

### When to pick it

- Pick it when both VAEs have the same architecture (the same weight names and shapes). Different families
  (for example an SDXL VAE and a Flux VAE) cannot be merged; the node stops with an "Architecture mismatch" error.
- To check the result, follow it with [VAE Inspect](#vae-inspect-vaeblockinspectormec) in compare mode, or turn on
  this node's `validation`.

### Minimal wiring

```
Load VAE (A) ──► vae_a ┐
Load VAE (B) ──► vae_b ├─ VAE Merge ──► vae ──► VAE Decode
(Load VAE (C) ─► vae_c)┘            └─► info / recipe_json / weight_map / validation_json
```

### Merge modes (`merge_mode`)

There are 13. `alpha` is the main weight (0 = all A, 1 = all B in `weighted_sum`).

| Mode | What it does with each weight tensor |
|---|---|
| `weighted_sum` | (1 − α)·A + α·B |
| `add_difference` | A + α·(B − C). Needs `vae_c` |
| `tensor_sum` | A + B, rescaled so its largest value equals A's largest value. Does not use `alpha` |
| `triple_sum` | (A + B + C) / 3. Needs `vae_c`; does not use `alpha` |
| `slerp` | Spherical interpolation between A and B (both flattened) at α |
| `sigmoid` | `weighted_sum` with α passed through an S-curve (1 / (1 + e^(−12·(α − 0.5)))) |
| `geometric` | Weighted geometric mean of the magnitudes of A and B, with A's sign |
| `max_abs` | Per value, whichever of A and B is larger in magnitude. Does not use `alpha` |
| `smooth_add_diff` | A + α·tanh(B − C). Needs `vae_c` |
| `distribution_xover` | If B's values spread wider (standard deviation) than A's in that tensor, `weighted_sum` at α; otherwise A |
| `dare_ties` | Drops a random 10% of (B − A), rescales the rest, keeps only the values whose sign agrees with the overall sign, adds α of it to A |
| `block_swap` | The whole tensor from B when its weight is 0.5 or more, else from A |
| `clamp_interp` | `weighted_sum`, then clamped to the range (min / max) of A's tensor |

The merge is computed in float32 and stored back in A's dtype. Values that come out NaN or infinite are replaced by
A's values, and `info` lists a warning for each such tensor.

### Key controls

- **Per-block weights.** Turn on `use_blocks` and set the block sliders (`block_conv_in`, `block_conv_out`,
  `block_norm_out`, `block_0` to `block_3`, `block_mid`). They match SD-style VAE weight names (`encoder.down.N`,
  `decoder.up.N`, `mid`, `conv_in`, ...). `block_0` covers `encoder.down.0` and `decoder.up.3` (the decoder block at
  the same resolution), `block_1` covers `encoder.down.1` and `decoder.up.2`, and so on. Weights that match no slider
  use `alpha`.
- **`auto_alpha`.** Sets the block weights from how similar A and B are in each block (cosine similarity): the less
  similar the block, the closer its weight goes to A. It replaces the manual sliders.
- **`block_weights_json`** (new 2026-11-19). Fine-grained weights: a JSON object of `"<regex>": weight`, matched
  against each weight's name, for example `{"decoder\\.up\\.0\\.": 0.2, "^encoder\\.": 0.0}`. It is applied after the
  sliders and `auto_alpha`; the first matching pattern wins. It is saved in the recipe.
- **`brightness` / `contrast`.** Applied after the merge to `decoder.conv_out` only. `brightness` adds 0.1 × its value
  to that layer's bias; `contrast` scales its weight and bias by (1 + value).
- **`validation`** (new 2026-11-19). `report` runs a check after the merge and writes it to `validation_json` and the
  recipe. `block_on_regression` stops with an error instead of returning a merge whose mean PSNR is more than
  `validation_max_psnr_drop_db` (default 3 dB) below the weaker of A and B.
- **`dry_run`.** Skips the merge and returns `vae_a` unchanged, with the similarity report, the recipe and the
  weight chart. Use it to plan block weights quickly.
- **`recipe_in`.** Paste a `recipe_json` from an earlier run to repeat that merge; it overrides the widget values.
- **`device`.** `cpu` (default), `cuda` or `auto`. `cuda` falls back to the CPU when no GPU is available.

`beta` is hidden: the merge math never reads it. It is still saved and still written to the recipe.

### Reading the outputs

- `vae`: the merged VAE (a copy of `vae_a` with the new weights).
- `info`: JSON with `merge_mode`, `alpha`, `architecture` (detected as `sd1x`, `sdxl`, `flux`, `mochi` or `unknown`),
  `key_count`, `block_similarity` (cosine per block), `auto_alpha_blocks`, `dtypes`, `warnings`, `postprocess_keys`
  and `status` (`ok` or `dry_run`).
- `recipe_json`: the settings as JSON (`schema` `mec.vae_merge.recipe.v1`), plus `block_weight_overrides` and the
  `validation` summary when used. It has no timestamp, so two recipes diff cleanly.
- `probe_report`: only with `reference_image` connected. The first frame, scaled down to 256 × 256 if larger, goes
  through encode → decode in A, B and the merge: `mse` and `psnr_db` for each, and `delta_psnr_vs_a_db`.
- `weight_map`: a chart image of the effective weight per block group (including `override` for weights set by
  `block_weights_json`, and `other`) next to the A-to-B cosine of that block. It is also produced on `dry_run`.
- `validation_json`: `passed`, `reason`, `merged_mean_psnr_db`, `a_mean_psnr_db`, `b_mean_psnr_db`,
  `non_finite_keys`, `shape_mismatch_keys`, and the full test results (`suites`). The test is five synthetic
  64 px images (flat grey, gradient, 2 px checker, colour bars, noise) through each VAE. Empty when `validation` is
  `off`.

### Common mistakes

- **"'add_difference' requires vae_c to be connected"** (also `triple_sum`, `smooth_add_diff`): connect a third VAE or
  pick a two-VAE mode.
- **"Architecture mismatch between vae_a and vae_b"** or **"Shape mismatch for key ..."**: the two VAEs are not the
  same architecture. They cannot be merged.
- **`block_weights_json` refused**: it must be a JSON object; each weight a number from 0 to 1; each key a valid regex
  that matches at least one weight of this VAE (the error shows what the weight names look like).
- **"validation failed"** with `block_on_regression`: nothing was returned. Lower the merge weights, or set
  `validation` to `report` to look at the numbers first.
- **Two runs of `dare_ties` differ.** Its 10% drop is random with no fixed seed, so a recipe does not repeat a
  `dare_ties` merge exactly.
- **Block sliders do nothing.** They act only while `use_blocks` is on, and `auto_alpha` replaces them. On a VAE whose
  weight names are not SD-style, no slider matches: use `block_weights_json` instead.
- **"could not copy vae_a"**: the copy that holds the merge did not fit in memory. Free memory or merge on the CPU.

## VAE Inspect (`VAEBlockInspectorMEC`)

**Menu:** 🐺 C2C/🧰 Core/Model Analysis · **Parameters:** [NODE_REFERENCE](../NODE_REFERENCE.md#vaeblockinspectormec)

### What it is for

It reads a VAE's weights and reports on them. It never changes the VAE.
- `inspect` mode: weight statistics per block of one VAE, and the tensors that look wrong (NaN / Inf, or far larger
  than similar tensors).
- `compare` mode: how similar two VAEs are, block by block (cosine similarity).
- `benchmark` (either mode, new 2026-11-19): runs the VAE on test images and reports the reconstruction error.

The VAE Similarity Analyser node was merged into this node (mode `compare`); saved workflows load as VAE Inspect.

### When to pick it

- After [VAE Merge](#vae-merge-vaemergemec), to see which blocks moved.
- To check a downloaded VAE for broken weights before using it.
- To read latent statistics instead, use Probe (C2C) ([utility-nodes.md](utility-nodes.md)).

### Minimal wiring

```
Load VAE ──► vae ──► VAE Inspect ──► report_json / block_heatmap / benchmark_json
Load VAE ──► vae_b ─┘   (vae_b only in compare mode)
```

### Key controls

- `mode`: `inspect` (default) or `compare`. Compare needs `vae_b`.
- `anomaly_threshold` (inspect, default 5): a tensor is flagged when its size (mean absolute value, scaled for the
  layer's width) is more than this many times the median of its own kind (the same name ending and rank, such as all
  4-D `weight` tensors). Kinds with fewer than 4 tensors are not judged. Lower = more flags.
- `include_per_tensor` (compare): adds one cosine entry per tensor to the JSON. Long.
- `benchmark` and `benchmark_resolutions` (default `64,128`): sizes in pixels, multiples of 8 from 32 to 1024, at most
  4. The benchmark runs the VAE, so it takes seconds per case on a CPU.

### Reading the outputs

- `report_json`.
  - Inspect: `blocks` (mean / std / abs_mean / count per block), `total_tensors`, `outlier_tensor_names`,
    `outliers_detail` (reason `non_finite` or `high_abs_mean` with `cohort` and `ratio_to_cohort_median`),
    `anomaly_score`.
  - Compare: `global_cosine`, `common_tensors`, `only_in_a` / `only_in_b` (first 50 names), `per_block_cosine`,
    `most_divergent_blocks`.
- `outlier_tensor_names` (inspect): JSON list of the flagged tensors.
- `anomaly_score` (inspect): 0 to 1. It is the share of flagged tensors, with NaN / Inf tensors counted five times; 0
  means nothing was flagged.
- `global_cosine` (compare): 1.0 means the two VAEs point the same way overall.
- `most_divergent_blocks` (compare): the 10 blocks with the lowest cosine, lowest first.
- `block_heatmap`: an image.
  - Inspect: one row per block, columns mean / std / abs mean / outlier count, each scaled over the blocks (dark =
    low, bright = high); a red marker on blocks with outliers.
  - Compare: 1 − cosine per block (red = different).
- `benchmark_json`: empty when `benchmark` is off. Otherwise:
  - `reconstruction`: per image type and size, `mse`, `psnr_db`, `max_abs_err`, `mean_shift_rgb` and `finite`, plus
    `mean_psnr_db_by_resolution`, `mean_psnr_db`, `all_finite`, `failed_cases` and `seconds`.
  - `latents`: decodes of random latents (seeds 0 to 2, scales 0.5 and 1.0) of the shape this VAE produces, with
    `out_min`, `out_max` and `clip_fraction` (share of values outside 0..1).
  - Compare also lists `psnr_delta` (B minus A) per case.
  - Precision modes are not tested (`not_included`): that would change the dtype of the connected VAE.

### Common mistakes

- **"connect the second VAE to vae_b"**: `compare` mode without `vae_b`.
- **"benchmark resolutions: ..."**: a size that is not a whole number, not a multiple of 8, outside 32 to 1024, or more
  than 4 sizes. The node checks this before any work.
- **A case in `failed_cases`**: that encode or decode raised an error. Its entry in `cases` has the error text and no
  numbers.
