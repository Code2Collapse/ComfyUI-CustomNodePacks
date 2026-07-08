# Live sampling preview — smart Auto

> The **C2C ▸ Sampler latent preview** setting controls the live denoising
> preview that appears inside a sampler node. Its **Auto** mode picks the right
> previewer per model on every run, so SD/SDXL/Flux and Wan/video samplers all
> preview correctly with no manual switching.

Setting: **Settings → C2C ▸ Sampler latent preview**.

---

## Why "Auto" has to be smart

ComfyUI's own Auto preview resolves to Latent2RGB, which has no color factors
for Wan/video latents — so a Wan sampler shows a **blank** preview. The usual
workaround is to force TAESD everywhere, but that's overkill for image models.
The C2C Auto instead chooses **per model**, and it does so inside the
previewer resolver so it stays correct on every sampler callback (it survives
ComfyUI's per-prompt preview reset):

| Latent type | Detected by | Previewer chosen |
|---|---|---|
| **Video** — Wan 2.1/2.2, HunyuanVideo, LTXV, Mochi, Cosmos | temporal latents (`latent_dimensions ≥ 3`), the `taew*`/`taehv*` decoder name, or the format's class name | **TAESD** — Wan/Kijai samplers route this to their own video previewer; falls back to a Wan-factor Latent2RGB if the decoder file is absent (never blank) |
| **Image** — SD 1.5, SDXL, Flux, SD3 | everything else | **Core Auto** — TAESD when the decoder is present (sharp), else fast Latent2RGB |

---

## The four options

| Option | Behaviour |
|---|---|
| **Auto — smart, per model** *(default, recommended)* | Chooses per the table above |
| **Force On — Wan/video-aware (TAESD)** | Always TAESD |
| **Force On — fast, SD/Flux only (Latent2RGB)** | Always Latent2RGB (blank for Wan) |
| **Off** | No sampler preview |

---

## Sharper Wan previews (optional)

For a crisp video preview instead of the rough Wan-factor fallback, drop the
Wan approx decoders into `ComfyUI/models/vae_approx/`:

- `taew2_1.safetensors` — Wan 2.1
- `taew2_2.safetensors` — Wan 2.2

(available from Kijai's WanVideo_comfy on Hugging Face). With those present,
Auto's TAESD path shows a proper decoded video preview.

---

## How it's implemented (safe by design)

This is **backend-only** and **purely additive**. It does not draw anything,
does not touch any node, and does not overlap ComfyUI's own UI — it only
ensures ComfyUI's *native* in-node previewer runs, and picks the best method
for the model. The `get_previewer` wrapper is fully guarded: any change in a
future ComfyUI release simply no-ops and leaves core untouched, and the
previewer can never return `None` (so previews never silently stop).

Opt out entirely with the environment variable `C2C_NO_FORCE_PREVIEW=1`.

---

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| Wan preview blank | Setting is on `Latent2RGB` — switch to **Auto** or **TAESD** |
| Wan preview rough/low-res | Add `taew2_1`/`taew2_2.safetensors` to `models/vae_approx/` |
| No preview at all | Setting is **Off**, or ComfyUI launched with `--preview-method none` and the guard is disabled by `C2C_NO_FORCE_PREVIEW=1` |
| Existing install still on `taesd` | The default changed to Auto for new installs; older installs keep the saved value — set it to **Auto** once |

---

## License

Apache-2.0. Backend-only, no core overlap.
