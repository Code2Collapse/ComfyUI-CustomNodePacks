# AsymFlow Sampler Patch

## AsymFlow Sampler Patch (Lakonik signal-shift) (`AsymFlowSamplerPatch`)
**Menu:** 🐺 C2C/🧰 Core/Sampling · **Parameters:** [NODE_REFERENCE](../NODE_REFERENCE.md#asymflowsamplerpatch)
**Works only with flow models.** No image-quality comparison has been recorded for this node, so this page makes no
claim about how the results look.

### What it is for
It replaces a flow model's noise schedule: the mapping from sampler timestep to noise level (sigma). It does not
change the sampler, the model weights or the number of steps, and it needs no retraining. Only the noise level each
step lands on changes.

The schedule is adapted from AsymFlow (Lakonik / LakonLab). For a timestep `t` between 0 and 1:

```
r     = sqrt(t / (1 − t))
sigma = r / (shift + r)
```

The output is a copy of the model (`model.clone()`) whose `model_sampling` is swapped for this schedule. The model's
own denoise and noise-scaling functions are kept.

### Which models
- **Accepted:** models whose sampling uses ComfyUI's `CONST` (flow / rectified-flow) or `COSMOS_RFLOW` prediction.
  The node's error message names Flux, SD3, Wan and H3 as examples.
- **Refused:** models that use EPS or v-prediction (SD 1.5, SDXL and similar). The node stops with "AsymFlow Sampler
  Patch only works with flow models (…)" and names the model's prediction type. Before 2026-11-16 (commit 9840802)
  it patched such models anyway and gave wrong output without an error.

### Where it goes
```
Load Checkpoint / UNet ──► (LoRA loaders) ──► AsymFlow Sampler Patch ──► model ──► KSampler
```
Put it between the model loader (and any LoRAs) and the sampler.

Do not also use a core `ModelSampling…` node (for example ModelSamplingSD3 or ModelSamplingFlux) on the same model
path. Both replace the same `model_sampling` object, so whichever runs last decides the schedule. One placed before
this node has no effect; one placed after it undoes it.

### Key controls
- `shift` (default 3.0, range 0.05–20):
  - 1 = a plain linear flow;
  - above 1 = more of the steps are spent at high noise;
  - below 1 = the opposite.
  The tooltip suggests above 1 for high-resolution / Flux-like models.
- `multiplier` (default 1000) scales the timesteps (the discretisation). Leave it at 1000 unless you know the model
  expects another value.

### Reading the output
- `model`: the same model with the AsymFlow schedule. Wire it to the sampler.

### Common mistakes
- **An SD 1.5 / SDXL checkpoint.** It is refused (see above). Remove the node for those models.
- **A core ModelSampling node after this one.** It replaces the schedule again (see "Where it goes").
- **Running outside ComfyUI.** The node needs ComfyUI's `comfy.model_sampling` module. Without it, it stops with
  "comfy.model_sampling is unavailable".
