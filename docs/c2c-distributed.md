# C2C Distributed: multi-GPU for ComfyUI (design, L9.38)

Status 2026-10-24: slices 0, 1 and 2a built and tested (GPU Inventory node; exact Ulysses / Ring / USP attention;
C2C Sequence Parallel node for Wan 2.1 / 2.2 and MiniMax H3 on one box). Speed-up is **not measured yet**: that needs a machine with
two or more GPUs (see "What is measured"). This file is the A7 design record: what the subsystem does, what it refuses
to do, how it fails, what it costs, and how to turn it off.

## What it is for

One ComfyUI, several GPUs, used together:

| Mode | What runs where | Speeds up | Where it comes from |
|---|---|---|---|
| **CFG / batch split** | cond and uncond (and batch chunks) of each step on different GPUs | one generation, up to ~2x with CFG > 1 | **ComfyUI core** (`MultiGPU CFG Split`, 0.39+) for any model; inside **C2C Sequence Parallel** the batch split of each call comes first and is combined with sequence parallelism (e.g. 2 x 4 on 8 GPUs) |
| **Sequence parallel (Ulysses / Ring / USP)** | every attention layer splits the token sequence over the GPUs | one long video, attention-bound models | **C2C Sequence Parallel** (this package), one box, any OS |
| **Data parallel** | each GPU or machine renders whole jobs: seeds, shots, batches | throughput | C2C Farm (`renderfarm/`); fan-out node = slice 3 |
| **FSDP (weight sharding)** | each GPU holds 1/N of the weights | a model that does not fit one GPU | slice 2b, Linux + NCCL |
| **Tile work-stealing** | upscale / refine tiles pulled by whichever GPU is free | big frames | slice 4 |

It lives in `c2c_distributed/` (a sub-package of this pack).

## Hard facts this design rests on (measured, not assumed)

| Fact | Measured | Consequence |
|---|---|---|
| Windows torch 2.9.0+cu130: `is_nccl_available()` False; gloo cannot create a device (`makeDeviceForHostname(): unsupported gloo device`, also with `GLOO_DEVICE_TRANSPORT` UV / TCP_LIBUV) | 2026-10-24, owner's PC | no torch.distributed collectives on Windows at all |
| ComfyUI core 0.39 ships `comfy/multigpu.py`: per-GPU deep clones reloaded from the loader (`deepclone_multigpu`), LoRA / patches re-synced every run (`match_multigpu_clones`), a persistent per-device thread pool, and the `MultiGPU CFG Split` / `Select Model/CLIP/VAE Device` nodes | read 2026-10-24 (`comfy_extras/nodes_multigpu.py`) | CFG split is core's; we reuse core's clones and thread pool instead of a worker-process pool |
| Core attention dispatch honours `transformer_options["optimized_attention_override"]`, called as `override(func, q, k, v, heads, mask=..., skip_reshape=..., skip_output_reshape=..., transformer_options=...)` | `comfy/ldm/modules/attention.py` `wrap_attn` | sequence-parallel attention plugs in without copying any model code |
| Wan (`WanModel.forward_orig`) and MiniMax H3 (`MiniMaxH3Model._forward`) run every transformer block through `patches_replace["dit"][("double_block", i)]` with the token stream as `img` and the RoPE table as `pe` / `rope_freqs` | read 2026-10-24 | tokens can be cut before block 0 and gathered after the last block through a documented hook |
| One process can drive several GPUs from threads; tensors move GPU-to-GPU with `.to(device)` (P2P / NVLink when present) | core does exactly this for CFG split | **the thread backend works on Windows**, no NCCL needed on one box |
| The cu130 wheel carries sm_75 / 80 / 86 / 90 / 100 / 120 kernels | `torch.cuda.get_arch_list()` | Turing to Blackwell; Hopper and B200 / GB200 native; B300 (sm_103) and GB10 / DGX Spark (sm_121) run the same-major cubins; the Inventory node reports this per GPU |
| Grace hosts (GH200 / GB200 / DGX Spark) are aarch64 Linux | NVIDIA platform docs | nothing in the package assumes x86 |
| Ulysses needs heads divisible by its degree | DeepSpeed-Ulysses, arXiv 2309.14509 | the planner splits into Ulysses x Ring (USP, arXiv 2405.07719) |

Sources and the survey of raylight / ComfyUI-Distributed: workspace `docs/evidence/L9.38/research.md`.

## How C2C Sequence Parallel runs a model call

```
sampler thread (rank 0, GPU 0)                 core's per-device threads (ranks 1..N-1)
  CALC_COND_BATCH wrapper: takes core's MultiGPU clones away from core's cond splitter for this model
  DIFFUSION_MODEL wrapper: batch items (cond / uncond) to GPU groups first - no exchange - then the SAME call
                           on every GPU of a group, inputs moved to each GPU's clone
     Wan / H3 forward (unchanged core code)        same, on the clone
       block 0 patch: cut tokens + RoPE rows       cut its own shard
       self-attention -> override:  Ulysses all-to-all (heads) / Ring K/V passes, exact
       cross-attention to text: local, no exchange
       last block patch: gather tokens             gather
     head / unpatchify on the whole sequence       (result discarded)
```

* `c2c_distributed/comm.py`: one `Comm` interface (all_gather, all_to_all with uneven sizes, send / recv, ring
  step, broadcast, barrier, sub-groups). `ThreadComm`: ranks as threads, tensors moved to the receiver's device,
  every collective times out (600 s default) and a failing rank releases the others. `TorchComm`:
  torch.distributed (NCCL) for slice 2b.
* `c2c_distributed/attention.py`: Ulysses, Ring (online-softmax merge of per-block log-sum-exp) and the USP hybrid
  over a `Mesh` (ring x ulysses). The pure-Ulysses path calls the model's own attention kernel (SDPA / flash /
  sage) on the gathered head group; Ring steps use an exact block kernel. Global attention masks are sliced per
  rank (raylight's USP path drops the mask).
* `c2c_distributed/usp.py`: the wrappers and patches above, model adapters (`WanModel`; `MiniMaxH3Model` with its
  packed stream and modulation segments re-mapped per shard), refusals.
* `c2c_distributed/planner.py`: Ulysses as large as heads and the fast-link domain allow, Ring for the rest;
  calls with too few tokens per GPU use fewer GPUs. The batch degree (`usp._layout`) is the largest divisor of the
  GPU count that divides the call's batch (Wan; H3 runs batch 1): a batch split needs no exchange, so it goes first.
  E.g. 8 GPUs, CFG on: cond and uncond on 4 GPUs each, Ulysses 4 inside each group.
* `c2c_distributed/inventory.py`: GPUs, architecture, kernels in this torch build, P2P and NVLink matrix, Grace /
  aarch64 host, NCCL / gloo usability measured in a subprocess, which modes can run.

No ComfyUI model code is copied (ComfyUI is GPL-3.0, this pack Apache-2.0): only core's extension points.

## Nodes

| Node | Does |
|---|---|
| **C2C GPU Inventory** | the report above, on the node and as JSON |
| **C2C Sequence Parallel (multi-GPU)** | MODEL -> MODEL; `gpus` (0 = all), `strategy` auto / ulysses / ring, `min_tokens_per_gpu`, `fast_link_gpus`. Put it last in the model chain. With one GPU the model passes through unchanged. |

## What is measured (2026-10-24, owner's PC: one RTX 4060 Laptop, CPU testbed)

| Check | Result | Evidence |
|---|---|---|
| Ulysses / Ring / USP attention vs one-device attention, 2-8 ranks, 37 tokens (uneven shards), with and without a mask, float64 | max error <= 1.9e-15 | `tests/test_c2c_distributed.py` (local) |
| Tiny Wan forward split over 2 / 3 / 4 ranks (Ulysses, Ring, USP 2x2), CPU float32 | max error < 2e-5 x output scale | same |
| Same on the GPU, real CUDA kernels, bf16 and fp16, 1920 tokens, 2-4 ranks sharing cuda:0 | Ulysses 2 / 4: **bit-identical** to the unsplit model; Ulysses 3 and Ring: error vs an fp32 reference equal to the unsplit model's own (bf16 0.68-0.69 % both, fp16 0.085 % both) | workspace `docs/evidence/L9.38/usp_gpu_check.json` |
| End to end: core loader (`load_diffusion_model`, model detection) + core KSampler (euler, CFG 4, 3 steps) on a tiny Wan, with vs without the node, 2 / 3 simulated ranks | latents differ by <= 1.05e-5 (latent magnitude 12.6) | `tests/test_c2c_distributed.py` |
| Same for MiniMax H3 (tiny H3, joint video + audio latent, text-to-video layout), 2 / 3 ranks | video and audio latents differ by <= 3.8e-6 (magnitude 5.3-5.5) | same |
| Batch split: Wan sampler with CFG 4 on 2 ranks (cond / uncond apart) and 4 ranks (batch 2 x Ulysses 2) | latents differ by <= 1.12e-5 | same |
| Batch split on the GPU, bf16 / fp16, batch 2 on 2 and 4 ranks | error vs fp32 equal to the unsplit batched call (bf16 0.686 % vs 0.687 %, fp16 0.085 % both) | `usp_gpu_check.json` |
| Guard: are those comparisons meaningful? Halving every attention output | changes the latents by > 1e-2 (Wan and H3), so a broken split could not pass | same |
| `TorchComm` over real torch.distributed, separate processes (WSL Arch Linux, torch 2.9.0+cpu, gloo), 2 and 4 ranks: all_gather, uneven all_to_all, ring step, sub-groups, broadcast; Ulysses / Ring / USP attention vs one process, float64 | all pass; attention <= 1.3e-15. Found and fixed: gloo has no list all_to_all (now all_to_all_single), and send / recv / broadcast / ring took group ranks where torch wants global ranks | workspace `docs/evidence/L9.38/torchcomm_gloo_check.json` |
| Speed-up on real GPUs | **not measured** - needs >= 2 GPUs | owner / cloud run |
| MiniMax H3 image-to-video / reference layouts | segment re-mapping unit-tested; not sampled end to end | - |

`C2C_DISTRIBUTED_SIMULATE=N` runs N ranks on the one device a box has, through the whole path (clones, sampler,
exchanges): a test of the setup on a single-GPU or CPU machine. It is slower than not splitting and takes N x the
model memory; never use it for speed.

## Failure modes and what happens

| Failure | Behaviour |
|---|---|
| Model class without an adapter (VACE, Wan Animate, S2V, Flux, ...) | refused at the node with the list of supported models |
| Another node patches the transformer blocks or sets an attention override after this one | refused at run time, naming the patch ("put C2C Sequence Parallel last") - never a silently wrong picture |
| ControlNet / control conditioning | refused at run time (not supported yet) |
| A user attention override set BEFORE the node (e.g. Sage attention) | kept: it computes the local attention on every rank |
| Heads not divisible by the Ulysses degree | planner picks Ulysses x Ring that divides, or Ring only, and says so |
| Too few tokens per GPU | that call runs on one GPU; the node's log says so |
| A rank raises (OOM, interrupt) | every other rank is released from its collective; the run fails with that rank's error; rank 0's own exception keeps its type (ComfyUI's interrupt still works) |
| A rank stalls | the collective times out after 600 s and names the rank |
| Loader without a reload factory (some custom loaders, GGUF) | refused: core's multi-GPU clones need it ("use a core loader") |
| Core's `MultiGPU CFG Split` also in the chain | this node takes the clones; its own batch split does the cond / uncond part per call |
| Batch does not divide the GPU count (3 items, 2 GPUs) | tokens are split instead |

## Budgets

* No cost when unused: importing starts nothing; the Inventory probes run only when its node runs (backend probes
  cached per ComfyUI run).
* Memory: one full model copy per GPU (as core's CFG split). Weight sharding is slice 2b.
* Exchange per attention call: Ulysses moves q, k, v and the output once each (3 + 1 all-to-alls of the shard);
  Ring passes K and V N-1 times. On PCIe without P2P the copies go through host memory - the Inventory node warns.

## Compatibility

* Nothing changes for workflows that do not use the new nodes; with one GPU the model is unchanged.
* Kill switch: `C2C_DISTRIBUTED=off` - the nodes refuse with that reason, and a model already patched runs as one
  normal single-GPU call.
* Licences: Apache-2.0 like the pack. raylight (Apache-2.0) and ComfyUI-Distributed (Apache-2.0) were read for
  behaviour; their ComfyUI-derived files (GPL-3.0) were not copied.

## Slices

| Slice | What | State |
|---|---|---|
| 0 | C2C GPU Inventory | done |
| 1 | Comm, exact Ulysses / Ring / USP attention, planner | done |
| 2a | C2C Sequence Parallel, one box, Wan + MiniMax H3 | done; speed to measure on >= 2 GPUs |
| 2b | Multi-process / multi-node over NCCL (`TorchComm`), FSDP2 weight sharding | Linux + NCCL box needed |
| 2c | Batch (CFG) split x sequence parallel | done (Wan) |
| 2d | ControlNet; more adapters (VACE, Animate, Flux, LTX); H3 cond / uncond on separate GPU groups (H3 calls run batch 1) | to do |
| 3 | Data-parallel fan-out to Farm backends, collector | to do |
| 4 | Tile work-stealing upscale / refine | to do |
