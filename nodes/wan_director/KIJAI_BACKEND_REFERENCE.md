# Kijai WanVideoWrapper Backend Reference

Authoritative mapping of Kijai `ComfyUI-WanVideoWrapper` types and chains, used by
`WanDirectorC2C` when `backend="kijai"`.

Source: vendored upstream at `third_party/ComfyUI-WanVideoWrapper/` (Apache-2.0).
Generated via thorough read of `nodes.py`, `nodes_sampler.py`, `nodes_model_loading.py`,
`nodes_utility.py`, `s2v/nodes.py`, `FlashVSR/flashvsr_nodes.py`.

---

## A. Canonical Types

| Type | Shape / Contents | Emitter | Consumer |
|---|---|---|---|
| `WANVIDEOMODEL` | wrapped transformer w/ patches | `WanVideoModelLoader` | `WanVideoSampler`, attention/block-swap modifiers |
| `WANVAE` | VAE with `.encode()` / `.decode()` | `WanVideoVAELoader`, `WanVideoFlashVSRDecoderLoader` | `WanVideoImageToVideoEncode`, `WanVideoDecode` |
| `WANTEXTENCODER` | `{"model": T5, "dtype": dtype}` | `LoadWanVideoT5TextEncoder` | `WanVideoTextEncode*` |
| `WANVIDEOTEXTEMBEDS` | `{"prompt_embeds":[B,512,4096], "negative_prompt_embeds":..., "echoshot":bool}` | `WanVideoTextEncode*`, `WanVideoApplyNAG` | `WanVideoSampler` |
| `WANVIDIMAGE_EMBEDS` | dict with `image_embeds[C,T,h/8,w/8]`, `num_frames`, `lat_h/w`, `mask`, `vae`, optional CLIP context | `WanVideoImageToVideoEncode`, `WanVideoAnimateEmbeds`, T2V latent encoder | `WanVideoSampler` |
| `WANVIDIMAGE_CLIPEMBEDS` | `{"clip_embeds":..., "negative_clip_embeds":...}` | `WanVideoClipVisionEncode` | embed assembly |
| `WANVIDLORA` | `[{"path","strength","name",...}]` | `WanVideoLoraSelect*` | `WanVideoModelLoader` |
| `WANCOMPILEARGS` | torch.compile dict | `WanVideoTorchCompileSettings` | model loader |
| `BLOCKSWAPARGS` | block-swap dict | `WanVideoBlockSwap` | model loader |
| `VRAM_MANAGEMENTARGS` | aggressive offload | `WanVideoVRAMManagement` | model loader |
| `FETAARGS` | Enhance-A-Video params | `WanVideoEnhanceAVideo` | sampler |
| `WANVIDCONTEXT` | context-window schedule | `WanVideoContextOptions` | sampler |
| `CACHEARGS` | TeaCache/MagCache/EasyCache | cache setting nodes | sampler |
| `SLGARGS` | skip-layer guidance | `WanVideoSetSkipLayerGuidance` | sampler |
| `LOOPARGS` | loop-sampling cfg | loop nodes | sampler |
| `EXPERIMENTALARGS` | experimental flags | experimental nodes | sampler |
| `SIGMAS` | custom noise schedule | scheduler nodes | sampler |
| `UNIANIMATE_POSE` | `{"pose","ref","strength",...}` | `WanVideoUnianimeLoader` | sampler |
| `FANTASYTALKING_EMBEDS` | audio embeds + projection | fantasy talking nodes | sampler |
| `UNI3C_EMBEDS` | 3D render control | Uni3C loader | sampler |
| `MULTITALK_EMBEDS` | `{"audio_features":[T], "audio_scale":float,...}` | MultiTalk processors | sampler |
| `FREEINITARGS` | FreeInit frequency filter | `WanVideoFreeInitSettings` | sampler |
| `AUDIO_ENCODER_OUTPUT` | audio features dict | S2V audio encoders | `WanVideoAddS2VEmbeds` |
| `VACEPATH` | extra model paths | `WanVideoVACEModelSelect` | model loader |

## B. Canonical Loader Chain (Wan 2.1 / 2.2)

```python
model      = WanVideoModelLoader(model=..., base_precision="bf16",
                                 quantization="disabled",
                                 load_device="offload_device",
                                 attention_mode="sdpa")            # -> WANVIDEOMODEL
lora       = WanVideoLoraSelect(lora="...", strength=1.0)          # optional -> WANVIDLORA
vae        = WanVideoVAELoader(model_name="Wan2_1_VAE_fp32.safetensors",
                               precision="fp32")                    # -> WANVAE
t5         = LoadWanVideoT5TextEncoder(model_name="umt5-xxl-enc-bf16.safetensors",
                                       precision="bf16")            # -> WANTEXTENCODER
embeds     = WanVideoTextEncodeCached(positive_prompt=..., negative_prompt=...,
                                      use_disk_cache=True)          # -> WANVIDEOTEXTEMBEDS
img_embeds = WanVideoImageToVideoEncode(vae=vae, width=832, height=480,
                                        num_frames=81,
                                        start_image=ref)            # -> WANVIDIMAGE_EMBEDS
samples, _ = WanVideoSampler(model, image_embeds=img_embeds,
                             text_embeds=embeds, steps=30, cfg=6.0,
                             shift=5.0, seed=42, scheduler="unipc")
images     = WanVideoDecode(vae=vae, samples=samples)
```

## C. Text Encoding

- UMT5-XXL T5, output `[B, 512, 4096]`.
- Pipe-separated prompts (`a|b|c`) → equally spaced segments over video.
- EchoShot tags `[1]...[2]...[3]` → multi-shot mode.
- Weight syntax `(text:1.2)` → per-segment multiplier.
- SHA256 prompt hash → disk cache under `text_embed_cache/`.
- NAG: `WanVideoApplyNAG(positive, negative, nag_scale=11.0, nag_tau=2.5, nag_alpha=0.25)`.

## D. Sampler — `WanVideoSampler`

Required: `model`, `image_embeds`, `steps`, `cfg`, `shift`, `seed`,
`force_offload`, `scheduler` (unipc / karras / exponential / simple / bpe /
gits / pisa / teacache / magcache / easycache / multitalk / ...),
`riflex_freq_index`.

Optional (20+): `text_embeds`, `samples` (v2v init), `denoise_strength`,
`feta_args`, `context_options`, `cache_args`, `batched_cfg`, `slg_args`,
`rope_function` (default/comfy/comfy_chunked/mocha), `loop_args`,
`experimental_args`, `sigmas`, `unianimate_poses`,
`fantasytalking_embeds`, `uni3c_embeds`, `multitalk_embeds`,
`freeinit_args`, `start_step`, `end_step`, `add_noise_to_samples`.

Returns `("LATENT","LATENT")` → `(samples, denoised_samples)`.

Dual-CFG Wan 2.2 high/low noise: detected via `in_dim` in state_dict
(16=T2V, 48=I2V/5B). Both expert noise schedules computed internally.
`cfg` may be a single float OR a list of per-step floats.

## E. Audio / S2V

Input audio: `{"waveform":[1,2,N], "sample_rate":16000}`.

Chain: `AudioEncoder(audio) -> AUDIO_ENCODER_OUTPUT ->
WanVideoAddS2VEmbeds(embeds, audio_encoder_output, frame_window_size=80,
audio_scale=1.0, pose_start_percent, pose_end_percent, enable_framepack)
-> (image_embeds, audio_frame_count)`.

Internal alignment: 50 fps → 30 fps → 16 fps buckets, 512-dim/frame.

## F. Image-to-Video

`WanVideoImageToVideoEncode(vae, width, height, num_frames, start_image,
end_image?, noise_aug_strength=0.0, start_latent_strength=1.0,
end_latent_strength=1.0, force_offload=True) -> WANVIDIMAGE_EMBEDS`

Image normalization: RGB [0,1] → [-1,1], lanczos resize.
Latents: `[C=16 or 48, T=(F-1)//4+2, H/8, W/8]`.
`num_frames` quantized: `((n-1)//4)*4 + 1`.
Resolution multiple of `VAE_STRIDE = (4, 8, 8)`.

## G. FlashVSR

```
flashvsr_decoder = WanVideoFlashVSRDecoderLoader(model_name=..., precision="bf16") -> WANVAE
image_embeds     = WanVideoAddFlashVSRInput(image_embeds, low_res_images, strength=1.0)
```
4× upscale; LQ frames at 1/4 resolution, `LQ_proj_in` projects to UNet space,
`Buffer_LQ4x_Proj` upsamples 4×.

## H. Differences vs Native ComfyUI Wan

| Aspect | Kijai | Native |
|---|---|---|
| Latent shape | `[C, T, H, W]`, C=16/48 | `[B, 4, H, W]` |
| VAE API | `WANVAE.encode([frames], device, tiled=)` → `{"samples":T}` | `VAE.encode(pixels) -> LATENT` |
| Text conditioning | T5 only (4096-dim) | CLIP + T5 hybrid |
| Sampler output | 2 LATENTs (samples + denoised) | 1 LATENT |
| Precision | `fp8_e4m3fn_scaled`, `fp16_fast` w/ matmul control | fp16, fp32 |
| Attention | Sage/Radial/UltraVico | SDPA |
| Memory | Explicit block-swap & offload device | ComfyUI default |
| RoPE | default/comfy/comfy_chunked/mocha | comfy ≤2D |
| Dual-CFG (2.2) | Implicit high/low noise expert routing | Not exposed |
| Context windows | Multi-window scheduling | n/a |
| LoRA | Diffusers / LyCORIS / Fun LoRA | Native LoRA |

## I. Director Sockets when `backend="kijai"`

```python
RETURN_TYPES = (
    "WANVIDEOMODEL",          # model
    "WANVAE",                 # vae
    "WANTEXTENCODER",         # text_encoder
    "WANVIDEOTEXTEMBEDS",     # text_embeds_pos
    "WANVIDEOTEXTEMBEDS",     # text_embeds_neg
    "WANVIDIMAGE_EMBEDS",     # image_embeds
    "WANVIDLORA",             # lora
    "WANCOMPILEARGS",         # compile_args
    "BLOCKSWAPARGS",          # block_swap_args
    "WANVIDCONTEXT",          # context_options
    "CACHEARGS",              # cache_args
    "SLGARGS",                # slg_args
    "FETAARGS",               # feta_args
    "FREEINITARGS",           # freeinit_args
    "LOOPARGS",               # loop_args
    "WANVIDIMAGE_CLIPEMBEDS", # clip_embeds
    "UNIANIMATE_POSE",        # poses
    "FANTASYTALKING_EMBEDS",  # audio_embeds_s2v
    "MULTITALK_EMBEDS",       # audio_embeds_multitalk
    "SIGMAS",                 # sigmas
    "FLOAT",                  # frame_rate
    "AUDIO",                  # combined_audio (raw mux)
    "IMAGE",                  # reference_image
    "STRING",                 # info
)
```

When `backend="native"` the first 6 sockets collapse to
`(MODEL, CLIP, VAE, CONDITIONING_pos, CONDITIONING_neg, LATENT)` and the
remaining Kijai-only sockets emit `None` (downstream nodes guard with
`if x is not None`).
