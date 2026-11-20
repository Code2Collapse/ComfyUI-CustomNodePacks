# Prompt Relay

Prompt Relay gives one video several prompts in a row: one prompt for the first part of the clip, another for the next
part, and so on, plus a global prompt that holds for the whole clip. It is a port of Gordon Chen's Prompt Relay
(<https://github.com/GordonChen19/Prompt-Relay>); attribution and licence notes are in
`nodes/prompt_relay/NOTICE.md`.

| Node | Use it to |
|---|---|
| [Prompt Relay Encode](#prompt-relay-encode-promptrelayencodec2c) | Encode the prompts and patch the model so each part of the video follows its own prompt |
| [Prompt Relay Advanced Options](#prompt-relay-advanced-options-promptrelayadvancedoptionsc2c) | Tune how hard and how wide the segments are; save and load settings as profiles |
| [Prompt Relay Restore (Kijai)](#prompt-relay-restore-kijai-promptrelayrestorekijaic2c) | Remove the patch from a Kijai WanVideo model |

**Status.** The commit history records unit tests with stand-in models, and checks against the WanVideoWrapper source
(96d9561). No run with real Wan or LTX weights is recorded: output quality is NOT VERIFIED. The 2026-11-19 additions
(segment report, role sections, safe restore, option profiles, capability report; commit 4ebaa55) were tested on the
CPU with stand-ins; the live UI is NOT VERIFIED.

## How it works, in plain words

1. The node joins your prompts into one text: the global prompt, then each local prompt in order. It encodes that text
   once and notes which tokens belong to which local prompt.
2. The clip is divided in time: each local prompt gets a run of frames (a **segment**).
3. The model is patched at its cross-attention, the step where the video "reads" the text. For each frame, the tokens
   of local prompts whose segment is far away get a penalty, so that frame mostly reads its own local prompt. Inside a
   flat zone around a segment's middle there is no penalty; past it the penalty grows quickly. The global prompt's
   tokens are never penalised, so they apply to every frame.
4. Only the positive (conditional) pass is changed. When the sampler marks the negative pass (ComfyUI's own samplers
   do), the negative prompt is not penalised.

Supported models: ComfyUI's own Wan (text-to-video and image-to-video cross-attention) and LTX (its video
cross-attention, and its audio cross-attention when the model has one), Kijai's WanVideoWrapper, and other models through a generic fallback that patches modules
named `cross_attn` or `attn2`.

## Prompt Relay Encode (`PromptRelayEncodeC2C`)

**Menu:** 🐺 C2C/🧰 Core/Prompt Relay · **Parameters:** [NODE_REFERENCE](../NODE_REFERENCE.md#promptrelayencodec2c)
Needs a video model (Wan or LTX) and its text encoder; the `kijai` backend needs the ComfyUI-WanVideoWrapper pack.

### What it is for

It turns a global prompt and a list of timed local prompts into conditioning plus a patched model (or, for Kijai,
text embeds plus a patched WanVideo model). The removed Prompt Relay Encode (Smart) and Prompt Relay Encode (Kijai)
nodes were merged into it (`backend` `smart` / `kijai`); saved workflows load as this node.

### When to pick it

- To make a clip change subject, action or camera over time from one sampling run.
- One prompt for the whole clip needs no Prompt Relay: use a normal text encode.

### Minimal wiring

`native` / `smart`:

```
Load Diffusion Model ──► model  ┐
Load CLIP ───────────► clip     ├─ Prompt Relay Encode ──► model ────► KSampler (model)
Empty video latent ──► latent   ┘                       └─► positive ─► KSampler (positive)
```

`kijai`:

```
WanVideoModelLoader ─────────► wan_model ┐
LoadWanVideoT5TextEncoder ───► wan_t5    ┴─ Prompt Relay Encode ──► wan_model ──────► WanVideoSampler
                                                                └─► wan_text_embeds ─► WanVideoSampler
```

Every backend also gives `segment_report`. The node shows only the sockets and widgets of the chosen `backend`;
switching the backend disconnects the inputs the new backend does not use.

### Key controls

- **`backend`**: `native` (MODEL + CLIP, prompts split by `|`), `smart` (MODEL + CLIP, segments read from
  `smart_prompt`), `kijai` (WanVideoWrapper).
- **`global_prompt`**: holds for the whole clip.
- **`role_sections`** (new 2026-11-19): an optional structured addition to the global prompt, one line per role:
  `subject:`, `action:`, `camera:`, `environment:`, `style:`. They are appended to `global_prompt` in that order, as
  sentences. All backends.
- **`local_prompts`** (`native`, `kijai`): the segment prompts, separated by `|`.
- **`segment_lengths`** (`native`, `kijai`): comma-separated frame counts in pixel frames, one per local prompt; empty
  = equal parts. They are converted to latent frames (4 pixel frames per latent frame for Wan).
- **`smart_prompt`** (`smart`): either `|`-separated parts, each with an optional weight tag such as `[2]` or a range
  `[0-24]`, or blocks under header lines that end in a number or range and a colon (`Scene 1:`, `Shot 0-24:`). A range
  header gives the segment a weight of its length. The weights set the segments' shares of the clip. Number words in
  headers ("Scene one:") need the optional `word2number` package. `normalize_by_tokens` scales each weight by the
  segment's token count. With an empty `global_prompt`, the first segment's text is used as the global prompt.
- **`epsilon`** (default 0.001): how sharp the segment edges are. The falloff width is 1 / ln(1/epsilon) latent
  frames: about 0.14 at 0.001, about 0.43 at 0.1. Larger = softer transitions.
- **`kijai` only:** `latent_frames` must be the clip's latent frame count, (pixel frames − 1) // 4 + 1 (81 pixel
  frames → 21). `negative_prompt` is encoded for the sampler's negative embeds. `encode_device` runs the T5 encode on
  `gpu` or `cpu`.
- **`relay_options`**: connect [Prompt Relay Advanced Options](#prompt-relay-advanced-options-promptrelayadvancedoptionsc2c)
  to tune the penalty.

### Reading the outputs

- `model` / `positive` (`native`, `smart`): the patched model and the conditioning for the sampler.
- `wan_model` / `wan_text_embeds` (`kijai`): the patched WanVideo model and the text embeds.
- `segment_report` (new 2026-11-19, every backend): JSON with `backend`, `arch` (`wan`, `ltx`, `kijai` or `generic`),
  `global_prompt`, `roles`, the joined `full_prompt`, `global_token_range`, and per segment its `text`,
  `token_range`, `token_count`, `latent_frame_range`, `approx_pixel_frames` (and `weight` for `smart`); plus
  `latent_frames`, `tokens_per_frame`, `temporal_stride` and `epsilon`. Read it to check that each prompt landed on the
  frames you meant.

### Common mistakes

- **The `kijai` patch outlives the node.** Kijai's sampler gives no per-run hook, so the masks are patched into the
  loaded model and stay after you delete this node. Run **Prompt Relay Restore (Kijai)** on the model, or reload it.
- **"At least one local prompt is required (separate with |)."**: `local_prompts` is empty.
- **"Number of segment_lengths (N) must match number of local prompts (M)"**: give one length per local prompt, or
  leave `segment_lengths` empty.
- **"Local prompt produced no tokens"**: a local prompt is empty or only punctuation.
- **"already patched by another node (e.g. KJNodes NAG)"**: another node changed the same cross-attention first.
  Stacking is not supported; remove the other node.
- **"could not locate cross-attention modules on this model"**: the generic fallback found nothing to patch. For a
  WanVideoWrapper model use the `kijai` backend.
- **"connect `wan_model` ... and `wan_t5`"** / **"connect `model`, `clip`, and `latent`"**: the inputs of the chosen
  backend are missing.
- **"Kijai T5 encoder is missing .tokenizer"**: update ComfyUI-WanVideoWrapper.
- **"role_sections line N ... must start with one of subject:, action:, ..."**: every non-empty line needs one of the
  five role names and a colon.
- **`latent_frames` left at 81 for an 81-frame clip** (`kijai`): 81 is the latent count of a 321-frame clip. Use the
  formula above.

## Prompt Relay Advanced Options (`PromptRelayAdvancedOptionsC2C`)

**Menu:** 🐺 C2C/🧰 Core/Prompt Relay · **Parameters:** [NODE_REFERENCE](../NODE_REFERENCE.md#promptrelayadvancedoptionsc2c)

### What it is for

Optional tuning for Prompt Relay Encode, wired to its `relay_options` input. Without it, every control is at its
neutral value.

### Minimal wiring

```
Prompt Relay Advanced Options ──► relay_options ──► Prompt Relay Encode
                              └─► capability_report
```

### Key controls

- `video_strength` (1.0): multiplier on the penalty. 0 turns segmentation off (every segment sees the whole clip).
- `video_window_scale` (1.0): scales the flat zone around each segment's middle (by default half the segment minus 2
  latent frames). Below 1 the falloff starts sooner; above 1 the zone is wider.
- `audio_epsilon`, `audio_strength`, `audio_window_scale`: the same for the LTX audio stream. `audio_epsilon` 0 =
  use the encoder's `epsilon`. Wan (native or Kijai) and the generic path have no audio stream and ignore these three.
- **Profiles** (new 2026-11-19): `profile_action` `save` stores the five values under `profile_name` (range-checked)
  in `user/c2c/prompt_relay_profiles.json` in your ComfyUI user folder; `load` uses a saved profile's values instead of
  the widgets; `none` uses the widgets. `profile_target` (`any`, `wan (native)`, `ltx (native)`, `kijai wan`,
  `generic`) is stored with the profile and picks which path the report describes.

### Reading the outputs

- `relay_options`: the bundle for Prompt Relay Encode.
- `capability_report` (new 2026-11-19): text. Per encoder path, which of the five controls it reads and which it
  ignores; notes for values that change nothing (strength or window scale at 1.0, `audio_epsilon` 0) or turn
  segmentation off (`video_strength` 0); and the profile saved or loaded.

### Common mistakes

- **"profile_action=save needs a profile_name"** (or `load`): type a name.
- **"no profile named ..."**: the error lists the saved names.
- **"the profile file ... is unreadable"** or **"not a JSON object"**: fix or delete
  `user/c2c/prompt_relay_profiles.json`.
- **Changing the audio controls on a Wan model**: no effect; the capability report says so.

## Prompt Relay Restore (Kijai) (`PromptRelayRestoreKijaiC2C`)

**Menu:** 🐺 C2C/🧰 Core/Prompt Relay · **Parameters:** [NODE_REFERENCE](../NODE_REFERENCE.md#promptrelayrestorekijaic2c)
Needs ComfyUI-WanVideoWrapper.

### What it is for

It removes a Prompt Relay `kijai` patch from a WanVideo model, putting back the original cross-attention of every
block. Only the `kijai` backend needs it: the `native` and `smart` backends put their patch on a copy of the model
(ComfyUI object patches), so the loaded model itself is not changed.

### Minimal wiring

```
WanVideoModelLoader ──► model ──► Prompt Relay Restore (Kijai) ──► model ──► WanVideoSampler
                                                              └─► patch_state
```

### Key controls

`on_foreign_patch` (new 2026-11-19) decides what happens when another node patched a block's cross-attention AFTER
Prompt Relay:
- `restore anyway` (default, the old behaviour): put Prompt Relay's saved original back, which drops the other patch.
- `keep foreign patch`: leave that block's current patch and remove only Prompt Relay's bookkeeping.
- `error`: stop before changing anything.

### Reading the outputs

- `model`: the same model, restored.
- `patch_state`: JSON with the state `before` and `after` (blocks found, patched, foreign patches), `restored`,
  `kept_foreign`, `failed`, `policy`, and a `verdict`: `clean`, `was not patched`, `restore incomplete` or `foreign
  patch kept`.

### Common mistakes

- **"blocks [...] were patched by another node after Prompt Relay; nothing was changed"**: you chose `error`. Pick
  `keep foreign patch` or `restore anyway`.
- **Verdict `restore incomplete`**: some blocks could not be restored (`failed` lists them). Reload the model.
