# Magnific / OmniScale sign-in nodes

Fifteen nodes that use Magnific's hosted service: image, video and music generation, voiceover, image and video
upscaling, skin enhancement, background removal, retouch, stock images, and pickers for your Magnific library.
The menu shows them as **OmniScale …**. Their node ids start with `Magnific`, so workflows saved with the vendor's
own Magnific pack load here.

**Read this first:**
- **You need a Magnific account and must sign in.** Nothing here runs offline or on your GPU. Every result is made
  on Magnific's servers.
- **Paid requests.** Generation, upscaling, voiceover, skin enhancement, background removal, retouch and stock
  downloads are billed per call by Magnific, in credits on your account. The table below says which nodes cost
  credits.
- **Not verified live.** The pack's tests never send a live paid request (commit 5fce802). The rules that show or
  hide sliders per mode follow the code comments and tooltips. They are NOT VERIFIED against the Magnific API
  (`nodes/magnific/nodes.py`).
- **Origin.** The nodes are ported from the vendor's magnific-comfyui 0.7.0. The vendor declares that pack
  UNLICENSED (`nodes/magnific/__init__.py`, CREDITS.md).

| Node | Use it to | Costs credits |
|---|---|---|
| [OmniScale Save To (project/folder)](#omniscale-save-to-projectfolder-magnificsaveto) | Choose where new results are stored | no |
| [OmniScale Library Reference](#omniscale-library-reference-magnificlibraryreference) | Attach Library characters, styles, elements, locations | no |
| [OmniScale Metadata (unpack)](#omniscale-metadata-unpack-magnificmetadata) | Split a result's metadata into sockets | no |
| [OmniScale Creation Picker](#omniscale-creation-picker-magnificcreationpicker) | Load an earlier result from your library | no |
| [OmniScale Generate Image](#omniscale-generate-image-magnificgenerateimage) | Make images from a prompt | yes, per image |
| [OmniScale Generate Video](#omniscale-generate-video-magnificgeneratevideo) | Make a video clip | yes |
| [OmniScale Generate Music](#omniscale-generate-music-magnificgeneratemusic) | Make music | yes |
| [OmniScale Voiceover (TTS)](#omniscale-voiceover-tts-magnificvoiceover) | Turn text into speech | yes |
| [OmniScale Upscale Image](#omniscale-upscale-image-magnificupscaleimage) | Upscale a picture | yes |
| [OmniScale Upscale Video](#omniscale-upscale-video-magnificupscalevideo) | Upscale a clip | yes |
| [OmniScale Skin Enhancer](#omniscale-skin-enhancer-magnificskinenhancer) | Enhance a portrait | yes |
| [OmniScale Remove Background](#omniscale-remove-background-magnificremovebackground) | Cut out the subject and get its mask | yes |
| [OmniScale Retouch (inpaint)](#omniscale-retouch-inpaint-magnificretouch) | Replace or erase a masked area | yes |
| [OmniScale Stock Search](#omniscale-stock-search-magnificstocksearch) | Get a stock image by search, by index or at random | yes (the download) |
| [OmniScale Stock Picker](#omniscale-stock-picker-magnificstockpicker) | Search stock and click the image you want | yes (the download) |

## How sign-in works

1. Use the ComfyUI menu **Magnific → Sign in**. Alternatively, use the **Sign in to Magnific** button, which appears
   on Save To, Creation Picker and Library Reference while you are signed out.
2. A new browser tab opens on Magnific's site, and a message shows a code. Approve the sign-in there.
3. When it succeeds, the nodes reload your projects, models and voices. If the message asks you to, press **R**
   (refresh node definitions).

What happens behind this:
- The sign-in is the OAuth device flow against `auth.magnific.com`. The requests go to `mcp.magnific.com`.
- Your sign-in is saved as a plain file at `~/.magnific/comfyui_auth.json` in your home folder (or wherever the
  environment variable `MAGNIFIC_AUTH_FILE` points). This is the same file the vendor's own pack uses, so signing in
  once covers both.
- **Magnific → Sign out** deletes the session and clears the cached lists. Results remembered for reuse are also
  forgotten (see "Credits" below), so they are never shared across accounts.
- The sign-in, sign-out and cost routes answer only the ComfyUI page opened on the same machine (a loopback address
  plus a per-process token). Starting a new sign-in while one account is already signed in is refused: sign out
  first.
- While you are signed out, the project, voice and asset lists show "— sign in (Magnific menu) and press R to
  refresh —". Running a node then stops with "Not signed in to Magnific".
- The lists of projects, folders, models, voices and Library assets are cached, so ComfyUI's node list never waits
  for Magnific. The last list is shown while a fresh one loads in the background. After a failed fetch, the node
  waits 60 s before trying again (commit 99651c6). Press R to see a list that has just changed.

**Want an API key instead?** That is the separate **ComfyUI-OmniScale** pack.

### When ComfyUI-OmniScale is installed, these nodes stand down
CustomNodePacks checks its neighbours in `custom_nodes` when it loads (`_c2c_omniscale_installed` in the pack's
`__init__.py`). A folder counts as OmniScale when:
- its name starts with `comfyui-omniscale` (any case);
- the name does not end in `.disabled`;
- it contains an `__init__.py`.

When such a folder exists, CustomNodePacks does **not** register its 15 Magnific nodes. The console says "[C2C]
ComfyUI-OmniScale is installed: its API-key nodes provide Magnific*; CustomNodePacks' sign-in copy is not
registered". OmniScale registers the same 15 node ids, so saved workflows still load. Those nodes take an API key
(Settings → C2C → OmniScale → API key), not a sign-in. The pack then registers 86 nodes instead of 101 (README).

- Set the environment variable `C2C_MAGNIFIC_KEEP_CNP=1` before starting ComfyUI to keep this copy registered
  anyway. Both packs then register the same ids, and ComfyUI keeps whichever loads last.
- While OmniScale provides the nodes, the **Magnific → Sign in / Sign out** menu shows a note pointing to the API key
  setting instead.

## Credits: what is charged and what protects you

- **Cost estimate on the node.** Eight nodes show an `estimated_cost` line: Generate Image, Upscale Image, Generate
  Video, Upscale Video, Generate Music, Voiceover, Skin Enhancer and Remove Background. It holds a credit figure from
  Magnific's own cost simulator, marked "≈" when it is not exact. It reads "sign in to estimate" while you are signed
  out. Generate Video needs a concrete model, not `auto`, before it can estimate. The Upscale Video estimate is for a
  5 s clip.
- **Count gate (Generate Image).** Each image is a separate paid generation. A `count` of up to 4 runs as it is.
  For 5 to 8, type the same number into `confirm_task_count`, or the run stops before anything is sent: "Not
  submitted: … Nothing was sent."
- **No double billing in one session.** ComfyUI can run a node again with unchanged inputs (for example after its
  cache entry was dropped). When an identical paid request (same tool, same arguments) already succeeded in this
  ComfyUI session, within the last 6 hours, the earlier result is returned instead of paying again. The console says
  "reused, not paid again". Uploads of identical bytes reuse their upload for 30 minutes, so re-running an upscale of
  the same picture counts as the identical request. A result that failed is never reused. Set
  `C2C_MAGNIFIC_DEDUPE=0` to turn the reuse off.
- **Checks before paying.** Most input problems (empty prompt, wrong mask size, too many references, a reference
  video together with keyframes) are caught before anything is uploaded or billed.
- **Version notice.** Magnific's update manifest is checked at most once an hour. If it marks this version as
  retired, the console warns, but the nodes keep running.

## Typical chains

```
Save To (project/folder) ──folder──► Generate Image ──images──► Save Image
                                          │
                                       metadata ──► Metadata (unpack) ──► seed / prompt / model ──► …

Library Reference (@hero) ──chain──► Library Reference (@style) ──library_references──► Generate Image / Video

Stock Search (query, index or random) ──image──► Upscale Image ──► Retouch ◄── mask (Mask Editor)

Load Image ──► Remove Background ──image, mask──► your compositing
Load Video ──► Upscale Video (preview on first, then off) ──video──► Save Video
Creation Picker (browse a project, click a thumbnail) ──image / video / audio──► …
```

Every generation and edit node has an optional `folder` input. Wire a Save To output into it to choose where the
result is stored in your Magnific library. Without it, results go to your **Personal** project.

Every generation and edit node also outputs `creation_identifier` (the result's Magnific id) and `metadata` (JSON).
Metadata (unpack) splits the JSON into sockets.

---

## OmniScale Save To (project/folder) (`MagnificSaveTo`)
**Menu:** 🐺 C2C/🧰 Core/Magnific · **Parameters:** [NODE_REFERENCE](../NODE_REFERENCE.md#magnificsaveto) · free

- **Use it to:** pick the project, and optionally a folder in it, where new results are stored.
- **Inputs that matter:** `project` (your projects; press R after creating one on magnific.com) and `folder`
  (narrowed to the chosen project; `(project root)` = no folder).
- **Outputs:** `folder`. Wire it into any generation or edit node's `folder` input.
- **Errors you may see:** "Your Magnific library has no projects"; "Could not load your projects — press R";
  "'X' is not a folder of 'Y'".

## OmniScale Library Reference (`MagnificLibraryReference`)
**Menu:** 🐺 C2C/🧰 Core/Magnific · **Parameters:** [NODE_REFERENCE](../NODE_REFERENCE.md#magnificlibraryreference) · free

- **Use it to:** attach a Library asset (a character, style, element or location) to Generate Image or Generate
  Video. Chain several nodes through `chain` to attach several assets.
- **Inputs that matter:**
  - `catalog`: `My Library` (your own and shared assets) or `Magnific` (the public catalog).
  - `type`: a filter.
  - `asset`: only assets that are ready are listed. Use the **Refresh library** button or R.
- **Outputs:** `library_references`, which is this asset plus everything chained into it. The generation node adds
  each asset's `@name` to the prompt when the prompt does not already contain it.
- **Errors you may see:**
  - "A generation takes at most 12 references". This counts Library assets and reference images together.
  - "'X' is no longer available in the … catalog".
  - "… does not take … references". Video takes characters, elements and styles, not locations. A specific model
    may accept fewer kinds.

## OmniScale Metadata (unpack) (`MagnificMetadata`)
**Menu:** 🐺 C2C/🧰 Core/Magnific · **Parameters:** [NODE_REFERENCE](../NODE_REFERENCE.md#magnificmetadata) · free,
no network

- **Use it to:** turn the `metadata` output of any Magnific node into plain sockets. For example, use `model` in a
  file name, the `prompt` in a text node, or the `seed` to replay a result.
- **Inputs that matter:** `metadata`, and `index` (which result of a batch; clamped to the last one).
- **Outputs:** `model`, `prompt`, `seed`, `creation_identifier`, `tool`, `url` and `json` (the whole entry).
- **What the JSON can hold:** `creation_identifier`, `tool`, `model`, `mode`, `api`, `prompt`, `negative_prompt`,
  `seed`, `width`, `height`, `aspect_ratio`, `resolution`, `duration`, `fps`, `created_at` and `url`.
  - These are read from Magnific after the run, on a best-effort basis.
  - If that lookup fails, the JSON holds only `creation_identifier`, and the node does not fail. The result is
    already paid for at that point.

## OmniScale Creation Picker (`MagnificCreationPicker`)
**Menu:** 🐺 C2C/🧰 Core/Magnific · **Parameters:** [NODE_REFERENCE](../NODE_REFERENCE.md#magnificcreationpicker) ·
free (downloads your own result)

- **Use it to:** bring an earlier result from your Magnific library back into the graph. Press **Browse** on the
  node, then click a thumbnail.
- **Inputs that matter:** `project`, `folder` and `content_type` (`image`, `video` or `audio`).
- **Outputs:** `image`, `video` and `audio`. Only the one that matches `content_type` is filled. Wiring one of the
  others fails downstream. Also `creation_identifier` and `metadata`.
- **Errors you may see:** "Nothing selected — press Browse on the node and click a thumbnail"; "The selected
  creation has no downloadable media (status: …)".

## OmniScale Generate Image (`MagnificGenerateImage`)
**Menu:** 🐺 C2C/🧰 Core/Magnific · **Parameters:** [NODE_REFERENCE](../NODE_REFERENCE.md#magnificgenerateimage) ·
**paid, one generation per image**

- **Sends:**
  - `prompt`, `model`, `aspect_ratio`, `resolution`, `count` and `seed`;
  - your reference images, each uploaded as a separate image;
  - your Library assets;
  - the `folder`.
- **Inputs that matter:**
  - `model`: `auto` lets Magnific choose. The list comes from your account.
  - `count`: 1–8. Above 4, set `confirm_task_count` to the same number.
  - `seed`: a batch uses seed, seed+1, … Models that honour the seed reproduce a result with the same seed and
    settings.
  - `reference_image` … `reference_image_4`: each can be a batch; one reference per image. Use the extra sockets
    for pictures of different sizes. `reference_type` applies to all of them: `image` = content to follow,
    `style` = look to copy.
- **Outputs:** `images` (one batch), `creation_identifiers` (one id per line) and `metadata`.
- **Errors you may see:** "Prompt is empty"; "Not submitted: … Set confirm_task_count to N"; "… takes at most 12
  references"; "Only N of M generations returned an image URL".

## OmniScale Generate Video (`MagnificGenerateVideo`)
**Menu:** 🐺 C2C/🧰 Core/Magnific · **Parameters:** [NODE_REFERENCE](../NODE_REFERENCE.md#magnificgeneratevideo) · **paid**

- **Sends:**
  - `prompt`, `model`, `duration`, `resolution`, `aspect_ratio`, `sound_effects` and `seed`;
  - the uploaded `start_frame` / `end_frame` / `reference_image` / `reference_video` (the video goes up as MP4);
  - your Library assets and the `folder`.
- **Inputs that matter:**
  - `model`: `auto` or a model from your account. `duration` (5 or 10 s) and `resolution` narrow to the chosen
    model.
  - `sound_effects`: only models with native audio use it.
  - `start_frame` / `end_frame` for image-to-video. `end_frame` works only on models that support it.
  - `reference_video` works only on models that support it, and cannot be combined with a start or end frame.
- **Outputs:** `video`, `creation_identifier` and `metadata`.
- **Errors you may see:** "Prompt is empty"; "A reference video cannot be combined with a start or end frame".
  The second one is refused before anything is uploaded.

## OmniScale Generate Music (`MagnificGenerateMusic`)
**Menu:** 🐺 C2C/🧰 Core/Magnific · **Parameters:** [NODE_REFERENCE](../NODE_REFERENCE.md#magnificgeneratemusic) · **paid**

- **Sends:** `prompt` (genre, mood, instruments, lyrics), `model`, `duration_seconds`, `instrumental` and `folder`.
- **Inputs that matter:**
  - `model` (default Google Lyria).
  - `duration_seconds` (10–300): only Lyria 3 Long and ElevenLabs use it; the other models are fixed at 30 s.
  - `instrumental` on = no vocals.
  - `seed` is **not sent**. Changing it only forces a new run, which is a new paid request.
- **Outputs:** `audio`, `creation_identifier` and `metadata`.
- **Before paying:** the node checks that this ComfyUI can decode the returned audio.

## OmniScale Voiceover (TTS) (`MagnificVoiceover`)
**Menu:** 🐺 C2C/🧰 Core/Magnific · **Parameters:** [NODE_REFERENCE](../NODE_REFERENCE.md#magnificvoiceover) · **paid**

- **Sends:** `text`, the chosen voice, `speed` and `folder`.
- **Inputs that matter:** `voice` (from your account; press R to refresh) and `speed` (0.7–1.2; 1 = normal).
- **Outputs:** `audio`, `creation_identifier` and `metadata`.
- **Errors you may see:** "Text is empty"; "The voice catalog is unavailable or empty"; "Unknown voice … — press R".

## OmniScale Upscale Image (`MagnificUpscaleImage`)
**Menu:** 🐺 C2C/🧰 Core/Magnific · **Parameters:** [NODE_REFERENCE](../NODE_REFERENCE.md#magnificupscaleimage) · **paid**

- **Sends:** the `image` (uploaded; at most 25 MB) and the settings below.
- **Inputs that matter:**
  - `scale`: 2x, 4x, 8x or 16x.
  - `precision`: `creative` invents new detail and uses the −10..10 sliders `creativity`, `hdr`, `resemblance` and
    `fractality`. `precision` stays faithful to the input and uses the 0..100 sliders `sharpness`, `grain` and
    `ultra_detail`.
  - `presets`: subtle, vivid or wild.
  - `engine` and `mode`: `Automatic` lets Magnific choose.
  - `prompt` (optional) guides the new detail.
- **Outputs:** `image`, `creation_identifier` and `metadata`.
- **Errors you may see:** "Image is N MB — Magnific accepts at most 25MB per upload"; "Upscale finished but
  returned no image URL".

## OmniScale Upscale Video (`MagnificUpscaleVideo`)
**Menu:** 🐺 C2C/🧰 Core/Magnific · **Parameters:** [NODE_REFERENCE](../NODE_REFERENCE.md#magnificupscalevideo) · **paid**

- **Sends:** the `video` (uploaded as MP4) and the settings of the chosen `mode`.
- **Inputs that matter:**
  - `mode`:
    - `magnific`: a creative upscale to `target_resolution` (the output width). It uses `preset`; the `custom`
      preset uses `creativity`, `sharpen`, `smart_grain`, `flavor`, `premium_quality` and `turbo`.
    - `magnific_precision`: a faithful upscale to `target_resolution`, using `strength`.
    - `topaz`: scales by `upscale_factor` (2 or 4).
  - `fps_boost` asks the server to raise the frame rate too. It is not sent in `topaz` mode.
  - `preview` on = a 12-frame preview run instead of the full clip.
- **Outputs:** `video`, `creation_identifier` and `metadata`.

## OmniScale Skin Enhancer (`MagnificSkinEnhancer`)
**Menu:** 🐺 C2C/🧰 Core/Magnific · **Parameters:** [NODE_REFERENCE](../NODE_REFERENCE.md#magnificskinenhancer) · **paid**

- **Sends:** the uploaded portrait, `version`, `sharpen`, `smart_grain`, `skin_detail`, `optimized_for` and `folder`.
- **Inputs that matter:**
  - `version`:
    - `faithful` keeps the face as it is; `skin_detail` applies.
    - `creative` is freer.
    - `flexible` aims at whatever `optimized_for` says (for example `enhance_skin`, `improve_lighting`,
      `no_make_up`).
- **Outputs:** `image`, `creation_identifier` and `metadata`.

## OmniScale Remove Background (`MagnificRemoveBackground`)
**Menu:** 🐺 C2C/🧰 Core/Magnific · **Parameters:** [NODE_REFERENCE](../NODE_REFERENCE.md#magnificremovebackground) ·
**paid**

- **Sends:** the uploaded `image` and `folder`.
- **Outputs:**
  - `image`: the cut-out, as RGB.
  - `mask`: the subject's alpha, 1 = kept and 0 = removed.
  - `creation_identifier` and `metadata`.
- The `mask` output can feed Retouch, or any other mask input.

## OmniScale Retouch (inpaint) (`MagnificRetouch`)
**Menu:** 🐺 C2C/🧰 Core/Magnific · **Parameters:** [NODE_REFERENCE](../NODE_REFERENCE.md#magnificretouch) · **paid**
(no on-node cost estimate)

- **Sends:** the first frame of `image` and the `mask` (as a black-and-white PNG), `mode`, `prompt` and `folder`.
- **Inputs that matter:**
  - `mask`: white = the area to change. It must have the same pixel size as the image. It is turned into pure
    black and white, so anything below 50% counts as not selected.
  - `mode`: `replace` paints what `prompt` says into the mask. `erase` removes what is there.
- **Size limit:** images whose longest side is over 2048 px (`MAGNIFIC_RETOUCH_MAX_EDGE`) are scaled down to fit, on
  a multiple of 8. The console says so, and the result comes back at the scaled size.
- **Outputs:** `image` (at the size it was sent), `creation_identifier` and `metadata`.
- **Errors you may see** (all raised before anything is uploaded):
  - "Mask is WxH but the image is WxH";
  - "The mask selects nothing";
  - "Replace mode needs a prompt …";
  - "The masked area is too thin to survive scaling".

## OmniScale Stock Search (`MagnificStockSearch`)
**Menu:** 🐺 C2C/🧰 Core/Magnific · **Parameters:** [NODE_REFERENCE](../NODE_REFERENCE.md#magnificstocksearch) ·
searching is free; the **download of the chosen item is paid**

- **Use it to:** fetch one stock image by query, without clicking. Good for automated graphs.
- **Inputs that matter:**
  - `query`.
  - The filters `content_type`, `license`, `orientation` and `ai_generated`. Vectors, PSDs and designs come back
    as JPG renders.
  - `selection`:
    - `index` takes `result_index`.
    - `random` picks between `range_from` and `range_to`, driven by `seed`. A fixed seed repeats the pick.
  - `folder`: also save the item to your library.
- **Outputs:**
  - `image`.
  - `info`: every usable result found, with the chosen one marked `>`. With `random`, the range and seed. A note
    when there were fewer results than the index (the last one is used). Where it was saved.
- **Errors you may see:** "Search query is empty"; "No usable stock image results for '…'".

## OmniScale Stock Picker (`MagnificStockPicker`)
**Menu:** 🐺 C2C/🧰 Core/Magnific · **Parameters:** [NODE_REFERENCE](../NODE_REFERENCE.md#magnificstockpicker) ·
searching is free; the **download of the picked item is paid**

- **Use it to:** search stock visually. Press **Search** on the node, then click a thumbnail. The click stores the
  item in the hidden `selected` field, and the run downloads that item.
- **Inputs that matter:** the same `query` and filters as Stock Search, plus `folder`.
- **Outputs:** `image`, and `info` (the stock id and type, and where it was saved).
- **Errors you may see:** "Nothing selected — press Search on the node and click a thumbnail".
