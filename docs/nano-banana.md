# Nano Banana · Gemini Image

## Nano Banana · Gemini Image (C2C) (`NanoBananaC2C`)
**Menu:** 🐺 C2C/🧰 Core/AI Image · **Parameters:** [NODE_REFERENCE](../NODE_REFERENCE.md#nanobananac2c)
**Paid online service.** The node needs a Google AI Studio (Gemini API) key and an internet connection. Every
generation call goes to Google and is billed to that key's account. The pack shows no price. Its tests use a stand-in
for the network and never send a live request (commit 253e39c).

### What it is for
It generates or edits images with Google's Gemini image models ("Nano Banana") and returns them as an IMAGE batch.
Nothing runs on your GPU.

The `model` list:
- `gemini-3-pro-image-preview` (Nano Banana Pro; the only one that takes `resolution`);
- `gemini-2.5-flash-image`;
- `gemini-2.5-flash-image-preview`;
- `gemini-2.0-flash-preview-image-generation`.

### Minimal wiring
```
                         ┌─ image_1 … image_4 (optional references)
Nano Banana · Gemini ────┤
  prompt, model, mode    └─► images ──► Save Image / Preview Image
                             text_response, info ──► Preview Any
```
For `text_to_image` nothing needs to be wired: type a prompt and run.

### The API key: where it is read from
The node looks for the key in this order. It never prints the key.
1. The `api_key` field on the node. This is kept for old workflows only. When you type a key there, the front end
   moves it into the C2C key store and clears the field, so it is not saved in the workflow or the image metadata. If
   the key store cannot take it, the key stays in the field and you are told it will be saved with the workflow.
2. The C2C key store, entry `GEMINI_API_KEY` (Settings > C2C AI). It lives in the operating system's credential
   store (Windows Credential Manager, macOS Keychain, Linux Secret Service) under the service name `c2c-comfy`. This
   needs the `keyring` Python package.
3. The environment variable `GEMINI_API_KEY`.
4. A Gemini key typed into the C2C Diagnostics sidebar (the pack's encrypted secrets file).
5. The environment variable `GOOGLE_API_KEY`.

You create the key at aistudio.google.com. With no key anywhere, the node stops with "No Gemini API key" and lists
these places.

### What is sent, and what it costs
Each run sends a request to `generativelanguage.googleapis.com` with your key in a request header. The request holds:
- the prompt text, after `prompt_style` and `mode` have added their wording (see Key controls);
- every reference image, as a PNG, scaled down to 2048 px on the longest side when larger;
- `temperature`, the response type (`response_modalities`), and `aspect_ratio` / `resolution` when not `auto`;
- `system_instruction`, when you fill it in.

What counts as a billed call:
- `batch_count` (1–4) makes that many calls, one after another. Each one is billed.
- `enhance_prompt` adds one more call before them: your prompt goes to the text model `gemini-2.5-flash` to be
  rewritten.
- `seed` is not sent to Google. It only affects ComfyUI's cache. It is also part of the reuse check below, so a new
  seed means a new paid request.

**Paid requests are not repeated within a session.** ComfyUI can run a node again with unchanged inputs, for example
after its cache entry was dropped. When an identical request (same model, prompt, settings, reference images, seed,
batch count and key) already succeeded in this ComfyUI session, within the last 6 hours, the node returns the earlier
images instead of paying again. `info` then shows `"reused": true`. The node remembers up to 8 results. Set the
environment variable `C2C_NANO_BANANA_DEDUPE=0` to turn this off. A run that stopped part-way is never reused.

### Key controls
- `mode`:
  - `text_to_image`: no reference needed.
  - `edit_image`: needs at least one reference image. It edits that image and asks the model to keep the rest.
  - `compose_images`: needs at least one reference image. It combines the references into one picture.
  - `style_transfer`: needs at least two images. The first is the content and the last is the style.
- `prompt_style` adds fixed wording around your prompt:
  - `raw` = your text as it is;
  - `cinematic` = a film-camera description;
  - `structured_json` = a JSON brief (a prompt that starts with `{` is passed through as your own brief);
  - `character_consistency` = keep the reference person's identity;
  - `product_shot` = a studio product description.
- `image_1` … `image_4`: reference images. Batches are flattened in order, up to 14 images in total.
- `resolution`: 1K / 2K / 4K, sent only for the `gemini-3` model.
- `response_modalities`: `IMAGE+TEXT` also returns the model's text; `IMAGE` returns the picture only.

### Reading the outputs
- `images`: one image per successful call, stacked as a batch. Images of another size are resized to the first one's
  size.
- `text_response`: the model's text reply. It is empty with `response_modalities` = `IMAGE`.
- `info` (JSON) has these fields:
  - `model`, `mode`, `prompt_style`, `aspect_ratio`, `resolution`;
  - `prompt_enhanced`: true when the enhancer rewrote the prompt;
  - `reference_images`, `images_out` and `size`;
  - `latency_s` and `tokens` (as reported by Google);
  - `partial`: present when a later call failed after some images were made; the images made are kept and returned;
  - `reused`: present when the result came from the session reuse.

### Common mistakes
- **Wrong or expired key.** "Google rejected the API key" (HTTP 401/403).
- **Model not available to your key or region.** HTTP 404. The message suggests `gemini-2.5-flash-image`, or
  enabling the preview model in Google AI Studio.
- **Quota.** HTTP 429: "Rate/quota limit hit". Wait, or raise the quota in Google AI Studio.
- **Unsupported setting.** HTTP 400 is often an aspect ratio or resolution that the chosen model does not take, or a
  prompt that is too long.
- **Missing references for the mode.** `edit_image` and `compose_images` need at least one image. `style_transfer`
  needs two.
- **The model refused or sent no picture.** "Gemini blocked this prompt (…)", "Gemini refused to render this image
  (SAFETY …)" or "Gemini answered with text only". Rephrase the prompt or change the references.
- **Enhancer failure.** When the `enhance_prompt` call fails, the node logs a warning and uses your original prompt.
  It does not stop.
