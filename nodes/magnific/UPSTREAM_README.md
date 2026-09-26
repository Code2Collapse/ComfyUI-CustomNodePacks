# Magnific for ComfyUI

Magnific nodes inside ComfyUI: image and video generation, the Magnific upscaler, retouch, skin enhancer, background removal, music, voiceover and stock search. Everything runs on your Magnific account — the nodes send the job to Magnific and return the result into your graph.

Results are also saved into the Magnific project and folder you pick with a **Magnific Save To** node. If you don't connect one, they go to your Personal project.

## What you get

| Node | What it does |
| --- | --- |
| Magnific Save To (project/folder) | Where results are saved, and the sign-in button |
| Magnific Generate Image | Text-to-image and image-to-image |
| Magnific Upscale Image | The Magnific upscaler |
| Magnific Retouch (inpaint) | Edit a region with a prompt |
| Magnific Skin Enhancer | Skin detail on portraits |
| Magnific Remove Background | Cut out the subject |
| Magnific Generate Video | Text-to-video and image-to-video |
| Magnific Upscale Video | Upscale an existing video |
| Magnific Generate Music | Music from a prompt |
| Magnific Voiceover (TTS) | Text to speech |
| Magnific Stock Search / Stock Picker | Find and pick stock media |
| Magnific Creation Picker | Reuse something you already created in Magnific |
| Magnific Library Reference | Attach a Library character, style, element or location to a generation |
| Magnific Metadata (unpack) | Read the model, prompt, seed and creation link out of any result |

## Before you start

- ComfyUI (desktop or a manual install) with Python 3.9 or newer.
- A Magnific account.
- Nothing else to install: the nodes use only what ComfyUI already ships.
- Only for the audio nodes (Generate Music, Voiceover): if your ComfyUI has no audio decoder, the node tells you to run `pip install av` in ComfyUI's Python environment and restart.

## Install

1. Download the latest `magnific-comfyui-<version>.zip` from https://www.magnific.com/plugins.
2. Extract it into your ComfyUI `custom_nodes/` folder. The zip already contains one folder, so you get:
   ```
   ComfyUI/custom_nodes/comfyui-magnific/
   ```
3. Restart ComfyUI.
4. Type "Magnific" in the node search. The nodes are there.

## Sign in

Your password never goes into ComfyUI — you approve the sign-in in your browser.

1. Add a **Magnific Save To (project/folder)** node. If you are not signed in, it shows a **Sign in to Magnific** button. A message at startup offers the same button.
2. Click it. Your browser opens the Magnific page. Approve the sign-in there.
3. Go back to ComfyUI. The node loads your projects and folders.

To sign out, delete the file `~/.magnific/comfyui_auth.json`.

## Use several reference images

**Magnific Generate Image** takes up to four reference sockets: `reference_image`, `reference_image_2`, `reference_image_3` and `reference_image_4`. Connect a picture to each one you need. A batch (for example from a **Batch Images** node) counts one reference per image, so a batch of three on one socket is three references.

One `reference_type` applies to every picture: `image` keeps a subject, product or person consistent; `style` borrows the look. A generation takes at most 12 references, Library references included; the node stops before uploading anything if the graph exceeds that.

Batching resizes every picture to the first one's size, so for pictures of different sizes use separate sockets.

## Use your Library (characters, styles, elements, locations)

The assets you trained or saved in your Magnific Library, and the ones in Magnific's public catalog, can guide a generation, like the `@name` mentions in the web app.

1. Add a **Magnific Library Reference** node. Choose a **catalog** (`My Library`: your own and shared assets; `Magnific`: the public catalog), optionally narrow the **type** (Characters, Styles, Elements, Locations), then pick an **asset**. The list shows `@name (Type · Origin)` grouped by type; only ready assets appear. Press **Refresh library** on the node (or R) after you create a new one.
2. Connect its `library_references` output to the `library_references` input of **Magnific Generate Image** or **Magnific Generate Video**.
3. To attach several assets, connect one Library Reference node into the `chain` input of the next, and connect the last one to the generation node. A generation takes at most 12 references, reference image included.

Write `@name` in the prompt where the asset should appear. If you forget, the node adds the token to the end of the prompt.

Not every model takes every type. Image models list the types they accept; video takes characters, elements and styles, never locations. The node stops with a message naming the model and the type before anything is sent, so pick another model or set the model to `auto` and let Magnific choose.

## Know which model made each result

Every node that creates or edits something has a `metadata` output next to `creation_identifier`: a JSON text with the model, prompt, seed, size, duration, creation date and the link to the creation on magnific.com (a batch gives one entry per image, in batch order). Connect it to **Magnific Metadata (unpack)** to get `model`, `prompt`, `seed`, `creation_identifier`, `tool` and `url` as plain sockets — for example `model` into the `filename_prefix` of Save Image, so each file says which model produced it. The `json` output re-emits the selected entry for a metadata-writing node of your choice.

ComfyUI re-encodes what a node returns, so the file itself carries no Magnific metadata; the node values are the record. A saved workflow keeps them.

## Reproduce a result with the seed

**Magnific Generate Image** and **Magnific Generate Video** send their `seed` to Magnific. Run the same prompt, model and settings with the same seed and you get the same or a near-identical result on models that honor a seed (Seedance among them); change any setting, resolution included, and the result changes with it. Leave the widget on `randomize` for variety, or set it to `fixed` and type the seed from a previous result's `metadata` to replay it. An image batch (`count` above 1) uses `seed`, `seed+1`, … per image, so every image is still reproducible on its own.

The `seed` on **Magnific Generate Music** is not sent: music generation takes no seed, and changing it only forces a new run.

## Update

The plugin tells you when a new version is available and links to the download page. It does not update itself.

To update: delete the folder `ComfyUI/custom_nodes/comfyui-magnific/`, then install the new zip as above and restart ComfyUI.

If your version is too old, the nodes stop running and point you to the download page. Install the new version to continue.

## Help

- Downloads and plugin help: https://www.magnific.com/plugins
- If a node fails, its error message says what to do. Keep it at hand if you contact support.
