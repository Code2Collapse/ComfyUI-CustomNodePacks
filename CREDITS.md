# Credits

Parts of this pack are ports of other people's open-source work. They are named
here, with the licence each upstream declares, because someone who fixes a bug
in a node should be able to find out whose idea it was.

`NOTICE` carries the formal licence text and the redistribution terms. This file
is the human one: who wrote the thing, and where to go if you want the original.

Licences below were read from the clones in `third_party/` on 2026-09-19, not
assumed. Where a project declares a licence only in its README or
`pyproject.toml` and ships no `LICENSE` file, that is said so.

---

## Node families ported into this pack

### ComfyUI_LayerStyle — chflame163
<https://github.com/chflame163/ComfyUI_LayerStyle> · **MIT** (declared in
`pyproject.toml` and README)

The source for two families:

- **`nodes/layer_effects/`** — Drop Shadow, Inner Shadow, Outer Glow, Inner
  Glow, Stroke, Color Overlay, Gradient Overlay, Gradient Map. One node per
  effect, carrying the union of upstream's V1/V2/V3 variants.
- **`nodes/mask_toolkit/`** — Mask From Color, Mask Gradient, Mask Grain, Mask
  Motion Blur, Edge Spread, plus the channel selector and Blend-If soft ends
  added to `LuminanceKeyerMEC`.

Reimplemented in torch rather than copied verbatim, for a concrete reason:
upstream's `blendmodes.py` imports the `blend_modes` pip package, which is not
installed in this ComfyUI, so a straight copy would have failed at import. The
blend table is checked against a numpy transcription of upstream's own formulas
in `tests/test_layer_effects.py`.

### ComfyUI-NKD-Basic-Tools — Nekodificador
<https://github.com/Nekodificador/ComfyUI-NKD-Basic-Tools> · **MIT** (declared in
`pyproject.toml`)

- **`nodes/frequency_grain/`** — Frequency Separate, Frequency Combine, Film Grain.
  Ported from `nkd_frequency.py`, `nkd_film_grain.py`, and selected helpers in
  `helpers.py` (colour transforms, mask resize, film-grain synthesis). Reimplemented
  as CNP V1 nodes with report outputs; upstream live-preview widget code omitted.

### Virtuoso Pack — Chris Freilich
<https://github.com/chrisfreilich/virtuoso-nodes>

The 30-mode blend table in `nodes/layer_effects/_blend.py`. Credited here
because ComfyUI_LayerStyle's own `py/imagefunc.py:2506` credits it — the debt
is owed one step further upstream than our direct source.

### comfyui-mcp — artokun
<https://github.com/artokun/comfyui-mcp>

The "compact tool mode" architecture behind `nodes/ai_workflow_builder.py`: do
not hand a small local model a 2000-node schema, score the live registry down to
a candidate set first.

---

## Not ported — reference only

Read while building, no code taken:

- **ComfyUI-NKD-VFX-Tools** — read as the standard for node UI quality.
  (Frequency Separate/Combine and Film Grain from NKD Basic Tools are ported in
  `nodes/frequency_grain/`.)
- **ComfyUI-Pixaroma** by pixaroma (<https://github.com/pixaroma/ComfyUI-Pixaroma>,
  MIT) — read for its front-end approach.

---

## Ported code that is NOT open source

- **magnific-comfyui 0.7.0** by Magnific / Freepik
  (<https://www.magnific.com/plugins>) — ported whole into `nodes/magnific/`
  and `js/magnific/`.

  Its `pyproject.toml` declares `license = { text = "UNLICENSED" }`. That is an
  explicit reservation of rights, not a missing licence, so it is recorded
  separately from the permissive ports above: **this pack has no right to
  redistribute it.** It is here because `third_party/` is gitignored and so is
  absent on the Linux box; if this repository is ever published, this
  directory comes out first.

  Four changes were made on the way in, each commented at the site that
  changed and pinned by `tests/test_magnific_port.py`: the version literal
  (upstream read a sibling `pyproject.toml` that does not exist here, and its
  failure path returned `0.0.0`), the update check no longer raises, the JS
  import depth gained a `../`, and the route mount is guarded against a
  missing `PromptServer.instance`. To re-sync, copy the upstream files over
  and re-apply those four — the tests name them.

  The fifteen node IDs are unchanged, so the vendor's own pack installed
  alongside this one registers all fifteen twice.

---

## Upstream projects whose licence could not be verified

Both of these declare `license = {file = "LICENSE"}` in `pyproject.toml` while
shipping no such file in the clone. Their licence is therefore **unknown**, not
permissive by default, and nothing from either has been ported:

- **ComfyUI-KJNodes** — kijai
- **WhatDreamsCost-ComfyUI**

`NOTICE` currently describes KJNodes as Apache-2.0. Nothing on disk supports
that, and the claim predates this file.

---

## Bundled third-party code

See `third_party/` and `THIRD_PARTY_LICENSES/` for vendored projects
(ProPainter, SAM3, the video stabiliser, Prompt Relay) and their own licences.

---

If your work is here and the attribution is wrong, thin, or you would rather it
were removed, open an issue — it will be fixed.
