# Load / Save Video (C2C): colour management, formats, Load -> Save link - design (A7, L9.53-L9.58)

Owner request + six answer rounds, verbatim: `docs/evidence/L9.52/owner_answers_2026-10-25_video.md` (workspace).
Research and sources: `docs/evidence/L9.53/research_2026-10-25.md`. Order (owner): visible bugs (L9.52, done) ->
Vault front end (L9.46) -> this. Nothing here is built yet.

## Constraints

- **No new nodes.** Everything lives in Load Video (C2C), Load Video Path (C2C) and Save Video (C2C). The
  model-boundary conversion happens INSIDE Load Video (owner).
- **Saved workflows never change behaviour.** New widgets are appended. A saved workflow keeps what it did: a
  missing new widget means "off / as before" (the `widget_migration.js` tail-pad). Defaults for NEW nodes may differ.
- Both renderers, inside subgraphs, Linux + Windows. Frontend 1.53.10 is the floor (owner production).
- **No silent guesses** (owner): an unknown colour space is a warning plus a choice, never "assume sRGB".
- Filenames alone are never enough (owner). OCIO file rules are a suggestion that needs confirming.
- PyOpenColorIO 2.5.2 is already a dependency. Its built-in configs carry ACES 2.0 studio-config v4.0.0 (55 spaces:
  ARRI LogC3 / LogC4, S-Log3 S-Gamut3(.Cine), Log3G10 RWG, V-Log V-Gamut, BMDFilm WG Gen5, Canon CinemaGamut,
  DaVinci WG, sRGB / Rec.709 / P3 / Rec.2020 encodings, ACEScg ...). Measured 2026-10-25.

## 1. Settings (C2C › Video › Colour Management)

| Setting | Values | Default |
|---|---|---|
| Mode | Automatic OCIO / Manual / Preserve Source | Automatic OCIO |
| OCIO config | `$OCIO` when set, else this path, else built-in `ocio://studio-config-latest` | (empty path) |
| Working space | any space of the active config | **open question Q1** |
| Default output space (Save with no Load link) | any space, or "same as working" | same as working |
| Display / View / Look (review copy, preview) | from the config | config default display + view |

The active config (and its source) is shown in both nodes' info line. A config that fails to load says so and falls
back to the built-in one, with a warning. It never silently uses a different config.

## 2. Load Video: what the file is

New appended widgets: `colorspace_in` = auto / any space; `working_space` = inherit / any space.

Detection. Every source reports a confidence, and the first confident answer wins:

1. **Camera vendor metadata** (high). ARRI `com.arri.camera.*` gamma + EI -> "ARRI LogC3 (EI800)" / "ARRI LogC4";
   Sony / RED / Canon / Panasonic / BMD keys likewise. A real camera sample is needed to verify which keys PyAV 17
   exposes (research note 4). Until then this source is marked UNVERIFIED and not shipped.
2. **EXR / DPX header** (high). An explicit colour-space attribute (`oiio:ColorSpace`, a Nuke-written name),
   chromaticities (AP0 / AP1 / Rec.709 / P3 / Rec.2020) plus a linear transfer for EXR, the DPX transfer field.
3. **Sidecar / Nuke script** (high). `<file>.c2c.json` or `<file>.colorspace` next to the file, or the Write node
   of a `.nk` the user points at.
4. **MOV / MP4 colour tags** (medium). They can lie (an ARRI LogC ProRes is tagged BT.709), so they are confirmed by
   the pixel check (5). They decide alone only when the check agrees.
5. **Pixel sanity check** (warn only). It reads the first frames' black floor, white peak and over-range: log-like
   (lifted blacks, no clipping), display-like, or scene-linear (values > 1, negatives). A disagreement with 1-4
   adds a warning naming both, and nothing changes.
6. **OCIO file rules** (suggestion). `config.getColorSpaceFromFilepath()`: shown as "suggested by the config's file
   rules - confirm", never used without a click.
7. Nothing confident -> **Unknown**: the node runs in Preserve Source for that clip and shows "Pick the colour space
   this file was rendered in".

The transform happens on decode, source -> working (OCIO CPU processor, float32, chunked like decode; GPU later).
Preserve Source = no transform. `video_info["c2c_colour"] = {source, detected_from, confidence, working,
config, transform}`.

## 3. Save Video: back to the file's space

Appended: `colorspace_out` (gets "auto (from Load)"), `transform` = automatic / explicit / bypass.
`colorspace_in` gets "auto (from Load)".

- **automatic**: in = the frames' actual working space (from the link, section 4), out = the Load's source space
  (reversed, owner). No Load link -> the default output space.
- **explicit**: the two widgets as set. **bypass**: no transform, tags still written.
- Same space in and out -> no transform at all (no double transform). A round trip Load -> Save with nothing between
  is a test: pixels identical within the codec's own error.
- Colour written INTO the file (owner): MOV / MP4 colour tags from the output space (never a tag the pixels do not
  have), EXR chromaticities + `oiio:ColorSpace`, DPX transfer.

## 4. The Load -> Save link (L9.55)

Already built for handles (L9.52): a wired `video_info` wins, otherwise Save traces its `images` / `latent` input
back through each node's first image input. Extended:

- The record carries colour, name, timecode, fps, pixel aspect, reel name and source frame numbers.
- **Ambiguous** (two Loads plausible, a node with several image inputs on the path): a "Follow Load Video #12 / #15"
  picker and a warning until chosen. The choice is stored on the Save node by node id and is stable until the graph
  changes (owner).
- **Live in the UI**: the frontend does the same trace on the graph (no server call) and shows the resolved values in
  grey next to "auto" widgets.

## 5. Naming (owner)

- The name is Load's `name_prefix` + source stem + Load's `name_suffix`. Both are appended Load widgets, empty by
  default: no suffix unless the user adds one. Save's own prefix / suffix win when set.
- The version folder is the default, and the file name carries the SAME version as the folder:
  `v004/shot010_comp_v004.mov` (the source's own `_v###` token is replaced). On collision, version up. The source
  path is never written.
- Sequences keep source frame numbers (1001...), shifted by the trimmed handles and In / Out.

## 6. Format UI (L9.54)

`format` keeps its saved values. New UI: **Format** (Video / Image Sequence / Custom) -> **Subformat** (H.264,
H.265, ProRes Proxy / LT / 422 / HQ / 4444 / 4444 XQ, DNxHR LB / SQ / HQ / HQX / 444, FFV1; OpenEXR, DPX, PNG, TIFF,
JPEG) -> that subformat's options only (EXR compression ZIP / PIZ / ZIPS / DWAA / DWAB / NONE + half / float + RGB /
RGBA; H.264/5 CRF + preset + pix_fmt + profile / bitrate / GOP; DPX 10 / 12 / 16 + packing; PNG 8 / 16 +
compression) + a collapsible Advanced (timecode, range, colour metadata, alpha). Custom = today's five-control
cascade. The last valid settings per subformat are remembered (node properties). The matrix from the server says
what this machine can write. Implementation: the Format / Subformat pair maps onto the existing five controls plus
appended option widgets, so the Python encode path stays one.

## 7. Review copy + burn-in (L9.57)

Off by default. When on, `<master>_review.mp4` is written next to the master: H.264, the config's Display / View
applied, half resolution. The burn-in is optional (name + version, frame + timecode, colour spaces, a custom template
`{shot} {version} {frame} {tc} {date} {artist}` per corner). It is written from the same frames in one pass, so it
costs no second decode.

## 8. Extra handles in model steps (L9.56)

Appended to Load: `extra_handles` = k (an integer), with +/- buttons stepping one model step (17 / 4 / 8 from the
grid) and a readout "k x step = frames -> total". The total stays on the grid by construction. Save trims the
recorded total (already exact, L9.52).

## 9. OCIO version and configs (L9.58) - owner 2026-10-26: "we need latest OCIO"

- **Require OpenColorIO >= 2.6.0** (released 2026-10-01; VFX Reference Platform CY2027 base). Wheels exist for
  CPython 3.10-3.14 on Windows / Linux / macOS. It goes in CNP's `requirements.txt`. The owner's production env
  updates through ComfyUI-Manager / pip. The testbed never installs into the production env (D0.08), so the colour
  module's tests run in a scratch venv with 2.6.0 until the owner's env has it. A 2.5 install keeps working, with a
  one-line "update OpenColorIO" note.
- What 2.6 brings that we use: the 2.6.0 built-in ACES configs, **ASWF colour-interop IDs** (standard identifiers
  for a colour space - written into files and read back, the most reliable detection a file can carry), display /
  view aliases, Apple Log 2, SMPTE ST 2136-1 CLF LUT read / write.
- Configs: the built-in ACES configs (no download), `$OCIO` / the setting path (the owner's own Nuke config).
  nuke-default and the 1.4 GB legacy ACES 1.x LUT config are not needed (owner).

## Failure modes

| Failure | Behaviour |
|---|---|
| OCIO config missing / broken | built-in config + warning naming the path |
| A space name not in the active config (a workflow from another studio) | the node says which name is missing; no transform until one is chosen |
| Detection sources disagree | the higher-confidence one is used, the warning names both |
| Unknown space | Preserve Source + "pick the space" |
| PyOpenColorIO missing | sRGB <-> linear only (today's fallback), and the node says so |

## Budgets

OCIO CPU transform: one processor per (src, dst) pair, cached. The cost is to be measured per megapixel before
shipping, against the decode's own cost. If it exceeds 25 % of decode time, the GPU path (OCIO GPU shader or a 3D
LUT on torch) comes first. Detection runs once per file: probe metadata plus at most 8 frames for the pixel check.

## Kill switches

`C2C_VIDEO_CM=off`: no detection, no transform, today's behaviour. `C2C_VIDEO_LINK=off`: no trace (wire only).
`C2C_SAVE_V2_MATRIX=off` keeps working.

## Owner answers (2026-10-26, verbatim in docs/evidence/L9.52/owner_answers_2026-10-26_cm.md)

- **Working space:** sRGB display by default, but a source that is already Rec.709 or sRGB is left as it is (no
  transform). Any other source converts to the working default. -> Automatic mode: Rec.709 / sRGB sources are
  preserved; log / wide-gamut / linear sources convert to sRGB display; Save reverses exactly what Load did.
- **Detection:** "use Nuke methods and OCIO methods, search the internet" -> research Nuke's Read behaviour (metadata
  keys it reads, how it picks the default from OCIO file rules) and OCIO's own (file rules, colour-interop IDs,
  `getColorSpaceFromFilepath`, heuristics) before building section 2. Vendor keys come from documented sources, not
  from guesses.
- **Configs / version:** no nuke-default, no legacy LUT config; the latest OCIO (2.6.0) - section 9.
