# WanDirector — timeline video direction inside a node

> **WanDirectorC2C** is a full timeline editor embedded in a ComfyUI node: an
> image/text track, an audio track, a Control-Video track, and four automation
> lanes (LoRA / camera / seed / pose), plus a live preview player. It compiles
> everything into a single `tracks_program` the backend hands to the Wan
> pipeline. This is the "direct a multi-shot video on a familiar timeline"
> workflow, with the professional editing tools you'd expect from an NLE.

The node ships in both **ComfyUI-CustomNodePacks** and
**ComfyUI-WanNodeExperiments** (identical; guarded against double
registration). Companion node: **WanDirectorExtraArgs** bundles the advanced
quality stack so the main node stays uncluttered.

---

## The timeline at a glance

```
┌─ toolbar ── + Text  + Image  + Video  + Audio   Split  Delete  ▶ − + Fit ──┐
│  ruler  0s ......... 1s ......... 2s ......... 3s .........   [I===O]       │
│  ● image/text lane   [ clip ][ clip ]                                       │
│  ● audio lane        [ wave~~~~~~~ ]                                         │
│  ● video lane        [🎞 prores.mov ]                                        │
│  ● LoRA  ● Cam  ● Seed  ● Pose   (automation lanes)                          │
│  ─ live preview player ─  ⏮ ◀ǀ ▶ ǀ▶ ⏭  🔁 1× [ ] ⌫   0:00                    │
│  ─ properties panel ─  (edit the selected clip / segment)                   │
└─────────────────────────────────────────────────────────────────────────────┘
```

- **Add clips** — the toolbar `+` buttons, or click the "+ add …" hint in an
  empty lane, or drag-and-drop a file onto a lane.
- **Move / trim** — drag a clip's middle to move it, drag an edge to trim.
- **Select** — click a clip to edit it in the properties panel below.

---

## Editing power tools (LTX-Director class)

| Tool | How | What it does |
|---|---|---|
| **Snapping** | drag a clip | Snaps to the playhead, in/out marks, every other clip's edges, and the timeline bounds (8-px tolerance). **Hold Shift** to bypass |
| **Work region (I/O)** | `I` / `O` keys | Set in / out points at the playhead (amber band on the ruler); `X` clears. Playback **loops inside the region** and starts from the in-point |
| **Toggleable tracks** | click the dot in a lane's left gutter | Mutes that lane — it dims, audio preview goes silent, and the **backend skips the muted track entirely** on render (with a note in the info output) |
| **Camera presets** | right-click a **Cam** segment | One-click Static / Pan ← → / Dolly In-Out / Zoom In-Out / Orbit ↺ ↻ |
| **Retake** | right-click an image/video clip | "🎲 Retake span (new seed)" — lays a fresh fixed seed exactly over that clip's frames (LTX-2 Retake semantics), so you re-roll just that shot |
| **Split at playhead** | `S`, or Split button, or right-click → Split | Cuts the selected clip at the playhead |

### Hotkeys (canvas focused)

| Key | Action | Key | Action |
|---|---|---|---|
| `Space` | Play / pause | `S` | Split at playhead |
| `←` / `→` | Step 1 frame | `Delete` | Delete selection |
| `Home` / `End` | Jump to start / end | `D` | Duplicate clip after itself |
| `I` / `O` / `X` | In / out / clear region | `Ctrl+C` / `Ctrl+V` | Copy / paste at playhead |
| `+` / `-` | Zoom in / out | `Ctrl+Z` | Undo (native ComfyUI history) |

---

## Real-time playback of any format (proxy workflow)

Browsers only decode web codecs (H.264 / VP9 / AV1). The Director plays
**ProRes, EXR sequences, DPX, MXF, DNxHD** and more in real time using the
industry proxy pattern (Resolve / Premiere):

1. Add a Control-Video clip in one of those formats — it appears as a **🎞
   tile**.
2. The first time you play it, the server transcodes a **frame-accurate H.264
   proxy** once (fps untouched, so frame *N* of the proxy is frame *N* of the
   source; scaled to ≤720p, never upscaled). You see "Building edit proxy… %"
   during that pass.
3. After that it plays instantly from cache. **Playback is trim-window
   accurate** and auto-advances through mixed image+video sequences.

The proxy is only ever used for *preview*. **Your render always reads the
original source** — source media is never modified. Proxies cache under
ComfyUI's temp with a disk ceiling you can tune (`WNE_PROXY_CACHE_MB`, default
2048 MB, oldest-first eviction). The same proxy path powers the **Video
Comparer** node.

---

## The six tracks → `tracks_program`

The backend (schema v2) parses and validates all tracks and emits one compact
`tracks_program` JSON alongside the standard outputs:

| Track | Segment fields | Purpose |
|---|---|---|
| **image / text** | prompt, imageFile, guideStrength | Per-segment prompt + optional reference image |
| **audio** | audioFile, trimStart | Audio bed (lip-sync / music) |
| **video** (control) | videoFile, trimStart, length | Control-video conditioning |
| **LoRA** | name, strength | Time-ranged LoRA application |
| **camera** | type (static/pan/zoom/orbit/dolly), params | Camera motion program |
| **seed** | seed, mode (fixed/increment/random_per_frame) | Seed schedule (drives Retake) |
| **pose** | poseFile, strength, interpolation | Pose conditioning ranges |

Muted tracks are excluded before compilation. Track warnings surface in the
node's `info` JSON.

---

## Parameters (main node)

The core inputs stay short by design; the advanced quality stack lives on
**WanDirectorExtraArgs**, which you connect to the main node's `extra_args`
input.

| Group | Parameters |
|---|---|
| Model | `backend`, `model_variant`, model/clip/vae inputs |
| Duration | `duration_frames`, `duration_seconds`, `frame_rate`, `display_mode` |
| Output | `custom_width`, `custom_height`, `resize_method` |
| Sampling | `cfg` (and `cfg_low_noise`, `ref_strength` on dual-cfg variants) |
| Audio | `audio_target` |
| Timeline | `global_prompt` + the hidden `timeline_data` state the editor writes |

Selecting a `model_variant` gates which advanced fields are relevant (single-
vs dual-cfg, reference image, EverAnimate, etc.).

---

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| "+ Video / + Audio" seems to do nothing | Older builds had a leak where a dead editor swallowed clicks — fixed. Refresh the browser to load current JS |
| Video clip shows "Building edit proxy…" forever | ffmpeg must be on PATH (it decodes ProRes/EXR/MXF). Check the server console for a proxy error |
| Node looks "damaged" / timeline hangs outside the node | Fixed — the node is no longer height-capped. Refresh; workflows saved with the old capped size self-heal on load |
| Muted track still shows in the render | The mute is applied at compile time — re-queue after toggling |
| Everything blank on the timeline canvas | Check the browser console; the tracks canvas needs the DOM widget mounted (works in the classic renderer; see Nodes 2.0 notes) |

---

## License

Apache-2.0. Timeline power-tool feature set researched against LTX Director
(WhatDreamsCost, used with permission) and LTX-2 Retake. Built on Kijai's
WanVideoWrapper and wuwukaka's WanAnimatePlus.
