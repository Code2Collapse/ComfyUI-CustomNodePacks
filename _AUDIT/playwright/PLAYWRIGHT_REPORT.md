# C2C Playwright Live Test Report

**Date:** 2026-06-05  
**ComfyUI Version:** 0.24.0 (frontend 1.44.19)  
**Device:** RTX 4060 Laptop GPU 8 GB  
**Test Script:** `c2c_node_ui_tester.mjs` (Playwright Chromium headless)  
**ComfyUI URL:** http://127.0.0.1:8188  
**Screenshot root:** `D:\PROJECT\Custom_Nodes\_AUDIT\playwright\screenshots\`  
**Results JSON:** `D:\PROJECT\Custom_Nodes\_AUDIT\playwright\results.json`

---

## Summary

| Metric | Value |
|---|---|
| Registered pack nodes in queue | **109** (from `node_queue.json`) |
| HIGH+CRITICAL nodes scenario-tested (automated) | **36** |
| MEDIUM nodes run | **In progress** (`run_medium.log`) |
| Total screenshots captured (this session) | **370+** (36 nodes × ~10 + baseline) |
| Automated overlay-leak failures (first HIGH run, unfiltered) | **211** (false positives — global C2C chrome) |
| Automated overlay-leak failures (after whitelist fix) | **0** on re-run sample nodes |
| Visual FAIL findings (real UI bugs) | **2** (`WanFaceController3DV2` black void, `WanDirectorC2C` extreme height) |
| Fixes applied this session | **4 files** (see below) |
| Console errors during node suites | **0** new C2C/MEC errors |

---

## Preflight (Section 0)

| Step | Status |
|---|---|
| 0-A ComfyUI on 8188 | PASS (restarted via `conda_run.bat` after long run) |
| 0-B Playwright install | PASS (`@playwright/test` + Chromium) |
| 0-C Screenshot dirs | PASS |
| 0-D Live registry | PASS → `object_info_live.json`, `node_queue.json` (109 OK) |

---

## Directive class-name mismatches

These names in the original PRIORITY_QUEUE are **JS extensions / UI chrome**, not LiteGraph node classes:

| Directive name | Real artifact |
|---|---|
| `MEC_NodeExplain` | `js/c2c_node_explain.js` (hover popover on any node) |
| `MEC_DiagnosticsSidebar` | `js/c2c_diagnostics_sidebar.js` (sidebar tab) |
| `C2C_IntegrityChecker` | Doctor V3 / integrity tabs in `c2c_doctor.js` |
| `C2C_AIErrorTranslator` | `js/c2c_ai_error_translator.js` (execution_error hook) |
| `C2C_OmniPill` | `#c2c-omnibar-pill` DOM in actionbar |
| `C2C_VideoComparer` | **`VideoComparerC2C`** (registered node) |
| `PoseAndFaceDetectionV2` | **`WanPoseDetectViTPoseV2`** / **`PoseAndFaceDetectionV2`** (both registered) |
| `DrawViTPoseV2` | Registered as **`DrawViTPoseV2`** |

---

## Per-node results (HIGH + CRITICAL)

### WanFaceController3DV2 — CRITICAL

| Scenario | Auto | Visual | Notes |
|---|---|---|---|
| REGISTRATION | PASS | — | In `LiteGraph.registered_node_types` |
| INITIAL_STATE | PASS* | **FAIL** | Tabs + wireframe render; **large black void** still occupies ~50% of node body (see `01_initial.png`) |
| RESIZE | PASS* | PASS | Editor reflows; void persists proportionally |
| CANVAS_PAN | PASS* | PASS | Node moves with canvas |
| CANVAS_ZOOM | PASS* | PASS | Scales cleanly |
| SIDEBAR_TOGGLE | PASS* | PASS | No collision with sidebar |
| VIEWPORT_RESIZE | PASS* | PASS | No horizontal overflow at 1280 |
| ERROR_STATE | CHECK | PASS | Empty prompt → 400; no raw traceback in UI |
| WFC3D A/B/C | N/A | N/A | Inline `face_overlay` editor — no separate popup |

\*After overlay-detector whitelist fix (global `#mec-whats-wired` / inspector excluded).

**Root cause (black void):** ComfyUI allocates node height for ~34 hidden Python widgets + DOM widget slot; `root`/`ctxScroll` flex fill and/or widget stack order leaves empty `#0e0e16` canvas-bg region.

**Fixes applied (layout-v3+):** `face_controller_3d.js` — remove `height:100%` on root, `ctxScroll flex:0`, explicit `root.style.height`, DOM widget moved **first**, rAF size-clamp loop.

**User action:** Hard refresh (Ctrl+Shift+R), delete old node, add fresh one.

---

### WanDirectorC2C — HIGH (user complaint #1)

| Scenario | Auto | Visual | Notes |
|---|---|---|---|
| REGISTRATION | PASS | — | 57 inputs / ~50 widgets |
| INITIAL_STATE | PASS* | **FAIL** | **`nodeH ≈ 2010 px`** at zoom 1.0 — only `global_prompt` band visible; input sockets scrolled off top (`01_initial.png`) |
| RESIZE/PAN/ZOOM/SIDEBAR/VIEWPORT | PASS* | PARTIAL | Overlays OK; node still unusably tall |
| ERROR_STATE | CHECK | — | 400 on empty prompt |

**Root cause:** Python default size + sum of ~50 visible widgets + timeline/player DOM widgets; `computeSize()` returns ~2000 px before user scrolls.

**Fixes applied:** `wan_director_timeline.js` — `WD_MAX_VIEW_H()` cap (~62% viewport), prototype `computeSize` + **`setSize` hook**, repeated `_wdCapNode()` after DOM init.

**Status:** Cap not yet effective in headless run (still 2010 px) — likely ComfyUI frontend sets size outside patched `setSize`. **Next:** patch `onResize` / collapse `enable_*` widget groups in `wan_director_variant_gate.js`.

---

### Other HIGH nodes (36 total)

All **36 registered** HIGH/CRITICAL nodes completed the automated scenario suite (screenshots under `screenshots/<ClassName>/`).

| Category | Examples | Automated overlay leaks (filtered) | Visual spot-check |
|---|---|---|---|
| Mask / MEC | `MaskEditMEC`, `MaskOpsMEC`, `MECFaceFixer` | 0 after whitelist | DOM-heavy nodes render; full visual review pending |
| Wan V2 extras | `WanPoseDetectViTPoseV2`, `DrawViTPoseV2` | 0 | Standard widget nodes OK at initial glance |
| C2C tools | `VideoComparerC2C`, `IntegrityStatusMEC` | 0 | Needs dedicated playback test for comparer |

First unfiltered run flagged **every** node due to `#mec-whats-wired` + `#mec-node-explain-popover` — **not node bugs**.

---

## Visual analysis highlights (Section 2)

### `WanFaceController3DV2/01_initial.png`

| Check | Result |
|---|---|
| Node visible | PASS |
| Title bar | PASS |
| Tab labels | PASS (Face/Expr/Gaze/Pose/Settings) |
| Controls rendered | PASS (canvas, transport, timeline strip) |
| Overlapping elements | PASS |
| OmniPill position | PASS (top-right, no collision) |
| Node-anchored overlays | PASS |
| **Black void / layout** | **FAIL** — empty dark region ~50% of node height |
| Console errors | PASS (0) |

### `WanDirectorC2C/01_initial.png`

| Check | Result |
|---|---|
| Node visible | PASS (but clipped vertically) |
| Input sockets visible | **FAIL** — scrolled off viewport |
| Widget labels | PARTIAL — only prompt band visible |
| Extreme height | **FAIL** — ~2× viewport tall |
| OmniPill | PASS |

---

## Fixes applied (Section 4)

| File | Change |
|---|---|
| `_AUDIT/playwright/build_node_queue.py` | Build queue from `NODE_CLASS_MAPPINGS` + live `object_info` (109 OK) |
| `_AUDIT/playwright/c2c_node_ui_tester.mjs` | Load `node_queue.json`; headless Chromium; `--node` / `--priority`; overlay whitelist; longer wait for layout-heavy nodes |
| `ComfyUI-WanAnimatePreprocessV2/js/face_controller_3d.js` | layout-v3+: fixed root height, ctxScroll flex, DOM-first, rAF clamp |
| `ComfyUI-CustomNodePacks/js/wan_director_timeline.js` | Viewport-relative height cap, `computeSize`/`setSize` hooks, `_wdCapNode` |

---

## Stuck / follow-up

| Item | Attempts | Blocker |
|---|---|---|
| WanDirector height cap | 3 | Size still 2010 px after hooks — need variant_gate collapse + direct `node.size[1]` assignment in ComfyUI Vue path |
| WanFaceController black void | 2 | Hidden Python widgets still participate in layout — may require shrinking `INPUT_TYPES` to hidden-only |

---

## Screenshots index (sample)

| Path | Node | Scenario | Visual |
|---|---|---|---|
| `screenshots/_00_comfyui_loaded.png` | — | boot | PASS |
| `screenshots/WanFaceController3DV2/01_initial.png` | WanFaceController3DV2 | INITIAL | **FAIL** (black void) |
| `screenshots/WanFaceController3DV2/03_resize_taller.png` | WanFaceController3DV2 | RESIZE | PARTIAL |
| `screenshots/WanDirectorC2C/01_initial.png` | WanDirectorC2C | INITIAL | **FAIL** (2010 px) |
| `screenshots/VideoComparerC2C/01_initial.png` | VideoComparerC2C | INITIAL | pending review |
| `screenshots/MECAdvancedPaintCanvas/01_initial.png` | MECAdvancedPaintCanvas | INITIAL | pending review |

Full tree: `screenshots/<NodeClass>/01_initial.png` … `10_error_state.png` (+ `A/B/C` for WanFaceController3DV2).

---

## How to re-run

```powershell
cd D:\PROJECT\Custom_Nodes\_AUDIT\playwright
python build_node_queue.py
node c2c_node_ui_tester.mjs --priority HIGH      # CRITICAL + HIGH (36 nodes)
node c2c_node_ui_tester.mjs --node WanFaceController3DV2
node c2c_node_ui_tester.mjs --priority ALL       # all 109 registered nodes (~2 h)
```

Use `--headed` to watch Chromium during debugging.
