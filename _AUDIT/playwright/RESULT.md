# Playwright C2C/MEC Sweep — Result

**Date:** 2026-06-05 (updated)  
**Runner:** `_AUDIT/playwright/c2c_node_ui_tester.mjs`  
**ComfyUI:** http://127.0.0.1:8188 (live)

---

## Executive summary

| Run | Nodes | Scenarios | PASS | FAIL | Console errors |
|-----|------:|----------:|-----:|-----:|---------------:|
| Baseline (`2026-06-05T12:42Z`) | 109 | 869 | 673 | **85** | 0 |
| After overlay fix (`2026-06-05T20:00Z`) | 109 | 875 | 763 | **0** | 0 |
| After FC3D resize fix (manual) | 1 | — | ✅ drag dw≥280 dh≥82 | 0 | 0 |

**Automated sweep:** 0 FAIL after popover + SAM gradient fixes.  
**Production UI gaps:** 2 CRITICAL nodes still need human visual sign-off (see §Production node assessment).  
**New this session:** Safe image drop + PNG metadata cleaner (non-invasive `app.handleFile` wrap).

---

## Line-by-line read of baseline failures (RESULT.md §Failures)

### Line 20–33: `DIV#mec-node-explain-popover` → `HARDCODED_FIXED_POSITION` (84 nodes)

| Line claim | Verdict | Notes |
|------------|---------|-------|
| Same failure on 84 nodes | **TRUE** — one global bug | Not 84 distinct node implementations |
| `position:fixed; top:190px` | **TRUE** | Popover used viewport-fixed coords |
| Triggered on `VIEWPORT_RESIZE` | **TRUE** | Popover left open after hover crossed title |
| Fix: `position:absolute` in `c2c_node_explain.js` | **RESOLVED** | Re-run: 0 FAIL on this pattern |
| Panel mode stays `fixed` | **INTENTIONAL** | Corner-docked; not flagged by leak detector |

### Line 35–45: `SamMultiMaskPickerMEC` → `JS_CRASH` on `addColorStop`

| Line claim | Verdict | Notes |
|------------|---------|-------|
| `var(--c2c-panelMid)` passed to canvas | **TRUE** | `_cssVar()` returned unresolved `var(...)` |
| Canvas2D cannot parse CSS vars | **TRUE** | Spec limitation |
| Fix: reject `var(` prefix, use hex fallback | **RESOLVED** | Spot-check 7/7 PASS |

### Line 47–57: Fixes applied (popover + SAM)

Both fixes verified in full 109-node re-run. **No regression** in console error count.

### Line 59–64: Verification

| Step | Status |
|------|--------|
| SamMultiMaskPickerMEC spot-check | ✅ 7/7 |
| NumberLerpMEC spot-check | ✅ 7/7 |
| Full 109-node sweep | ✅ 763 PASS / 0 FAIL |

### Line 66–68: STUCK items

Baseline report: **None** (for automated FAIL class).  
**Updated:** Visual/production gaps remain (§Production node assessment) — not counted as automated FAIL.

### Line 70–73: Next-step suggestions

| Suggestion | Status |
|------------|--------|
| 112 `CHECK_VISUALLY` ERROR_STATE | Still manual; not automated FAIL |
| Allowlist for docked `position:fixed` | Partially done in tester whitelist |
| Safe corrupt-metadata drop | **DONE** — `c2c_safe_image_drop.js` |
| PNG metadata strip API | **DONE** — `POST /c2c/doctor/clean_png` |

---

## Production node assessment (PLAYWRIGHT_REPORT + FULL_AUDIT cross-read)

Automated sweep passes **mechanics** (resize, pan, zoom, overlay leak). It does **not** prove premium layout. Below is the honest per-node UI/backend status for production-facing nodes.

### WanFaceController3DV2 — CRITICAL

| Check | Automated | Visual / production | Root cause |
|-------|-----------|---------------------|------------|
| Registered | PASS | — | — |
| Resize | PASS (after fix) | PARTIAL | Was: LiteGraph `node.size` not array-like; `_relayout` forced minHeight=680 |
| Layout void | not auto-tested | **FAIL → FIX IN PROGRESS** | ~34 Python optional widgets still in `INPUT_TYPES`; JS hides chrome but LiteGraph may reserve height before hide |
| Backend `/c2c/fc3d_preview` | not in sweep | PASS (logic) | Python `run()` contract intact |
| Duplicate title | — | **FIXED** | Internal title bar removed; frame hint in tab bar |

**Fixes landed (2026-06-05):** `_fc3dWriteNodeSize`, `_fillAvailableHeight`, auto-grow block, `_fc3dOrigSetSize` for init, **`addWidget` hook** (hide chrome at creation), **legacy inflated height shrink** on workflow load.  
**Still optional:** Collapse Python `INPUT_TYPES` to JSON-only for cold-start without JS (saved workflows keep working via widgets).

### WanDirectorC2C — HIGH

| Check | Automated | Visual / production | Root cause |
|-------|-----------|---------------------|------------|
| Registered | PASS | — | — |
| Height cap | PASS* | Was **FAIL ~2010px** | ~50 widgets + timeline DOM `computeSize` ~474px + visible prompts |
| Sockets visible | PASS* | **FIXED** (compact.js) | Gated widgets collapsed; `Show advanced` toggle |
| Backend director | not in sweep | PASS | `director_node.py` + variant gate intact |

**Fixes landed:** `wan_director_compact.js`, `wan_director_variant_gate.js`, `_writeNodeSize` in cap paths.  
**User action:** Hard refresh; add fresh node; confirm height ≤ ~62% viewport.

### VideoComparerC2C — HIGH

| Check | Automated | Visual | Gap |
|-------|-----------|--------|-----|
| Overlay leak | PASS | — | — |
| Playback on empty node | CHECK_VISUALLY | **UNKNOWN** | Fresh node may have empty file combos — needs default mode + upload CTAs |
| Backend | not in sweep | PASS | Python node + JS player exist |

### SamMultiMaskPickerMEC — MEDIUM

| Check | Status |
|-------|--------|
| ADD_TO_CANVAS crash | **RESOLVED** (gradient CSS var) |
| Full visual | pending golden screenshot |

### IntegrityStatusMEC / Doctor V3 — CRITICAL

| Check | Status |
|-------|--------|
| Overlay leak | PASS |
| Pack integrity tab (DEF-003) | PARTIAL — not re-verified live this session |
| Plain-English errors | RESOLVED (`c2c_ai_error_translator.js`) |

### Global C2C chrome (not node bugs)

| Artifact | Issue | Status |
|----------|-------|--------|
| `#mec-node-explain-popover` | Fixed viewport coords | **FIXED** |
| `#mec-whats-wired` / Inspector | Tester false positives | Whitelisted in tester |
| `c2c_metadata_inspector.js` | Blocked **all** PNG drops with text chunks | **FIXED** — defers to safe drop |
| 66× `position:fixed` in JS | Dock vs leak ambiguity | Ongoing triage (`_c2c_top_dock.js` pattern) |

---

## New: Safe image drop (non-invasive)

**Design principle:** Wrap `app.handleFile` (same pattern as ComfyUI-Manager). **No capture-phase `preventDefault`** on canvas drops unless SafeImageDrop is absent (legacy inspector fallback only).

| File | Role |
|------|------|
| `js/c2c_png_metadata.js` | Bounds-safe chunk reader + `assessWorkflowMetadata()` + `loadImageOnly()` |
| `js/c2c_safe_image_drop.js` | `app.handleFile` wrap; corrupt JSON → image-only; Shift=Comfy default; Alt=force image |
| `js/c2c_metadata_inspector.js` | Modal only via `app.__c2cMetaInspectPrompt`; no longer hijacks every PNG |
| `nodes/c2c_doctor.py` | `POST /c2c/doctor/clean_png` strips workflow tEXt/iTXt chunks |

**Settings (ComfyUI → Settings → C2C):**

- `C2C ▸ Safe image drop` — on by default  
- `C2C ▸ Safe drop mode` — Auto | Image only (no metadata) | Comfy default  

**ComfyUI behavior preserved when:**

- Valid workflow PNG + user confirms in metadata inspector → `loadGraphData`  
- Shift+drop → original Comfy `handleFile`  
- Non-image files → untouched  

---

## CHECK_VISUALLY bucket (112 scenarios)

These are **not FAIL**. The runner queues an empty `/prompt` and flags ERROR_STATE for human review (toast/notification sanity). Treat as a backlog, not a regression.

---

## Verification commands

```powershell
cd D:\PROJECT\Custom_Nodes\_AUDIT\playwright
python build_node_queue.py
node c2c_node_ui_tester.mjs --priority ALL          # ~2 h, expect 0 FAIL
node c2c_node_ui_tester.mjs --node WanFaceController3DV2
node fc3d_resize_test.mjs                           # resize dw/dh thresholds
```

Hard refresh ComfyUI (`Ctrl+Shift+R`) after pulling JS changes.

---

## Remaining work for “flawless UI + working backend”

| Priority | Item | Owner |
|----------|------|-------|
| P0 | `face_controller_3d.py` — collapse optional widgets to hidden JSON | Python |
| P0 | Re-screenshot WanFaceController + WanDirector after hard refresh | Visual QA |
| P1 | VideoComparerC2C — empty-state upload UX | JS |
| P1 | Doctor V3 integrity tab live verify (DEF-003) | JS + Python |
| P2 | Automate CHECK_VISUALLY via golden PNG diff | Tester |
| P2 | Triage 66× `position:fixed` against dock allowlist | JS audit |

---

## STUCK items

None for **automated FAIL**. Production visual sign-off on WanFaceController3DV2 and VideoComparerC2C remains open.
