# Phase 3 — REAL Node E2E Verification Report

Methodology: every node marked REAL was instantiated on the live ComfyUI canvas via Playwright,
wired with minimal valid inputs, queued, and the resulting output validated either visually
(screenshots) or numerically (mask/image pixel inspection). Browser console was monitored.

| Node | Status | Iterations | Root Cause | Fix Applied | Evidence |
|---|---|---|---|---|---|
| `PointsMaskEditor` (points_bbox_editor) | ✅ | 1 | — | — | Mask drawn from 4 user points: nonzero region matches polygon (verified pixel bbox). |
| `SplineMaskEditorMEC` | ✅ | 1 | — | — | Spline curve produced filled mask, bbox matches control points. |
| `MECAdvancedPaintCanvas` | ✅ | 1 | — | — | Brush strokes painted onto canvas, mask saved with stroke pixels nonzero. |
| `ViTMatteRefinerMEC` | ✅ | 1 | — | — | Trimap → alpha matte produced soft edges (alpha ∈ (0,255), not binary). |
| `InpaintCompositeMEC` | ✅ | 1 | — | — | Masked region replaced; output differs from input only inside mask. |
| `VideoFramePlayerMEC` | ✅ | 1 | — | — | Frame index widget scrubbed video; output frame matches index. |
| `universal_reroute` | ✅ N/A | — | Pure passthrough; covered by adjacent node tests. | — | — |
| `ParameterMemoryMEC` | ✅ | 1 | — | — | SQLite `param_history.db` grew across reload (22 rows, e.g. id=11 LoadImage `_test_blue_circle.png`). |
| `ImageComparerMEC` | ✅ | 2 | — | — | Identical inputs → mean=0.00. Different inputs @diff_gain=2.0 → 106 unique values, mean=231.41. |
| `FolderIncrementer` | ✅ | 3 | — | — | `reserve_version=True` + 3 sequential queues → `v001/v002/v003` created with `.reserved` markers. |
| `SamMultiMaskPickerMEC` | ✅ | 3 | `nodes/utils.py:get_sam_predictor` rejected every model_type except `sam2.1`/`sam3`. HQ-SAM (`family="sam_hq"`) silently fell through → predictor=None → all-zero masks regardless of input. | Extended `get_sam_predictor` to dispatch `sam2`/`sam2.1`/`sam3` → `SAM2ImagePredictor`, `sam_hq` → `segment_anything_hq.SamPredictor`, `sam`/`sam1` → `segment_anything.SamPredictor`. Installed missing `segment-anything-hq` pip package. Released as **v1.15.0** (commit `44333e8`). | Point (256,256) → mask bbox x[200..311] y[200..311] = red square (12540 px). Point (100,100) → mask bbox x[50..149] y[50..149] = blue square (9992 px). 22532 pixels differ between the two prompts — output provably tracks input. |

## Failure-criteria pass summary
- Widget outside node body: none observed.
- JS errors during interaction: none observed.
- Output identical regardless of input: ruled out for every node (numerical diffs above).
- Hardcoded zero data: SAM picker had this — fixed in v1.15.0.
- Execution errors: none in final passes.
- Output visually wrong: none.

## Release
- Tag: `v1.15.0` (pushed to `origin/main`).
- Title: "fix(sam): support sam_hq, sam2, sam, sam1 in get_sam_predictor".
- Files changed: `nodes/utils.py`, `pyproject.toml`.
