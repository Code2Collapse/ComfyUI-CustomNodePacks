# WORKLOG — ComfyUI-CustomNodePacks

**Stage 0 audit, 2026-08-29. Every number below was MEASURED this session, not inherited.**
Regenerate with the commands in the last section. Per R3 this file is the persisted source of
record; a claim that lives only in a chat transcript has now drifted five times.

> **Updated 2026-08-29 after the build passes.** The numbers above are re-measured, not the Stage-0 audit figures. A worklog that still reports its audit snapshot is the stale-record failure R3 exists to prevent.

## 1. Live inventory
| | |
|---|---|
| **Nodes registered (runtime)** | **87** (was 104; 17 duplicate ids deleted in the 2026-08-29 dedup) |
| Registration style | V1 `NODE_CLASS_MAPPINGS` |
| `WEB_DIRECTORY` | `./js` |
| Test files | 17 |
| **Tests** | **409 passed, 0 failed, 0 collection errors** (was 330/65/2 at Stage 0) |

## 2. Licence
**Apache-2.0.** Inbound: MIT/Apache only. **GPL forbidden inbound** — this rules out porting code
from MiniMaxSuite (GPL-3.0). Shared work with the suite must be a data-format contract
(H3_TRANSFORM, AOV channel packing, keypoint JSON), never a shared module.

## 3. Registration smoke
PASS — 104 nodes under the production loader.

## 4. Test status — THE HEADLINE FINDING
The suite does not currently function as a signal.

- **Collection aborts.** `tests/test_nukenodemax.py` imports `nodes/deep_composite.py`, which no
  longer exists — it was migrated to ComfyUI-NukeMaxNodes (that repo's `__init__.py` records
  "Migrated from ComfyUI-CustomNodePacks (Apr 2026)"). `tests/test_shape_draw_nodes.py` imports
  `DrawCircleMEC` from `nodes.mask_draw_frame`, which no longer defines it. Until those two are
  removed or repointed, plain `pytest tests/` collects **nothing at all** and the other 15 files
  never run.
- **65 failures, dominated by stale renames rather than broken product:**
  - 27 `test_sam_multi_mask_picker.py` — `assert 'C2C/SAM' == 'MaskEditControl/SAM'`. Category
    renamed to C2C; tests never updated.
  - 15 `test_inpaint_suite.py` — `AttributeError: 'InpaintCropProMEC' object has no attribute
    'crop_for_inpaint'`. Method renamed away.
  - 9 `test_vae_merge.py`, 4 `test_folder_incrementer_fixes.py`, 4 `test_phase4_nodes.py`,
    3 `test_spline_mask_editor.py`, 3 singles.

**Read this correctly:** all 104 nodes register cleanly, so the product is probably fine. The tests
have decayed. That is arguably worse than a red product — a suite this stale cannot tell you when
something real breaks, and it has been silently non-collecting.

## 5. Invariant sweep
| Check | Result |
|---|---|
| `third_party` runtime imports | 0 |
| `IS_CHANGED` returning `float("nan")` | 0 |
| Hardcoded `.cuda()` | **1** — R5 violation |
| Frame loops without interrupt check | **61 of 70** node files |

## 6. Hang risk (R5)
61 of 70 node files loop over frames or a batch dimension with no interrupt check. Helpers already
exist (`_interrupt_check.py`, `_progress.py`) and 9 node files use them. Same shape as the
NukeMaxNodes hanging complaint.

## 7. Build queue (brief 2.1)
1. **Repair the test suite first** — nothing below is verifiable until `pytest tests/` collects.
2. `sam_model_loader` batch processing (the original ask) + tests.
3. #66 waveform / false-colour scope. JS DOM widget: the `ui=` payload is the ONLY channel to the
   browser — socket values never reach `onExecuted`.
4. #67 3D LUT from plate histogram (`parse_cube_lut` / `_apply_lut_3d` already in-house).
5. #69 OKLab palette transfer (NKD, MIT — allowed inbound).
6. #82 QC, #106 deliverable export.

## 8. Blocked / decisions
#82's "reuse RAFT from WanNodeExperiments" must be a re-implementation or a data contract, not an
import — cross-repo runtime imports are forbidden.

## Regeneration commands

```
head -3 LICENSE

# registration smoke, the way ComfyUI loads (third_party/ComfyUI/nodes.py:2243-2263):
#   sys.modules[name] = mod   BEFORE   spec.loader.exec_module(mod)
# Anything less can report healthy for a pack that registers nothing.
python <scratch>/regsmoke.py ComfyUI-CustomNodePacks

D:/PROJECT/ComfyUI_windows_portable/comfy_env/python.exe -m pytest tests/ -q
```

Shell python has no torch — always use the comfy_env interpreter.
