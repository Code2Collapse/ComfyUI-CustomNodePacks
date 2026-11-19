# Deprecated compatibility nodes: migration and parity (L9.176, A25)

One page per deprecated node: what replaces it, how each input and output maps, and the parity test result.
Every number on these pages comes from `tests/test_l9176_deprecated_parity.py` (run 2026-11-19, CPU, torch from
`comfy_env`, core 0.39.0 at `D:/PROJECT/_c2c_testbed/ComfyUI-owner` for the core comparisons). The full generated
replacement table for every merged node is in [../MIGRATION.md](../MIGRATION.md).

How migration works (decision D0.14): a node id that is **really gone** is migrated when a saved workflow loads
(`js/_c2c_migrate_core.js`) and when an API prompt names it (`nodes/_legacy_replacements.py`, on_prompt hook).
An id that is **still registered** is never migrated by either path. It runs as itself and, from L9.176, logs one
warning per server run that names its replacement.

| Node id | Status | Replaced by | Parity | Page |
|---|---|---|---|---|
| VideoStabilizerClassicMEC | removed; `C2C_LEGACY_STABILIZERS=1` restores it | VideoStabilizerMEC, method=classic | bit-exact | [VideoStabilizerClassicMEC](VideoStabilizerClassicMEC.md) |
| VideoStabilizerFlowMEC | removed; same switch | VideoStabilizerMEC, method=dense_flow | bit-exact | [VideoStabilizerFlowMEC](VideoStabilizerFlowMEC.md) |
| VideoStabilizerAutoMEC | removed; same switch | VideoStabilizerMEC, method from force_backend | bit-exact | [VideoStabilizerAutoMEC](VideoStabilizerAutoMEC.md) |
| MaskCompositeAdvanced | registered, deprecated | Mask Tools (C2C) combine / core masks, by operation | exact for 7 of 8 operations; blend has none | [MaskCompositeAdvanced](MaskCompositeAdvanced.md) |
| MaskMath | registered, deprecated | Mask Tools (C2C) grade, by operation | 7 of 11 operations, max error 1.2e-7 | [MaskMath](MaskMath.md) |
| MaskPreviewOverlay | registered, deprecated | none with the same picture | not applicable | [MaskPreviewOverlay](MaskPreviewOverlay.md) |
| BBoxFromMask | registered, deprecated | none | not applicable | [BBoxFromMask](BBoxFromMask.md) |
| BBoxToMask | registered, deprecated | core Create Solid Mask + Combine Masks | bit-exact | [BBoxToMask](BBoxToMask.md) |
| BBoxPad | registered, deprecated | none | not applicable | [BBoxPad](BBoxPad.md) |
| BBoxCrop | registered, deprecated | core Crop Image + Crop Mask | bit-exact | [BBoxCrop](BBoxCrop.md) |
| SimpleWatermarkRemover | not a C2C node (ComfyUI-AusBoss id) | n/a | n/a | [SimpleWatermarkRemover](SimpleWatermarkRemover.md) |

The seven registered nodes stay registered because none has a full replacement. Removing any of them with automatic
migration would drop operations that no current node has. That is an owner decision, listed on each page.
