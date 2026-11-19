# BBoxToMask (legacy): core Create Solid Mask + Combine Masks

**Status:** still registered and marked DEPRECATED. It is not migrated automatically: the replacement is two core
nodes, and a migration row maps one node to one node. It logs one warning per run that describes the replacement.

| Old input | Core replacement |
|---|---|
| image_width, image_height | Create Solid Mask: value 0, width, height (the destination) |
| bbox `[x, y, w, h]` | Create Solid Mask: value 1, width w, height h (the source), then Combine Masks with x, y and operation `add` |
| reference_image (optional) | no single input; read its size with Get Image Size and wire it to the first Solid Mask |

**Parity:** bit-exact (`torch.equal`) for box `[6, 4, 10, 7]` on a 32x24 canvas. The core `execute` bodies were taken
from the installed core 0.39.0 file and run (`test_bbox_to_mask_matches_core_solid_mask_plus_combine`).

Difference: the old node clipped a box with a negative x or y. Core x and y start at 0.

**Owner decision (not taken):** keep it, or remove it. Removal needs a two-node expansion, which the migration does
not support.
