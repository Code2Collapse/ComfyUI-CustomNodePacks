# BBoxFromMask (legacy): no equivalent node

**Status:** still registered and marked DEPRECATED. It is not migrated. It logs one warning per run saying there is
no replacement. Keep using it.

| Output | Notes |
|---|---|
| bbox (BBOX `[x, y, w, h]`), x, y, width, height (INT), bbox_str (JSON) | No C2C or core 0.39 node returns this BBOX type. Core's BOUNDING_BOX (`{x, y, width, height}`) is a different type and cannot be wired in its place |

Known behaviour, unchanged:
- It reads frame 0 only.
- An empty mask returns the full frame.

**Parity:** not applicable. There is no successor. The output is unchanged by the notice (tested).

**Owner decision (not taken):** keep it, or add a BOUNDING_BOX output to a current node and migrate. Adding that
output is a new feature, so it is out of scope for L9.176.
