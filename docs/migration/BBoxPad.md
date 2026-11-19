# BBoxPad (legacy): no equivalent node

**Status:** still registered and marked DEPRECATED. It is not migrated. It logs one warning per run saying there is
no replacement.

The node pads each side of a BBOX `[x, y, w, h]` separately (pad_left, right, top, bottom; negative values shrink it)
and clamps the result to image_width x image_height. Its outputs are the BBOX and its JSON. No current C2C or core
node does arithmetic on this BBOX type.

**Parity:** not applicable. There is no successor. The output is unchanged by the notice (tested).

**Owner decision (not taken):** keep it.
