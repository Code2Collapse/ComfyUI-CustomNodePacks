# MaskPreviewOverlay (legacy): no equivalent node

**Status:** still registered and marked DEPRECATED. It is an output node with one IMAGE output, `preview`. It is not
migrated, because no current node draws the same picture. Its five display modes are overlay, mask_only,
side_by_side, checkerboard and edge_highlight, with edge contours and a bbox. It logs one warning per run that points
to the nearest tools:
- Mask Tools (C2C): its `image` output is the matte as a grey image.
- MaskOpsMEC: its preview output.

| Old input | Nearest current tool |
|---|---|
| image, mask, display_mode = mask_only | Mask Tools (any mode), `image` output = the matte as grey. Not the same node graph |
| the other display modes, colours, opacity, edge_width, show_bbox, bbox | none |

**Parity:** not applicable. There is no successor to compare with. The test checks that the deprecation notice leaves
the output unchanged (two runs, `torch.equal`), that the notice is logged once, and that the signature is unchanged.

**Owner decision (not taken):** keep it as is (recommended, since nothing replaces it), or extend a current preview
node first.
