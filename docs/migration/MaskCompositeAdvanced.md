# MaskCompositeAdvanced (legacy): replaced by operation

**Status:** still registered and marked DEPRECATED (category Legacy). It is not migrated automatically, because no
single node does all eight operations. It logs one warning per run that names the replacement for each operation.
Saved workflows keep running unchanged. There is no kill switch, because nothing was removed.

| Old operation | Replacement | Parity (2x24x32 masks, seeded) |
|---|---|---|
| union, max | Mask Tools (C2C), mode `combine`, op `union` | bit-exact |
| intersect, min | Mask Tools `combine`, op `intersect` | bit-exact |
| subtract | Mask Tools `combine`, op `subtract` | bit-exact |
| difference (\|a-b\|) | Mask Tools `combine`, op `xor` (it computes \|a-b\|) | bit-exact |
| xor (binarised at 0.5) | core Combine Masks (MaskComposite), `xor`, x=y=0 | bit-exact, including pixels at exactly 0.5 |
| blend | none | no replacement (Mask Tools has no weighted mix) |

| Old input | Replacement |
|---|---|
| mask_a / mask_b | mask / mask_b (Mask Tools), or destination / source (core) |
| invert_a, invert_b | core Invert Mask in front: bit-exact (tested with union) |
| threshold > 0 | core Threshold Mask after the combine. **Differs:** core keeps values `> t`, the old node kept values `>= t`. In the test, 93 of 1536 pixels differ, all where the value is exactly 0.5 |
| different batch sizes or frame sizes | the old node broadcast the batch and resized mask_b bilinearly; Mask Tools `combine` raises ValueError instead (tested) |

**Owner decision (not taken):** whether to remove this node with automatic migration. The `c2c_by_value` rows can
send each operation to its own successor. `blend`, `threshold` with `>=`, and the size handling would change or be lost.
