# MaskMath (legacy): replaced by Mask Tools (C2C) grade, by operation

**Status:** still registered and marked DEPRECATED. It is not migrated automatically, because four operations have
no replacement. It logs one warning per run. Saved workflows run unchanged.

Every mapping is to Mask Tools mode `grade`. Inputs not listed stay at their defaults: gain 1, gamma 1, clamp on,
invert off, black point 0, white point 1, lift 0, offset 0.

| Old operation (value_a, value_b) | grade settings | Max abs error (2x24x32 seeded mask) |
|---|---|---|
| invert | invert = on | 0 |
| multiply_scalar (a) | gain = a (range 0..16) | 0 |
| add_scalar (a) | offset = a (range -1..1) | 0 (a = 0.25 and -0.4) |
| gamma (g) | gamma = g (range 0.05..8) | 0 |
| power (p) | gamma = 1/p (p in 0.125..20) | 0 (p = 2); the test allows 1e-6 for 1/(1/p) rounding |
| remap_range (lo, hi) | black point = lo, white point = hi | 1.2e-7 (float rounding of the scale and shift) |
| contrast (c, mid) | gain = c, offset = mid*(1-c) | 6.0e-8 |
| clamp (a, b) | none | no replacement |
| quantize | none | no replacement |
| threshold_hysteresis | none | no replacement |
| abs_diff_from_value | none | no replacement |

Limits of the mapping:
- A negative `multiply_scalar` or `contrast` factor cannot be set: gain starts at 0.
- Values outside the grade ranges cannot be set either.
- The old node clamped its input to 0..1 before power or gamma. Grade only clamps below 0. Masks are already 0..1, so the results only differ for out-of-range inputs.

**Owner decision (not taken):** whether to remove the node with migration. The four operations without a
replacement would be lost.
