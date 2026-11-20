# Mask Refiner

Mask Refiner cleans a mask you already have, guided by the image: it fills holes, removes specks, snaps the edge to
the image's colour edges, recovers thin strands and softens or hardens the result. It uses no model weights. After a
run its QC viewer shows the plate with the refined matte.

| Node | Use it to |
|---|---|
| [Mask Refiner](#mask-refiner-maskrefinemec) (`MaskRefineMEC`) | Clean and edge-snap an existing mask or matte, still or video |

**Where the older matting nodes went.** These were removed; their work is now done by other nodes:

- Matting Node (MEC): use [Mask + Matting](mask-matting-pipeline.md) (`MaskOpsMEC`) with a `matter`.
- ViTMatte Refiner (`ViTMatteRefinerMEC`): replaced by Mask Refiner (commit 93fc880). The same edge-method choice
  (ViTMatte, guided filter, multi-scale guided, colour-aware, Laplacian blend) is offered by Mask + Matting, backend
  `SAM + ViTMatte`, control `sam_refine_method`.
- Trimap Generator (`TrimapGeneratorMEC`): folded into Mask + Matting as `enable_advanced_trimap` (commit 47641f1).
- Luminance Keyer (`LuminanceKeyerMEC`): Mask Tools (`MaskToolsMEC`), `mode` = `key: luma`; saved workflows load as
  Mask Tools. See [mask-editing.md](mask-editing.md).

These three ids (Matting Node, ViTMatte Refiner, Trimap Generator) have no automatic migration entry in
[MIGRATION.md](MIGRATION.md): a workflow saved with them shows them as missing nodes.

---

## Mask Refiner (`MaskRefineMEC`)

**Menu:** 🐺 C2C/🧰 Core/Pipeline · **Parameters:** [NODE_REFERENCE](../NODE_REFERENCE.md#maskrefinemec)

No model weights. Some stages need an optional Python package and are skipped without it (the `info` output lists
them): `opencv-contrib-python` (joint bilateral, domain transform), `scikit-image` (thin-strand recovery), `scipy`
(hole fill, speck removal), `pydensecrf` (DenseCRF).

### What it is for

A segmenter's mask is often a little off: holes in the eyes or mouth, specks, an edge that bleeds into the
background, hair cut off. Mask Refiner fixes that without re-running segmentation. The stages run in a fixed order:
hole fill → morphology → thin-strand recovery → joint bilateral → guided filter → DenseCRF → edge snap → extra
cascade passes → feather → gamma → threshold. Each stage has its own switch.

### When to pick it

- You have a mask (from SAM, a keyer, a paint node) and want cleaner edges: this node.
- You have no mask yet, or need a soft alpha for hair and fur from a matting model: [Mask + Matting](mask-matting-pipeline.md).
  Its `alpha` can feed this node.
- You want to grow, shrink, blur, key or combine masks: Mask Tools ([mask-editing.md](mask-editing.md)).
- You want to stabilise a mask over time or check its drift: Mask Track ([video-temporal-bbox.md](video-temporal-bbox.md)).
  Mask Refiner's `enable_temporal_smooth` only evens out flicker across the batch.

### Minimal wiring

```
Load Image ──image──────────────► Mask Refiner ──mask──► (inpaint, composite)
SAM / Mask + Matting ──mask─────►      │
                                       ├─ alpha    (soft, before threshold)
                                       ├─ preview  (image × alpha)
                                       └─ info
```

`image` is required: the edge stages read the image's colours. A video works the same way, one frame per mask.

### Key controls

- `preset`: `balanced`, `fast`, `hair`, `aggressive` or `crf_heavy`. It sets the numbers (radii, strengths,
  iterations) for every stage you switch on.
- `auto_edge_lock` (on by default) with `subject_class`: pins the edge to real image edges. It always switches on the
  guided filter and edge snap, and the subject adds more:
  - `face`: joint bilateral on, morphology off, a light feather (0.6) when `feather_sigma` is 0;
  - `hair`: thin-strand recovery on, morphology off;
  - `garment`: hole fill on;
  - `object`: hole fill on, morphology `close` when `morph_op` is `none`;
  - `hard_surface`: morphology off, a hard cut at 0.5 when `threshold` is 0;
  - `general`: guided filter and edge snap only.
- The stage switches (`enable_hole_fill`, `morph_op`, `enable_thin_recover`, `enable_joint_bilateral`,
  `enable_guided_filter`, `enable_dense_crf`, `enable_edge_snap`) and `cascade_passes` (repeat the edge stages with
  halved radii).
- Finish: `feather_sigma` (blur the edge), `gamma` (above 1 thins semi-transparent edges), `threshold` (above 0 cuts
  a hard 0 / 1 mask).
- Power stages (folded): domain transform, colour decontamination, unsharp alpha, anti-alias, chroma lock, speck
  removal, temporal smooth (video only), and `advanced_overrides_json` to set any preset number, for example
  `{"gf_radius": 12, "crf_iterations": 8}`. Your JSON wins over the preset and the subject profile.
- Integrity (folded): `enable_integrity_check` adds per-frame coverage and drift stats to `info`.

Every input with its tooltip: [NODE_REFERENCE](../NODE_REFERENCE.md#maskrefinemec).

### The QC viewer

After a run, the node shows the C2C Matte viewer in QC mode: **Plate → Render**, the plate with the refined matte,
scrubbable, with the **Comp / Alpha / Checker / Overlay** views and a wipe (drag on the frame). Three chips grade the
result green / amber / red, measured against the mask you fed in:

- **Coverage**: share of the input mask that the refined matte still covers;
- **Edge detail**: edge sharpness of the refined matte compared with the input mask;
- **Flicker**: frame-to-frame change of the matte edge after following the motion.

The thresholds were measured on synthetic clips ([design/matte_viewer.md](design/matte_viewer.md)), not on your
footage. The viewer shows at most 72 proxy frames at 640 px and measures QC on at most 16 frame pairs, so a long clip
costs little extra. It saves nothing in the workflow and never changes the outputs. Turn it off with the C2C setting
**Matte viewer** (Settings → C2C → Masks) or `C2C_VIEWER=0`. The same viewer appears on Mask Track.

### Reading the outputs

| Output | What it is |
|---|---|
| `mask` | The refined mask, after `threshold` |
| `alpha` | The same before `threshold` (soft) |
| `preview` | The image times `alpha` |
| `info` | JSON report |

`info` fields:
- `ran`: the stages that ran, with their numbers (for example `feather_s0.60`, `threshold@0.50`);
- `skipped`: stages switched on but skipped, with the missing package (`dense_crf(missing:pydensecrf)`);
- `auto_edge_lock`: the subject class used, or `null`;
- `deps`: which optional packages are installed;
- `input_sum`, `output_sum`: mask area before and after, to spot a refine that ate the subject;
- `overrides_ignored`: keys in `advanced_overrides_json` the node does not know;
- `integrity` (with the check on): `flagged_count`, `flagged_frames` and per frame `area`, `area_ratio`,
  `centroid_delta`, `iou_prev` and `warn` (mask dropout, area drop, centroid jump, low IoU).
  `integrity_drop_threshold` and `integrity_jump_threshold` set when a frame is flagged.

### Common mistakes

| What you see | Why | What to do |
|---|---|---|
| Turning off the guided filter or edge snap changes nothing | `auto_edge_lock` forces both on | Turn `auto_edge_lock` off |
| `morph_op` has no effect | `subject_class` `face`, `hair` or `hard_surface` turns morphology off | Use `general` or `garment`, or turn the lock off |
| The mask comes out hard | `subject_class` = `hard_surface` cuts at 0.5 when `threshold` is 0 | Pick another subject, or use `alpha` |
| A stage is listed in `skipped` | Its Python package is missing | Install the package named there |
| "advanced_overrides_json is not valid JSON" / "must be a JSON object" | The text is not a JSON object | Write it like `{"gf_radius": 12}` |
| `enable_temporal_smooth` does nothing | It needs more than one frame | Feed a video batch |
| "MaskRefineMEC expects IMAGE tensor [B,H,W,C]" / "MASK tensor [H,W] or [B,H,W]" | Wrong input type on `image` or `mask` | Wire an IMAGE and a MASK |
