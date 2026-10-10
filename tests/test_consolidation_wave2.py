"""L7.65 consolidation wave 2: every merged node gives the same results through its successor.

Each case feeds the OLD node's inputs through its row in nodes/_legacy_replacements.json exactly as the workflow
migration does (renames, set_value, c2c_values translations), runs the successor, and compares every mapped output
with what the old node returned. The old nodes are gone, so their logic is kept below as reference copies of the last
shipped version (git 2785b91) - short functions, verbatim.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

PACK = Path(__file__).resolve().parents[1]
TABLE = {r["old_node_id"]: r for r in json.loads((PACK / "nodes" / "_legacy_replacements.json").read_text("utf-8"))}


# ── reference copies of the removed nodes (2785b91) ──────────────────────────────────────────────────────────────

def old_dimensions_snap(width, height, multiple, direction):
    m = max(1, int(multiple))

    def _snap(v):
        if direction == "down":    return max(m, (v // m) * m)
        if direction == "up":      return max(m, ((v + m - 1) // m) * m)
        return max(m, int(round(v / m)) * m)
    return (int(_snap(width)), int(_snap(height)))


def old_batch_split(images, mode, index, ratio):
    b = images.shape[0]
    if mode == "ratio":
        cut = max(0, min(b, int(round(b * ratio))))
    else:
        cut = max(0, min(b, int(index)))
    a, r = images[:cut].contiguous(), images[cut:].contiguous()
    return (a, r, int(a.shape[0]), int(r.shape[0]))


def old_frame_extractor(images, frame_index=0, mode="first"):
    B = images.shape[0]
    is_video = B > 1
    if mode == "first":
        idx = 0
    elif mode == "last":
        idx = B - 1
    elif mode == "middle":
        idx = B // 2
    else:  # specific_frame
        idx = min(frame_index, B - 1)
    frame = images[idx].unsqueeze(0)
    return (frame, B, is_video)


def old_mask_area_probe(mask, threshold):
    t = mask
    if t.ndim == 2:
        t_b = t.unsqueeze(0)
    elif t.ndim == 3:
        t_b = t
    else:
        t_b = t.reshape(-1, *t.shape[-2:])
    per = (t_b > float(threshold)).float().mean(dim=(-2, -1)) * 100.0
    cmin = float(per.min().item())
    cmax = float(per.max().item())
    cmean = float(per.mean().item())
    report = (f"frames={t_b.shape[0]} thr={threshold:.2f} "
              f"coverage mean={cmean:.2f}% min={cmin:.2f}% max={cmax:.2f}%")
    return (mask, report, cmean, cmin, cmax)


# ── the migration, in Python (mirrors js/_c2c_migrate_core.js for values) ───────────────────────────────────────

def migrate_kwargs(old_id: str, old: dict) -> dict:
    row = TABLE[old_id]
    new = {}
    for m in row["input_mapping"]:
        if "set_value" in m:
            new[m["new_id"]] = m["set_value"]
        elif m["old_id"] in old:
            new[m["new_id"]] = old[m["old_id"]]
    for v in row.get("c2c_values") or []:
        if "set" in v:
            new[v["new_id"]] = v["set"]
            continue
        if v["old_id"] not in old:
            continue
        x = old[v["old_id"]]
        if "map" in v:
            x = v["map"][("true" if x else "false") if isinstance(x, bool) else str(x)]   # JS String(true) = "true"
        if v.get("fn") == "log2":
            x = math.log2(float(x))
        if v.get("clamp"):
            x = min(v["clamp"][1], max(v["clamp"][0], x))
        new[v["new_id"]] = x
    return new


def same(a, b):
    if a is b:
        return True
    if torch.is_tensor(a) or torch.is_tensor(b):
        return (torch.is_tensor(a) and torch.is_tensor(b) and a.shape == b.shape
                and torch.allclose(a, b, rtol=0.0, atol=0.0, equal_nan=True))      # bit-equal, NaN == NaN
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    return a == b


def check(old_id, old_kwargs, old_outputs, run_new):
    new_kwargs = migrate_kwargs(old_id, old_kwargs)
    new_outputs = run_new(**new_kwargs)
    for m in TABLE[old_id]["output_mapping"]:
        a, b = old_outputs[m["old_idx"]], new_outputs[m["new_idx"]]
        assert same(a, b), f"{old_id} output {m['old_idx']} -> {m['new_idx']}: {a!r} != {b!r}"


CLIP = torch.rand(9, 6, 5, 3, generator=torch.Generator().manual_seed(1))


@pytest.mark.parametrize("w,h,mult,direction", [(1000, 600, 64, "down"), (1000, 600, 64, "up"),
                                                (1000, 600, 64, "nearest"), (8, 9, 16, "down"), (1919, 1081, 8, "nearest")])
def test_dimensions_snap_through_size(w, h, mult, direction):
    from nodes.helpers.helpers import AspectPresetMEC
    old = {"width": w, "height": h, "multiple": mult, "direction": direction}
    check("DimensionsSnapMEC", old, old_dimensions_snap(**old),
          lambda **k: AspectPresetMEC().pick(base=1024, **k))


def test_size_presets_unchanged_for_saved_aspect_presets():
    """The survivor's own saved values ([preset, base, multiple]) give what Aspect Ratio Preset gave."""
    from nodes.helpers.helpers import AspectPresetMEC
    n = AspectPresetMEC()
    assert n.pick("16:9 landscape", 1024, 64) == (1024, 576)
    assert n.pick("Wan 480p land", 0, 64) == (832, 448)
    assert n.pick("9:16 portrait", 1280, 16) == (720, 1280)


@pytest.mark.parametrize("mode,index,ratio", [("index", 3, 0.5), ("index", 0, 0.5), ("index", 99, 0.5),
                                              ("ratio", 0, 0.3), ("ratio", 0, 1.0), ("ratio", 0, 0.0)])
def test_batch_split_through_batch_range(mode, index, ratio):
    from nodes.helpers.helpers import ImageBatchSliceMEC
    old = {"images": CLIP, "mode": mode, "index": index, "ratio": ratio}
    check("ImageBatchSplitMEC", old, old_batch_split(**old),
          lambda **k: ImageBatchSliceMEC().slice(start=0, end=-1, step=1, **k))


@pytest.mark.parametrize("mode,frame_index", [("first", 0), ("last", 0), ("middle", 0), ("specific_frame", 4),
                                              ("specific_frame", 50)])
@pytest.mark.parametrize("batch", [CLIP, CLIP[:1]])
def test_frame_extractor_through_batch_range(mode, frame_index, batch):
    from nodes.helpers.helpers import ImageBatchSliceMEC
    old = {"images": batch, "frame_index": frame_index, "mode": mode}
    check("VideoFrameExtractorMEC", old, old_frame_extractor(**old),
          lambda **k: ImageBatchSliceMEC().slice(start=0, end=-1, step=1, **k))


def test_batch_range_saved_slices_unchanged():
    from nodes.helpers.helpers import ImageBatchSliceMEC
    out, count, rest, rest_count, total, is_video = ImageBatchSliceMEC().slice(CLIP, 1, -2, 2)
    assert torch.equal(out, CLIP[1:8:2]) and count == 4 and rest_count == 0 and total == 9 and is_video


@pytest.mark.parametrize("threshold", [0.0, 0.5, 0.99])
@pytest.mark.parametrize("shape", [(3, 8, 8), (8, 8), (2, 2, 8, 8)])
def test_mask_area_probe_through_probe(threshold, shape):
    from nodes.helpers.helpers import ImageStatsProbeMEC
    mask = torch.rand(*shape, generator=torch.Generator().manual_seed(2))
    old = {"mask": mask, "threshold": threshold}
    check("MaskAreaProbeMEC", old, old_mask_area_probe(**old), lambda **k: ImageStatsProbeMEC().probe(**k))


def test_latent_inspector_through_probe():
    """Its body moved verbatim into nodes.vae_latent_inspector.inspect_latent; outputs land at 9-13."""
    from nodes.helpers.helpers import ImageStatsProbeMEC
    from nodes.vae_latent_inspector import inspect_latent
    t = torch.randn(1, 4, 6, 6, generator=torch.Generator().manual_seed(3))
    t[0, 0, 0, 0] = float("nan")
    lat = {"samples": t}
    info, verdict, nans, infs = inspect_latent(lat)
    old = (lat, info, verdict, nans, infs)
    check("VAELatentInspectorMEC", {"latent": lat, "fail_on_corrupt": False}, old,
          lambda **k: ImageStatsProbeMEC().probe(**k))
    assert verdict == "corrupt" and nans == 1
    with pytest.raises(ValueError, match="NaN=1"):
        ImageStatsProbeMEC().probe(**migrate_kwargs("VAELatentInspectorMEC", {"latent": lat, "fail_on_corrupt": True}))


def test_probe_needs_an_input():
    from nodes.helpers.helpers import ImageStatsProbeMEC
    with pytest.raises(ValueError, match="connect an image, a mask or a latent"):
        ImageStatsProbeMEC().probe()


def test_vae_similarity_through_vae_inspect():
    from nodes.model_analysis import VAEBlockInspectorMEC, _compare

    class V:
        def __init__(self, sd):
            self._sd = sd

        def state_dict(self):
            return self._sd

    g = torch.Generator().manual_seed(4)
    a = V({"decoder.conv_in.weight": torch.randn(4, 4, generator=g), "encoder.conv_in.weight": torch.randn(3, generator=g)})
    b = V({"decoder.conv_in.weight": torch.randn(4, 4, generator=g), "encoder.conv_in.weight": torch.randn(3, generator=g)})
    for per_tensor in (False, True):
        old = {"vae_a": a, "vae_b": b, "include_per_tensor": per_tensor}
        check("VAESimilarityAnalyserMEC", old, _compare(a, b, per_tensor),
              lambda **k: VAEBlockInspectorMEC().inspect(**k))
    with pytest.raises(ValueError, match="vae_b"):
        VAEBlockInspectorMEC().inspect(a, mode="compare")


@pytest.mark.parametrize("args", [
    {"balance_mode": "off", "balance_strength": 1.0, "saturation": 1.0, "contrast_restore": 0.0, "chroma_cleanup": 0.0},
    {"balance_mode": "grey world", "balance_strength": 0.7, "saturation": 0.8, "contrast_restore": 0.3,
     "chroma_cleanup": 0.4},
    {"balance_mode": "white point", "balance_strength": 1.0, "saturation": 1.2, "contrast_restore": 0.0,
     "chroma_cleanup": 0.0, "restore_unchanged": 0.0, "restore_radius": 8},
])
def test_vae_clean_through_vae_decode(args):
    from nodes.hdr_color_science import C2CVAEQualityDecode
    from nodes.vae_clean import VAECleanMEC
    img = torch.rand(2, 16, 16, 3, generator=torch.Generator().manual_seed(5)) * 0.8 + 0.1
    img[..., 0] *= 1.15                                                      # a red cast to correct
    old = {"image": img, **args}
    check("VAECleanMEC", old, VAECleanMEC().clean(**old), lambda **k: C2CVAEQualityDecode().decode(**k))


def test_vae_clean_match_reference_through_vae_decode():
    from nodes.hdr_color_science import C2CVAEQualityDecode
    from nodes.vae_clean import VAECleanMEC
    g = torch.Generator().manual_seed(6)
    img, ref = torch.rand(1, 16, 16, 3, generator=g), torch.rand(1, 16, 16, 3, generator=g)
    old = {"image": img, "reference": ref, "balance_mode": "match reference", "balance_strength": 1.0,
           "saturation": 1.0, "contrast_restore": 0.0, "chroma_cleanup": 0.0, "restore_unchanged": 0.05,
           "restore_radius": 4}
    check("VAECleanMEC", old, VAECleanMEC().clean(**old), lambda **k: C2CVAEQualityDecode().decode(**k))


def test_vae_decode_refuses_a_latent_and_an_image_together():
    from nodes.hdr_color_science import C2CVAEQualityDecode
    with pytest.raises(ValueError, match="not both"):
        C2CVAEQualityDecode().decode(samples={"samples": torch.zeros(1, 4, 2, 2)}, vae=object(),
                                     image=torch.zeros(1, 4, 4, 3))
    with pytest.raises(ValueError, match="connect a latent"):
        C2CVAEQualityDecode().decode()


def test_helpers_smoke_script_passes():
    """nodes/helpers/_test_smoke.py was never run by the suite - which is how Seed List's "hash" mode shipped with a
    NameError (fixed in L7.65 wave 2). Run it here."""
    import subprocess
    import sys
    r = subprocess.run([sys.executable, str(PACK / "nodes" / "helpers" / "_test_smoke.py")], capture_output=True,
                       text=True, encoding="utf-8", timeout=300, cwd=str(PACK))
    assert r.returncode == 0 and "ALL 9 HELPER NODES PASSED" in r.stdout, (r.stdout[-1500:], r.stderr[-1500:])
