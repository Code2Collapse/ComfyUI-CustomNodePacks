"""L7.65 P08: Smart Image Crop and Inpaint Mask Prepare became modes of Inpaint Crop Pro; Smart Image Stitch, Inpaint
Paste Back and Context Inpainter became modes of Inpaint Stitch Pro. Same method as test_consolidation_wave2.py: the
old node's inputs go through its row in nodes/_legacy_replacements.json (as the workflow migration moves them) into
the survivor, and every mapped output is compared bit for bit with the old node's own output.

The reference is the old code as shipped at d97cc13, read with `git show` into a temporary package. Where the history
is missing (a depth-1 CI clone) the in-tree engine classes are the reference instead, and
test_engines_are_the_shipped_code pins their source to d97cc13 by hash, so the two cannot drift apart unnoticed.
"""
from __future__ import annotations

import ast
import hashlib
import importlib
import json
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest
import torch

from tests.test_consolidation_wave2 import TABLE, check, migrate_kwargs, same

PACK = Path(__file__).resolve().parents[1]
REF_COMMIT = "d97cc13"
P08_ROWS = ("SmartImageCropMEC", "InpaintMaskPrepareMEC", "SmartImageStitcherMEC", "InpaintPasteBackMEC",
            "MECContextInpainter")
# sha256 of each engine class's source at d97cc13 (ast.get_source_segment); computed 2026-10-15 from `git show`
ENGINE_HASHES = {
    ("nodes/smart_crop.py", "SmartImageCropMEC"): "82543a8b4ef8f8bbdd3c51518d3f8c1893f8aa09ff218ee6627e24378b5c49f0",
    ("nodes/smart_crop.py", "SmartImageStitcherMEC"):
        "e0c68ce654033ccad26bda2846b0dfd6b8664f4918110fe607f9f3e6a2a0a9d3",
    ("nodes/inpaint_suite.py", "InpaintMaskPrepareMEC"):
        "a90ba3c28e576288261f335f5b39eb197de66e04306d1bfe9dcbc0d43199b160",
    ("nodes/inpaint_suite.py", "InpaintPasteBackMEC"):
        "ac567d8ea2da79c93adb75f48a0956cd4241946bd862c78b00d92fa7fa1cf4e6",
    ("nodes/mec_paint_suite.py", "MECContextInpainter"):
        "9b7e5649993d14a3e69847223aa0ecd8e3f3fa474836efb3e64947adbddd84f1",
}


@pytest.fixture(autouse=True)
def temp_dir(monkeypatch, tmp_path):
    """The preview PNGs go to ComfyUI's temp folder; the test stub of folder_paths has none."""
    fp = sys.modules.get("folder_paths")
    if fp is not None:
        monkeypatch.setattr(fp, "get_temp_directory", lambda: str(tmp_path), raising=False)
    return tmp_path


def _load_reference(tmp: Path):
    """The removed nodes as shipped at d97cc13 (a temp package), or the in-tree engines when git has no history."""
    pkg = tmp / "nodes_ref_d97cc13"
    try:
        pkg.mkdir(parents=True, exist_ok=True)
        for name in ("smart_crop.py", "inpaint_suite.py", "mec_paint_suite.py", "_is_changed_util.py"):
            text = subprocess.run(["git", "show", f"{REF_COMMIT}:nodes/{name}"], cwd=str(PACK), check=True,
                                  capture_output=True, text=True, encoding="utf-8", timeout=60).stdout
            (pkg / name).write_text(text, encoding="utf-8")
        (pkg / "__init__.py").write_text("", encoding="utf-8")
        sys.path.insert(0, str(tmp))
        try:
            mods = {n: importlib.import_module(f"nodes_ref_d97cc13.{n}")
                    for n in ("smart_crop", "inpaint_suite", "mec_paint_suite")}
        finally:
            sys.path.remove(str(tmp))
        source = f"git {REF_COMMIT}"
    except (subprocess.SubprocessError, OSError):
        from nodes import inpaint_suite, mec_paint_suite, smart_crop
        mods = {"smart_crop": smart_crop, "inpaint_suite": inpaint_suite, "mec_paint_suite": mec_paint_suite}
        source = "in-tree engines"
    return types.SimpleNamespace(
        source=source,
        SmartImageCropMEC=mods["smart_crop"].SmartImageCropMEC,
        SmartImageStitcherMEC=mods["smart_crop"].SmartImageStitcherMEC,
        InpaintMaskPrepareMEC=mods["inpaint_suite"].InpaintMaskPrepareMEC,
        InpaintPasteBackMEC=mods["inpaint_suite"].InpaintPasteBackMEC,
        MECContextInpainter=mods["mec_paint_suite"].MECContextInpainter,
    )


@pytest.fixture(scope="module")
def ref(tmp_path_factory):
    r = _load_reference(tmp_path_factory.mktemp("ref"))
    print(f"P08 reference: {r.source}")
    yield r
    for k in [m for m in sys.modules if m.startswith("nodes_ref_d97cc13")]:
        sys.modules.pop(k, None)


def deep_same(a, b):
    """same() of test_consolidation_wave2, also through lists and tuples (a Pro stitcher holds lists of tensors)."""
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(deep_same(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(deep_same(a[k], b[k]) for k in a)
    return same(a, b)


def run_crop(**k):
    from nodes.inpaint_suite import InpaintCropProMEC
    return InpaintCropProMEC().inpaint_crop(**k)["result"]


def run_stitch(**k):
    from nodes.inpaint_suite import InpaintStitchProMEC
    return InpaintStitchProMEC().inpaint_stitch(**k)


def through_successor(old_id: str, **old):
    """What a migrated node of type ``old_id`` returns, in the OLD node's output order (for the tests that were
    written against the removed classes)."""
    row = TABLE[old_id]
    out = (run_crop if row["new_node_id"] == "InpaintCropProMEC" else run_stitch)(**migrate_kwargs(old_id, old))
    by_old = {m["old_idx"]: m["new_idx"] for m in row["output_mapping"]}
    return tuple(out[by_old[i]] for i in range(len(by_old)))


# ── the reference itself ──────────────────────────────────────────────────────────────────────────────────────────

def test_engines_are_the_shipped_code():
    """The survivors call these classes; their source is the d97cc13 source, byte for byte."""
    for (rel, name), want in ENGINE_HASHES.items():
        text = (PACK / rel).read_text(encoding="utf-8")
        node = next(n for n in ast.parse(text).body if isinstance(n, ast.ClassDef) and n.name == name)
        got = hashlib.sha256(ast.get_source_segment(text, node).encode("utf-8")).hexdigest()
        assert got == want, f"{rel}:{name} differs from {REF_COMMIT} - update the reference deliberately"


def test_the_removed_ids_are_not_registered_and_have_rows():
    from nodes import mec_paint_suite, smart_crop
    assert not smart_crop.NODE_CLASS_MAPPINGS
    assert "MECContextInpainter" not in mec_paint_suite.NODE_CLASS_MAPPINGS
    init = (PACK / "__init__.py").read_text(encoding="utf-8")
    for old in P08_ROWS:
        assert f'"{old}":' not in init, old
        assert old in TABLE
    assert {TABLE[o]["new_node_id"] for o in P08_ROWS} == {"InpaintCropProMEC", "InpaintStitchProMEC"}


# ── shared plates ─────────────────────────────────────────────────────────────────────────────────────────────────

def plate(B=1, h=96, w=160, seed=0):
    g = torch.Generator().manual_seed(seed)
    yy = torch.linspace(0, 1, h).view(1, h, 1, 1).expand(B, h, w, 1)
    xx = torch.linspace(0, 1, w).view(1, 1, w, 1).expand(B, h, w, 1)
    return (torch.cat([xx, yy, xx * yy], -1) * 0.85 + torch.rand(B, h, w, 3, generator=g) * 0.15).clamp(0, 1)


def boxes(B, rects, h=96, w=160, value=1.0):
    m = torch.zeros(B, h, w)
    for b in range(B):
        x0, y0, x1, y1 = rects[b % len(rects)]
        m[b, y0:y1, x0:x1] = value
    return m


SMART_DEFAULTS = dict(resolution_mode="Automatic", max_resolution=2048, min_resolution=768, manual_width=1024,
                      manual_height=1024, mask_grow_pixels=32, patch_mask_holes=True, no_mask_mode="Bypass",
                      force_divisibility=128)


def _holed():
    m = boxes(1, [(40, 20, 100, 70)])
    m[0, 35:50, 60:75] = 0.0
    return m


SMART_CASES = [
    ("default", 1, lambda: boxes(1, [(50, 30, 90, 60)]), {}),
    ("manual", 1, lambda: boxes(1, [(50, 30, 90, 60)]), {"resolution_mode": "Manual", "manual_width": 640,
                                                          "manual_height": 384}),
    ("shrink", 1, lambda: boxes(1, [(40, 20, 120, 80)]), {"mask_grow_pixels": -8, "force_divisibility": 64}),
    ("faint", 1, lambda: boxes(1, [(50, 30, 90, 60)], value=3.0 / 255.0), {"patch_mask_holes": False}),
    ("holes", 1, _holed, {"mask_grow_pixels": 0, "min_resolution": 256, "max_resolution": 1024}),
    ("border", 1, lambda: boxes(1, [(0, 0, 25, 30)]), {"mask_grow_pixels": 8, "force_divisibility": 32}),
    ("batch", 3, lambda: boxes(3, [(10, 10, 40, 30), (60, 20, 150, 90), (0, 0, 0, 0)]), {"force_divisibility": 16}),
    ("empty-bypass", 2, lambda: torch.zeros(2, 96, 160), {}),
    ("empty-resize", 1, lambda: torch.zeros(1, 96, 160), {"no_mask_mode": "Resize Full Image"}),
    ("empty-resize-tall", 1, lambda: torch.zeros(1, 160, 96), {"no_mask_mode": "Resize Full Image",
                                                              "min_resolution": 256, "max_resolution": 512}),
    ("empty-crop", 1, lambda: torch.zeros(1, 96, 160), {"no_mask_mode": "Crop Full Image"}),
]


def _smart_old(case):
    _id, B, mk, over = case
    m = mk()
    img = plate(B=B, h=m.shape[1], w=m.shape[2])
    return {"image": img, "mask": m, **SMART_DEFAULTS, **over}


# ── Smart Image Crop -> Inpaint Crop Pro, crop_mode "smart aspect" ───────────────────────────────────────────────

@pytest.mark.parametrize("case", SMART_CASES, ids=[c[0] for c in SMART_CASES])
def test_smart_image_crop_through_inpaint_crop(ref, case):
    old = _smart_old(case)
    check("SmartImageCropMEC", old, ref.SmartImageCropMEC().crop(**old), run_crop)


@pytest.mark.parametrize("case", SMART_CASES, ids=[c[0] for c in SMART_CASES])
def test_plan_crop_is_the_engines_geometry(case):
    """The live preview's Smart geometry (no pixels) is the stitcher the engine returns, key for key."""
    from nodes.smart_crop import SmartImageCropMEC, plan_crop
    old = _smart_old(case)
    img, m = old.pop("image"), old.pop("mask")
    geometry, msk = plan_crop(m, tuple(img.shape[:3]), **old)
    st = SmartImageCropMEC().crop(img, m, **old)[2]
    assert same(geometry, st)
    assert msk.shape[1:] == m.shape[1:]


def test_smart_aspect_preview_shows_the_executed_windows():
    old = _smart_old(SMART_CASES[6])                                      # a batch, per-frame windows
    kw = migrate_kwargs("SmartImageCropMEC", old)
    from nodes.inpaint_suite import InpaintCropProMEC
    out = InpaintCropProMEC().inpaint_crop(preview_frame=1, **kw)
    st, crop = out["result"][0], out["result"][1]
    (plan,) = out["ui"]["c2c_crop_plan"]
    assert plan["mode"] == "smart aspect" and plan["frames_total"] == 3 and plan["stable"] is False
    assert [f["window"] for f in plan["frames"]] == [[st["x"][j], st["y"][j], st["w"][j], st["h"][j]]
                                                    for j in range(3)]
    assert plan["output_size"] == [int(crop.shape[2]), int(crop.shape[1])]
    assert out["ui"]["images"] and len(out["ui"]["images"]) == 1


def test_smart_aspect_without_a_mask_is_an_empty_mask():
    img = plate()
    st, crop, *_ = run_crop(image=img, crop_mode="smart aspect")
    assert st["bypass"] is True and torch.equal(crop, img)
    with pytest.raises(ValueError, match="connect an image"):
        run_crop(crop_mode="smart aspect", mask=boxes(1, [(1, 1, 5, 5)]))


# ── Inpaint Mask Prepare -> Inpaint Crop Pro, crop_mode "mask prepare" ───────────────────────────────────────────

PREPARE_DEFAULTS = dict(fill_holes=True, remove_small_regions=True, min_region_area=100, grow_pixels=4,
                        inpaint_edge_mode="hard_binary", stitch_edge_mode="gaussian", stitch_feather_radius=16,
                        temporal_smooth=False, temporal_sigma=1.5)


def _noisy_mask(B=4, h=64, w=80, seed=3):
    g = torch.Generator().manual_seed(seed)
    m = boxes(B, [(10, 10, 50, 40), (14, 12, 54, 44), (18, 8, 58, 40), (12, 14, 52, 48)], h=h, w=w)
    m[:, 20:26, 24:30] = 0.0                                              # a hole
    m[:, 55:58, 70:73] = 1.0                                              # a speck
    return (m + torch.rand(B, h, w, generator=g) * 0.2).clamp(0, 1)


PREPARE_CASES = [
    ("default", {}, False),
    ("edge-aware", {"stitch_edge_mode": "edge_aware", "stitch_feather_radius": 9}, True),
    ("temporal", {"temporal_smooth": True, "temporal_sigma": 2.0, "inpaint_edge_mode": "slight_feather"}, False),
    ("raw", {"fill_holes": False, "remove_small_regions": False, "grow_pixels": 0}, True),
    ("small-area", {"min_region_area": 5, "grow_pixels": 11}, False),
]


@pytest.mark.parametrize("case", PREPARE_CASES, ids=[c[0] for c in PREPARE_CASES])
def test_mask_prepare_through_inpaint_crop(ref, case):
    _id, over, with_ref = case
    m = _noisy_mask()
    old = {"mask": m, **PREPARE_DEFAULTS, **over}
    if with_ref:
        old["reference_image"] = plate(B=4, h=64, w=80, seed=5)
    check("InpaintMaskPrepareMEC", old, ref.InpaintMaskPrepareMEC().prepare_mask(**old), run_crop)


def test_mask_prepare_2d_mask_and_no_image():
    m = _noisy_mask(B=1)[0]
    st, cropped, im, bm, info, dbg = run_crop(mask=m, crop_mode="mask prepare")
    assert st is None and cropped is None and im.shape == (1, 64, 80) and dbg.shape == (1, 64, 80, 3)
    with pytest.raises(ValueError, match="connect a mask"):
        run_crop(image=plate(), crop_mode="mask prepare")


def test_mask_prepare_stitches_back_over_the_whole_frame():
    """With an image, mask prepare hands Inpaint Stitch Pro a whole-frame stitcher: an unchanged image comes back
    unchanged, a new one lands where the prepared blend mask says."""
    img = plate(B=2, h=64, w=80)
    m = _noisy_mask(B=2)
    st, cropped, im, bm, _info, _dbg = run_crop(image=img, mask=m, crop_mode="mask prepare")
    assert cropped is img
    out, used, _i = run_stitch(stitcher=st, inpainted_image=img)
    assert torch.allclose(out, img, atol=1e-6)
    assert torch.allclose(used, bm, atol=1e-6)
    white, _u, _i = run_stitch(stitcher=st, inpainted_image=torch.ones_like(img))
    a = bm.unsqueeze(-1)
    assert torch.allclose(white, a + (1.0 - a) * img, atol=1e-6)            # alpha over the plate, by the mask
    assert torch.equal(white[bm == 0], img[bm == 0])


# ── Smart Image Stitch -> Inpaint Stitch Pro, stitch_mode "smart" ────────────────────────────────────────────────

def _smart_stitcher(ref, case_index=0, **over):
    old = _smart_old(SMART_CASES[case_index])
    old.update(over)
    crop, crop_mask, st, _ov, _info = ref.SmartImageCropMEC().crop(**old)
    return old["image"], crop, crop_mask, st


STITCH_CASES = [
    ("box", 0, {"feather_pixels": 32, "blend_mode": "Box Feather"}, False, {}),
    ("mask-feather", 0, {"feather_pixels": 12, "blend_mode": "Mask Feather"}, True, {}),
    ("mask-feather-no-mask", 0, {"feather_pixels": 12, "blend_mode": "Mask Feather"}, False, {}),
    ("hard", 5, {"feather_pixels": 0, "blend_mode": "Hard Paste"}, False, {}),
    ("colour", 0, {"feather_pixels": 8, "blend_mode": "Box Feather", "enable_color_match": True,
                   "color_match_amount": 0.7}, True, {}),
    ("batch", 6, {"feather_pixels": 16, "blend_mode": "Mask Feather"}, True, {}),
    ("bypass", 7, {"feather_pixels": 8}, False, {}),
    ("full-restore", 8, {"resize_full_image_output": "Restore Original Size"}, False, {}),
    ("full-keep", 8, {"resize_full_image_output": "Keep Resized Image"}, False, {}),
]


@pytest.mark.parametrize("case", STITCH_CASES, ids=[c[0] for c in STITCH_CASES])
def test_smart_image_stitch_through_inpaint_stitch(ref, case):
    _id, ci, over, with_mask, _x = case
    img, crop, crop_mask, st = _smart_stitcher(ref, ci)
    processed = (crop * 0.4 + 0.3).clamp(0, 1)
    old = {"original_image": img, "processed_image": processed, "stitcher": st, "feather_pixels": 32,
           "blend_mode": "Box Feather", "resize_full_image_output": "Restore Original Size",
           "enable_color_match": False, "color_match_amount": 0.35, **over}
    if with_mask:
        old["mask"] = crop_mask
    check("SmartImageStitcherMEC", old, ref.SmartImageStitcherMEC().stitch(**old), run_stitch)


@pytest.mark.parametrize("case", [c for c in STITCH_CASES if "colour" not in c[0]],
                         ids=[c[0] for c in STITCH_CASES if "colour" not in c[0]])
def test_smart_stitch_blend_mask_used_is_the_applied_weight(ref, case):
    """A white crop over a black plate comes back as exactly the weight the stitch applied: blend_mask_used."""
    _id, ci, over, with_mask, _x = case
    img, crop, crop_mask, st = _smart_stitcher(ref, ci)
    black = torch.zeros_like(img)
    kw = {"stitch_mode": "smart", "stitcher": st, "original_image": black, "inpainted_image": torch.ones_like(crop),
          **{"smart_" + k: v for k, v in over.items()}}
    if with_mask:
        kw["mask"] = crop_mask
    out, used, _info = run_stitch(**kw)
    assert torch.equal(out[..., 0], used)


# ── Inpaint Paste Back -> Inpaint Stitch Pro, stitch_mode "paste back" ───────────────────────────────────────────

def _pro_crop(B=1, **over):
    from tests.test_inpaint_crop_geometry import DEFAULTS
    img = plate(B=B)
    m = boxes(B, [(40, 20, 90, 60), (60, 30, 120, 80), (0, 0, 30, 40)])
    st, cropped, *_ = run_crop(image=img, mask=m, **{**DEFAULTS, **over})
    return img, st, cropped


PASTE_CASES = [
    ("bicubic", 1, {"upscale_method": "bicubic", "feather_edges": False, "feather_radius": 16}, {}),
    ("lanczos-feather", 1, {"upscale_method": "lanczos", "feather_edges": True, "feather_radius": 12}, {}),
    ("area-feather-0", 1, {"upscale_method": "area", "feather_edges": True, "feather_radius": 0}, {}),
    ("per-frame", 3, {"upscale_method": "bilinear", "feather_edges": True, "feather_radius": 6},
     {"wan_stable_crop": False}),
    ("no-resize", 1, {"upscale_method": "nearest", "feather_edges": False, "feather_radius": 16},
     {"output_resize_to_target_size": False}),
]


@pytest.mark.parametrize("case", PASTE_CASES, ids=[c[0] for c in PASTE_CASES])
def test_paste_back_through_inpaint_stitch(ref, case):
    _id, B, over, crop_over = case
    img, st, cropped = _pro_crop(B, **crop_over)
    old = {"stitcher": st, "inpainted_image": (cropped * 0.5 + 0.25).clamp(0, 1), **over}
    check("InpaintPasteBackMEC", old, ref.InpaintPasteBackMEC().paste_back(**old), run_stitch)


def test_paste_back_one_crop_for_a_batch(ref):
    img, st, cropped = _pro_crop(1)
    inp = torch.cat([cropped, cropped * 0.5, cropped * 0.2], 0)
    old = {"stitcher": st, "inpainted_image": inp, "upscale_method": "bicubic", "feather_edges": True,
           "feather_radius": 8}
    check("InpaintPasteBackMEC", old, ref.InpaintPasteBackMEC().paste_back(**old), run_stitch)


@pytest.mark.parametrize("case", PASTE_CASES, ids=[c[0] for c in PASTE_CASES])
def test_paste_back_blend_mask_used_is_the_applied_alpha(case):
    _id, B, over, crop_over = case
    img, st, cropped = _pro_crop(B, **crop_over)
    black = dict(st, canvas_image=[torch.zeros_like(c) for c in st["canvas_image"]])
    out, used, _info = run_stitch(stitcher=black, inpainted_image=torch.ones_like(cropped), stitch_mode="paste back",
                                  **{"paste_" + k: v for k, v in over.items()})
    assert used.shape == out.shape[:3]
    assert torch.allclose(out[..., 0], used, atol=1e-6)


# ── Context Inpainter -> Inpaint Stitch Pro, stitch_mode "context blend" ─────────────────────────────────────────

CONTEXT_DEFAULTS = dict(crop_padding=1.2, blend_softness=8.0, mask_expansion_blend=0, enable_color_correction=True,
                        enable_lightness_rescue=True, enable_differential_diffusion=False, sampling_mask_blur_size=21,
                        sampling_mask_blur_strength=1.0)
CONTEXT_CASES = [
    ("default", {}, 1, False),
    ("dark-rescue", {"enable_color_correction": False}, 1, False),
    ("diff-diffusion", {"enable_differential_diffusion": True, "mask_expansion_blend": 5}, 1, False),
    ("erode-hard", {"mask_expansion_blend": -3, "blend_softness": 0.0, "sampling_mask_blur_size": 0}, 1, False),
    ("resized-batch", {"crop_padding": 1.8, "sampling_mask_blur_strength": 0.4}, 3, True),
    ("prompts", {"face_positive_prompt": "a [SEP] b", "face_negative_prompt": "c"}, 1, False),
]


@pytest.mark.parametrize("case", CONTEXT_CASES, ids=[c[0] for c in CONTEXT_CASES])
def test_context_inpainter_through_inpaint_stitch(ref, case):
    _id, over, B, other_size = case
    orig = plate(B=B, h=72, w=96, seed=7)
    fake = (orig * 0.55).clamp(0, 1) if not other_size else (plate(B=B, h=48, w=64, seed=8) * 0.5)
    m = boxes(B, [(20, 15, 60, 50)], h=72, w=96)
    old = {"original_image": orig, "mask": m, "inpainted_image": fake, **CONTEXT_DEFAULTS, **over}
    check("MECContextInpainter", old, ref.MECContextInpainter().execute(**old), run_stitch)


def test_context_blend_says_what_it_did():
    orig = plate(B=2, h=40, w=40)
    out, used, info = run_stitch(stitch_mode="context blend", original_image=orig, mask=boxes(2, [(5, 5, 20, 20)], 40, 40),
                                 inpainted_image=orig * 0.5, stitcher={"x": 1})
    assert out.shape[0] == 1 and "2 given, frame 0 blended" in info and "not used in this mode" in info
    for missing in ("original_image", "mask", "inpainted_image"):
        kw = {"stitch_mode": "context blend", "original_image": orig, "mask": torch.zeros(1, 40, 40),
              "inpainted_image": orig, missing: None}
        with pytest.raises(ValueError, match=f"connect {missing}"):
            run_stitch(**kw)


# ── the stitcher layouts: each mode says which one it reads ──────────────────────────────────────────────────────

def test_a_stitcher_of_the_other_layout_names_the_mode_to_use(ref):
    img, crop, _cm, smart_st = _smart_stitcher(ref, 0)
    _img, pro_st, cropped = _pro_crop(1)
    with pytest.raises(ValueError, match="set stitch_mode to 'smart'"):
        run_stitch(stitcher=smart_st, inpainted_image=crop)
    with pytest.raises(ValueError, match="set stitch_mode to 'smart'"):
        run_stitch(stitcher=smart_st, inpainted_image=crop, stitch_mode="paste back")
    with pytest.raises(ValueError, match="set stitch_mode to 'blend' or 'paste back'"):
        run_stitch(stitcher=pro_st, inpainted_image=cropped, stitch_mode="smart", original_image=img)
    with pytest.raises(ValueError, match="connect original_image"):
        run_stitch(stitcher=smart_st, inpainted_image=crop, stitch_mode="smart")
    with pytest.raises(ValueError, match="connect the stitcher"):
        run_stitch(inpainted_image=cropped)
    with pytest.raises(ValueError, match="connect inpainted_image"):
        run_stitch(stitcher=pro_st)
    with pytest.raises(ValueError, match="unknown stitch_mode"):
        run_stitch(stitcher=pro_st, inpainted_image=cropped, stitch_mode="glue")
    with pytest.raises(ValueError, match="unknown crop_mode"):
        run_crop(image=img, crop_mode="glue")
    with pytest.raises(ValueError, match="connect an image"):
        run_crop(mask=boxes(1, [(1, 1, 5, 5)]))


# ── the defaults are the old nodes: survivors' own saves are unchanged ──────────────────────────────────────────

def test_pro_and_blend_outputs_unchanged_and_preview_appended():
    """crop_mode pro / stitch_mode blend run the L7.99 code: five outputs as before plus the preview picture."""
    from tests.test_inpaint_crop_geometry import DEFAULTS
    img = plate(B=2)
    m = boxes(2, [(40, 20, 90, 60), (50, 25, 100, 70)])
    from nodes.inpaint_suite import InpaintCropProMEC
    out = InpaintCropProMEC().inpaint_crop(image=img, mask=m, **DEFAULTS)
    explicit = InpaintCropProMEC().inpaint_crop(image=img, mask=m, crop_mode="pro", **DEFAULTS)
    a, b = out["result"], explicit["result"]
    assert len(a) == 6 and all(deep_same(x, y) for x, y in zip(a[:5], b[:5]))
    preview = a[5]
    assert preview.ndim == 4 and preview.shape[0] == 1 and preview.shape[-1] == 3
    s1 = run_stitch(stitcher=a[0], inpainted_image=a[1])
    s2 = run_stitch(stitcher=a[0], inpainted_image=a[1], stitch_mode="blend")
    assert all(deep_same(x, y) for x, y in zip(s1, s2))


def test_mode_inputs_are_the_engines_widgets_and_append_after_the_old_ones():
    from nodes import inpaint_suite as S
    from nodes.mec_paint_suite import MECContextInpainter
    from nodes.smart_crop import SmartImageCropMEC, SmartImageStitcherMEC

    crop = S.InpaintCropProMEC.INPUT_TYPES()["optional"]
    names = list(crop)
    assert names[names.index("preview_frame") + 1:] == ["crop_mode", *S._SMART_CROP_KEYS, *S._PREPARE_KEYS]
    for keys, prefix, cls, skip in ((S._SMART_CROP_KEYS, "smart_", SmartImageCropMEC, {"image", "mask"}),
                                    (S._PREPARE_KEYS, "prepare_", S.InpaintMaskPrepareMEC, {"mask"})):
        old = {k: v for k, v in cls.INPUT_TYPES()["required"].items() if k not in skip}
        assert [prefix + k for k in old] == list(keys)
        for k, v in old.items():
            new = crop[prefix + k]
            assert list(new[0]) == list(v[0]) if isinstance(v[0], (list, tuple)) else new[0] == v[0]
            assert {a: b for a, b in new[1].items() if a != "tooltip"} == \
                   {a: b for a, b in v[1].items() if a != "tooltip"}
    stitch = S.InpaintStitchProMEC.INPUT_TYPES()
    assert list(stitch["required"]) == ["blend_mode_override", "color_match", "stitch_temporal_sigma",
                                        "stitch_dilate_px"]
    opt = list(stitch["optional"])
    assert opt[:3] == ["stitcher", "inpainted_image", "roto_quality_override"]
    assert opt[3:] == ["stitch_mode", "original_image", "mask", *S._PASTE_KEYS, *S._SMART_STITCH_KEYS,
                       *S._CONTEXT_KEYS]
    for keys, prefix, cls, skip in (
            (S._PASTE_KEYS, "paste_", S.InpaintPasteBackMEC, {"stitcher", "inpainted_image"}),
            (S._SMART_STITCH_KEYS, "smart_", SmartImageStitcherMEC, {"original_image", "processed_image", "stitcher"}),
            (S._CONTEXT_KEYS, "context_", MECContextInpainter, {"original_image", "mask", "inpainted_image"})):
        assert [prefix + k for k in cls.INPUT_TYPES()["required"] if k not in skip] == list(keys)
    # the live preview knows every crop setting a mode adds
    assert set(S._SMART_CROP_KEYS) | set(S._PREPARE_KEYS) | {"crop_mode", "roto_quality"} <= set(S._LIVE_KEYS)


# ── the browser migration (js/_c2c_migrate_core.js, run under Node) moves links and values to the right slots ─────

NODE = shutil.which("node")


def _frontend_shape(cls):
    """The successor as frontend 1.53.10 builds it (addInputs: every socket input first, in definition order, then
    one widget socket per widget), its widget names and default values."""
    spec = cls.INPUT_TYPES()
    items = [(n, s) for g in ("required", "optional") for n, s in (spec.get(g) or {}).items()]
    is_widget = lambda s: isinstance(s[0], (list, tuple)) or s[0] in ("INT", "FLOAT", "STRING", "BOOLEAN")
    sockets = [{"name": n, "type": s[0]} for n, s in items if not is_widget(s)]
    widgets = [n for n, s in items if is_widget(s)]
    wsock = [{"name": n, "type": "COMBO" if isinstance(s[0], (list, tuple)) else s[0], "widget": {"name": n}}
             for n, s in items if is_widget(s)]
    vals = []
    for n, s in items:
        if is_widget(s):
            o = s[1] if len(s) > 1 and isinstance(s[1], dict) else {}
            vals.append(o.get("default", s[0][0] if isinstance(s[0], (list, tuple)) else None))
    return {"inputs": sockets + wsock, "outputs": [{"name": n, "type": t} for n, t in zip(cls.RETURN_NAMES,
                                                                                      cls.RETURN_TYPES)],
            "widgets": widgets, "values": vals}


@pytest.mark.skipif(NODE is None, reason="node not on PATH")
@pytest.mark.parametrize("old_id", P08_ROWS)
def test_saved_node_migrates_with_its_links_and_values(old_id):
    from nodes.inpaint_suite import InpaintCropProMEC, InpaintStitchProMEC
    fixture = json.loads((PACK / "tests" / "fixtures_legacy_signatures.json").read_text("utf-8"))[old_id]
    row = TABLE[old_id]
    new_cls = InpaintCropProMEC if row["new_node_id"] == "InpaintCropProMEC" else InpaintStitchProMEC
    shape = _frontend_shape(new_cls)
    sockets = [i for i in fixture["inputs"] if not i["widget"]]
    widget_ids = [i["name"] for i in fixture["inputs"] if i["widget"]]
    old_vals = []
    for i in fixture["inputs"]:
        if i["widget"]:
            old_vals.append(i["options"][-1] if i["type"] == "COMBO" else
                            {"INT": 7, "FLOAT": 0.25, "BOOLEAN": True, "STRING": "kept"}[i["type"]])
    links, nodes = [], []
    src_outputs = []
    for k, s in enumerate(sockets):                                   # one source per old socket
        lid = 100 + k
        links.append([lid, 10 + k, 0, 1, k, s["type"]])
        nodes.append({"id": 10 + k, "type": "Src", "outputs": [{"name": "o", "type": s["type"], "links": [lid]}]})
        src_outputs.append(lid)
    old_outputs = []
    for k, t in enumerate(fixture["outputs"]):                        # one target per old output
        lid = 200 + k
        links.append([lid, 1, k, 50 + k, 0, t])
        nodes.append({"id": 50 + k, "type": "Dst", "inputs": [{"name": "i", "type": t, "link": lid}]})
        old_outputs.append({"name": f"out{k}", "type": t, "links": [lid]})
    nodes.append({"id": 1, "type": old_id, "inputs": [{"name": s["name"], "type": s["type"], "link": 100 + k}
                                                      for k, s in enumerate(sockets)],
                  "outputs": old_outputs, "widgets_values": old_vals, "properties": {}})
    wf = {"nodes": nodes, "links": links}
    script = f"""
import {{ migrateWorkflow }} from {json.dumps((PACK / "js" / "_c2c_migrate_core.js").as_uri())};
const wf = {json.dumps(wf)};
const shape = {json.dumps(shape)};
const r = migrateWorkflow(wf, {json.dumps([row])}, (t) => t !== {json.dumps(old_id)}, () => shape);
console.log(JSON.stringify({{ wf, r }}));
"""
    p = subprocess.run([NODE, "--input-type=module", "-e", script], capture_output=True, text=True,
                       encoding="utf-8", timeout=60)
    assert p.returncode == 0, p.stderr
    res = json.loads(p.stdout)
    assert not res["r"]["problems"], res["r"]["problems"]
    node = next(n for n in res["wf"]["nodes"] if n["id"] == 1)
    assert node["type"] == row["new_node_id"]
    names = [i["name"] for i in node["inputs"]]
    by_id = {link[0]: link for link in res["wf"]["links"]}
    for m in row["input_mapping"]:
        if "old_id" not in m:
            assert node["widgets_values"][shape["widgets"].index(m["new_id"])] == m["set_value"]
            continue
        if m["old_id"] in widget_ids:                                 # values move by position
            assert node["widgets_values"][shape["widgets"].index(m["new_id"])] == \
                   old_vals[widget_ids.index(m["old_id"])], m
        else:                                                         # links move to the new socket's slot
            k = [s["name"] for s in sockets].index(m["old_id"])
            slot = names.index(m["new_id"])
            assert node["inputs"][slot]["link"] == 100 + k and by_id[100 + k][4] == slot, m
    for m in row["output_mapping"]:
        assert node["outputs"][m["new_idx"]]["links"] == [200 + m["old_idx"]]
        assert by_id[200 + m["old_idx"]][2] == m["new_idx"]


# ── the live crop preview follows the new options and still equals the executed crop (L7.99) ─────────────────────

@pytest.fixture
def live():
    mod = pytest.importorskip("nodes._inpaint_preview_live")
    mod.clear()
    yield mod
    mod.clear()


def _run_live(img, mask, uid, **over):
    from tests.test_inpaint_crop_geometry import DEFAULTS
    from nodes.inpaint_suite import InpaintCropProMEC
    out = InpaintCropProMEC().inpaint_crop(img, mask=mask, unique_id=uid, **{**DEFAULTS, **over})
    return out["result"], out["ui"]


LIVE_CHANGES = [
    {"crop_mode": "smart aspect"},
    {"crop_mode": "smart aspect", "smart_min_resolution": 256, "smart_force_divisibility": 32},
    {"crop_mode": "smart aspect", "smart_resolution_mode": "Manual", "smart_manual_width": 512,
     "smart_manual_height": 256},
    {"crop_mode": "smart aspect", "smart_mask_grow_pixels": -8, "smart_patch_mask_holes": False},
    {"crop_mode": "mask prepare"},
    {"crop_mode": "mask prepare", "prepare_grow_pixels": 10, "prepare_remove_small_regions": False},
    {"roto_quality": True},
    {"roto_quality": True, "context_from_mask_extend_factor": 1.6},
]


@pytest.mark.parametrize("change", LIVE_CHANGES, ids=[",".join(f"{k}={v}" for k, v in c.items()) for c in LIVE_CHANGES])
@pytest.mark.parametrize("start", ["pro", "smart aspect", "mask prepare"])
def test_live_plan_equals_a_real_run_in_every_mode(live, start, change):
    """From a run in any mode, a changed mode or mode setting previews exactly the windows a run with it executes."""
    from tests.test_inpaint_crop_geometry import box
    img = plate(B=3, h=240, w=320)
    mask = box(3, 20, 60, 70, 140, dx=40)
    _res, ui0 = _run_live(img, mask, "7", crop_mode=start)
    assert ui0.get("c2c_crop_plan"), ui0
    got = live.render("7", change)
    assert got.get("images"), got
    _res, ui = _run_live(img, mask, "8", **{"crop_mode": start, **change})
    (want,) = ui["c2c_crop_plan"]
    assert got["plan"] == want


def test_live_smart_empty_mask_modes(live):
    img = plate(B=1, h=120, w=200)
    empty = torch.zeros(1, 120, 200)
    _run_live(img, empty, "7", crop_mode="smart aspect")
    for change in ({"smart_no_mask_mode": "Resize Full Image"}, {"smart_no_mask_mode": "Crop Full Image"},
                   {"smart_no_mask_mode": "Bypass"}):
        got = live.render("7", change)
        _res, ui = _run_live(img, empty, "8", crop_mode="smart aspect", **change)
        assert got["plan"] == ui["c2c_crop_plan"][0], change


def test_live_mask_prepare_without_an_image_stays_stale(live):
    from nodes.inpaint_suite import InpaintCropProMEC
    InpaintCropProMEC().inpaint_crop(mask=_noisy_mask(B=1), crop_mode="mask prepare", unique_id="9")
    assert live.render("9", {})["stale"] is True
