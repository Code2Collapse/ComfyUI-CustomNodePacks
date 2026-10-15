"""L7.65 P05: one Matte node - SAM + ViTMatte, SeC + MatAnyone2 and Background Remover are backends of MaskOps.

Same method as tests/test_consolidation_wave2.py: the old node's inputs go through its row in
nodes/_legacy_replacements.json (as the workflow migration moves them) into MaskOpsMEC, and every mapped output is
compared with what the old node returned. The old nodes are kept as MaskOps' engines, byte for byte the d97cc13 files
(pinned below), so the reference is the shipped code called the way the old node was called.

Two kinds of case per row:
* the inference functions monkeypatched to deterministic fakes that RECORD their arguments - the migrated node must
  make the same calls with the same arguments and return the same tensors;
* the model-free paths (no weights: GrabCut, guided filters, optical flow) run for real.

`info` is compared as JSON: MaskOps adds two keys ("backend", "maskops"); SAM + ViTMatte's `elapsed_sec` is wall
clock. Everything else is bit-equal. Design: docs/design/matte_unified.md.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from tests.test_consolidation_wave2 import TABLE, migrate_kwargs, same

PACK = Path(__file__).resolve().parents[1]
W, H, CX, CY, R = 96, 72, 48, 36, 16

# The engines must stay the shipped code: sha256 of each file at d97cc13 (line endings normalised to LF).
ENGINE_SHA256 = {
    "nodes/sam_vitmatte_pipeline.py": "703865263fb84e5fdffde3eeb27246d5eb8aca151b2d6d3a2f7c43c3ef374783",
    "nodes/sec_matanyone_pipeline.py": "e4dfe42f2b02d94debf76180adc8d49c3edf80669787f665235d351f06e33163",
    "nodes/background_remover.py": "9207e2b229b26b29f6cc3be5d02b152ed23d8819a9ff42a49ebbe5a12883fcb0",
}
OLD_RETURN_NAMES = (
    "mask", "alpha", "preview", "trimap", "bbox", "bbox_json", "score", "info",
    "despilled", "lightwrap_rgba", "edge_mask", "inside_mask", "outside_mask",
    "luma_key_mask", "problem_regions", "severity", "suggested_method", "object_alphas",
)

# MaskOpsMEC's inputs at d97cc13 (read from that commit's INPUT_TYPES), in order
INPUT_ORDER = {
    "required": [
        "image", "segmenter", "matter", "model", "matter_model", "precision", "attention", "offload",
        "subject_preset", "trimap_dilate", "trimap_erode", "edge_radius", "edge_mode", "edge_threshold",
        "edge_feather", "individual_objects", "tracking_direction", "frame_annotation", "object_id",
        "max_frames_to_track", "memory_size", "start_frame", "end_frame", "auto_download", "seed", "tta_flip",
        "multiscale", "post_refine", "refine_radius", "refine_iterations", "despill", "despill_strength",
        "preserve_skin", "lightwrap_strength", "lightwrap_radius", "edge_band_radius", "premultiply",
        "enable_luma_key", "luma_mode", "luma_low", "luma_high", "luma_gamma", "luma_falloff", "luma_invert",
        "luma_mix", "enable_advanced_trimap", "trimap_inner_scale", "trimap_outer_scale", "trimap_smooth",
        "trimap_threshold", "auto_quality", "auto_disambiguate", "quality_mode", "enable_diagnose",
        "diag_ring_width", "diag_blur_threshold", "diag_brightness_threshold", "robust_propagation",
        "robust_confidence_threshold", "robust_reanchor_method", "robust_blend_alpha",
    ],
    "optional": [
        "positive_coords", "negative_coords", "pos_bbox", "neg_bbox", "normal_bbox", "text_prompt",
        "external_mask", "external_trimap", "holdout_mask", "core_mask", "pipeline", "scene_prompts",
        "band_scale", "matte_tile", "temporal_stabilise", "matte_overlap", "matte_tile_batch",
    ],
}

# ── helpers ──────────────────────────────────────────────────────────────────────────────────────────────────────

def scene(b=1, seed=3):
    """A red disk on a noisy grey plate (B,H,W,3) and its ground truth."""
    rng = np.random.default_rng(seed)
    frames = []
    for i in range(b):
        g = (0.45 + 0.06 * rng.standard_normal((H, W, 1))).clip(0, 1).repeat(3, -1)
        yy, xx = np.mgrid[:H, :W]
        disk = (xx - CX - i) ** 2 + (yy - CY) ** 2 <= R ** 2
        g[disk] = [0.85, 0.12, 0.10]
        frames.append(g)
    return torch.from_numpy(np.stack(frames)).float()


def _defaults(cls):
    spec = cls.INPUT_TYPES()
    out = {}
    for name, s in {**spec["required"], **spec.get("optional", {})}.items():
        opts = s[1] if len(s) > 1 and isinstance(s[1], dict) else {}
        if "default" in opts:
            out[name] = opts["default"]
        elif isinstance(s[0], (list, tuple)) and s[0]:
            out[name] = s[0][0]
    return out


@pytest.fixture(scope="module")
def maskops():
    node = pytest.importorskip("nodes.mask_matting.node", reason="MaskOps unavailable")
    return node.MaskOpsMEC


def run_maskops(maskops, **migrated):
    kw = _defaults(maskops)
    kw.update(migrated)
    return maskops().execute(**kw)


def _info_equal(old_info: str, new_info: str):
    a, b = json.loads(old_info), json.loads(new_info)
    a.pop("elapsed_sec", None)
    b.pop("elapsed_sec", None)
    assert b.pop("backend") in ("SAM + ViTMatte", "SeC + MatAnyone2", "remove background")
    assert isinstance(b.pop("maskops"), dict)
    assert a == b


def check_row(old_id, old_kwargs, old_out, new_out):
    for m in TABLE[old_id]["output_mapping"]:
        a, b = old_out[m["old_idx"]], new_out[m["new_idx"]]
        if m["new_idx"] == 7:                                  # info: JSON, see the module docstring
            _info_equal(a, b)
            continue
        assert same(a, b), f"{old_id} output {m['old_idx']} -> {m['new_idx']} differs"


def storage_bytes(outputs, exclude=()):
    """Bytes of the unique tensor storages in `outputs`, not counting storages shared with `exclude` (the input)."""
    skip = {t.untyped_storage().data_ptr() for t in exclude if torch.is_tensor(t)}
    seen = {}
    for t in outputs:
        if torch.is_tensor(t):
            st = t.untyped_storage()
            if st.data_ptr() not in skip:
                seen[st.data_ptr()] = st.nbytes()
    return sum(seen.values())


# ── the engines are the shipped code, the survivor keeps its outputs ─────────────────────────────────────────────

@pytest.mark.parametrize("path", sorted(ENGINE_SHA256))
def test_engine_modules_are_the_shipped_code(path):
    data = (PACK / path).read_bytes().replace(b"\r\n", b"\n")
    assert hashlib.sha256(data).hexdigest() == ENGINE_SHA256[path], (
        f"{path} changed: it is the equivalence reference of a migration row. Re-run this file's equivalence "
        "tests against the d97cc13 code before updating the hash.")


def test_maskops_keeps_its_18_outputs_and_appends_foreground(maskops):
    assert maskops.RETURN_NAMES[:18] == OLD_RETURN_NAMES
    assert maskops.RETURN_NAMES[18] == "foreground" and maskops.RETURN_TYPES[18] == "IMAGE"
    assert len(maskops.RETURN_TYPES) == len(maskops.RETURN_NAMES) == len(maskops.OUTPUT_TOOLTIPS) == 19


def test_maskops_keeps_its_78_inputs_in_order(maskops):
    """D7.65b: new inputs are appended; the old ones keep name, order and type (saved widgets are positional)."""
    spec = maskops.INPUT_TYPES()
    assert list(spec["required"]) == INPUT_ORDER["required"]
    assert list(spec["optional"])[:len(INPUT_ORDER["optional"])] == INPUT_ORDER["optional"]
    from nodes.mask_matting import _engines
    assert list(spec["optional"])[len(INPUT_ORDER["optional"]):] == list(_engines.appended_inputs())


@pytest.mark.parametrize("old_id, module, cls_name", [
    ("SAMViTMattePipelineMEC", "nodes.sam_vitmatte_pipeline", "SAMViTMattePipelineMEC"),
    ("SeCMatAnyonePipelineMEC", "nodes.sec_matanyone_pipeline", "SeCMatAnyonePipelineMEC"),
    ("BackgroundRemoverMEC", "nodes.background_remover", "BackgroundRemoverMEC"),
])
def test_every_mapped_widget_has_the_old_spec(maskops, old_id, module, cls_name):
    """Same type, default, range and options: every saved value of the old node validates on the survivor. Where
    the survivor's own input is reused (precision, subject_preset, auto_download, ...), its spec must match too."""
    import importlib
    old = getattr(importlib.import_module(module), cls_name).INPUT_TYPES()
    old = {**old["required"], **old.get("optional", {})}
    new = maskops.INPUT_TYPES()
    new = {**new["required"], **new["optional"]}
    keys = ("default", "min", "max", "step", "multiline", "forceInput")
    for m in TABLE[old_id]["input_mapping"]:
        if "old_id" not in m:
            continue
        o, n = old[m["old_id"]], new[m["new_id"]]
        if isinstance(o[0], (list, tuple)):
            assert list(n[0]) == list(o[0]), f"{old_id}.{m['old_id']} options"
        else:
            assert n[0] == o[0], f"{old_id}.{m['old_id']} type"
        oo = o[1] if len(o) > 1 else {}
        no = n[1] if len(n) > 1 else {}
        if m["old_id"] == "allow_download":
            continue                                           # auto_download: same BOOLEAN, default False
        for k in keys:
            if k in oo:
                assert no.get(k) == oo.get(k), f"{old_id}.{m['old_id']} {k}: {no.get(k)!r} != {oo.get(k)!r}"


def test_migrated_rows_switch_off_what_the_old_nodes_never_made():
    for old_id in ("SAMViTMattePipelineMEC", "SeCMatAnyonePipelineMEC", "BackgroundRemoverMEC"):
        sets = {m["new_id"]: m["set_value"] for m in TABLE[old_id]["input_mapping"] if "set_value" in m}
        assert sets["enable_diagnose"] is False and sets["split_masks"] is False
        assert sets["backend"] != "segment + matte"


# ── SAM + ViTMatte ───────────────────────────────────────────────────────────────────────────────────────────────

def _svm_old_kwargs(sam_model, image, **over):
    from nodes.sam_vitmatte_pipeline import SAMViTMattePipelineMEC
    kw = _defaults(SAMViTMattePipelineMEC)
    kw.update({"sam_model": sam_model, "image": image,
               "points_json": json.dumps([{"x": CX, "y": CY, "label": 1}])})
    kw.update(over)
    return kw


@pytest.fixture
def fake_sam(monkeypatch):
    """A deterministic SAM: three candidate disks around the first point (or the box centre); records every call."""
    svm = pytest.importorskip("nodes.sam_vitmatte_pipeline")
    calls = []

    def predictor(model, model_type, img_np):
        calls.append(("set_image", model_type, img_np.shape, float(img_np.mean())))
        return {"img": img_np}

    def predict(pred, model_info, point_coords=None, point_labels=None, box=None, mask_input=None,
                multimask_output=True):
        calls.append(("predict", None if point_coords is None else np.asarray(point_coords).tolist(),
                      None if point_labels is None else np.asarray(point_labels).tolist(),
                      None if box is None else np.asarray(box).tolist(),
                      None if mask_input is None else float(np.asarray(mask_input).sum()), multimask_output))
        hh, ww = pred["img"].shape[:2]
        no_points = point_coords is None or not len(point_coords)
        if no_points and box is None:
            # no prompt at all: the old node keeps its existing_mask then (SAM's failure path)
            raise RuntimeError("fake SAM: no prompt")
        # The geometry ignores the prompts (they are compared through the call log): a disk with a hole and a
        # 3x3 speck far from it, so fill_holes / min_region_size change the result.
        yy, xx = np.mgrid[:hh, :ww]
        d2 = (xx - CX) ** 2 + (yy - CY) ** 2
        masks = np.stack([(d2 <= r ** 2) & (d2 > 9) for r in (R - 3, R, R + 4)])
        masks[:, hh - 5:hh - 2, ww - 5:ww - 2] = True
        return masks, np.array([0.71, 0.93, 0.64], np.float32), None

    def vitmatte(img_t, mask_t, edge_radius, trimap_input=None):
        calls.append(("vitmatte", tuple(img_t.shape), float(mask_t.sum()), int(edge_radius),
                      None if trimap_input is None else float(trimap_input.sum())))
        k = torch.ones(1, 1, 5, 5) / 25.0
        return torch.nn.functional.conv2d(mask_t[None, None].float(), k, padding=2)[0, 0].clamp(0, 1)

    monkeypatch.setattr(svm, "get_sam_predictor", predictor)
    monkeypatch.setattr(svm, "sam_predict", predict)
    monkeypatch.setattr(svm, "refine_with_vitmatte", vitmatte)
    model = {"model": torch.nn.Linear(1, 1), "model_type": "sam2.1", "device": "cpu", "offload_to_cpu": False,
             "dtype": torch.float32}
    return svm, model, calls


@pytest.mark.parametrize("over", [
    {},
    {"subject_type": "hair"},
    {"refine_method": "vitmatte", "sam_iterations": 3, "edge_contrast": 1.4, "mask_index": 2,
     "multimask_output": True},
    {"refine_method": "guided_filter", "score_threshold": 0.8, "trimap_dilate": 9, "trimap_erode": 4},
    {"bbox_json": json.dumps([CX - R - 4, CY - R - 4, CX + R + 4, CY + R + 4]), "points_json": "[]",
     "fill_holes_enabled": False, "min_region_size": 0},
    {"batch_mode": True},
    # every widget off its default (subject_type custom: a preset would overwrite the rest)
    {"sam_iterations": 3, "refine_method": "vitmatte", "edge_radius": 7, "detail_preservation": 0.6,
     "edge_contrast": 1.3, "fill_holes_enabled": False, "min_region_size": 10, "multimask_output": False,
     "mask_index": 1, "score_threshold": 0.5, "trimap_dilate": 5, "trimap_erode": 3, "batch_mode": True,
     "bbox_json": json.dumps([CX - 20, CY - 20, CX + 20, CY + 20])},
    {"refine_method": "multi_scale_guided", "edge_radius": 5, "detail_preservation": 0.3, "edge_contrast": 0.7,
     "min_region_size": 200, "mask_index": 2, "sam_iterations": 1},
])
def test_sam_vitmatte_calls_the_same_code_with_the_same_arguments(maskops, fake_sam, over):
    svm, model, calls = fake_sam
    image = scene(b=2)
    old = _svm_old_kwargs(model, image, **over)
    expected = svm.SAMViTMattePipelineMEC().execute(**old)
    old_calls = list(calls)
    calls.clear()
    out = run_maskops(maskops, **migrate_kwargs("SAMViTMattePipelineMEC", old))
    assert calls == old_calls and old_calls
    check_row("SAMViTMattePipelineMEC", old, expected, out)
    assert len(out) == 19


def test_sam_vitmatte_sockets_move_with_their_row(maskops, fake_sam):
    """bbox -> pos_bbox, existing_mask -> external_mask, trimap -> external_trimap reach the same engine call."""
    svm, model, calls = fake_sam
    image = scene()
    em = torch.zeros(1, H, W)
    em[0, CY - 10:CY + 10, CX - 10:CX + 10] = 1.0
    tri = torch.full((1, H, W), 0.5)
    old = _svm_old_kwargs(model, image, bbox=[CX - 20, CY - 20, CX + 20, CY + 20], existing_mask=em, trimap=tri,
                          refine_method="vitmatte")
    expected = svm.SAMViTMattePipelineMEC().execute(**old)
    old_calls = list(calls)
    calls.clear()
    out = run_maskops(maskops, **migrate_kwargs("SAMViTMattePipelineMEC", old))
    assert calls == old_calls
    check_row("SAMViTMattePipelineMEC", old, expected, out)
    assert json.loads(out[7])["maskops"]["trimap"] == "external"


def test_sam_vitmatte_existing_mask_is_kept_when_sam_gives_nothing(maskops, fake_sam):
    """With no prompt the fake SAM fails, and the old node keeps the existing mask: the migrated one must too."""
    svm, model, _calls = fake_sam
    em = torch.zeros(1, H, W)
    em[0, 10:30, 20:60] = 1.0
    old = _svm_old_kwargs(model, scene(), points_json="[]", existing_mask=em, refine_method="guided_filter",
                          fill_holes_enabled=False, min_region_size=0)
    expected = svm.SAMViTMattePipelineMEC().execute(**old)
    assert float(expected[1].sum()) == float(em.sum())            # the coarse mask IS the existing mask
    out = run_maskops(maskops, **migrate_kwargs("SAMViTMattePipelineMEC", old))
    check_row("SAMViTMattePipelineMEC", old, expected, out)


@pytest.fixture
def model_free_sam(monkeypatch):
    sml = pytest.importorskip("nodes.sam_model_loader")

    def _no_model(*_a, **_k):
        raise FileNotFoundError("SAM model 'x.pt' not found.")
    monkeypatch.setattr(sml.SAMModelLoaderMEC, "_resolve_path", staticmethod(_no_model))
    (sam,) = sml.SAMModelLoaderMEC().load("x.pt", "auto", "cpu", False, "float32")
    assert sam["model_type"] == "model-free"
    return sam


@pytest.mark.parametrize("over", [
    {"refine_method": "guided_filter"},
    {"refine_method": "multi_scale_guided", "subject_type": "skin_face"},
    {"refine_method": "color_aware", "edge_contrast": 0.8},
    {"refine_method": "laplacian_blend", "subject_type": "soft_glow"},
    {"refine_method": "auto", "batch_mode": True},
])
def test_sam_vitmatte_model_free_for_real(maskops, model_free_sam, monkeypatch, over):
    svm = pytest.importorskip("nodes.sam_vitmatte_pipeline")
    # "ViTMatte is not installed": its loader gives up, the refinement falls back the way it always did
    monkeypatch.setattr(svm, "refine_with_vitmatte", lambda *a, **k: None)
    image = scene(b=2)
    old = _svm_old_kwargs(model_free_sam, image, **over)
    expected = svm.SAMViTMattePipelineMEC().execute(**old)
    out = run_maskops(maskops, **migrate_kwargs("SAMViTMattePipelineMEC", old))
    check_row("SAMViTMattePipelineMEC", old, expected, out)
    info = json.loads(out[7])
    assert info["approximate"] is True and info["backend"] == "SAM + ViTMatte"


def test_sam_vitmatte_without_a_sam_model_says_so(maskops):
    with pytest.raises(ValueError, match="connect a SAM Model Loader"):
        run_maskops(maskops, image=scene(), backend="SAM + ViTMatte")


# ── SeC + MatAnyone2 ─────────────────────────────────────────────────────────────────────────────────────────────

def _sec_old_kwargs(image, **over):
    from nodes.sec_matanyone_pipeline import SeCMatAnyonePipelineMEC
    kw = _defaults(SeCMatAnyonePipelineMEC)
    kw.update({"image": image, "points_json": json.dumps([{"x": CX, "y": CY, "label": 1}])})
    kw.update(over)
    return kw


@pytest.fixture
def fake_sec(monkeypatch):
    sec = pytest.importorskip("nodes.sec_matanyone_pipeline")
    us = pytest.importorskip("nodes.unified_segmentation")
    calls = []

    from nodes._download_policy import downloads_allowed

    def segment(self, **k):
        calls.append(("segment", {n: (tuple(v.shape) if torch.is_tensor(v) else v) for n, v in k.items()},
                      downloads_allowed()))
        img = k["image"]
        b, hh, ww = img.shape[:3]
        yy, xx = torch.meshgrid(torch.arange(hh), torch.arange(ww), indexing="ij")
        m = (((xx - CX) ** 2 + (yy - CY) ** 2) <= R ** 2).float()
        return m.expand(b, hh, ww).clone(), 0.875, {"fake": True}

    def matanyone(self, image, mask, n_warmup, B, Hh, Ww, device):
        calls.append(("matanyone2", tuple(image.shape), float(mask.sum()), n_warmup, B, Hh, Ww))
        return (mask * 0.9 + 0.05).clamp(0, 1)

    def vitmatte(self, image, mask, variant, edge_radius, B, Hh, Ww, device):
        calls.append(("vitmatte", variant, tuple(image.shape), float(mask.sum()), edge_radius, B))
        return (mask * 0.8 + 0.1).clamp(0, 1)

    monkeypatch.setattr(us.UnifiedSegmentation, "segment", segment)
    monkeypatch.setattr(sec, "clear_cache", lambda *a, **k: calls.append(("clear_cache",)))
    monkeypatch.setattr(sec.SeCMatAnyonePipelineMEC, "_run_matanyone2", matanyone)
    monkeypatch.setattr(sec.SeCMatAnyonePipelineMEC, "_run_vitmatte", vitmatte)
    return sec, calls


@pytest.mark.parametrize("b, over", [
    (1, {}),
    (3, {}),                                                   # auto -> MatAnyone2 on a clip
    (3, {"matting_backend": "vitmatte_base", "edge_radius": 9, "text_prompt": "a red ball"}),
    (1, {"matting_backend": "matanyone2", "n_warmup": 2, "precision": "fp32", "segmentation_model": "sec_4b"}),
    (2, {"edge_refine_method": "multi_scale_guided", "fill_holes_enabled": False, "min_region_size": 0,
         "bbox": [CX - 20, CY - 20, CX + 20, CY + 20], "keep_model_loaded": False}),
    # every widget off its default
    (2, {"segmentation_model": "sam3", "text_prompt": "ball", "matting_backend": "vitmatte_small",
         "edge_radius": 6, "n_warmup": 9, "precision": "bf16", "fill_holes_enabled": False, "min_region_size": 3,
         "edge_refine_method": "guided_filter", "keep_model_loaded": False, "allow_download": True,
         "bbox_json": json.dumps([CX - 20, CY - 20, CX + 20, CY + 20]), "points_json": "[]"}),
])
def test_sec_calls_the_same_code_with_the_same_arguments(maskops, fake_sec, monkeypatch, b, over):
    sec, calls = fake_sec
    monkeypatch.setenv("C2C_NO_DOWNLOADS", "0")                 # so the Allow Download flag is visible to the fake
    old = _sec_old_kwargs(scene(b=b), **over)
    expected = sec.SeCMatAnyonePipelineMEC().process(**old)
    old_calls = list(calls)
    calls.clear()
    out = run_maskops(maskops, **migrate_kwargs("SeCMatAnyonePipelineMEC", old))
    assert calls == old_calls and old_calls
    check_row("SeCMatAnyonePipelineMEC", old, expected, out)


def test_sec_a_wired_text_prompt_wins_over_the_typed_one(maskops, fake_sec):
    sec, calls = fake_sec
    old = _sec_old_kwargs(scene(), text_prompt="wired text")
    expected = sec.SeCMatAnyonePipelineMEC().process(**old)
    calls.clear()
    kw = migrate_kwargs("SeCMatAnyonePipelineMEC", {**old, "text_prompt": "typed text"})
    out = run_maskops(maskops, **kw, text_prompt="wired text")
    assert calls[0][1]["text_prompt"] == "wired text"
    check_row("SeCMatAnyonePipelineMEC", old, expected, out)


def _missing(*_a, **_k):
    from nodes._download_policy import ModelNotDownloaded
    raise ModelNotDownloaded("some_model", target="models/x/")


@pytest.mark.parametrize("b, over", [
    (1, {"matting_backend": "vitmatte_small"}),
    (3, {"matting_backend": "vitmatte_small", "edge_refine_method": "guided_filter"}),
    (2, {"matting_backend": "vitmatte_base", "positive_coords": json.dumps([[CX, CY]]), "points_json": "[]"}),
    # the coordinate sockets reach the model-free segmenter only (as in the old node); a negative point inside the
    # disk changes its mask
    (1, {"matting_backend": "vitmatte_small", "positive_coords": json.dumps([[CX - 6, CY]]),
         "negative_coords": json.dumps([[CX + R - 4, CY]]), "points_json": "[]"}),
])
def test_sec_model_free_for_real(maskops, monkeypatch, b, over):
    sec = pytest.importorskip("nodes.sec_matanyone_pipeline")
    us = pytest.importorskip("nodes.unified_segmentation")
    monkeypatch.setattr(us.UnifiedSegmentation, "segment", lambda self, **k: _missing())
    monkeypatch.setattr(sec.SeCMatAnyonePipelineMEC, "_run_vitmatte", lambda self, *a, **k: _missing())
    old = _sec_old_kwargs(scene(b=b), **over)
    expected = sec.SeCMatAnyonePipelineMEC().process(**old)
    out = run_maskops(maskops, **migrate_kwargs("SeCMatAnyonePipelineMEC", old))
    check_row("SeCMatAnyonePipelineMEC", old, expected, out)
    info = json.loads(out[7])
    assert info["approximate"] is True and set(info["model_free"]) == {"segmentation", "matting"}


# ── Background Remover ───────────────────────────────────────────────────────────────────────────────────────────

class _FakeNet(torch.nn.Module):
    """'Red is foreground': logits from the red channel, at the processor's size."""

    def __init__(self, calls):
        super().__init__()
        self.w = torch.nn.Parameter(torch.ones(1))
        self.calls = calls

    def forward(self, pixel_values):
        self.calls.append(("forward", tuple(pixel_values.shape), float(pixel_values.sum())))
        red = pixel_values[:, :1] - pixel_values[:, 1:2]
        return type("Out", (), {"logits": 12.0 * red * self.w})()


class _FakeProcessor:
    def __init__(self, calls):
        self.calls = calls

    def __call__(self, images, return_tensors="pt"):
        a = torch.from_numpy(np.asarray(images, dtype=np.float32) / 255.0).permute(2, 0, 1)[None]
        self.calls.append(("processor", tuple(a.shape)))
        return {"pixel_values": torch.nn.functional.interpolate(a, size=(32, 48), mode="bilinear",
                                                                 align_corners=False)}


@pytest.fixture
def fake_bg(monkeypatch):
    br = pytest.importorskip("nodes.background_remover")
    calls = []

    from nodes._download_policy import downloads_allowed

    def load(name, precision="fp32", device="cpu"):
        calls.append(("load", name, precision, device, downloads_allowed()))
        return {"model": _FakeNet(calls), "processor": _FakeProcessor(calls)}
    monkeypatch.setattr(br, "get_or_load_model", load)
    monkeypatch.setattr(br, "clear_cache", lambda *a, **k: calls.append(("clear_cache",)))
    return br, calls


def _bg_old_kwargs(image, **over):
    from nodes.background_remover import BackgroundRemoverMEC
    kw = _defaults(BackgroundRemoverMEC)
    kw["image"] = image
    kw.update(over)
    return kw


@pytest.mark.parametrize("over", [
    {},
    {"threshold": 0.0},
    {"threshold": 0.3, "invert": True, "mask_blur": 3, "model_name": "rmbg_2.0"},
    {"keep_model_loaded": False, "model_name": "birefnet_portrait"},
    {"model_name": "rmbg_2.0", "threshold": 0.7, "invert": True, "mask_blur": 5, "keep_model_loaded": False,
     "allow_download": True},                                  # every widget off its default
])
def test_background_remover_calls_the_same_code_with_the_same_arguments(maskops, fake_bg, monkeypatch, over):
    br, calls = fake_bg
    monkeypatch.setenv("C2C_NO_DOWNLOADS", "0")                 # so the Allow Download flag is visible to the fake
    old = _bg_old_kwargs(scene(b=2), **over)
    expected = br.BackgroundRemoverMEC().remove_bg(**old)
    old_calls = list(calls)
    calls.clear()
    out = run_maskops(maskops, **migrate_kwargs("BackgroundRemoverMEC", old))
    assert calls == old_calls and old_calls
    check_row("BackgroundRemoverMEC", old, expected, out)


def test_background_remover_keeps_an_rgba_foreground(maskops, fake_bg):
    br, _calls = fake_bg
    rgba = torch.cat([scene(), torch.ones(1, H, W, 1)], dim=-1)
    old = _bg_old_kwargs(rgba)
    expected = br.BackgroundRemoverMEC().remove_bg(**old)
    out = run_maskops(maskops, **migrate_kwargs("BackgroundRemoverMEC", old))
    check_row("BackgroundRemoverMEC", old, expected, out)
    assert out[18].shape[-1] == 4


@pytest.mark.parametrize("over", [{}, {"threshold": 0.0, "mask_blur": 2}, {"invert": True}])
def test_background_remover_model_free_for_real(maskops, monkeypatch, over):
    br = pytest.importorskip("nodes.background_remover")
    monkeypatch.setattr(br, "get_or_load_model", _missing)
    old = _bg_old_kwargs(scene(), **over)
    expected = br.BackgroundRemoverMEC().remove_bg(**old)
    out = run_maskops(maskops, **migrate_kwargs("BackgroundRemoverMEC", old))
    check_row("BackgroundRemoverMEC", old, expected, out)
    assert json.loads(out[7])["approximate"] is True


# ── the shared finishing stage ───────────────────────────────────────────────────────────────────────────────────

def test_finishing_applies_to_an_engine_backend(maskops, monkeypatch):
    """holdout / core / hard edge / despill / split / diagnostics act on the engine's matte like on segment + matte."""
    br = pytest.importorskip("nodes.background_remover")
    monkeypatch.setattr(br, "get_or_load_model", _missing)
    image = scene()
    hold = torch.zeros(1, H, W)
    hold[0, :, :CX] = 1.0
    out = run_maskops(maskops, image=image, backend="remove background", holdout_mask=hold, edge_mode="hard",
                      despill="red", despill_strength=1.0, preserve_skin=False,
                      lightwrap_strength=0.5)
    alpha = out[1]
    assert float(alpha[0, :, :CX].max()) == 0.0                     # the holdout cut it
    assert set(torch.unique(alpha).tolist()) <= {0.0, 1.0}          # hard edge
    assert out[8].shape == image.shape and not same(out[8], image)  # despilled
    assert out[9].shape == (1, H, W, 4) and float(out[9][..., 3].sum()) > 0
    assert float(out[11].sum()) > 0 and float(out[12].sum()) > 0    # inside / outside made (split_masks default on)
    assert torch.allclose(out[18], (out[8] * alpha.unsqueeze(-1)).clamp(0, 1))
    info = json.loads(out[7])
    assert info["maskops"]["finishing"]["holdout"] is True and info["maskops"]["diagnose"]["enabled"] is True


def test_segment_and_matte_gives_a_foreground_and_can_skip_the_split(maskops):
    image = scene()
    kw = dict(image=image, segmenter="classic", matter="guided_filter", pipeline="cascade (legacy)",
              positive_coords=json.dumps([[CX, CY]]), enable_diagnose=False, auto_quality=False)
    out = run_maskops(maskops, **kw)
    assert len(out) == 19 and out[18] is out[2]                    # premultiplied preview = the foreground
    off = run_maskops(maskops, **kw, split_masks=False, premultiply=False)
    assert float(off[10].abs().sum()) == 0.0 and off[10].untyped_storage().nbytes() <= 4
    assert same(off[1], out[1])                                    # the matte itself is untouched
    assert torch.allclose(off[18], (off[8] * off[1].unsqueeze(-1)).clamp(0, 1))
    assert json.loads(out[7])["backend"] == "segment + matte"


# ── memory: a migrated node costs what the old one did ───────────────────────────────────────────────────────────

def test_migrated_engines_add_no_full_size_copies(maskops, fake_sec, fake_bg, fake_sam):
    """Unique tensor storage the node returns, not counting the input image (design section 5)."""
    image = scene(b=3)
    sec, _ = fake_sec
    old = _sec_old_kwargs(image)
    old_out = sec.SeCMatAnyonePipelineMEC().process(**old)
    new_out = run_maskops(maskops, **migrate_kwargs("SeCMatAnyonePipelineMEC", old))
    assert storage_bytes(new_out, exclude=[image]) - storage_bytes(old_out, exclude=[image]) <= 64

    br, _ = fake_bg
    old = _bg_old_kwargs(image)
    old_out = br.BackgroundRemoverMEC().remove_bg(**old)
    new_out = run_maskops(maskops, **migrate_kwargs("BackgroundRemoverMEC", old))
    assert storage_bytes(new_out, exclude=[image]) - storage_bytes(old_out, exclude=[image]) <= 64

    svm, model, _ = fake_sam
    old = _svm_old_kwargs(model, image, batch_mode=True)
    old_out = svm.SAMViTMattePipelineMEC().execute(**old)
    new_out = run_maskops(maskops, **migrate_kwargs("SAMViTMattePipelineMEC", old))
    extra = storage_bytes(new_out, exclude=[image]) - storage_bytes(old_out, exclude=[image])
    assert extra - new_out[18].untyped_storage().nbytes() <= 64     # only the new foreground


# ── kill switch ──────────────────────────────────────────────────────────────────────────────────────────────────

def test_kill_switch_brings_the_old_nodes_back(monkeypatch):
    from nodes.mask_matting import _engines
    monkeypatch.delenv("C2C_LEGACY_MATTE_NODES", raising=False)
    assert _engines.legacy_matte_mappings() == ({}, {})
    monkeypatch.setenv("C2C_LEGACY_MATTE_NODES", "1")
    maps, disp = _engines.legacy_matte_mappings()
    assert set(maps) == {"SAMViTMattePipelineMEC", "SeCMatAnyonePipelineMEC", "BackgroundRemoverMEC"}
    assert set(disp) == set(maps) and all("legacy" in v for v in disp.values())


# ── what each backend reads (the rule tables the spec serves) ───────────────────────────────────────────────────

def _vis():
    return pytest.importorskip("nodes.mask_matting._visibility")


@pytest.mark.parametrize("backend", ["SAM + ViTMatte", "SeC + MatAnyone2", "remove background"])
def test_each_backend_shows_its_own_controls_only(maskops, backend):
    V = _vis()
    vals = {**_defaults(maskops), "backend": backend}
    shown = set(V.visible("", "", vals))
    reads = set(V.ENGINE_READS[backend])
    assert {n for n in reads if V.toggle_allows(n, vals)} <= shown
    others = set().union(*(set(v) for k, v in V.ENGINE_READS.items() if k != backend)) - reads
    assert not shown & others
    assert not shown & {"segmenter", "matter", "model", "matter_model", "pipeline", "scene_prompts"}
    assert set(V.SOCKETS) <= shown


def test_segment_and_matte_never_shows_an_engine_control(maskops):
    V = _vis()
    for seg, mat in (("sam3.1", "vitmatte"), ("birefnet", "none"), ("classic", "rvm")):
        for pipeline in ("onyx", "cascade (legacy)"):
            vals = {**_defaults(maskops), "segmenter": seg, "matter": mat, "pipeline": pipeline}
            shown = V.visible(seg, mat, vals)
            assert shown == ["*"] or not set(shown) & V.BACKEND_ONLY


def test_the_spec_serves_the_backends():
    V = _vis()
    spec = V.build_spec()
    assert spec["native"] == "segment + matte"
    assert set(spec["backends"]) == {"SAM + ViTMatte", "SeC + MatAnyone2", "remove background"}
    assert set(spec["backend_only"]) == set(V.BACKEND_ONLY) and "backend" in spec["always"]
    assert "sam_model" in spec["sockets"]
