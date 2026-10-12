"""L7.91: MaskOps masks on a fresh install - no weights, no third_party/, no internet - and says it is approximate.

The model-free engine (nodes/mask_matting/_classic.py: GrabCut, DIS optical flow, guided filter) stands in when the
chosen model cannot be loaded, and the result is labelled (info["approximate"], info["model_free"]) - never as SAM.
A bug is not a missing model: it still raises. C2C_MODEL_FALLBACK=off restores the plain error.

Test images are synthetic (a red disk on a textured ground); quality on real footage is not claimed here.
"""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

PACK = Path(__file__).resolve().parents[1]
if str(PACK) not in sys.path:
    sys.path.insert(0, str(PACK))
for mod_name in ("folder_paths", "comfy", "comfy.utils", "server"):
    if mod_name not in sys.modules:
        stub = types.ModuleType(mod_name)
        stub.__path__ = []
        sys.modules[mod_name] = stub

from nodes.mask_matting import _classic as C  # noqa: E402
from nodes.mask_matting import _onyx_pipeline as op  # noqa: E402

W, H, CX, CY, R = 256, 192, 128, 96, 40


def scene(cx=CX, cy=CY, seed=7):
    rng = np.random.default_rng(seed)
    g = (0.45 + 0.08 * rng.standard_normal((H, W, 1))).clip(0, 1).repeat(3, -1)
    yy, xx = np.mgrid[:H, :W]
    disk = (xx - cx) ** 2 + (yy - cy) ** 2 <= R ** 2
    g[disk] = [0.85, 0.12, 0.10]
    return (g * 255).astype(np.uint8), disk


def iou(m, gt):
    m = np.asarray(m) > 0.5
    return float(np.logical_and(m, gt).sum()) / float(np.logical_or(m, gt).sum())


def as_image(*frames_u8):
    return torch.from_numpy(np.stack(frames_u8)).float().div(255.0)


# ── engine ───────────────────────────────────────────────────────────────────────────────────────────────────────

def test_grabcut_from_a_point_and_from_a_box():
    img, gt = scene()
    m, _ = C.grabcut_segment(img, [(CX, CY)])
    assert iou(m, gt) >= 0.85
    m, _ = C.grabcut_segment(img, box=(CX - R - 6, CY - R - 6, CX + R + 6, CY + R + 6))
    assert iou(m, gt) >= 0.85


def test_tracker_follows_a_moving_disk():
    from PIL import Image
    frames = [scene(CX - 10 + 4 * i, CY + 2 * i, seed=7 + i) for i in range(6)]
    tr = C.ClassicVideoTracker()
    st = tr.init_state([Image.fromarray(f) for f, _ in frames])
    tr.add_prompt(st, 0, points=[[(CX - 10) / W, CY / H]], point_labels=[1], obj_id=1)
    got = {fi: out["out_binary_masks"][0] for fi, out in tr.propagate_in_video(st, 0, 6, False)}
    assert sorted(got) == list(range(6))
    for fi, (_f, gt) in enumerate(frames):
        assert iou(got[fi], gt) >= 0.8, fi


def test_guided_matte_keeps_the_trimap_certainties():
    img, gt = scene()
    tri = torch.zeros(1, H, W)
    tri[0][torch.from_numpy(gt)] = 1.0
    tri[0, CY - R - 3:CY - R + 3, :] = 0.5                         # an unknown band across the top edge
    a = C.guided_matte(as_image(img), tri)
    assert float(a.min()) >= 0.0 and float(a.max()) <= 1.0
    assert float(a[0, CY, CX]) == 1.0 and float(a[0, 5, 5]) == 0.0


def test_missing_model_vs_bug():
    assert C.is_missing_model_error(ImportError("No module named sam3"))
    assert C.is_missing_model_error(FileNotFoundError("weights"))
    assert C.is_missing_model_error(RuntimeError("SAM 3.1 vendored library not present."))
    assert C.is_missing_model_error(OSError("We couldn't connect to 'https://huggingface.co' to load the files"))
    assert not C.is_missing_model_error(TypeError("unexpected keyword"))
    assert not C.is_missing_model_error(RuntimeError("shape mismatch [1, 3] vs [1, 4]"))


def test_kill_switch(monkeypatch):
    assert C.fallback_allowed()
    for v in ("off", "OFF", "0", "false"):
        monkeypatch.setenv("C2C_MODEL_FALLBACK", v)
        assert not C.fallback_allowed()


# ── registry ─────────────────────────────────────────────────────────────────────────────────────────────────────

def test_model_free_backends_are_registered_and_ready():
    from nodes.mask_matting.matters import all_matters
    from nodes.mask_matting.segmenters import all_segmenters
    assert all_segmenters()["classic"].STATUS == "ready"
    assert all_matters()["guided_filter"].STATUS == "ready"
    # listed after the model backends, so no default and no saved value moves
    assert list(all_segmenters())[-1] == "classic"
    assert list(all_matters())[-1] == "guided_filter"


def test_classic_segmenter_refuses_text_honestly():
    from nodes.mask_matting.segmenters.classic_backend import ClassicSegmenter
    img, _ = scene()
    with pytest.raises(ValueError, match="Text prompts need a model"):
        ClassicSegmenter().segment(as_image(img), text_prompt="red disk")


def test_classic_segmenter_video_bidirectional():
    from nodes.mask_matting.segmenters.classic_backend import ClassicSegmenter
    frames = [scene(CX - 8 + 4 * i, CY, seed=3 + i) for i in range(5)]
    out = ClassicSegmenter().segment(as_image(*[f for f, _ in frames]), positive_points=[(CX, CY)],
                                     frame_annotation=2, tracking_direction="bidirectional")
    assert out["info"]["approximate"] is True and "SAM" not in out["info"]["backend"]
    for i, (_f, gt) in enumerate(frames):
        assert iou(out["mask"][i].numpy(), gt) >= 0.8, i


# ── ONYX: SAM 3.1 / matter missing ───────────────────────────────────────────────────────────────────────────────

def _legacy(**kw):
    base = dict(frame_annotation=0, tracking_direction="forward", start_frame=0, max_frames_to_track=0,
                positive_coords="", negative_coords="", pos_bbox=None, neg_bbox=None, normal_bbox=None,
                text_prompt="")
    base.update(kw)
    return op.LegacyPrompts(**base)


def _onyx(**kw):
    img, _ = scene()
    args = dict(scene_raw="", legacy=_legacy(positive_coords=json.dumps([{"x": CX, "y": CY}])),
                matter_key="none", matter_model="(auto)", matter_cls=None, sam31_model_name="(auto)", device="cpu",
                precision="fp32", attention="auto", offload="none", auto_download=False, band_scale=1.0,
                matte_tile=256, temporal_stabilise=False, post_refine="none", external_mask=None,
                external_trimap=None)
    args.update(kw)
    return op.run_onyx_pipeline(as_image(img), **args)


def _no_sam31(*_a, **_k):
    raise RuntimeError("SAM 3.1 is not installed: its library is not part of ComfyUI-CustomNodePacks")


def test_onyx_without_sam31_masks_and_says_so(monkeypatch):
    monkeypatch.setattr(op, "ensure_sam31_video_model", _no_sam31)
    res = _onyx()
    _img, gt = scene()
    assert iou(res.mask_coarse[0].numpy(), gt) >= 0.8
    info = res.pipeline_info
    assert info["approximate"] is True and info["fallback_from"]["wanted"] == "sam3.1"
    assert "not installed" in info["fallback_from"]["why"] and "SAM" not in info["tracker"]


def test_onyx_matter_missing_uses_the_guided_filter(monkeypatch):
    monkeypatch.setattr(op, "ensure_sam31_video_model", _no_sam31)

    class NoWeights:
        MODELS_KEY = ""

        def __init__(self, **_k):
            pass

        def load(self):
            raise FileNotFoundError("vitmatte weights not found in models/vitmatte/")

    res = _onyx(matter_key="vitmatte", matter_cls=NoWeights)
    assert res.pipeline_info["matter_fallback"]["wanted"] == "vitmatte"
    a = res.alpha_combined
    assert float(a.min()) >= 0.0 and float(a.max()) <= 1.0 and float(a[0, CY, CX]) > 0.9


def test_onyx_text_without_a_text_model_is_refused(monkeypatch):
    monkeypatch.setattr(op, "ensure_sam31_video_model", _no_sam31)
    with pytest.raises(ValueError, match="Text prompts need a model"):
        _onyx(legacy=_legacy(text_prompt="red disk"))


def test_onyx_kill_switch_and_bugs_still_raise(monkeypatch):
    monkeypatch.setattr(op, "ensure_sam31_video_model", _no_sam31)
    monkeypatch.setenv("C2C_MODEL_FALLBACK", "off")
    with pytest.raises(RuntimeError, match="not installed"):
        _onyx()
    monkeypatch.delenv("C2C_MODEL_FALLBACK")

    def bug(*_a, **_k):
        raise TypeError("a real bug")
    monkeypatch.setattr(op, "ensure_sam31_video_model", bug)
    with pytest.raises(TypeError):
        _onyx()


# ── MaskOps node: what the user sees ─────────────────────────────────────────────────────────────────────────────

def _maskops_kwargs(node_mod, **over):
    spec = node_mod.MaskOpsMEC.INPUT_TYPES()
    kw = {}
    for group in ("required", "optional"):
        for name, s in (spec.get(group) or {}).items():
            opts = s[1] if len(s) > 1 and isinstance(s[1], dict) else {}
            if isinstance(s[0], list):
                kw[name] = opts.get("default", s[0][0])
            elif "default" in opts:
                kw[name] = opts["default"]
    kw.update(enable_diagnose=False, auto_quality=False)
    kw.update(over)
    return kw


@pytest.fixture
def missing_backends(monkeypatch):
    """A segmenter and a matter that are registered but not installed, the way sam3.1 / vitmatte are on a fresh
    install."""
    from nodes.mask_matting import matters, segmenters
    from nodes.mask_matting.matters import BaseMatter
    from nodes.mask_matting.segmenters import BaseSegmenter

    class NotInstalledSeg(BaseSegmenter):
        KEY, STATUS, SUPPORTS_MODES = "not_installed_seg", "missing-deps", {"points", "bbox", "text", "video"}

        def segment(self, *_a, **_k):
            raise RuntimeError("Not-installed segmenter: library not present")

    class NotInstalledMatter(BaseMatter):
        KEY, STATUS = "not_installed_matter", "missing-deps"

        def load(self):
            raise ImportError("No module named 'not_installed_matter'")

    monkeypatch.setitem(segmenters._REGISTRY, NotInstalledSeg.KEY, NotInstalledSeg)
    monkeypatch.setitem(matters._REGISTRY, NotInstalledMatter.KEY, NotInstalledMatter)
    return NotInstalledSeg.KEY, NotInstalledMatter.KEY


def test_maskops_cascade_falls_back_and_labels_it(missing_backends):
    node = pytest.importorskip("nodes.mask_matting.node")
    seg, mat = missing_backends
    img, gt = scene()
    out = node.MaskOpsMEC().execute(**_maskops_kwargs(
        node, image=as_image(img), segmenter=seg, matter=mat, pipeline="cascade (legacy)",
        positive_coords=json.dumps([{"x": CX, "y": CY}])))
    names = node.MaskOpsMEC.RETURN_NAMES
    mask, info = out[names.index("mask")], json.loads(out[names.index("info")])
    assert iou(mask[0].numpy(), gt) >= 0.8
    assert info["segmenter"] == "classic" and info["matter"] == "guided_filter"
    assert info["approximate"] is True
    assert info["model_free"]["segmenter"]["wanted"] == seg and info["model_free"]["matter"]["wanted"] == mat


def test_maskops_onyx_default_on_a_fresh_install(monkeypatch):
    node = pytest.importorskip("nodes.mask_matting.node")
    monkeypatch.setattr(op, "ensure_sam31_video_model", _no_sam31)
    img, gt = scene()
    out = node.MaskOpsMEC().execute(**_maskops_kwargs(
        node, image=as_image(img), matter="none", pipeline="onyx",
        positive_coords=json.dumps([{"x": CX, "y": CY}])))
    info = json.loads(out[node.MaskOpsMEC.RETURN_NAMES.index("info")])
    assert iou(out[0][0].numpy(), gt) >= 0.8
    assert info["segmenter"] == "classic" and info["approximate"] is True       # never "sam3.1"
    assert info["model_free"]["segmenter"]["wanted"] == "sam3.1"


def test_maskops_without_a_fallback_reports_nothing_new(missing_backends):
    """A run that needs no fallback carries no model_free / approximate keys - the output is what it was."""
    node = pytest.importorskip("nodes.mask_matting.node")
    img, _ = scene()
    out = node.MaskOpsMEC().execute(**_maskops_kwargs(
        node, image=as_image(img), segmenter="classic", matter="guided_filter", pipeline="cascade (legacy)",
        positive_coords=json.dumps([{"x": CX, "y": CY}])))
    info = json.loads(out[node.MaskOpsMEC.RETURN_NAMES.index("info")])
    assert "model_free" not in info and info["segmenter"] == "classic"


# ── small fixes that came with it ────────────────────────────────────────────────────────────────────────────────

def test_points_as_documented_pairs():
    from nodes.mask_matting.utils import parse_points
    assert parse_points("[[10, 20], [30, 40]]") == ([(10.0, 20.0), (30.0, 40.0)], [])
    assert parse_points('[{"x": 1, "y": 2, "label": 0}]') == ([], [(1.0, 2.0)])
    assert parse_points('{"positive": [[5, 6]], "negative": []}') == ([(5.0, 6.0)], [])


def test_no_model_folders_created_at_start(tmp_path, monkeypatch):
    from nodes.mask_matting import utils
    fp = types.SimpleNamespace(models_dir=str(tmp_path), add_model_folder_path=lambda *a, **k: None)
    monkeypatch.setattr(utils, "folder_paths", fp)
    utils._register_backend_folders()
    assert list(tmp_path.iterdir()) == []
