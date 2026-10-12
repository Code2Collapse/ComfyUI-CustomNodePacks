"""L7.98: the SAM nodes without a SAM model - a labelled model-free stand-in instead of an error.

SAM Model Loader with no model on disk (or no `sam2` package, or no internet for a [download]) hands out a SAM_MODEL
whose model_type is "model-free"; get_sam_predictor answers it with GrabCut behind SAM's predictor API, and the SAM
nodes label their info output approximate. C2C_MODEL_FALLBACK=off keeps the old error. Synthetic image only.
"""
from __future__ import annotations

import json

import numpy as np
import pytest
import torch

W, H, CX, CY, R = 256, 192, 128, 96, 40


def scene():
    rng = np.random.default_rng(11)
    g = (0.45 + 0.08 * rng.standard_normal((H, W, 1))).clip(0, 1).repeat(3, -1)
    yy, xx = np.mgrid[:H, :W]
    disk = (xx - CX) ** 2 + (yy - CY) ** 2 <= R ** 2
    g[disk] = [0.85, 0.12, 0.10]
    return (g * 255).astype(np.uint8), disk


def iou(m, gt):
    m = np.asarray(m) > 0.5
    return float(np.logical_and(m, gt).sum()) / float(np.logical_or(m, gt).sum())


@pytest.fixture
def sml():
    # imported inside the test, never at collection (see test_sam_model_loader_names.py)
    return pytest.importorskip("nodes.sam_model_loader")


def _no_model(*_a, **_k):
    raise FileNotFoundError("SAM model '(place model in models/sams/ or models/sam2/)' not found.")


def test_loader_hands_out_a_labelled_stand_in(sml, monkeypatch):
    monkeypatch.setattr(sml.SAMModelLoaderMEC, "_resolve_path", staticmethod(_no_model))
    (sam,) = sml.SAMModelLoaderMEC().load("(place model in models/sams/ or models/sam2/)", "auto", "cpu", False,
                                          "float32")
    assert sam["model_type"] == "model-free" and sam["model"] is None and sam["model_free"] is True
    assert "not found" in sam["fallback_from"]["why"]


def test_loader_kill_switch_and_bugs(sml, monkeypatch):
    monkeypatch.setattr(sml.SAMModelLoaderMEC, "_resolve_path", staticmethod(_no_model))
    monkeypatch.setenv("C2C_MODEL_FALLBACK", "off")
    with pytest.raises(FileNotFoundError):
        sml.SAMModelLoaderMEC().load("x.pt", "auto", "cpu", False, "float32")
    monkeypatch.delenv("C2C_MODEL_FALLBACK")

    def bug(*_a, **_k):
        raise TypeError("a real bug")
    monkeypatch.setattr(sml.SAMModelLoaderMEC, "_resolve_path", staticmethod(bug))
    with pytest.raises(TypeError):
        sml.SAMModelLoaderMEC().load("x.pt", "auto", "cpu", False, "float32")


def test_predictor_api_points_box_and_refinement():
    utils = pytest.importorskip("nodes.utils")
    img, gt = scene()
    pred = utils.get_sam_predictor(None, "model-free", img)
    masks, scores, low = pred.predict(point_coords=np.array([[CX, CY]], np.float32),
                                      point_labels=np.array([1], np.int32), multimask_output=True)
    assert masks.shape == (1, H, W) and scores.shape == (1,) and low.shape == (1, 256, 256)
    assert iou(masks[0], gt) >= 0.85
    masks, _s, _l = pred.predict(box=np.array([CX - R - 6, CY - R - 6, CX + R + 6, CY + R + 6], np.float32))
    assert iou(masks[0], gt) >= 0.85
    masks, _s, _l = pred.predict(point_coords=np.array([[CX, CY]], np.float32), point_labels=np.array([1]),
                                 mask_input=low)                       # the generator's refinement loop
    assert iou(masks[0], gt) >= 0.85


def test_sam_mask_generator_end_to_end(sml, monkeypatch):
    gen = pytest.importorskip("nodes.sam_mask_generator")
    monkeypatch.setattr(sml.SAMModelLoaderMEC, "_resolve_path", staticmethod(_no_model))
    (sam,) = sml.SAMModelLoaderMEC().load("x.pt", "auto", "cpu", False, "float32")
    img, gt = scene()
    image = torch.from_numpy(img).float().div(255.0).unsqueeze(0)
    out = gen.SAMMaskGeneratorMEC().generate(
        sam, image, json.dumps([{"x": CX, "y": CY, "label": 1}]), "", "", "", "none", 0.25, 0.3,
        True, 0, 0.0, False, refine_iterations=2, edge_refine="none")
    mask, info = out[0], json.loads(out[-1])
    assert iou(mask[0].numpy(), gt) >= 0.85
    assert info["model_type"] == "model-free" and info["approximate"] is True
    assert "not found" in info["model_free"]["why"]


def test_text_only_with_the_stand_in_is_refused_honestly(sml, monkeypatch):
    gen = pytest.importorskip("nodes.sam_mask_generator")
    monkeypatch.setattr(sml.SAMModelLoaderMEC, "_resolve_path", staticmethod(_no_model))
    (sam,) = sml.SAMModelLoaderMEC().load("x.pt", "auto", "cpu", False, "float32")
    img, _gt = scene()
    image = torch.from_numpy(img).float().div(255.0).unsqueeze(0)
    with pytest.raises(ValueError, match="Text prompts need a model"):
        gen.SAMMaskGeneratorMEC().generate(sam, image, "[]", "", "red disk", "", "none", 0.25, 0.3,
                                           True, 0, 0.0, False)


def test_multi_mask_picker_labels_the_stand_in(sml, monkeypatch):
    picker = pytest.importorskip("nodes.sam_multi_mask_picker")
    monkeypatch.setattr(sml.SAMModelLoaderMEC, "_resolve_path", staticmethod(_no_model))
    (sam,) = sml.SAMModelLoaderMEC().load("x.pt", "auto", "cpu", False, "float32")
    img, gt = scene()
    image = torch.from_numpy(img).float().div(255.0).unsqueeze(0)
    node = picker.SamMultiMaskPickerMEC()
    fn = getattr(node, node.FUNCTION)
    out = fn(image=image, model_name="sam2.1_hiera_base_plus", points_json=json.dumps([{"x": CX, "y": CY, "label": 1}]),
             bbox_json="", precision="fp32", selected_index=0, sam_model=sam)
    out = out["result"] if isinstance(out, dict) else out          # the node also sends thumbnails to the UI
    info = json.loads(out[-1])
    assert iou(out[0][0].numpy(), gt) >= 0.85
    assert info["approximate"] is True and info["model_type"] == "model-free"
    assert info["scores_raw"][1:] == [0.0, 0.0]            # one real mask; the picker pads its three slots with empties
