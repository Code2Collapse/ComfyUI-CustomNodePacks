"""L7.114: the Matte viewer's hidden helper nodes (nodes/c2c_viewer): Plate Proxy, Detect, Viewer Result.

No weights here: Detect runs with a fake detector that speaks the same protocol as CNP's SAM 3.1 backend
(detect(frame, text, pos, neg, boxes) -> [{mask, box, score}]). The temp folder is redirected to pytest's tmp_path.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest
import torch

from nodes.c2c_viewer import _common as C
from nodes.c2c_viewer import detect as D
from nodes.c2c_viewer.plate_proxy import C2CViewerPlateProxy
from nodes.c2c_viewer.result import C2CViewerResult

H, W = 200, 300


@pytest.fixture(autouse=True)
def temp(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "temp_root", lambda: str(tmp_path))
    D._loaded.clear()
    yield tmp_path
    D._loaded.clear()


def plate(b=4):
    g = torch.linspace(0, 1, W).view(1, 1, W, 1).expand(b, H, W, 3).clone()
    for i in range(b):
        g[i, 50:150, 100 + i:200 + i] = torch.tensor([0.9, 0.2, 0.1])
    return g


def payload(out):
    assert out["result"] == ()
    (p,) = out["ui"]["c2c_viewer"]
    json.dumps(p)                                    # the browser gets it as JSON
    return p


def readback(tmp_path, ref):
    from PIL import Image
    return Image.open(os.path.join(tmp_path, ref["subfolder"], ref["filename"]))


# ── registration: present for a prompt, absent from the menu ─────────────────────────────────────────────────────

def test_the_helpers_register_hidden_with_mask_matting():
    import nodes.mask_matting as mm
    for name in ("C2CViewerPlateProxy", "C2CViewerDetect", "C2CViewerResult"):
        cls = mm.NODE_CLASS_MAPPINGS[name]
        assert cls.DEV_ONLY is True                  # frontend 1.53: dev_only -> not in library / search / menu
        assert cls.OUTPUT_NODE is True and cls.RETURN_TYPES == ()   # a helper prompt runs them
        assert "hidden" in cls.DESCRIPTION


# ── Plate Proxy ──────────────────────────────────────────────────────────────────────────────────────────────────

def test_plate_proxy_writes_small_frames_once(temp):
    img = plate(4)
    p = payload(C2CViewerPlateProxy().execute(img, max_side=150))
    assert (p["count"], p["width"], p["height"], p["proxy"], p["step"]) == (4, W, H, [150, 100], 1)
    assert [f["index"] for f in p["frames"]] == [0, 1, 2, 3]
    im = readback(temp, p["frames"][2])
    assert im.size == (150, 100) and im.mode == "RGB"
    path = os.path.join(temp, p["frames"][0]["subfolder"], p["frames"][0]["filename"])
    mtime = os.path.getmtime(path)
    again = payload(C2CViewerPlateProxy().execute(img, max_side=150))
    assert again["fingerprint"] == p["fingerprint"] and os.path.getmtime(path) == mtime
    other = img.clone()
    other[0, 0, 0, 0] = 0.123
    assert payload(C2CViewerPlateProxy().execute(other, max_side=150))["fingerprint"] != p["fingerprint"]


def test_a_long_clip_shows_every_kth_frame_and_the_last():
    idx, k = C.sample_indices(1001, 240)
    assert k == 5 and idx[0] == 0 and idx[-1] == 1000 and len(idx) <= 241
    assert C.sample_indices(17, 240) == (list(range(17)), 1)


def test_proxy_size_keeps_the_aspect():
    assert C.proxy_size(3840, 2160) == (960, 540)
    assert C.proxy_size(640, 480) == (640, 480)       # never enlarged


# ── Detect ───────────────────────────────────────────────────────────────────────────────────────────────────────

class FakeDetector:
    def __init__(self):
        self.calls = []
        self.unloaded = 0

    def detect(self, frame, text, pos, neg, boxes):
        self.calls.append((frame.shape, text, list(pos), list(neg), list(boxes)))
        h, w = frame.shape[:2]
        out = []
        if text == "box":
            m = np.zeros((h, w), np.float32)
            m[50:150, 100:200] = 1
            out.append({"mask": m, "box": [100, 50, 200, 150], "score": 0.6})
            m2 = np.zeros((h, w), np.float32)
            m2[10:30, 10:40] = 1
            out.append({"mask": m2, "box": None, "score": 0.9})
        if pos:
            m = np.zeros((h, w), np.float32)
            m[60:100, 120:160] = 1
            out.append({"mask": m, "box": None, "score": 0.5})
        return out

    def unload(self):
        self.unloaded += 1


@pytest.fixture
def fake(monkeypatch):
    det = FakeDetector()
    monkeypatch.setattr(D, "DETECTOR_FACTORY", lambda tag, path: det)
    monkeypatch.setattr(D, "sam_checkpoints", lambda: [("sam3.1/sam3.1_multiplex_fp16.safetensors", "X:/s.st")])
    return det


def test_detect_returns_one_mask_per_object(temp, fake):
    p = payload(D.C2CViewerDetect().execute(plate(3), 1, "sam3.1/sam3.1_multiplex_fp16.safetensors",
                                            text="box, nothing", max_side=150))
    assert fake.calls[0][1] == "box" and fake.calls[1][1] == "nothing"     # one call per comma-separated word
    assert [o["score"] for o in p["objects"]] == [0.9, 0.6]                # strongest first
    big = p["objects"][1]
    assert big["label"] == "box" and big["box"] == [round(100 / W, 5), 0.25, round(200 / W, 5), 0.75]
    assert abs(big["area"] - (100 * 100) / (W * H)) < 1e-6
    im = readback(temp, big["mask"])
    assert im.size == (150, 100) and im.mode == "LA"
    alpha = np.asarray(im)[..., 1]
    assert alpha[50, 75] == 255 and alpha[5, 5] == 0                       # the mask is in the alpha channel
    small = p["objects"][0]
    assert small["box"] == [round(10 / W, 5), 0.05, round(40 / W, 5), 0.15]    # box from the mask when none given
    assert p["message"] == "" and p["frame"] == 1 and p["model"].startswith("sam3.1/")


def test_a_click_without_find_detects_from_the_point(temp, fake):
    pts = json.dumps([{"x": 0.5, "y": 0.4, "label": 1}, {"x": 0.1, "y": 0.1, "label": 0}])
    p = payload(D.C2CViewerDetect().execute(plate(2), 0, "sam3.1/sam3.1_multiplex_fp16.safetensors", points=pts))
    (_shape, text, pos, neg, _boxes), = fake.calls
    assert text == "" and pos == [(150.0, 80.0)] and neg == [(30.0, 20.0)]
    assert len(p["objects"]) == 1 and p["objects"][0]["label"] == "clicked"


def test_nothing_found_is_said_on_the_frame(temp, fake):
    p = payload(D.C2CViewerDetect().execute(plate(2), 0, "sam3.1/sam3.1_multiplex_fp16.safetensors",
                                            text="cat"))
    assert p["objects"] == [] and p["message"] == "Nothing found for 'cat' on frame 0. Try other words, or click the subject."


def test_detect_refuses_in_words(temp, fake):
    node = D.C2CViewerDetect()
    with pytest.raises(ValueError, match="plate has 2 frames"):
        node.execute(plate(2), 5, "sam3.1/sam3.1_multiplex_fp16.safetensors", text="x")
    with pytest.raises(ValueError, match="Type what to find"):
        node.execute(plate(2), 0, "sam3.1/sam3.1_multiplex_fp16.safetensors")
    with pytest.raises(ValueError, match="no longer on this machine"):
        node.execute(plate(2), 0, "sam3/gone.pt", text="x")


def test_the_detector_loads_once(temp, fake, monkeypatch):
    made = []
    monkeypatch.setattr(D, "DETECTOR_FACTORY", lambda tag, path: made.append(path) or fake)
    for _ in range(3):
        D.C2CViewerDetect().execute(plate(2), 0, "sam3.1/sam3.1_multiplex_fp16.safetensors", text="box")
    assert made == ["X:/s.st"]


# ── the model slot ───────────────────────────────────────────────────────────────────────────────────────────────

def test_the_strongest_checkpoint_comes_first(tmp_path, monkeypatch):
    from nodes.mask_matting import utils
    files = {"sam3": ["sam3.pt", "sam3.1_multiplex_fp16.safetensors"],          # the same file, both folders
             "sam3.1": ["sam3.1_multiplex_fp32.safetensors", "sam3.1_multiplex_fp16.safetensors", "sam3.1.pt"]}
    for key, names in files.items():
        for n in names:
            (tmp_path / n).write_bytes(b"x")
    monkeypatch.setattr(utils, "list_backend_files", lambda key: files.get(key, []))
    monkeypatch.setattr(utils, "resolve_backend_weight", lambda key, name: str(tmp_path / name))
    got = [t for t, _p in D.sam_checkpoints()]
    assert got == ["sam3.1/sam3.1_multiplex_fp16.safetensors", "sam3.1/sam3.1_multiplex_fp32.safetensors",
                   "sam3.1/sam3.1.pt", "sam3/sam3.pt"]


def test_no_checkpoint_is_a_missing_entry_with_folder_and_link(monkeypatch):
    from nodes.mask_matting import utils
    monkeypatch.setattr(utils, "list_backend_files", lambda key: [])
    monkeypatch.setattr(utils, "backend_first_root", lambda key: "/models/" + key)
    assert D.model_choices() == ["MISSING: sam3.1_multiplex_fp16.safetensors"]
    spec = D.C2CViewerDetect.INPUT_TYPES()["required"]["model"]
    info = spec[1]["c2c_missing"]
    assert info["file"] == "sam3.1_multiplex_fp16.safetensors" and info["folder"].endswith("sam3.1")
    assert info["url"].startswith("https://huggingface.co/Comfy-Org/sam3.1")
    with pytest.raises(ValueError) as e:
        D.C2CViewerDetect().execute(plate(2), 0, spec[0][0], text="person")
    msg = str(e.value)
    assert "sam3.1_multiplex_fp16.safetensors" in msg and info["folder"] in msg and info["url"] in msg
    assert "press R" in msg


# ── Viewer Result ────────────────────────────────────────────────────────────────────────────────────────────────

def test_result_writes_plate_and_alpha_and_measures(temp):
    pytest.importorskip("cv2")
    img = plate(6)
    alpha = torch.zeros(3, H, W)
    alpha[:, 50:150, 102:202] = 1.0
    coarse = alpha.clone()
    p = payload(C2CViewerResult().execute(img, alpha, start=2, coarse=coarse, max_side=150))
    assert (p["start"], p["count"]) == (2, 3)
    assert [f["index"] for f in p["frames"]] == [2, 3, 4]
    assert readback(temp, p["frames"][0]["alpha"]).mode == "LA"
    assert readback(temp, p["frames"][0]["plate"]).size == (150, 100)
    q = p["qc"]
    assert q["coverage"]["value"] == 1.0 and q["coverage"]["level"] == "green"
    assert q["flicker"]["pairs"] == 2 and set(q["thresholds"]) == {"coverage", "edge_detail", "flicker"}


def test_result_refuses_a_render_that_does_not_fit_the_plate(temp):
    with pytest.raises(ValueError, match="plate has 2"):
        C2CViewerResult().execute(plate(2), torch.zeros(3, H, W), start=0)
    with pytest.raises(ValueError, match="plate is 300x200"):
        C2CViewerResult().execute(plate(2), torch.zeros(1, 10, 10), start=0)
