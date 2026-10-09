"""Tests for C2C video loader nodes (S1a)."""

from __future__ import annotations

import importlib.util
import math
import os
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

PACK = Path(__file__).resolve().parents[1]
C2C_VIDEO = PACK / "nodes" / "c2c_video"


def _load_c2c_video():
    name = "c2c_video"
    if name in sys.modules and hasattr(sys.modules[name], "probe_file"):
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name,
        C2C_VIDEO / "__init__.py",
        submodule_search_locations=[str(C2C_VIDEO)],
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_nodes_load(cv, tmp_input: Path, monkeypatch):
    fp = types.SimpleNamespace(
        get_input_directory=lambda: str(tmp_input),
        get_annotated_filepath=lambda p: os.path.join(str(tmp_input), os.path.basename(p)),
        exists_annotated_filepath=lambda p: os.path.isfile(os.path.join(str(tmp_input), os.path.basename(p))),
    )
    monkeypatch.setitem(sys.modules, "folder_paths", fp)

    comfy_utils = types.ModuleType("comfy.utils")
    comfy_utils.ProgressBar = lambda total: types.SimpleNamespace(
        update_absolute=lambda *a, **k: None,
    )
    mm = types.ModuleType("comfy.model_management")
    mm.throw_exception_if_processing_interrupted = lambda: None
    monkeypatch.setitem(sys.modules, "comfy", types.ModuleType("comfy"))
    monkeypatch.setitem(sys.modules, "comfy.utils", comfy_utils)
    monkeypatch.setitem(sys.modules, "comfy.model_management", mm)

    name = "c2c_video.nodes_load"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name,
        C2C_VIDEO / "nodes_load.py",
        submodule_search_locations=[str(C2C_VIDEO)],
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def cv():
    return _load_c2c_video()


@pytest.fixture(autouse=True)
def _clean_video_state(cv):
    cv.reset_demux_call_count()
    cv.drain_container_pool()
    yield
    cv.drain_container_pool()


@pytest.fixture(autouse=True)
def _private_index_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("C2C_VIDEO_INDEX_DIR", str(tmp_path / "index_cache"))


@pytest.fixture
def nl(cv, tmp_path, monkeypatch):
    inp = tmp_path / "input"
    inp.mkdir()
    return _load_nodes_load(cv, inp, monkeypatch)


@pytest.fixture
def cfr_clip(cv, tmp_path):
    path = str(tmp_path / "cfr.mp4")
    cv.make_test_clip("h264_cfr_b", path, frames=60, width=256, height=256)
    return path


@pytest.fixture
def exr_seq(cv, tmp_path):
    d = str(tmp_path / "exr_seq")
    cv.make_test_sequence(d, frames=8, width=256, height=64)
    return d


@pytest.fixture
def alpha_png_dir(cv, tmp_path):
    pytest.importorskip("cv2")
    import cv2

    d = tmp_path / "alpha_png"
    d.mkdir()
    for i in range(4):
        rgba = np.zeros((64, 64, 4), dtype=np.uint8)
        rgba[..., 0] = 200
        rgba[..., 3] = int(255 * (i + 1) / 4)
        cv2.imwrite(str(d / f"frame_{i:04d}.png"), rgba)
    return str(d)


def _linked_prompt(uid: str) -> dict:
    return {"99": {"class_type": "PreviewImage", "inputs": {"images": [uid, 0]}}}


def _run_path(nl, path, uid="1", **kw):
    defaults = {
        "force_rate": 0,
        "custom_width": 0,
        "custom_height": 0,
        "frame_load_cap": 0,
        "skip_first_frames": 0,
        "select_every_nth": 1,
        "format": "None",
        "vae": None,
        "sequence_fps": 24,
        "prompt": _linked_prompt(uid),
        "unique_id": uid,
    }
    defaults.update(kw)
    return nl.LoadVideoPathC2C().load_video(video=path, **defaults)


def test_input_types_order_and_defaults(nl):
    req = nl.LoadVideoC2C.INPUT_TYPES()["required"]
    assert list(req.keys()) == [
        "video", "force_rate", "custom_width", "custom_height",
        "frame_load_cap", "skip_first_frames", "select_every_nth",
    ]
    assert req["force_rate"][1]["default"] == 0
    assert req["custom_width"][1]["default"] == 0
    assert req["custom_height"][1]["default"] == 0
    assert req["frame_load_cap"][1]["default"] == 0
    assert req["skip_first_frames"][1]["default"] == 0
    assert req["select_every_nth"][1]["default"] == 1

    path_req = nl.LoadVideoPathC2C.INPUT_TYPES()["required"]
    assert list(path_req.keys())[:7] == list(req.keys())
    assert nl.LoadVideoPathC2C.INPUT_TYPES()["optional"]["sequence_fps"][1]["default"] == 24

    img_req = nl.LoadImagesPathC2C.INPUT_TYPES()["required"]
    assert list(img_req.keys()) == ["directory"]
    img_opt = nl.LoadImagesPathC2C.INPUT_TYPES()["optional"]
    assert img_opt["image_load_cap"][1]["default"] == 0
    assert img_opt["skip_first_images"][1]["default"] == 0
    assert img_opt["select_every_nth"][1]["default"] == 1


def test_defaults_shape_and_indices(nl, cfr_clip):
    image, count, _audio, info, handle, mask = _run_path(nl, cfr_clip)
    assert count == 60
    assert image.shape == (60, 256, 256, 3)
    assert image.dtype == torch.float32
    cv = _load_c2c_video()
    indices = [cv.decode_index(image[i].cpu().numpy()) for i in range(60)]
    assert indices == list(range(60))
    assert info["loaded_frame_count"] == 60


def test_skip_nth_cap_selection(nl, cfr_clip):
    image, count, *_ = _run_path(
        nl, cfr_clip, skip_first_frames=5, select_every_nth=3, frame_load_cap=4,
    )
    assert count == 4
    cv = _load_c2c_video()
    got = [cv.decode_index(image[i].cpu().numpy()) for i in range(4)]
    assert got == [5, 8, 11, 14]


def test_force_rate_12(nl, cfr_clip):
    image, count, _a, info, handle, _m = _run_path(nl, cfr_clip, force_rate=12)
    assert info["loaded_fps"] == 12
    cv = _load_c2c_video()
    got = [cv.decode_index(image[i].cpu().numpy()) for i in range(count)]
    want = []
    j = 0
    while math.ceil(j * 24 / 12 - 1e-9) <= 59:
        want.append(math.ceil(j * 24 / 12 - 1e-9))
        j += 1
    assert got == want


def test_custom_width_and_wan_format(nl, cfr_clip):
    image, count, *_ = _run_path(nl, cfr_clip, custom_width=128)
    assert image.shape[1:3] == (128, 128)

    image2, count2, *_ = _run_path(nl, cfr_clip, format="Wan")
    assert count2 % 4 == 1
    assert image2.shape[2] % 8 == 0 and image2.shape[1] % 8 == 0


def test_budget_raises(nl, cfr_clip, monkeypatch):
    pytest.importorskip("psutil")
    import psutil

    tiny = MagicMock()
    tiny.available = 1024
    monkeypatch.setattr(psutil, "virtual_memory", lambda: tiny)
    with pytest.raises(RuntimeError, match="frame_load_cap"):
        _run_path(nl, cfr_clip)


def test_mask_only_linked_decodes(nl, cfr_clip):
    image, count, _audio, _info, _handle, mask = nl.LoadVideoPathC2C().load_video(
        video=cfr_clip,
        force_rate=0,
        custom_width=0,
        custom_height=0,
        frame_load_cap=0,
        skip_first_frames=0,
        select_every_nth=1,
        prompt={"9": {"class_type": "MaskPreview", "inputs": {"mask": ["1", 5]}}},
        unique_id="1",
    )
    assert image is None
    assert mask is not None
    assert hasattr(mask, "shape")


def test_unlinked_image_skips_decode(nl, cfr_clip, monkeypatch):
    def _boom(*a, **k):
        raise AssertionError("decode should not run")

    monkeypatch.setattr(nl, "iter_chunks", _boom)
    image, count, audio, info, handle, mask = nl.LoadVideoPathC2C().load_video(
        video=cfr_clip,
        force_rate=0,
        custom_width=0,
        custom_height=0,
        frame_load_cap=0,
        skip_first_frames=0,
        select_every_nth=1,
        prompt={"9": {"inputs": {"x": ["other", 0]}}},
        unique_id="1",
    )
    assert image is None
    assert mask is None
    assert count == 60
    assert info["loaded_frame_count"] == 60


def test_exr_path_preserves_hdr(nl, exr_seq):
    image, count, *_ = _run_path(nl, exr_seq)
    assert float(image.max()) > 1.0


def test_png_alpha_mask(nl, alpha_png_dir):
    image, mask, count, handle = nl.LoadImagesPathC2C().load_images(
        directory=alpha_png_dir,
        prompt=_linked_prompt("1"),
        unique_id="1",
    )
    assert count == 4
    assert mask.shape[0] == 4
    assert mask.shape[1] == 64
    pytest.importorskip("cv2")
    import cv2

    raw = cv2.imread(str(Path(alpha_png_dir) / "frame_0000.png"), cv2.IMREAD_UNCHANGED)
    alpha = raw[..., 3].astype(np.float32) / 255.0
    expected = 1.0 - alpha
    np.testing.assert_allclose(mask[0].cpu().numpy(), expected, atol=1e-5)


def _audio_rms(wf, sr: int, start_sec: float = 0.0, window_sec: float = 0.1) -> float:
    i0 = int(start_sec * sr)
    i1 = int((start_sec + window_sec) * sr)
    chunk = wf[0, 0, i0:i1]
    return float(torch.sqrt((chunk ** 2).mean()).item())


def _make_offset_audio_clip(cv, path: str, frames: int = 48) -> None:
    pytest.importorskip("av")
    import av
    from av import AudioFrame

    sr = 48000
    container = av.open(path, mode="w")
    try:
        vstream = container.add_stream("libx264", rate=24)
        vstream.width = 256
        vstream.height = 256
        vstream.pix_fmt = "yuv420p"
        astream = container.add_stream("aac", rate=sr)
        astream.layout = "mono"
        for i in range(frames):
            frame = av.VideoFrame.from_ndarray(
                cv.encode_index_pattern(i, 256, 256), format="rgb24",
            )
            frame.pts = i
            for pkt in vstream.encode(frame):
                container.mux(pkt)
        silent = np.zeros(sr, dtype=np.float32)
        tone = (0.25 * np.sin(2 * np.pi * 440.0 * np.arange(sr, dtype=np.float32) / sr)).astype(np.float32)
        wave = np.concatenate([silent, tone])
        aframe = AudioFrame.from_ndarray(
            wave.reshape(1, -1), format="fltp", layout=astream.layout.name,
        )
        aframe.sample_rate = sr
        aframe.pts = 0
        for pkt in astream.encode(aframe):
            container.mux(pkt)
        for pkt in vstream.encode(None):
            container.mux(pkt)
        for pkt in astream.encode(None):
            container.mux(pkt)
    finally:
        container.close()


def test_audio_offset_trim(nl, cv, tmp_path):
    path = str(tmp_path / "offset_audio.mp4")
    _make_offset_audio_clip(cv, path, frames=48)

    _i0, c0, audio0, *_ = _run_path(nl, path, skip_first_frames=0)
    sr0 = audio0["sample_rate"]
    wf0 = audio0["waveform"]
    expect0 = int(c0 / 24.0 * sr0)
    assert abs(wf0.shape[-1] - expect0) <= max(1, int(expect0 * 0.01))
    assert _audio_rms(wf0, sr0, 0.0, 0.1) < 0.01

    _i1, c1, audio1, *_ = _run_path(nl, path, skip_first_frames=24)
    sr1 = audio1["sample_rate"]
    wf1 = audio1["waveform"]
    expect1 = int(c1 / 24.0 * sr1)
    assert abs(wf1.shape[-1] - expect1) <= max(1, int(expect1 * 0.01))
    assert _audio_rms(wf1, sr1, 0.0, 0.1) > 0.05


def test_select_every_nth_keeps_the_audio_span(nl, cv, tmp_path):
    # VHS parity (L7.69): every 2nd frame of a 24 fps clip is a 12 fps clip, and its audio covers the whole span.
    path = str(tmp_path / "nth_audio.mp4")
    _make_offset_audio_clip(cv, path, frames=48)
    _i, count, audio, info, *_ = _run_path(nl, path, select_every_nth=2)
    assert count == 24
    assert info["loaded_fps"] == 12
    sr = audio["sample_rate"]
    expect = int(2.0 * sr)
    assert abs(audio["waveform"].shape[-1] - expect) <= max(1, int(expect * 0.01))


def test_validate_inputs_and_missing_is_changed(nl, tmp_path):
    assert nl.LoadVideoPathC2C.VALIDATE_INPUTS(None) is True
    assert "enter a video" in nl.LoadVideoPathC2C.VALIDATE_INPUTS("")
    missing = str(tmp_path / "nope.mp4")
    assert "nothing found" in nl.LoadVideoPathC2C.VALIDATE_INPUTS(missing)
    assert nl.LoadVideoPathC2C.IS_CHANGED(missing).startswith("missing:")

    assert nl.LoadImagesPathC2C.VALIDATE_INPUTS(None) is True
    assert "enter a directory" in nl.LoadImagesPathC2C.VALIDATE_INPUTS("")
    assert "nothing found" in nl.LoadImagesPathC2C.VALIDATE_INPUTS(missing)


def test_is_changed_on_mtime(nl, cfr_clip):
    before = nl.LoadVideoPathC2C.IS_CHANGED(cfr_clip)
    os.utime(cfr_clip, None)
    after = nl.LoadVideoPathC2C.IS_CHANGED(cfr_clip)
    assert before != after


def test_video_info_ten_fields(nl):
    info = {
        "source_fps": 24.0,
        "source_frame_count": 60,
        "source_duration": 2.5,
        "source_width": 256,
        "source_height": 256,
        "loaded_fps": 12.0,
        "loaded_frame_count": 30,
        "loaded_duration": 2.5,
        "loaded_width": 128,
        "loaded_height": 128,
    }
    out = nl.VideoInfoC2C().get_video_info(info)
    assert out == (
        24.0, 60, 2.5, 256, 256,
        12.0, 30, 2.5, 128, 128,
    )


def test_is_output_linked_string_uid(nl):
    assert nl.is_output_linked({"n": {"inputs": {"a": ["42", 0]}}}, 42, 0)
    assert not nl.is_output_linked({"n": {"inputs": {"a": ["42", 1]}}}, 42, 0)
    assert nl.is_output_linked(None, "1", 0)
