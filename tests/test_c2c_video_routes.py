"""Tests for C2C video loader probe/preview routes (S1b)."""

from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path

import pytest

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


def _load_routes(cv, tmp_path, monkeypatch):
    temp = tmp_path / "temp"
    temp.mkdir()
    fp = types.SimpleNamespace(
        get_temp_directory=lambda: str(temp),
        get_input_directory=lambda: str(tmp_path / "input"),
        get_annotated_filepath=lambda p: os.path.join(str(tmp_path / "input"), os.path.basename(p)),
    )
    monkeypatch.setitem(sys.modules, "folder_paths", fp)

    name = "c2c_video.routes"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            name,
            C2C_VIDEO / "routes.py",
            submodule_search_locations=[str(C2C_VIDEO)],
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[name]


def _load_nodes_load(cv, tmp_path, monkeypatch):
    name = "c2c_video.nodes_load"
    if name in sys.modules:
        return sys.modules[name]
    inp = tmp_path / "input"
    inp.mkdir(exist_ok=True)
    fp = types.SimpleNamespace(
        get_input_directory=lambda: str(inp),
        get_annotated_filepath=lambda p: os.path.join(str(inp), os.path.basename(p)),
    )
    monkeypatch.setitem(sys.modules, "folder_paths", fp)
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
def routes(cv, tmp_path, monkeypatch):
    return _load_routes(cv, tmp_path, monkeypatch)


@pytest.fixture
def nl(cv, tmp_path, monkeypatch):
    return _load_nodes_load(cv, tmp_path, monkeypatch)


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


def _video_params(path, **kw):
    base = {
        "force_rate": "0",
        "skip_first_frames": "0",
        "select_every_nth": "1",
        "frame_load_cap": "0",
        "format": "None",
        "sequence_fps": "24",
        "max_frames": "240",
    }
    base.update({k: str(v) for k, v in kw.items()})
    return {"path": path, **base}


def test_probe_payload_cfr_clip(routes, cfr_clip):
    params = routes.parse_request_params(_video_params(cfr_clip))
    payload = routes.probe_payload(params)
    assert payload["kind"] == "file"
    assert payload["width"] == 256
    assert payload["height"] == 256
    assert payload["frame_count"] == 60
    assert payload["selection"]["count"] == 60
    assert payload["codec"]
    assert "matrix" in payload["colour"]
    assert payload["index_mode"] in ("auto", "exact")


def test_probe_selection_matches_node_handle(routes, nl, cfr_clip):
    params = routes.parse_request_params(_video_params(
        cfr_clip, skip_first_frames="5", select_every_nth="3", frame_load_cap="4",
    ))
    payload = routes.probe_payload(params)
    raw = nl.resolve_probe(cfr_clip)
    _, loaded = nl.build_loaded_handle(
        raw,
        force_rate=0,
        skip_first_frames=5,
        select_every_nth=3,
        frame_load_cap=4,
        format_name="None",
        custom_width=0,
        custom_height=0,
        vae=None,
    )
    sel = list(loaded.selected_indices())
    assert payload["selection"]["count"] == len(loaded)
    assert payload["selection"]["first_source"] == sel[0]
    assert payload["selection"]["last_source"] == sel[-1]


def test_ensure_preview_creates_proxy(routes, cfr_clip):
    pytest.importorskip("av")
    import av

    params = routes.parse_request_params(_video_params(cfr_clip, max_frames="30"))
    path = routes.ensure_preview(params)
    assert os.path.isfile(path)
    with av.open(path) as container:
        stream = container.streams.video[0]
        frames = sum(1 for _ in container.decode(stream))
    assert frames == 30
    assert frames <= 30
    with av.open(path) as container:
        stream = container.streams.video[0]
        w, h = stream.codec_context.width, stream.codec_context.height
    assert max(w, h) <= 640
    assert w % 2 == 0 and h % 2 == 0


def test_preview_cache_hit(routes, cfr_clip):
    params = routes.parse_request_params(_video_params(cfr_clip, max_frames="16"))
    routes._encode_calls = 0
    p1 = routes.ensure_preview(params)
    mtime1 = os.path.getmtime(p1)
    calls_after_first = routes._encode_calls
    assert calls_after_first >= 1
    p2 = routes.ensure_preview(params)
    assert p1 == p2
    assert os.path.getmtime(p2) == mtime1
    assert routes._encode_calls == calls_after_first


def test_bad_path_error(routes, tmp_path):
    missing = str(tmp_path / "nope.mp4")
    params = routes.parse_request_params({"path": missing})
    with pytest.raises(routes.RouteError) as exc:
        routes.probe_payload(params)
    assert exc.value.status == 404
    assert "Traceback" not in exc.value.message
    assert exc.value.message


def test_linear_exr_not_black(routes, exr_seq):
    pytest.importorskip("av")
    params = routes.parse_request_params({
        "directory": exr_seq,
        "skip_first_images": "0",
        "select_every_nth": "1",
        "image_load_cap": "0",
        "max_frames": "8",
    })
    path = routes.ensure_preview(params)
    import av

    peak = 0.0
    with av.open(path) as container:
        for frame in container.decode(container.streams.video[0]):
            arr = frame.to_ndarray(format="rgb24")
            peak = max(peak, float(arr.max()) / 255.0)
    assert peak > 0.05


# ── review fixes: this is an HTTP route, so it reads media and nothing else ──

def test_upload_name_cannot_leave_the_input_folder(routes, cfr_clip, tmp_path, monkeypatch):
    """core joins the name onto input/ like this; "../" must not escape it."""
    import shutil
    inp = tmp_path / "input"
    inp.mkdir(exist_ok=True)
    outside = tmp_path / "outside.mp4"
    shutil.copy(cfr_clip, outside)
    fp = sys.modules["folder_paths"]
    monkeypatch.setattr(fp, "get_annotated_filepath", lambda p: os.path.normpath(os.path.join(str(inp), p)))
    params = routes.parse_request_params({"filename": "../outside.mp4", "type": "input"})
    with pytest.raises(routes.RouteError) as err:
        routes.probe_payload(params)
    assert err.value.status == 400 and "input folder" in err.value.message
    shutil.copy(cfr_clip, inp / "inside.mp4")
    ok = routes.probe_payload(routes.parse_request_params({"filename": "inside.mp4", "type": "input"}))
    assert ok["selection"]["count"] == 60


def test_path_route_refuses_non_media_files(routes, tmp_path):
    secret = tmp_path / "notes.txt"
    secret.write_text("not a video")
    with pytest.raises(routes.RouteError) as err:
        routes.probe_payload(routes.parse_request_params(_video_params(str(secret))))
    assert err.value.status == 400 and "not a video" in err.value.message


def test_max_frames_is_bounded(routes, cfr_clip):
    params = routes.parse_request_params(_video_params(cfr_clip, max_frames=10**9))
    assert params["max_frames"] == routes._MAX_FRAMES_CAP


def test_preview_is_faststart(routes, cfr_clip):
    """The index (moov) sits before the media (mdat), so a browser plays it
    before the download finishes."""
    path = routes.ensure_preview(routes.parse_request_params(_video_params(cfr_clip)))
    if not path.endswith(".mp4"):
        pytest.skip("no H.264 encoder: webm preview")
    head = open(path, "rb").read()
    assert 0 <= head.find(b"moov") < head.find(b"mdat")


def test_alpha_source_preview(routes, cv, tmp_path):
    clip = str(tmp_path / "prores4444.mov")
    cv.make_test_clip("prores4444", clip, frames=6, width=256, height=64)
    path = routes.ensure_preview(routes.parse_request_params(_video_params(clip)))
    import av
    with av.open(path) as c:
        assert sum(1 for _ in c.decode(video=0)) == 6


def test_sequence_colour_tags(routes, exr_seq, tmp_path):
    """Float EXR is scene-linear (the preview applies a view transform); PNG is
    display-encoded sRGB. Neither has a YUV matrix."""
    exr = routes.probe_payload(routes.parse_request_params(_video_params(exr_seq)))
    assert exr["colour"]["transfer"] == "linear" and exr["colour"]["matrix"] == "rgb"
    import cv2
    import numpy as np
    d = tmp_path / "pngs"
    d.mkdir()
    for i in range(3):
        cv2.imwrite(str(d / f"p_{i:04d}.png"), np.full((32, 48, 3), 90, np.uint8))
    png = routes.probe_payload(routes.parse_request_params({"directory": str(d)}))
    assert png["colour"]["transfer"] == "srgb" and png["colour"]["matrix"] == "rgb"
