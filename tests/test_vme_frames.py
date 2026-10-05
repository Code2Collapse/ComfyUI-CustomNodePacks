"""CPU tests for VME frame planning + thumbnails (L2.03)."""

from __future__ import annotations

import importlib.util
import os
import shutil
import sys
import types
from pathlib import Path

import numpy as np
import pytest

PACK = Path(__file__).resolve().parents[1]
C2C_VIDEO = PACK / "nodes" / "c2c_video"

FRAME_COUNTS = (1, 2, 31, 32, 33, 125, 500)


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


def _load_frames(cv, tmp_path, monkeypatch):
    inp = tmp_path / "input"
    inp.mkdir(exist_ok=True)
    fp = types.SimpleNamespace(
        get_input_directory=lambda: str(inp),
        get_annotated_filepath=lambda p: os.path.join(str(inp), os.path.basename(p)),
        get_temp_directory=lambda: str(tmp_path / "temp"),
    )
    monkeypatch.setitem(sys.modules, "folder_paths", fp)
    name = "c2c_video.frames"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name,
        C2C_VIDEO / "frames.py",
        submodule_search_locations=[str(C2C_VIDEO)],
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


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


def _default_video_widgets(path: str) -> dict:
    return {
        "video": path,
        "force_rate": 0,
        "skip_first_frames": 0,
        "select_every_nth": 1,
        "frame_load_cap": 0,
        "format": "None",
        "custom_width": 0,
        "custom_height": 0,
        "sequence_fps": 24,
    }


def _loaded_count(nl, path: str, **widgets) -> int:
    w = _default_video_widgets(path)
    w.update(widgets)
    raw = nl.resolve_probe(path, sequence_fps=float(w["sequence_fps"]))
    _, loaded = nl.build_loaded_handle(
        raw,
        force_rate=float(w["force_rate"]),
        skip_first_frames=int(w["skip_first_frames"]),
        select_every_nth=int(w["select_every_nth"]),
        frame_load_cap=int(w["frame_load_cap"]),
        format_name=str(w["format"]),
        custom_width=int(w["custom_width"]),
        custom_height=int(w["custom_height"]),
        vae=None,
    )
    return len(loaded)


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
def frames(cv, tmp_path, monkeypatch):
    mod = _load_frames(cv, tmp_path, monkeypatch)
    mod.reset_thumb_cache()
    yield mod
    mod.reset_thumb_cache()


@pytest.fixture
def nl(cv, tmp_path, monkeypatch):
    return _load_nodes_load(cv, tmp_path, monkeypatch)


@pytest.fixture
def routes(cv, tmp_path, monkeypatch):
    return _load_routes(cv, tmp_path, monkeypatch)


def _make_clip(cv, tmp_path, n: int) -> str:
    path = str(tmp_path / f"clip_{n}.mp4")
    cv.make_test_clip("h264_cfr_b", path, frames=n, width=256, height=256)
    return path


@pytest.mark.parametrize("n", FRAME_COUNTS)
def test_plan_count_matches_loader_default(frames, nl, cv, tmp_path, n):
    path = _make_clip(cv, tmp_path, n)
    widgets = _default_video_widgets(path)
    plan = frames.plan_frames("LoadVideoPathC2C", widgets)
    assert plan["count"] == _loaded_count(nl, path)
    assert plan["count"] == n
    assert plan["width"] == 256
    assert plan["height"] == 256


def test_plan_count_with_trim_nth_cap_force_rate(frames, nl, cv, tmp_path):
    path = _make_clip(cv, tmp_path, 125)
    raw = nl.resolve_probe(path)
    half_fps = float(raw.fps) / 2.0
    widgets = _default_video_widgets(path)
    widgets.update({
        "skip_first_frames": 3,
        "select_every_nth": 2,
        "frame_load_cap": 10,
        "force_rate": half_fps,
    })
    plan = frames.plan_frames("LoadVideoPathC2C", widgets)
    expected = _loaded_count(
        nl, path,
        skip_first_frames=3,
        select_every_nth=2,
        frame_load_cap=10,
        force_rate=half_fps,
    )
    assert plan["count"] == expected


@pytest.mark.parametrize("node_type", ["LoadVideoPathC2C", "VHS_LoadVideoPath"])
def test_vhs_video_path_matches_c2c(frames, cv, tmp_path, node_type):
    path = _make_clip(cv, tmp_path, 33)
    widgets = _default_video_widgets(path)
    widgets["skip_first_frames"] = 2
    widgets["select_every_nth"] = 2
    c2c = frames.plan_frames("LoadVideoPathC2C", widgets)
    vhs = frames.plan_frames(node_type, widgets)
    assert vhs["count"] == c2c["count"]
    assert vhs["width"] == c2c["width"]
    assert vhs["height"] == c2c["height"]


def test_thumb_decodes_selected_batch_index(frames, nl, cv, tmp_path):
    path = _make_clip(cv, tmp_path, 32)
    widgets = _default_video_widgets(path)
    widgets["skip_first_frames"] = 1
    widgets["select_every_nth"] = 2
    plan = frames.plan_frames("LoadVideoPathC2C", widgets)
    raw = nl.resolve_probe(path)
    _, loaded = nl.build_loaded_handle(
        raw,
        force_rate=0,
        skip_first_frames=1,
        select_every_nth=2,
        frame_load_cap=0,
        format_name="None",
        custom_width=0,
        custom_height=0,
        vae=None,
    )
    sel = list(loaded.selected_indices())
    for i in (0, len(sel) // 2, len(sel) - 1):
        png = frames.encode_thumb(plan["token"], i, max_px=0, fmt="png")
        try:
            from PIL import Image
            import io

            arr = np.asarray(Image.open(io.BytesIO(png)).convert("RGB"))
        except ImportError:
            import cv2

            arr = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        assert cv.decode_index(arr) == sel[i]


def test_load_images_path_directory(frames, nl, cv, tmp_path):
    import cv2

    d = tmp_path / "png_seq"
    d.mkdir()
    n = 12
    for i in range(n):
        rgb = cv.encode_index_pattern(i, 256, 256)
        cv2.imwrite(str(d / f"frame_{i:04d}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    widgets = {
        "directory": str(d),
        "image_load_cap": 0,
        "skip_first_images": 0,
        "select_every_nth": 1,
    }
    plan = frames.plan_frames("LoadImagesPathC2C", widgets)
    loaded = nl.build_images_handle(str(d), skip_first_images=0, select_every_nth=1, image_load_cap=0)
    assert plan["count"] == len(loaded) == n


def test_upload_path_escape_rejected(frames, cv, tmp_path, monkeypatch):
    inp = tmp_path / "input"
    inp.mkdir(exist_ok=True)
    outside = tmp_path / "outside.mp4"
    cv.make_test_clip("h264_cfr_b", str(outside), frames=8, width=256, height=256)
    fp = sys.modules["folder_paths"]
    monkeypatch.setattr(fp, "get_annotated_filepath", lambda p: os.path.normpath(os.path.join(str(inp), p)))
    widgets = {
        "video": "../outside.mp4",
        "force_rate": 0,
        "skip_first_frames": 0,
        "select_every_nth": 1,
        "frame_load_cap": 0,
        "format": "None",
        "custom_width": 0,
        "custom_height": 0,
    }
    with pytest.raises(frames.PlanError) as err:
        frames.plan_frames("LoadVideoC2C", widgets)
    assert err.value.status == 400


def test_non_media_file_rejected(frames, tmp_path):
    bad = tmp_path / "notes.txt"
    bad.write_text("not video")
    with pytest.raises(frames.PlanError) as err:
        frames.plan_frames("LoadVideoPathC2C", _default_video_widgets(str(bad)))
    assert err.value.status == 400


def test_thumb_lru_stays_under_byte_budget(frames, cv, tmp_path, monkeypatch):
    monkeypatch.setattr(frames, "_THUMB_BUDGET", 256 * 1024)
    frames.reset_thumb_cache()
    path = _make_clip(cv, tmp_path, 8)
    plan = frames.plan_frames("LoadVideoPathC2C", _default_video_widgets(path))
    for i in range(8):
        frames.encode_thumb(plan["token"], i, max_px=0, fmt="png")
    count, total = frames.thumb_cache_stats()
    assert count > 0
    assert total <= frames._THUMB_BUDGET


def test_out_of_range_thumb_index(frames, cv, tmp_path):
    path = _make_clip(cv, tmp_path, 4)
    plan = frames.plan_frames("LoadVideoPathC2C", _default_video_widgets(path))
    with pytest.raises(frames.PlanError) as err:
        frames.encode_thumb(plan["token"], 99, max_px=64, fmt="jpeg")
    assert err.value.status == 400
