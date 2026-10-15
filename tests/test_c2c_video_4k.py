"""L2.53 (owner A14): the 4K pack's fast path, presets, seconds, resolution presets and audio option in Load Video (C2C)
/ Load Video Path (C2C).

The fast decoder must give the exact reader's frames: same frames (burned-in indices), same order, same pixels.
``rgb48`` (what auto uses) is checked BIT-IDENTICAL; ``nv12`` (GPU conversion, decoder=fast on CUDA - run here on the
CPU, same arithmetic) is checked for identical frames and a measured colour bound."""
from __future__ import annotations

import json
import os
import subprocess

import numpy as np
import pytest
import torch

from tests.test_c2c_video_nodes import (  # noqa: F401 - fixtures used by name
    _clean_video_state, _linked_prompt, _load_c2c_video, _private_index_cache, _run_path, cfr_clip, cv, exr_seq, nl,
)


@pytest.fixture
def fd(nl):
    import c2c_video.fast_decode as fd

    fd._FFMPEG.clear()
    yield fd
    fd._FFMPEG.clear()


def _ffmpeg():
    import c2c_video.fast_decode as fd

    return fd.find_ffmpeg()


needs_ffmpeg = pytest.mark.skipif(_load_c2c_video() is None or _ffmpeg() is None,
                                  reason="ffmpeg is not installed (the fast path needs it)")


def _exact(h):
    from c2c_video.reader import iter_chunks

    return np.concatenate(list(iter_chunks(h, 7, fmt="float32")))


def _fast(fd, h, mode="rgb48"):
    return np.concatenate(list(fd.iter_chunks_fast(h, 7, mode, device="cpu")))


@pytest.fixture
def vfr_clip(cv, tmp_path):
    path = str(tmp_path / "vfr.mp4")
    cv.make_test_clip("h264_vfr", path, frames=60, width=256, height=256)
    return path


@pytest.fixture
def long_clip(cv, tmp_path):
    path = str(tmp_path / "long.mp4")
    cv.make_test_clip("h264_cfr_b", path, frames=120, width=256, height=256)   # 5 s: the fast path seeks
    return path


@pytest.fixture
def colour_clip(tmp_path):
    """Saturated colour blocks with hard edges (where chroma handling shows), 8-bit 4:2:0, tagged BT.709."""
    import av

    from c2c_video.testmedia import _rgb_to_yuv, _tag_bt709

    path = str(tmp_path / "colour.mp4")
    rng = np.random.default_rng(3)
    c = av.open(path, mode="w")
    try:
        st = c.add_stream("libx264", rate=24)
        st.width, st.height, st.pix_fmt = 256, 144, "yuv420p"
        st.options = {"crf": "12"}
        _tag_bt709(st)
        for i in range(8):
            blocks = rng.integers(0, 256, size=(9, 16, 3), dtype=np.uint8)
            fr = _rgb_to_yuv(np.ascontiguousarray(np.kron(blocks, np.ones((16, 16, 1), np.uint8))), "yuv420p")
            fr.pts = i
            for p in st.encode(fr):
                c.mux(p)
        for p in st.encode(None):
            c.mux(p)
    finally:
        c.close()
    return path


# ── the fast reader gives the exact reader's frames ───────────────────────────────────────────────────────────────
@needs_ffmpeg
@pytest.mark.parametrize("name", ["whole", "skip_nth_cap", "late", "rate12", "rate30", "scaled_odd", "crop",
                                  "crop_scale"])
@pytest.mark.parametrize("clip", ["cfr_clip", "vfr_clip", "long_clip"])
def test_fast_rgb48_is_bit_identical_to_exact(fd, cv, clip, name, request):
    from c2c_video.nodes_load import source_times_for_handle

    raw = cv.probe_file(request.getfixturevalue(clip))
    h = {
        "whole": lambda: raw,
        "skip_nth_cap": lambda: raw.trimmed(5).every(3).capped(10),
        "late": lambda: raw.trimmed(raw.frame_count - 25).capped(20),
        "rate12": lambda: raw.retimed(12, source_times_for_handle(raw)),
        "rate30": lambda: raw.retimed(30, source_times_for_handle(raw)).capped(40),     # repeats frames
        "scaled_odd": lambda: raw.scaled(101, 77),
        "crop": lambda: raw.cropped(0.25, 0.125, 0.5, 0.5),
        "crop_scale": lambda: raw.cropped(0.1, 0.1, 0.5, 0.7).scaled(64, 90),
    }[name]()
    want, got = _exact(h), _fast(fd, h)
    assert got.shape == want.shape
    assert np.array_equal(got, want)
    if not h.is_scaled and h.crop is None:
        assert [cv.decode_index(f) for f in got] == list(h.selected_indices())


@needs_ffmpeg
def test_fast_rgb48_rotation_and_mpegts(fd, cv, cfr_clip, long_clip, tmp_path):
    exe = _ffmpeg()
    rot = str(tmp_path / "rot.mp4")
    ts = str(tmp_path / "long.ts")
    subprocess.run([exe, "-v", "error", "-y", "-display_rotation", "90", "-i", cfr_clip, "-c", "copy", rot], check=True)
    subprocess.run([exe, "-v", "error", "-y", "-i", long_clip, "-c", "copy", ts], check=True)
    r = cv.probe_file(rot)
    assert r.rotation in (90, 270)
    for h in (r, r.trimmed(7).every(2).scaled(99, 51)):
        assert np.array_equal(_fast(fd, h), _exact(h))
    t = cv.probe_file(ts)                                   # MPEG-TS: timestamps start at 1.4 s, seeks overshoot
    for h in (t, t.trimmed(80).capped(12)):
        got = _fast(fd, h)
        assert np.array_equal(got, _exact(h))
        assert [cv.decode_index(f) for f in got] == list(h.selected_indices())


@needs_ffmpeg
def test_fast_colour_identical_on_colour_edges(fd, cv, colour_clip):
    h = cv.probe_file(colour_clip)
    assert np.array_equal(_fast(fd, h), _exact(h))
    assert np.array_equal(_fast(fd, h.scaled(171, 97)), _exact(h.scaled(171, 97)))


@needs_ffmpeg
def test_nv12_gpu_conversion_same_frames_measured_colour(fd, cv, cfr_clip, colour_clip):
    """decoder=fast on CUDA: nv12 + our chroma upsample. Run on the CPU here (same arithmetic). Frames identical;
    colour within the bound measured on 1080p / 4K testsrc2 (mean 0.47/255, max 25.7/255 at saturated edges)."""
    h = cv.probe_file(cfr_clip).trimmed(3).every(4)
    got = _fast(fd, h, "nv12")
    assert [cv.decode_index(f) for f in got] == list(h.selected_indices())
    c = cv.probe_file(colour_clip)
    d = np.abs(_fast(fd, c, "nv12") - _exact(c))
    assert d.max() * 255 <= 32.0 and d.mean() * 255 <= 1.0, (d.max() * 255, d.mean() * 255)


@needs_ffmpeg
def test_selftest_passes_with_this_ffmpeg(fd):
    ok, msg = fd.selftest()
    assert ok, msg


# ── the decision ─────────────────────────────────────────────────────────────────────────────────────────────────
def test_plan_exact_for_what_the_fast_path_does_not_take(fd, cv, tmp_path, exr_seq):
    seq = cv.probe_sequence(exr_seq)
    assert fd.plan_decoder(seq, "fast").mode == "exact"
    p10 = str(tmp_path / "ten.mp4")
    cv.make_test_clip("h264_10bit", p10, frames=4)
    assert "10-bit" in fd.plan_decoder(cv.probe_file(p10), "fast").note
    pr = str(tmp_path / "alpha.mov")
    cv.make_test_clip("prores4444", pr, frames=3)
    assert fd.plan_decoder(cv.probe_file(pr), "auto").mode == "exact"
    assert fd.plan_decoder(seq, "exact").note == ""


@needs_ffmpeg
def test_plan_auto_is_exact_unless_allowed_and_fast_on_cuda_is_nv12(fd, cv, cfr_clip, monkeypatch):
    """auto = the fast path only where it is faster with the same colour. Measured (docs/evidence/L2.53): the identical
    ffmpeg path is not faster than the exact reader on the CPU and the GPU path is not identical, so auto is exact;
    C2C_VIDEO_AUTO opts a machine in."""
    h = cv.probe_file(cfr_clip)
    monkeypatch.setattr(fd, "AUTO_MIN_PIXELS", 0)
    assert fd.plan_decoder(h, "auto").mode == "exact"
    assert fd.plan_decoder(h, "auto", cuda=True).mode == "exact"
    monkeypatch.setenv("C2C_VIDEO_AUTO", "fast")
    assert fd.plan_decoder(h, "auto").mode == "rgb48"                  # after the self-check proved it identical
    assert fd.plan_decoder(h, "auto", cuda=True).mode == "rgb48"
    monkeypatch.setenv("C2C_VIDEO_AUTO", "gpu")
    assert fd.plan_decoder(h, "auto", cuda=True).mode == "nv12"
    monkeypatch.setattr(fd, "AUTO_MIN_PIXELS", 10 ** 12)
    assert fd.plan_decoder(h, "auto", cuda=True).mode == "exact"       # a small load: start-up outweighs any gain
    monkeypatch.delenv("C2C_VIDEO_AUTO")
    assert fd.plan_decoder(h, "fast").mode == "rgb48"
    assert fd.plan_decoder(h, "fast", cuda=True).mode == "nv12"
    assert fd.plan_decoder(h.cropped(0, 0, 0.5, 0.5), "fast", cuda=True).mode == "rgb48"
    monkeypatch.setenv("C2C_VIDEO_FAST", "rgb48")
    assert fd.plan_decoder(h, "fast", cuda=True).mode == "rgb48"
    monkeypatch.setenv("C2C_VIDEO_FAST", "0")                          # the kill switch
    assert fd.plan_decoder(h, "fast", cuda=True).mode == "exact"


def test_no_ffmpeg_means_exact(fd, cv, cfr_clip, monkeypatch):
    fd._FFMPEG["path"] = None
    plan = fd.plan_decoder(cv.probe_file(cfr_clip), "fast")
    assert plan.mode == "exact" and "ffmpeg" in plan.note


@needs_ffmpeg
def test_node_fast_equals_exact_and_info_says_which(nl, cfr_clip):
    exact = _run_path(nl, cfr_clip, decoder="exact", skip_first_frames=4, select_every_nth=2, frame_load_cap=9)
    fast = _run_path(nl, cfr_clip, decoder="fast", skip_first_frames=4, select_every_nth=2, frame_load_cap=9)
    assert torch.equal(exact[0], fast[0])
    assert json.loads(exact[6])["decoder"]["used"].startswith("exact")
    assert json.loads(fast[6])["decoder"]["used"].startswith("fast")


def test_fast_failure_falls_back_to_the_exact_frames(nl, cfr_clip, monkeypatch):
    import c2c_video.nodes_load as m

    # the test harness loads nodes_load as a package with its own copies of its submodules: use ITS classes
    DecoderPlan, FastDecodeError = m.DecoderPlan, m.FastDecodeError

    def broken(h, size, plan):
        raise FastDecodeError("simulated: frame 3 was not decoded")
        yield  # pragma: no cover

    monkeypatch.setattr(m, "plan_decoder", lambda h, d, cuda=False: DecoderPlan(d, "rgb48"))
    monkeypatch.setattr(m, "iter_chunks_for", broken)
    out = _run_path(nl, cfr_clip, decoder="fast", frame_load_cap=6)
    ref = _run_path(nl, cfr_clip, decoder="exact", frame_load_cap=6)
    assert torch.equal(out[0], ref[0])
    info = json.loads(out[6])["decoder"]
    assert info["used"].startswith("exact") and "simulated" in info["note"]


# ── presets, seconds, resolution, divisible_by, audio ─────────────────────────────────────────────────────────────
def test_old_presets_unchanged_and_minimax_h3_added(nl):
    assert nl.FORMAT_PRESETS["LTXV"] == {"rate": 24, "dim_multiple": 32, "frames": (8, 1)}
    assert nl.FORMAT_PRESETS["Wan"] == {"rate": 16, "dim_multiple": 8, "frames": (4, 1)}
    assert nl.FORMAT_PRESETS["Hunyuan"] == {"rate": 24, "dim_multiple": 16, "frames": (4, 1)}
    assert nl.FORMAT_PRESETS["MiniMax H3"] == {"rate": 24, "dim_multiple": 32, "frames": (17, 5)}
    assert nl.FORMAT_CHOICES[:7] == ["None", "AnimateDiff", "Mochi", "LTXV", "Hunyuan", "Cosmos", "Wan"]


def test_minimax_h3_snaps_frames_and_size(nl, cfr_clip, cv):
    image, count, *_ = _run_path(nl, cfr_clip, format="MiniMax H3")
    assert count == 56 and count % 17 == 5                 # 60 frames -> 56 (5, 22, 39, 56)
    assert image.shape[1] % 32 == 0 and image.shape[2] % 32 == 0
    image, count, *_ = _run_path(nl, cfr_clip, format="MiniMax H3", frame_load_cap=30)
    assert count == 22
    with pytest.raises(RuntimeError, match="at least 5"):
        _run_path(nl, cfr_clip, format="MiniMax H3", frame_load_cap=4)


def test_seconds_cap(nl, cfr_clip, cv):
    image, count, *_ = _run_path(nl, cfr_clip, seconds_cap=1.0)                 # 24 fps
    assert count == 24
    assert [cv.decode_index(image[i].numpy()) for i in (0, 23)] == [0, 23]
    _i, count, *_ = _run_path(nl, cfr_clip, seconds_cap=1.0, skip_first_frames=10, select_every_nth=2)
    assert count == 12                                                           # frames 10..33 every 2nd
    _i, count, *_ = _run_path(nl, cfr_clip, seconds_cap=1.0, frame_load_cap=7)
    assert count == 7                                                            # the smaller cap wins
    _i, count, *_ = _run_path(nl, cfr_clip, seconds_cap=2.0, force_rate=12)
    assert count == 24                                                           # seconds at the forced rate
    _i, count, *_ = _run_path(nl, cfr_clip, seconds_cap=1.0, format="MiniMax H3")
    assert count == 22                                                           # 24 -> snapped to 17n+5
    assert nl.seconds_to_frames(2.7, 24) == 64 and nl.seconds_to_frames(0, 24) == 0


def test_resolution_presets_longest_side(nl):
    f = nl.resolution_target_size
    assert f(1920, 1080, "custom", 0, 0, 1) == (1920, 1080)
    assert f(1920, 1080, "custom", 640, 0, 1) == (640, 360)
    assert f(1920, 1080, "source", 640, 0, 1) == (1920, 1080)          # source ignores the custom sizes
    assert f(1920, 1080, "720p", 0, 0, 1) == (1280, 720)
    assert f(1080, 1920, "720p", 0, 0, 1) == (720, 1280)               # portrait: longest side
    assert f(3840, 2160, "480p", 0, 0, 1) == (854, 480)
    assert f(1920, 1080, "2160p (4K)", 0, 0, 32) == (3840, 2176)
    assert f(1280, 720, "1024 wide", 0, 0, 16) == (1024, 576)
    assert f(1920, 1080, "not a preset", 0, 0, 1) == (1920, 1080)


def test_resolution_and_divisible_by_on_the_node(nl, cfr_clip):
    img, *_ = _run_path(nl, cfr_clip, resolution="480p", frame_load_cap=2)
    assert tuple(img.shape[1:3]) == (854, 854)
    img, *_ = _run_path(nl, cfr_clip, custom_width=100, frame_load_cap=2)
    assert tuple(img.shape[1:3]) == (100, 100)                          # divisible_by 1: as before
    img, *_ = _run_path(nl, cfr_clip, custom_width=100, divisible_by=16, frame_load_cap=2)
    assert tuple(img.shape[1:3]) == (96, 96)
    img, *_ = _run_path(nl, cfr_clip, custom_width=100, divisible_by=8, format="MiniMax H3", frame_load_cap=5)
    assert tuple(img.shape[1:3]) == (96, 96)                            # the preset raises 8 to 32


def test_audio_when_missing(nl, cfr_clip):
    _i, count, audio, *_rest = _run_path(nl, cfr_clip, frame_load_cap=12)
    assert audio is not None and audio["waveform"].abs().max() == 0       # silence (the old behaviour, default)
    assert audio["waveform"].shape[-1] == round(12 / 24 * audio["sample_rate"])
    _i, count, audio, *_rest = _run_path(nl, cfr_clip, frame_load_cap=12, audio_when_missing="none")
    assert audio is None
    assert json.loads(_rest[-1])["audio"] == "none"


def test_new_inputs_are_appended_with_defaults_that_change_nothing(nl, cfr_clip):
    for cls, before in ((nl.LoadVideoC2C, ["vae", "format", "crop_x", "crop_y", "crop_w", "crop_h"]),
                        (nl.LoadVideoPathC2C, ["vae", "format", "sequence_fps", "crop_x", "crop_y", "crop_w",
                                               "crop_h"])):
        t = cls.INPUT_TYPES()
        opt = list(t["optional"])
        assert opt == before + ["decoder", "seconds_cap", "resolution", "divisible_by", "audio_when_missing"]
        assert [t["optional"][k][1]["default"] for k in opt[len(before):]] == ["auto", 0.0, "custom", 1, "silence"]
        assert cls.RETURN_NAMES[:6] == ("IMAGE", "frame_count", "audio", "video_info", "video", "mask")
        assert cls.RETURN_NAMES[6:] == ("info",)
    # a workflow saved before L2.53 sends only the old inputs: same frames as with every new input at its default
    old = _run_path(nl, cfr_clip, skip_first_frames=3, select_every_nth=2, frame_load_cap=9, custom_width=128)
    new = _run_path(nl, cfr_clip, skip_first_frames=3, select_every_nth=2, frame_load_cap=9, custom_width=128,
                    decoder="auto", seconds_cap=0.0, resolution="custom", divisible_by=1, audio_when_missing="silence")
    assert torch.equal(old[0], new[0]) and old[1] == new[1] and old[3] == new[3]


def test_info_output(nl, cfr_clip):
    info = json.loads(_run_path(nl, cfr_clip, skip_first_frames=12, frame_load_cap=24, custom_width=128)[6])
    assert info["source"]["width"] == 256 and info["source"]["frames"] == 60
    assert info["loaded"]["frames"] == 24 and info["loaded"]["width"] == 128
    assert info["loaded"]["start_seconds"] == 0.5 and info["loaded"]["end_seconds"] == 1.5
    assert info["decoder"]["asked"] == "auto"


def test_probe_reports_what_a_run_loads(cv, cfr_clip, tmp_path, monkeypatch):
    from tests.test_c2c_video_routes import _load_routes   # the routes tests' own loader (module state is shared)

    routes = _load_routes(cv, tmp_path, monkeypatch)

    params = routes.parse_request_params({"path": cfr_clip, "skip_first_frames": "12", "frame_load_cap": "0",
                                          "seconds_cap": "1", "custom_width": "128", "format": "MiniMax H3",
                                          "decoder": "exact"})
    data = routes.probe_payload(params)
    assert data["selection"]["count"] == 22                             # 1 s = 24 frames -> 22 (17n+5)
    assert data["load"]["frames"] == 22 and data["load"]["width"] == 128
    assert data["load"]["start_seconds"] == 0.5
    assert data["load"]["decoder"]["used"].startswith("exact")
    data2 = routes.probe_payload(routes.parse_request_params({"path": cfr_clip}))    # an older front end
    assert data2["selection"]["count"] == 60 and data2["load"]["frames"] == 60
