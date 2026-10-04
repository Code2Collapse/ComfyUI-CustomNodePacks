"""c2c_video against files made by the ffmpeg CLI - real muxers and encoders,
not PyAV's. Each case once failed or was blind in a way synthetic PyAV clips
could not show:

* MPEG-TS seeks land on the NEXT keyframe (even for frame 0, even for the
  last GOP, where they land at end of file);
* FFmpeg 8 gives the encoder a 1/framerate clock, so "VFR" made without
  -enc_time_base is silently constant-rate;
* dropped frames keep every timestamp on the frame grid;
* the display matrix (phone portrait video) is only exposed per frame.

Every frame carries its own index as a block pattern, so a wrong frame is
caught, not just a wrong count."""

from __future__ import annotations

import importlib.util
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("av")
cv2 = pytest.importorskip("cv2")
if shutil.which("ffmpeg") is None:
    pytest.skip("ffmpeg CLI not on PATH", allow_module_level=True)

_PKG = Path(__file__).resolve().parents[1] / "nodes" / "c2c_video"
_spec = importlib.util.spec_from_file_location("c2c_video_rw", _PKG / "__init__.py",
                                               submodule_search_locations=[str(_PKG)])
cv = importlib.util.module_from_spec(_spec)
sys.modules["c2c_video_rw"] = cv
_spec.loader.exec_module(cv)

N, W, H = 150, 320, 192
RATE = ["-framerate", "30000/1001"]
X264 = ["-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p"]
TAG709 = ["-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
          "-color_range", "tv", "-vf", "scale=out_color_matrix=bt709:out_range=tv"]
VFR = ["-enc_time_base:v", "1/90000", "-fps_mode", "passthrough"]

CASES = {
    "gop250_bframes.mp4": X264 + ["-g", "250", "-bf", "3"] + TAG709,
    "ms_timebase.mkv": X264 + ["-g", "60", "-bf", "2"] + TAG709,
    "opengop.ts": X264 + ["-g", "48", "-bf", "3", "-x264-params", "open-gop=1"] + TAG709,
    "gop250.ts": X264 + ["-g", "250", "-bf", "3"] + TAG709,
    "prores.mov": ["-c:v", "prores_ks", "-profile:v", "3", "-pix_fmt", "yuv422p10le"] + TAG709,
    "jitter.mp4": ["-vf", "settb=1/90000,setpts=N*3003+if(mod(N\\,7)\\,0\\,1170)"] + X264 + VFR,
    "drops.mp4": ["-vf", "settb=1/90000,setpts=(N+floor(N/5))*3003"] + X264 + VFR,
    "rare_drop.mkv": ["-vf", "settb=1/90000,setpts=(N+floor(N/100))*3003"] + X264 + ["-g", "30"] + VFR,
}


@pytest.fixture(scope="module")
def media(tmp_path_factory):
    root = tmp_path_factory.mktemp("rw")
    os.environ["C2C_VIDEO_INDEX_DIR"] = str(root / "index_cache")
    frames = root / "png"
    frames.mkdir()
    for i in range(N):
        img = cv.encode_index_pattern(i, W, H)
        img[64:, :, 0] = (i * 7) % 256
        cv2.imwrite(str(frames / f"f_{i:05d}.png"), img[..., ::-1])
    src = RATE + ["-i", str(frames / "f_%05d.png")]
    out = {}
    for name, args in CASES.items():
        path = root / name
        subprocess.run(["ffmpeg", "-v", "error", "-y"] + src + args + [str(path)], check=True)
        out[name] = str(path)
    for deg in ("90", "-90", "180"):
        rot = root / f"rot_{deg}.mp4"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-display_rotation", deg, "-i", out["gop250_bframes.mp4"],
                        "-c", "copy", str(rot)], check=True)
        ref = root / f"rot_{deg}.png"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(rot), "-frames:v", "1", str(ref)], check=True)
        out[f"rot_{deg}"] = (str(rot), str(ref))
    yield out
    cv.drain_container_pool()
    os.environ.pop("C2C_VIDEO_INDEX_DIR", None)


def _ids(batch):
    return [cv.decode_index(b) for b in batch]


@pytest.mark.parametrize("name", list(CASES))
def test_every_frame_is_the_right_frame(media, name):
    h = cv.probe_file(media[name], index="auto")
    assert len(h) == N
    rng = random.Random(7)
    for pos in ([rng.randrange(N) for _ in range(30)],
                list(reversed(range(0, N, 11))),
                list(range(3, N, 5)),
                [N - 1, 0, N - 1],
                [N - 1]):
        assert _ids(cv.read(h, pos, fmt="uint8")) == pos, name
    streamed = []
    for chunk in cv.iter_chunks(h.every(3), 16, fmt="uint8"):
        streamed += _ids(chunk)
    assert streamed == list(range(0, N, 3))


@pytest.mark.parametrize("name", ["jitter.mp4", "drops.mp4"])
def test_vfr_is_caught_at_probe(media, name):
    """A counted container (MP4) whose frames are not on the constant-rate
    grid, or leave slots empty, must get the exact index up front."""
    h = cv.probe_file(media[name], index="auto")
    assert cv.get_index_table(h.source.stat_key).is_exact


def test_counted_cfr_skips_the_demux(media):
    cv.reset_demux_call_count()
    h = cv.probe_file(media["gop250_bframes.mp4"], index="auto")
    assert not cv.get_index_table(h.source.stat_key).is_exact
    assert cv.demux_call_count() == 0


def test_prores_colour_tags_come_from_the_bitstream(media):
    h = cv.probe_file(media["prores.mov"])
    assert (h.colour.matrix, h.colour.range) == ("bt709", "tv")
    assert h.bit_depth == 10


@pytest.mark.parametrize("deg", ["90", "-90", "180"])
def test_rotation_matches_ffmpeg_autorotate(media, deg):
    clip, ref_png = media[f"rot_{deg}"]
    h = cv.probe_file(clip)
    ours = cv.read(h, [0], fmt="uint8")[0]
    ref = cv2.imread(ref_png)[..., ::-1]
    assert ours.shape == ref.shape
    assert (h.height, h.width) == ref.shape[:2]
    assert float(np.abs(ours.astype(np.int16) - ref).mean()) < 3.0


# ── 12- and 16-bit MOV (ProRes 4444 XQ, CineForm, PNG-in-MOV) ──
# A 16-bit ramp, one 12-bit level per column. Before the fix every high-bit
# source was scaled by 65535/65283 (the calibration white was built at 8 bits).

HBD_W = 4096
HBD_RAMP = (np.arange(HBD_W, dtype=np.uint32) * 16).astype(np.uint16)
HBD_CASES = {
    # name: (ffmpeg args, reported bits, min distinct levels, max mean error in 16-bit units)
    "prores4444xq.mov": (["-c:v", "prores_ks", "-profile:v", "5", "-pix_fmt", "yuva444p10le"], 12, 1500, 40),
    "cineform12.mov": (["-c:v", "cfhd", "-pix_fmt", "gbrp12le"], 12, 4000, 20),
    "png16.mov": (["-c:v", "png", "-pix_fmt", "rgb48be"], 16, 4096, 0.5),
}


@pytest.fixture(scope="module")
def hbd_media(tmp_path_factory):
    root = tmp_path_factory.mktemp("hbd")
    src = root / "src"
    src.mkdir()
    for i in range(3):
        img = np.empty((96, HBD_W, 3), np.uint16)
        img[:] = HBD_RAMP[None, :, None]
        img[:48] = cv.encode_index_pattern(i, HBD_W, 48).astype(np.uint16) * 257
        cv2.imwrite(str(src / f"f_{i:04d}.png"), img[..., ::-1])
    out = {}
    for name, (args, *_rest) in HBD_CASES.items():
        path = root / name
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", "24", "-i", str(src / "f_%04d.png")]
                       + args + [str(path)], check=True)
        out[name] = str(path)
    return out


@pytest.mark.parametrize("name", list(HBD_CASES))
def test_high_bit_depth_mov_keeps_precision(hbd_media, name):
    _args, bits, min_levels, max_err = HBD_CASES[name]
    h = cv.probe_file(hbd_media[name])
    assert h.bit_depth == bits
    b = cv.read(h, range(len(h)), fmt="uint16")
    assert _ids(b) == list(range(len(h)))
    row = b[0, 70, :, 1].astype(np.float64)
    assert len(np.unique(row)) >= min_levels
    slope = np.polyfit(HBD_RAMP.astype(np.float64), row, 1)[0]
    assert abs(slope - 1.0) < 1e-3, slope
    assert float(np.abs(b[0, 60:, :, 1].astype(np.int64) - HBD_RAMP[None, :].astype(np.int64)).mean()) <= max_err
