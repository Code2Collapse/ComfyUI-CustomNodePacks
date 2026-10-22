"""8K through Load Video (C2C) and Save Video (C2C), owner A16 Phase 2 ("4K/8K"). OPT-IN: C2C_TEST_8K=1 (an 8K frame
is 0.4 GB as float32; the tests compare frame by frame to stay near 2 GB). UHD-8K 7680x4320 and DCI 8K 8192x4320.

* decode: the fast ffmpeg path (rgb48) gives the exact reader's pixels, frame for frame;
* the node loads an 8K selection, and refuses one that does not fit in RAM before decoding anything;
* encode: H.265 (CPU), ProRes 422 HQ and FFV1 write 8K files that decode back at the full size and frame count
  (H.264 is checked too: libx264 accepts 8K up to level 6.2, the NVENC encoders do not - not tested here, CPU box).
"""
from __future__ import annotations

import os
import subprocess

import numpy as np
import pytest
import torch

from tests.test_c2c_video_4k import _ffmpeg, fd  # noqa: F401 - fixture used by name
from tests.test_c2c_video_nodes import (  # noqa: F401 - fixtures used by name
    _clean_video_state, _private_index_cache, _run_path, cv, nl,
)
from tests.test_save_video_4k import _encode, av, fp_dirs  # noqa: F401 - fixture used by name

pytestmark = pytest.mark.skipif(os.environ.get("C2C_TEST_8K") != "1" or _ffmpeg() is None,
                                reason="opt-in: C2C_TEST_8K=1 and ffmpeg (8K frames need about 2 GB of RAM)")

SIZES = [(7680, 4320, "uhd8k"), (8192, 4320, "dci8k")]


@pytest.fixture(params=SIZES, ids=[s[2] for s in SIZES])
def clip8k(request, tmp_path):
    w, h, name = request.param
    path = str(tmp_path / f"{name}.mp4")
    subprocess.run([_ffmpeg(), "-v", "error", "-f", "lavfi", "-i", f"testsrc2=size={w}x{h}:rate=24:duration=0.125",
                    "-c:v", "libx265", "-pix_fmt", "yuv420p", "-x265-params", "log-level=error", path], check=True)
    return path, w, h


def test_8k_fast_rgb48_is_the_exact_readers_pixels(fd, cv, clip8k):
    from c2c_video.reader import iter_chunks

    path, w, h = clip8k
    raw = cv.probe_file(path)
    assert (raw.width, raw.height) == (w, h) and raw.frame_count == 3
    n = 0
    for want, got in zip(iter_chunks(raw, 1, fmt="float32"), fd.iter_chunks_fast(raw, 1, "rgb48", device="cpu")):
        assert want.shape == got.shape == (1, h, w, 3)
        assert np.array_equal(want, got)
        n += 1
    assert n == 3


def test_8k_node_loads_and_refuses_what_does_not_fit(nl, clip8k, monkeypatch):
    path, w, h = clip8k
    out = _run_path(nl, path, decoder="exact", frame_load_cap=2)
    assert tuple(out[0].shape) == (2, h, w, 3)
    del out
    import psutil

    class _VM:
        available = 512 * 1024 ** 2        # 0.5 GB free: not even one 8K frame fits in the budget
    monkeypatch.setattr(psutil, "virtual_memory", lambda: _VM())
    with pytest.raises(RuntimeError, match="RAM is free"):
        _run_path(nl, path, decoder="exact", frame_load_cap=3)


@pytest.mark.parametrize("label", ["MP4 H.265", "MOV ProRes 422 HQ", "MKV FFV1 (lossless)", "MP4 H.264"])
def test_8k_saves_decode_back_at_full_size(fp_dirs, label):
    w, h = 7680, 4320
    yy = torch.linspace(0, 1, h)[:, None].expand(h, w)
    xx = torch.linspace(0, 1, w)[None, :].expand(h, w)
    frames = torch.stack([torch.stack([xx, yy, (xx + yy + k / 2) % 1], dim=-1) for k in range(2)])
    path = _encode(label, frames, prefix=f"t8k/{label.split()[1].lower()}")
    del frames
    with av.open(path) as c:
        s = c.streams.video[0]
        got = [(f.width, f.height) for f in c.decode(s)]
    assert got == [(w, h)] * 2
