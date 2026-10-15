"""L2.53: crop in Load Video (C2C) / Load Video Path (C2C) - the WhatDreamsCost-style crop tool writes crop_x / crop_y /
crop_w / crop_h (fractions of the frame); the run decodes each frame at full size, cuts the rectangle out, then
resizes it (custom_width / custom_height). The crop must equal slicing the uncropped frames, exactly."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from tests.test_c2c_video_nodes import (  # noqa: F401 - fixtures used by name
    _clean_video_state, _load_c2c_video, _private_index_cache, _run_path, cfr_clip, cv, exr_seq, nl,
)


def test_crop_inputs_are_appended_after_the_old_ones(nl):
    v = nl.LoadVideoC2C.INPUT_TYPES()
    # the crop inputs come right after the old ones; L2.53's inputs are appended after them (test_c2c_video_4k.py)
    assert list(v["optional"])[:6] == ["vae", "format", "crop_x", "crop_y", "crop_w", "crop_h"]
    p = nl.LoadVideoPathC2C.INPUT_TYPES()
    assert list(p["optional"])[:7] == ["vae", "format", "sequence_fps", "crop_x", "crop_y", "crop_w", "crop_h"]
    for t in (v, p):
        defaults = [t["optional"][k][1]["default"] for k in ("crop_x", "crop_y", "crop_w", "crop_h")]
        assert defaults == [0.0, 0.0, 1.0, 1.0]


def test_crop_equals_a_slice_of_the_full_frames(nl, cfr_clip):
    full = _run_path(nl, cfr_clip, frame_load_cap=4)[0]
    crop, count, _a, info, handle, _m, _s = _run_path(nl, cfr_clip, frame_load_cap=4,
                                                  crop_x=0.25, crop_y=0.125, crop_w=0.5, crop_h=0.5)
    assert count == 4 and crop.shape == (4, 128, 128, 3)
    assert torch.equal(crop, full[:, 32:160, 64:192])
    assert handle.crop == (64, 32, 128, 128) and handle.crop_source == (256, 256)
    assert info["loaded_width"] == 128 and info["loaded_height"] == 128


def test_crop_then_resize(nl, cfr_clip):
    from c2c_video.reader import _resize_image

    full = _run_path(nl, cfr_clip, frame_load_cap=3)[0]
    out = _run_path(nl, cfr_clip, frame_load_cap=3, crop_x=0.25, crop_y=0.0, crop_w=0.5, crop_h=1.0,
                    custom_width=64)[0]
    assert out.shape == (3, 128, 64, 3)                     # 128 x 256 crop, width 64 keeps its aspect
    ref = np.stack([_resize_image(np.ascontiguousarray(f[:, 64:192].numpy()), 64, 128) for f in full])
    assert np.allclose(out.numpy(), ref, atol=1e-6)


def test_no_crop_is_the_same_handle(cv, cfr_clip):
    h = cv.probe_file(cfr_clip)
    assert h.cropped(0, 0, 1, 1) is h
    assert h.cropped(0.5, 0.5, 0, 0.2) is h                  # empty rectangle = no crop


def test_crop_of_a_crop_composes_and_scale_comes_last(cv, cfr_clip):
    h = cv.probe_file(cfr_clip)
    c = h.cropped(0.5, 0.0, 0.5, 1.0).cropped(0.5, 0.25, 0.5, 0.5)
    assert c.crop == (192, 64, 64, 128) and c.crop_source == (256, 256) and (c.width, c.height) == (64, 128)
    with pytest.raises(ValueError):
        h.scaled(128, 128).cropped(0.1, 0.1, 0.5, 0.5)
    assert h.cropped(0.1, 0.1, 0.5, 0.5).fingerprint() != h.fingerprint()


def test_crop_image_sequence(nl, exr_seq):
    full = _run_path(nl, exr_seq, frame_load_cap=2)[0]
    crop = _run_path(nl, exr_seq, frame_load_cap=2, crop_x=0.5, crop_y=0.25, crop_w=0.25, crop_h=0.5)[0]
    assert crop.shape == (2, 32, 64, 3)
    assert torch.equal(crop, full[:, 16:48, 128:192])
