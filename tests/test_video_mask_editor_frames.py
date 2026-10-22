"""L9.35 (owner 2026-10-23: 'the mask is sticking throughout rather than frame-by-frame'): a painted frame applies to
itself; interpolation / hold between and outside painted frames are explicit choices; input_mask is a per-frame base;
painted keyframes survive a server restart (L9.16)."""
import uuid

import numpy as np
import pytest
import torch

from nodes import video_mask_editor as vme


@pytest.fixture(autouse=True)
def _private_store(tmp_path, monkeypatch):
    monkeypatch.setenv("C2C_VME_DIR", str(tmp_path / "vme"))


@pytest.fixture
def sid():
    s = f"test-{uuid.uuid4()}"
    yield s
    vme._SESSIONS.pop(s, None)


def disk(v=255, hw=(32, 48), box=(8, 16, 12, 24)):
    m = np.zeros(hw, np.uint8)
    y0, y1, x0, x1 = box
    m[y0:y1, x0:x1] = v
    return m


def run(sid, n=8, tween="distance_transform", outside=None, input_mask=None, hw=(32, 48)):
    img = torch.zeros(n, hw[0], hw[1], 3)
    kw = {} if outside is None else {"outside_keys": outside}
    mask, info = vme.VideoMaskEditorMEC().execute(img, sid, tween, 0.0, 0.0, input_mask, **kw)
    return mask, info


def painted(mask):
    return [int(i) for i in range(mask.shape[0]) if float(mask[i].max()) > 0.5]


def test_one_painted_frame_masks_only_that_frame(sid):
    vme._get_session(sid, create=True)["keyframes"][2] = disk()
    mask, info = run(sid)
    assert painted(mask) == [2]                  # before 2026-10-23: all 8 frames
    assert '"painted_frames":[2]' in info and "outside=empty" in info


def test_hold_outside_is_the_old_behaviour_on_request(sid):
    vme._get_session(sid, create=True)["keyframes"][2] = disk()
    mask, _ = run(sid, outside="hold")
    assert painted(mask) == list(range(8))


@pytest.mark.parametrize("tween,between", [("none", []), ("hold", [3, 4, 5]), ("distance_transform", [3, 4, 5]),
                                           ("linear", [3, 4, 5])])
def test_between_painted_frames(sid, tween, between):
    s = vme._get_session(sid, create=True)
    s["keyframes"][2] = disk(box=(4, 12, 4, 12))
    s["keyframes"][6] = disk(box=(20, 28, 30, 40))
    mask, _ = run(sid, tween=tween)
    got = painted(mask) if tween != "linear" else [i for i in range(8) if float(mask[i].max()) > 0.2]
    assert got == sorted([2, 6] + between)
    if tween == "hold":
        assert torch.equal(mask[4], mask[2])
    if tween == "distance_transform":                    # the shape travels from the first box to the second
        ys, xs = torch.nonzero(mask[4] > 0.5, as_tuple=True)
        assert 10 < float(xs.float().mean()) < 30


def test_input_mask_is_a_per_frame_base(sid):
    base = torch.zeros(8, 32, 48)
    for i in range(8):
        base[i, i * 3:i * 3 + 2, :] = 1.0                 # a different stripe on every frame
    vme._get_session(sid, create=True)["keyframes"][5] = disk()
    mask, info = run(sid, input_mask=base)
    for i in range(8):
        if i == 5:
            assert torch.equal(mask[5], torch.from_numpy(disk()).float() / 255.0)
        else:
            assert torch.equal(mask[i], base[i]), i      # before: every frame got base[0]
    assert "input=7" in info and "painted=1" in info


def test_short_input_mask_repeats_its_last_frame(sid):
    base = torch.zeros(2, 32, 48)
    base[1, :4] = 1.0
    vme._get_session(sid, create=True)
    mask, _ = run(sid, input_mask=base)
    assert float(mask[0].max()) == 0.0 and all(torch.equal(mask[i], base[1]) for i in range(1, 8))


def test_keyframes_survive_a_restart(sid):
    vme._disk_put(sid, 3, disk())
    vme._disk_put(sid, 7, disk(128))
    vme._SESSIONS.pop(sid, None)                          # a server restart empties the RAM store
    s = vme._get_session(sid)
    assert s is not None and sorted(s["keyframes"]) == [3, 7]
    assert int(s["keyframes"][7].max()) == 128
    vme._disk_delete(sid, 3)
    vme._SESSIONS.pop(sid, None)
    assert sorted(vme._get_session(sid)["keyframes"]) == [7]
    vme._disk_delete(sid)
    vme._SESSIONS.pop(sid, None)
    assert vme._get_session(sid) is None


def test_bad_session_ids_never_touch_the_disk(tmp_path):
    for bad in ("", "../../x", "a/b", "short"):
        assert vme._disk_dir(bad) is None
    vme._disk_put("../escape-attempt", 0, disk())
    assert not (tmp_path / "escape-attempt").exists()


def test_effective_frame_reports_how_it_was_made(sid):
    kfs = {2: disk(box=(4, 12, 4, 12)), 6: disk(box=(20, 28, 30, 40))}
    srt = sorted(kfs)
    none = vme._base_getter(None, (32, 48))
    assert vme._frame_mask(2, (32, 48), kfs, srt, "distance_transform", "empty", none)[1] == "painted"
    assert vme._frame_mask(4, (32, 48), kfs, srt, "distance_transform", "empty", none)[1] == "interpolated 2-6"
    assert vme._frame_mask(4, (32, 48), kfs, srt, "hold", "empty", none)[1] == "held from 2"
    assert vme._frame_mask(4, (32, 48), kfs, srt, "none", "empty", none) == (None, "empty")
    assert vme._frame_mask(0, (32, 48), kfs, srt, "none", "empty", none) == (None, "empty")
    assert vme._frame_mask(7, (32, 48), kfs, srt, "none", "hold", none)[1] == "held from 6"


def test_old_api_prompts_still_validate():
    spec = vme.VideoMaskEditorMEC.INPUT_TYPES()
    assert "outside_keys" in spec["optional"] and "outside_keys" not in spec["required"]
    assert spec["required"]["tween_mode"][0][:3] == ["distance_transform", "linear", "hold"]
