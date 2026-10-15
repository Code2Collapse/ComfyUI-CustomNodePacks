"""L7.114 (owner A14): the viewer's selection (`picks`) on Mask + Matting.

* empty = the node runs exactly as before (the new code is never reached);
* set = it becomes the chosen backend's prompts (recording stand-ins show exactly what each backend receives);
* every bad selection is refused with a sentence that says what to do;
* the preview chunk (`range`) cuts the plate and moves the prompt frame;
* C2C_VIEWER=0 only changes the front end: the same selection gives the same result.
"""
from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from nodes.mask_matting import _picks

H, W, CX, CY, R = 72, 96, 48, 36, 16


def plate(b=3, seed=3):
    rng = np.random.default_rng(seed)
    frames = []
    for i in range(b):
        g = (0.45 + 0.06 * rng.standard_normal((H, W, 1))).clip(0, 1).repeat(3, -1)
        yy, xx = np.mgrid[:H, :W]
        g[(xx - CX - i) ** 2 + (yy - CY) ** 2 <= R ** 2] = [0.85, 0.12, 0.10]
        frames.append(g)
    return torch.from_numpy(np.stack(frames)).float()


def sel(**over):
    base = {"v": 1, "frame": 0, "text": "disk",
            "subject": {"x": CX / W, "y": CY / H, "box": [0.30, 0.25, 0.70, 0.75], "label": "disk"},
            "excluded": [{"x": 0.05, "y": 0.05, "label": "corner"}],
            "add": [{"x": 0.75, "y": 0.5}], "cut": [{"x": 0.5, "y": 0.3}],
            "plate": {"frames": 3, "w": W, "h": H}}
    base.update(over)
    return json.dumps(base)


@pytest.fixture(scope="module")
def maskops():
    node = pytest.importorskip("nodes.mask_matting.node", reason="MaskOps unavailable")
    return node.MaskOpsMEC


def defaults(cls):
    spec = cls.INPUT_TYPES()
    out = {}
    for name, s in {**spec["required"], **spec.get("optional", {})}.items():
        opts = s[1] if len(s) > 1 and isinstance(s[1], dict) else {}
        if "default" in opts:
            out[name] = opts["default"]
        elif isinstance(s[0], (list, tuple)) and s[0]:
            out[name] = s[0][0]
    return out


def run(cls, **kw):
    args = defaults(cls)
    args.update(kw)
    return cls().execute(**args)


@pytest.fixture(autouse=True)
def _fresh_cache(monkeypatch):
    from nodes.mask_matting import _stage_cache
    monkeypatch.setattr(_stage_cache, "_CACHE", None)
    monkeypatch.delenv("C2C_MATTE_CACHE_MB", raising=False)


# ── the schema ───────────────────────────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("text", ["", "   ", None, "{}", json.dumps({"v": 1, "frame": 4, "text": "person"})])
def test_nothing_clicked_is_no_selection(text):
    assert _picks.parse(text) is None


@pytest.mark.parametrize("text, words", [
    ("{not json", "not valid JSON"),
    ("[1, 2]", "not a JSON object"),
    (json.dumps({"v": 2, "subject": {"x": 0.5, "y": 0.5}}), "newer version"),
    (json.dumps({"v": 1, "excluded": [{"x": 0.1, "y": 0.1}]}), "no subject"),
    (json.dumps({"v": 1, "subject": {"x": 1.4, "y": 0.5}}), "outside the picture"),
    (json.dumps({"v": 1, "subject": {"x": "a", "y": 0.5}}), "not a number"),
    (json.dumps({"v": 1, "frame": -2, "subject": {"x": 0.5, "y": 0.5}}), "frame -2"),
])
def test_a_bad_selection_is_refused_in_words(text, words):
    with pytest.raises(ValueError, match=words) as e:
        _picks.parse(text)
    assert "pick again" in str(e.value) or "Update" in str(e.value)


def test_the_selection_in_pixels():
    p = _picks.parse(sel())
    pr = _picks.prompts(p, W, H)
    # subject first, then the added part; negatives = excluded objects, then cut parts
    assert pr.positive == [(CX, CY), (0.75 * W, 0.5 * H)]
    assert pr.negative == [(0.05 * W, 0.05 * H), (0.5 * W, 0.3 * H)]
    # the box grows to take in the added part (x 0.75 > 0.70) plus a 4 % margin, in pixels
    x0, y0, x1, y1 = pr.box
    assert x1 > 0.75 * W and x0 < 0.30 * W and y0 < 0.25 * H and y1 > 0.75 * H
    assert json.loads(pr.points_json()) == [{"x": CX, "y": CY, "label": 1}, {"x": 72.0, "y": 36.0, "label": 1},
                                            {"x": 4.8, "y": 3.6, "label": 0}, {"x": 48.0, "y": 21.6, "label": 0}]


def test_a_point_on_the_far_edge_stays_inside_the_picture():
    p = _picks.parse(json.dumps({"v": 1, "subject": {"x": 1.0, "y": 1.0}}))
    assert _picks.prompts(p, W, H).positive == [(W - 1, H - 1)]


def test_the_plate_checks():
    p = _picks.parse(sel(frame=2))
    with pytest.raises(ValueError, match="picked on frame 2, but the plate has 2 frames"):
        _picks.check_plate(p, 2, W, H)
    with pytest.raises(ValueError, match="changed shape"):
        _picks.check_plate(p, 3, H, W)                     # portrait now
    notes = _picks.check_plate(p, 3, W * 2, H * 2)          # same shape, bigger: points are relative
    assert any("still land on the same spot" in n for n in notes)


def test_a_preview_range_must_hold_the_pick_frame():
    p = _picks.parse(sel(frame=1, range={"start": 1, "count": 2}))
    assert _picks.frame_range(p, 3) == (1, 3)
    assert _picks.prompts(p, W, H, frame_offset=1).frame == 0
    with pytest.raises(ValueError, match="Preview from the pick frame"):
        _picks.frame_range(_picks.parse(sel(frame=0, range={"start": 1, "count": 2})), 3)


# ── empty = unchanged ────────────────────────────────────────────────────────────────────────────────────────────

def test_picks_is_appended_last_and_empty_by_default(maskops):
    spec = maskops.INPUT_TYPES()
    opt = list(spec["optional"])
    assert opt[-1] == "picks"
    kind, opts = spec["optional"]["picks"]
    assert kind == "STRING" and opts["default"] == "" and opts["c2c_viewer"] is True


def test_empty_picks_never_reaches_the_viewer_code(maskops, monkeypatch):
    """The node with picks="" runs the path it always ran: the selection code and the stage cache are booby-trapped,
    and the outputs equal a call that does not pass picks at all."""
    from nodes.mask_matting import _stage_cache

    def boom(*_a, **_k):
        raise AssertionError("viewer code reached with an empty selection")

    image = plate(2)
    kw = dict(image=image, segmenter="classic", matter="guided_filter", pipeline="cascade (legacy)",
              positive_coords=json.dumps([[CX, CY]]))
    base = defaults(maskops)
    base.pop("picks")
    ref = maskops().execute(**{**base, **kw})
    for name in ("check_plate", "prompts", "frame_range"):
        monkeypatch.setattr(_picks, name, boom)
    monkeypatch.setattr(_stage_cache, "get_cache", boom)
    monkeypatch.setattr(_stage_cache, "fingerprint", boom)
    for empty in ("", "  ", "{}"):
        out = maskops().execute(**{**base, **kw, "picks": empty})
        for a, b in zip(ref, out):
            assert (torch.equal(a, b) if torch.is_tensor(a) else a == b)
        assert "viewer" not in json.loads(out[7])


# ── set = the backend's prompts ──────────────────────────────────────────────────────────────────────────────────

def _record_classic(monkeypatch):
    from nodes.mask_matting.segmenters import get_segmenter_cls
    cls = get_segmenter_cls("classic")
    calls = []
    orig = cls.segment

    def rec(self, image_bhwc, **kw):
        calls.append(kw)
        return orig(self, image_bhwc, **kw)

    monkeypatch.setattr(cls, "segment", rec)
    return calls


def test_cascade_prompts_come_from_the_selection(maskops, monkeypatch):
    calls = _record_classic(monkeypatch)
    out = run(maskops, image=plate(3), segmenter="classic", matter="guided_filter", pipeline="cascade (legacy)",
              positive_coords=json.dumps([[1, 1]]), text_prompt="", picks=sel(frame=1))
    kw = calls[0]
    assert kw["positive_points"] == [(CX, CY), (72.0, 36.0)]
    assert kw["negative_points"] == [(4.8, 3.6), (48.0, 21.6)]
    assert kw["bbox"] is not None and kw["frame_annotation"] == 1
    assert kw["tracking_direction"] == "bidirectional"            # picked mid-clip: frames before it are tracked
    info = json.loads(out[7])["viewer"]
    assert info["replaced_inputs"] == ["positive_coords"]
    assert info["prompts"] == {"positive": 2, "negative": 2, "box": True, "frame": 1}
    assert any("bidirectional" in n for n in info["notes"])


def test_onyx_gets_the_selection_as_its_legacy_prompts(maskops, monkeypatch):
    from nodes.mask_matting import _onyx_pipeline as op
    seen = {}
    orig = op.run_onyx_pipeline

    def rec(img, **kw):
        seen.update(kw)
        return orig(img, **kw)

    monkeypatch.setattr(op, "run_onyx_pipeline", rec)
    em = torch.zeros(1, H, W)
    run(maskops, image=plate(3), matter="guided_filter", pipeline="onyx", external_mask=em,
        scene_prompts='{"version": 1}', picks=sel(frame=0))
    leg = seen["legacy"]
    assert json.loads(leg.positive_coords) == [[CX, CY], [72.0, 36.0]]
    assert json.loads(leg.negative_coords) == [[4.8, 3.6], [48.0, 21.6]]
    assert leg.frame_annotation == 0 and leg.tracking_direction == "forward"
    assert seen["scene_raw"] == "" and seen["external_mask"] is None    # ONYX reads external_mask as a prompt
    assert seen["stage_cache"] is not None and seen["plate_fp"]


def test_sam_vitmatte_gets_points_json_and_the_box(maskops, monkeypatch):
    from nodes.mask_matting import _engines
    got = {}

    def fake(image, kw):
        got.update(kw)
        a = torch.zeros(image.shape[0], H, W)
        return _engines.EngineResult(mask=a, alpha=a, preview=None, score=0.5, bbox=[0, 0, 1, 1], edge_mask=None,
                                     foreground=None, info={})

    monkeypatch.setitem(_engines._RUNNERS, _engines.SAM_VITMATTE, fake)
    out = run(maskops, image=plate(3), backend=_engines.SAM_VITMATTE, points_json='[{"x":1,"y":1,"label":1}]',
              picks=sel(frame=2))
    pts = json.loads(got["points_json"])
    assert [p["label"] for p in pts] == [1, 1, 0, 0] and pts[0] == {"x": CX, "y": CY, "label": 1}
    assert len(json.loads(got["bbox_json"])) == 4 and got["pos_bbox"] is None
    info = json.loads(out[7])["viewer"]
    assert info["replaced_inputs"] == ["points_json"]


def test_sec_refuses_a_pick_after_the_first_frame(maskops):
    from nodes.mask_matting import _engines
    with pytest.raises(ValueError, match="first frame of the clip, but the subject was picked on frame 1"):
        run(maskops, image=plate(3), backend=_engines.SEC_MATANYONE, picks=sel(frame=1))


def test_remove_background_only_takes_the_frames(maskops, monkeypatch):
    from nodes.mask_matting import _engines
    got = {}

    def fake(image, kw):
        got["frames"] = int(image.shape[0])
        a = torch.ones(image.shape[0], H, W)
        return _engines.EngineResult(mask=a, alpha=a, preview=None, score=None, bbox=None, edge_mask=None,
                                     foreground=None, info={})

    monkeypatch.setitem(_engines._RUNNERS, _engines.REMOVE_BG, fake)
    out = run(maskops, image=plate(3), backend=_engines.REMOVE_BG, picks=sel(frame=1, range={"start": 1, "count": 2}))
    assert got["frames"] == 2 and out[1].shape[0] == 2
    assert any("no prompts" in n for n in json.loads(out[7])["viewer"]["notes"])


def test_a_preview_chunk_renders_only_its_frames(maskops, monkeypatch):
    calls = _record_classic(monkeypatch)
    out = run(maskops, image=plate(3), segmenter="classic", matter="guided_filter", pipeline="cascade (legacy)",
              picks=sel(frame=1, range={"start": 1, "count": 2}))
    assert out[1].shape[0] == 2 and out[0].shape[0] == 2
    assert calls[0]["frame_annotation"] == 0                      # frame 1 of the plate = frame 0 of the chunk
    assert json.loads(out[7])["viewer"]["frames"] == [1, 3]


def test_a_stale_selection_stops_before_any_work(maskops, monkeypatch):
    calls = _record_classic(monkeypatch)
    with pytest.raises(ValueError, match="picked on frame 5"):
        run(maskops, image=plate(3), segmenter="classic", matter="guided_filter", pipeline="cascade (legacy)",
            picks=sel(frame=5))
    assert calls == []


def test_the_kill_switch_changes_the_front_end_not_the_result(maskops, monkeypatch):
    kw = dict(image=plate(2), segmenter="classic", matter="guided_filter", pipeline="cascade (legacy)",
              picks=sel(frame=0, plate={"frames": 2, "w": W, "h": H}))
    on = run(maskops, **kw)
    monkeypatch.setenv("C2C_VIEWER", "0")
    assert maskops.INPUT_TYPES()["optional"]["picks"][1]["c2c_viewer"] is False
    off = run(maskops, **kw)
    for a, b in zip(on[:7], off[:7]):
        assert (torch.equal(a, b) if torch.is_tensor(a) else a == b)
