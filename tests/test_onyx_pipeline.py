"""CPU-only tests for the ONYX pipeline module and MaskOpsMEC wiring."""
from __future__ import annotations

import inspect
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

PACK = Path(__file__).resolve().parents[1]
if str(PACK) not in sys.path:
    sys.path.insert(0, str(PACK))

for mod_name in ("folder_paths", "comfy", "comfy.utils", "server"):
    if mod_name not in sys.modules:
        stub = types.ModuleType(mod_name)
        stub.__path__ = []
        sys.modules[mod_name] = stub

from nodes.mask_matting import _onyx_pipeline as op
from nodes.mask_matting.node import MaskOpsMEC


class FakeVideoModel:
    def __init__(self):
        self.calls: list[dict] = []
        self._seeded: dict[tuple[int, int], dict] = {}

    def init_state(self, resource_path, offload_video_to_cpu=False, **kw):
        n = len(resource_path)
        return {"num_frames": n, "frames": n, "seeded": self._seeded}

    def reset_state(self, inference_state):
        inference_state.clear()

    def add_prompt(
        self,
        inference_state,
        frame_idx,
        points=None,
        point_labels=None,
        obj_id=None,
        rel_coordinates=True,
        **kw,
    ):
        assert kw.get("text_str") is None
        self.calls.append({
            "frame_idx": int(frame_idx),
            "points": points,
            "point_labels": point_labels,
            "obj_id": int(obj_id),
            "rel_coordinates": rel_coordinates,
        })
        self._seeded[(int(obj_id), int(frame_idx))] = {
            "points": points,
            "labels": point_labels,
        }
        return frame_idx, {}

    def propagate_in_video(
        self,
        inference_state,
        start_frame_idx=None,
        max_frame_num_to_track=None,
        reverse=False,
    ):
        B = inference_state["num_frames"]
        start = int(start_frame_idx or 0)
        cap = int(max_frame_num_to_track or B)
        if reverse:
            frames = list(range(start, max(start - cap, -1), -1))
        else:
            frames = list(range(start, min(B, start + cap)))
        for fi in frames:
            obj_ids = sorted({oid for (oid, _) in self._seeded.keys()})
            if not obj_ids:
                continue
            masks_t = []
            for _oid in obj_ids:
                m = torch.zeros(8, 8, dtype=torch.bool)
                m[2:6, 2:6] = True
                masks_t.append(m)
            yield fi, {
                "out_obj_ids": torch.tensor(obj_ids, dtype=torch.int64),
                "out_binary_masks": torch.stack(masks_t),
                "out_probs": torch.tensor([0.9] * len(obj_ids), dtype=torch.float32),
            }


class FakeMatter:
    KEY = "fake"

    def __init__(self):
        self.matte_calls = 0

    def load(self):
        return None

    def matte(self, image_bhwc, coarse_mask, *, trimap=None, **kw):
        self.matte_calls += 1
        tri = trimap.float().clamp(0, 1)
        alpha = tri.clone()
        alpha = torch.where(tri >= 0.99, torch.ones_like(alpha), alpha)
        alpha = torch.where(tri <= 0.01, torch.zeros_like(alpha), alpha)
        alpha = torch.where((tri > 0.01) & (tri < 0.99), torch.full_like(alpha, 0.75), alpha)
        return {"alpha": alpha}


def _legacy(**kw) -> op.LegacyPrompts:
    base = dict(
        frame_annotation=0,
        tracking_direction="forward",
        start_frame=0,
        max_frames_to_track=0,
        positive_coords='[{"x": 10, "y": 10, "label": 1}]',
        negative_coords="",
        pos_bbox=None,
        neg_bbox=None,
        normal_bbox=None,
        text_prompt="",
    )
    base.update(kw)
    return op.LegacyPrompts(**base)


def _img(b=3, h=32, w=48):
    return torch.rand(b, h, w, 3)


@pytest.fixture
def fake_video():
    return FakeVideoModel()


@pytest.fixture
def fake_matter():
    return FakeMatter()


def test_parse_scene_valid():
    raw = json.dumps({
        "version": 1,
        "objects": [{
            "id": 1, "label": "a", "enabled": True, "opacity": 1.0,
            "keyframes": [{"frame": 0, "points": {"positive": [[1, 2]]}}],
        }],
        "tracking": {"direction": "forward", "start_frame": 0, "max_frames": 0},
    })
    scene = op.parse_scene_prompts(raw, B=5, H=32, W=48)
    assert scene.objects[0].id == 1
    assert scene.objects[0].keyframes[0].points_pos == [(1.0, 2.0)]


def test_parse_scene_errors():
    with pytest.raises(ValueError, match="not valid JSON"):
        op.parse_scene_prompts("{bad", B=1, H=8, W=8)
    with pytest.raises(ValueError, match="version must be 1"):
        op.parse_scene_prompts('{"version":2,"objects":[]}', B=1, H=8, W=8)
    objs = [{"id": i + 1, "enabled": True, "opacity": 1,
             "keyframes": [{"frame": 0, "points": {"positive": [[1, 1]]}}]}
            for i in range(17)]
    with pytest.raises(ValueError, match="Too many objects"):
        op.parse_scene_prompts(json.dumps({"version": 1, "objects": objs}), B=1, H=8, W=8)


def test_legacy_mapping():
    scene = op.legacy_to_scene(
        B=4, H=16, W=16,
        legacy=_legacy(tracking_direction="bidirectional", positive_coords='[{"x": 4, "y": 5, "label": 1}]'),
        external_mask=None,
    )
    assert scene.objects[0].id == 1
    assert scene.tracking.direction == "both"
    assert scene.objects[0].keyframes[0].points_pos == [(4.0, 5.0)]


def test_seed_rel_coords(fake_video):
    state = fake_video.init_state([None, None])
    kf = op.KeyframeSpec(frame=0, points_pos=[(24.0, 16.0)])
    pts, lbls = op.keyframe_to_tracker_prompt(kf, W=48, H=32)
    fake_video.add_prompt(state, 0, points=pts, point_labels=lbls, obj_id=1, rel_coordinates=True)
    p = fake_video.calls[0]["points"][0]
    assert abs(p[0] - 24 / 48) < 1e-4
    assert abs(p[1] - 16 / 32) < 1e-4


def test_seed_box_labels_order():
    kf = op.KeyframeSpec(frame=0, box=(0, 0, 20, 20), points_pos=[(5.0, 5.0)])
    _, lbls = op.keyframe_to_tracker_prompt(kf, W=40, H=40)
    assert lbls[:2] == [2, 3]


def test_seed_one_call_per_object_frame(fake_video):
    state = fake_video.init_state([None, None, None])
    scene = op.SceneSpec(
        version=1,
        objects=[
            op.ObjectSpec(id=1, keyframes=[op.KeyframeSpec(frame=0, points_pos=[(1, 1)])]),
            op.ObjectSpec(id=2, keyframes=[op.KeyframeSpec(frame=1, points_pos=[(2, 2)])]),
        ],
        tracking=op.TrackingSpec(),
    )
    op.seed_all_objects(
        fake_video, state, scene,
        img_bhwc=_img(3), seg_one_fn=lambda *a, **k: (torch.zeros(8, 8).numpy(), 0.0),
        external_mask=None,
    )
    assert len(fake_video.calls) == 2


def test_seed_16_object_cap():
    objs = [{"id": i, "enabled": True, "opacity": 1,
             "keyframes": [{"frame": 0, "points": {"positive": [[1, 1]]}}]}
            for i in range(1, 18)]
    with pytest.raises(ValueError, match="Too many objects"):
        op.parse_scene_prompts(json.dumps({"version": 1, "objects": objs}), B=2, H=8, W=8)


def test_text_seed_no_detector_text(fake_video):
    state = fake_video.init_state([None])

    def seg_one(frame, pos, neg, bbox, neg_bbox, text):
        assert text == "cat"
        assert pos == []
        m = np.zeros(frame.shape[:2], dtype=np.float32)
        m[4:12, 4:12] = 1.0
        return m, 0.9

    scene = op.SceneSpec(
        version=1,
        objects=[op.ObjectSpec(id=1, keyframes=[op.KeyframeSpec(frame=0, text="cat")])],
        tracking=op.TrackingSpec(),
    )
    op.seed_all_objects(
        fake_video, state, scene, img_bhwc=_img(1), seg_one_fn=seg_one, external_mask=None,
    )
    assert len(fake_video.calls) == 1
    assert fake_video.calls[0]["point_labels"][:2] == [2, 3]


def test_text_seed_passes_points_to_seg_one(fake_video):
    state = fake_video.init_state([None])
    received = {}

    def seg_one(frame, pos, neg, bbox, neg_bbox, text):
        received["pos"] = pos
        received["text"] = text
        m = np.zeros(frame.shape[:2], dtype=np.float32)
        m[4:12, 4:12] = 1.0
        return m, 0.9

    scene = op.SceneSpec(
        version=1,
        objects=[op.ObjectSpec(
            id=2,
            label="boom microphone",
            keyframes=[op.KeyframeSpec(
                frame=12,
                text="boom microphone",
                points_pos=[(20.0, 24.0)],
            )],
        )],
        tracking=op.TrackingSpec(),
    )
    img = _img(20, 32, 48)
    op.seed_all_objects(
        fake_video, state, scene, img_bhwc=img, seg_one_fn=seg_one, external_mask=None,
    )
    assert received["text"] == "boom microphone"
    assert received["pos"] == [(20.0, 24.0)]
    assert fake_video.calls[0]["frame_idx"] == 12


def test_text_seed_empty_raises_plain_english(fake_video):
    state = fake_video.init_state([None] * 15)

    def seg_one_empty(*a, **k):
        return np.zeros((32, 48), dtype=np.float32), 0.0

    scene = op.SceneSpec(
        version=1,
        objects=[op.ObjectSpec(
            id=2,
            label="boom microphone",
            keyframes=[op.KeyframeSpec(frame=12, text="boom microphone")],
        )],
        tracking=op.TrackingSpec(),
    )
    with pytest.raises(ValueError, match="Object 2: 'boom microphone' found nothing on frame 12"):
        op.seed_all_objects(
            fake_video, state, scene, img_bhwc=_img(15), seg_one_fn=seg_one_empty,
            external_mask=None,
        )


def test_propagate_tensor_outputs(fake_video):
    state = fake_video.init_state([None, None, None, None, None])
    fake_video._seeded[(1, 2)] = {}
    scene = op.SceneSpec(
        version=1,
        objects=[op.ObjectSpec(id=1, keyframes=[op.KeyframeSpec(frame=2, points_pos=[(1, 1)])])],
        tracking=op.TrackingSpec(direction="forward", start_frame=2, max_frames=0),
    )
    coarse, scores = op.propagate_objects(fake_video, state, scene, B=5, H=8, W=8)
    assert coarse[0, 2].sum() > 0
    assert scores[1] > 0.0


def test_all_foreground_tile_skips_matter(fake_matter):
    img = _img(1, 64, 64)
    tri = torch.ones(64, 64)
    coarse = torch.ones(64, 64)
    fake_matter.matte_calls = 0
    op.refine_frame_tiled(
        fake_matter, img[0], tri, coarse, matte_tile=32, tile_overlap=8, margin=0,
    )
    assert fake_matter.matte_calls == 0


def test_unknown_tile_calls_matter(fake_matter):
    img = _img(1, 64, 64)
    tri = torch.zeros(64, 64)
    tri[20:44, 20:44] = 0.5
    coarse = torch.zeros(64, 64)
    fake_matter.matte_calls = 0
    op.refine_frame_tiled(
        fake_matter, img[0], tri, coarse, matte_tile=32, tile_overlap=8, margin=0,
    )
    assert fake_matter.matte_calls > 0


def test_mask_input_seed(fake_video):
    state = fake_video.init_state([None, None])
    ext = torch.zeros(2, 8, 8)
    ext[1, 2:6, 2:6] = 1.0
    scene = op.SceneSpec(
        version=1,
        objects=[op.ObjectSpec(
            id=1, keyframes=[op.KeyframeSpec(frame=1, mask_source="input")],
        )],
        tracking=op.TrackingSpec(),
    )
    op.seed_all_objects(
        fake_video, state, scene, img_bhwc=_img(2),
        seg_one_fn=lambda *a, **k: (torch.zeros(8, 8).numpy(), 0.0),
        external_mask=ext,
    )
    assert fake_video.calls[0]["frame_idx"] == 1


def test_propagate_directions(fake_video):
    state = fake_video.init_state([None, None, None, None, None])
    fake_video._seeded[(1, 2)] = {}
    scene = op.SceneSpec(
        version=1,
        objects=[op.ObjectSpec(id=1, keyframes=[op.KeyframeSpec(frame=2, points_pos=[(1, 1)])])],
        tracking=op.TrackingSpec(direction="both", start_frame=0, max_frames=0),
    )
    coarse, _ = op.propagate_objects(fake_video, state, scene, B=5, H=8, W=8)
    assert coarse.shape == (1, 5, 8, 8)
    assert coarse[0, 2].sum() > 0
    assert coarse[0, 4].sum() > 0
    assert coarse[0, 0].sum() > 0


def test_trimap_band_scales():
    coarse = torch.zeros(1, 1, 1080, 1920)
    coarse[0, 0, 500:580, 900:980] = 1.0
    tri_small = op.trimap_resaware(coarse, 1.0)
    coarse4k = torch.zeros(1, 1, 2160, 3840)
    coarse4k[0, 0, 1000:1160, 1800:1960] = 1.0
    tri_large = op.trimap_resaware(coarse4k, 1.0)
    unk_small = int((tri_small == 0.5).sum())
    unk_large = int((tri_large == 0.5).sum())
    assert unk_large > unk_small


def test_trimap_band_min_3px():
    assert op.band_px(8, 8, 1.0) >= 3


def test_tiled_equals_untiled_fake_matter(fake_matter):
    img = _img(1, 64, 64)
    tri = torch.full((1, 1, 64, 64), 0.5)
    tri[0, 0, 20:44, 20:44] = 1.0
    tri[0, 0, 10:54, 10:54] = 0.5
    coarse = (tri == 1.0).float()
    tiled = op.refine_object_tiled(fake_matter, img, tri, coarse, matte_tile=32)
    flat = fake_matter.matte(img, coarse, trimap=tri[0])["alpha"].unsqueeze(0).unsqueeze(0)
    diff = (tiled - flat).abs().max()
    assert diff.item() < 1e-3


def test_tiled_only_unknown_bbox(fake_matter):
    img = _img(1, 64, 64)
    tri = torch.zeros(1, 1, 64, 64)
    tri[0, 0, 30:34, 30:34] = 1.0
    tri[0, 0, 28:36, 28:36] = 0.5
    coarse = (tri >= 0.5).float()
    out = op.refine_object_tiled(fake_matter, img, tri, coarse, matte_tile=24)
    assert out[0, 0, 0, 0].item() == 0.0
    assert out[0, 0, 32, 32].item() > 0.0


def test_combine_opacity():
    alpha = torch.zeros(2, 1, 4, 4)
    alpha[0, 0].fill_(1.0)
    alpha[1, 0].fill_(1.0)
    objs = [
        op.ObjectSpec(id=1, enabled=True, opacity=1.0),
        op.ObjectSpec(id=2, enabled=True, opacity=0.5),
    ]
    comb = op.combine_alphas(alpha, objs)
    assert comb.max().item() == 1.0


def test_object_alphas_shape():
    alpha = torch.rand(2, 3, 8, 8)
    objs = [op.ObjectSpec(id=1), op.ObjectSpec(id=2)]
    packed = op.pack_object_alphas(alpha, objs)
    assert packed.shape == (6, 8, 8)


def test_maskops_return_types():
    assert len(MaskOpsMEC.RETURN_TYPES) == 18
    assert MaskOpsMEC.RETURN_NAMES[-1] == "object_alphas"


def test_execute_default_pipeline_legacy():
    sig = inspect.signature(MaskOpsMEC.execute)
    assert sig.parameters["pipeline"].default == "cascade (legacy)"


def test_run_onyx_legacy_integration(fake_video, fake_matter):
    img = _img(2, 32, 48)
    res = op.run_onyx_pipeline(
        img,
        scene_raw="",
        legacy=_legacy(positive_coords='[{"x": 12, "y": 12, "label": 1}]'),
        matter_key="none",
        matter_model="(auto)",
        matter_cls=None,
        sam31_model_name="(auto)",
        device="cpu",
        precision="fp32",
        attention="auto",
        offload="none",
        auto_download=False,
        band_scale=1.0,
        matte_tile=32,
        temporal_stabilise=False,
        post_refine="none",
        external_mask=None,
        external_trimap=None,
        video_model=fake_video,
        seg_one_fn=lambda f, *a, **k: (np.zeros(f.shape[:2], np.float32), 0.0),
        matter_instance=fake_matter,
    )
    assert res.mask_coarse.shape == (2, 32, 48)
    assert res.alpha_combined.shape == (2, 32, 48)
    assert res.object_alphas.shape[0] == 2
