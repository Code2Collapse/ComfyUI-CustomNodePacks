"""CPU tests for ViTMatte object-aware tiling (no weights)."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

PACK = Path(__file__).resolve().parents[1]
if str(PACK) not in sys.path:
    sys.path.insert(0, str(PACK))

vm = pytest.importorskip(
    "nodes.mask_matting.matters.vitmatte_backend",
    reason="mask_matting matters unavailable",
)
ViTMatteMatter = vm.ViTMatteMatter


class StubProcessor:
    """Mimic HF processor: pixel_values [1,4,h,w] from RGB + trimap."""

    def __call__(self, images, trimaps, return_tensors="pt"):
        img = np.asarray(images, dtype=np.float32)
        if img.ndim == 3:
            img = img[np.newaxis, ...]
        tri = np.asarray(trimaps, dtype=np.float32)
        if tri.ndim == 2:
            tri = tri[np.newaxis, ...]
        rgb = img[0] / 255.0
        t = tri[0] / 255.0
        if t.ndim == 2:
            t = t[np.newaxis, ...]
        pv = np.concatenate([rgb.transpose(2, 0, 1), t], axis=0)
        return {"pixel_values": torch.from_numpy(pv).unsqueeze(0).float()}


class StubModel:
    """Pointwise luminance alpha; counts forwards and batch sizes."""

    def __init__(self, oom_on_batch_gt: int = 0):
        self.forward_calls = 0
        self.batch_sizes: list[int] = []
        self.oom_on_batch_gt = oom_on_batch_gt

    def __call__(self, **inputs):
        pv = inputs["pixel_values"]
        b = pv.shape[0]
        self.forward_calls += 1
        self.batch_sizes.append(b)
        if self.oom_on_batch_gt and b > self.oom_on_batch_gt:
            raise torch.cuda.OutOfMemoryError("stub OOM")
        rgb = pv[:, :3, :, :]
        lum = 0.299 * rgb[:, 0] + 0.587 * rgb[:, 1] + 0.114 * rgb[:, 2]
        return type("Out", (), {"alphas": lum.unsqueeze(1)})()


@pytest.fixture
def matter():
    m = ViTMatteMatter(device="cpu", precision="fp32")
    m._model = StubModel()
    m._processor = StubProcessor()
    m._dtype = torch.float32
    return m


def _frame_u8(h: int, w: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.random((h, w, 3)) * 255).astype(np.uint8)


def _mask_disk(h: int, w: int, cy: int, cx: int, r: int) -> np.ndarray:
    yy, xx = np.ogrid[:h, :w]
    return ((yy - cy) ** 2 + (xx - cx) ** 2 <= r * r).astype(np.float32)


def _whole_frame_alpha(matter, frame_u8, tri_u8, edge_radius=4, **kw):
    """Reference: one tile large enough to cover the frame."""
    h, w = frame_u8.shape[:2]
    m_np = (tri_u8 > 127).astype(np.float32)
    bbox = (0, 0, w, h)
    opts = dict(kw)
    opts.setdefault("tile_size", max(h, w) + 64)
    out = matter._matte_multi_frame(
        frame_u8, [m_np], [bbox], h, w, edge_radius,
        external_trimaps=[tri_u8],
        **opts,
    )
    return out[0].numpy()


def test_verify_tile_logic_passes():
    ViTMatteMatter._verify_tile_logic()


def test_tile_placement_property():
    rng = np.random.default_rng(42)
    tile, overlap = 512, 64
    for _ in range(200):
        H = int(rng.integers(1, 3001))
        W = int(rng.integers(1, 3001))
        pad = 8
        y0 = int(rng.integers(0, H))
        x0 = int(rng.integers(0, W))
        y1 = int(rng.integers(y0 + 1, H + 1))
        x1 = int(rng.integers(x0 + 1, W + 1))
        bbox = (x0, y0, x1, y1)
        positions = ViTMatteMatter._object_tile_positions(
            bbox, H, W, pad, tile_size=tile, tile_overlap=overlap,
        )
        px0, py0 = max(0, x0 - pad), max(0, y0 - pad)
        px1, py1 = min(W, x1 + pad), min(H, y1 + pad)
        assert positions, "non-empty bbox must yield at least one tile"
        for tx, ty in positions:
            assert 0 <= tx < W and 0 <= ty < H
            assert tx + min(tile, W - tx) <= W
            assert ty + min(tile, H - ty) <= H
        xs = sorted({p[0] for p in positions})
        ys = sorted({p[1] for p in positions})
        assert xs[0] <= px0
        assert xs[-1] + tile >= px1 or W <= tile
        assert ys[0] <= py0
        assert ys[-1] + tile >= py1 or H <= tile
        if len(xs) > 1:
            for a, b in zip(xs, xs[1:]):
                assert (a + tile) - b >= overlap
        if len(ys) > 1:
            for a, b in zip(ys, ys[1:]):
                assert (a + tile) - b >= overlap


@pytest.mark.parametrize("h,w", [(300, 200), (512, 512), (1300, 900)])
def test_tiled_matches_whole_frame_stub(matter, h, w):
    frame = _frame_u8(h, w, seed=h + w)
    tri = np.zeros((h, w), dtype=np.uint8)
    tri[20:h - 20, 20:w - 20] = 127
    tri[40:h - 40, 40:w - 40] = 255
    ref = _whole_frame_alpha(matter, frame, tri, tile_size=128, tile_overlap=32)
    matter._model.forward_calls = 0
    m_np = (tri > 127).astype(np.float32)
    ys, xs = np.where(tri > 0)
    bbox = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    tiled = matter._matte_multi_frame(
        frame, [m_np], [bbox], h, w, 4,
        tile_size=128, tile_overlap=32, tile_batch=4,
        external_trimaps=[tri],
    )[0].numpy()
    mask = tri > 0
    np.testing.assert_allclose(tiled[mask], ref[mask], rtol=1e-4, atol=1e-4)


def test_hair_edge_tiled_beats_downscale_path(matter):
    h, w = 600, 800
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    rng = np.random.default_rng(7)
    for _ in range(80):
        y = int(rng.integers(50, h - 50))
        x0 = int(rng.integers(20, w - 200))
        x1 = x0 + int(rng.integers(80, 200))
        frame[y, x0:x1] = 220
    tri = np.zeros((h, w), dtype=np.uint8)
    tri[40:h - 40, 30:w - 30] = 127
    tri[45:h - 45, 35:w - 35] = 255
    m_np = (tri > 127).astype(np.float32)
    ys, xs = np.where(tri > 0)
    bbox = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    ref = _whole_frame_alpha(matter, frame, tri, tile_size=h + 64, tile_overlap=32)
    tiled = matter._matte_multi_frame(
        frame, [m_np], [bbox], h, w, 4,
        tile_size=128, tile_overlap=64, tile_batch=2,
        external_trimaps=[tri],
    )[0].numpy()

    def _down_up_path():
        import torch.nn.functional as F
        fr = torch.from_numpy(frame).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        tri_t = torch.from_numpy(tri).float().unsqueeze(0).unsqueeze(0) / 255.0
        small = 512
        fr_s = F.interpolate(fr, size=(small, small), mode="bilinear", align_corners=False)
        tri_s = F.interpolate(tri_t, size=(small, small), mode="nearest")
        fr_u8 = (fr_s[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        tri_u8 = (tri_s[0, 0].numpy() * 255).astype(np.uint8)
        a_s = _whole_frame_alpha(
            matter, fr_u8, tri_u8, tile_size=small + 32, tile_overlap=32,
        )
        a_t = torch.from_numpy(a_s).unsqueeze(0).unsqueeze(0)
        a_up = F.interpolate(a_t, size=(h, w), mode="bilinear", align_corners=False)[0, 0].numpy()
        return a_up

    down = _down_up_path()
    unk = (tri == 127)
    sad_tiled = float(np.abs(tiled[unk] - ref[unk]).sum())
    sad_down = float(np.abs(down[unk] - ref[unk]).sum())
    assert sad_tiled < sad_down


def test_two_touching_objects_independent(matter):
    h, w = 256, 256
    frame = _frame_u8(h, w, seed=99)
    m0 = np.zeros((h, w), np.float32)
    m1 = np.zeros((h, w), np.float32)
    m0[60:200, 40:120] = 1.0
    m1[60:200, 120:200] = 1.0
    b0 = (40, 60, 120, 200)
    b1 = (120, 60, 200, 200)
    alphas = matter._matte_multi_frame(
        frame, [m0, m1], [b0, b1], h, w, 4,
        tile_size=96, tile_overlap=32, tile_batch=4,
    )
    excl0 = m0 > 0.5
    excl1 = m1 > 0.5
    only0 = excl0 & ~excl1
    only1 = excl1 & ~excl0
    assert float(alphas[0].numpy()[only1].max()) < 1e-3
    assert float(alphas[1].numpy()[only0].max()) < 1e-3


def test_overlapping_objects_keep_their_shared_pixels(matter):
    """Ownership masking must not cut a hole where two objects genuinely overlap."""
    h, w = 256, 256
    frame = np.full((h, w, 3), 230, np.uint8)            # bright: the luminance stub gives high alpha
    m0 = np.zeros((h, w), np.float32)
    m1 = np.zeros((h, w), np.float32)
    m0[60:200, 40:150] = 1.0
    m1[60:200, 110:200] = 1.0                             # columns 110..149 are shared
    alphas = matter._matte_multi_frame(
        frame, [m0, m1], [(40, 60, 150, 200), (110, 60, 200, 200)], h, w, 4,
        tile_size=96, tile_overlap=32, tile_batch=4,
    )
    shared = (m0 > 0.5) & (m1 > 0.5)
    merged = np.maximum(alphas[0].numpy(), alphas[1].numpy())
    assert float(merged[shared].min()) > 0.5
    assert float(alphas[0].numpy()[shared].min()) > 0.5 and float(alphas[1].numpy()[shared].min()) > 0.5


def test_empty_mask_no_forward_no_load():
    m = ViTMatteMatter(device="cpu", precision="fp32")
    assert m._model is None
    img = torch.rand(1, 64, 64, 3)
    coarse = torch.zeros(1, 64, 64)
    out = m.matte(img, coarse, tile_size=128)
    assert m._model is None
    assert out["alpha"].sum() == 0


def test_external_trimap_honoured(matter):
    h, w = 128, 128
    frame = _frame_u8(h, w)
    coarse = torch.zeros(1, h, w)
    tri = torch.zeros(1, h, w)
    tri[0, 40:88, 40:88] = 0.5
    tri[0, 50:78, 50:78] = 1.0
    out = matter.matte(
        torch.from_numpy(frame).float().div(255.0).unsqueeze(0),
        coarse,
        trimap=tri,
        tile_size=64,
        tile_overlap=16,
    )
    alpha = out["alpha"][0].numpy()
    assert alpha[44, 44] > 0.05
    assert alpha[10, 10] == 0.0


def test_onyx_zero_coarse_with_trimap_returns_unknown_alpha(matter):
    """ONYX calls matte(coarse=0, trimap=sub_tri); must not return all zeros."""
    h, w = 64, 64
    img = torch.rand(1, h, w, 3)
    coarse = torch.zeros(1, h, w)
    tri = torch.zeros(1, h, w)
    tri[0, 20:44, 20:44] = 0.5
    tri[0, 26:38, 26:38] = 1.0
    out = matter.matte(img, coarse, trimap=tri, tile_size=32, tile_overlap=8)
    alpha = out["alpha"][0]
    assert float(alpha[30, 30]) > 0.05
    assert float(alpha[5, 5]) == 0.0


def test_tile_batch_forward_count(matter):
    h, w = 300, 300
    frame = _frame_u8(h, w)
    tri = np.zeros((h, w), dtype=np.uint8)
    tri[10:260, 10:260] = 127
    tri[40:230, 40:230] = 255
    m_np = (tri > 127).astype(np.float32)
    bbox = (10, 10, 260, 260)
    matter._model.forward_calls = 0
    matter._model.batch_sizes = []
    matter._matte_multi_frame(
        frame, [m_np], [bbox], h, w, 4,
        tile_size=100, tile_overlap=25, tile_batch=3,
        external_trimaps=[tri],
    )
    # The tile count comes from the placement helper (pad = 2 * edge_radius), not a guess:
    # region 2..268 = 266 px with tile 100 / overlap 25 needs 4 tiles per axis -> 16 jobs.
    n_tiles = len(matter._object_tile_positions(bbox, h, w, 2 * 4, tile_size=100, tile_overlap=25))
    assert n_tiles > 3
    assert matter._model.forward_calls == math.ceil(n_tiles / 3)
    assert sum(matter._model.batch_sizes) == n_tiles
    assert all(b == 3 for b in matter._model.batch_sizes[:-1])


def test_oom_fallback_same_result(matter):
    h, w = 400, 400
    frame = _frame_u8(h, w, seed=3)
    tri = np.zeros((h, w), dtype=np.uint8)
    tri[50:350, 50:350] = 127
    tri[100:300, 100:300] = 255
    m_np = (tri > 127).astype(np.float32)
    bbox = (50, 50, 350, 350)

    matter_ok = ViTMatteMatter(device="cpu", precision="fp32")
    matter_ok._model = StubModel()
    matter_ok._processor = StubProcessor()
    matter_ok._dtype = torch.float32
    ref = matter_ok._matte_multi_frame(
        frame, [m_np], [bbox], h, w, 4,
        tile_size=128, tile_overlap=32, tile_batch=4,
        external_trimaps=[tri],
    )[0]

    matter_oom = ViTMatteMatter(device="cpu", precision="fp32")
    matter_oom._model = StubModel(oom_on_batch_gt=1)
    matter_oom._processor = StubProcessor()
    matter_oom._dtype = torch.float32
    got = matter_oom._matte_multi_frame(
        frame, [m_np], [bbox], h, w, 4,
        tile_size=128, tile_overlap=32, tile_batch=4,
        external_trimaps=[tri],
    )[0]
    torch.testing.assert_close(got, ref, atol=1e-5, rtol=1e-4)


def test_b3_video_frames_independent(matter):
    B, h, w = 3, 96, 96
    img = torch.stack([
        torch.from_numpy(_frame_u8(h, w, seed=i)).float().div(255.0)
        for i in range(B)
    ])
    masks = torch.zeros(B, h, w)
    for i in range(B):
        masks[i, 20 + i * 5:70, 20:70] = 1.0
    out = matter.matte(img, masks, tile_size=48, tile_overlap=16)
    assert out["alpha"].shape == (B, h, w)
    for i in range(B):
        assert out["alpha"][i, 25 + i * 5, 30] > 0.05


@pytest.mark.parametrize("h,w", [(1, 1), (7, 5), (64, 48)])
def test_tiny_images(matter, h, w):
    frame = _frame_u8(h, w)
    m_np = np.ones((h, w), np.float32)
    bbox = (0, 0, w, h)
    alphas = matter._matte_multi_frame(
        frame, [m_np], [bbox], h, w, 2,
        tile_size=64, tile_overlap=16, tile_batch=2,
    )
    assert alphas[0].shape == (h, w)
    assert float(alphas[0].max()) > 0.0
