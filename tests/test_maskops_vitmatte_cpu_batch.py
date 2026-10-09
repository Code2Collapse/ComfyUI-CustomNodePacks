"""L2.12: on the CPU, MaskOps runs ViTMatte tiles one at a time.

Measured with the real hustvl/vitmatte-small-composition-1k weights on 1536x1152 hair plates (docs/evidence/L2.12 in
the work area): the node's default 1024 px tiles in a batch of 4 peaked at 7.66 GB and took 9.3 s; one tile at a time
2.54 GB and 5.8 s. Batch size does not change which tiles run, so the matte is the same.
"""
from __future__ import annotations

import pytest


@pytest.fixture
def node():
    # imported inside the test, never at collection (see test_maskops_legacy_values.py)
    return pytest.importorskip("nodes.mask_matting.node")


def test_cpu_runs_one_tile_at_a_time(node):
    kw = node._vitmatte_tile_kwargs(1024, 64, 4, "cpu")
    assert kw == {"tile_size": 1024, "tile_overlap": 64, "tile_batch": 1}


def test_gpu_keeps_the_batch_setting(node):
    assert node._vitmatte_tile_kwargs(1024, 64, 4, "cuda")["tile_batch"] == 4
    assert node._vitmatte_tile_kwargs(512, 128, 8, "cuda:1")["tile_batch"] == 8


def test_tile_floor_and_batch_floor(node):
    kw = node._vitmatte_tile_kwargs(16, 32, 0, "cuda")
    assert kw["tile_size"] == 64 and kw["tile_batch"] == 1


def test_same_matte_one_tile_at_a_time():
    # the claim the CPU rule rests on: batching changes how tiles are grouped, not the result
    # (real weights: 1024/128 batch 4 and batch 1 scored identically on all three plates, tile_sweep.json)
    np = pytest.importorskip("numpy")
    torch = pytest.importorskip("torch")
    vm = pytest.importorskip("nodes.mask_matting.matters.vitmatte_backend")

    class Proc:          # HF processor shape: pixel_values [1,4,h,w] from RGB + trimap
        def __call__(self, images, trimaps, return_tensors="pt"):
            rgb = np.asarray(images, np.float32).transpose(2, 0, 1) / 255.0
            tri = np.asarray(trimaps, np.float32)[None] / 255.0
            return {"pixel_values": torch.from_numpy(np.concatenate([rgb, tri], 0)).unsqueeze(0)}

    class Model:         # a per-pixel alpha, so tile grouping cannot matter unless blending is wrong
        def __call__(self, pixel_values):
            lum = pixel_values[:, :3].mean(1, keepdim=True) * 0.8 + pixel_values[:, 3:4] * 0.2
            return type("Out", (), {"alphas": lum})()

    rng = np.random.default_rng(5)
    frame = (rng.random((300, 300, 3)) * 255).astype(np.uint8)
    tri = np.zeros((300, 300), dtype=np.uint8)
    tri[20:280, 20:280] = 127
    tri[60:240, 60:240] = 255
    m_np = (tri > 127).astype(np.float32)

    def run(batch):
        m = vm.ViTMatteMatter(device="cpu", precision="fp32")
        m._model, m._processor, m._dtype = Model(), Proc(), torch.float32
        return m._matte_multi_frame(frame, [m_np], [(20, 20, 280, 280)], 300, 300, 4,
                                    tile_size=128, tile_overlap=32, tile_batch=batch, external_trimaps=[tri])[0]

    torch.testing.assert_close(run(1), run(4), atol=1e-6, rtol=1e-5)
