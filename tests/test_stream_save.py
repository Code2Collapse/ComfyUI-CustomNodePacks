"""L7.59 stream save: Save Video (C2C) with latent + vae decodes and encodes as one stream.

Exactness against core's REAL VAE classes is proven outside pytest (core is stubbed here): docs/evidence/L7.59/
stream_exactness.json - Wan 2.1, Wan 2.2, MiniMax H3, LTX all torch.equal to core's own decode. These tests cover the
plumbing with stand-in VAEs that follow the same protocols: the frame stream, the node, the file, the warnings.
"""
from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

PACK = Path(__file__).resolve().parents[1]
if str(PACK) not in sys.path:
    sys.path.insert(0, str(PACK))

av = pytest.importorskip("av")


@pytest.fixture
def sd():
    from nodes.c2c_video import stream_decode
    return stream_decode


def _chunks(frames, sizes):
    i = 0
    for s in sizes:
        yield frames[i:i + s]
        i += s


def test_frame_stream_reads_in_order_once(sd):
    frames = torch.arange(10 * 2 * 2 * 3, dtype=torch.float32).view(10, 2, 2, 3)
    fs = sd.FrameStream(_chunks(frames, [1, 4, 4, 1]), 10, 2, 2, 3)
    assert fs.shape == (10, 2, 2, 3) and fs.ndim == 4
    assert torch.equal(fs[0:3], frames[0:3])
    assert torch.equal(fs[3], frames[3])
    assert torch.equal(fs[4:10], frames[4:10]) and torch.equal(fs.last_frame, frames[9:10])
    with pytest.raises(sd.StreamDecodeError, match="already handed on"):
        fs[2]


def test_frame_stream_reports_a_short_decoder(sd):
    fs = sd.FrameStream(_chunks(torch.zeros(3, 2, 2, 3), [3]), 5, 2, 2, 3)
    with pytest.raises(sd.StreamDecodeError, match="produced 3 frames, 5 were expected"):
        fs[0:5]


# ── stand-in VAEs following core's protocols ────────────────────────────────────────────────────────────────────
def _frame_values(t_out, h, w):
    """Decoded frame k is a flat grey of k / t_out (in [-1, 1] before process_output)."""
    k = torch.arange(t_out, dtype=torch.float32).view(1, 1, t_out, 1, 1) / max(1, t_out) * 2 - 1
    return k.expand(1, 3, t_out, h, w).clone()


def _wan_like_module():
    m = types.ModuleType("standin.wan.vae")

    class Decoder:
        def __init__(self):
            self.calls = 0

        def __call__(self, x, feat_cache=None, feat_idx=None):
            # causal: the first step makes 1 frame, every later latent frame makes 4
            n_lat = x.shape[2]
            out_t = 1 if self.calls == 0 else 4 * n_lat
            start = 0 if self.calls == 0 else self.produced
            self.calls += 1
            self.produced = start + out_t
            full = _frame_values(self.total, x.shape[3] * 8, x.shape[4] * 8)
            return [full[:, :, start:start + out_t]]

    class WanVAE:
        def __init__(self, t_lat):
            self.decoder = Decoder()
            self.decoder.total = 1 + 4 * (t_lat - 1)
            self.conv2 = lambda z: z

    WanVAE.__module__ = m.__name__                            # what the stream's family check reads
    m.WanVAE = WanVAE
    m.count_cache_layers = lambda dec: 3
    sys.modules[m.__name__] = m
    return m


class _ChunkedFSM:
    comfy_has_chunked_io = True

    def decode_output_shape(self, shape):
        return (shape[0], 3, 1 + 4 * (shape[2] - 1), shape[3] * 8, shape[4] * 8)

    def decode(self, z, output_buffer=None):
        shape = self.decode_output_shape(z.shape)
        full = _frame_values(shape[2], shape[3], shape[4])
        for t0 in range(0, shape[2], 3):                       # finished ranges, in order
            output_buffer[:, :, t0:t0 + 3, :, :].copy_(full[:, :, t0:t0 + 3])
        return output_buffer


class _WholeFSM:
    pass


def _vae(fsm, whole_frames=None):
    v = types.SimpleNamespace(first_stage_model=fsm, vae_dtype=torch.float32, device=torch.device("cpu"),
                              patcher=object(), disable_offload=False)
    v.memory_used_decode = lambda shape, dtype: 1
    v.process_output = lambda img: img.add_(1.0).div_(2.0).clamp_(0.0, 1.0)
    v.spacial_compression_decode = lambda: 8
    if whole_frames is not None:
        v.decode = lambda z: whole_frames
    return v


@pytest.fixture
def node(tmp_path, monkeypatch):
    out = tmp_path / "out"
    temp = tmp_path / "temp"
    out.mkdir(); temp.mkdir()
    fp = types.SimpleNamespace(
        get_output_directory=lambda: str(out), get_temp_directory=lambda: str(temp),
        get_save_image_path=lambda prefix, odir, w=0, h=0: (os.path.join(odir, os.path.dirname(os.path.normpath(prefix))),
                                                              os.path.basename(os.path.normpath(prefix)), 1,
                                                              os.path.dirname(os.path.normpath(prefix)), prefix))
    monkeypatch.setitem(sys.modules, "folder_paths", fp)
    mm = types.ModuleType("comfy.model_management")
    mm.load_models_gpu = lambda models, memory_required=0, force_full_load=False: None
    mm.throw_exception_if_processing_interrupted = lambda: None
    import comfy
    monkeypatch.setitem(sys.modules, "comfy.model_management", mm)
    monkeypatch.setattr(comfy, "model_management", mm, raising=False)
    from nodes.save_video import SaveVideoC2C
    return SaveVideoC2C(), out


def _decoded(path):
    from nodes.c2c_video.reader import _stamp, _to_rgb, _white_level
    outs = []
    with av.open(path) as c:
        for f in c.decode(c.streams.video[0]):
            deep = any(s in f.format.name for s in ("10", "12", "16"))
            fmt = "rgb48le" if deep else "rgb24"
            yuv = f.format.name.startswith("yuv")
            _stamp(f, "bt709" if yuv else None, "tv" if yuv else "pc")
            arr = _to_rgb(f, fmt).to_ndarray().astype(np.float64)
            outs.append(arr / _white_level(f.format.name, "bt709" if yuv else None, "tv" if yuv else "pc", fmt))
    return np.stack(outs)


@pytest.mark.parametrize("kind", ["wan_like", "chunked"])
def test_stream_save_writes_every_frame_in_order(node, kind):
    n, out = node
    t_lat = 4
    fsm = _wan_like_module().WanVAE(t_lat) if kind == "wan_like" else _ChunkedFSM()
    latent = {"samples": torch.zeros(1, 16, t_lat, 2, 3)}
    res = n.execute(format="MKV FFV1 (lossless)", fps=24.0, filename_prefix=f"s/{kind}", quality=80,
                    latent=latent, vae=_vae(fsm))
    ui = res["ui"]["c2c_save_video"][0]
    assert ui["frames"] == 13 and (ui["width"], ui["height"]) == (24, 16)
    assert ui["stream"]["exact"] and ui["stream"]["bounded"]
    got = _decoded(str(out / "s" / ui["filename"]))
    want = (np.arange(13) / 13 * 2 - 1 + 1) / 2                  # frame k grey level after process_output
    assert got.shape[0] == 13
    assert np.allclose(got[:, 0, 0, 0], want, atol=1 / 65535 * 2)
    assert res["result"][1].shape[0] == 1                          # only the last frame comes back, not the batch


def test_whole_decode_is_reported_not_hidden(node):
    n, out = node
    whole = torch.full((1, 5, 16, 24, 3), 0.5)
    res = n.execute(format="MKV FFV1 (lossless)", fps=24.0, filename_prefix="s/whole", quality=80,
                    latent={"samples": torch.zeros(1, 16, 2, 2, 3)}, vae=_vae(_WholeFSM(), whole))
    ui = res["ui"]["c2c_save_video"][0]
    assert ui["stream"]["bounded"] is False and ui["frames"] == 5
    assert any("no exact streaming decode" in w for w in ui["warnings"])


def test_images_and_latent_together_are_refused(node):
    n, _ = node
    from nodes.c2c_video.save_encode import SaveVideoError
    with pytest.raises(SaveVideoError, match="not both"):
        n.execute(format="MKV FFV1 (lossless)", fps=24.0, filename_prefix="s/x", quality=80,
                  images=torch.zeros(2, 8, 8, 3), latent={"samples": torch.zeros(1, 16, 2, 1, 1)}, vae=_vae(_ChunkedFSM()))
    with pytest.raises(SaveVideoError, match="both latent and vae"):
        n.execute(format="MKV FFV1 (lossless)", fps=24.0, filename_prefix="s/x", quality=80,
                  latent={"samples": torch.zeros(1, 16, 2, 1, 1)})


def test_disk_tiles_equal_the_spatial_tiled_decode(sd, tmp_path):
    # Real-weight proof (Wan 2.1): docs/evidence/L7.59/tiled_stream_exactness.py. Here: a causal stand-in whose output
    # depends on the latent content, so every tile differs and the blend is really exercised.
    from nodes._vae_tiled import decode_wan_spatial_tiled

    mod = types.ModuleType("standin2.wan.vae")

    class Decoder:
        def __call__(self, x, feat_cache=None, feat_idx=None):
            reps = 1 if x.shape[2] == 1 and feat_cache[0] is None else 4
            feat_cache[0] = True
            up = x[:, :3].repeat_interleave(8, -1).repeat_interleave(8, -2)        # 8x spatial, content-dependent
            return [torch.tanh(up.repeat_interleave(reps, 2))]

    class WanVAE:
        def __init__(self):
            self.decoder = Decoder()
            self.conv2 = lambda z: z

    WanVAE.__module__ = mod.__name__
    mod.WanVAE, mod.count_cache_layers = WanVAE, (lambda dec: 1)
    sys.modules[mod.__name__] = mod
    fsm = WanVAE()
    vae = _vae(fsm)

    def full_decode(z):                       # what core's VAE.decode returns for the stand-in
        out = torch.cat(list(sd._wan_steps(fsm, z, "wan21")), 2).float()
        return vae.process_output(out).movedim(1, -1)

    vae.decode = full_decode
    z = torch.randn(1, 16, 3, 10, 14, generator=torch.Generator().manual_seed(1))
    ref = decode_wan_spatial_tiled(vae, z, 4, 1)
    got = torch.cat(list(sd._wan_tiles_via_disk(vae, fsm, z, "wan21", 32, str(tmp_path))), 0)
    assert torch.equal(ref.reshape(-1, *ref.shape[-3:]), got)
    assert not list(tmp_path.iterdir())                          # tile files removed
