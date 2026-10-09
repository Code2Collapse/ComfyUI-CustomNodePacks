"""CPU tests for VAE precision restrictions (R1-R3) and C2CVAEQualityDecode fixes."""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest
import torch
import torch.nn as nn

PACK = Path(__file__).resolve().parents[1]
if str(PACK) not in sys.path:
    sys.path.insert(0, str(PACK))

from nodes._vae_tiled import (  # noqa: E402
    _build_feather_mask,
    _plan_tile_spans,
    _standard_decode,
    decode_wan_spatial_tiled,
    standard_output_mapping,
    unclamped_output,
    vae_compute_dtype,
)


def _core_process_output(t):
    return t.add_(1.0).div_(2.0).clamp_(0.0, 1.0)


def _unclamped_process_output(t):
    return t.add_(1.0).div_(2.0)


class FakeWeight(nn.Module):
    def __init__(self, dtype=torch.float32):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, dtype=dtype))


class LegacyFakeVAE:
    def __init__(self, weight_dtype=torch.float32, dynamic: bool = False):
        self.first_stage_model = FakeWeight(weight_dtype)
        self.vae_dtype = weight_dtype
        self._dynamic = dynamic
        self.last_input_dtype = None

    def is_dynamic(self) -> bool:
        return self._dynamic

    def spacial_compression_decode(self) -> int:
        return 8

    process_output = staticmethod(_core_process_output)

    def _prepare_decode_input(self, x: torch.Tensor) -> torch.Tensor:
        vae_dt = getattr(self, "vae_dtype", None)
        if vae_dt is not None and x.dtype != vae_dt:
            x = x.to(vae_dt)
        self.last_input_dtype = x.dtype
        w = self.first_stage_model.weight
        if self.is_dynamic():
            if x.dtype != w.dtype:
                self._weight_snapshot = w.data.clone()
                w.data = self._weight_snapshot.to(x.dtype)
        elif x.dtype != w.dtype:
            raise RuntimeError(
                f"Input type ({x.dtype}) and bias type ({w.dtype}) should be the same"
            )
        return x

    def _finish_decode(self) -> None:
        snap = getattr(self, "_weight_snapshot", None)
        if snap is not None:
            self.first_stage_model.weight.data = snap
            del self._weight_snapshot

    def _decode_tensor(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 5:
            b, _c, t_lat, h, w = x.shape
            t_out = 1 + 4 * (t_lat - 1)
            raw = x.new_zeros(b, t_out, h * 8, w * 8, 3)
            for bi in range(b):
                for ti in range(t_out):
                    raw[bi, ti] = x[bi, 0, min(ti, t_lat - 1)].mean() * 0.01
            return self.process_output(raw * 2.0 - 1.0)
        raw = x.new_zeros(*x.shape[:3], 3)
        raw[:] = x.mean(dim=1, keepdim=True).permute(0, 2, 3, 1) * 0.01
        return self.process_output(raw * 2.0 - 1.0)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        x = self._prepare_decode_input(x)
        try:
            return self._decode_tensor(x)
        finally:
            self._finish_decode()


class LocalSpatialFakeVAE(LegacyFakeVAE):
    """Decode each spatial position from its latent cell only (no cross-tile mixing)."""

    def _decode_tensor(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError("expected 5D")
        b, _c, t_lat, h, w = x.shape
        t_out = 1 + 4 * (t_lat - 1)
        up = 8
        raw = x.new_zeros(b, t_out, h * up, w * up, 3)
        cell = x[:, 0].mean(dim=1)
        for yi in range(h):
            for xi in range(w):
                v = cell[:, yi, xi]
                raw[:, :, yi * up:(yi + 1) * up, xi * up:(xi + 1) * up, :] = (
                    v.view(b, 1, 1, 1, 1)
                )
        return self.process_output(raw * 2.0 - 1.0)


class DynamicFp32RejectVAE(LegacyFakeVAE):
    def __init__(self):
        super().__init__(weight_dtype=torch.bfloat16, dynamic=True)
        self.vae_dtype = torch.bfloat16

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        if self.vae_dtype == torch.float32:
            raise RuntimeError(
                "Input type (torch.float32) and bias type (torch.bfloat16) should be the same"
            )
        return super().decode(x)


# ── R1 / B1 ─────────────────────────────────────────────────────────────────

def test_vae_compute_dtype_legacy_restores_after_success():
    vae = LegacyFakeVAE(weight_dtype=torch.bfloat16)
    vae.vae_dtype = torch.bfloat16
    x = torch.randn(1, 4, 2, 8, 8, dtype=torch.bfloat16)
    with vae_compute_dtype(vae, torch.float32):
        _standard_decode(vae, x)
    assert vae.vae_dtype == torch.bfloat16
    assert vae.first_stage_model.weight.dtype == torch.bfloat16


def test_vae_compute_dtype_legacy_restores_after_exception():
    vae = LegacyFakeVAE(weight_dtype=torch.bfloat16)
    vae.vae_dtype = torch.bfloat16

    def bad_decode(_x):
        raise RuntimeError("decode failed")

    vae.decode = bad_decode
    with pytest.raises(RuntimeError, match="decode failed"):
        with vae_compute_dtype(vae, torch.float32):
            vae.decode(torch.zeros(1))
    assert vae.vae_dtype == torch.bfloat16
    assert vae.first_stage_model.weight.dtype == torch.bfloat16


def test_vae_compute_dtype_dynamic_leaves_weights():
    vae = LegacyFakeVAE(weight_dtype=torch.bfloat16, dynamic=True)
    vae.vae_dtype = torch.bfloat16
    x = torch.randn(1, 4, 2, 4, 4, dtype=torch.bfloat16)
    with vae_compute_dtype(vae, torch.float32):
        _standard_decode(vae, x)
    assert vae.vae_dtype == torch.bfloat16
    assert vae.first_stage_model.weight.dtype == torch.bfloat16


def test_vae_compute_dtype_noop_when_already_set():
    vae = LegacyFakeVAE()
    before = vae.first_stage_model.weight.data.clone()
    with vae_compute_dtype(vae, torch.float32):
        pass
    assert torch.equal(vae.first_stage_model.weight.data, before)


# ── R2 ──────────────────────────────────────────────────────────────────────

def test_standard_output_mapping_accepts_core_lambda():
    vae = LegacyFakeVAE()
    assert standard_output_mapping(vae) is True


def test_standard_output_mapping_rejects_identity():
    vae = LegacyFakeVAE()
    vae.process_output = lambda t: t
    assert standard_output_mapping(vae) is False


def test_unclamped_output_allows_values_above_one():
    vae = LegacyFakeVAE()
    with unclamped_output(vae):
        out = vae.process_output(torch.tensor([3.0]))
        assert float(out) > 1.0
    assert standard_output_mapping(vae)


def test_unclamped_output_restores_process_output():
    vae = LegacyFakeVAE()
    orig = vae.process_output
    with unclamped_output(vae):
        assert vae.process_output is not orig
    assert vae.process_output is orig


# ── R3 / B5 / B6 ────────────────────────────────────────────────────────────

def test_plan_tile_spans_covers_range():
    for total in range(1, 65):
        for tile in range(1, 41):
            for overlap in range(0, 65):
                tile_eff = max(1, tile)
                eff_overlap = min(max(0, overlap), (tile_eff - 1) // 2)
                spans = _plan_tile_spans(total, tile, overlap)
                assert spans[0][0] == 0
                assert spans[-1][1] == total
                starts = [s[0] for s in spans]
                assert starts == sorted(starts)
                assert len(starts) == len(set(starts))
                for i in range(len(spans) - 1):
                    assert spans[i][1] > spans[i][0]
                    assert spans[i + 1][0] <= spans[i][1]
                    if eff_overlap > 0:
                        a0, a1 = spans[i]
                        b0, b1 = spans[i + 1]
                        if (a1 - a0) == tile_eff and (b1 - b0) == tile_eff:
                            assert spans[i + 1][0] < spans[i][1]


def test_tiled_frame_count_matches_plain():
    vae = LegacyFakeVAE()
    lat = torch.randn(1, 4, 5, 16, 16)
    plain = _standard_decode(vae, lat)
    tiled = decode_wan_spatial_tiled(vae, lat, tile_latent=8, overlap_latent=2)
    assert plain.shape[1] == tiled.shape[1]
    assert plain.shape[1] == 1 + 4 * (5 - 1)


def test_tiled_effective_overlap_leaves_no_zero_weight_pixels():
    vae = LegacyFakeVAE()
    lat = torch.randn(1, 4, 2, 16, 16)
    tile_latent = 2
    overlap_latent = 4
    factor = 8
    eff = min(max(0, overlap_latent), (tile_latent - 1) // 2)
    overlap_px = eff * factor
    h_tiles = _plan_tile_spans(16, tile_latent, eff)
    w_tiles = _plan_tile_spans(16, tile_latent, eff)
    out_h, out_w = 16 * factor, 16 * factor
    weight = torch.zeros(1, 1, out_h, out_w, 1)
    for h_start, h_end in h_tiles:
        for w_start, w_end in w_tiles:
            mask = _build_feather_mask(
                (h_end - h_start) * factor, (w_end - w_start) * factor,
                overlap_px,
                h_start == 0, h_end == 16, w_start == 0, w_end == 16,
                torch.float32, lat.device,
            )
            oh, ow = h_start * factor, w_start * factor
            weight[:, :, oh:oh + mask.shape[2], ow:ow + mask.shape[3], :] += mask[:1, :1]
    assert float(weight.min()) > 0.0
    tiled = decode_wan_spatial_tiled(vae, lat, tile_latent, overlap_latent)
    plain = _standard_decode(vae, lat)
    assert tiled.shape == plain.shape
    assert torch.isfinite(tiled).all()


def test_tiled_matches_plain_on_local_fake():
    vae = LocalSpatialFakeVAE()
    lat = torch.randn(1, 4, 3, 24, 24)
    plain = _standard_decode(vae, lat)
    tiled = decode_wan_spatial_tiled(vae, lat, tile_latent=8, overlap_latent=2)
    assert tiled.shape == plain.shape
    assert torch.allclose(tiled, plain, atol=1e-5)


def test_standard_decode_returns_five_d():
    vae = LegacyFakeVAE()
    lat = torch.randn(1, 4, 2, 4, 4)
    out = _standard_decode(vae, lat)
    assert out.ndim == 5


# ── C2CVAEQualityDecode node ────────────────────────────────────────────────

@pytest.fixture
def decode_node():
    for mod in ("folder_paths", "comfy", "comfy.utils"):
        if mod not in sys.modules:
            stub = types.ModuleType(mod)
            stub.__path__ = []
            sys.modules[mod] = stub
    from nodes.hdr_color_science import C2CVAEQualityDecode
    return C2CVAEQualityDecode()


def test_decode_never_casts_latent(decode_node):
    vae = LegacyFakeVAE()
    lat = torch.randn(1, 4, 2, 8, 8, dtype=torch.float16)
    (out,) = decode_node.decode({"samples": lat}, vae, False, 0, False, 1.0, True)
    assert lat.dtype == torch.float16


def test_decode_reshapes_five_d_to_four_d(decode_node):
    vae = LegacyFakeVAE()
    lat = torch.randn(1, 4, 3, 4, 4)
    (out,) = decode_node.decode({"samples": lat}, vae, False, 0, False, 1.0, True)
    assert out.ndim == 4
    assert out.shape[0] == 1 + 4 * (3 - 1)


def test_decode_trims_c2c_source_frames(decode_node):
    vae = LegacyFakeVAE()
    lat = torch.randn(1, 4, 5, 4, 4)
    (out,) = decode_node.decode(
        {"samples": lat, "c2c_source_frames": 3}, vae, False, 0, False, 1.0, True,
    )
    assert out.shape[0] == 3


def test_decode_tiled_failure_is_runtime_error_not_fallback(decode_node):
    vae = LegacyFakeVAE()

    def boom(_x):
        raise ValueError("tiled internal failure")

    vae.decode = boom
    lat = torch.randn(1, 4, 2, 32, 32)
    with pytest.raises(RuntimeError, match="Spatial-tiled VAE decode failed"):
        decode_node.decode({"samples": lat}, vae, False, 256, False, 1.0, True, tile_mode="manual")


def test_b1_legacy_fp32_decode_succeeds(decode_node):
    vae = LegacyFakeVAE(weight_dtype=torch.bfloat16)
    vae.vae_dtype = torch.bfloat16
    lat = torch.randn(1, 4, 2, 4, 4, dtype=torch.bfloat16)
    (out,) = decode_node.decode({"samples": lat}, vae, True, 0, False, 1.0, True)
    assert out.ndim == 4
    assert vae.vae_dtype == torch.bfloat16
    assert vae.first_stage_model.weight.dtype == torch.bfloat16


def test_dynamic_fp32_retry(decode_node):
    vae = DynamicFp32RejectVAE()
    lat = torch.randn(1, 4, 2, 4, 4, dtype=torch.bfloat16)
    (out,) = decode_node.decode({"samples": lat}, vae, True, 0, False, 1.0, True)
    assert out.ndim == 4
    assert vae.vae_dtype == torch.bfloat16


def test_dynamic_fp32_retry_not_on_other_errors(decode_node):
    vae = DynamicFp32RejectVAE()

    def boom(_x):
        raise RuntimeError("out of memory")

    vae.decode = boom
    lat = torch.randn(1, 4, 2, 4, 4, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="out of memory"):
        decode_node.decode({"samples": lat}, vae, True, 0, False, 1.0, True)


# ── L7.42: apply_aces on an sRGB-encoded decode ─────────────────────────────

def _ramp_decode(node, monkeypatch, exposure=1.0):
    ramp = torch.linspace(0.0, 1.0, 256).view(1, 1, 256, 1).expand(1, 4, 256, 3).contiguous()
    monkeypatch.setattr(node, "_run_decode", lambda vae, latent, tile_size: ramp.clone())
    (out,) = node.decode({"samples": torch.zeros(1, 4, 1, 1)}, LegacyFakeVAE(), False, 0, True, exposure, True)
    return ramp, out


def test_aces_decode_matches_the_tonemap_node_from_srgb(decode_node, monkeypatch):
    from nodes.hdr_color_science import C2CACESTonemap
    for exposure in (1.0, 1.6):
        ramp, out = _ramp_decode(decode_node, monkeypatch, exposure)
        (ref,) = C2CACESTonemap().apply_tonemap(ramp, "sRGB", exposure, 1.0, 1.0, "sRGB (gamma)")
        torch.testing.assert_close(out, ref, atol=1e-6, rtol=1e-5)


def test_aces_decode_no_double_gamma(decode_node, monkeypatch):
    # sRGB 0.4614 is scene mid-grey 0.18. ACES(0.18) = 0.2669 linear = 0.553 sRGB. Treating the encoded value as
    # linear (the bug) gave 0.79: every mid-tone lifted toward white.
    ramp, out = _ramp_decode(decode_node, monkeypatch)
    i = int(torch.argmin((ramp[0, 0, :, 0] - 0.4614).abs()))
    assert abs(float(out[0, 0, i, 0]) - 0.553) < 0.01
    assert float(out[0, 0, 0, 0]) == 0.0                       # black stays black
    assert torch.all(out[0, 0, 1:, 0] >= out[0, 0, :-1, 0])     # monotonic


# ── L7.59: tile_mode off / auto / manual ────────────────────────────────────

class _GpuLikeWanVAE(LegacyFakeVAE):
    """Core's Wan 2.1 estimate on a pretend 8 GB card (the decode itself is the fake's)."""

    device = types.SimpleNamespace(type="cuda")

    def memory_used_decode(self, shape, dtype):
        return (2200 if shape[2] <= 4 else 7000) * shape[3] * shape[4] * 64 * 2   # bytes, bf16


def _free(monkeypatch, gb):
    import comfy
    mm = types.ModuleType("comfy.model_management")
    mm.get_free_memory = lambda dev=None: gb * 2**30
    monkeypatch.setitem(sys.modules, "comfy.model_management", mm)
    monkeypatch.setattr(comfy, "model_management", mm, raising=False)


def test_auto_picks_the_largest_tile_that_fits(decode_node, monkeypatch):
    _free(monkeypatch, 6.5)
    vae = _GpuLikeWanVAE()
    lat_2k = torch.zeros(1, 16, 5, 135, 256)          # 2048x1080, 17 frames: ~31 GB untiled by core's estimate
    assert decode_node._auto_tile_px(vae, lat_2k, False) == 512     # what the 8 GB measurement found best
    small = torch.zeros(1, 16, 5, 30, 40)              # 320x240: fits -> untiled
    assert decode_node._auto_tile_px(vae, small, False) == 0
    assert decode_node._auto_tile_px(LegacyFakeVAE(), lat_2k, False) == 0   # no GPU device: untiled reference


def test_off_and_manual_and_auto_route_the_decode(decode_node, monkeypatch):
    seen = []
    monkeypatch.setattr(decode_node, "_run_decode", lambda vae, latent, tile: seen.append(tile) or torch.zeros(1, 4, 8, 8, 3))
    monkeypatch.setattr(decode_node, "_auto_tile_px", lambda vae, latent, fp32: 512)
    lat = torch.zeros(1, 4, 2, 4, 4)
    for mode, size, want in (("off", 256, 0), ("manual", 256, 256), ("manual", 0, 0), ("auto", 0, 512), ("auto", 256, 512)):
        decode_node.decode({"samples": lat}, LegacyFakeVAE(), False, size, False, 1.0, True, tile_mode=mode)
        assert seen[-1] == want, (mode, size)


def test_image_latents_tile_through_core(decode_node):
    calls = {}

    class ImgVAE:
        def spacial_compression_decode(self):
            return 8

        def decode_tiled(self, latent, tile_x, tile_y, overlap):
            calls.update(tile_x=tile_x, tile_y=tile_y, overlap=overlap)
            return torch.zeros(1, 64, 64, 3)

    out = decode_node._run_decode(ImgVAE(), torch.zeros(1, 4, 8, 8), 512)
    assert calls == {"tile_x": 64, "tile_y": 64, "overlap": 16} and out.shape == (1, 64, 64, 3)


def test_fp32_copy_leaves_the_shared_vae_alone(monkeypatch):
    # L7.59: casting the shared VAE in place broke core's pinned weights on a GPU; fp32 now uses a separate copy
    import nodes._vae_tiled as VT

    built = []

    class FakeCoreVAE:
        def __init__(self, sd=None, dtype=None, **_k):
            self.sd, self.dtype = sd, dtype
            built.append(self)

        def get_sd(self):
            return {"w": torch.ones(3, dtype=torch.bfloat16)}

    fake_sd = types.ModuleType("comfy.sd")
    fake_sd.VAE = FakeCoreVAE
    monkeypatch.setitem(sys.modules, "comfy.sd", fake_sd)
    monkeypatch.setattr(VT, "_FP32_COPY", {})
    shared = FakeCoreVAE.__new__(FakeCoreVAE)          # the user's VAE (not built through __init__)
    copy = VT.fp32_vae_copy(shared)
    assert copy is not shared and copy.dtype == torch.float32 and copy.sd["w"].dtype == torch.float32
    assert shared.get_sd()["w"].dtype == torch.bfloat16
    assert VT.fp32_vae_copy(shared) is copy and len(built) == 1          # cached
    assert VT.fp32_vae_copy(LegacyFakeVAE()) is None                    # not a core VAE: old path
