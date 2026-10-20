"""L2.54: Universal mode's operation adapters (nodes/_autobatch_adapters.py).

The reference below is core Grow Mask's algorithm (comfy_extras/nodes_mask.py, GrowMask.execute): per frame, per
iteration scipy.ndimage.grey_dilation / grey_erosion with a 3x3 footprint (cross when tapered). Written here from its
documented behaviour so the test does not need ComfyUI."""
from __future__ import annotations

import numpy as np
import pytest
import scipy.ndimage
import torch

from nodes import _autobatch_adapters as AD


class _Out:
    def __init__(self, *args):
        self.args = args

    @property
    def result(self):
        return self.args


def core_grow(mask, expand, tapered_corners):
    c = 0 if tapered_corners else 1
    kernel = np.array([[c, 1, c], [1, 1, 1], [c, 1, c]])
    mask = mask.reshape((-1, mask.shape[-2], mask.shape[-1]))
    out = []
    for m in mask:
        o = m.numpy()
        for _ in range(abs(expand)):
            o = scipy.ndimage.grey_erosion(o, footprint=kernel) if expand < 0 else scipy.ndimage.grey_dilation(o, footprint=kernel)
        out.append(torch.from_numpy(o))
    return _Out(torch.stack(out, dim=0))


@pytest.fixture(autouse=True)
def _fresh(tmp_path, monkeypatch):
    monkeypatch.setenv("C2C_AUTOBATCH_PROFILES", str(tmp_path / "profiles.json"))
    monkeypatch.delenv("C2C_AUTOBATCH_ADAPTERS", raising=False)
    monkeypatch.setattr(AD, "_PROFILES", None)
    AD.DECISIONS.clear()
    for a in AD.ADAPTERS.values():
        a.state.clear()
    yield
    for a in AD.ADAPTERS.values():
        a.state.clear()


def _mask(n=6, h=40, w=56, seed=0):
    g = torch.Generator().manual_seed(seed)
    m = (torch.rand(n, h, w, generator=g) > 0.9).float() * torch.rand(n, h, w, generator=g)
    m[:, 0, :] = torch.rand(n, w, generator=g)
    m[:, :, -1] = torch.rand(n, h, generator=g)
    return m


@pytest.mark.parametrize("expand", [0, 1, 4, -1, -3])
@pytest.mark.parametrize("tapered", [True, False])
def test_batched_grow_is_bit_identical_to_core(expand, tapered):
    m = _mask()
    ref = core_grow(m, expand, tapered).result[0]
    got = AD.grow_mask_batched(m, expand, tapered)
    assert got.shape == ref.shape and torch.equal(got, ref)


def test_inputs_are_never_mutated():
    m = _mask()
    keep = m.clone()
    AD.grow_mask_batched(m, 3, True)
    AD.grow_mask_batched(m, -2, False)
    assert torch.equal(m, keep)


def test_validation_passes_against_the_original_and_catches_a_wrong_adapter(monkeypatch):
    a = AD.ADAPTERS["GrowMask"]
    assert AD.validate(a, core_grow) == (True, "self-test passed")
    a.state.clear()
    monkeypatch.setattr(a, "run", lambda args: AD.grow_mask_batched(args["mask"], args["expand"] + 1, args["tapered_corners"]))
    ok, reason = AD.validate(a, core_grow)
    assert not ok and "differs" in reason


def test_measured_choice_is_persisted_and_reused(tmp_path):
    calls = {"n": 0}

    def original(**kw):
        calls["n"] += 1
        return core_grow(**kw)

    args = {"mask": _mask(n=12, h=64, w=96), "expand": 3, "tapered_corners": True}
    used, res = AD.try_adapter("GrowMask", original, args)
    first_calls = calls["n"]
    prof = AD.status()["profiles"]
    assert len(prof) == 1
    (key, p), = prof.items()
    assert p["choice"] in ("adapter", "original") and p["measured_frames"] == 4
    assert (tmp_path / "profiles.json").is_file()
    if used:
        assert torch.equal(res[0], core_grow(**args).result[0])
    # second call: no new measurement (only the self-test cache and the stored profile)
    calls["n"] = 0
    AD.try_adapter("GrowMask", original, args)
    assert calls["n"] == (0 if p["choice"] == "adapter" else 0)
    assert first_calls >= 1


def test_profiles_survive_a_restart(tmp_path, monkeypatch):
    args = {"mask": _mask(n=8), "expand": 2, "tapered_corners": False}
    AD.try_adapter("GrowMask", core_grow, args)
    monkeypatch.setattr(AD, "_PROFILES", None)          # a new process
    assert len(AD.status()["profiles"]) == 1


def test_kill_switch_and_unsupported_inputs(monkeypatch):
    args = {"mask": _mask(), "expand": 2, "tapered_corners": True}
    monkeypatch.setenv("C2C_AUTOBATCH_ADAPTERS", "0")
    assert AD.try_adapter("GrowMask", core_grow, args) == (False, None)
    monkeypatch.delenv("C2C_AUTOBATCH_ADAPTERS")
    assert AD.try_adapter("GrowMask", core_grow, {**args, "expand": 2.0})[0] is False      # not an int
    assert AD.try_adapter("NotAdapted", core_grow, args) == (False, None)


def test_adapter_error_falls_back_to_the_original(monkeypatch):
    a = AD.ADAPTERS["GrowMask"]
    args = {"mask": _mask(n=16, h=96, w=128), "expand": 6, "tapered_corners": False}
    AD.validate(a, core_grow)
    key = AD.profile_key(a, a.state["fp"], args["mask"])
    AD._profiles()[key] = {"choice": "adapter", "original_s_per_frame": 1.0, "adapter_s_per_frame": 0.1, "measured_frames": 4}

    def boom(_args):
        raise RuntimeError("simulated")

    real_run = a.run
    monkeypatch.setattr(a, "run", boom)
    try:
        assert AD.try_adapter("GrowMask", core_grow, args) == (False, None)
    finally:
        monkeypatch.setattr(a, "run", real_run)
    assert "fell back" in AD.DECISIONS[-1]["reason"]


def test_a_changed_original_is_revalidated():
    a = AD.ADAPTERS["GrowMask"]
    AD.validate(a, core_grow)
    fp1 = a.state["fp"]

    def core_grow_v2(mask, expand, tapered_corners):   # a different source text = a core update
        return core_grow(mask, expand, tapered_corners)

    AD.validate(a, core_grow_v2)
    assert a.state["fp"] != fp1 and a.state["ok"] is True


# ── Feather Mask (slice 3) ─────────────────────────────────────────────────────────────────────────────────────────

def core_feather(mask, left, top, right, bottom):
    """Core Feather Mask's algorithm (comfy_extras/nodes_mask.py FeatherMask.execute, core v0.39), line for line."""
    output = mask.reshape((-1, mask.shape[-2], mask.shape[-1])).clone()
    left = min(left, output.shape[-1])
    right = min(right, output.shape[-1])
    top = min(top, output.shape[-2])
    bottom = min(bottom, output.shape[-2])
    for x in range(left):
        output[:, :, x] *= (x + 1.0) / left
    for x in range(right):
        output[:, :, -(x + 1)] *= (x + 1) / right
    for y in range(top):
        output[:, y, :] *= (y + 1) / top
    for y in range(bottom):
        output[:, -(y + 1), :] *= (y + 1) / bottom
    return _Out(output)


@pytest.mark.parametrize("sides", [(0, 0, 0, 0), (1, 0, 0, 0), (0, 0, 0, 1), (3, 7, 5, 2), (30, 30, 30, 30),
                                   (56, 40, 56, 40), (9999, 9999, 9999, 9999), (20, 0, 50, 0)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_strip_feather_is_bit_identical_to_core(sides, dtype):
    m = torch.rand(5, 40, 56, generator=torch.Generator().manual_seed(4)).to(dtype)
    ref = core_feather(m, *sides).result[0]
    got = AD.feather_mask_strips(m, *sides)
    assert got.dtype == ref.dtype and torch.equal(got, ref)


def test_strip_feather_2d_mask_and_no_mutation():
    m = torch.rand(33, 47)
    keep = m.clone()
    assert torch.equal(AD.feather_mask_strips(m, 4, 5, 6, 7), core_feather(m, 4, 5, 6, 7).result[0])
    assert torch.equal(m, keep)


def test_feather_adapter_validates_and_refuses_what_it_cannot_reproduce():
    a = AD.ADAPTERS["FeatherMask"]
    assert AD.validate(a, core_feather) == (True, "self-test passed")
    base = {"mask": torch.rand(4, 8, 8), "left": 2, "top": 2, "right": 2, "bottom": 2}
    assert a.supports(base)
    assert not a.supports({**base, "mask": base["mask"].half()})          # core rounds once in float, a half ramp twice
    assert not a.supports({**base, "left": 2.0})
    assert not a.supports({**base, "left": True})
    assert not a.supports({**base, "top": -1})


def test_feather_through_try_adapter_gives_core_s_result():
    args = {"mask": torch.rand(12, 64, 96), "left": 16, "top": 8, "right": 16, "bottom": 8}
    used, res = AD.try_adapter("FeatherMask", core_feather, args)
    prof = next(iter(AD.status()["profiles"].values()))
    assert prof["measured_frames"] == 4 and prof["choice"] in ("adapter", "original")
    if used:
        assert torch.equal(res[0], core_feather(**args).result[0])
