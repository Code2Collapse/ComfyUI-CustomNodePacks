"""VAE merge tests.

VAEMergeMEC returns FOUR outputs — RETURN_NAMES = (vae, info, recipe_json,
probe_report) at nodes/vae_merge.py:81-82, and every return statement in
merge() yields four. These tests unpacked two, so all nine failed with
"too many values to unpack". The product grew outputs deliberately and is
internally consistent; the tests were stale. Arity updated, and the two new
outputs are now asserted rather than discarded.
"""
import json
import sys
import types
from copy import deepcopy

import pytest
import torch

# Stub ComfyUI modules so the node module imports cleanly.
for mod_name in ("folder_paths", "comfy", "comfy.utils"):
    if mod_name not in sys.modules:
        stub = types.ModuleType(mod_name)
        stub.__path__ = []
        sys.modules[mod_name] = stub

from nodes.vae_merge import (  # noqa: E402
    VAEMergeMEC,
    _classify_key,
    _detect_architecture,
    _extract_state_dict,
    _slerp,
)


# --------------------------------------------------------------------- #
#  Dummy VAE wrapper                                                    #
# --------------------------------------------------------------------- #

class _DummyInner(torch.nn.Module):
    """Minimal stand-in for ComfyUI's first_stage_model — exposes
    state_dict/load_state_dict over a fixed set of named buffers that
    look like SD-1.5 VAE keys."""

    def __init__(self, seed: int = 0, dtype=torch.float32):
        super().__init__()
        torch.manual_seed(seed)
        # SD-1.5-style key names (subset, enough to drive block detection)
        spec = {
            "encoder.conv_in.weight":       (4, 3, 3, 3),
            "encoder.conv_in.bias":         (4,),
            "encoder.down.0.block.0.norm1.weight": (4,),
            "encoder.down.1.block.0.conv1.weight": (4, 4, 3, 3),
            "encoder.down.2.block.0.conv1.weight": (4, 4, 3, 3),
            "encoder.mid.block_1.norm1.weight":    (4,),
            "encoder.norm_out.weight":      (4,),
            "encoder.conv_out.weight":      (4, 4, 1, 1),
            "decoder.conv_in.weight":       (4, 4, 3, 3),
            "decoder.up.0.block.0.norm1.weight":   (4,),
            "decoder.up.1.block.0.conv1.weight":   (4, 4, 3, 3),
            "decoder.mid.block_1.norm1.weight":    (4,),
            "decoder.norm_out.weight":      (4,),
            "decoder.conv_out.weight":      (3, 4, 3, 3),
            "decoder.conv_out.bias":        (3,),
        }
        for k, shape in spec.items():
            self.register_buffer(k.replace(".", "__"), torch.randn(*shape, dtype=dtype))
        self._spec = spec

    def state_dict(self, *args, **kwargs):
        # Map dotted-name keys (what ComfyUI / merge code expects) back
        # from our buffer-safe underscore names.
        return {k: getattr(self, k.replace(".", "__")).clone() for k in self._spec}

    def load_state_dict(self, sd, strict=False):
        missing, unexpected = [], []
        for k, v in sd.items():
            attr = k.replace(".", "__")
            if hasattr(self, attr):
                getattr(self, attr).copy_(v)
            else:
                unexpected.append(k)
        for k in self._spec:
            if k not in sd:
                missing.append(k)
        return type("R", (), {"missing_keys": missing, "unexpected_keys": unexpected})()


class _DummyVAE:
    def __init__(self, seed: int = 0, dtype=torch.float32):
        self.first_stage_model = _DummyInner(seed=seed, dtype=dtype)


def _make_vae(seed=0, dtype=torch.float32):
    return _DummyVAE(seed=seed, dtype=dtype)


# --------------------------------------------------------------------- #
#  Helpers                                                               #
# --------------------------------------------------------------------- #

def _close(a: torch.Tensor, b: torch.Tensor, tol=1e-5) -> bool:
    return torch.allclose(a, b, atol=tol, rtol=tol)


# --------------------------------------------------------------------- #
#  Tests                                                                 #
# --------------------------------------------------------------------- #

def test_classify_key_basic():
    assert _classify_key("encoder.conv_in.weight") == "block_conv_in"
    assert _classify_key("decoder.conv_out.bias") == "block_conv_out"
    assert _classify_key("encoder.norm_out.weight") == "block_norm_out"
    assert _classify_key("encoder.down.0.block.0.norm1.weight") == "block_0"
    assert _classify_key("decoder.up.3.block.0.norm1.weight") == "block_0"
    assert _classify_key("encoder.mid.block_1.norm1.weight") == "block_mid"
    assert _classify_key("totally.unrelated.key") is None


def test_detect_architecture_sd1x():
    keys = ["encoder.down.2.block.0.conv1.weight", "decoder.up.0.foo.bar"]
    assert _detect_architecture(keys) in {"sd1x", "sdxl"}


def test_extract_state_dict_from_wrapper():
    vae = _make_vae()
    sd, inner = _extract_state_dict(vae)
    assert "encoder.conv_in.weight" in sd
    assert inner is vae.first_stage_model


def test_extract_state_dict_from_dict():
    sd_in = {"k": torch.zeros(2)}
    sd, inner = _extract_state_dict(sd_in)
    assert sd == {"k": sd_in["k"]}
    assert inner is None


def test_weighted_sum_alpha_zero_returns_a():
    a = _make_vae(seed=1)
    b = _make_vae(seed=2)
    sd_a_before = a.first_stage_model.state_dict()
    node = VAEMergeMEC()
    out, info_str, recipe_json, probe_report = node.merge(vae_a=a, vae_b=b, merge_mode="weighted_sum", alpha=0.0)
    info = json.loads(info_str)
    assert info["status"] == "ok"
    sd_out = out.first_stage_model.state_dict()
    for k in sd_out:
        assert _close(sd_out[k], sd_a_before[k])


def test_weighted_sum_alpha_one_returns_b():
    a = _make_vae(seed=1)
    b = _make_vae(seed=2)
    sd_b = b.first_stage_model.state_dict()
    node = VAEMergeMEC()
    out, _, _recipe, _probe = node.merge(vae_a=a, vae_b=b, merge_mode="weighted_sum", alpha=1.0)
    sd_out = out.first_stage_model.state_dict()
    for k in sd_out:
        assert _close(sd_out[k], sd_b[k])


def test_weighted_sum_midpoint_average():
    a = _make_vae(seed=1)
    b = _make_vae(seed=2)
    sd_a = a.first_stage_model.state_dict()
    sd_b = b.first_stage_model.state_dict()
    node = VAEMergeMEC()
    out, _, _recipe, _probe = node.merge(vae_a=a, vae_b=b, merge_mode="weighted_sum", alpha=0.5)
    sd_out = out.first_stage_model.state_dict()
    for k in sd_out:
        assert _close(sd_out[k], 0.5 * sd_a[k] + 0.5 * sd_b[k], tol=1e-4)


def test_add_difference_requires_vae_c():
    a = _make_vae(seed=1)
    b = _make_vae(seed=2)
    node = VAEMergeMEC()
    with pytest.raises(ValueError, match=r"vae_c"):
        node.merge(vae_a=a, vae_b=b, merge_mode="add_difference", alpha=0.5)


def test_triple_sum_requires_vae_c():
    node = VAEMergeMEC()
    with pytest.raises(ValueError, match=r"vae_c"):
        node.merge(vae_a=_make_vae(1), vae_b=_make_vae(2),
                   merge_mode="triple_sum", alpha=0.5)


def test_triple_sum_average():
    a, b, c = _make_vae(1), _make_vae(2), _make_vae(3)
    sd_a = a.first_stage_model.state_dict()
    sd_b = b.first_stage_model.state_dict()
    sd_c = c.first_stage_model.state_dict()
    node = VAEMergeMEC()
    out, _, _recipe, _probe = node.merge(vae_a=a, vae_b=b, vae_c=c,
                        merge_mode="triple_sum", alpha=0.5, beta=0.5)
    sd_out = out.first_stage_model.state_dict()
    for k in sd_out:
        assert _close(sd_out[k], (sd_a[k] + sd_b[k] + sd_c[k]) / 3.0, tol=1e-4)


def test_block_weights_override_alpha():
    """When use_blocks=True, block_conv_in slider controls keys matching
    encoder.conv_in.* regardless of the global alpha."""
    a = _make_vae(seed=1)
    b = _make_vae(seed=2)
    sd_a = a.first_stage_model.state_dict()
    sd_b = b.first_stage_model.state_dict()
    node = VAEMergeMEC()
    out, _, _recipe, _probe = node.merge(
        vae_a=a, vae_b=b,
        merge_mode="weighted_sum",
        alpha=0.0,            # default: keep A
        use_blocks=True,
        block_conv_in=1.0,    # but override conv_in to keep B
    )
    sd_out = out.first_stage_model.state_dict()
    # conv_in keys should equal B (block override = 1.0)
    assert _close(sd_out["encoder.conv_in.weight"], sd_b["encoder.conv_in.weight"])
    # norm_out keys (block_norm_out default 0.5) → midpoint
    assert _close(sd_out["decoder.norm_out.weight"],
                  0.5 * sd_a["decoder.norm_out.weight"] + 0.5 * sd_b["decoder.norm_out.weight"],
                  tol=1e-4)


def test_architecture_mismatch_raises():
    """Two VAEs with disjoint keys must raise a descriptive ValueError."""
    a = _make_vae()
    sd_a = a.first_stage_model.state_dict()
    bad = {"some.totally.different.key": torch.zeros(4)}
    node = VAEMergeMEC()
    with pytest.raises(ValueError, match=r"Architecture mismatch"):
        node.merge(vae_a=sd_a, vae_b=bad, merge_mode="weighted_sum", alpha=0.5)


def test_info_json_well_formed():
    node = VAEMergeMEC()
    _, info_str, recipe_json, probe_report = node.merge(vae_a=_make_vae(1), vae_b=_make_vae(2),
                             merge_mode="weighted_sum", alpha=0.5)
    info = json.loads(info_str)
    assert info["merge_mode"] == "weighted_sum"
    assert info["alpha"] == 0.5
    assert info["status"] == "ok"
    assert info["key_count"] > 0
    assert "timestamp" in info
    assert "dtypes" in info


def test_dtype_preserved():
    a = _make_vae(seed=1, dtype=torch.float16)
    b = _make_vae(seed=2, dtype=torch.float16)
    node = VAEMergeMEC()
    out, _, _recipe, _probe = node.merge(vae_a=a, vae_b=b, merge_mode="weighted_sum", alpha=0.5)
    for v in out.first_stage_model.state_dict().values():
        assert v.dtype == torch.float16


def test_brightness_contrast_only_touch_decoder_conv_out():
    a = _make_vae(seed=1)
    b = _make_vae(seed=1)  # identical → merged should equal a
    sd_a = a.first_stage_model.state_dict()
    node = VAEMergeMEC()
    out, _, _recipe, _probe = node.merge(vae_a=a, vae_b=b, merge_mode="weighted_sum",
                        alpha=0.5, brightness=0.1, contrast=0.1)
    sd_out = out.first_stage_model.state_dict()
    # decoder.conv_out keys should differ
    assert not _close(sd_out["decoder.conv_out.weight"], sd_a["decoder.conv_out.weight"])
    # but encoder.norm_out should be unchanged
    assert _close(sd_out["encoder.norm_out.weight"], sd_a["encoder.norm_out.weight"])


def test_slerp_endpoints():
    a = torch.randn(4, 4)
    b = torch.randn(4, 4)
    assert torch.allclose(_slerp(a, b, 0.0), a, atol=1e-5)
    assert torch.allclose(_slerp(a, b, 1.0), b, atol=1e-5)


def test_invalid_input_returns_passthrough_with_error_info():
    """Non-VAE-shaped input should not crash; returns vae_a + error info."""
    node = VAEMergeMEC()
    bad = object()
    # _extract_state_dict raises TypeError -> caught in except Exception branch
    out, info_str, recipe_json, probe_report = node.merge(vae_a=bad, vae_b=bad,
                               merge_mode="weighted_sum", alpha=0.5)
    info = json.loads(info_str)
    assert info["status"] == "error"
    assert "error" in info
    assert out is bad


def test_recipe_json_and_probe_report_are_populated():
    """The two outputs added since these tests were written must carry content.

    Added during the 2026-08 triage: all nine tests here failed on arity because
    VAEMergeMEC grew `recipe_json` and `probe_report`. Fixing the unpacking alone
    would have left both outputs completely untested, so they are asserted here.
    """
    import json as _json

    node = VAEMergeMEC()
    a, b = _make_vae(1), _make_vae(2)
    _out, info_str, recipe_json, probe_report = node.merge(
        vae_a=a, vae_b=b, merge_mode="weighted_sum", alpha=0.5
    )

    assert isinstance(recipe_json, str) and recipe_json.strip(), "recipe_json is empty"
    recipe = _json.loads(recipe_json)  # must be valid JSON, not a bare string
    assert isinstance(recipe, dict) and recipe, "recipe_json decoded to nothing useful"

    assert isinstance(probe_report, str), "probe_report must be a STRING output"
    assert isinstance(_json.loads(info_str), dict), "info must remain valid JSON"
