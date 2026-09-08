"""CustomNodePacks-owned nodes only.

The color_science / render_pass / plate_tools / geometry_nodes / metadata_nodes
tests MOVED to ComfyUI-NukeMaxNodes/tests/test_migrated_vfx_nodes.py together
with their code (2026-08-29 dedup). What stays here is what this pack still
owns: exr_io (LoadEXRMEC / SaveEXRMEC, the OpenImageIO path) and model_analysis.
"""
from __future__ import annotations

import json
import os
import struct
from pathlib import Path

import numpy as np
import pytest
import torch

from nodes.exr_io import LoadEXRMEC, SaveEXRMEC
from nodes.model_analysis import VAEBlockInspectorMEC, VAESimilarityAnalyserMEC


# ──────────────────────────────────────────────────────────────────────
#  Fixtures
# ──────────────────────────────────────────────────────────────────────

@pytest.fixture
def small_image():
    torch.manual_seed(0)
    return torch.rand(1, 16, 24, 3)


@pytest.fixture
def video_batch():
    torch.manual_seed(1)
    return torch.rand(4, 16, 16, 3)


# ──────────────────────────────────────────────────────────────────────
#  Color science
# ──────────────────────────────────────────────────────────────────────

class TestEXRIO:
    def test_save_load_round_trip(self, tmp_path, small_image):
        out = tmp_path / "test.exr"
        save = SaveEXRMEC()
        info_json, = save.save(small_image, str(out), half_float=False)
        # If backend wrote a *_fallback.tif, accept that the file may differ.
        info = json.loads(info_json)
        if info["frames"][0].get("backend") == "tiff_fallback":
            pytest.skip("EXR write fell back to TIFF; round-trip not meaningful")
        assert out.exists()
        load = LoadEXRMEC()
        loaded, _ = load.load(str(out))
        assert loaded.shape == small_image.shape
        assert torch.allclose(loaded, small_image, atol=5e-3)

    def test_load_missing(self):
        with pytest.raises(FileNotFoundError):
            LoadEXRMEC().load("/nope/missing.exr")


# ──────────────────────────────────────────────────────────────────────
#  Render passes
# ──────────────────────────────────────────────────────────────────────

class _FakeVAE:
    def __init__(self, sd):
        self._sd = sd
    def state_dict(self):
        return self._sd


class TestVAESimilarity:
    def test_identical_models_cosine_one(self):
        sd = {
            "encoder.conv_in.weight": torch.randn(4, 3, 3, 3),
            "decoder.conv_out.weight": torch.randn(3, 4, 3, 3),
        }
        v = _FakeVAE(sd)
        report, cos, divergent = VAESimilarityAnalyserMEC().analyse(v, v)
        assert abs(cos - 1.0) < 1e-5
        d = json.loads(report)
        assert d["common_tensors"] == 2

    def test_orthogonal_low_cosine(self):
        a = torch.randn(8)
        sd_a = {"x.weight": a}
        sd_b = {"x.weight": -a}
        report, cos, divergent = VAESimilarityAnalyserMEC().analyse(_FakeVAE(sd_a), _FakeVAE(sd_b))
        assert cos < -0.99

    def test_only_in_a(self):
        sd_a = {"a.weight": torch.zeros(3), "b.weight": torch.zeros(3)}
        sd_b = {"a.weight": torch.zeros(3)}
        report, _, divergent = VAESimilarityAnalyserMEC().analyse(_FakeVAE(sd_a), _FakeVAE(sd_b))
        d = json.loads(report)
        assert d["only_in_a"] == ["b.weight"]


class TestVAEBlockInspector:
    def test_stats_basic(self):
        sd = {"encoder.conv_in.weight": torch.full((4, 3, 3, 3), 0.5)}
        report, outliers, anomaly = VAEBlockInspectorMEC().inspect(_FakeVAE(sd))
        d = json.loads(report)
        assert "blocks" in d
        # Some block key should report the constant 0.5
        means = [b["mean"] for b in d["blocks"].values()]
        assert any(abs(m - 0.5) < 1e-6 for m in means)


def test_analyser_and_inspector_extra_outputs_are_populated():
    """Assert the outputs these tests previously did not unpack at all.

    VAESimilarityAnalyserMEC returns (report_json, global_cosine,
    most_divergent_blocks) and VAEBlockInspectorMEC returns (report_json,
    outlier_tensor_names, anomaly_score) — nodes/model_analysis.py:69-70,159-160.
    The suite unpacked 2 and 1 respectively, so the trailing outputs shipped with
    no coverage.
    """
    import json as _json

    sd = {"decoder.conv_out.weight": torch.ones(4, 4)}
    report, cos, divergent = VAESimilarityAnalyserMEC().analyse(_FakeVAE(sd), _FakeVAE(sd))
    assert isinstance(_json.loads(report), dict)
    assert isinstance(cos, float)
    assert isinstance(divergent, str), "most_divergent_blocks must be a STRING output"

    report2, outliers, anomaly = VAEBlockInspectorMEC().inspect(_FakeVAE(sd))
    assert isinstance(_json.loads(report2), dict)
    assert isinstance(outliers, str), "outlier_tensor_names must be a STRING output"
    assert isinstance(anomaly, float), "anomaly_score must be a FLOAT output"
