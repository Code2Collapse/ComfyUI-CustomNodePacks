"""Tests for Phase-3 VFX nodes."""
from __future__ import annotations

import json
import os
import struct
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pytest
import torch

from nodes.batch_version_manager import BatchVersionManagerMEC, _safe, _scan_max_version
from nodes.model_metadata_extractor import (
    ModelMetadataExtractorMEC,
    _read_safetensors_header,
    _detect_model_kind,
)
from nodes.temporal_consistency_checker import TemporalConsistencyCheckerMEC, _mask_iou
from nodes.vae_latent_inspector import VAELatentInspectorMEC, _verdict


# ══════════════════════════════════════════════════════════════════════
#  VAELatentInspectorMEC
# ══════════════════════════════════════════════════════════════════════

class TestVAELatentInspector:
    def test_healthy_latent(self):
        latent = {"samples": torch.randn(1, 4, 8, 8)}
        out = VAELatentInspectorMEC().inspect(latent)
        assert out[2] == "healthy"
        assert out[3] == 0  # nan_count
        assert out[4] == 0  # inf_count
        info = json.loads(out[1])
        assert info["shape"] == [1, 4, 8, 8]
        assert len(info["channels"]) == 4

    def test_nan_detection(self):
        t = torch.randn(1, 4, 4, 4)
        t[0, 0, 0, 0] = float("nan")
        latent = {"samples": t}
        out = VAELatentInspectorMEC().inspect(latent)
        assert out[2] == "corrupt"
        assert out[3] == 1

    def test_inf_detection(self):
        t = torch.randn(1, 4, 4, 4)
        t[0, 1, 2, 3] = float("inf")
        latent = {"samples": t}
        out = VAELatentInspectorMEC().inspect(latent)
        assert out[2] == "corrupt"
        assert out[4] == 1

    def test_low_contrast(self):
        latent = {"samples": torch.full((1, 4, 4, 4), 0.5)}
        out = VAELatentInspectorMEC().inspect(latent)
        assert out[2] == "low_contrast"

    def test_saturated(self):
        latent = {"samples": torch.full((1, 4, 4, 4), 100.0)}
        out = VAELatentInspectorMEC().inspect(latent)
        assert out[2] == "saturated"

    def test_passthrough_identity(self):
        t = torch.randn(1, 4, 8, 8)
        latent = {"samples": t}
        out = VAELatentInspectorMEC().inspect(latent)
        assert out[0] is latent

    def test_fail_on_corrupt_raises(self):
        t = torch.full((1, 4, 4, 4), float("nan"))
        with pytest.raises(ValueError):
            VAELatentInspectorMEC().inspect({"samples": t}, fail_on_corrupt=True)

    def test_invalid_input(self):
        with pytest.raises(ValueError):
            VAELatentInspectorMEC().inspect({})

    def test_verdict_helper(self):
        assert _verdict(0.0, 1.0, 0.3, 0, 0) == "healthy"
        assert _verdict(0.0, 0.0, 0.0, 0, 0) == "low_contrast"
        assert _verdict(-100.0, 100.0, 50.0, 0, 0) == "saturated"
        assert _verdict(0.0, 1.0, 0.3, 1, 0) == "corrupt"


# ══════════════════════════════════════════════════════════════════════
#  EXRMetadataReaderMEC
# ══════════════════════════════════════════════════════════════════════

def _make_minimal_exr(path: Path, width: int = 4, height: int = 3) -> None:
    """Write a minimal but valid OpenEXR header (no scanlines/data, header only).

    The file is intentionally truncated (no pixel offsets, no data) — it
    only exercises the header parser. Magic + version + attrs + null-name
    terminator. The pure-Python reader doesn't decode pixels so this works.
    """
    with open(path, "wb") as fh:
        fh.write(struct.pack("<i", 20000630))   # magic
        fh.write(struct.pack("<i", 2))          # version=2, no flags
        # channels attr (chlist) — single 'R' float channel
        fh.write(b"channels\x00")
        fh.write(b"chlist\x00")
        chlist = b"R\x00" + struct.pack("<i", 2) + b"\x00\x00\x00\x00" + struct.pack("<i", 1) + struct.pack("<i", 1) + b"\x00"
        fh.write(struct.pack("<i", len(chlist)))
        fh.write(chlist)
        # compression = ZIP (3)
        fh.write(b"compression\x00")
        fh.write(b"compression\x00")
        fh.write(struct.pack("<i", 1))
        fh.write(b"\x03")
        # dataWindow box2i = (0, 0, w-1, h-1)
        fh.write(b"dataWindow\x00")
        fh.write(b"box2i\x00")
        fh.write(struct.pack("<i", 16))
        fh.write(struct.pack("<4i", 0, 0, width - 1, height - 1))
        # displayWindow box2i
        fh.write(b"displayWindow\x00")
        fh.write(b"box2i\x00")
        fh.write(struct.pack("<i", 16))
        fh.write(struct.pack("<4i", 0, 0, width - 1, height - 1))
        # lineOrder
        fh.write(b"lineOrder\x00")
        fh.write(b"lineOrder\x00")
        fh.write(struct.pack("<i", 1))
        fh.write(b"\x00")
        # pixelAspectRatio
        fh.write(b"pixelAspectRatio\x00")
        fh.write(b"float\x00")
        fh.write(struct.pack("<i", 4))
        fh.write(struct.pack("<f", 1.0))
        # screenWindowCenter v2f
        fh.write(b"screenWindowCenter\x00")
        fh.write(b"v2f\x00")
        fh.write(struct.pack("<i", 8))
        fh.write(struct.pack("<2f", 0.0, 0.0))
        # screenWindowWidth float
        fh.write(b"screenWindowWidth\x00")
        fh.write(b"float\x00")
        fh.write(struct.pack("<i", 4))
        fh.write(struct.pack("<f", 1.0))
        # End-of-header marker (empty attribute name)
        fh.write(b"\x00")


class TestBatchVersionManager:
    def test_safe_token(self):
        assert _safe("hello world", "x") == "hello_world"
        assert _safe("../../etc/passwd", "x") == "etc_passwd"
        assert _safe("", "x") == "x"
        assert _safe("ok-name.v1", "x") == "ok-name.v1"

    def test_no_disk_writes_when_reserve_false(self, tmp_path: Path):
        out = BatchVersionManagerMEC().allocate(
            root=str(tmp_path), show="s", shot="sh010", task="comp", reserve=False,
        )
        path = Path(out[0])
        assert out[1] == 1
        assert out[2] == "v001"
        assert not path.exists()

    def test_reserve_creates_lock(self, tmp_path: Path):
        out = BatchVersionManagerMEC().allocate(
            root=str(tmp_path), show="s", shot="sh010", task="comp", reserve=True,
        )
        path = Path(out[0])
        assert path.is_dir()
        assert (path / ".lock").is_file()
        info = json.loads((path / ".lock").read_text())
        assert info["version"] == 1

    def test_increment_after_existing(self, tmp_path: Path):
        BatchVersionManagerMEC().allocate(
            root=str(tmp_path), show="s", shot="sh", task="t", reserve=True,
        )
        out = BatchVersionManagerMEC().allocate(
            root=str(tmp_path), show="s", shot="sh", task="t", reserve=True,
        )
        assert out[1] == 2
        assert out[2] == "v002"

    def test_scan_max_version(self, tmp_path: Path):
        d = tmp_path / "task"
        d.mkdir()
        (d / "v001").mkdir()
        (d / "v003").mkdir()
        (d / "not_a_version").mkdir()
        assert _scan_max_version(d) == 3

    def test_padding(self, tmp_path: Path):
        out = BatchVersionManagerMEC().allocate(
            root=str(tmp_path), show="s", shot="sh", task="t",
            reserve=False, padding=4,
        )
        assert out[2] == "v0001"

    def test_path_uses_forward_slash(self, tmp_path: Path):
        out = BatchVersionManagerMEC().allocate(
            root=str(tmp_path), show="s", shot="sh", task="t", reserve=False,
        )
        assert "\\" not in out[0]

    def test_min_version_floor(self, tmp_path: Path):
        out = BatchVersionManagerMEC().allocate(
            root=str(tmp_path), show="s", shot="sh", task="t",
            reserve=False, min_version=10,
        )
        assert out[1] == 10


# ══════════════════════════════════════════════════════════════════════
#  TemporalConsistencyCheckerMEC
# ══════════════════════════════════════════════════════════════════════

class TestTemporalConsistencyChecker:
    def test_pixel_diff_identical_frames(self):
        img = torch.ones(3, 4, 4, 3) * 0.5
        out = TemporalConsistencyCheckerMEC().check("pixel_diff", image=img)
        assert out[2] == pytest.approx(0.0)

    def test_pixel_diff_changing_frames(self):
        img = torch.zeros(3, 4, 4, 3)
        img[1] = 1.0
        out = TemporalConsistencyCheckerMEC().check("pixel_diff", image=img)
        assert out[2] > 0.5

    def test_mask_iou_identical(self):
        m = torch.zeros(3, 4, 4)
        m[:, 1:3, 1:3] = 1.0
        out = TemporalConsistencyCheckerMEC().check("mask_iou", mask=m)
        # All pairs identical → instability score 0
        assert out[2] == pytest.approx(0.0)

    def test_mask_iou_helper(self):
        a = torch.zeros(4, 4)
        a[1:3, 1:3] = 1.0
        b = torch.zeros(4, 4)
        b[1:3, 1:3] = 1.0
        assert _mask_iou(a, b) == 1.0
        c = torch.zeros(4, 4)  # disjoint (empty)
        d = torch.ones(4, 4)
        # IoU of empty vs full = 0
        assert _mask_iou(c, d) == 0.0
        # both empty special case → 1.0
        assert _mask_iou(c, c) == 1.0

    def test_too_short_sequence(self):
        m = torch.zeros(1, 4, 4)
        out = TemporalConsistencyCheckerMEC().check("mask_iou", mask=m)
        assert out[2] == 0.0
        rep = json.loads(out[3])
        assert "too short" in rep["note"].lower()

    def test_mask_iou_requires_mask(self):
        with pytest.raises(ValueError):
            TemporalConsistencyCheckerMEC().check("mask_iou")

    def test_passthrough(self):
        img = torch.rand(2, 4, 4, 3)
        out = TemporalConsistencyCheckerMEC().check("pixel_diff", image=img)
        assert out[0] is img


# ══════════════════════════════════════════════════════════════════════
#  ModelMetadataExtractorMEC
# ══════════════════════════════════════════════════════════════════════

def _write_minimal_safetensors(path: Path, custom_meta: dict | None = None) -> None:
    header = {
        "weight_a": {"dtype": "F32", "shape": [4, 4], "data_offsets": [0, 64]},
        "weight_b": {"dtype": "F16", "shape": [8], "data_offsets": [64, 80]},
    }
    if custom_meta is not None:
        header["__metadata__"] = custom_meta
    hdr_bytes = json.dumps(header).encode("utf-8")
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(hdr_bytes)))
        fh.write(hdr_bytes)
        fh.write(b"\x00" * 80)  # fake tensor data


class TestModelMetadataExtractor:
    def test_safetensors_header(self, tmp_path: Path):
        f = tmp_path / "m.safetensors"
        _write_minimal_safetensors(f, custom_meta={"author": "MEC", "format": "test"})
        meta = _read_safetensors_header(str(f))
        assert meta["format"] == "safetensors"
        assert meta["tensor_count"] == 2
        assert meta["total_params"] == 4 * 4 + 8
        assert meta["metadata"]["author"] == "MEC"

    def test_node_returns_kind_and_params(self, tmp_path: Path):
        f = tmp_path / "m.safetensors"
        _write_minimal_safetensors(f)
        out = ModelMetadataExtractorMEC().extract(str(f), compute_fingerprint=True)
        meta = json.loads(out[0])
        assert out[2] == 24  # 16 + 8
        assert len(out[3]) == 64  # sha256 hex length
        assert meta["fingerprint_sha256"] == out[3]

    def test_torch_zip(self, tmp_path: Path):
        f = tmp_path / "m.pt"
        with zipfile.ZipFile(f, "w") as zf:
            zf.writestr("archive/data.pkl", b"\x80\x04N.")  # pickle: protocol 4, NONE, STOP
            zf.writestr("archive/data/0", b"\x00" * 16)
        out = ModelMetadataExtractorMEC().extract(str(f), compute_fingerprint=False)
        meta = json.loads(out[0])
        assert meta["format"] == "torch_zip"
        assert meta["member_count"] == 2

    def test_missing_file(self):
        with pytest.raises(FileNotFoundError):
            ModelMetadataExtractorMEC().extract("nonexistent.safetensors")

    def test_detect_kind_lora(self):
        meta = {"tensors": [{"name": "lora_unet_down.lora_up.weight"}], "format": "safetensors"}
        assert _detect_model_kind(meta) == "lora"

    def test_detect_kind_unknown(self):
        meta = {"tensors": [], "format": "safetensors"}
        assert _detect_model_kind(meta) == "unknown"

    def test_fingerprint_stable(self, tmp_path: Path):
        f = tmp_path / "m.safetensors"
        _write_minimal_safetensors(f)
        a = ModelMetadataExtractorMEC().extract(str(f))[3]
        b = ModelMetadataExtractorMEC().extract(str(f))[3]
        assert a == b
