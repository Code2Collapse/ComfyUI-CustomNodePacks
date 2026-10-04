"""ViTMatte's alpha must land on the frame's own pixels.

Found 2026-09-29: the HF processor PADS a frame bottom/right to a multiple of
32 (1080x1920 -> 1088x1920) and never resizes it, but the backend RESIZED the
padded alpha back to (H, W). That stretched the matte: it slid off the true
edges, more the further down/right it was (~8 px at the bottom of 1080p, 28 %
of the height on a 100 px frame). The padding is now cropped off.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

PACK = Path(__file__).resolve().parents[1]
if str(PACK) not in sys.path:
    sys.path.insert(0, str(PACK))

vm = pytest.importorskip("nodes.mask_matting.matters.vitmatte_backend",
                         reason="mask_matting matters unavailable")


def test_padding_is_cropped_not_stretched():
    H, W = 100, 150
    padded = torch.zeros(1, 1, 128, 160)
    padded[..., 60:, :] = 1.0              # an edge at row 60 of the real frame
    out = vm.unpad_alpha(padded, H, W)
    assert out.shape[-2:] == (H, W)
    col = out[0, 0, :, 10]
    assert col[59] == 0 and col[60] == 1, "the edge must stay on row 60"


def test_a_resizing_processor_is_still_mapped_back():
    small = torch.zeros(1, 1, 50, 75)
    assert vm.unpad_alpha(small, 100, 150).shape[-2:] == (100, 150)


def test_the_processor_really_pads_and_does_not_resize():
    """The assumption the crop rests on, checked against the real processor."""
    tr = pytest.importorskip("transformers")
    import numpy as np
    p = tr.VitMatteImageProcessor()
    out = p(images=np.zeros((1080, 1920, 3), np.uint8), trimaps=np.zeros((1080, 1920), np.uint8),
            return_tensors="pt")
    assert tuple(out["pixel_values"].shape[-2:]) == (1088, 1920)
