"""L2.14: MaskOpsMEC accepts the combo values old saves carry.

Core's NodeReplace copies MaskMattingMEC values verbatim; the SAM 1/2 segmenters were removed (531eb78); the
"  [missing-deps]" badge comes and goes with what is installed; weight choices differ per machine. VALIDATE_INPUTS
normalises them the way execute() does and refuses only a backend that does not exist at all.
"""
from __future__ import annotations

from pathlib import Path

import pytest

PACK = Path(__file__).resolve().parents[1]


@pytest.fixture
def node():
    # Imported inside the test, never at collection: the SAM loader it pulls in caches whether folder_paths exists
    # at import time, and other suites stub folder_paths first.
    return pytest.importorskip("nodes.mask_matting.node")


def test_badge_and_removed_segmenters_normalise(node):
    N = node._normalize_legacy_choice
    assert N("segmenter", "sam3.1  [missing-deps]") == "sam3.1"
    for old in ("sam2", "sam2.1", "sam2.1  [missing-deps]", "sam1"):
        assert N("segmenter", old) == "sam3.1"
    assert N("matter", "matanyone  [missing-deps]") == "matanyone"


def test_unknown_weight_falls_back_to_auto_and_known_tail_maps(node, monkeypatch):
    N = node._normalize_legacy_choice
    monkeypatch.setattr(node, "_all_weight_files", lambda: ["(auto)", "vitmatte/vitmatte-small.safetensors",
                                                           "[preset:sam3.1] sam3.pt"])
    assert N("weight", "sam2.1/sam2.1_hiera_large.pt") == "(auto)"
    assert N("weight", "vitmatte-small.safetensors") == "vitmatte/vitmatte-small.safetensors"
    assert N("weight", "sam3.pt") == "[preset:sam3.1] sam3.pt"
    assert N("weight", "") == "(auto)"


def test_validate_accepts_legacy_and_refuses_unknown_backends(node):
    M = node.MaskOpsMEC
    assert M.VALIDATE_INPUTS("sam2.1  [missing-deps]", "vitmatte", "sam2/x.pt", "(auto)") is True
    assert M.VALIDATE_INPUTS("auto_best", "none", "(auto)", "(auto)") is True
    msg = M.VALIDATE_INPUTS("no_such_backend", "none", "(auto)", "(auto)")
    assert isinstance(msg, str) and "Unknown segmenter" in msg


def test_execute_normalises_before_use():
    src = (PACK / "nodes" / "mask_matting" / "node.py").read_text(encoding="utf-8")
    a = src.index("def execute(self, image, segmenter")
    body = src[a:]
    assert body.index('segmenter = _normalize_legacy_choice("segmenter", segmenter)') < body.index("seg_key = _strip_badge(segmenter)")
