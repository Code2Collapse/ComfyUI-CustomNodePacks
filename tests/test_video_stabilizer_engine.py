"""The video stabilizer ENGINE is in the repo and works without any model weights (L7.90, 2026-10-12).

It used to be loaded from the gitignored third_party/ folder, so on every fresh install the four Video Stabilizer
nodes raised "not vendored"; the other tests in this area mock the engine and never noticed. Both backends are
OpenCV (Lucas-Kanade tracking / DIS optical flow): a clip with known hand-shake must come out steadier.

The suite stubs `comfy`, so the engine's comfy_api / comfy.utils imports get a minimal stand-in here; the real
server run is tools/c2c_exec_audit.py (docs/evidence/L7.90).
"""
from __future__ import annotations

import importlib
from pathlib import Path
import sys
import types

import numpy as np
import pytest
import torch


class _Anything:
    """Any attribute / call / base class the engine's module body touches at import time."""

    def __init__(self, *a, **k):
        pass

    def __call__(self, *a, **k):
        return _Anything()

    def __getattr__(self, name):
        return _Anything

    def __mro_entries__(self, bases):
        return (object,)


@pytest.fixture
def V(monkeypatch):
    io = types.ModuleType("comfy_api.latest.io")
    io.__getattr__ = lambda name: _Anything                                   # type: ignore[attr-defined]
    latest = types.ModuleType("comfy_api.latest")
    latest.io = io
    latest.ComfyExtension = object
    latest.__getattr__ = lambda name: _Anything                               # type: ignore[attr-defined]
    utils = types.ModuleType("comfy.utils")

    class ProgressBar:
        def __init__(self, *_a, **_k):
            pass

        def update(self, *_a, **_k):
            pass

        def update_absolute(self, *_a, **_k):
            pass

    utils.ProgressBar = ProgressBar
    monkeypatch.setitem(sys.modules, "comfy_api", types.ModuleType("comfy_api"))
    monkeypatch.setitem(sys.modules, "comfy_api.latest", latest)
    monkeypatch.setitem(sys.modules, "comfy.utils", utils)
    if "comfy" in sys.modules:
        monkeypatch.setattr(sys.modules["comfy"], "utils", utils, raising=False)
    # A private copy of the module (same package, so its relative imports resolve): reloading the shared one would
    # swap the classes other test files imported.
    import importlib.util

    import nodes as pack_nodes
    path = Path(pack_nodes.__file__).resolve().parent / "video_stabilizer_mec.py"
    name = "nodes._test_video_stabilizer_engine_copy"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, mod)
    spec.loader.exec_module(mod)
    return mod


def _shaky_clip(n=12, h=96, w=128, seed=4):
    rng = np.random.default_rng(seed)
    big = rng.random((h + 40, w + 40, 3)).astype(np.float32)
    big = (big + np.roll(big, 3, 0) + np.roll(big, 3, 1)) / 3.0             # some structure to track
    shifts = rng.integers(-6, 7, size=(n, 2))
    return torch.from_numpy(np.stack([big[20 + dy:20 + dy + h, 20 + dx:20 + dx + w] for dy, dx in shifts]))


def _jitter(frames: torch.Tensor) -> float:
    """Mean frame-to-frame translation (pixels) by phase correlation."""
    import cv2
    g = frames.mean(-1).numpy().astype(np.float32)
    return float(np.mean([np.hypot(*cv2.phaseCorrelate(g[i], g[i + 1])[0]) for i in range(len(g) - 1)]))


def test_the_engine_is_in_the_repo_and_loads(V):
    assert V._VENDOR.is_dir() and (V._VENDOR / "LICENSE").is_file()
    assert "third_party" not in str(V._VENDOR)
    assert V.HAS_STABILIZER, V._IMPORT_ERROR


@pytest.mark.parametrize("method", ["classic", "raft_flow"])
def test_handshake_is_reduced_without_weights(V, method):
    assert V.HAS_STABILIZER, V._IMPORT_ERROR
    clip = _shaky_clip()
    out, mask, info = V.VideoStabilizerMEC().stabilize(clip, method, "manual", 16.0, "127, 127, 127", "crop_and_pad",
                                                       "translation", True, 1.0, 0.9, 0.8, 12, False)
    assert out.shape[0] == clip.shape[0] and out.shape[-1] == 3
    assert mask.shape[0] == clip.shape[0]
    assert _jitter(out) < 0.5 * _jitter(clip), (method, _jitter(clip), _jitter(out), str(info)[:200])
