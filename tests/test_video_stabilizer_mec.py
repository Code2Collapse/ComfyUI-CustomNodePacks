"""CPU tests for nodes/video_stabilizer_mec.py: the method alias that keeps migrated graphs valid (ledger L2.02).

VideoStabilizerAutoMEC (deprecated) named raft_flow "flow" in its force_backend combo. Its core NodeReplace row
maps force_backend -> method and core copies values verbatim, so a migrated graph can arrive with method="flow".
"""
from __future__ import annotations

import inspect

import torch

from nodes.video_stabilizer_mec import VideoStabilizerMEC, VideoStabilizerAutoMEC


def test_validate_inputs_accepts_every_method_and_the_flow_alias():
    for m in ("auto", "classic", "raft_flow", "flow"):
        assert VideoStabilizerMEC.VALIDATE_INPUTS(method=m) is True


def test_validate_inputs_rejects_unknown_methods_with_a_readable_reason():
    msg = VideoStabilizerMEC.VALIDATE_INPUTS(method="optical")
    assert isinstance(msg, str) and "raft_flow" in msg and "'optical'" in msg


def test_validate_inputs_names_only_method():
    # core 0.36 skips its own list/range checks for every input VALIDATE_INPUTS names (all of them with **kwargs):
    # naming only `method` keeps preset/framing_mode/strength... validated by core.
    spec = inspect.getfullargspec(VideoStabilizerMEC.VALIDATE_INPUTS)
    assert spec.args == ["cls", "method"] and spec.varkw is None


def test_flow_alias_reaches_the_backend_as_raft_flow(monkeypatch):
    seen = {}

    def fake_impl(self, frames, method, *rest):
        seen["method"] = method
        return (frames, torch.zeros(frames.shape[:3]), "")

    monkeypatch.setattr("nodes.video_stabilizer_mec._require_stabilizer", lambda: None)
    monkeypatch.setattr(VideoStabilizerMEC, "_stabilize_impl", fake_impl)
    frames = torch.rand(2, 8, 8, 3)
    VideoStabilizerMEC().stabilize(frames, "flow", "handheld_light", 24.0, "127, 127, 127",
                                   "crop_and_pad", "similarity", False, 0.7, 0.5, 0.6, 12, True)
    assert seen["method"] == "raft_flow"


def test_auto_shim_translates_flow_the_same_way(monkeypatch):
    # the replacement row must behave like the shim it replaces
    seen = {}
    monkeypatch.setattr(VideoStabilizerMEC, "stabilize", lambda self, **kw: seen.update(kw) or ("ok",))
    VideoStabilizerAutoMEC().stabilize(torch.rand(1, 8, 8, 3), 24.0, "flow", "handheld_light", "127, 127, 127")
    assert seen["method"] == "raft_flow" and seen["preset"] == "handheld_light"
