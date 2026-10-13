"""L7.99 slice 2: Inpaint Crop Pro's live preview (a changed setting, no run) shows exactly the windows a run with that
setting executes. Same geometry functions; this checks it against real runs, per setting."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_inpaint_crop_geometry import DEFAULTS, H, W, box, plate  # noqa: E402 - the same plates and settings


@pytest.fixture(autouse=True)
def temp_dir(monkeypatch, tmp_path):
    import sys
    fp = sys.modules.get("folder_paths")
    if fp is not None:
        monkeypatch.setattr(fp, "get_temp_directory", lambda: str(tmp_path), raising=False)
    return tmp_path


@pytest.fixture
def mods():
    suite = pytest.importorskip("nodes.inpaint_suite")
    live = pytest.importorskip("nodes._inpaint_preview_live")
    live.clear()
    yield suite, live
    live.clear()


def run(suite, img, mask, uid="7", **over):
    kw = dict(DEFAULTS)
    kw.update(over)
    out = suite.InpaintCropProMEC().inpaint_crop(img, mask=mask, unique_id=uid, **kw)
    return out["result"], out["ui"]


CHANGES = [
    {"context_from_mask_extend_factor": 1.6}, {"context_from_mask_extend_factor": 0.7},
    {"aspect_preset": "16:9"}, {"aspect_preset": "Auto"}, {"output_target_width": 896, "output_target_height": 512},
    {"mask_expand_pixels": 12}, {"mask_blend_pixels": 0}, {"mask_invert": True}, {"auto_context_factor": True},
    {"output_resize_to_target_size": False}, {"output_padding": "64"}, {"wan_align_multiple": 64},
    {"preresize": True}, {"extend_for_outpainting": True, "extend_right_factor": 1.4},
    {"wan_stable_crop": False}, {"wan_mask_polarity": "preserve_subject"}, {"mask_fill_holes": False},
]


@pytest.mark.parametrize("change", CHANGES, ids=[",".join(c) for c in CHANGES])
def test_live_plan_equals_a_real_run_with_the_new_setting(mods, change):
    suite, live = mods
    img = plate(B=3)
    mask = box(3, 20, 60, 70, 140, dx=40)
    run(suite, img, mask)                                               # the run the preview starts from
    got = live.render("7", change)
    assert got.get("images"), got
    (_res, ui) = run(suite, img, mask, uid="8", **change)               # a real run with the new setting
    (want,) = ui["c2c_crop_plan"]
    assert got["plan"] == want


def test_stale_answers_instead_of_guessing(mods):
    suite, live = mods
    assert live.render("nope", {})["stale"] is True                     # never ran
    img = plate(B=3)
    run(suite, img, box(3, 20, 60, 70, 140))
    out = live.render("7", {"preview_frame": 2})                         # the cached frame is 0
    assert out["stale"] is True and "frame 2" in out["reason"]


def test_masks_are_kept_exactly_and_within_budget(mods, monkeypatch):
    suite, live = mods
    g = torch.Generator().manual_seed(3)
    binary = (torch.rand(2, 33, 47, generator=g) > 0.5).float()
    soft = torch.rand(2, 33, 47, generator=g)
    for t in (binary, soft):
        p = live._pack(t)
        assert torch.equal(live._unpack(p), t)
    assert live._pack(binary)["data"].nbytes <= (binary.numel() + 7) // 8   # one bit per pixel
    monkeypatch.setattr(live, "BUDGET_BYTES", 30_000)
    img = plate()
    assert live.store("a", shape_bhw=(1, H, W), mask=box(1, 0, 0, 10, 10), opt_mask=None, frame=img[0],
                      frame_index=0, settings={}) is False               # the frame alone is over the budget
    assert live.get("a") is None


def test_lru_keeps_the_budget(mods, monkeypatch):
    suite, live = mods
    small = torch.zeros(1, 20, 20, 3)
    one = live._nbytes({"frame": (small[0] * 255).to(torch.uint8).numpy(), "mask": None, "opt": None})
    monkeypatch.setattr(live, "BUDGET_BYTES", one * 2 + 10)
    for uid in ("a", "b", "c"):
        assert live.store(uid, shape_bhw=(1, 20, 20), mask=None, opt_mask=None, frame=small[0], frame_index=0,
                          settings={})
    assert live.get("a") is None and live.get("b") is not None and live.get("c") is not None
