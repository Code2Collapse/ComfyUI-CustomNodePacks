"""Ask for N frames, get N frames, whatever the model's grid is.

The point: the grid is the model's problem, not the user's. Ask for 64 on
MiniMax H3 (17n+5) and the node should render 73 and hand back 64 — not 73,
and not 56.

The property that carries the most weight is the last section: the trim obeys
a TARGET, not a remembered count. Samplers return one frame more or fewer than
asked often enough that subtracting a count is how a clip ends up short with
nothing to explain it.

CPU-only, torch only.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

PACK = Path(__file__).resolve().parents[1]
if str(PACK) not in sys.path:
    sys.path.insert(0, str(PACK))

from nodes._frame_grid import (  # noqa: E402
    CUSTOM,
    GRID_NAMES,
    GRIDS,
    FrameGridError,
    Grid,
    describe_fit,
    plan_fit,
    resolve,
    seconds_to_frames,
    trim_to,
)
from nodes.av_handles import AVHandlesMEC, forget_all  # noqa: E402


@pytest.fixture(autouse=True)
def clean():
    forget_all()
    yield
    forget_all()


def clip(n):
    t = torch.zeros(n, 8, 8, 3)
    for i in range(n):
        t[i] = i / 1000.0          # each frame identifiable
    return t


def node():
    return AVHandlesMEC()


# ── the grids themselves ────────────────────────────────────────────────────

def test_h3_accepts_the_counts_it_is_known_for():
    """124 for five seconds is the figure everyone quotes; it is 17*7+5."""
    g = GRIDS["MiniMax H3 (17n+5)"]
    for n in (5, 22, 39, 56, 73, 124):
        assert g.accepts(n), n
    for n in (6, 23, 64, 120, 123):
        assert not g.accepts(n), n


def test_wan_and_ltx_grids():
    wan = GRIDS["Wan 2.x (4n+1)"]
    ltx = GRIDS["LTX / LTX-2 (8n+1)"]
    assert wan.accepts(81) and wan.accepts(121) and not wan.accepts(80)
    assert ltx.accepts(121) and not ltx.accepts(120)


def test_cogvideox_accepts_its_usual_length():
    assert GRIDS["CogVideoX (8n+1)"].accepts(49)     # 8*6+1


def test_no_grid_accepts_anything():
    g = GRIDS["none (any frame count)"]
    for n in (1, 7, 64, 999):
        assert g.accepts(n)
    assert g.next_at_or_above(64) == 64


def test_nothing_below_the_models_floor_is_accepted():
    """A count under the floor is not a short clip, it is a broken one."""
    g = GRIDS["MiniMax H3 (17n+5)"]
    assert not g.accepts(0) and not g.accepts(4)


def test_next_at_or_above_never_rounds_down():
    """Rounding down silently delivers fewer frames than were asked for, and
    a short clip has to be rendered again."""
    for name in GRID_NAMES:
        if name == CUSTOM:
            continue
        g = GRIDS[name]
        for want in range(1, 200):
            got = g.next_at_or_above(want)
            assert got >= want, f"{name}: {want} -> {got}"
            assert g.accepts(got), f"{name}: {got} is not accepted"


def test_a_count_already_on_the_grid_does_not_move():
    assert GRIDS["MiniMax H3 (17n+5)"].next_at_or_above(124) == 124
    assert GRIDS["Wan 2.x (4n+1)"].next_at_or_above(81) == 81


# ── the fit plan ────────────────────────────────────────────────────────────

def test_the_example_from_the_brief():
    """64 frames on H3: render 73, deliver 64."""
    plan = plan_fit(64, GRIDS["MiniMax H3 (17n+5)"])
    assert plan.target == 64
    assert plan.generate == 73
    assert plan.pad == 9
    assert not plan.already_fits


def test_a_target_that_already_fits_pads_nothing():
    plan = plan_fit(73, GRIDS["MiniMax H3 (17n+5)"])
    assert plan.already_fits and plan.pad == 0


def test_the_wasted_frames_are_reported():
    """At 17n+5 an unlucky target costs 16 frames of render time, which is
    worth seeing before the render rather than after it."""
    plan = plan_fit(58, GRIDS["MiniMax H3 (17n+5)"])
    assert plan.generate == 73 and plan.wasted == 15


def test_a_nonsense_target_is_named():
    with pytest.raises(FrameGridError, match="does not mean anything"):
        plan_fit(0, GRIDS["Wan 2.x (4n+1)"])


# ── custom grids ────────────────────────────────────────────────────────────

def test_a_custom_grid_can_be_built():
    g = resolve(CUSTOM, step=6, plus=1)
    assert g.accepts(7) and g.accepts(13) and not g.accepts(8)


def test_a_custom_step_below_one_is_refused_with_the_alternative():
    with pytest.raises(FrameGridError, match="'any frame count'"):
        resolve(CUSTOM, step=0, plus=1)


def test_an_unknown_model_lists_the_ones_that_exist():
    with pytest.raises(FrameGridError, match="Choose one of"):
        resolve("Sora (made up)")


# ── trimming to a target, not by a count ────────────────────────────────────

def test_trim_to_uses_what_arrived():
    assert trim_to(73, 64) == 9


def test_trim_to_absorbs_a_sampler_that_returned_an_extra_frame():
    """THE robustness property. Pad 64->73 and the obvious trim is 'remove 9'.
    A sampler that hands back 74 then leaves 65 — silently, in a finished
    clip. Trimming TO 64 gives 64 from whatever arrives."""
    assert trim_to(74, 64) == 10
    assert trim_to(72, 64) == 8


def test_trim_to_refuses_to_invent_frames():
    with pytest.raises(FrameGridError, match="look upstream"):
        trim_to(60, 64)


# ── seconds ─────────────────────────────────────────────────────────────────

def test_five_seconds_on_h3_is_the_expected_124():
    assert seconds_to_frames(5.0, GRIDS["MiniMax H3 (17n+5)"]) == 124


def test_seconds_snap_up_to_the_grid():
    n = seconds_to_frames(3.0, GRIDS["Wan 2.x (4n+1)"])
    assert GRIDS["Wan 2.x (4n+1)"].accepts(n)


def test_an_fps_override_is_used():
    a = seconds_to_frames(2.0, GRIDS["Wan 2.x (4n+1)"], fps=16.0)
    b = seconds_to_frames(2.0, GRIDS["Wan 2.x (4n+1)"], fps=32.0)
    assert b > a


def test_a_nonsense_fps_is_named():
    with pytest.raises(FrameGridError, match="fps must be positive"):
        seconds_to_frames(2.0, GRIDS["Wan 2.x (4n+1)"], fps=-1)


# ── the round trip through the node ─────────────────────────────────────────

@pytest.mark.parametrize("want,model,expect_gen", [
    (64, "MiniMax H3 (17n+5)", 73),
    (64, "Wan 2.x (4n+1)", 65),
    (120, "LTX / LTX-2 (8n+1)", 121),
    (100, "Hunyuan Video (4n+1)", 101),
    (40, "Mochi (6n+1)", 43),
])
def test_fit_then_trim_lands_on_exactly_what_was_asked(want, model, expect_gen):
    n = node()
    src = clip(want)
    padded, _, total, _, _ = n.run(mode="fit", target_frames=want,
                                   model=model, images=src)
    assert total == expect_gen, f"{model}: generated {total}, wanted {expect_gen}"
    back, _, out, _, _ = n.run(mode="trim", images=padded)
    assert out == want
    assert torch.equal(back, src), "the wrong frames came off"


def test_fit_survives_a_sampler_that_changed_the_frame_count():
    """The reason the target is recorded and not just the count."""
    n = node()
    src = clip(64)
    padded, _, total, _, _ = n.run(mode="fit", target_frames=64,
                                   model="MiniMax H3 (17n+5)", images=src)
    assert total == 73
    # the "sampler" hands back one extra frame
    from_model = torch.cat([padded, padded[-1:]], dim=0)
    assert from_model.shape[0] == 74
    _, _, out, _, info = n.run(mode="trim", images=from_model)
    assert out == 64, "trimming by the remembered count would have given 65"
    assert "target" in info


def test_fit_with_no_target_uses_the_clips_own_length():
    n = node()
    padded, _, total, _, _ = n.run(mode="fit", target_frames=0,
                                   model="MiniMax H3 (17n+5)", images=clip(64))
    assert total == 73
    _, _, out, _, _ = n.run(mode="trim", images=padded)
    assert out == 64


def test_fit_on_a_clip_that_already_fits_changes_nothing():
    n = node()
    src = clip(73)
    padded, _, total, _, info = n.run(mode="fit", target_frames=73,
                                      model="MiniMax H3 (17n+5)", images=src)
    assert total == 73 and torch.equal(padded, src)
    assert "already a length" in info


def test_fit_moves_the_audio_too():
    n = node()
    a = {"waveform": torch.ones(1, 2, 24000), "sample_rate": 24000}
    _, out_a, total, _, _ = n.run(mode="fit", target_frames=64,
                                  model="MiniMax H3 (17n+5)",
                                  images=clip(64), audio=a, manual_fps=24.0)
    assert total == 73
    # 9 frames at 24fps = 0.375s = 9000 samples
    assert out_a["waveform"].shape[-1] == 24000 + 9000


def test_fit_needs_images_and_says_so():
    """Audio alone has no frame count to fit, and the message says what to
    use instead rather than just refusing."""
    a = {"waveform": torch.ones(1, 2, 1000), "sample_rate": 1000}
    with pytest.raises(Exception, match="fit mode needs images"):
        node().run(mode="fit", target_frames=64, audio=a)


def test_nothing_connected_at_all_is_named():
    with pytest.raises(Exception, match="nothing to put handles on"):
        node().run(mode="fit", target_frames=64)


def test_fit_refuses_to_shorten():
    """It pads; asking it to reach a shorter target is a different job."""
    with pytest.raises(Exception, match="fit only pads"):
        node().run(mode="fit", target_frames=20,
                   model="none (any frame count)", images=clip(64))


# ── the report ──────────────────────────────────────────────────────────────

def test_the_report_states_the_render_cost():
    text = describe_fit(plan_fit(58, GRIDS["MiniMax H3 (17n+5)"]),
                        "MiniMax H3 (17n+5)")
    assert "thrown away" in text and "%" in text


def test_the_report_explains_the_target_rule():
    text = describe_fit(plan_fit(64, GRIDS["MiniMax H3 (17n+5)"]), "H3")
    assert "TARGET, not by a count" in text


def test_the_report_says_when_nothing_was_needed():
    text = describe_fit(plan_fit(124, GRIDS["MiniMax H3 (17n+5)"]), "H3")
    assert "already a length" in text
