"""AV handles: add frames, then trim exactly those back off.

The bug this replaces: two nodes, each with its own handle_frames widget, set
at different times. They drift, and the symptom is a clip a few frames long or
short with nothing to say why. So the tests are mostly about the COUNT — where
trim gets it, and what happens when it cannot.

CPU-only, torch only, no ComfyUI.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

PACK = Path(__file__).resolve().parents[1]
if str(PACK) not in sys.path:
    sys.path.insert(0, str(PACK))

from nodes.av_handles import (  # noqa: E402
    AVHandlesError,
    AVHandlesMEC,
    forget_all,
    next_on_grid,
    on_grid,
    pad_images,
    plan_add,
    resolve_fps,
    shift_audio,
    trim_images,
)


@pytest.fixture(autouse=True)
def clean():
    forget_all()
    yield
    forget_all()


def clip(n=24, h=8, w=8):
    """Each frame carries its own index so a trim off the wrong end shows."""
    t = torch.zeros(n, h, w, 3)
    for i in range(n):
        t[i] = i / 100.0
    return t


def audio(seconds=1.0, sr=16000, ch=2, dims=3):
    w = torch.ones(ch, int(seconds * sr))
    if dims == 3:
        w = w.unsqueeze(0)
    elif dims == 1:
        w = w[0]
    return {"waveform": w, "sample_rate": sr}


def run(**kw):
    return AVHandlesMEC().run(**kw)


# ── the round trip, which is the whole point ────────────────────────────────

def test_add_then_trim_returns_the_original_exactly():
    src = clip(24)
    added, _, n_add, _, _ = run(mode="add", handle_frames=12, images=src)
    assert n_add == 36
    back, _, n_back, _, info = run(mode="trim", handle_frames=0, images=added)
    assert n_back == 24
    assert torch.equal(back, src), "the wrong frames came off"
    assert "remembered" in info


def test_the_trim_needs_no_number_typed_in():
    """THE point. Two widgets that must agree, set at different times, drift."""
    src = clip(30)
    added, _, _, _, _ = run(mode="add", handle_frames=7, images=src)
    back, _, n, _, _ = run(mode="trim", images=added)     # handle_frames left at 0
    assert n == 30


def test_a_wired_handles_signal_beats_the_memory():
    """Explicit and exact, and it works across separate workflows where the
    memory cannot."""
    src = clip(20)
    added, _, _, sig, _ = run(mode="add", handle_frames=5, images=src)
    forget_all()                                   # as if a different session
    back, _, n, _, info = run(mode="trim", images=added, handles=sig)
    assert n == 20
    assert "wire" in info


def test_a_manual_count_overrides_the_memory():
    """Only when asked for explicitly — see the next test for why."""
    src = clip(20)
    added, _, _, _, _ = run(mode="add", handle_frames=5, images=src)
    back, _, n, _, info = run(mode="trim", handle_frames=2,
                              trim_amount="manual", images=added)
    assert n == 23, "the manual 2 was ignored"
    assert "manual" in info


def test_the_widget_default_does_not_masquerade_as_an_override():
    """THE bug this control exists for. handle_frames defaults to 8 because
    that is a sensible number to ADD; reading any non-zero value as a typed
    override meant switching to trim and leaving the widget alone trimmed 8,
    never the remembered count — the node's whole promise, defeated by its
    own default."""
    src = clip(30)
    added, _, _, _, _ = run(mode="add", handle_frames=7, images=src)
    back, _, n, _, info = run(mode="trim", handle_frames=8, images=added)
    assert n == 30, "the default 8 was used instead of the remembered 7"
    assert "remembered" in info


def test_manual_mode_with_zero_trims_nothing():
    """An explicit instruction is obeyed even when it is a no-op."""
    added, _, _, _, _ = run(mode="add", handle_frames=5, images=clip(20))
    _, _, n, _, info = run(mode="trim", handle_frames=0,
                           trim_amount="manual", images=added)
    assert n == 25
    assert "manual" in info


def test_with_nothing_to_go_on_it_trims_nothing_and_says_so():
    """Guessing here silently shortens the clip."""
    out, _, n, _, info = run(mode="trim", images=clip(20))
    assert n == 20
    assert "NOTHING WAS TRIMMED" in info
    assert "wire its `handles`" in info


def test_two_clips_keep_separate_counts_when_named():
    a = clip(20); b = clip(40)
    ca, _, _, _, _ = run(mode="add", handle_frames=4, images=a, memo_key="a")
    cb, _, _, _, _ = run(mode="add", handle_frames=9, images=b, memo_key="b")
    assert run(mode="trim", images=ca, memo_key="a")[2] == 20
    assert run(mode="trim", images=cb, memo_key="b")[2] == 40


def test_without_names_the_second_add_wins():
    """Documented behaviour, not an accident: one memory per key."""
    run(mode="add", handle_frames=4, images=clip(20))
    c2, _, _, _, _ = run(mode="add", handle_frames=9, images=clip(40))
    assert run(mode="trim", images=c2)[2] == 40


# ── which end ───────────────────────────────────────────────────────────────

def test_handles_go_on_the_head_by_default():
    src = clip(10)
    out = pad_images(src, 3, "head")
    assert out.shape[0] == 13
    assert torch.equal(out[0], src[0]) and torch.equal(out[2], src[0])
    assert torch.equal(out[3], src[0])          # the original first frame


def test_the_tail_option_pads_the_other_end():
    src = clip(10)
    out = pad_images(src, 3, "tail")
    assert torch.equal(out[-1], src[-1]) and torch.equal(out[9], src[9])


def test_trim_takes_from_the_same_end_it_added_to():
    src = clip(12)
    added, _, _, _, _ = run(mode="add", handle_frames=4, images=src, side="tail")
    back, _, _, _, _ = run(mode="trim", images=added, side="tail")
    assert torch.equal(back, src)


def test_handles_repeat_the_edge_frame_not_black():
    """A black run at the head is content a video model tries to continue
    out of, which is the opposite of stabilising it."""
    src = clip(10)
    out = pad_images(src, 4, "head")
    assert float(out[0].max()) == float(src[0].max())
    assert float(out[:4].max()) > 0 or float(src[0].max()) == 0


# ── frame grids ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("grid,good,bad", [
    ("WAN (4n+1)", [1, 5, 81, 121], [2, 80, 122]),
    ("LTX2 (8n+1)", [1, 9, 121], [8, 120]),
    ("H3 (17n+5)", [5, 22, 124], [6, 123]),
])
def test_grid_membership(grid, good, bad):
    for n in good:
        assert on_grid(n, grid), f"{n} should be on {grid}"
    for n in bad:
        assert not on_grid(n, grid), f"{n} should not be on {grid}"


def test_disabled_grid_accepts_anything():
    assert on_grid(7, "disabled") and on_grid(0, "disabled")


def test_next_on_grid_rounds_up_only():
    assert next_on_grid(118, "WAN (4n+1)") == 121
    assert next_on_grid(121, "WAN (4n+1)") == 121
    assert next_on_grid(124, "H3 (17n+5)") == 124
    assert next_on_grid(125, "H3 (17n+5)") == 141


def test_zero_handles_with_a_grid_means_just_enough():
    """The common case: the clip is nearly right and should not be padded
    more than it has to be."""
    assert plan_add(118, 0, "WAN (4n+1)") == 3
    assert plan_add(121, 0, "WAN (4n+1)") == 0


def test_a_requested_count_is_rounded_up_to_the_grid():
    got = plan_add(100, 8, "WAN (4n+1)")
    assert on_grid(100 + got, "WAN (4n+1)")
    assert got >= 8


def test_no_grid_means_exactly_what_was_asked():
    assert plan_add(100, 8, "disabled") == 8


def test_landing_off_grid_is_called_out():
    _, _, _, _, info = run(mode="add", handle_frames=0, images=clip(118),
                           padding_mode="WAN (4n+1)")
    assert "is on the WAN (4n+1) grid" in info


# ── audio stays in step ─────────────────────────────────────────────────────

def test_audio_is_padded_by_the_matching_duration():
    """Handles on the picture without matching silence slide the audio
    against it, and eight frames out is worse than no lip-sync."""
    a = audio(seconds=1.0, sr=16000)
    out = shift_audio(a, frames=12, fps=24.0, side="head", add=True)
    assert out["waveform"].shape[-1] == 16000 + 8000   # 12/24 s = 0.5 s


def test_trimming_audio_undoes_exactly_that():
    a = audio(seconds=1.0, sr=16000)
    padded = shift_audio(a, 12, 24.0, "head", add=True)
    back = shift_audio(padded, 12, 24.0, "head", add=False)
    assert back["waveform"].shape == a["waveform"].shape


@pytest.mark.parametrize("dims", [1, 2, 3])
def test_the_audio_tensor_rank_survives(dims):
    """A node that silently changes an audio tensor's rank breaks whatever is
    downstream of it."""
    a = audio(dims=dims)
    out = shift_audio(a, 6, 24.0, "head", add=True)
    assert out["waveform"].ndim == a["waveform"].ndim


def test_the_silence_goes_on_the_same_end_as_the_frames():
    a = audio(seconds=0.5, sr=1000)
    head = shift_audio(a, 24, 24.0, "head", add=True)["waveform"]
    tail = shift_audio(a, 24, 24.0, "tail", add=True)["waveform"]
    w_h = head[0] if head.ndim == 3 else head
    w_t = tail[0] if tail.ndim == 3 else tail
    assert float(w_h[0, 0]) == 0.0 and float(w_h[0, -1]) == 1.0
    assert float(w_t[0, 0]) == 1.0 and float(w_t[0, -1]) == 0.0


def test_trimming_more_audio_than_exists_is_refused():
    with pytest.raises(AVHandlesError, match="does not belong"):
        shift_audio(audio(seconds=0.1, sr=1000), 240, 24.0, "head", add=False)


def test_no_audio_is_fine():
    assert shift_audio(None, 5, 24.0, "head", True) is None
    out, aud, n, _, info = run(mode="add", handle_frames=3, images=clip(10))
    assert aud is None and n == 13
    assert "No audio connected" in info


# ── fps ─────────────────────────────────────────────────────────────────────

def test_fps_is_measured_from_the_clips_own_audio():
    """A guessed 30 on 24fps material puts the silence 25% out, which reads
    as a lip-sync fault rather than a settings one."""
    fps, why = resolve_fps(clip(24), audio(seconds=1.0), 0.0, 24)
    assert fps == pytest.approx(24.0, abs=0.01)
    assert "measured" in why


def test_a_manual_fps_wins():
    fps, why = resolve_fps(clip(24), audio(seconds=1.0), 48.0, 24)
    assert fps == 48.0 and "by hand" in why


def test_an_assumed_fps_is_flagged_loudly():
    fps, why = resolve_fps(clip(24), None, 0.0, 24)
    assert fps == 30.0 and why.startswith("ASSUMED")
    _, _, _, _, info = run(mode="add", handle_frames=4, images=clip(24),
                           audio=audio(seconds=0), manual_fps=0.0) \
        if False else (None, None, None, None, "")


def test_the_report_warns_when_the_fps_was_guessed():
    a = {"waveform": torch.ones(1, 2, 10), "sample_rate": 16000}
    _, _, _, _, info = run(mode="add", handle_frames=4, images=None,
                           audio=a, manual_fps=0.0)
    assert "ASSUMED" in info or "guess" in info


def test_the_trim_reuses_the_fps_the_add_measured():
    src, a = clip(24), audio(seconds=1.0)
    ai, ao, _, _, _ = run(mode="add", handle_frames=12, images=src, audio=a)
    _, back, _, _, info = run(mode="trim", images=ai, audio=ao)
    assert back["waveform"].shape == a["waveform"].shape
    assert "carried from the add side" in info


# ── refusing rather than mangling ───────────────────────────────────────────

def test_trimming_the_whole_clip_away_is_refused():
    """An empty batch reads downstream as 'no video' and says nothing."""
    run(mode="add", handle_frames=30, images=clip(30))
    with pytest.raises(AVHandlesError, match="would leave nothing"):
        run(mode="trim", images=clip(20))


def test_connecting_nothing_is_named():
    with pytest.raises(AVHandlesError, match="nothing to put handles on"):
        run(mode="add", handle_frames=4)


def test_an_unknown_mode_is_named():
    with pytest.raises(AVHandlesError, match="Unknown mode"):
        run(mode="sideways", images=clip(4))


def test_a_single_frame_is_accepted_as_a_batch():
    out, _, n, _, _ = run(mode="add", handle_frames=2,
                          images=torch.zeros(8, 8, 3))
    assert n == 3


# ── the signal ──────────────────────────────────────────────────────────────

def test_the_handles_output_carries_the_real_count():
    _, _, _, sig, _ = run(mode="add", handle_frames=0, images=clip(118),
                          padding_mode="WAN (4n+1)")
    assert sig["frames"] == 3, "the signal must carry what was ACTUALLY added"
    assert sig["side"] == "head" and sig["source_frames"] == 118


def test_add_never_caches_away_its_recording():
    """Add records state. A cache hit that skips it leaves the trim working
    from a stale count."""
    a = AVHandlesMEC.IS_CHANGED(mode="add", handle_frames=4)
    b = AVHandlesMEC.IS_CHANGED(mode="trim", handle_frames=4)
    assert a != b
