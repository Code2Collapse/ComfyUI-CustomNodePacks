"""Two upgrades to the live sampler preview: fetch the decoder, and stop
showing frame 0 of a video forever.

Both sit in _c2c_preview_guard.py, on the wrapper around
`latent_preview.get_previewer`. That function is the one place EVERY sampler
goes through - core's, and WanVideoWrapper's, and anyone else's - which is why
the work belongs there and not in a node or a JS overlay. Nothing here draws
anything; ComfyUI's own previewer still renders the picture.

Frame 0 is the worst frame a video model could show. It is usually the one
pinned to the conditioning image, so it looks correct from the first step
whether or not the rest of the clip is working, and it never changes. You
watch a still picture for two minutes and learn nothing.

No network. The fetch is tested through its table and its guards; the one test
that would download is the URL-shape check, which only inspects strings.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path

import pytest

PACK = Path(__file__).resolve().parents[1]
GUARD_SRC = PACK / "nodes" / "_c2c_preview_guard.py"


@pytest.fixture(scope="module")
def guard():
    """Load the guard with ComfyUI absent, so nothing patches anything.

    Its install functions all begin with `import latent_preview` inside a try,
    so with no ComfyUI on the path they no-op and leave the module's own
    helpers available to test directly.
    """
    assert GUARD_SRC.exists()
    spec = importlib.util.spec_from_file_location("_c2c_preview_guard_test", GUARD_SRC)
    mod = importlib.util.module_from_spec(spec)
    saved = sys.modules.get("latent_preview")
    sys.modules.pop("latent_preview", None)
    try:
        spec.loader.exec_module(mod)
    finally:
        if saved is not None:
            sys.modules["latent_preview"] = saved
    return mod


class FakeLatent:
    """Stands in for a torch tensor: only shape, ndim and slicing are used."""

    def __init__(self, shape, tag="root"):
        self.shape = shape
        self.ndim = len(shape)
        self.tag = tag

    def __getitem__(self, key):
        # x0[:, :, i:i+1] - record which frame was taken
        sl = key[2]
        shape = list(self.shape)
        shape[2] = sl.stop - sl.start
        out = FakeLatent(tuple(shape), tag=f"frame{sl.start}")
        out.frame = sl.start
        return out


class Recorder:
    """A previewer that records what it was handed."""

    def __init__(self):
        self.seen = []

    def decode_latent_to_preview_image(self, fmt, x0):
        self.seen.append(getattr(x0, "frame", None))
        return ("JPEG", object(), 512)

    def decode_latent_to_preview(self, x0):
        self.seen.append(getattr(x0, "frame", None))
        return object()


def video_format():
    fmt = types.SimpleNamespace()
    fmt.latent_dimensions = 3
    fmt.taesd_decoder_name = "lighttaew2_2"
    return fmt


def image_format():
    fmt = types.SimpleNamespace()
    fmt.latent_dimensions = 2
    fmt.taesd_decoder_name = "taesd_decoder"
    return fmt


# ── the frame policy ───────────────────────────────────────────────────────

def test_sweep_shows_a_different_frame_each_time(guard):
    """The whole point. Core shows frame 0 on all 20 steps of a 20-step
    sample; this shows 20 different frames of the clip."""
    inner = Recorder()
    prev = guard._FramePickingPreviewer(inner, "sweep")
    x0 = FakeLatent((1, 16, 21, 60, 104))
    for _ in range(8):
        prev.decode_latent_to_preview_image("JPEG", x0)
    assert inner.seen == [0, 1, 2, 3, 4, 5, 6, 7]


def test_sweep_wraps_at_the_end_of_the_clip(guard):
    inner = Recorder()
    prev = guard._FramePickingPreviewer(inner, "sweep")
    x0 = FakeLatent((1, 16, 3, 60, 104))
    for _ in range(7):
        prev.decode_latent_to_preview_image("JPEG", x0)
    assert inner.seen == [0, 1, 2, 0, 1, 2, 0]


def test_middle_is_stable_and_is_not_frame_zero(guard):
    """For anyone who finds a moving preview distracting - still a far better
    single frame than the conditioning frame."""
    inner = Recorder()
    prev = guard._FramePickingPreviewer(inner, "middle")
    x0 = FakeLatent((1, 16, 21, 60, 104))
    for _ in range(4):
        prev.decode_latent_to_preview_image("JPEG", x0)
    assert inner.seen == [10, 10, 10, 10]


def test_an_image_latent_is_handed_over_untouched(guard):
    """4-D means no temporal axis. Slicing one would corrupt the preview."""
    inner = Recorder()
    prev = guard._FramePickingPreviewer(inner, "sweep")
    x0 = FakeLatent((1, 4, 64, 64))
    got = prev.decode_latent_to_preview_image("JPEG", x0)
    assert got is not None
    assert inner.seen == [None], "an image latent was sliced"


def test_a_single_frame_clip_is_left_alone(guard):
    inner = Recorder()
    prev = guard._FramePickingPreviewer(inner, "sweep")
    prev.decode_latent_to_preview_image("JPEG", FakeLatent((1, 16, 1, 60, 104)))
    assert inner.seen == [None]


def test_an_unexpected_shape_falls_through_rather_than_failing(guard):
    """This runs inside a sampler callback. Raising here would take down the
    sample over a preview."""
    class Hostile:
        ndim = 5
        shape = (1, 16, 8, 60, 104)

        def __getitem__(self, key):
            raise RuntimeError("not sliceable")

    inner = Recorder()
    prev = guard._FramePickingPreviewer(inner, "sweep")
    assert prev.decode_latent_to_preview_image("JPEG", Hostile()) is not None


def test_attributes_of_the_wrapped_previewer_still_reach_through(guard):
    """Samplers do reach into a previewer - Kijai's checks for its own
    attributes. An opaque wrapper would break them."""
    inner = Recorder()
    inner.taesd = "sentinel"
    prev = guard._FramePickingPreviewer(inner, "sweep")
    assert prev.taesd == "sentinel"


def test_the_policy_only_applies_to_video(guard):
    inner = Recorder()
    assert guard._with_frame_policy(inner, image_format()) is inner
    assert guard._with_frame_policy(inner, video_format()) is not inner


def test_first_opts_back_into_core_behaviour(guard, monkeypatch):
    inner = Recorder()
    monkeypatch.setattr(guard, "_VIDEO_FRAME_POLICY", "first")
    assert guard._with_frame_policy(inner, video_format()) is inner


def test_no_previewer_stays_no_previewer(guard):
    """get_previewer returning None means the user turned previews off."""
    assert guard._with_frame_policy(None, video_format()) is None


# ── fetching a decoder ─────────────────────────────────────────────────────

def test_only_urls_that_were_actually_checked_are_listed(guard):
    """Four candidates were dropped during development because they 404'd or
    were gated. A guessed URL is a silent failure on every single sample,
    which is worse than saying "I cannot fetch this one" - so the table holds
    only what was verified, and the video decoders are deliberately absent."""
    src = guard._TAE_SOURCES
    assert set(src) == {"taesd_decoder", "taesdxl_decoder",
                        "taesd3_decoder", "taef1_decoder"}
    for name, url in src.items():
        assert url.startswith("https://"), f"{name} is not https"
        assert name in url, f"{url} does not look like the {name} file"
    for gated in ("taehv", "lighttaew2_1", "lighttaew2_2", "lighttaehy1_5"):
        assert gated not in src, (
            f"{gated} is back in the table - it was removed because the source "
            "does not resolve. Re-check it before adding it again.")


def test_a_missing_source_says_what_to_do_instead_of_guessing(guard, monkeypatch, caplog):
    """The video path. It must be actionable, not silent."""
    monkeypatch.setattr(guard, "_taesd_decoder_present", lambda fmt: False)
    monkeypatch.setattr(guard, "_vae_approx_dir", lambda: os.sep + "models")
    monkeypatch.setattr(guard, "_fetch_tried", set())
    started = []
    monkeypatch.setattr(guard.threading, "Thread",
                        lambda *a, **k: started.append(k) or pytest.fail("downloaded"))
    with caplog.at_level("INFO"):
        guard._ensure_decoder(video_format())
    text = caplog.text
    assert "lighttaew2_2" in text
    assert "models" in text
    assert not started


def test_it_does_not_fetch_a_decoder_that_is_already_there(guard, monkeypatch):
    monkeypatch.setattr(guard, "_taesd_decoder_present", lambda fmt: True)
    monkeypatch.setattr(guard, "_fetch_tried", set())
    monkeypatch.setattr(guard.threading, "Thread",
                        lambda *a, **k: pytest.fail("fetched one we already have"))
    guard._ensure_decoder(image_format())


def test_it_tries_once_per_decoder_not_once_per_sample(guard, monkeypatch):
    """get_previewer runs on every sampler call. Retrying a failing download
    each time would hammer the host and stall nothing visibly."""
    monkeypatch.setattr(guard, "_taesd_decoder_present", lambda fmt: False)
    monkeypatch.setattr(guard, "_vae_approx_dir", lambda: os.sep + "models")
    monkeypatch.setattr(guard, "_fetch_tried", set())
    calls = []

    class FakeThread:
        def __init__(self, *a, **k):
            calls.append(k.get("name"))

        def start(self):
            pass

    monkeypatch.setattr(guard.threading, "Thread", FakeThread)
    for _ in range(5):
        guard._ensure_decoder(image_format())
    assert len(calls) == 1, f"attempted {len(calls)} times"


def test_the_env_var_turns_it_off(guard, monkeypatch):
    monkeypatch.setenv("C2C_NO_TAE_DOWNLOAD", "1")
    monkeypatch.setattr(guard, "_taesd_decoder_present", lambda fmt: False)
    monkeypatch.setattr(guard, "_fetch_tried", set())
    monkeypatch.setattr(guard.threading, "Thread",
                        lambda *a, **k: pytest.fail("downloaded despite the opt-out"))
    guard._ensure_decoder(image_format())


def test_the_fetch_never_raises_into_a_sampler(guard, monkeypatch):
    """It is called from get_previewer, inside the sampling loop. An exception
    here would kill a run over a preview."""
    def boom(_fmt):
        raise RuntimeError("folder_paths exploded")

    monkeypatch.setattr(guard, "_taesd_decoder_present", boom)
    monkeypatch.setattr(guard, "_fetch_tried", set())
    guard._ensure_decoder(image_format())   # must not raise


def test_a_short_download_is_rejected_rather_than_saved(guard, tmp_path, monkeypatch):
    """An HTML error page that returns 200 is a few KB. Saved as a decoder it
    loads, fails deep inside, and looks like a corrupt install."""
    import io

    class FakeResp:
        def __init__(self, data):
            self._b = io.BytesIO(data)

        def read(self, n):
            return self._b.read(n)

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(guard.__dict__.setdefault("urllib", types.SimpleNamespace()),
                        "request", types.SimpleNamespace(), raising=False)
    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen",
                        lambda *a, **k: FakeResp(b"<html>404</html>"))
    guard._download_tae("taesd_decoder", "https://example.invalid/x.pth", str(tmp_path))
    assert not list(tmp_path.glob("taesd_decoder*")), "a 15-byte 'decoder' was saved"


def test_a_good_download_lands_atomically(guard, tmp_path, monkeypatch):
    """Written to .part and renamed, so an interrupted fetch cannot be picked
    up as a model."""
    import io
    import urllib.request

    payload = b"\x00" * (2 << 20)

    class FakeResp:
        def __init__(self):
            self._b = io.BytesIO(payload)

        def read(self, n):
            return self._b.read(n)

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: FakeResp())
    guard._download_tae("taesd_decoder", "https://example.invalid/taesd_decoder.pth",
                        str(tmp_path))
    got = list(tmp_path.glob("taesd_decoder*"))
    assert [p.name for p in got] == ["taesd_decoder.pth"]
    assert got[0].stat().st_size == len(payload)
    assert not list(tmp_path.glob("*.part")), "the temp file was left behind"


def test_the_user_can_add_a_source_without_editing_the_file(guard, tmp_path, monkeypatch):
    """The video decoders cannot be fetched from here, so there has to be a
    way to point at one that does not involve a code change."""
    import json

    (tmp_path / "_c2c_tae_sources.json").write_text(
        json.dumps({"lighttaew2_2": "https://example.test/lighttaew2_2.safetensors"}),
        encoding="utf-8")
    monkeypatch.setattr(guard, "_vae_approx_dir", lambda: str(tmp_path))
    sources = guard._tae_sources()
    assert sources["lighttaew2_2"] == "https://example.test/lighttaew2_2.safetensors"
    assert "taesd_decoder" in sources, "the built-ins were lost"


def test_a_broken_sources_file_does_not_break_previews(guard, tmp_path, monkeypatch):
    (tmp_path / "_c2c_tae_sources.json").write_text("{not json", encoding="utf-8")
    monkeypatch.setattr(guard, "_vae_approx_dir", lambda: str(tmp_path))
    assert "taesd_decoder" in guard._tae_sources()


# ── it stays out of core's way ─────────────────────────────────────────────

def test_it_still_hooks_the_one_place_every_sampler_reaches(guard):
    """get_previewer, not a node and not a JS overlay. That is what makes this
    work for third-party samplers without them knowing about it."""
    src = GUARD_SRC.read_text(encoding="utf-8")
    assert "latent_preview.get_previewer = (" in src
    assert "_ensure_decoder(latent_format)" in src
    assert "_with_frame_policy(" in src


def test_it_draws_nothing_of_its_own(guard):
    """The standing rule for this pack: never overlap core's UI. The preview
    picture is rendered by ComfyUI, from ComfyUI's own previewer."""
    src = GUARD_SRC.read_text(encoding="utf-8")
    for forbidden in ("document.", "addDOMWidget", "<canvas"):
        assert forbidden not in src, f"{forbidden} - this is backend-only"
