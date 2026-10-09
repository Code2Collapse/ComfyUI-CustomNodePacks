"""L7.57: the SeedVR2 preview hook (nodes/seedvr2_preview.py).

A stand-in module with the numz pack's names and signatures (src.interfaces.video_upscaler + the four phases) drives
the wrappers; a contract test reads the REAL pack in third_party with ast so a change of theirs shows up here.
"""
from __future__ import annotations

import ast
import io
import sys
import types
from collections import namedtuple
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")
Image = pytest.importorskip("PIL.Image")

PACK = Path(__file__).resolve().parents[1]
REF = PACK.parent / "third_party" / "ComfyUI-SeedVR2_VideoUpscaler"
MOD_NAME = "custom_nodes.SeedVR2-standin.src.interfaces.video_upscaler"


@pytest.fixture
def sp(monkeypatch):
    if str(PACK) not in sys.path:
        sys.path.insert(0, str(PACK))
    import importlib

    mod = importlib.import_module("nodes.seedvr2_preview")
    events = []
    monkeypatch.setattr(mod, "_send", lambda ev, data: events.append((ev, data)))
    monkeypatch.setattr(mod, "LIVE_MIN_INTERVAL_S", 0.0)
    monkeypatch.setattr(mod, "_STRIPS", type(mod._STRIPS)())
    monkeypatch.delenv("C2C_SEEDVR2_PREVIEW", raising=False)
    ctx_mod = types.ModuleType("comfy_execution.utils")
    EC = namedtuple("EC", "prompt_id node_id list_index")
    ctx_mod.get_executing_context = lambda: EC("p1", "7", None)
    monkeypatch.setitem(sys.modules, "comfy_execution", types.ModuleType("comfy_execution"))
    monkeypatch.setitem(sys.modules, "comfy_execution.utils", ctx_mod)
    mod.events = events
    return mod


def _standin(monkeypatch, *, batches=3, per=4, h=96, w=128, prepend=0, rgba=False):
    """The pack's node module as far as the hook sees it: same names, same signatures, same ctx contract."""
    m = types.ModuleType(MOD_NAME)

    class SeedVR2VideoUpscaler:
        pass

    def encode_all_batches(runner, ctx=None, images=None, debug=None, progress_callback=None):
        ctx["input_images"] = images
        return ctx

    def upscale_all_batches(runner, ctx=None, debug=None, progress_callback=None, seed=0):
        return ctx

    def decode_all_batches(runner, ctx=None, debug=None, progress_callback=None, cache_model=False):
        T = batches * per
        C = 4 if rgba else 3
        ctx["final_video"] = torch.empty((T, h, w, C))
        ctx["decode_batch_info"] = []
        for b in range(batches):
            s, e = b * per, (b + 1) * per
            level = (b + 1) / batches * 2 - 1                    # decode writes [-1, 1]
            ctx["final_video"][s:e] = level
            ctx["decode_batch_info"].append((s, e, b, per))
            if progress_callback:
                progress_callback(b + 1, batches, 1, "Phase 3: Decoding")
        return ctx

    def postprocess_all_batches(ctx=None, debug=None, progress_callback=None, color_correction="lab",
                                prepend_frames=0, temporal_overlap=0, batch_size=1):
        ctx["final_video"] = (ctx["final_video"] + 1) / 2
        if prepend_frames:
            ctx["final_video"] = ctx["final_video"][prepend_frames:]
        if progress_callback:
            progress_callback(1, 1, 1, "Phase 4: Post-processing")
        del ctx["input_images"]                                   # theirs releases the input here
        return ctx

    for f in (SeedVR2VideoUpscaler, encode_all_batches, upscale_all_batches, decode_all_batches,
              postprocess_all_batches):
        setattr(m, f.__name__, f)
    monkeypatch.setitem(sys.modules, MOD_NAME, m)

    def run_node(images, their_progress=None):
        # what their node does: module globals, keyword arguments
        ctx = {}
        ctx = m.encode_all_batches(None, ctx=ctx, images=images, debug=None, progress_callback=their_progress)
        ctx = m.decode_all_batches(None, ctx=ctx, debug=None, progress_callback=their_progress, cache_model=False)
        ctx = m.postprocess_all_batches(ctx=ctx, debug=None, progress_callback=their_progress,
                                        prepend_frames=prepend, batch_size=per)
        return ctx["final_video"]

    m.run_node = run_node
    return m


def _decode_jpeg(data_or_url):
    raw = data_or_url
    if isinstance(raw, str):
        import base64
        raw = base64.b64decode(raw.split(",", 1)[1])
    return np.asarray(Image.open(io.BytesIO(raw)).convert("L"), dtype=np.float64)


def test_install_wraps_once_and_kill_switch(sp, monkeypatch):
    m = _standin(monkeypatch)
    assert sp.install() == 1
    dec = m.decode_all_batches
    assert sp.install() == 1 and m.decode_all_batches is dec          # idempotent
    m2 = _standin(monkeypatch)
    monkeypatch.setenv("C2C_SEEDVR2_PREVIEW", "0")
    assert sp.install() == 0 and not getattr(m2.decode_all_batches, sp._MARK, False)


def test_live_frame_per_batch_with_before_and_after(sp, monkeypatch):
    m = _standin(monkeypatch, batches=3, per=4)
    sp.install()
    theirs = []
    images = torch.full((12, 48, 64, 3), 0.25)
    out = m.run_node(images, their_progress=lambda *a: theirs.append(a))
    live = [d for e, d in sp.events if e == "c2c.seedvr2.preview"]
    assert [d["frame"] for d in live] == [3, 7, 11]
    assert [d["batch"] for d in live] == [1, 2, 3] and live[0]["batches"] == 3
    assert live[0]["node"] == "7" and live[0]["prompt_id"] == "p1"
    # batch levels -1/3, 1/3, 1 in [-1, 1] -> 1/3, 2/3, 1 on screen
    for d, want in zip(live, (1 / 3, 2 / 3, 1.0)):
        assert abs(_decode_jpeg(d["after"]).mean() / 255 - want) < 0.02
        assert abs(_decode_jpeg(d["before"]).mean() / 255 - 0.25) < 0.02
    assert len(theirs) == 4                                            # their callback still sees every step
    assert out.shape == (12, 96, 128, 3)                               # their result untouched


def test_throttle_keeps_first_and_last(sp, monkeypatch):
    monkeypatch.setattr(sp, "LIVE_MIN_INTERVAL_S", 3600.0)
    m = _standin(monkeypatch, batches=5)
    sp.install()
    m.run_node(torch.rand(20, 32, 32, 3))
    assert [d["batch"] for e, d in sp.events if e == "c2c.seedvr2.preview"] == [1, 5]


def test_strip_for_scrub_and_route_frames(sp, monkeypatch):
    m = _standin(monkeypatch, batches=2, per=4, prepend=2)
    sp.install()
    images = torch.linspace(0, 1, 8).view(8, 1, 1, 1).expand(8, 24, 32, 3).contiguous()
    m.run_node(images)
    done = [d for e, d in sp.events if e == "c2c.seedvr2.done"]
    assert len(done) == 1
    d = done[0]
    assert d["key"] == "p1:7" and d["count"] == 6 and d["frames"] == list(range(6))
    assert d["before"] is True and d["alpha"] is False and (d["width"], d["height"]) == (128, 96)
    # "before" skips the prepended frames, as their output does
    first_before = _decode_jpeg(sp.strip_frame("p1:7", 0, "before")).mean() / 255
    assert abs(first_before - float(images[2, 0, 0, 0])) < 0.02
    assert sp.strip_frame("p1:7", 6, "after") is None and sp.strip_frame("nope", 0, "after") is None


def test_rgba_output_keeps_an_alpha_strip(sp, monkeypatch):
    m = _standin(monkeypatch, batches=1, per=3, rgba=True)
    sp.install()
    m.run_node(torch.rand(3, 16, 16, 4))
    d = [d for e, d in sp.events if e == "c2c.seedvr2.done"][0]
    assert d["alpha"] is True and sp.strip_frame(d["key"], 2, "alpha") is not None


def test_our_failure_never_breaks_their_run(sp, monkeypatch, caplog):
    m = _standin(monkeypatch, batches=2)
    sp.install()

    def boom(*_a, **_k):
        raise RuntimeError("preview failed")

    monkeypatch.setattr(sp, "_small", boom)
    out = m.run_node(torch.rand(8, 16, 16, 3))
    assert out.shape[0] == 8 and float(out.max()) <= 1.0
    assert not [e for e, _ in sp.events]
    assert "SeedVR2 preview stopped" in caplog.text


def test_changed_signature_is_left_alone(sp, monkeypatch):
    m = _standin(monkeypatch)

    def decode_all_batches(runner, ctx=None, debug=None, on_progress=None):   # renamed parameter
        return ctx

    m.decode_all_batches = decode_all_batches
    assert sp.install() == 0 and m.decode_all_batches is decode_all_batches


def test_only_the_last_three_runs_are_kept(sp, monkeypatch):
    import comfy_execution.utils as cu

    EC = namedtuple("EC", "prompt_id node_id list_index")
    m = _standin(monkeypatch, batches=1, per=2)
    sp.install()
    for k in range(5):
        monkeypatch.setattr(cu, "get_executing_context", lambda k=k: EC(f"p{k}", "7", None))
        m.run_node(torch.rand(2, 16, 16, 3))
    assert list(sp._STRIPS) == ["p2:7", "p3:7", "p4:7"]


@pytest.mark.skipif(not REF.is_dir(), reason="reference clone third_party/ComfyUI-SeedVR2_VideoUpscaler absent")
def test_contract_with_the_real_pack():
    node_src = (REF / "src" / "interfaces" / "video_upscaler.py").read_text(encoding="utf-8")
    phases_src = (REF / "src" / "core" / "generation_phases.py").read_text(encoding="utf-8")
    node_tree, phases_tree = ast.parse(node_src), ast.parse(phases_src)
    imported = {a.name for n in ast.walk(node_tree) if isinstance(n, ast.ImportFrom)
                and (n.module or "").endswith("core.generation_phases") for a in n.names}
    assert {"encode_all_batches", "upscale_all_batches", "decode_all_batches", "postprocess_all_batches"} <= imported
    assert "class SeedVR2VideoUpscaler" in node_src
    defs = {n.name: n for n in phases_tree.body if isinstance(n, ast.FunctionDef)}
    for name in ("decode_all_batches", "postprocess_all_batches"):
        assert "progress_callback" in [a.arg for a in defs[name].args.args], name
    dec = ast.get_source_segment(phases_src, defs["decode_all_batches"])
    assert '"Phase 3: Decoding"' in dec and "ctx['decode_batch_info'].append" in dec and "ctx['final_video']" in dec
    post = ast.get_source_segment(phases_src, defs["postprocess_all_batches"])
    assert "del ctx['input_images']" in post            # why the input is read BEFORE theirs runs
    assert "prepend_frames" in [a.arg for a in defs["postprocess_all_batches"].args.args]
