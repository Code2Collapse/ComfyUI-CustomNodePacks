"""Which of MaskOps' 68 widgets are shown for a given pair of models.

MaskOpsMEC has 68 parameters and most belong to one backend or one optional
feature. All 68 on screen is worse than clutter: a visible control that is
ignored teaches the wrong thing about the node, and you spend an afternoon
adjusting `trimap_dilate` on a matter that has no trimap.

The two properties that matter most are at the bottom: every widget named in
the spec must really exist on the node (or the rule silently does nothing),
and a backend that has NOT declared must get everything shown (silence is not
a claim).

CPU-only; skips cleanly when the registry will not load.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

PACK = Path(__file__).resolve().parents[1]
if str(PACK) not in sys.path:
    sys.path.insert(0, str(PACK))

# Nothing is done to sys.path or to ComfyUI's modules here. The pack conftest
# already arranges the environment, and every other mask test imports plainly.
# Two attempts at "helping" broke this file: pushing ComfyUI's root on first
# made `nodes` resolve to ComfyUI's own nodes.py, and importing `server` early
# perturbed the order enough that folder_paths arrived without a __file__.
vis = pytest.importorskip("nodes.mask_matting._visibility",
                          reason="mask_matting registry unavailable")


def shown(seg: str, mat: str):
    return vis.widgets_for(seg, mat)


# ── the point of the whole thing ────────────────────────────────────────────

def _defaults(cls):
    """Every widget's default value, as the node would open."""
    spec = cls.INPUT_TYPES()
    fields = {**(spec.get("required") or {}), **(spec.get("optional") or {})}
    out = {}
    for name, sp in fields.items():
        opts = sp[1] if isinstance(sp, (list, tuple)) and len(sp) > 1 \
            and isinstance(sp[1], dict) else {}
        if "default" in opts:
            out[name] = opts["default"]
        elif isinstance(sp, (list, tuple)) and isinstance(sp[0], (list, tuple)) \
                and sp[0]:
            out[name] = sp[0][0]          # a combo opens on its first entry
    return out


def _maskops():
    node = pytest.importorskip("nodes.mask_matting.node",
                               reason="MaskOps unavailable")
    cls = getattr(node, "MaskOpsMEC", None)
    if cls is None:
        pytest.skip("MaskOpsMEC not found")
    return cls


def test_a_declared_pair_hides_a_real_share_of_the_widgets():
    """Measured at DEFAULT settings, which is the number the user sees.

    The earlier version measured the candidate set instead, and so reported a
    far smaller number than the node has ever shown."""
    cls = _maskops()
    vals = _defaults(cls)
    on_screen = vis.visible("sam3.1", "vitmatte", vals)
    assert on_screen != ["*"], "sam3.1 should have declared its parameters"
    assert 30 <= len(on_screen) <= 55, (
        f"{len(on_screen)} on screen — the rule is either doing nothing or "
        "hiding too much")


def test_switching_a_feature_on_reveals_its_controls():
    """THE regression. Each of these had a TOGGLES entry and no way into the
    candidate set, so its switch revealed nothing. A toggle rule RESTRICTS
    when a widget shows; it never grants."""
    cls = _maskops()
    off = _defaults(cls)
    on = dict(off, enable_luma_key=True, enable_advanced_trimap=True,
              enable_diagnose=True, robust_propagation=True,
              despill="green", edge_mode="hard+feather",
              post_refine="guided", lightwrap_strength=0.5)
    before = set(vis.visible("sam3.1", "vitmatte", off))
    after = set(vis.visible("sam3.1", "vitmatte", on))
    for name in ("luma_low", "trimap_smooth", "diag_ring_width",
                 "robust_blend_alpha", "despill_strength", "edge_feather",
                 "refine_radius", "lightwrap_radius"):
        assert name not in before, f"{name} shown while its switch is off"
        assert name in after, (
            f"{name} stays hidden with its switch ON — its toggle rule can "
            "never fire because nothing puts it in the candidate set")


def test_every_toggled_widget_is_reachable():
    """The same bug as a property, so a new toggle cannot repeat it."""
    cand = set(vis.widgets_for("sam3.1", "vitmatte"))
    unreachable = sorted(set(vis.TOGGLES) - cand)
    assert not unreachable, (
        "These widgets have a toggle rule but are in no candidate list, so "
        "they can never be shown at all:\n  " + "\n  ".join(unreachable))


def test_the_despill_rule_uses_the_widgets_real_off_value():
    """`despill` turns off with "off"; gating on "none" left its controls on
    screen permanently and read as the rule not running."""
    cls = _maskops()
    spec = cls.INPUT_TYPES()
    fields = {**(spec.get("required") or {}), **(spec.get("optional") or {})}
    for widget, rule in vis.TOGGLES.items():
        if not isinstance(rule, tuple):
            continue
        gate, off_value = rule
        sp = fields.get(gate)
        if not (isinstance(sp, (list, tuple)) and sp
                and isinstance(sp[0], (list, tuple))):
            continue                      # not a combo: nothing to check
        assert off_value in sp[0], (
            f"{widget} is gated on {gate} != {off_value!r}, but {gate} has no "
            f"such choice ({list(sp[0])}) — the rule can never fire")


def test_a_salient_segmenter_shows_less_than_a_prompted_one():
    """BiRefNet takes no prompt at all, so its text box and point fields have
    no business being on screen."""
    salient = shown("birefnet", "vitmatte")
    prompted = shown("sam3.1", "vitmatte")
    assert salient != ["*"] and prompted != ["*"]
    assert len(salient) < len(prompted)


def test_a_backend_that_cannot_take_text_does_not_show_a_text_box():
    """Typing in a box that is ignored is the single most confusing thing
    this node can do."""
    assert "text_prompt" not in shown("birefnet", "vitmatte")
    assert "text_prompt" in shown("sam3.1", "vitmatte")


def test_a_still_segmenter_does_not_show_the_tracking_window():
    w = shown("sam3.1", "vitmatte")
    for name in ("memory_size", "max_frames_to_track", "tracking_direction"):
        assert name not in w, f"{name} shown for a non-video backend"


def test_a_video_tracker_does_show_the_tracking_window():
    w = shown("sec", "rvm")
    assert w != ["*"]
    for name in ("memory_size", "start_frame", "end_frame",
                 "tracking_direction"):
        assert name in w, f"{name} missing for a video backend"


def test_a_trimap_widget_only_appears_for_a_matter_that_uses_one():
    """ViTMatte builds a trimap; RVM does not."""
    assert "trimap_dilate" in shown("sam3.1", "vitmatte")
    assert "trimap_dilate" not in shown("sam3.1", "rvm")


# ── silence is not a claim ──────────────────────────────────────────────────

def test_an_undeclared_backend_shows_everything():
    """PARAMS of None means the backend has not said. Hiding a control that
    might matter is worse than showing one that does not."""
    class Undeclared:
        KEY = "made-up"
        PARAMS = None
        SUPPORTS_MODES = {"auto"}

    real = vis._seg_classes
    vis._seg_classes = lambda: {"made-up": Undeclared}
    try:
        assert shown("made-up", "vitmatte") == ["*"]
    finally:
        vis._seg_classes = real


def test_an_empty_declaration_is_not_the_same_as_silence():
    """() is a real claim — 'I read nothing extra' — and must show the core
    only. Collapsing the two would make silence look like a claim."""
    class ReadsNothing:
        KEY = "quiet"
        PARAMS = ()
        SUPPORTS_MODES = {"auto"}
        NEEDS_VIDEO = False

    real = vis._seg_classes
    vis._seg_classes = lambda: {"quiet": ReadsNothing}
    try:
        w = shown("quiet", "vitmatte")
        assert w != ["*"]
        assert "text_prompt" not in w
        assert "image" in w
    finally:
        vis._seg_classes = real


def test_an_unknown_segmenter_falls_back_to_the_core():
    w = shown("not-a-real-backend", "vitmatte")
    assert "image" in w and "segmenter" in w


# ── what must never be folded ───────────────────────────────────────────────

def test_every_socket_survives_every_combination():
    """A hidden input that is WIRED is a connection the user can neither see
    nor undo."""
    for seg in ("sam3.1", "birefnet", "sec"):
        for mat in ("vitmatte", "rvm"):
            w = shown(seg, mat)
            if w == ["*"]:
                continue
            for s in vis.SOCKETS:
                assert s in w, f"{s} folded away for {seg}+{mat}"


def test_the_model_pickers_are_always_shown():
    """Hiding the control that decides the rules would be unrecoverable."""
    for seg in ("sam3.1", "birefnet", "sec"):
        w = shown(seg, "vitmatte")
        if w == ["*"]:
            continue
        assert "segmenter" in w and "matter" in w


# ── the spec the front-end receives ─────────────────────────────────────────

def test_the_spec_carries_every_registered_backend():
    spec = vis.build_spec()
    from nodes.mask_matting.segmenters import all_segmenters
    from nodes.mask_matting.matters import all_matters
    assert set(spec["segmenters"]) == set(all_segmenters())
    assert set(spec["matters"]) == set(all_matters())


def test_the_spec_reports_availability_so_the_ui_can_say_so():
    """A backend whose weights are missing should be visibly unavailable, not
    silently selectable and then a failure at run time."""
    spec = vis.build_spec()
    for info in spec["segmenters"].values():
        assert info["status"] in ("ready", "experimental", "missing-deps",
                                  "unknown")


def test_the_spec_marks_which_matters_are_temporal():
    """A per-frame matter on video flickers — the commonest complaint about
    an otherwise good matte — so the UI has to be able to warn."""
    spec = vis.build_spec()
    assert spec["matters"]["rvm"]["temporal"] is True
    assert spec["matters"]["vitmatte"]["temporal"] is False


# ── THE parity check ────────────────────────────────────────────────────────

def test_every_widget_named_in_the_spec_exists_on_the_node():
    """A rule naming a widget that does not exist does nothing, silently, and
    looks like it works. This is the check that catches a rename."""
    node = pytest.importorskip(
        "nodes.mask_matting.node", reason="MaskOps unavailable")
    cls = None
    for name in dir(node):
        obj = getattr(node, name)
        if isinstance(obj, type) and hasattr(obj, "INPUT_TYPES") \
                and name.startswith("MaskOps"):
            cls = obj
            break
    if cls is None:
        pytest.skip("MaskOpsMEC not found")

    spec_in = cls.INPUT_TYPES()
    real = set(spec_in.get("required") or {}) | set(spec_in.get("optional") or {})

    named = set(vis.ALWAYS) | set(vis.SOCKETS) | set(vis.TOGGLES) | \
        set(vis.VIDEO_WINDOW) | set(vis.FEATURE)
    for names in vis.BY_MODE.values():
        named |= set(names)
    for rule in vis.TOGGLES.values():
        named.add(rule[0] if isinstance(rule, tuple) else rule)

    missing = sorted(named - real)
    assert not missing, (
        "These widgets are named by a visibility rule but do not exist on "
        "MaskOpsMEC, so the rule silently does nothing:\n  "
        + "\n  ".join(missing))


def test_no_widget_on_the_node_is_unreachable():
    """The mirror of the parity check. A widget in no list can never be shown
    for any combination — a control the user simply cannot get to."""
    cls = _maskops()
    spec = cls.INPUT_TYPES()
    real = set(spec.get("required") or {}) | set(spec.get("optional") or {})

    named = set(vis.ALWAYS) | set(vis.SOCKETS) | set(vis.FEATURE) | \
        set(vis.TOGGLES) | set(vis.VIDEO_WINDOW)
    for names in vis.BY_MODE.values():
        named |= set(names)
    orphans = sorted(real - named)
    assert not orphans, (
        "These widgets exist on MaskOpsMEC but appear in no visibility list, "
        "so they are hidden for every combination:\n  " + "\n  ".join(orphans))


def test_a_backend_param_names_a_real_widget_or_is_declared_unexposed():
    """A backend's PARAMS are ITS OWN kwargs, not widget names — `bbox` covers
    three widgets and `multimask` has none. An unmapped, undeclared name is a
    rule that silently does nothing."""
    cls = _maskops()
    spec = cls.INPUT_TYPES()
    real = set(spec.get("required") or {}) | set(spec.get("optional") or {})

    bad = []
    for kind, classes in (("segmenter", vis._seg_classes()),
                          ("matter", vis._mat_classes())):
        for key, backend in classes.items():
            for name in (getattr(backend, "PARAMS", None) or ()):
                mapped = vis.PARAM_TO_WIDGET.get(name)
                if mapped:
                    gone = [w for w in mapped if w not in real]
                    if gone:
                        bad.append(f"{kind} {key}: {name} -> {gone} (no such widget)")
                elif name not in vis.NOT_EXPOSED and name not in real:
                    bad.append(f"{kind} {key}: {name} (unmapped, not a widget)")
    assert not bad, (
        "Add each to PARAM_TO_WIDGET (it has a widget under another name) or "
        "to NOT_EXPOSED (the node fixes it):\n  " + "\n  ".join(sorted(bad)))


def test_the_report_warns_about_a_per_frame_matter_on_video():
    text = vis.describe("sec", "vitmatte", 68)
    assert "flickers" in text


def test_the_report_explains_an_undeclared_backend():
    class Undeclared:
        KEY = "x"
        PARAMS = None
        SUPPORTS_MODES = set()

    real = vis._seg_classes
    vis._seg_classes = lambda: {"x": Undeclared}
    try:
        assert "has not declared" in vis.describe("x", "vitmatte", 68)
    finally:
        vis._seg_classes = real


# ── ViTMatte tiling widgets (cascade + ONYX matte_tile) ─────────────────────

def _visible(seg: str, mat: str, pipeline: str):
    cls = _maskops()
    vals = dict(_defaults(cls), pipeline=pipeline)
    return set(vis.visible(seg, mat, vals))


def test_matte_tile_shows_on_onyx_and_cascade_vitmatte():
    assert "matte_tile" in _visible("sam3.1", "vitmatte", "onyx")
    assert "matte_tile" in _visible("sam3.1", "vitmatte", "cascade (legacy)")


def test_matte_overlap_and_batch_show_on_cascade_vitmatte_only():
    cascade = _visible("sam3.1", "vitmatte", "cascade (legacy)")
    onyx = _visible("sam3.1", "vitmatte", "onyx")
    assert "matte_overlap" in cascade
    assert "matte_tile_batch" in cascade
    assert "matte_overlap" not in onyx
    assert "matte_tile_batch" not in onyx


def test_vitmatte_tiling_widgets_hidden_for_other_matters():
    shown = _visible("sam3.1", "rvm", "cascade (legacy)")
    for name in ("matte_tile", "matte_overlap", "matte_tile_batch"):
        assert name not in shown
