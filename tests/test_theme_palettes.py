"""The palette engine: four variants, 75 authored keys each, 108 derived.

What this replaced, and why it matters more than it sounds: a sweep had turned
every inline hex in the codebase into a named token - the right move - but it
pasted the SAME 108 values into all three variants. They were byte-identical.
108 of 183 keys, the majority of the palette, did not respond to the theme at
all, and 30 of them were near-black sitting in `latte`, the LIGHT one. Any
widget reaching for `panelDeep` or `scrimDark4` painted a black block onto a
white page, and no care taken in the widget could fix it, because the widget
was doing exactly what it should.

So the relationships are measured off mocha once and re-applied. The tests
below are mostly about the ways that can go wrong, because each one was hit
while building it:

  chips tracking the ground's lightness  -> every chip went dark on OLED
  surfaces MIRRORING on a light theme    -> latte's deep panels went white
  copying mocha's saturation             -> night's panels went grey on indigo
  scaling by the palette's own instead   -> night's panels went vivid
  clamping at the rails                  -> 17 OLED surfaces collapsed onto
                                            one black and elevation vanished

The JS is executed through node, not reimplemented here. A Python copy of the
maths would pass while the shipped file was broken.
"""

from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
from pathlib import Path

import pytest

PACK = Path(__file__).resolve().parents[1]
THEME = PACK / "js" / "_c2c_theme.js"
# The other packs ship their own copy, with local import edits. The palette
# block has to stay in step or a node's panel changes colour depending on
# which pack's module happened to load first.
COPIES = [
    PACK.parent / "ComfyUI-WanNodeExperiments" / "web" / "js" / "_c2c_theme.js",
    PACK.parent / "ComfyUI-WanAnimatePreprocessV2" / "js" / "_c2c_theme.js",
]

VARIANTS = ("night", "mocha", "oled", "latte")
DARK = {"night": True, "mocha": True, "oled": True, "latte": False}

pytestmark = pytest.mark.skipif(shutil.which("node") is None,
                                reason="node is not on PATH")


def palette_block(path: Path) -> str:
    """The engine, lifted out of the shipped file so the test runs what ships."""
    s = path.read_text(encoding="utf-8")
    return s[s.index("const CORE = {"):s.index("\nlet _variant = ")]


@pytest.fixture(scope="module")
def pal(tmp_path_factory) -> dict:
    probe = tmp_path_factory.mktemp("theme") / "probe.mjs"
    probe.write_text(palette_block(THEME) +
                     "\nprocess.stdout.write(JSON.stringify(PALETTES));\n",
                     encoding="utf-8")
    out = subprocess.run([shutil.which("node"), str(probe)],
                         capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr[:800]
    return json.loads(out.stdout)


# ── colour helpers (measurement only - the maths under test lives in JS) ───

def rgb(h: str):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))


def lin(v):
    return v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4


def lum(h):
    r, g, b = (lin(v) for v in rgb(h))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast(a, b):
    lo, hi = sorted((lum(a) + 0.05, lum(b) + 0.05))
    return hi / lo


def hsl(h):
    r, g, b = rgb(h)
    mx, mn = max(r, g, b), min(r, g, b)
    l = (mx + mn) / 2
    if mx == mn:
        return 0.0, 0.0, l
    d = mx - mn
    s = d / (2 - mx - mn) if l > 0.5 else d / (mx + mn)
    return 0.0, s, l


# ── shape ──────────────────────────────────────────────────────────────────

def test_all_four_variants_exist(pal):
    assert set(pal) == set(VARIANTS)


def test_every_variant_carries_the_same_keys(pal):
    """A key missing from one variant is `undefined` in a style string, which
    the browser drops silently - the element keeps whatever it had."""
    ref = set(pal["night"])
    assert len(ref) == 183
    for name in VARIANTS:
        assert set(pal[name]) == ref, sorted(set(pal[name]) ^ ref)[:8]


def test_every_value_is_a_hex_colour(pal):
    for name, palette in pal.items():
        for key, value in palette.items():
            assert re.fullmatch(r"#[0-9a-f]{6}", value), f"{name}.{key} = {value!r}"


def test_the_derive_table_covers_every_shade_that_is_not_authored(pal):
    """183 keys, 75 authored per variant, so 108 rules. A key in neither list
    would simply not exist."""
    assert len(RULES) == 108
    assert not (set(RULES) - set(pal["night"]))


def test_the_derived_shades_are_no_longer_identical_across_variants(pal):
    """The original defect, stated as a test. 108 keys were byte-identical in
    every variant; anything above a handful means the derivation stopped
    running and the literals came back."""
    keys = set(pal["night"])
    same = [k for k in keys if len({pal[v][k] for v in VARIANTS}) == 1]
    # White, black and the two translucency anchors are legitimately shared.
    assert len(same) <= 6, f"{len(same)} keys do not respond to the variant: {sorted(same)[:12]}"


# ── the light theme, which is what was actually broken ─────────────────────
#
# Scoped to the DERIVED keys only. The first version of these tests matched on
# the name and swept up the authored core with it, then failed on
# `oled.scrimDark = #000000` - which is not a derivation fault, it is a
# true-black theme's scrim, chosen on purpose.

def _rules() -> dict:
    """The DERIVE table, read out of the shipped file: key -> kind letter."""
    src = THEME.read_text(encoding="utf-8")
    body = src[src.index("const DERIVE = {"):]
    body = body[:body.index("\n};")]
    return dict(re.findall(r'(\w+):\s*\["(\w)"', body))


RULES = _rules()
SURFACE_KINDS = ("S", "T")


def surfaces(palette):
    return {k: v for k, v in palette.items()
            if RULES.get(k) in SURFACE_KINDS}


def chips(palette):
    return {k: v for k, v in palette.items() if RULES.get(k) == "C"}


def test_the_light_theme_has_no_black_surfaces(pal):
    """41 of 41 were near-black before. A panel background darker than its own
    page is not a theme bug you can style around."""
    bad = {k: v for k, v in surfaces(pal["latte"]).items() if lum(v) < 0.18}
    assert not bad, f"black surfaces on the light theme: {sorted(bad)[:10]}"


def test_surfaces_stay_near_their_own_ground(pal):
    """A surface is the page, one step up or down. If it is not within reach
    of the ground it is not a surface, whatever it is named."""
    for name in VARIANTS:
        bg = pal[name]["bg"]
        for key, value in surfaces(pal[name]).items():
            assert contrast(value, bg) < 6.0, \
                f"{name}.{key} is {contrast(value, bg):.1f}:1 from its own ground"


def test_chips_are_readable_on_their_own_ground(pal):
    """Chips are foreground. Anchoring them to the ground's LIGHTNESS was the
    first attempt and it sent every chip dark on OLED; they carry absolute
    lightness now, mirrored on a light theme."""
    for name in VARIANTS:
        bg = pal[name]["bg"]
        poor = [k for k, v in chips(pal[name]).items() if contrast(v, bg) < 1.8]
        assert len(poor) <= 3, f"{name}: {len(poor)} unreadable chips, e.g. {poor[:6]}"


# ── elevation, which a clamp destroys ──────────────────────────────────────

def test_surfaces_stay_distinct_from_one_another(pal):
    """Clamping at the rails looked fine on average and collapsed seventeen
    OLED surfaces onto one black: a panel, a modal scrim and the page all the
    same colour, so nothing had an edge. The offsets fold back instead."""
    for name in VARIANTS:
        vals = surfaces(pal[name])
        distinct = len(set(vals.values()))
        assert distinct >= len(vals) - 3, \
            f"{name}: only {distinct} distinct values across {len(vals)} surfaces"


def test_nothing_is_pinned_to_pure_black_or_white(pal):
    for name in VARIANTS:
        pinned = [k for k, v in surfaces(pal[name]).items()
                  if v in ("#000000", "#ffffff")]
        assert not pinned, f"{name}: {pinned} sit on the rail"


# ── the dark theme people already use must not move ────────────────────────

def test_mocha_barely_changed(pal):
    """The whole point of measuring the relationships rather than guessing
    them: whoever is on mocha should not notice this happened."""
    import subprocess as sp
    before = sp.run(["git", "show", "HEAD:js/_c2c_theme.js"], cwd=PACK,
                    capture_output=True, text=True, timeout=60)
    if before.returncode != 0:
        pytest.skip("no committed version to compare against")
    old = {}
    seg = before.stdout
    if "    mocha: {" not in seg:
        pytest.skip("the committed version predates the named palettes")
    body = seg[seg.index("    mocha: {"):]
    body = body[:re.search(r"^    \},$", body, re.M).start()]
    old = dict(re.findall(r'(\w+):\s*"(#[0-9a-fA-F]{6})"', body))
    shared = set(old) & set(pal["mocha"])
    if len(shared) < 100:
        pytest.skip("not enough overlap to compare")

    def move(a, b):
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(rgb(a), rgb(b))) / 3) * 100

    moved = [move(old[k], pal["mocha"][k]) for k in shared]
    mean = sum(moved) / len(moved)
    big = sum(1 for m in moved if m > 8)
    assert mean < 5.0, f"mocha shifted {mean:.1f}/100 on average"
    assert big <= 20, f"{big} mocha shades moved visibly"


# ── the night palette, which is the identity ───────────────────────────────

def test_night_is_the_default(pal):
    src = THEME.read_text(encoding="utf-8")
    assert 'let _variant = "night";' in src
    assert "export let C = { ...PALETTES.night };" in src
    assert 'defaultValue: "night"' in src


def test_night_is_offered_in_the_settings(pal):
    src = THEME.read_text(encoding="utf-8")
    block = src[src.index('id: "c2c.theme.variant"'):]
    block = block[:block.index("onChange")]
    for v in VARIANTS:
        assert f'value: "{v}"' in block, f"{v} is not selectable"


def test_night_reads_as_night(pal):
    """Dark, and blue-violet rather than neutral - otherwise it is just
    another grey theme with a different name."""
    night = pal["night"]
    assert lum(night["bg"]) < 0.03, "the ground is not dark"
    _, s, _ = hsl(night["bg"])
    assert s > 0.25, f"the ground is nearly neutral (saturation {s:.2f})"
    r, g, b = rgb(night["bg"])
    assert b > r and b > g, "the ground is not blue-leaning"


def test_night_keeps_its_hue_through_the_surfaces(pal):
    """The failure this catches: copying mocha's absolute saturation left the
    panels grey on an indigo ground, so the identity stopped at the page."""
    tinted = sum(1 for k, v in surfaces(pal["night"]).items() if hsl(v)[1] > 0.10)
    total = len(surfaces(pal["night"]))
    assert tinted > total * 0.7, \
        f"only {tinted}/{total} night surfaces carry any hue"


def test_night_body_text_is_comfortably_readable(pal):
    assert contrast(pal["night"]["fg"], pal["night"]["bg"]) >= 7.0


def test_every_variant_passes_aa_for_body_text(pal):
    for name in VARIANTS:
        c = contrast(pal[name]["fg"], pal[name]["bg"])
        assert c >= 4.5, f"{name} body text is {c:.1f}:1"


# ── the copies in the other packs ──────────────────────────────────────────

@pytest.mark.parametrize("copy", COPIES, ids=lambda p: p.parents[1].name)
def test_the_other_packs_carry_the_same_engine(copy):
    """Three packs ship this file. They diverge deliberately at the imports -
    the other two cannot reach CNP-only modules - but the palette block must
    not drift, or a node's panel changes colour depending on which pack's
    module the browser loaded first."""
    if not copy.exists():
        pytest.skip(f"{copy.parents[1].name} is not in this workspace")
    assert palette_block(copy) == palette_block(THEME), (
        f"{copy.parents[1].name}'s palette block has drifted from CNP's")


@pytest.mark.parametrize("copy", COPIES, ids=lambda p: p.parents[1].name)
def test_the_copies_also_default_to_night(copy):
    if not copy.exists():
        pytest.skip(f"{copy.parents[1].name} is not in this workspace")
    src = copy.read_text(encoding="utf-8")
    assert 'let _variant = "night";' in src
    assert 'defaultValue: "night"' in src


@pytest.mark.parametrize("copy", COPIES, ids=lambda p: p.parents[1].name)
def test_the_copies_keep_their_own_imports(copy):
    """Guard against someone "fixing" the drift with a byte copy: the other
    packs do not ship _c2c_native_offsets.js or c2c_omnibar.js, and importing
    them 404s, which takes down the whole theme chain and every widget with
    it."""
    if not copy.exists():
        pytest.skip(f"{copy.parents[1].name} is not in this workspace")
    src = copy.read_text(encoding="utf-8")
    assert 'import "./c2c_omnibar.js"' not in src, \
        "a CNP-only import was copied in - this 404s and kills the pack's UI"
