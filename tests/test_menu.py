"""One menu root for every Code2Collapse pack - "🐺 C2C/<pack>/<family>".

_c2c_menu.py ships as an identical copy in every pack (packs cannot import
each other: a cross-pack import fails on a standalone install), and each
pack's __init__.py applies it once at registration. These tests hold the
copies in step, pin the mapping rules, and check every pack is wired.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pytest

WORKSPACE = Path(__file__).resolve().parents[2]
CANONICAL = WORKSPACE / "ComfyUI-CustomNodePacks" / "_c2c_menu.py"

# pack folder -> the <pack> label its nodes appear under
PACKS = {
    "ComfyUI-CustomNodePacks": "\U0001F9F0 Core",
    "ComfyUI-NukeMaxNodes": "\U0001F3AC NukeMax",
    "ComfyUI-MiniMaxSuite": "\U0001F39E️ MiniMax H3",
    "ComfyUI-WanNodeExperiments": "\U0001F30A Wan Experiments",
    "ComfyUI-WanAnimatePreprocessV2": "\U0001F9CD Wan Animate Preprocess",
    "ComfyUI-GLM_Image": "\U0001F5BC️ GLM Image",
    "ComfyUI-WanAnimalPreprocessor": "\U0001F43E Wan Animal Preprocess",
}


def _load_menu():
    spec = importlib.util.spec_from_file_location("_c2c_menu_under_test", CANONICAL)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


menu = _load_menu()
ROOT = menu.MENU_ROOT


def _present(pack):
    if not (WORKSPACE / pack).exists():
        pytest.skip(f"{pack} is not in this workspace")


# ── the copies ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("pack", sorted(PACKS))
def test_every_pack_ships_the_same_menu(pack):
    _present(pack)
    path = WORKSPACE / pack / "_c2c_menu.py"
    assert path.exists(), f"{pack} has no _c2c_menu.py - its nodes stay outside the root"
    assert path.read_bytes() == CANONICAL.read_bytes(), f"{pack}'s _c2c_menu.py has drifted"


@pytest.mark.parametrize("pack", sorted(PACKS))
def test_every_pack_applies_it_under_its_own_label(pack):
    """Wired in __init__.py, guarded, and after the mappings are final - a
    placement that runs before the last .update() misses those nodes."""
    _present(pack)
    src = (WORKSPACE / pack / "__init__.py").read_text(encoding="utf-8")
    call = re.search(r"(rebrand_v1 as _c2c_menu_rebrand|rebrand_v3\()", src)
    assert call, f"{pack} never applies the menu root"
    # the sources spell the label as escapes ("\U0001F9F0 Core"); compare case-blind
    label_literal = PACKS[pack].encode("unicode_escape").decode("ascii")
    assert label_literal.upper() in src.upper(), f"{pack} uses a different label"
    before = src[: call.start()]
    after = src[call.end():]
    assert "try:" in before[-400:], f"{pack}: the menu call is not guarded"
    if "rebrand_v1" in call.group(0):
        assert "NODE_CLASS_MAPPINGS.update(" not in after and not re.search(
            r"^NODE_CLASS_MAPPINGS\s*=", after, re.M
        ), f"{pack}: mappings change after the menu is applied"


# ── the mapping rules ───────────────────────────────────────────────────────

def test_legacy_roots_are_stripped_repeatedly():
    assert menu.menu_category("C2C/MEC/Mask", "Core", strip=("C2C", "MEC")) == f"{ROOT}/Core/Mask"


def test_a_whole_path_rename_wins_over_a_head_rename():
    rename = {"KJNodes/masking": "Mask", "KJNodes": "KJ"}
    assert menu.menu_category("KJNodes/masking", "W", rename=rename) == f"{ROOT}/W/Mask"
    assert menu.menu_category("KJNodes/other", "W", rename=rename) == f"{ROOT}/W/KJ/other"


def test_an_empty_rename_drops_the_segment():
    assert menu.menu_category("Legacy/Color", "P", rename={"Legacy": ""}) == f"{ROOT}/P/Color"


def test_a_category_that_is_only_legacy_root_lands_at_the_pack():
    assert menu.menu_category("WanAnimalPreprocess", "A", strip=("WanAnimalPreprocess",)) == f"{ROOT}/A"
    assert menu.menu_category(None, "A") == f"{ROOT}/A"
    assert menu.menu_category("", "A") == f"{ROOT}/A"


def test_segments_are_trimmed_and_empties_dropped():
    assert menu.menu_category(" NukeMax / Color //", "N", strip=("NukeMax",)) == f"{ROOT}/N/Color"


def test_applying_it_twice_changes_nothing():
    once = menu.menu_category("MEC/Mask", "Core", strip=("MEC",))
    assert menu.menu_category(once, "Other", strip=("MEC",), rename={"Mask": "X"}) == once


# ── V1 packs ────────────────────────────────────────────────────────────────

def test_rebrand_v1_rewrites_each_class_once():
    class A:
        CATEGORY = "MEC/Mask"

    class B:
        CATEGORY = "Masking"

    # an alias: two ids for one class must not map it twice
    mappings = {"A": A, "A_alias": A, "B": B}
    n = menu.rebrand_v1(mappings, "Core", strip=("MEC",), rename={"Masking": "Mask"})
    assert n == 2
    assert A.CATEGORY == B.CATEGORY == f"{ROOT}/Core/Mask"
    assert menu.rebrand_v1(mappings, "Core", strip=("MEC",)) == 0


def test_rebrand_v1_never_raises_on_a_class_that_refuses_the_attribute():
    class Frozen(type):
        def __setattr__(cls, name, value):
            raise AttributeError("frozen")

    class Locked(metaclass=Frozen):
        CATEGORY = "MEC/Mask"

    class Fine:
        CATEGORY = "MEC/Paint"

    assert menu.rebrand_v1({"L": Locked, "F": Fine}, "Core", strip=("MEC",)) == 1
    assert Locked.CATEGORY == "MEC/Mask"
    assert Fine.CATEGORY == f"{ROOT}/Core/Paint"


# ── V3 packs ────────────────────────────────────────────────────────────────

class _Schema:
    def __init__(self, category):
        self.category = category


def _v3(category):
    calls = []

    class Node:
        _CATEGORY = "stale cache"

        @classmethod
        def define_schema(cls):
            calls.append(cls)
            return _Schema(category)

    return Node, calls


def test_rebrand_v3_maps_every_fresh_schema_and_clears_the_cache():
    """ComfyUI calls define_schema afresh per /object_info and caches the
    category in _CATEGORY - so the schema is wrapped and the cache cleared."""
    Node, calls = _v3("MiniMax H3/Finish")
    assert menu.rebrand_v3([Node], "MMX", strip=("MiniMax H3",)) == 1
    assert Node._CATEGORY is None
    assert Node.define_schema().category == f"{ROOT}/MMX/Finish"
    assert Node.define_schema().category == f"{ROOT}/MMX/Finish"
    assert calls == [Node, Node]


def test_rebrand_v3_wraps_only_once():
    Node, _ = _v3("MiniMax H3/Finish")
    menu.rebrand_v3([Node], "MMX", strip=("MiniMax H3",))
    assert menu.rebrand_v3([Node], "MMX", strip=("MiniMax H3",)) == 0
    assert Node.define_schema().category == f"{ROOT}/MMX/Finish"


def test_rebrand_v3_skips_classes_without_a_schema():
    class NotANode:
        pass

    assert menu.rebrand_v3([NotANode], "MMX") == 0
