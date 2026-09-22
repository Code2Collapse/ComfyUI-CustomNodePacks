"""Per-output naming under one shared version.

The shape this serves: four Video Combines - a mov, an exr sequence, a png
sequence and a masked mov - one Folder Version Incrementer, one version. Each
output needs its own name; none of them may change the version.

CPU-only, no ComfyUI, no disk.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PACK = Path(__file__).resolve().parents[1]
if str(PACK) not in sys.path:
    sys.path.insert(0, str(PACK))

from _version_variant import (  # noqa: E402
    SUFFIX_MODES,
    VersionVariantError,
    apply_variant,
    describe,
    split_path,
)

PREFIX = "shot_a/09-22-2026/v004/shot_a"


# ── the three modes ─────────────────────────────────────────────────────────

def test_filename_mode_keeps_one_folder():
    out = apply_variant(PREFIX, suffix="_masked", mode="filename")
    assert out["filename_prefix"] == "shot_a/09-22-2026/v004/shot_a_masked"


def test_subfolder_mode_nests_inside_the_version():
    """Where an exr sequence should go - thousands of files that must not
    share a directory with anything else."""
    out = apply_variant(PREFIX, suffix="_exr", mode="subfolder",
                        version_string="v004")
    assert out["filename_prefix"] == "shot_a/09-22-2026/v004/exr/shot_a"


def test_folder_mode_makes_a_separate_tree():
    out = apply_variant(PREFIX, suffix="_masked", mode="folder")
    assert out["filename_prefix"] == "shot_a_masked/09-22-2026/v004/shot_a"


def test_the_version_is_never_changed_by_any_mode():
    """This node renames; it must not be able to create, skip or disagree
    about a version. That is the whole reason it does not touch disk."""
    for mode in SUFFIX_MODES:
        out = apply_variant(PREFIX, suffix="_x", mode=mode,
                            version_string="v004")
        assert "v004" in out["filename_prefix"], mode
        assert "v005" not in out["filename_prefix"], mode


def test_an_unknown_mode_is_named():
    with pytest.raises(VersionVariantError, match="Unknown mode"):
        apply_variant(PREFIX, suffix="_x", mode="sideways")


# ── the four-output shape ───────────────────────────────────────────────────

def test_four_outputs_land_under_one_version_with_four_names():
    """The request, end to end."""
    outs = [
        apply_variant(PREFIX, suffix="", mode="filename"),
        apply_variant(PREFIX, suffix="_exr", mode="subfolder", version_string="v004"),
        apply_variant(PREFIX, suffix="_png", mode="subfolder", version_string="v004"),
        apply_variant(PREFIX, suffix="_masked", mode="filename"),
    ]
    paths = [o["filename_prefix"] for o in outs]
    assert len(set(paths)) == 4, f"two outputs would collide: {paths}"
    assert all("09-22-2026/v004" in p for p in paths), paths


def test_an_untagged_output_is_left_alone():
    """One output usually needs no tag - the main mov."""
    out = apply_variant(PREFIX, suffix="", prefix="", mode="subfolder")
    assert out["filename_prefix"] == PREFIX


# ── prefixes ────────────────────────────────────────────────────────────────

def test_a_prefix_goes_before_the_name():
    """A prefix faked with a suffix sorts in the wrong order, which is the
    entire reason to use one."""
    out = apply_variant(PREFIX, prefix="FINAL_", mode="filename")
    assert out["basename"] == "FINAL_shot_a"


def test_a_prefix_and_suffix_can_both_apply():
    out = apply_variant(PREFIX, prefix="FINAL_", suffix="_v2", mode="filename")
    assert out["basename"] == "FINAL_shot_a_v2"


# ── subfolder placement ─────────────────────────────────────────────────────

def test_a_separator_only_tag_is_refused():
    """It would make a folder with no name."""
    with pytest.raises(VersionVariantError, match="no name"):
        apply_variant(PREFIX, suffix="__", mode="subfolder")


def test_a_leading_separator_is_a_join_hint_not_part_of_the_name():
    out = apply_variant(PREFIX, suffix="_masked", mode="subfolder",
                        version_string="v004")
    assert "/masked/" in out["filename_prefix"]
    assert "/_masked/" not in out["filename_prefix"]


def test_without_a_version_string_the_folder_goes_on_the_end():
    """Same result whenever the incrementer was not already nesting."""
    out = apply_variant(PREFIX, suffix="_masked", mode="subfolder")
    assert out["filename_prefix"] == "shot_a/09-22-2026/v004/masked/shot_a"


def test_a_subfolder_lands_after_the_version_even_when_already_nested():
    """The incrementer's own suffix_mode may already have nested something;
    the tag still belongs directly under the version, not at the bottom."""
    nested = "shot_a/09-22-2026/v004/wan/shot_a"
    out = apply_variant(nested, suffix="_masked", mode="subfolder",
                        version_string="v004")
    assert out["filename_prefix"] == "shot_a/09-22-2026/v004/masked/wan/shot_a"


# ── folder mode ─────────────────────────────────────────────────────────────

def test_folder_mode_tags_the_top_folder_not_a_middle_one():
    """The top folder is the one the incrementer treats as the shot."""
    out = apply_variant(PREFIX, suffix="_masked", mode="folder")
    assert out["filename_prefix"].startswith("shot_a_masked/")
    assert out["folder_name"] == "shot_a_masked"


def test_folder_mode_on_a_bare_name_is_refused_with_the_alternative():
    with pytest.raises(VersionVariantError, match="mode 'filename'"):
        apply_variant("justaname", suffix="_x", mode="folder")


# ── path handling ───────────────────────────────────────────────────────────

def test_windows_separators_survive():
    """Rebuilding with the wrong separator makes ONE FILE whose name
    contains a backslash, instead of a directory - on the other OS, and
    silently."""
    sep = chr(92)
    win = sep.join(["shot_a", "09-22-2026", "v004", "shot_a"])
    out = apply_variant(win, suffix="_masked", mode="filename")
    assert out["filename_prefix"] == win + "_masked"
    assert "/" not in out["filename_prefix"], "a forward slash crept in"


def test_windows_separators_survive_a_subfolder_insert():
    sep = chr(92)
    win = sep.join(["shot_a", "09-22-2026", "v004", "shot_a"])
    out = apply_variant(win, suffix="_exr", mode="subfolder",
                        version_string="v004")
    assert out["filename_prefix"] == sep.join(
        ["shot_a", "09-22-2026", "v004", "exr", "shot_a"])
    assert "/" not in out["filename_prefix"]


def test_an_empty_prefix_is_named_with_what_to_connect():
    with pytest.raises(VersionVariantError, match="Connect this node"):
        split_path("")


def test_an_extension_is_added_only_to_the_filename_output():
    out = apply_variant(PREFIX, suffix="_masked", mode="filename",
                        extension="mov")
    assert out["output_filename"].endswith(".mov")
    assert not out["filename_prefix"].endswith(".mov"), \
        "the prefix must stay an extension-free prefix for save nodes"


def test_a_dotted_extension_is_not_double_dotted():
    out = apply_variant(PREFIX, mode="filename", suffix="_x", extension=".mov")
    assert out["output_filename"].endswith(".mov")
    assert not out["output_filename"].endswith("..mov")


def test_the_subfolder_output_has_no_basename_on_it():
    out = apply_variant(PREFIX, suffix="_exr", mode="subfolder",
                        version_string="v004")
    assert out["subfolder_path"] == "shot_a/09-22-2026/v004/exr"


# ── the report ──────────────────────────────────────────────────────────────

def test_the_report_states_that_the_version_is_not_changed_here():
    out = apply_variant(PREFIX, suffix="_masked", mode="filename",
                        version_string="v004")
    text = describe(out, mode="filename", prefix="", suffix="_masked",
                    version_string="v004")
    assert "NOT" in text and "renames" in text


def test_the_report_explains_an_untagged_output():
    out = apply_variant(PREFIX, mode="subfolder")
    text = describe(out, mode="subfolder", prefix="", suffix="",
                    version_string="v004")
    assert "No prefix or suffix" in text


def test_the_report_explains_that_folder_mode_still_shares_the_version():
    """The surprising one: a separate top-level tree that is still on the
    same version."""
    out = apply_variant(PREFIX, suffix="_masked", mode="folder")
    text = describe(out, mode="folder", prefix="", suffix="_masked",
                    version_string="v004")
    assert "separate top-level tree" in text
    assert "shares the version" in text


def test_the_report_shows_the_resulting_path():
    out = apply_variant(PREFIX, suffix="_exr", mode="subfolder",
                        version_string="v004")
    text = describe(out, mode="subfolder", prefix="", suffix="_exr",
                    version_string="v004")
    assert out["filename_prefix"] in text
