"""Tests for folder_incrementer sanitization, override, and reserve_version."""

import sys
import types
import tempfile
from pathlib import Path

import pytest

# Stub ComfyUI modules
for mod_name in ("folder_paths", "comfy", "comfy.utils", "server"):
    if mod_name not in sys.modules:
        stub = types.ModuleType(mod_name)
        stub.__path__ = []
        sys.modules[mod_name] = stub

from folder_incrementer import (  # noqa: E402
    FolderIncrementer,
    FolderIncrementerSet,
    _sanitize_folder_name,
)


# ─────────────────────────── Sanitization ──────────────────────────────


class TestSanitize:
    def test_strips_illegal_windows_chars(self):
        assert _sanitize_folder_name('a<b>c:d"e/f\\g|h?i*j') == "a_b_c_d_e_f_g_h_i_j"

    def test_strips_trailing_dots_and_spaces(self):
        assert _sanitize_folder_name("name. . ") == "name"

    def test_caps_length(self):
        long = "x" * 250
        out = _sanitize_folder_name(long, max_length=100)
        assert len(out) <= 100

    def test_reserved_windows_name_prefixed(self):
        assert _sanitize_folder_name("CON") == "_CON"
        assert _sanitize_folder_name("com1") == "_com1"

    def test_empty_returns_fallback(self):
        assert _sanitize_folder_name("", fallback="default") == "default"
        assert _sanitize_folder_name("...", fallback="x") == "x"

    def test_passes_through_safe_name(self):
        assert _sanitize_folder_name("SC_30_SHT50") == "SC_30_SHT50"


# ─────────────────────────── increment() ──────────────────────────────


class TestIncrement:
    def test_no_disk_writes_when_reserve_version_false(self, tmp_path):
        """Default behaviour: NO directory should be created on disk."""
        node = FolderIncrementer()
        out = node.increment(
            source_filename="myrender.png",
            base_path=str(tmp_path),
            reserve_version=False,
        )
        # Nothing under tmp_path should exist yet (no children created).
        assert not any(tmp_path.iterdir())
        # Output paths should still be sensible.
        version_string, version_num, folder_name, sub, prefix, output, *_ = out
        assert version_string == "v001"
        assert version_num == 1
        assert folder_name == "myrender"

    def test_reserve_version_true_creates_dir_with_marker(self, tmp_path):
        node = FolderIncrementer()
        out = node.increment(
            source_filename="render.exr",
            base_path=str(tmp_path),
            reserve_version=True,
        )
        version_string, version_num, folder_name, *_ = out
        # The created dir must exist
        date_dirs = list((tmp_path / folder_name).iterdir())
        assert len(date_dirs) == 1
        ver_dir = date_dirs[0] / version_string
        assert ver_dir.is_dir()
        assert (ver_dir / ".reserved").is_file()

    def test_folder_name_override_wins(self, tmp_path):
        node = FolderIncrementer()
        _, _, folder_name, *_ = node.increment(
            source_filename="anything.png",
            base_path=str(tmp_path),
            folder_name_override="custom_shot",
        )
        assert folder_name == "custom_shot"

    def test_folder_name_override_sanitized(self, tmp_path):
        node = FolderIncrementer()
        _, _, folder_name, *_ = node.increment(
            source_filename="anything.png",
            base_path=str(tmp_path),
            folder_name_override='bad<name>:"foo"',
        )
        assert "<" not in folder_name
        assert ">" not in folder_name
        assert ":" not in folder_name
        assert '"' not in folder_name

    def test_windows_backslash_input_filename(self, tmp_path):
        """A Windows-style absolute path passed as source_filename should
        still extract just the stem cross-platform."""
        node = FolderIncrementer()
        _, _, folder_name, *_ = node.increment(
            source_filename=r"C:\Users\artist\renders\shot_010.png",
            base_path=str(tmp_path),
        )
        assert folder_name == "shot_010"

    def test_illegal_chars_in_source_filename_sanitized(self, tmp_path):
        node = FolderIncrementer()
        _, _, folder_name, *_ = node.increment(
            source_filename='bad<>name.png',
            base_path=str(tmp_path),
        )
        assert "<" not in folder_name and ">" not in folder_name

    def test_version_increments_after_reservation(self, tmp_path):
        node = FolderIncrementer()
        a = node.increment(source_filename="x.png", base_path=str(tmp_path),
                           reserve_version=True)
        b = node.increment(source_filename="x.png", base_path=str(tmp_path),
                           reserve_version=True)
        assert a[1] == 1
        assert b[1] == 2

    def test_path_style_linux_uses_forward_slash(self, tmp_path):
        node = FolderIncrementer()
        _, _, _, sub, *_ = node.increment(
            source_filename="x.png",
            base_path=str(tmp_path),
            path_style="linux",
        )
        assert "/" in sub
        assert "\\" not in sub

    def test_path_style_windows_uses_backslash(self, tmp_path):
        node = FolderIncrementer()
        _, _, _, sub, *_ = node.increment(
            source_filename="x.png",
            base_path=str(tmp_path),
            path_style="windows",
        )
        assert "\\" in sub


class TestSetVersion:
    def test_set_uses_sanitized_label(self, tmp_path):
        node = FolderIncrementerSet()
        status, next_ver = node.set_version(
            label='bad<>label',
            value=2,
            base_path=str(tmp_path),
        )
        # No illegal char on disk
        for child in tmp_path.iterdir():
            assert "<" not in child.name and ">" not in child.name
        assert next_ver == 3


# ── numbered_still_mode: frame index vs shot number ──────────────────────────
#
# The 2026-08-01 frame-sequence fix stripped ANY trailing 2-8 digit token from a
# still stem so a numbered sequence grouped under one folder. That also ate shot
# numbers: shot_010 / shot_020 / shot_030 all became "shot", silently mixing
# unrelated plates into a single sequence — and it contradicted the sibling
# version regex, which was deliberately anchored NOT to eat a shot number.
#
# Resolved 2026-08-29 by digit width, defaulting toward preserving identity:
# false grouping is destructive and invisible, false splitting is merely visible
# and inconvenient, so `auto` errs toward keeping the number.

class TestNumberedStillMode:
    def _folder(self, tmp_path, source, **over):
        node = FolderIncrementer()
        _, _, folder_name, *_ = node.increment(
            source_filename=source, base_path=str(tmp_path), **over
        )
        return folder_name

    def test_auto_keeps_short_shot_number(self, tmp_path):
        # INVARIANT: <=3 bare digits reads as a shot number and is preserved.
        assert self._folder(tmp_path, r"C:\Users\artist\renders\shot_010.png") == "shot_010"

    def test_auto_keeps_shot_numbers_distinct(self, tmp_path):
        # INVARIANT: the destructive case — sibling shots must NOT collapse together.
        names = {
            self._folder(tmp_path, f"shot_{n}.png") for n in ("010", "020", "030")
        }
        assert names == {"shot_010", "shot_020", "shot_030"}, (
            f"shot numbers collapsed into {names} — unrelated plates would share a folder"
        )

    def test_auto_strips_four_digit_frame_index(self, tmp_path):
        # INVARIANT: >=4 bare digits reads as a frame index and is stripped.
        assert self._folder(tmp_path, "plate.0001.png") == "plate"

    def test_auto_strips_frame_index_but_keeps_shot_number(self, tmp_path):
        # INVARIANT: the combined real-world case — frame token goes, shot stays.
        assert self._folder(tmp_path, "shot_010.1001.exr") == "shot_010"

    def test_hash_token_stripped_in_every_mode(self, tmp_path):
        # INVARIANT: #### is explicit frame syntax and never part of a real name.
        for mode in ("auto", "sequence", "identity"):
            assert self._folder(
                tmp_path, "render_####.exr", numbered_still_mode=mode
            ) == "render", f"#### survived in mode {mode}"

    def test_printf_token_stripped_in_every_mode(self, tmp_path):
        # INVARIANT: %04d likewise.
        for mode in ("auto", "sequence", "identity"):
            assert self._folder(
                tmp_path, "render.%04d.exr", numbered_still_mode=mode
            ) == "render", f"%04d survived in mode {mode}"

    def test_sequence_mode_restores_pre_fix_behaviour(self, tmp_path):
        # INVARIANT: `sequence` is the escape hatch for bare-numbered frames —
        # it must still strip a short digit run, which is what `auto` refuses to do.
        assert self._folder(
            tmp_path, "shot_010.png", numbered_still_mode="sequence"
        ) == "shot"

    def test_identity_mode_never_strips_bare_digits(self, tmp_path):
        # INVARIANT: `identity` keeps even a 4-digit run, for shot_0100 naming.
        assert self._folder(
            tmp_path, "plate.0001.png", numbered_still_mode="identity"
        ) == "plate.0001"

    def test_movie_container_digits_never_stripped(self, tmp_path):
        # INVARIANT: only still extensions carry frame tokens; take_002.mov keeps
        # its number in every mode, or a real name is destroyed.
        for mode in ("auto", "sequence", "identity"):
            assert self._folder(
                tmp_path, "take_002.mov", numbered_still_mode=mode
            ) == "take_002", f"movie digits stripped in mode {mode}"

    def test_mode_is_advertised_on_the_node(self, tmp_path):
        # INVARIANT: the widget exists and defaults to the safe mode.
        spec = FolderIncrementer.INPUT_TYPES()["required"]["numbered_still_mode"]
        assert list(spec[0]) == ["auto", "sequence", "identity"]
        assert spec[1]["default"] == "auto"
