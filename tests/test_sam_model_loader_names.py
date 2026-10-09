"""L2.13: SAMModelLoaderMEC accepts both spellings of a model, in both orders.

The model list offers '[download] x' until x is on disk and 'x' after it, so a saved workflow broke either right
after the first download or on another machine. VALIDATE_INPUTS accepts both, and _resolve_path finds the file
or downloads it whichever spelling was saved.
"""
from __future__ import annotations

import pytest

NAME = "sam2.1_hiera_large.pt"


@pytest.fixture
def sml():
    # Imported inside the test, never at collection: the loader caches whether folder_paths exists at import time,
    # and other suites stub folder_paths first (an import here at collection broke the mask-editor tests).
    return pytest.importorskip("nodes.sam_model_loader")


@pytest.fixture
def L(sml):
    return sml.SAMModelLoaderMEC


def _patch_list(L, monkeypatch, options):
    monkeypatch.setattr(L, "INPUT_TYPES", classmethod(lambda cls: {"required": {"model_name": (options, {})}}))


def test_download_spelling_validates_after_the_file_is_on_disk(L, monkeypatch):
    _patch_list(L, monkeypatch, [NAME])                       # downloaded: the list now says plain x
    assert L.VALIDATE_INPUTS(f"[download] {NAME}") is True


def test_plain_spelling_validates_where_the_file_is_missing(L, monkeypatch):
    _patch_list(L, monkeypatch, [f"[download] {NAME}"])        # another machine: only the download entry
    assert L.VALIDATE_INPUTS(NAME) is True


def test_unknown_missing_model_gets_a_plain_error(L, monkeypatch):
    _patch_list(L, monkeypatch, [f"[download] {NAME}"])
    msg = L.VALIDATE_INPUTS("my_private_sam.pt")
    assert isinstance(msg, str) and "not installed" in msg


def test_plain_registry_name_missing_on_disk_downloads_on_load_only(sml, L, monkeypatch, tmp_path):
    called = {}
    monkeypatch.setattr(sml, "HAS_FOLDER_PATHS", False)
    monkeypatch.setattr(L, "_auto_download", staticmethod(lambda n: called.setdefault("name", n) or str(tmp_path / n)))
    out = L._resolve_path(NAME, download_missing_registry=True)       # what load() passes
    assert called["name"] == NAME and out.endswith(NAME)
    called.clear()
    with pytest.raises(FileNotFoundError):                              # a probe never downloads
        L._resolve_path(NAME)
    assert not called


def test_download_spelling_uses_the_file_once_present(sml, L, monkeypatch, tmp_path):
    f = tmp_path / NAME
    f.write_bytes(b"x")
    monkeypatch.setattr(sml, "HAS_FOLDER_PATHS", False)
    monkeypatch.setattr(L, "_auto_download", staticmethod(lambda n: pytest.fail("must not download a present file")))
    assert L._resolve_path(f"[download] {f}") == str(f)
