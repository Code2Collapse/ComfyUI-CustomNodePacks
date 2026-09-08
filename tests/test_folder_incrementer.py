"""Regression tests for folder_incrementer IS_CHANGED."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load_folder_incrementer():
    root = Path(__file__).resolve().parents[1]
    import sys
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    spec = importlib.util.spec_from_file_location(
        "folder_incrementer",
        root / "folder_incrementer.py",
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def fi():
    return _load_folder_incrementer()


def test_is_changed_not_nan(fi, tmp_path, monkeypatch):
    monkeypatch.setattr(fi, "_get_output_dir", lambda: str(tmp_path))
    out = fi.FolderIncrementer.IS_CHANGED(label="test", prefix="v", padding=3)
    assert out != "nan"
    assert isinstance(out, str)
    assert len(out) == 32


def test_is_changed_stable_for_same_inputs(fi, tmp_path, monkeypatch):
    monkeypatch.setattr(fi, "_get_output_dir", lambda: str(tmp_path))
    kwargs = {"label": "stable", "prefix": "v", "padding": 3, "date_format": "MM-DD-YYYY"}
    a = fi.FolderIncrementer.IS_CHANGED(**kwargs)
    b = fi.FolderIncrementer.IS_CHANGED(**kwargs)
    assert a == b


def test_is_changed_changes_when_version_dir_added(fi, tmp_path, monkeypatch):
    monkeypatch.setattr(fi, "_get_output_dir", lambda: str(tmp_path))
    kwargs = {"label": "proj", "prefix": "v", "padding": 3, "date_format": "MM-DD-YYYY"}
    before = fi.FolderIncrementer.IS_CHANGED(**kwargs)
    fmt = fi.DATE_FORMAT_MAP["MM-DD-YYYY"]
    from datetime import datetime
    date_dir = tmp_path / "proj" / datetime.now().strftime(fmt)
    (date_dir / "v001").mkdir(parents=True)
    after = fi.FolderIncrementer.IS_CHANGED(**kwargs)
    assert before != after
