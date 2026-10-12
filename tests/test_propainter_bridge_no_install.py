"""ProPainter is used only when the user already has it - never downloaded, never installed (L7.92, owner 2026-10-12).

The bridge used to fetch sczhou/ProPainter (S-Lab License 1.0, non-commercial) from GitHub into third_party/ on its
first import - at every fresh ComfyUI start - and to download its weights on first use. Each test imports a private
copy of the module from a temporary pack folder, so the real third_party/ of this checkout is not seen.
"""
from __future__ import annotations

import importlib.util
import shutil
import sys
import urllib.request
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "nodes" / "propainter_bridge.py"


def _no_network(*_a, **_k):
    raise AssertionError("the ProPainter bridge must not touch the network")


def _load(tmp_path, monkeypatch, name):
    pack = tmp_path / "pack" / "nodes"
    pack.mkdir(parents=True)
    shutil.copy(SRC, pack / "propainter_bridge.py")
    monkeypatch.setattr(urllib.request, "urlopen", _no_network)
    monkeypatch.setattr(urllib.request, "urlretrieve", _no_network)
    monkeypatch.setattr(sys, "path", list(sys.path))
    for k in [k for k in sys.modules if k == "model" or k.startswith(("model.", "RAFT"))]:
        monkeypatch.delitem(sys.modules, k)
    spec = importlib.util.spec_from_file_location(name, pack / "propainter_bridge.py")
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, mod)          # @dataclass looks its module up there
    spec.loader.exec_module(mod)
    for k in [k for k in sys.modules if k == "model" or k.startswith(("model.", "RAFT"))]:
        monkeypatch.delitem(sys.modules, k, raising=False)
    return mod


def test_missing_propainter_downloads_and_creates_nothing(tmp_path, monkeypatch):
    monkeypatch.delenv("C2C_PROPAINTER_DIR", raising=False)
    before = sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*"))
    br = _load(tmp_path, monkeypatch, "_pp_bridge_missing")
    assert br.HAS_PROPAINTER is False and br._VENDORED_DIR is None
    after = sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*") if "__pycache__" not in p.parts)
    assert [p for p in after if p not in before and not p.startswith("pack")] == []      # no models/ folder
    assert not (tmp_path / "pack" / "third_party").exists()                               # no fetched source
    with pytest.raises(br.ProPainterMissingError) as e:
        br.require_propainter()
    msg = str(e.value)
    assert "non-commercial" in msg and "C2C_PROPAINTER_DIR" in msg and "does not download" in msg


def test_missing_weight_is_not_downloaded(tmp_path, monkeypatch):
    br = _load(tmp_path, monkeypatch, "_pp_bridge_weights")
    monkeypatch.setattr(br, "_MODELS_ROOT", str(tmp_path / "models" / "propainter"))
    with pytest.raises(FileNotFoundError) as e:
        br.ensure_weight("ProPainter.pth")
    assert "does not download" in str(e.value) and "releases/download" in str(e.value)
    assert not (tmp_path / "models").exists()


def test_a_users_own_checkout_is_used(tmp_path, monkeypatch):
    co = tmp_path / "my_propainter"
    (co / "model").mkdir(parents=True)
    (co / "RAFT" / "utils").mkdir(parents=True)
    (co / "model" / "__init__.py").write_text("")
    (co / "model" / "propainter.py").write_text("class InpaintGenerator:\n    pass\n")
    (co / "model" / "recurrent_flow_completion.py").write_text("class RecurrentFlowCompleteNet:\n    pass\n")
    (co / "RAFT" / "__init__.py").write_text("class RAFT:\n    pass\n")
    (co / "RAFT" / "utils" / "__init__.py").write_text("")
    (co / "RAFT" / "utils" / "utils.py").write_text("class InputPadder:\n    pass\n")
    monkeypatch.setenv("C2C_PROPAINTER_DIR", str(co))
    br = _load(tmp_path, monkeypatch, "_pp_bridge_user")
    assert br._VENDORED_DIR == str(co.resolve()) or Path(br._VENDORED_DIR) == co
    assert br.HAS_PROPAINTER is True
    br.require_propainter()                                                               # no error
