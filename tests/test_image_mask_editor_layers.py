"""CPU tests for ImageMaskEditorC2C layer sidecar (store only)."""
from __future__ import annotations

import io
import sys
import types
import uuid
from pathlib import Path

import numpy as np
import pytest

PACK = Path(__file__).resolve().parents[1]
if str(PACK) not in sys.path:
    sys.path.insert(0, str(PACK))

from nodes.image_mask_editor import store

try:
    from PIL import Image
except ImportError:
    Image = None


@pytest.fixture
def mask_root(tmp_path, monkeypatch):
    inp = tmp_path / "input"
    inp.mkdir()
    temp = tmp_path / "temp"
    temp.mkdir()
    fp = types.SimpleNamespace(
        get_input_directory=lambda: str(inp),
        get_temp_directory=lambda: str(temp),
    )
    monkeypatch.setitem(sys.modules, "folder_paths", fp)
    return inp


def _png_bytes(arr: np.ndarray) -> bytes:
    assert Image is not None
    buf = io.BytesIO()
    Image.fromarray(arr.astype(np.uint8), mode="L").save(buf, format="PNG")
    return buf.getvalue()


def _eid() -> str:
    return uuid.uuid4().hex


def _plain_manifest(active: str = "base") -> dict:
    return {
        "version": 1,
        "layers": [{
            "id": active,
            "name": "Layer 1",
            "mode": "add",
            "visible": True,
            "locked": False,
        }],
        "active": active,
    }


def test_layers_manifest_round_trip(mask_root):
    eid = _eid()
    man = {
        "version": 1,
        "layers": [
            {"id": "a", "name": "Base", "mode": "add", "visible": True, "locked": False},
            {"id": "b", "name": "Cut", "mode": "subtract", "visible": True, "locked": False},
        ],
        "active": "a",
    }
    store.put_layers_manifest(eid, man)
    got = store.get_layers_manifest(eid)
    assert got == man
    arr = np.full((8, 8), 200, dtype=np.uint8)
    store.put_layer_frame(eid, "a", 0, _png_bytes(arr))
    body = store.get_layer_frame_png(eid, "a", 0)
    assert body is not None
    decoded = store.get_frame(eid, 0)
    assert decoded is None


def test_manifest_validation(mask_root):
    eid = _eid()
    with pytest.raises(ValueError):
        store.put_layers_manifest(eid, {"version": 1, "layers": [], "active": "x"})
    too_many = {
        "version": 1,
        "layers": [
            {"id": f"l{i}", "name": "L", "mode": "add", "visible": True, "locked": False}
            for i in range(9)
        ],
        "active": "l0",
    }
    with pytest.raises(ValueError):
        store.put_layers_manifest(eid, too_many)
    with pytest.raises(ValueError):
        store.put_layers_manifest(eid, {
            "version": 1,
            "layers": [{"id": "UPPER", "name": "x", "mode": "add", "visible": True, "locked": False}],
            "active": "UPPER",
        })
    with pytest.raises(ValueError):
        store.put_layers_manifest(eid, {
            "version": 1,
            "layers": [{"id": "a", "name": "x", "mode": "xor", "visible": True, "locked": False}],
            "active": "a",
        })
    with pytest.raises(ValueError):
        store.put_layers_manifest(eid, {
            "version": 1,
            "layers": [{"id": "a", "name": "x" * 41, "mode": "add", "visible": True, "locked": False}],
            "active": "a",
        })


def test_clear_removes_sidecar(mask_root):
    eid = _eid()
    store.put_layers_manifest(eid, {
        "version": 1,
        "layers": [
            {"id": "a", "name": "A", "mode": "add", "visible": True, "locked": False},
            {"id": "b", "name": "B", "mode": "add", "visible": True, "locked": False},
        ],
        "active": "a",
    })
    store.put_layer_frame(eid, "a", 0, _png_bytes(np.full((4, 4), 128, dtype=np.uint8)))
    ed = mask_root / "c2c_masks" / eid
    assert (ed / "layers.json").is_file()
    assert (ed / "layers" / "a" / "00000.png").is_file()
    store.clear(eid)
    assert not (ed / "layers.json").exists()
    assert not (ed / "layers").exists()


def test_copy_includes_sidecar(mask_root):
    src, dst = _eid(), _eid()
    store.put_frame(src, 0, _png_bytes(np.full((8, 8), 255, dtype=np.uint8)))
    store.put_layers_manifest(src, {
        "version": 1,
        "layers": [
            {"id": "a", "name": "A", "mode": "add", "visible": True, "locked": False},
            {"id": "b", "name": "B", "mode": "subtract", "visible": True, "locked": False},
        ],
        "active": "a",
    })
    store.put_layer_frame(src, "b", 0, _png_bytes(np.full((8, 8), 64, dtype=np.uint8)))
    ds = store.digest(src)
    store.copy(src, dst)
    assert store.get_layers_manifest(dst) == store.get_layers_manifest(src)
    assert store.get_layer_frame_png(dst, "b", 0) is not None
    assert store.digest(dst) == ds


def test_delete_frame_removes_layer_pngs(mask_root):
    eid = _eid()
    store.put_layers_manifest(eid, {
        "version": 1,
        "layers": [
            {"id": "a", "name": "A", "mode": "add", "visible": True, "locked": False},
            {"id": "b", "name": "B", "mode": "add", "visible": True, "locked": False},
        ],
        "active": "a",
    })
    store.put_layer_frame(eid, "a", 2, _png_bytes(np.full((4, 4), 200, dtype=np.uint8)))
    store.put_layer_frame(eid, "b", 2, _png_bytes(np.full((4, 4), 100, dtype=np.uint8)))
    store.put_frame(eid, 2, _png_bytes(np.full((4, 4), 255, dtype=np.uint8)))
    ed = mask_root / "c2c_masks" / eid
    store.delete_frame(eid, 2)
    assert not (ed / "layers" / "a" / "00002.png").exists()
    assert not (ed / "layers" / "b" / "00002.png").exists()
    assert not (ed / "00002.png").exists()


def test_digest_unchanged_by_sidecar_only(mask_root):
    eid = _eid()
    store.put_frame(eid, 0, _png_bytes(np.full((8, 8), 128, dtype=np.uint8)))
    d0 = store.digest(eid)
    store.put_layers_manifest(eid, {
        "version": 1,
        "layers": [
            {"id": "a", "name": "A", "mode": "add", "visible": True, "locked": False},
            {"id": "b", "name": "B", "mode": "add", "visible": True, "locked": False},
        ],
        "active": "a",
    })
    store.put_layer_frame(eid, "b", 0, _png_bytes(np.zeros((8, 8), dtype=np.uint8)))
    assert store.digest(eid) == d0


def test_single_visible_add_elides_sidecar(mask_root):
    eid = _eid()
    store.put_layers_manifest(eid, _plain_manifest())
    ed = mask_root / "c2c_masks" / eid
    assert not (ed / "layers.json").exists()
    assert not (ed / "layers").exists()
    store.put_layers_manifest(eid, {
        "version": 1,
        "layers": [{
            "id": "base",
            "name": "Layer 1",
            "mode": "subtract",
            "visible": True,
            "locked": False,
        }],
        "active": "base",
    })
    assert (ed / "layers.json").is_file()
    store.put_layers_manifest(eid, _plain_manifest())
    assert not (ed / "layers.json").exists()
