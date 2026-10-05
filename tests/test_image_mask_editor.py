"""CPU tests for ImageMaskEditorC2C store + node."""
from __future__ import annotations

import hashlib
import io
import sys
import types
import uuid
from pathlib import Path

import numpy as np
import pytest
import torch

PACK = Path(__file__).resolve().parents[1]
if str(PACK) not in sys.path:
    sys.path.insert(0, str(PACK))

from nodes.image_mask_editor import frames, store
from nodes.image_mask_editor.node import ImageMaskEditorC2C
from nodes.smart_crop import grow_or_shrink

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    import cv2
except ImportError:
    cv2 = None


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


def test_store_round_trip(mask_root):
    eid = _eid()
    arr = np.full((16, 24), 200, dtype=np.uint8)
    png = _png_bytes(arr)
    sha = store.put_frame(eid, 0, png)
    assert sha == store.frame_sha1((16, 24), arr)
    got = store.get_frame(eid, 0)
    assert got is not None and got.shape == (16, 24)
    assert int(got.mean()) == 200
    assert store.list_frames(eid) == [0]
    assert store.digest(eid) != ""


def test_resave_rewrites_one_file_per_frame(mask_root):
    """Saving a frame again replaces its PNG: the store never grows per save (ComfyUI's editor adds four
    clipspace files per save)."""
    eid = _eid()
    for v in (10, 200, 77):
        store.put_frame(eid, 0, _png_bytes(np.full((8, 8), v, dtype=np.uint8)))
    names = sorted(p.name for p in (mask_root / "c2c_masks" / eid).iterdir())
    assert [n for n in names if n.endswith(".png")] == ["00000.png"]
    assert not [n for n in names if n.endswith(".tmp")]
    assert int(store.get_frame(eid, 0).mean()) == 77


def test_digest_changes_on_edit_not_otherwise(mask_root):
    eid = _eid()
    d0 = store.digest(eid)
    assert d0 == ""
    store.put_frame(eid, 0, _png_bytes(np.zeros((8, 8), dtype=np.uint8)))
    d1 = store.digest(eid)
    assert d1 != d0
    store.put_frame(eid, 0, _png_bytes(np.zeros((8, 8), dtype=np.uint8)))
    assert store.digest(eid) == d1


def test_invalid_ids_and_frames_rejected(mask_root):
    with pytest.raises(ValueError):
        store.put_frame("../evil", 0, _png_bytes(np.zeros((4, 4), dtype=np.uint8)))
    with pytest.raises(ValueError):
        store.put_frame("short", 0, _png_bytes(np.zeros((4, 4), dtype=np.uint8)))
    eid = _eid()
    with pytest.raises(ValueError):
        store.put_frame(eid, 100_001, _png_bytes(np.zeros((4, 4), dtype=np.uint8)))


def test_copy_duplicates_frames(mask_root):
    src, dst = _eid(), _eid()
    store.put_frame(src, 0, _png_bytes(np.full((8, 8), 128, dtype=np.uint8)))
    store.put_frame(src, 2, _png_bytes(np.full((8, 8), 64, dtype=np.uint8)))
    ds = store.digest(src)
    store.copy(src, dst)
    assert store.list_frames(dst) == [0, 2]
    assert store.digest(dst) == ds
    assert store.get_frame(dst, 2) is not None


def _ref_gaussian_blur_np(mask_bhw: np.ndarray, blur_px: int) -> np.ndarray:
    """Independent reference matching smart_crop.gaussian_blur semantics."""
    if blur_px <= 0:
        return mask_bhw.astype(np.float32)
    b, h, w = mask_bhw.shape
    sigma = float(blur_px) * 0.33
    size = 2 * int(3.0 * sigma + 0.5) + 1
    size = min(size if size % 2 else size + 1, max(3, 2 * min(h, w) - 1))
    if size % 2 == 0:
        size -= 1
    sigma = max(0.1, min(sigma, size / 6.0))
    out = np.empty_like(mask_bhw, dtype=np.float32)
    if cv2 is not None:
        for i in range(b):
            out[i] = cv2.GaussianBlur(
                mask_bhw[i].astype(np.float32), (size, size), sigma,
            )
        return np.clip(out, 0.0, 1.0)
    # numpy separable fallback
    radius = size // 2
    x = np.arange(size, dtype=np.float32) - radius
    k = np.exp(-0.5 * (x / sigma) ** 2)
    k /= k.sum()
    for i in range(b):
        row = mask_bhw[i].astype(np.float32)
        tmp = np.apply_along_axis(lambda r: np.convolve(r, k, mode="same"), 1, row)
        out[i] = np.apply_along_axis(lambda r: np.convolve(r, k, mode="same"), 0, tmp)
    return np.clip(out, 0.0, 1.0)


def _ref_grow_np(mask_2d: np.ndarray, pixels: int) -> np.ndarray:
    """Independent of smart_crop: square (2p+1) dilation / erosion with OpenCV's default borders (outside the
    image never adds coverage when growing, never removes it when shrinking), binarised at > 0 like the node."""
    if pixels == 0:
        return mask_2d.astype(np.float32)
    if cv2 is None:
        pytest.skip("OpenCV is needed for the independent grow/shrink reference")
    k = np.ones((2 * abs(pixels) + 1,) * 2, np.uint8)
    src = mask_2d.astype(np.float32)
    out = cv2.dilate(src, k) if pixels > 0 else cv2.erode(src, k)
    return (out > 0).astype(np.float32)


def _image(b=1, h=32, w=48):
    return torch.rand(b, h, w, 3)


def _result(out):
    if isinstance(out, dict):
        return out["result"]
    return out


def _run(image, editor_id="", **kw):
    node = ImageMaskEditorC2C()
    defaults = {
        "frame_mode": "per_frame",
        "grow": 0,
        "feather": 0.0,
        "threshold": 0.0,
        "invert": False,
        "input_mask": None,
        "combine": "replace",
    }
    defaults.update(kw)
    return _result(node.execute(image, editor_id, **defaults))


def _run_full(image, editor_id="", **kw):
    node = ImageMaskEditorC2C()
    defaults = {
        "frame_mode": "per_frame",
        "grow": 0,
        "feather": 0.0,
        "threshold": 0.0,
        "invert": False,
        "input_mask": None,
        "combine": "replace",
    }
    defaults.update(kw)
    return node.execute(image, editor_id, **defaults)


def _ime_dir(mask_root, eid: str) -> Path:
    return mask_root.parent / "temp" / "c2c_ime" / eid


def test_output_shapes_b1_and_b3(mask_root):
    eid = _eid()
    store.put_frame(eid, 0, _png_bytes(np.full((32, 48), 255, dtype=np.uint8)))
    m1, p1, _ = _run(_image(1), eid)
    assert m1.shape == (1, 32, 48)
    assert p1.shape == (1, 32, 48, 3)
    store.put_frame(eid, 1, _png_bytes(np.full((32, 48), 128, dtype=np.uint8)))
    store.put_frame(eid, 2, _png_bytes(np.full((32, 48), 64, dtype=np.uint8)))
    m3, p3, _ = _run(_image(3), eid)
    assert m3.shape == (3, 32, 48)
    assert p3.shape == (3, 32, 48, 3)


def test_empty_editor_id_runs_without_raise(mask_root):
    im = _image(2)
    m, p, info = _run(im, "")
    assert m.shape == (2, 32, 48)
    assert torch.all(m == 0)
    assert "no mask painted yet" in info


def test_per_frame_vs_shared(mask_root):
    eid = _eid()
    store.put_frame(eid, 1, _png_bytes(np.full((32, 48), 255, dtype=np.uint8)))
    m, _, _ = _run(_image(3), eid, frame_mode="per_frame")
    assert float(m[0].max()) == 0.0
    assert float(m[1].max()) == 1.0
    assert float(m[2].max()) == 0.0
    store.clear(eid)
    store.put_frame(eid, 0, _png_bytes(np.full((32, 48), 200, dtype=np.uint8)))
    m2, _, _ = _run(_image(3), eid, frame_mode="shared")
    assert float(m2[0].mean()) > 0.7
    assert float(m2[2].mean()) > 0.7


def test_resize_path(mask_root):
    eid = _eid()
    store.put_frame(eid, 0, _png_bytes(np.full((16, 16), 255, dtype=np.uint8)))
    _, _, info = _run(_image(1, 32, 48), eid)
    assert "resized" in info


def test_combine_modes(mask_root):
    eid = _eid()
    store.put_frame(eid, 0, _png_bytes(np.full((32, 48), 255, dtype=np.uint8)))
    inp = torch.zeros(1, 32, 48)
    inp[:, 8:24, 8:24] = 0.5
    m_rep, _, _ = _run(_image(1), eid, input_mask=inp, combine="replace")
    assert float(m_rep.mean()) > 0.9
    m_add, _, _ = _run(_image(1), eid, input_mask=inp, combine="add")
    assert float(m_add.min()) >= 0.0
    assert float(m_add.max()) <= 1.0
    assert float(m_add.mean()) > float(m_rep.mean()) * 0.5
    m_sub, _, _ = _run(_image(1), eid, input_mask=inp, combine="subtract")
    assert float(m_sub.max()) < 0.6
    m_int, _, _ = _run(_image(1), eid, input_mask=inp, combine="intersect")
    assert float(m_int[:, 8:24, 8:24].mean()) > 0.4
    assert float(m_int[:, 0, 0]) == 0.0


def test_grow_feather_threshold_invert(mask_root):
    eid = _eid()
    arr = np.zeros((32, 48), dtype=np.uint8)
    arr[12:20, 18:30] = 255
    store.put_frame(eid, 0, _png_bytes(arr))
    base = torch.from_numpy(arr.astype(np.float32) / 255.0).unsqueeze(0)
    # full-array golden values, not means
    m_g, _, _ = _run(_image(1), eid, grow=2)
    assert np.array_equal(m_g[0].numpy(), _ref_grow_np(base[0].numpy(), 2))
    m_s, _, _ = _run(_image(1), eid, grow=-2)
    assert np.array_equal(m_s[0].numpy(), _ref_grow_np(base[0].numpy(), -2))
    m_f, _, _ = _run(_image(1), eid, feather=8.0)
    ref_f = _ref_gaussian_blur_np(base.numpy(), 8)
    assert float(np.abs(m_f.numpy() - ref_f).max()) < 1e-4
    m_t, _, _ = _run(_image(1), eid, feather=4.0, threshold=0.5)
    assert set(torch.unique(m_t).tolist()).issubset({0.0, 1.0})
    ref_t = (_ref_gaussian_blur_np(base.numpy(), 4) >= 0.5).astype(np.float32)
    assert float(np.abs(m_t.numpy() - ref_t).sum()) <= 2.0   # at most a couple of exact-0.5 ties
    m_i, _, _ = _run(_image(1), eid, threshold=0.5, invert=True)
    assert np.array_equal(m_i[0].numpy(), 1.0 - (arr >= 128).astype(np.float32))   # 6.25% painted -> 93.75%


def test_is_changed_stable_vs_changed(mask_root):
    eid = _eid()
    im = _image(1)
    k0 = ImageMaskEditorC2C.IS_CHANGED(
        im, eid, "per_frame", 0, 0.0, 0.0, False,
    )
    store.put_frame(eid, 0, _png_bytes(np.full((32, 48), 255, dtype=np.uint8)))
    k1 = ImageMaskEditorC2C.IS_CHANGED(
        im, eid, "per_frame", 0, 0.0, 0.0, False,
    )
    assert k1 != k0
    k2 = ImageMaskEditorC2C.IS_CHANGED(
        im, eid, "per_frame", 0, 0.0, 0.0, False,
    )
    assert k2 == k1
    store.put_frame(eid, 0, _png_bytes(np.full((32, 48), 255, dtype=np.uint8)))
    assert ImageMaskEditorC2C.IS_CHANGED(
        im, eid, "per_frame", 0, 0.0, 0.0, False,
    ) == k1


def test_record_frames_b1_and_b3(mask_root):
    eid = _eid()
    im1 = _image(1)
    out1 = _run_full(im1, eid)
    payload1 = out1["ui"]["c2c_ime_frames"][0]
    d1 = _ime_dir(mask_root, eid)
    assert payload1["count"] == 1
    assert payload1["width"] == 48
    assert payload1["height"] == 32
    assert payload1["subfolder"] == f"c2c_ime/{eid}"
    assert payload1["type"] == "temp"
    assert len(payload1["version"]) == 16
    assert (d1 / "00000.png").is_file()
    assert (d1 / "t_00000.jpg").is_file()

    im3 = _image(3)
    out3 = _run_full(im3, eid)
    payload3 = out3["ui"]["c2c_ime_frames"][0]
    assert payload3["count"] == 3
    for i in range(3):
        assert (d1 / f"{i:05d}.png").is_file()
        assert (d1 / f"t_{i:05d}.jpg").is_file()


def test_record_frames_idempotent_same_image(mask_root):
    eid = _eid()
    im = _image(1)
    out1 = _run_full(im, eid)
    v1 = out1["ui"]["c2c_ime_frames"][0]["version"]
    d = _ime_dir(mask_root, eid)
    mtimes = {p.name: p.stat().st_mtime_ns for p in d.iterdir() if p.is_file()}
    out2 = _run_full(im, eid)
    for p in d.iterdir():
        if p.is_file():
            assert p.stat().st_mtime_ns == mtimes[p.name]
    assert out2["ui"]["c2c_ime_frames"][0]["version"] == v1


def test_record_frames_version_changes_on_pixel_edit(mask_root):
    eid = _eid()
    im = _image(1)
    out1 = _run_full(im, eid)
    v1 = out1["ui"]["c2c_ime_frames"][0]["version"]
    im2 = im.clone()
    im2[0, 0, 0, 0] = 1.0 - im2[0, 0, 0, 0]
    out2 = _run_full(im2, eid)
    v2 = out2["ui"]["c2c_ime_frames"][0]["version"]
    assert v2 != v1


def test_record_frames_batch_shrink_deletes_extra(mask_root):
    eid = _eid()
    _run_full(_image(3), eid)
    d = _ime_dir(mask_root, eid)
    assert (d / "00002.png").is_file()
    _run_full(_image(1), eid)
    assert not (d / "00001.png").exists()
    assert not (d / "00002.png").exists()
    assert not (d / "t_00001.jpg").exists()
    assert not (d / "t_00002.jpg").exists()
    assert (d / "00000.png").is_file()


def test_record_frames_over_budget_writes_nothing(mask_root, monkeypatch):
    eid = _eid()
    monkeypatch.setattr(frames, "RECORD_BUDGET_PIXELS", 100)
    out = _run_full(_image(1, 32, 48), eid)
    assert out["ui"]["c2c_ime_frames"] == []
    assert any("input too large" in part for part in out["result"][2].split("; "))
    assert not _ime_dir(mask_root, eid).exists()


def test_record_frames_invalid_editor_id(mask_root):
    out = _run_full(_image(1), "short")
    assert out["ui"]["c2c_ime_frames"] == []
    m, _, _ = _run(_image(1), "short")
    assert m.shape == (1, 32, 48)
    assert not (mask_root.parent / "temp" / "c2c_ime").exists()


def test_record_frames_failing_temp_dir_still_returns_mask(mask_root, monkeypatch):
    eid = _eid()
    store.put_frame(eid, 0, _png_bytes(np.full((32, 48), 255, dtype=np.uint8)))

    def _boom(*_a, **_k):
        raise OSError("disk full")

    monkeypatch.setattr(Path, "mkdir", _boom)
    m, _, info = _run(_image(1), eid)
    assert m.shape == (1, 32, 48)
    assert float(m.max()) == 1.0
    assert "frame recording failed" in info


def test_routes_idempotent():
    from nodes.image_mask_editor.routes import _ROUTES_REGISTERED, register_routes

    class FakeRoutes:
        def __init__(self):
            self.items = []

        def post(self, path):
            def deco(fn):
                self.items.append(("POST", path, fn))
                return fn
            return deco

        def get(self, path):
            def deco(fn):
                self.items.append(("GET", path, fn))
                return fn
            return deco

        def delete(self, path):
            def deco(fn):
                self.items.append(("DELETE", path, fn))
                return fn
            return deco

    srv = types.SimpleNamespace(routes=FakeRoutes())
    import nodes.image_mask_editor.routes as routes_mod
    routes_mod._ROUTES_REGISTERED = False
    register_routes(srv)
    n1 = len(srv.routes.items)
    register_routes(srv)
    assert len(srv.routes.items) == n1
    routes_mod._ROUTES_REGISTERED = _ROUTES_REGISTERED
