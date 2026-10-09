"""CPU tests for Image Mask Editor smart backends (SAM + refine)."""
from __future__ import annotations

import importlib.util
import io
import json
import sys
import types
import uuid
from pathlib import Path

import numpy as np
import pytest

PACK = Path(__file__).resolve().parents[1]
if str(PACK) not in sys.path:
    sys.path.insert(0, str(PACK))

from nodes.image_mask_editor import smart

try:
    from PIL import Image
except ImportError:
    Image = None


@pytest.fixture(autouse=True)
def _reset_smart():
    smart.reset_state_for_tests()
    yield
    smart.reset_state_for_tests()


@pytest.fixture
def mask_root(tmp_path, monkeypatch):
    inp = tmp_path / "input"
    inp.mkdir()
    temp = tmp_path / "temp"
    temp.mkdir()
    vit = tmp_path / "models" / "vitmatte"
    vit.mkdir(parents=True)
    fp = types.SimpleNamespace(
        get_input_directory=lambda: str(inp),
        get_temp_directory=lambda: str(temp),
        base_path=str(tmp_path),
        models_dir=str(tmp_path / "models"),
        folder_names_and_paths={
            "sams": ([str(tmp_path / "models" / "sam2")], {".pt"}),
            "vitmatte": ([str(vit)], set()),
        },
    )

    def _get_list(key):
        if key == "sams":
            d = tmp_path / "models" / "sam2"
            d.mkdir(parents=True, exist_ok=True)
            return ["tiny.pt", "[download] big.pt"]
        return []

    def _get_full(key, name):
        if key == "sams" and name == "tiny.pt":
            p = tmp_path / "models" / "sam2" / "tiny.pt"
            p.write_bytes(b"x" * 100)
            return str(p)
        return None

    fp.get_filename_list = _get_list
    fp.get_full_path = _get_full
    fp.get_folder_paths = lambda k: fp.folder_names_and_paths.get(k, ([], set()))[0]
    monkeypatch.setitem(sys.modules, "folder_paths", fp)
    # The SAM loader binds folder_paths when it is first imported. If another suite imported it earlier, the
    # sys.modules stub above never reaches it: patch its own reference too, so these tests do not depend on order.
    loader = sys.modules.get("nodes.sam_model_loader")
    if loader is not None:
        monkeypatch.setattr(loader, "folder_paths", fp, raising=False)
        monkeypatch.setattr(loader, "HAS_FOLDER_PATHS", True, raising=False)
    return tmp_path


def _eid() -> str:
    return uuid.uuid4().hex


def _patch_sam2_installed(monkeypatch):
    """list_sam_models / default backend need sam2 present unless testing runtime missing."""
    real_find_spec = importlib.util.find_spec

    def _find_spec(name, *a, **k):
        if name == "sam2":
            return object()
        return real_find_spec(name, *a, **k)

    monkeypatch.setattr(importlib.util, "find_spec", _find_spec)


def _use_default_sam_backend():
    with smart._LOCK:
        smart._BACKENDS.pop("sam", None)


def _png_rgb(arr: np.ndarray) -> bytes:
    assert Image is not None
    buf = io.BytesIO()
    Image.fromarray(arr.astype(np.uint8), mode="RGB").save(buf, format="PNG")
    return buf.getvalue()


def _png_l(arr: np.ndarray) -> bytes:
    assert Image is not None
    buf = io.BytesIO()
    Image.fromarray(arr.astype(np.uint8), mode="L").save(buf, format="PNG")
    return buf.getvalue()


def _meta(**kw):
    base = {
        "editor_id": _eid(),
        "frame_key": "a" * 40,
        "seq": 1,
        "width": 32,
        "height": 24,
        "model": "stub-model",
        "points": [[8, 8, 1]],
        "box": None,
    }
    base.update(kw)
    return base


class StubSam:
    embed_count = 0
    predict_count = 0
    last_points = None
    last_labels = None
    last_box = None
    load_count = 0
    release_count = 0

    def reset(self):
        StubSam.embed_count = 0
        StubSam.predict_count = 0
        StubSam.last_points = None
        StubSam.last_labels = None
        StubSam.last_box = None
        StubSam.load_count = 0
        StubSam.release_count = 0

    def load(self, model_name):
        StubSam.load_count += 1
        return {"model": model_name}

    def embed(self, handle, rgb_u8):
        StubSam.embed_count += 1
        return {"handle": handle, "h": rgb_u8.shape[0], "w": rgb_u8.shape[1]}

    def predict(self, emb, point_coords, point_labels, box):
        StubSam.predict_count += 1
        StubSam.last_points = None if point_coords is None else point_coords.tolist()
        StubSam.last_labels = None if point_labels is None else point_labels.tolist()
        StubSam.last_box = None if box is None else box.tolist()
        h, w = emb["h"], emb["w"]
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[4:20, 4:20] = 255
        return mask, 0.91

    def release(self, handle):
        StubSam.release_count += 1


class StubMatte:
    available_ok = True
    matte_count = 0
    last_trimap = None

    def reset(self):
        StubMatte.matte_count = 0
        StubMatte.last_trimap = None
        self.available_ok = True

    def available(self):
        if self.available_ok:          # instance state: a test flips it on the registered stub
            return True, ""
        return False, "ViTMatte weights not found locally."

    def matte(self, rgb_u8, trimap_f32):
        StubMatte.matte_count += 1
        StubMatte.last_trimap = trimap_f32.copy()
        return np.clip(trimap_f32, 0, 1).astype(np.float32)


@pytest.fixture
def stubs():
    sam = StubSam()
    matte = StubMatte()
    smart.register_backend("sam", sam)
    smart.register_backend("matte", matte)
    sam.reset()
    matte.reset()
    return sam, matte


def test_sam_first_call_need_image(stubs):
    st, _ = stubs
    meta = _meta()
    status, body, _ = smart.handle_sam(meta)
    assert status == 409
    assert body.get("need_image") is True
    assert st.embed_count == 0
    assert st.predict_count == 0


def test_sam_with_image_returns_mask(stubs):
    st, _ = stubs
    meta = _meta()
    rgb = np.zeros((24, 32, 3), dtype=np.uint8)
    status, body, hdr = smart.handle_sam(meta, _png_rgb(rgb))
    assert status == 200
    assert hdr.get("X-IME-Score") == "0.91"
    arr = np.array(Image.open(io.BytesIO(body)))
    assert arr.shape == (24, 32)
    assert st.embed_count == 1
    assert st.predict_count == 1


def test_sam_same_frame_key_embed_once(stubs):
    st, _ = stubs
    eid = _eid()
    rgb = np.zeros((24, 32, 3), dtype=np.uint8)
    png = _png_rgb(rgb)
    fk = "b" * 40
    m1 = _meta(editor_id=eid, frame_key=fk, seq=1)
    m2 = _meta(editor_id=eid, frame_key=fk, seq=2, points=[[10, 10, 0]])
    smart.handle_sam(m1, png)
    smart.handle_sam(m2)
    assert st.embed_count == 1
    assert st.predict_count == 2


def test_sam_new_frame_key_embeds_again(stubs):
    st, _ = stubs
    eid = _eid()
    rgb = np.zeros((24, 32, 3), dtype=np.uint8)
    png = _png_rgb(rgb)
    smart.handle_sam(_meta(editor_id=eid, frame_key="c" * 40, seq=1), png)
    smart.handle_sam(_meta(editor_id=eid, frame_key="d" * 40, seq=2), png)
    assert st.embed_count == 2


def test_sam_negative_and_box_reach_predict(stubs):
    st, _ = stubs
    meta = _meta(
        points=[[5, 5, 1], [20, 20, 0]],
        box=[2, 2, 28, 20],
    )
    rgb = np.zeros((24, 32, 3), dtype=np.uint8)
    smart.handle_sam(meta, _png_rgb(rgb))
    assert st.last_labels == [1, 0]
    assert st.last_box == [2.0, 2.0, 28.0, 20.0]


def test_sam_stale_seq_superseded(stubs):
    st, _ = stubs
    eid = _eid()
    rgb = np.zeros((24, 32, 3), dtype=np.uint8)
    png = _png_rgb(rgb)
    smart.handle_sam(_meta(editor_id=eid, seq=5), png)
    before = st.predict_count
    status, body, _ = smart.handle_sam(_meta(editor_id=eid, seq=3), png)
    assert status == 409
    assert body.get("superseded") is True
    assert st.predict_count == before


def test_sam_bad_inputs_400(stubs):
    status, body, _ = smart.handle_sam({"editor_id": "short", "seq": 0})
    assert status == 400
    assert "error" in body
    meta = _meta(points=[[100, 100, 1]])
    rgb = np.zeros((24, 32, 3), dtype=np.uint8)
    status, body, _ = smart.handle_sam(meta, _png_rgb(rgb))
    assert status == 400


def test_release_drops_embedding_and_model(stubs):
    st, _ = stubs
    eid = _eid()
    rgb = np.zeros((24, 32, 3), dtype=np.uint8)
    smart.handle_sam(_meta(editor_id=eid), _png_rgb(rgb))
    assert st.load_count == 1
    smart.handle_release(eid)
    assert st.release_count == 1
    st.embed_count = 0
    status, body, _ = smart.handle_sam(_meta(editor_id=eid, seq=10))
    assert status == 409
    assert body.get("need_image") is True
    assert st.embed_count == 0


def test_refine_matte_called_with_trimap(stubs):
    _, mt = stubs
    rgb = np.full((16, 16, 3), 128, dtype=np.uint8)
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[4:12, 4:12] = 255
    band = np.zeros((16, 16), dtype=np.uint8)
    band[6:10, 6:10] = 255
    meta = {"editor_id": _eid(), "seq": 1, "bbox": [0, 0, 15, 15]}
    status, _, _ = smart.handle_refine(meta, _png_rgb(rgb), _png_l(mask), _png_l(band))
    assert status == 200
    assert mt.matte_count == 1
    tri = mt.last_trimap
    assert tri is not None
    assert set(np.unique(tri[tri > 0]).tolist()).issubset({0.5, 1.0})
    assert float(tri[band > 0].mean()) == 0.5
    assert float(tri[(band == 0) & (mask >= 128)].mean()) == 1.0


def test_refine_empty_band_no_matte(stubs):
    _, mt = stubs
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    mask = np.full((8, 8), 200, dtype=np.uint8)
    band = np.zeros((8, 8), dtype=np.uint8)
    meta = {"editor_id": _eid(), "seq": 1, "bbox": [0, 0, 7, 7]}
    mask_png = _png_l(mask)
    status, body, _ = smart.handle_refine(meta, _png_rgb(rgb), mask_png, _png_l(band))
    assert status == 200
    assert body == mask_png
    assert mt.matte_count == 0


def test_refine_no_weights_409(stubs):
    _, mt = stubs
    mt.available_ok = False
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    mask = np.zeros((8, 8), dtype=np.uint8)
    band = np.full((8, 8), 255, dtype=np.uint8)
    meta = {"editor_id": _eid(), "seq": 1, "bbox": [0, 0, 7, 7]}
    status, body, _ = smart.handle_refine(meta, _png_rgb(rgb), _png_l(mask), _png_l(band))
    assert status == 409
    assert body.get("need_weights") is True


def test_sam_models_sort_and_default(mask_root, monkeypatch):
    _patch_sam2_installed(monkeypatch)
    sam_dir = mask_root / "models" / "sam2"
    sam_dir.mkdir(parents=True, exist_ok=True)
    big = sam_dir / "big.pt"
    tiny = sam_dir / "tiny.pt"
    big.write_bytes(b"x" * 500)
    tiny.write_bytes(b"x" * 50)

    from nodes.sam_model_loader import _DOWNLOAD_REGISTRY

    monkeypatch.setitem(_DOWNLOAD_REGISTRY, "big.pt", {"repo_id": "x", "filename": "big.pt"})

    out = smart.list_sam_models()
    models = out["models"]
    dl_idx = next(i for i, n in enumerate(models) if n.startswith("[download]"))
    inst_idx = next(i for i, n in enumerate(models) if n == "tiny.pt")
    assert inst_idx < dl_idx
    assert out["default"] == "tiny.pt"


def test_sam_models_no_default_when_nothing_installed(mask_root, monkeypatch):
    """With no SAM file on disk the editor must not pre-select a "[download]" entry: picking one is the
    user's explicit choice (the first click would otherwise start a multi-GB download)."""
    monkeypatch.setattr(smart, "_sam_model_files", lambda: ["[download] sam2.1_hiera_tiny.pt", "[download] sam3.pt"])
    monkeypatch.setattr(smart, "_is_installed_model", lambda name: False)
    out = smart.list_sam_models()
    assert out["default"] == ""
    assert out["models"] == ["[download] sam2.1_hiera_tiny.pt", "[download] sam3.pt"]


def test_sam_models_never_list_the_loader_placeholder(mask_root, monkeypatch):
    """The loader node's combo placeholder "(place model in ...)" is not a model: the editor list carries only
    installed files and "[download]" entries (its own "choose a model" option covers the empty case)."""
    _patch_sam2_installed(monkeypatch)
    import folder_paths

    monkeypatch.setattr(folder_paths, "get_filename_list", lambda key: [])
    from nodes.sam_model_loader import SAMModelLoaderMEC

    monkeypatch.setattr(SAMModelLoaderMEC, "_scan_extra_paths", staticmethod(lambda files: None))
    names = smart._sam_model_files()
    assert names and all(n.startswith("[download]") for n in names)
    assert not any(n.startswith("(") for n in smart.list_sam_models()["models"])


def test_sam_auto_download_uses_models_dir(tmp_path, monkeypatch):
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    fp = types.SimpleNamespace(
        models_dir=str(models_dir),
        base_path=str(tmp_path),
        folder_names_and_paths={},
    )
    monkeypatch.setitem(sys.modules, "folder_paths", fp)
    monkeypatch.setattr(
        "nodes.sam_model_loader.HAS_FOLDER_PATHS",
        True,
        raising=False,
    )
    monkeypatch.setattr(
        "nodes.sam_model_loader.folder_paths",
        fp,
        raising=False,
    )

    from nodes.sam_model_loader import SAMModelLoaderMEC, _DOWNLOAD_REGISTRY

    name = "sam2.1_hiera_tiny.pt"
    monkeypatch.setitem(
        _DOWNLOAD_REGISTRY,
        name,
        {"repo_id": "facebook/sam2.1-hiera-tiny", "filename": name},
    )

    dest = models_dir / "sam2" / name

    def _fake_hf_hub_download(**kwargs):
        local_dir = Path(kwargs["local_dir"])
        local_dir.mkdir(parents=True, exist_ok=True)
        dest_path = local_dir / kwargs["filename"]
        dest_path.write_bytes(b"fake-checkpoint")
        return str(dest_path)

    hf_mod = types.ModuleType("huggingface_hub")
    hf_mod.hf_hub_download = _fake_hf_hub_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", hf_mod)

    path = SAMModelLoaderMEC._auto_download(name)
    assert path == str(dest)
    assert dest.is_file()
    assert SAMModelLoaderMEC._resolve_path(f"[download] {name}") == str(dest)


def test_default_sam_backend_respects_comfy_cpu_device(monkeypatch):
    captured = {}

    def _fake_load(self, model_name, model_type, device, offload_to_cpu, dtype):
        captured["device"] = device
        captured["dtype"] = dtype
        return ({"model": None, "model_type": "sam2.1"},)

    mm = types.SimpleNamespace(
        get_torch_device=lambda: types.SimpleNamespace(type="cpu"),
    )
    monkeypatch.setitem(sys.modules, "comfy.model_management", mm)
    monkeypatch.setattr(
        "nodes.sam_model_loader.SAMModelLoaderMEC.load",
        _fake_load,
    )

    backend = smart._DefaultSamBackend()
    backend.load("tiny.pt")
    assert captured["device"] == "cpu"
    assert captured["dtype"] == "float32"


def test_move_to_inference_device_respects_comfy_cpu(monkeypatch):
    import torch
    from nodes.sam_model_loader import move_to_inference_device

    moved_to = []

    class _FakeModel:
        def to(self, device):
            moved_to.append(device)
            return self

    mm = types.SimpleNamespace(
        get_torch_device=lambda: types.SimpleNamespace(type="cpu"),
    )
    monkeypatch.setitem(sys.modules, "comfy.model_management", mm)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    wrapper = {
        "model": _FakeModel(),
        "device": "cpu",
        "offload_to_cpu": True,
    }
    target = move_to_inference_device(wrapper)
    assert target == "cpu"
    assert moved_to == ["cpu"]


def test_handle_sam_runtime_missing_returns_400_before_acquire(monkeypatch):
    _use_default_sam_backend()
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a, **k: None)

    def _fail_acquire(_model_name):
        raise AssertionError("_acquire_model must not be called when sam2 is missing")

    monkeypatch.setattr(smart, "_acquire_model", _fail_acquire)

    meta = _meta(model="[download] sam2.1_hiera_tiny.pt")
    rgb = np.zeros((24, 32, 3), dtype=np.uint8)
    status, body, _ = smart.handle_sam(meta, _png_rgb(rgb))
    assert status == 400
    assert "sam2" in body.get("error", "").lower()


def test_list_sam_models_unavailable_when_runtime_missing(mask_root, monkeypatch):
    _use_default_sam_backend()
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a, **k: None)
    monkeypatch.setattr(
        smart,
        "_sam_model_files",
        lambda: ["[download] sam2.1_hiera_tiny.pt", "tiny.pt"],
    )
    monkeypatch.setattr(smart, "_is_installed_model", lambda name: name == "tiny.pt")

    out = smart.list_sam_models()
    assert out["default"] == ""
    assert set(out["unavailable"].keys()) == {"tiny.pt", "[download] sam2.1_hiera_tiny.pt"}
    assert "sam2" in out["unavailable"]["tiny.pt"].lower()

