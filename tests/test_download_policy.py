"""L7.97: model weights are downloaded only with the node's "Allow Download" (owner 2026-10-12).

Installed copies are found first - this pack's folder, another pack's folder, the Hugging Face cache; nothing is
fetched without permission; a missing model gives the labelled model-free result where one exists, otherwise a
message naming the model, where it goes and how to allow the download. No test touches the network.
"""
from __future__ import annotations

import json
import os
import types
import urllib.request

import numpy as np
import pytest
import torch

W, H, CX, CY, R = 256, 192, 128, 96, 40


def scene(skin=False):
    rng = np.random.default_rng(5)
    g = (0.45 + 0.08 * rng.standard_normal((H, W, 1))).clip(0, 1).repeat(3, -1)
    yy, xx = np.mgrid[:H, :W]
    disk = (xx - CX) ** 2 + (yy - CY) ** 2 <= R ** 2
    g[disk] = [0.88, 0.62, 0.50] if skin else [0.85, 0.12, 0.10]          # a skin tone / a red disk
    return (g * 255).astype(np.uint8), disk


def iou(m, gt):
    m = np.asarray(m) > 0.5
    return float(np.logical_and(m, gt).sum()) / float(np.logical_or(m, gt).sum())


def as_image(u8):
    return torch.from_numpy(u8).float().div(255.0).unsqueeze(0)


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def refuse(*_a, **_k):
        raise AssertionError("no test may reach the network")
    monkeypatch.setattr(urllib.request, "urlopen", refuse)
    monkeypatch.delenv("C2C_NO_DOWNLOADS", raising=False)


@pytest.fixture
def pol():
    return pytest.importorskip("nodes._download_policy")


@pytest.fixture
def mm(monkeypatch, tmp_path, pol):
    mm = pytest.importorskip("nodes.model_manager")
    monkeypatch.setattr(mm, "_MODELS_DIR", str(tmp_path))
    monkeypatch.setattr(mm, "folder_paths", None)
    monkeypatch.setattr(pol, "hf_cached_snapshot", lambda repo: None)
    monkeypatch.setattr(pol, "hf_cached_file", lambda repo, f: None)
    calls = []
    monkeypatch.setattr(mm, "ensure_downloaded", lambda name, dest=None: calls.append(name) or "<downloaded>")
    mm._test_download_calls = calls
    return mm


# ── the policy ───────────────────────────────────────────────────────────────────────────────────────────────────

def test_off_unless_a_node_allows_it(pol, monkeypatch, tmp_path):
    assert pol.downloads_allowed() is False
    with pol.allow_downloads(True):
        assert pol.downloads_allowed() is True
        assert pol.hf_kwargs("org/repo") == {}
    with pol.allow_downloads(False):
        assert pol.hf_kwargs("org/repo") == {"local_files_only": True}
        assert pol.hf_kwargs(str(tmp_path)) == {}                        # a local folder is never a download
    monkeypatch.setenv("C2C_NO_DOWNLOADS", "1")
    with pol.allow_downloads(True):
        assert pol.downloads_allowed() is False                          # the machine-wide switch wins


def test_torch_hub_needs_permission(pol, monkeypatch, tmp_path):
    monkeypatch.setattr(torch.hub, "get_dir", lambda: str(tmp_path))
    with pytest.raises(pol.ModelNotDownloaded):
        pol.require_hub_repo("PeterL1n/RobustVideoMatting")
    (tmp_path / "PeterL1n_RobustVideoMatting_master").mkdir()
    pol.require_hub_repo("PeterL1n/RobustVideoMatting")                 # cached code: fine
    with pytest.raises(pol.ModelNotDownloaded, match="weights"):
        pol.require_hub_repo("PeterL1n/RobustVideoMatting", weights_missing=True)


# ── model_manager: find first, download only when allowed ────────────────────────────────────────────────────────

def test_directory_model_missing_is_not_downloaded(mm, pol):
    with pytest.raises(pol.ModelNotDownloaded) as e:
        mm.get_model_path("birefnet_general")
    assert "birefnet" in str(e.value) and "allow_download" in str(e.value) and "huggingface.co" in str(e.value)
    assert mm._test_download_calls == []
    with pol.allow_downloads(True):
        assert mm.get_model_path("birefnet_general") == "<downloaded>"
    assert mm._test_download_calls == ["birefnet_general"]


def test_another_packs_copy_and_the_hf_cache_are_reused(mm, pol, tmp_path, monkeypatch):
    other = tmp_path / "BiRefNet" / "BiRefNet"                             # e.g. a BiRefNet pack's folder
    other.mkdir(parents=True)
    (other / "config.json").write_text("{}")
    assert mm.get_model_path("birefnet_general") == str(other)
    monkeypatch.setattr(pol, "hf_cached_snapshot", lambda repo: "/hf/cache/segformer" if "segformer" in repo else None)
    assert mm.get_model_path("segformer_clothes") == "/hf/cache/segformer"
    assert mm._test_download_calls == []


def test_file_model_found_in_any_known_folder(mm, pol, monkeypatch, tmp_path):
    f = tmp_path / "elsewhere" / "sam2.1_hiera_base_plus.safetensors"
    f.parent.mkdir()
    f.write_bytes(b"x")
    fp = types.SimpleNamespace(folder_names_and_paths={"other_pack_sams": ([str(f.parent)], set())},
                               get_full_path=lambda key, name: str(f) if name == f.name else None)
    monkeypatch.setattr(mm, "folder_paths", fp)
    assert mm.get_model_path("sam2.1_hiera_base_plus") == str(f)
    monkeypatch.setattr(mm, "folder_paths", None)
    with pytest.raises(pol.ModelNotDownloaded) as e:
        mm.get_model_path("sam2.1_hiera_base_plus")
    assert not (tmp_path / "sam2").exists()                              # no folder made for a download that never ran
    # exact diagnostics: the absolute folder and the file name, separately and as one path
    assert e.value.target == os.path.abspath(str(tmp_path / "sam2")) and e.value.file_name == f.name
    assert e.value.expected_path == os.path.join(e.value.target, f.name)
    assert e.value.target in str(e.value) and f"'{f.name}'" in str(e.value)


# ── UnifiedSegmentation (reached through SeC + MatAnyone): approved fix A, owner 2026-10-13 ─────────────────────

_US_NAME = "sam2.1_hiera_base_plus"


@pytest.fixture
def us(monkeypatch, tmp_path, mm, pol):
    """UnifiedSegmentation with an empty models folder, no folder_paths, an empty HF cache, and a downloader that
    fails the test if it is reached."""
    us = pytest.importorskip("nodes.unified_segmentation")
    monkeypatch.setattr(us, "_MODELS_DIR", str(tmp_path))
    monkeypatch.setattr(us, "folder_paths", None)
    import huggingface_hub

    def no_hub(*_a, **_k):
        raise AssertionError("hf_hub_download must not be called")
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", no_hub)
    return us


def _entries(root):
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*"))


def test_unified_segmentation_missing_model_no_network_no_folder(us, pol, tmp_path):
    reg = us.MODEL_REGISTRY[_US_NAME]
    before = _entries(tmp_path)
    for need_dl in (False, True):                                       # a '[download] ' name does not override it
        with pytest.raises(pol.ModelNotDownloaded) as e:
            us.UnifiedSegmentation._resolve(reg, need_dl=need_dl)
        assert e.value.target == os.path.abspath(str(tmp_path / reg["model_dir"]))
        assert e.value.file_name == reg["filename"]
        assert e.value.target in str(e.value) and reg["filename"] in str(e.value)
        assert "allow_download" in str(e.value) and reg["repo_id"] in str(e.value)
    assert _entries(tmp_path) == before                                 # no model folder, no .cache marker


def test_unified_segmentation_download_guard_creates_nothing(us, pol, tmp_path):
    reg = us.MODEL_REGISTRY[_US_NAME]
    dest = tmp_path / "sam2"
    with pytest.raises(pol.ModelNotDownloaded):
        us.UnifiedSegmentation._download(reg, str(dest))                 # even when called directly
    assert not dest.exists()


def test_unified_segmentation_reuses_a_copy_in_another_known_folder(us, mm, tmp_path, monkeypatch):
    reg = us.MODEL_REGISTRY[_US_NAME]
    f = tmp_path / "other_pack" / reg["filename"]
    f.parent.mkdir()
    f.write_bytes(b"x")
    fp = types.SimpleNamespace(folder_names_and_paths={"other_pack_sams": ([str(f.parent)], set())},
                               get_full_path=lambda key, name: str(f) if name == f.name else None)
    monkeypatch.setattr(mm, "folder_paths", fp)
    assert us.UnifiedSegmentation._resolve(reg, need_dl=False) == str(f)
    assert not (tmp_path / reg["model_dir"]).exists()


def test_unified_segmentation_reuses_the_hf_cache(us, pol, tmp_path, monkeypatch):
    reg = us.MODEL_REGISTRY[_US_NAME]
    monkeypatch.setattr(pol, "hf_cached_file",
                        lambda repo, fn: "/hf/cache/" + fn if (repo, fn) == (reg["repo_id"], reg["filename"]) else None)
    assert us.UnifiedSegmentation._resolve(reg, need_dl=False) == "/hf/cache/" + reg["filename"]
    assert _entries(tmp_path) == []


def test_unified_segmentation_permitted_download_saves_into_its_folder(us, pol, tmp_path, monkeypatch):
    reg = us.MODEL_REGISTRY[_US_NAME]
    calls = []

    def fake_hub(repo_id, filename, local_dir):
        calls.append((repo_id, filename, local_dir))
        out = os.path.join(local_dir, filename)
        with open(out, "wb") as fh:
            fh.write(b"weights")
        return out
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_hub)
    dest = tmp_path / reg["model_dir"]
    assert not dest.exists()
    with pol.allow_downloads(True):
        path = us.UnifiedSegmentation._resolve(reg, need_dl=False)
    assert calls == [(reg["repo_id"], reg["filename"], os.path.abspath(str(dest)))]
    assert os.path.isfile(path) and os.path.dirname(path) == os.path.abspath(str(dest))   # the folder only now


def test_sec_pipeline_without_permission_uses_the_model_free_path(us, monkeypatch):
    """End to end through the node the gap was found in: SeC + MatAnyone, Allow Download off, no weights - a
    labelled model-free result, no download attempted, nothing created under models/."""
    sec = pytest.importorskip("nodes.sec_matanyone_pipeline")
    monkeypatch.setattr(sec.SeCMatAnyonePipelineMEC, "_run_vitmatte", lambda self, *a, **k: _missing())
    img, gt = scene()
    out = sec.SeCMatAnyonePipelineMEC().process(as_image(img), _US_NAME, "",
                                                json.dumps([{"x": CX, "y": CY, "label": 1}]), "", "vitmatte_small",
                                                8, 5, "fp32", False, 0)
    info = json.loads(out[-1])
    assert info["model_free"]["segmentation"]["wanted"] == _US_NAME
    assert "downloading is off" in info["model_free"]["segmentation"]["why"]
    assert iou(out[1][0].numpy(), gt) >= 0.8


def test_sam_loader_message_names_the_absolute_folder_and_file(monkeypatch, tmp_path):
    sml = pytest.importorskip("nodes.sam_model_loader")
    L = sml.SAMModelLoaderMEC
    fp = types.SimpleNamespace(folder_names_and_paths={}, models_dir=str(tmp_path), base_path=str(tmp_path),
                               get_full_path=lambda *a: None)
    monkeypatch.setattr(sml, "folder_paths", fp, raising=False)
    monkeypatch.setattr(sml, "HAS_FOLDER_PATHS", True)
    monkeypatch.setattr(L, "_auto_download", staticmethod(lambda n: pytest.fail("must not download")))
    from nodes._download_policy import ModelNotDownloaded
    name = "sam2.1_hiera_large.pt"
    with pytest.raises(ModelNotDownloaded) as e:
        L._resolve_path(f"[download] {name}", download_missing_registry=True)
    want = os.path.abspath(str(tmp_path / "sam2"))
    assert e.value.target == want and e.value.file_name == name
    assert os.path.isabs(e.value.target) and want in str(e.value) and f"'{name}'" in str(e.value)
    assert not (tmp_path / "sam2").exists()
    # the stand-in the node hands out carries the same message
    (sam,) = L().load(f"[download] {name}", "auto", "cpu", False, "float32")
    assert want in sam["fallback_from"]["why"] and name in sam["fallback_from"]["why"]


# ── nodes: model-free where one exists, a plain message where not ────────────────────────────────────────────────

def _missing(*_a, **_k):
    from nodes._download_policy import ModelNotDownloaded
    raise ModelNotDownloaded("some_model", target="models/x/")


def test_background_remover_model_free(monkeypatch):
    br = pytest.importorskip("nodes.background_remover")
    monkeypatch.setattr(br, "get_or_load_model", _missing)
    img, gt = scene()
    fg, mask, info = br.BackgroundRemoverMEC().remove_bg(as_image(img), "birefnet_general", 0.5, False, 0)
    info = json.loads(info)
    assert iou(mask[0].numpy(), gt) >= 0.8
    assert info["approximate"] is True and info["model_status"].startswith("missing")
    assert "border" in info["model_free"]["used"]


def test_semantic_segment_skin_only_without_the_model(monkeypatch):
    ss = pytest.importorskip("nodes.semantic_segment")
    monkeypatch.setattr(ss, "get_or_load_model", _missing)
    img, gt = scene(skin=True)
    mask, info = ss.SemanticSegmentMEC().parse(as_image(img), "segformer_face", "skin,hair", 0.5, False)
    info = json.loads(info)
    assert iou(mask[0].numpy(), gt) >= 0.8
    assert info["approximate"] is True and info["needs_the_model"] == ["hair"]
    with pytest.raises(ValueError, match="needs the model"):
        ss.SemanticSegmentMEC().parse(as_image(img), "segformer_clothes", "upper_clothes", 0.5, False)


def test_multi_mask_picker_without_a_loader(monkeypatch):
    pk = pytest.importorskip("nodes.sam_multi_mask_picker")
    monkeypatch.setattr(pk, "get_or_load_model", _missing)
    img, gt = scene()
    out = pk.SamMultiMaskPickerMEC().pick_mask(as_image(img), "sam2.1_hiera_base_plus",
                                               json.dumps([{"x": CX, "y": CY, "label": 1}]), "", "fp32", 0)
    out = out["result"] if isinstance(out, dict) else out
    info = json.loads(out[-1])
    assert iou(out[0][0].numpy(), gt) >= 0.85 and info["approximate"] is True


def test_sam_loader_download_entry_needs_permission(monkeypatch):
    sml = pytest.importorskip("nodes.sam_model_loader")
    L = sml.SAMModelLoaderMEC
    monkeypatch.setattr(sml, "HAS_FOLDER_PATHS", False)
    called = []
    monkeypatch.setattr(L, "_auto_download", staticmethod(lambda n: called.append(n) or "/x/" + n))
    (sam,) = L().load("[download] sam2.1_hiera_large.pt", "auto", "cpu", False, "float32")
    assert called == [] and sam["model_type"] == "model-free"
    assert "downloading is off" in sam["fallback_from"]["why"]
    monkeypatch.setattr(L, "_load_sam2_family", lambda *a, **k: (object(), "stub"))
    (sam,) = L().load("[download] sam2.1_hiera_large.pt", "auto", "cpu", False, "float32", allow_download=True)
    assert called == ["sam2.1_hiera_large.pt"] and sam["load_method"] == "stub"


def test_sec_pipeline_model_free(monkeypatch):
    sec = pytest.importorskip("nodes.sec_matanyone_pipeline")
    us = pytest.importorskip("nodes.unified_segmentation")
    monkeypatch.setattr(us.UnifiedSegmentation, "segment", lambda self, **k: _missing())
    monkeypatch.setattr(sec.SeCMatAnyonePipelineMEC, "_run_vitmatte", lambda self, *a, **k: _missing())
    img, gt = scene()
    out = sec.SeCMatAnyonePipelineMEC().process(as_image(img), "sam2.1_hiera_base_plus", "",
                                                json.dumps([{"x": CX, "y": CY, "label": 1}]), "", "vitmatte_small",
                                                8, 5, "fp32", False, 0)
    info = json.loads(out[-1])
    assert iou(out[1][0].numpy(), gt) >= 0.8
    assert info["approximate"] is True and set(info["model_free"]) == {"segmentation", "matting"}


# ── L7.97 gaps (owner status note 2026-10-13): LocateAnything + SAM 3.1 name the absolute folder and file ───────────

_LA = "nvidia/LocateAnything-3B"


def _hf_folder(p, weights=True):
    p.mkdir(parents=True)
    (p / "config.json").write_text("{}")
    if weights:
        (p / "model.safetensors").write_bytes(b"x")
    return p


def test_transformers_repo_missing_names_the_absolute_folder(mm, pol, tmp_path):
    with pytest.raises(pol.ModelNotDownloaded) as e:
        mm.resolve_transformers_repo(_LA, "LLM")
    want = os.path.abspath(str(tmp_path / "LLM" / "LocateAnything-3B"))
    assert e.value.target == want and os.path.isabs(e.value.target) and want in str(e.value)
    assert "huggingface.co/nvidia/LocateAnything-3B" in str(e.value) and "allow_download" in str(e.value)
    assert not (tmp_path / "LLM").exists()                               # nothing created without permission


def test_transformers_repo_found_in_own_folder_other_pack_or_cache(mm, pol, tmp_path, monkeypatch):
    own = _hf_folder(tmp_path / "LLM" / "LocateAnything-3B")
    assert mm.resolve_transformers_repo(_LA, "LLM") == str(own)
    other = tmp_path / "other"
    other.mkdir()
    assert mm.resolve_transformers_repo(str(other), "LLM") == str(other)   # a local folder is used as given
    (own / "model.safetensors").unlink()                                 # half a folder is not an installed copy
    pack = _hf_folder(tmp_path / "VLM" / "LocateAnything-3B")
    assert mm.resolve_transformers_repo(_LA, "LLM") == str(pack)
    (pack / "model.safetensors").unlink()
    cache = _hf_folder(tmp_path / "hf" / "snap")
    monkeypatch.setattr(pol, "hf_cached_snapshot", lambda repo: str(cache) if repo == _LA else None)
    assert mm.resolve_transformers_repo(_LA, "LLM") == str(cache)
    (cache / "model.safetensors").unlink()                               # e.g. only the tokenizer was cached
    with pytest.raises(pol.ModelNotDownloaded):
        mm.resolve_transformers_repo(_LA, "LLM")


def test_transformers_repo_permitted_download_goes_into_its_folder(mm, pol, tmp_path, monkeypatch):
    hub = pytest.importorskip("huggingface_hub")
    got = []
    monkeypatch.setattr(hub, "snapshot_download", lambda **k: got.append(k) or k["local_dir"])
    with pol.allow_downloads(True):
        p = mm.resolve_transformers_repo(_LA, "LLM")
    want = os.path.abspath(str(tmp_path / "LLM" / "LocateAnything-3B"))
    assert p == want and got == [{"repo_id": _LA, "local_dir": want}]


def test_transformers_repo_bad_local_path(mm, pol, tmp_path):
    bad = str(tmp_path / "no" / "such" / "folder")
    with pytest.raises(FileNotFoundError) as e:
        mm.resolve_transformers_repo(bad, "LLM")
    assert not isinstance(e.value, pol.ModelNotDownloaded) and os.path.abspath(bad) in str(e.value)


def test_locate_anything_node_message_and_no_transformers_load(mm, pol, tmp_path, monkeypatch):
    la = pytest.importorskip("nodes.locate_anything")
    tf = pytest.importorskip("transformers")
    for cls in ("AutoModel", "AutoTokenizer", "AutoProcessor"):
        monkeypatch.setattr(getattr(tf, cls), "from_pretrained",
                            classmethod(lambda c, *a, **k: pytest.fail("must not load before the model is found")))
    monkeypatch.setattr(la.LocateAnythingGroundingMEC, "_model_cache", {})
    img = torch.zeros(1, 32, 32, 3)
    with pytest.raises(pol.ModelNotDownloaded) as e:
        la.LocateAnythingGroundingMEC().run(img, "person", device="cpu")
    assert os.path.abspath(str(tmp_path / "LLM" / "LocateAnything-3B")) in str(e.value)
    assert not (tmp_path / "LLM").exists()


def test_locate_anything_segmenter_same_message(mm, pol, tmp_path):
    seg = pytest.importorskip("nodes.mask_matting.segmenters.locate_anything_backend")
    pytest.importorskip("transformers")
    s = seg.LocateAnythingSegmenter(device="cpu")
    with pytest.raises(pol.ModelNotDownloaded) as e:
        s.load()
    assert os.path.abspath(str(tmp_path / "LLM" / "LocateAnything-3B")) in str(e.value)
    assert "'auto_download'" in str(e.value)                           # MaskOps' own switch, not the node's


@pytest.fixture
def mm_utils(monkeypatch, tmp_path):
    """MaskOps' folder look-up as in a real ComfyUI: models_dir set, sam3.1 registered there."""
    utils = pytest.importorskip("nodes.mask_matting.utils")
    fp = types.SimpleNamespace(models_dir=str(tmp_path), base_path=str(tmp_path),
                               get_folder_paths=lambda k: [str(tmp_path / "sam3.1")] if k == "sam3.1" else [])
    monkeypatch.setattr(utils, "folder_paths", fp)
    return utils


def test_backend_first_root_survives_a_partial_folder_paths(monkeypatch):
    utils = pytest.importorskip("nodes.mask_matting.utils")
    monkeypatch.setattr(utils, "folder_paths", types.SimpleNamespace())    # no models_dir: it raised AttributeError
    assert utils.backend_first_root("birefnet").replace("\\", "/").endswith("models/BiRefNet")


def test_maskops_preset_message_absolute_folder_file_and_source(pol, monkeypatch, mm_utils, tmp_path):
    node = pytest.importorskip("nodes.mask_matting.node")
    utils = mm_utils
    monkeypatch.setattr(node, "list_backend_files", lambda key: [])
    monkeypatch.setattr(node, "download_preset", lambda *a, **k: pytest.fail("must not download"))
    name = "sam3.1_multiplex_fp16.safetensors"
    with pytest.raises(pol.ModelNotDownloaded) as e:
        node._resolve_model_choice(f"[preset:sam3.1] {name}", "sam3.1", auto_download=False)
    root = os.path.abspath(str(tmp_path / "sam3.1"))
    assert e.value.target == root and e.value.file_name == name
    assert root in str(e.value) and "huggingface.co/Comfy-Org/sam3.1" in str(e.value)
    assert "'auto_download'" in str(e.value) and "models/sam3.1/" not in str(e.value)
    with pytest.raises(FileNotFoundError):                               # callers catching FileNotFoundError still do
        node._resolve_model_choice(f"[preset:sam3.1] {name}", "sam3.1", auto_download=False)


def test_sam31_missing_checkpoint_and_library_messages_are_absolute(monkeypatch, mm_utils, tmp_path):
    b = pytest.importorskip("nodes.mask_matting.segmenters.sam31_backend")
    onyx = pytest.importorskip("nodes.mask_matting._onyx_pipeline")
    root = os.path.abspath(str(tmp_path / "sam3.1"))
    assert os.path.abspath(mm_utils.backend_first_root("sam3.1")) == root
    monkeypatch.setattr(b, "_have_vendor", lambda: True)
    monkeypatch.setattr(b, "resolve_backend_weight", lambda key, name: None)
    with pytest.raises(FileNotFoundError) as e:
        b.SAM31Segmenter(model_name="", device="cpu").load()             # "(auto)" with nothing installed
    assert root in str(e.value) and "'sam3.1_multiplex_fp16.safetensors'" in str(e.value)
    with pytest.raises(FileNotFoundError) as e:
        b.SAM31Segmenter(model_name="mine.safetensors", device="cpu").load()
    assert root in str(e.value) and "'mine.safetensors'" in str(e.value)
    monkeypatch.setattr(b, "_have_vendor", lambda: False)
    with pytest.raises(RuntimeError) as e:
        onyx.ensure_sam31_video_model(model_name="", device="cpu", precision="fp32", attention="auto",
                                      offload="none", auto_download=False)
    assert root in str(e.value) and str(b._vendor_root()) in str(e.value) and "models/sam3.1/" not in str(e.value)


def test_maskops_vitmatte_missing_names_the_absolute_folder(mm, pol, tmp_path, monkeypatch, mm_utils):
    vb = pytest.importorskip("nodes.mask_matting.matters.vitmatte_backend")
    tf = pytest.importorskip("transformers")
    for cls in ("VitMatteForImageMatting", "VitMatteImageProcessor"):
        monkeypatch.setattr(getattr(tf, cls), "from_pretrained",
                            classmethod(lambda c, *a, **k: pytest.fail("must not load before the model is found")))
    m = vb.ViTMatteMatter(model_name="", device="cpu", precision="fp32")
    with pytest.raises(pol.ModelNotDownloaded) as e:
        m.load()
    want = os.path.abspath(str(tmp_path / "vitmatte" / "vitmatte-small-composition-1k"))
    assert e.value.target == want and want in str(e.value)
    assert "huggingface.co/hustvl/vitmatte-small-composition-1k" in str(e.value)
    assert "'auto_download'" in str(e.value)
    assert not (tmp_path / "vitmatte").exists()


@pytest.mark.parametrize("mod, cls, folder, repo", [
    ("salient_backend", "BiRefNetSegmenter", "BiRefNet", "ZhengPeng7/BiRefNet"),
    ("salient_backend", "RMBG2Segmenter", "RMBG", "briaai/RMBG-2.0"),
    ("experimental_backend", "DISSegmenter", "dis", "briaai/RMBG-1.4"),
])
def test_maskops_segmenters_missing_name_the_absolute_folder(mm, pol, tmp_path, monkeypatch, mm_utils,
                                                             mod, cls, folder, repo):
    m = pytest.importorskip(f"nodes.mask_matting.segmenters.{mod}")
    tf = pytest.importorskip("transformers")
    monkeypatch.setattr(tf.AutoModelForImageSegmentation, "from_pretrained",
                        classmethod(lambda c, *a, **k: pytest.fail("must not load before the model is found")))
    with pytest.raises(pol.ModelNotDownloaded) as e:
        getattr(m, cls)(model_name="", device="cpu").load()
    want = os.path.abspath(str(tmp_path / folder / repo.split("/")[-1]))
    assert e.value.target == want and f"huggingface.co/{repo}" in str(e.value)
    assert "'auto_download'" in str(e.value)
    assert not (tmp_path / folder).exists()


def test_maskops_dinov2_reanchor_missing_names_the_absolute_folder(mm, pol, tmp_path, monkeypatch):
    ra = pytest.importorskip("nodes.mask_matting._reanchor")
    pytest.importorskip("transformers")
    monkeypatch.setattr(ra, "_DINO_MODEL", None)
    with pytest.raises(pol.ModelNotDownloaded) as e:
        ra._load_dinov2("cpu")
    assert e.value.target == os.path.abspath(str(tmp_path / "dino" / "dinov2-small"))
    assert "'auto_download'" in str(e.value)
