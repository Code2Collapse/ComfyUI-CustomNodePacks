"""C2C memory guard — release pack caches when ComfyUI frees VRAM.

CPU-only tests use a fake ``comfy.model_management``. One optional CUDA test
uses the real ComfyUI install when present.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import patch

import pytest

WORKSPACE = Path(__file__).resolve().parents[2]
PACK = Path(__file__).resolve().parents[1]
CANONICAL = PACK / "nodes" / "_c2c_memguard.py"
WAP_COPY = WORKSPACE / "ComfyUI-WanAnimatePreprocessV2" / "nodes_extras" / "_c2c_memguard.py"
COMFY_ROOT = Path("D:/PROJECT/ComfyUI_windows_portable/ComfyUI")


def _load_guard(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def fake_mm(monkeypatch):
    """Inject a recorder fake comfy.model_management into sys.modules."""
    calls = {
        "free_memory": [],
        "soft_empty_cache": 0,
        "gc_collect": 0,
    }

    def free_memory(memory_required, device, *args, **kwargs):
        calls["free_memory"].append((memory_required, device, args, kwargs))
        return []

    def soft_empty_cache(force=False):
        calls["soft_empty_cache"] += 1

    mm = types.SimpleNamespace(
        free_memory=free_memory,
        soft_empty_cache=soft_empty_cache,
    )
    monkeypatch.setitem(sys.modules, "comfy.model_management", mm)
    monkeypatch.setitem(sys.modules, "comfy", types.SimpleNamespace(model_management=mm))

    # Fresh guard module per test
    name = f"_c2c_memguard_test_{id(calls)}"
    guard = _load_guard(CANONICAL, name)
    sys.modules[name] = guard
    return guard, mm, calls


class FakeCudaDevice:
    type = "cuda"


class FakeCpuDevice:
    type = "cpu"


def test_byte_identical_copy_in_wap():
    if not WAP_COPY.parent.parent.exists():
        pytest.skip("WAP pack not in workspace")
    assert WAP_COPY.exists()
    assert WAP_COPY.read_text(encoding="utf-8") == CANONICAL.read_text(encoding="utf-8")


def test_install_twice_wraps_once(fake_mm):
    guard, mm, _calls = fake_mm
    assert guard.install() is True
    assert guard.install() is False
    assert getattr(mm, "_c2c_memguard_installed") is True
    assert callable(getattr(mm, "_c2c_memguard_orig_free_memory", None))
    assert mm.free_memory is not getattr(mm, "_c2c_memguard_orig_free_memory")


def test_free_memory_cuda_calls_release_before_orig(fake_mm):
    guard, mm, calls = fake_mm
    released = []

    guard.register("a", lambda: released.append("a"), lambda: True)
    guard.install()

    mm.free_memory(1e9, FakeCudaDevice(), keep_loaded=["x"], for_dynamic=True)

    assert released == ["a"]
    assert len(calls["free_memory"]) == 1
    mem_req, device, args, kwargs = calls["free_memory"][0]
    assert mem_req == 1e9
    assert device.type == "cuda"
    assert args == ()
    assert kwargs == {"keep_loaded": ["x"], "for_dynamic": True}


def test_free_memory_cpu_skips_release(fake_mm):
    guard, mm, calls = fake_mm
    released = []

    guard.register("a", lambda: released.append("a"), lambda: True)
    guard.install()
    mm.free_memory(1e9, FakeCpuDevice())

    assert released == []
    assert len(calls["free_memory"]) == 1


def test_free_memory_none_device_skips_release(fake_mm):
    guard, mm, _calls = fake_mm
    released = []

    guard.register("a", lambda: released.append("a"), lambda: True)
    guard.install()
    mm.free_memory(1e9, None)

    assert released == []


def test_raising_release_does_not_stop_others_or_orig(fake_mm):
    guard, mm, calls = fake_mm
    seen = []

    def bad():
        seen.append("bad")
        raise RuntimeError("boom")

    guard.register("bad", bad, lambda: True)
    guard.register("good", lambda: seen.append("good"), lambda: True)
    guard.install()
    mm.free_memory(1e9, FakeCudaDevice())

    assert seen == ["bad", "good"]
    assert len(calls["free_memory"]) == 1


def test_unload_all_models_path_triggers_release(fake_mm):
    guard, mm, calls = fake_mm
    released = []

    guard.register("a", lambda: released.append("a"), lambda: True)
    guard.install()

    def unload_all_models():
        mm.free_memory(1e30, FakeCudaDevice())

    mm.unload_all_models = unload_all_models
    mm.unload_all_models()

    assert released == ["a"]
    assert len(calls["free_memory"]) == 1


def test_shared_registry_across_two_module_copies(fake_mm):
    guard_a, mm, _calls = fake_mm
    guard_b = _load_guard(WAP_COPY, f"_c2c_memguard_wap_{id(mm)}")

    guard_a.register("from_a", lambda: None, lambda: False)
    guard_b.register("from_b", lambda: None, lambda: False)

    reg = getattr(mm, "_c2c_memguard_registry")
    assert "from_a" in reg and "from_b" in reg


def test_nothing_loaded_skips_gc_and_soft_empty_cache(fake_mm):
    guard, mm, calls = fake_mm

    guard.register("empty", lambda: (_ for _ in ()).throw(AssertionError("should not run")),
                   lambda: False)
    guard.install()

    with patch.object(guard, "gc") as mock_gc:
        guard.release_all("test")
        mock_gc.collect.assert_not_called()

    assert calls["soft_empty_cache"] == 0


def test_only_loaded_entries_are_released(fake_mm):
    guard, mm, calls = fake_mm
    released = []

    guard.register("loaded", lambda: released.append("loaded"), lambda: True)
    guard.register("empty", lambda: released.append("empty"), lambda: False)
    guard.install()

    with patch.object(guard, "gc") as mock_gc:
        n = guard.release_all("test")
        mock_gc.collect.assert_called_once()

    assert n == 1
    assert released == ["loaded"]
    assert calls["soft_empty_cache"] == 1


def test_raising_is_loaded_counts_as_loaded(fake_mm):
    guard, _mm, _calls = fake_mm
    released = []

    def boom():
        raise RuntimeError("is_loaded broke")

    guard.register("x", lambda: released.append("x"), boom)
    n = guard.release_all("test")

    assert n == 1
    assert released == ["x"]


@pytest.mark.parametrize("registry_name,module_import,cache_attr,seed", [
    ("cnp.control_backends", "nodes._control_backends", "_CACHE", lambda c: c.update({"x": 1})),
    ("cnp.vitmatte", "nodes.utils", "_vitmatte_model", lambda c: setattr(c, "_vitmatte_model", object())),
    ("cnp.dinov2", "nodes.mask_matting._reanchor", "_DINO_MODEL", lambda c: setattr(c, "_DINO_MODEL", (1, 2, 3))),
    ("cnp.raft", "nodes.mask_matting.temporal_node", "_RAFT_MODEL", lambda c: setattr(c, "_RAFT_MODEL", (1, 2))),
    ("cnp.unified_segmentation", "nodes.unified_segmentation", "_cache",
     lambda c: c.update({"model": object(), "name": "x"})),
])
def test_cnp_cache_site_registers_and_release_clears(fake_mm, registry_name, module_import, cache_attr, seed, monkeypatch):
    guard, mm, _calls = fake_mm
    if str(PACK) not in sys.path:
        sys.path.insert(0, str(PACK))
    # unified_segmentation reads folder_paths.base_path at import; the test
    # environment's folder_paths may be a stub without it.
    try:
        import folder_paths
        monkeypatch.setattr(folder_paths, "base_path", str(COMFY_ROOT), raising=False)
    except Exception:
        pass

    mod = importlib.import_module(module_import)
    importlib.reload(mod)

    reg = getattr(mm, "_c2c_memguard_registry", {})
    assert registry_name in reg

    if cache_attr == "_CACHE":
        seed(mod._CACHE)
        assert mod._CACHE
        guard.release_all("test")
        assert not mod._CACHE
    elif cache_attr == "_vitmatte_model":
        seed(mod)
        assert mod._vitmatte_model is not None
        guard.release_all("test")
        assert mod._vitmatte_model is None
    elif cache_attr == "_DINO_MODEL":
        seed(mod)
        assert mod._DINO_MODEL is not None
        guard.release_all("test")
        assert mod._DINO_MODEL is None
    elif cache_attr == "_RAFT_MODEL":
        seed(mod)
        assert mod._RAFT_MODEL is not None
        guard.release_all("test")
        assert mod._RAFT_MODEL is None
    elif cache_attr == "_cache":
        seed(mod._cache)
        assert mod._cache.get("model") is not None
        guard.release_all("test")
        assert mod._cache.get("model") is None


@pytest.mark.parametrize("registry_name,module_import,cache_attr,seed", [
    ("wap.pose3d_nlf", "nodes_extras.pose3d_nlf", "_NLF_CACHE", lambda c: c.update({"k": 1})),
    ("wap.gaze_ethxgaze", "nodes_extras.gaze_ethxgaze", "_MODEL_CACHE", lambda c: c.update({"k": 1})),
])
def test_wap_cache_site_registers_and_release_clears(fake_mm, registry_name, module_import, cache_attr, seed):
    guard, mm, _calls = fake_mm
    wap_root = WORKSPACE / "ComfyUI-WanAnimatePreprocessV2"
    if not wap_root.exists():
        pytest.skip("WAP pack not in workspace")
    # Import under a parent PACKAGE, as ComfyUI does: WAP's modules use
    # relative imports (`from ..`) that fail when nodes_extras is top-level.
    # The parent is a bare package - WAP's own __init__ (every node) is not run.
    pkg_name = "c2c_wap_test_pkg"
    if pkg_name not in sys.modules:
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = [str(wap_root)]
        sys.modules[pkg_name] = pkg
    mod = importlib.import_module(f"{pkg_name}.{module_import}")
    importlib.reload(mod)

    reg = getattr(mm, "_c2c_memguard_registry", {})
    assert registry_name in reg

    cache = getattr(mod, cache_attr)
    seed(cache)
    assert cache
    guard.release_all("test")
    assert not cache


_CUDA_PROBE = r'''
import importlib.util, json, sys
sys.path.insert(0, sys.argv[1])
import torch
import comfy.model_management as mm            # the REAL one
spec = importlib.util.spec_from_file_location("_c2c_memguard_cuda", sys.argv[2])
guard = importlib.util.module_from_spec(spec); spec.loader.exec_module(guard)
hold = []
guard.register("cuda_test", lambda: hold.clear(), lambda: bool(hold))
hold.append(torch.zeros(64 * 1024 * 1024, device="cuda", dtype=torch.float32))   # 256 MB
torch.cuda.synchronize(); before = torch.cuda.memory_allocated()
mm.free_memory(1e9, torch.device("cuda:0"))     # what load_models_gpu does before loading
torch.cuda.synchronize(); after = torch.cuda.memory_allocated()
print(json.dumps({"before": before, "after": after, "hold": len(hold)}))
'''


@pytest.mark.skipif(not COMFY_ROOT.is_dir(), reason="ComfyUI install not found")
def test_cuda_tensor_released_on_real_free_memory(tmp_path):
    """End to end on the real core: a model cache holding 256 MB of VRAM is
    released the moment ComfyUI asks for memory. Run in a clean process - the
    other tests put a FAKE comfy.model_management in sys.modules."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    import json
    import subprocess
    probe = tmp_path / "cuda_probe.py"
    probe.write_text(_CUDA_PROBE, encoding="utf-8")
    p = subprocess.run([sys.executable, str(probe), str(COMFY_ROOT), str(CANONICAL)],
                       capture_output=True, text=True, timeout=300, cwd=str(COMFY_ROOT))
    line = [l for l in p.stdout.splitlines() if l.startswith("{")]
    assert line, (p.stdout[-800:], p.stderr[-1500:])
    r = json.loads(line[-1])
    assert r["hold"] == 0, "the cache was not released"
    assert r["before"] - r["after"] >= 200 * 1024 * 1024, r
