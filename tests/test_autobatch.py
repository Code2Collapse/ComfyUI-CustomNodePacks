"""CPU tests for nodes/_autobatch.py — no ComfyUI on sys.path."""
from __future__ import annotations

import json
import inspect
import logging
import sys
import types
from pathlib import Path
import pytest
import torch

PACK = Path(__file__).resolve().parents[1]
if str(PACK) not in sys.path:
    sys.path.insert(0, str(PACK))

from nodes import _autobatch as ab


@pytest.fixture(autouse=True)
def _reset_autobatch_state(monkeypatch):
    monkeypatch.delenv("C2C_AUTOBATCH", raising=False)
    monkeypatch.delenv("C2C_AUTOBATCH_CONFIG", raising=False)
    ab._WRAPPED.clear()
    ab._MACHINERY_FAILED.clear()
    ab._DICT_RETURN_LABELS.clear()
    ab._CONFIG_MTIME = None
    ab._CONFIG_CACHE = None
    ab._PROMPT_HOOK_REGISTERED = False
    ab._ROUTES_REGISTERED = False
    ab._WRAP_STRICT.clear()
    ab._SAFE.clear()
    ab._UNSAFE.clear()
    ab._REFUSAL_LOGGED.clear()
    ab._MEASURED_BPF.clear()
    ab._CALL_HISTORY.clear()
    yield


@pytest.fixture
def config_dir(tmp_path, monkeypatch):
    cfg_path = tmp_path / "autobatch.json"
    monkeypatch.setenv("C2C_AUTOBATCH_CONFIG", str(cfg_path))
    fp = types.SimpleNamespace(get_user_directory=lambda: str(tmp_path))
    monkeypatch.setitem(sys.modules, "folder_paths", fp)
    return cfg_path


def _write_config(path: Path, **kwargs) -> None:
    data = {"enabled": False, "curated": False, "nodes": {}}
    data.update(kwargs)
    path.write_text(json.dumps(data), encoding="utf-8")


def _per_frame_mul(image, mask=None, **_kw):
    if mask is None:
        return (image * 0.5,)
    m = mask
    if m.ndim == 3:
        m = m.unsqueeze(-1)
    return (image * m,)


def test_bit_exact_chunked_vs_unchunked_with_broadcast_mask():
    b, h, w = 20, 8, 8
    image = torch.arange(b * h * w * 3, dtype=torch.float32).reshape(b, h, w, 3) / 1000.0
    mask = torch.linspace(0, 1, h * w, dtype=torch.float32).reshape(1, h, w)
    frames = {"image": image, "mask": mask}
    static: dict = {}

    tiny = 1
    chunked = ab.run_chunked(
        _per_frame_mul, frames, static,
        label="test.mul", budget_bytes=tiny,
    )
    direct = ab.run_chunked(
        _per_frame_mul, frames, static,
        label="test.mul", budget_bytes=10 ** 12,
    )
    assert len(chunked) == len(direct) == 1
    assert torch.equal(chunked[0], direct[0])


def test_calibration_first_chunk_smaller():
    sizes: list[int] = []

    def fn(image, **_kw):
        sizes.append(int(image.shape[0]))
        return (image + 1.0,)

    image = torch.zeros(30, 4, 4, 3)
    ab.run_chunked(
        fn, {"image": image}, {},
        label="cal", budget_bytes=20000,
    )
    assert sizes == [8, 20, 2]
    assert sum(sizes) == 30


def test_oom_halving_logged(caplog):
    caplog.set_level(logging.INFO, logger="C2C.autobatch")

    def fn(image, **_kw):
        if image.shape[0] > 4:
            raise MemoryError("simulated OOM")
        return (image.clone(),)

    image = torch.zeros(24, 2, 2, 3)
    out = ab.run_chunked(
        fn, {"image": image}, {},
        label="oom", budget_bytes=2400,
    )
    assert out[0].shape[0] == 24
    assert any("halving" in r.message.lower() for r in caplog.records)


def test_oom_chunk_one_plain_english():
    def fn(image, **_kw):
        raise MemoryError("simulated OOM")

    image = torch.zeros(10, 2, 2, 3)
    with pytest.raises(ab.AutobatchError, match="single frame"):
        ab.run_chunked(
            fn, {"image": image}, {},
            label="oom1", budget_bytes=100,
        )


def test_abort_non_tensor_output_differs():
    calls = {"n": 0}

    def fn(image, **_kw):
        calls["n"] += 1
        meta = "a" if calls["n"] == 1 else "b"
        return (image.clone(), meta)

    image = torch.zeros(20, 2, 2, 3)
    with pytest.raises(ab.AutobatchError, match="non-tensor output"):
        ab.run_chunked(
            fn, {"image": image}, {},
            label="meta", budget_bytes=400,
        )


def test_abort_frame_count_mismatch():
    def fn(image, **_kw):
        return (image[: max(1, image.shape[0] - 1)].clone(),)

    image = torch.zeros(20, 2, 2, 3)
    with pytest.raises(ab.AutobatchError, match="frame count"):
        ab.run_chunked(
            fn, {"image": image}, {},
            label="drop", budget_bytes=400,
        )


def test_kill_switch_single_call(monkeypatch):
    calls = {"n": 0}

    def fn(image, **_kw):
        calls["n"] += 1
        return (image.clone(),)

    monkeypatch.setenv("C2C_AUTOBATCH", "0")
    image = torch.zeros(50, 2, 2, 3)
    ab.run_chunked(
        fn, {"image": image}, {},
        label="kill", budget_bytes=1,
    )
    assert calls["n"] == 1


def test_decorator_on_fake_node():
    class FakeScale:
        @ab.autobatch(frames=("image",))
        def execute(self, image, strength=1.0):
            return (image * strength,)

    node = FakeScale()
    image = torch.ones(24, 3, 3, 3)
    chunked = node.execute(image, 2.0)
    direct = FakeScale.execute.__wrapped__(node, image, 2.0)
    assert torch.equal(chunked[0], direct[0])


def test_decorator_on_fake_node_keyword():
    class FakeScale:
        @ab.autobatch(frames=("image",))
        def execute(self, image, strength=1.0):
            return (image * strength,)

    node = FakeScale()
    image = torch.ones(24, 3, 3, 3)
    chunked = node.execute(image, strength=2.0)
    direct = FakeScale.execute.__wrapped__(node, image=image, strength=2.0)
    assert torch.equal(chunked[0], direct[0])


def test_install_uninstall_wrappers(config_dir, caplog):
    caplog.set_level(logging.WARNING, logger="C2C.autobatch")
    _write_config(
        config_dir,
        enabled=True,
        nodes={
            "GoodNode": {"frames": ["image"]},
            "BadOutput": {"frames": ["image"]},
            "BadList": {"frames": ["image"]},
        },
    )

    class GoodNode:
        FUNCTION = "execute"
        OUTPUT_NODE = False
        INPUT_IS_LIST = False

        def execute(self, image):
            return (image,)

    class BadOutput:
        FUNCTION = "execute"
        OUTPUT_NODE = True
        INPUT_IS_LIST = False

        def execute(self, image):
            return (image,)

    class BadList:
        FUNCTION = "execute"
        OUTPUT_NODE = False
        INPUT_IS_LIST = True

        def execute(self, image):
            return (image,)

    mappings = {
        "GoodNode": GoodNode,
        "BadOutput": BadOutput,
        "BadList": BadList,
    }
    orig = GoodNode.execute
    wrapped = ab.install_wrappers(mappings, ab.load_config())
    assert "GoodNode" in wrapped
    assert GoodNode.execute is not orig
    assert getattr(GoodNode, ab._ORIG_ATTR) is orig
    assert any("OUTPUT_NODE" in r.message for r in caplog.records)
    assert any("INPUT_IS_LIST" in r.message for r in caplog.records)

    ab.uninstall_wrappers()
    assert GoodNode.execute is orig
    assert not ab._WRAPPED


def test_machinery_failure_unwraps_and_reruns(monkeypatch):
    class WrapMe:
        FUNCTION = "execute"
        OUTPUT_NODE = False
        INPUT_IS_LIST = False

        def execute(self, image):
            return (image + 1.0,)

    mappings = {"WrapMe": WrapMe}
    cfg = {
        "enabled": True,
        "curated": False,
        "nodes": {"WrapMe": {"frames": ["image"], "max_frames": 4}},
    }
    orig = WrapMe.execute
    ab.install_wrappers(mappings, cfg)

    def _boom(*_a, **_k):
        raise RuntimeError("machinery bug")

    monkeypatch.setattr(ab, "_allocate_output_buffers", _boom)
    image = torch.zeros(12, 2, 2, 3)
    out = WrapMe().execute(image)
    assert torch.allclose(out[0], image + 1.0)
    assert WrapMe.execute is orig
    assert "WrapMe" in ab._MACHINERY_FAILED


def test_on_prompt_reload_on_mtime_change(config_dir, monkeypatch):
    fake_nodes = types.ModuleType("nodes")
    fake_nodes.NODE_CLASS_MAPPINGS = {}
    monkeypatch.setitem(sys.modules, "nodes", fake_nodes)

    _write_config(config_dir, enabled=True, nodes={})
    payload = {"prompt": 1}
    assert ab.on_prompt(payload) is payload
    first_mtime = ab._CONFIG_MTIME

    class NodeA:
        FUNCTION = "execute"
        OUTPUT_NODE = False
        INPUT_IS_LIST = False

        def execute(self, image):
            return (image,)

    fake_nodes.NODE_CLASS_MAPPINGS = {"NodeA": NodeA}
    _write_config(
        config_dir,
        enabled=True,
        nodes={"NodeA": {"frames": ["image"]}},
    )
    import os
    os.utime(config_dir, (config_dir.stat().st_atime, config_dir.stat().st_mtime + 2))

    assert ab.on_prompt(payload) is payload
    assert ab._CONFIG_MTIME != first_mtime
    assert "NodeA" in ab._WRAPPED

    assert ab.on_prompt(payload) is payload
    assert len(ab._WRAPPED) == 1


def test_dict_return_two_calls_then_one(caplog):
    caplog.set_level(logging.INFO, logger="C2C.autobatch")
    calls = {"n": 0}

    def fn(image, **_kw):
        calls["n"] += 1
        if image.shape[0] < 20:
            return {"ui": {}, "result": (image.clone(),)}
        return {"ui": {}, "result": (image.clone(),)}

    image = torch.zeros(20, 2, 2, 3)
    out1 = ab.run_chunked(
        fn, {"image": image}, {},
        label="DictNode", budget_bytes=400,
    )
    assert calls["n"] == 2
    assert isinstance(out1, dict)
    assert any("ui dict" in r.message for r in caplog.records)

    calls["n"] = 0
    out2 = ab.run_chunked(
        fn, {"image": image}, {},
        label="DictNode", budget_bytes=400,
    )
    assert calls["n"] == 1
    assert isinstance(out2, dict)


class _FakeInterrupt(BaseException):
    """Like comfy.model_management.InterruptProcessingException: a BaseException, not an Exception."""


def test_interrupt_between_chunks(monkeypatch):
    checks = {"n": 0}

    def interrupt():
        checks["n"] += 1
        if checks["n"] >= 2:
            raise _FakeInterrupt()

    monkeypatch.setattr(ab, "_throw_interrupt", interrupt)

    def fn(image, **_kw):
        return (image.clone(),)

    image = torch.zeros(24, 2, 2, 3)
    with pytest.raises(_FakeInterrupt):
        ab.run_chunked(
            fn, {"image": image}, {},
            label="intr", budget_bytes=500,
        )
    assert checks["n"] >= 2


def test_progress_updates_sum_to_batch(monkeypatch):
    updates: list[int] = []

    class FakeBar:
        def __init__(self, total):
            self.total = total

        def update(self, n):
            updates.append(int(n))

    monkeypatch.setattr(ab, "_progress_bar", FakeBar)

    def fn(image, **_kw):
        return (image.clone(),)

    b = 25
    image = torch.zeros(b, 2, 2, 3)
    ab.run_chunked(
        fn, {"image": image}, {},
        label="prog", budget_bytes=600,
    )
    assert sum(updates) == b


def test_autobatch_error_not_unwrapped_by_wrapper():
    class BadMeta:
        FUNCTION = "execute"
        OUTPUT_NODE = False
        INPUT_IS_LIST = False

        calls = 0

        def execute(self, image):
            BadMeta.calls += 1
            if BadMeta.calls == 1:
                return (image.clone(), "a")
            return (image.clone(), "b")

    mappings = {"BadMeta": BadMeta}
    cfg = {
        "enabled": True,
        "curated": False,
        "nodes": {"BadMeta": {"frames": ["image"], "max_frames": 4}},
    }
    ab.install_wrappers(mappings, cfg)
    image = torch.zeros(16, 2, 2, 3)
    with pytest.raises(ab.AutobatchError):
        BadMeta().execute(image)
    assert getattr(BadMeta, ab._ORIG_ATTR, None) is not None


def test_node_exception_propagates_unchanged():
    class BoomNode:
        FUNCTION = "execute"
        OUTPUT_NODE = False
        INPUT_IS_LIST = False

        def execute(self, image):
            raise ValueError("node broke")

    mappings = {"BoomNode": BoomNode}
    cfg = {
        "enabled": True,
        "curated": False,
        "nodes": {"BoomNode": {"frames": ["image"], "max_frames": 4}},
    }
    ab.install_wrappers(mappings, cfg)
    with pytest.raises(ValueError, match="node broke"):
        BoomNode().execute(torch.zeros(8, 2, 2, 3))


def test_node_valueerror_mid_chunk_same_instance():
    err = ValueError("chunk fail")
    calls = {"n": 0}

    class FailThird:
        FUNCTION = "execute"
        OUTPUT_NODE = False
        INPUT_IS_LIST = False

        def execute(self, image):
            calls["n"] += 1
            if calls["n"] == 3:
                raise err
            return (image.clone(),)

    budget_mb = 2400 / (2 ** 20)
    mappings = {"FailThird": FailThird}
    cfg = {
        "enabled": True,
        "curated": False,
        "budget_mb": budget_mb,
        "nodes": {"FailThird": {"frames": ["image"], "max_frames": 4}},
    }
    ab.install_wrappers(mappings, cfg)
    wrapped_before = FailThird.execute
    with pytest.raises(ValueError) as excinfo:
        FailThird().execute(torch.zeros(24, 2, 2, 3))
    assert excinfo.value is err
    assert calls["n"] == 3
    assert FailThird.execute is wrapped_before
    assert "FailThird" in ab._WRAPPED


def test_missing_config_means_disabled(config_dir):
    _write_config(config_dir, enabled=True, nodes={"X": {"frames": ["image"]}})
    assert config_dir.is_file()
    config_dir.unlink()
    cfg = ab.load_config()
    assert cfg["enabled"] is False
    ab.on_prompt({})
    assert not ab._WRAPPED


class _FakeNodeOutput:
    def __init__(self, *args, ui=None, expand=None, block_execution=None):
        self.args = args
        self.ui = ui
        self.expand = expand
        self.block_execution = block_execution


class _FakeV3Base:
    define_schema = object()


def test_v3_install_uninstall_restores_identical_execute(config_dir):
    class V3Scale(_FakeV3Base):
        @classmethod
        def execute(cls, image, strength=1.0):
            return _FakeNodeOutput(image * strength)

    mappings = {"V3Scale": V3Scale}
    cfg = {
        "enabled": True,
        "curated": False,
        "nodes": {"V3Scale": {"frames": ["image"], "max_frames": 4}},
    }
    # A classmethod read through the class is a new bound method each time: compare the raw descriptor.
    orig = inspect.getattr_static(V3Scale, "execute")
    wrapped = ab.install_wrappers(mappings, cfg)
    assert "V3Scale" in wrapped
    assert inspect.getattr_static(V3Scale, "execute") is not orig
    assert inspect.getattr_static(V3Scale, ab._ORIG_ATTR) is orig

    ab.uninstall_wrappers()
    assert inspect.getattr_static(V3Scale, "execute") is orig
    assert not ab._WRAPPED


def test_v3_wrapped_execute_chunks_node_output():
    calls: list[int] = []

    class V3Mul(_FakeV3Base):
        @classmethod
        def execute(cls, image, strength=1.0):
            calls.append(int(image.shape[0]))
            return _FakeNodeOutput(image * strength)

    mappings = {"V3Mul": V3Mul}
    cfg = {
        "enabled": True,
        "curated": False,
        "nodes": {"V3Mul": {"frames": ["image"], "max_frames": 4}},
    }
    orig_func = V3Mul.execute.__func__
    ab.install_wrappers(mappings, cfg)
    image = torch.ones(12, 3, 3, 3)
    out = V3Mul.execute(image, strength=2.0)
    chunked_calls = list(calls)
    direct = orig_func(V3Mul, image=image, strength=2.0)
    assert isinstance(out, _FakeNodeOutput)
    assert chunked_calls == [4, 4, 4]          # max_frames caps every chunk, calibration included
    assert torch.equal(out.args[0], direct.args[0])


def test_v3_node_output_with_ui_runs_once_then_remembered(caplog):
    caplog.set_level(logging.INFO, logger="C2C.autobatch")
    calls = {"n": 0}

    class V3Ui(_FakeV3Base):
        @classmethod
        def execute(cls, image):
            calls["n"] += 1
            return _FakeNodeOutput(image.clone(), ui={"preview": True})

    mappings = {"V3Ui": V3Ui}
    cfg = {
        "enabled": True,
        "curated": False,
        "nodes": {"V3Ui": {"frames": ["image"], "max_frames": 4}},
    }
    ab.install_wrappers(mappings, cfg)
    image = torch.zeros(16, 2, 2, 3)
    out1 = V3Ui.execute(image)
    assert calls["n"] == 2
    assert out1.ui == {"preview": True}
    assert any("not chunked" in r.message for r in caplog.records)

    calls["n"] = 0
    out2 = V3Ui.execute(image)
    assert calls["n"] == 1
    assert out2.ui == {"preview": True}


def test_v3_async_execute_refused_with_warning(caplog):
    caplog.set_level(logging.WARNING, logger="C2C.autobatch")

    class V3Async(_FakeV3Base):
        @classmethod
        async def execute(cls, image):
            return _FakeNodeOutput(image)

    mappings = {"V3Async": V3Async}
    cfg = {
        "enabled": True,
        "curated": False,
        "nodes": {"V3Async": {"frames": ["image"], "max_frames": 4}},
    }
    wrapped = ab.install_wrappers(mappings, cfg)
    assert "V3Async" not in wrapped
    assert any("async execute" in r.message for r in caplog.records)


def test_v3_tuple_return_stays_tuple():
    calls: list[int] = []

    class V3Tuple(_FakeV3Base):
        @classmethod
        def execute(cls, image):
            calls.append(int(image.shape[0]))
            return (image + 1.0,)

    mappings = {"V3Tuple": V3Tuple}
    cfg = {
        "enabled": True,
        "curated": False,
        "nodes": {"V3Tuple": {"frames": ["image"], "max_frames": 4}},
    }
    orig_func = V3Tuple.execute.__func__
    ab.install_wrappers(mappings, cfg)
    image = torch.zeros(10, 2, 2, 3)
    out = V3Tuple.execute(image)
    chunked_calls = list(calls)
    direct = orig_func(V3Tuple, image=image)
    assert isinstance(out, tuple)
    assert not isinstance(out, _FakeNodeOutput)
    assert chunked_calls == [4, 4, 2]
    assert torch.equal(out[0], direct[0])


def _fake_v1_info(image_type="IMAGE", output_type="IMAGE"):
    return {
        "input": {"required": {"image": [image_type, {}]}, "optional": {}},
        "output": [output_type],
        "output_node": False,
        "output_is_list": False,
        "is_input_list": False,
    }


class _FakeV1Image:
    FUNCTION = "execute"
    OUTPUT_NODE = False
    INPUT_IS_LIST = False
    RETURN_TYPES = ("IMAGE",)

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",)}, "optional": {}}

    def execute(self, image):
        return (image * 0.5,)


class _FakeV3Image(_FakeV3Base):
    @classmethod
    def GET_NODE_INFO_V1(cls):
        return _fake_v1_info()

    @classmethod
    def execute(cls, image):
        return _FakeNodeOutput(image * 0.5)


class _FakeOutputNode(_FakeV1Image):
    OUTPUT_NODE = True


class _FakeListNode(_FakeV1Image):
    OUTPUT_IS_LIST = [True]


class _FakeVideoBatchNode(_FakeV1Image):
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",)}, "optional": {}}

    @classmethod
    def GET_NODE_INFO_V1(cls):
        info = _fake_v1_info()
        return info


def test_discover_v1_and_v3_image_nodes():
    mappings = {
        "V1Image": _FakeV1Image,
        "V3Image": _FakeV3Image,
    }
    cfg = {"mode": "universal", "enabled": True, "allow": [], "never": []}
    found = ab.discover_universal_candidates(mappings, {}, cfg)
    assert "V1Image" in found
    assert "V3Image" in found
    assert found["V1Image"]["frames"] == ["image"]
    assert found["V3Image"]["frames"] == ["image"]


def test_discover_skips_output_node_list_excluded_never():
    class BatchNameNode(_FakeV1Image):
        pass

    class NeverNode(_FakeV1Image):
        pass

    mappings = {
        "BadOutput": _FakeOutputNode,
        "BadList": _FakeListNode,
        "ImageBatch": _FakeVideoBatchNode,
        "VideoBatchNode": BatchNameNode,
        "NeverNode": NeverNode,
    }
    display = {"VideoBatchNode": "Video Batch Helper"}
    cfg = {
        "mode": "universal",
        "enabled": True,
        "allow": [],
        "never": ["NeverNode"],
    }
    found = ab.discover_universal_candidates(mappings, display, cfg)
    assert "BadOutput" not in found
    assert "BadList" not in found
    assert "ImageBatch" not in found
    assert "VideoBatchNode" not in found
    assert "NeverNode" not in found


def test_probe_passes_per_frame_op(config_dir):
    calls = {"n": 0}

    class ProbeOk:
        FUNCTION = "execute"
        OUTPUT_NODE = False
        INPUT_IS_LIST = False
        RETURN_TYPES = ("IMAGE",)

        @classmethod
        def INPUT_TYPES(cls):
            return {"required": {"image": ("IMAGE",)}, "optional": {}}

        def execute(self, image):
            calls["n"] += 1
            return (image * 0.5,)

    mappings = {"ProbeOk": ProbeOk}
    cfg = {
        "mode": "universal",
        "enabled": True,
        "max_frames": 4,
        "nodes": {},
    }
    ab.install_wrappers(mappings, cfg)
    image = torch.ones(12, 2, 2, 3)
    out = ProbeOk().execute(image)
    assert out[0].shape == image.shape
    assert "ProbeOk" in ab._SAFE
    assert calls["n"] >= 2


def test_probe_fails_temporal_op(caplog):
    caplog.set_level(logging.INFO, logger="C2C.autobatch")
    calls = {"n": 0}

    class MeanShift:
        FUNCTION = "execute"
        OUTPUT_NODE = False
        INPUT_IS_LIST = False
        RETURN_TYPES = ("IMAGE",)

        @classmethod
        def INPUT_TYPES(cls):
            return {"required": {"image": ("IMAGE",)}, "optional": {}}

        def execute(self, image):
            calls["n"] += 1
            mean = image.mean(dim=0, keepdim=True)
            return ((image - mean).clone(),)

    mappings = {"MeanShift": MeanShift}
    cfg = {
        "mode": "universal",
        "enabled": True,
        "max_frames": 4,
        "allow": ["MeanShift"],
        "nodes": {},
    }
    ab.install_wrappers(mappings, cfg)
    image = torch.randn(12, 2, 2, 3)
    out = MeanShift().execute(image)
    direct = (image - image.mean(dim=0, keepdim=True)).clone()
    assert torch.allclose(out[0], direct)
    # First time: calibration chunk + the probe frame alone + the full-batch fallback.
    assert calls["n"] == 3
    assert "MeanShift" in ab._UNSAFE
    calls["n"] = 0
    out2 = MeanShift().execute(image)          # known unsafe now: one plain call
    assert calls["n"] == 1 and torch.allclose(out2[0], direct)
    assert any("probe" in r.message.lower() or "diff" in r.message.lower() for r in caplog.records)


def test_refuse_latent_alongside(caplog):
    caplog.set_level(logging.INFO, logger="C2C.autobatch")
    calls = {"n": 0}

    class WithLatent:
        FUNCTION = "execute"
        OUTPUT_NODE = False
        INPUT_IS_LIST = False
        RETURN_TYPES = ("IMAGE",)

        @classmethod
        def INPUT_TYPES(cls):
            return {
                "required": {"image": ("IMAGE",), "latent": ("LATENT",)},
                "optional": {},
            }

        def execute(self, image, latent):
            calls["n"] += 1
            return (image.clone(),)

    mappings = {"WithLatent": WithLatent}
    cfg = {
        "mode": "universal",
        "enabled": True,
        "max_frames": 4,
        "allow": ["WithLatent"],
        "nodes": {},
    }
    ab.install_wrappers(mappings, cfg)
    image = torch.zeros(8, 2, 2, 3)
    latent = {"samples": torch.zeros(8, 4, 2, 2)}
    WithLatent().execute(image, latent)
    assert calls["n"] == 1
    assert any("LATENT" in r.message or "non-frame" in r.message for r in caplog.records)


def test_anomaly_fallback_universal_vs_strict_explicit():
    class DropFrames:
        FUNCTION = "execute"
        OUTPUT_NODE = False
        INPUT_IS_LIST = False
        RETURN_TYPES = ("IMAGE",)
        calls = 0

        @classmethod
        def INPUT_TYPES(cls):
            return {"required": {"image": ("IMAGE",)}, "optional": {}}

        def execute(self, image):
            DropFrames.calls += 1
            return (image[: max(1, image.shape[0] - 1)].clone(),)

    mappings = {"DropFrames": DropFrames}
    universal_cfg = {
        "mode": "universal",
        "enabled": True,
        "max_frames": 4,
        "allow": ["DropFrames"],          # its name matches the cross-frame word "frame": allow forces the wrap
        "nodes": {},
    }
    ab.install_wrappers(mappings, universal_cfg)
    image = torch.zeros(12, 2, 2, 3)
    out = DropFrames().execute(image)
    assert out[0].shape[0] == 11              # the node's own full-batch result (it drops one frame)
    assert "DropFrames" in ab._UNSAFE
    ab.uninstall_wrappers()
    DropFrames.calls = 0

    strict_cfg = {
        "enabled": True,
        "curated": False,
        "max_frames": 4,
        "nodes": {"DropFrames": {"frames": ["image"]}},
    }
    ab.install_wrappers(mappings, strict_cfg)
    with pytest.raises(ab.AutobatchError):
        DropFrames().execute(image)


def test_validate_config_and_atomic_save(config_dir):
    with pytest.raises(ValueError, match="mode"):
        ab.validate_config({"mode": "bogus"})
    with pytest.raises(ValueError, match="allow"):
        ab.validate_config({"allow": "ImageBlur"})
    saved = ab.save_config({
        "mode": "universal",
        "enabled": True,
        "allow": ["ImageBlur"],
        "never": [],
        "nodes": {},
    })
    assert saved["mode"] == "universal"
    assert saved["allow"] == ["ImageBlur"]
    loaded = json.loads(config_dir.read_text(encoding="utf-8"))
    assert loaded["mode"] == "universal"
    assert loaded["allow"] == ["ImageBlur"]


def test_fast_path_no_extra_calls():
    calls = {"n": 0}

    class FastPath:
        FUNCTION = "execute"
        OUTPUT_NODE = False
        INPUT_IS_LIST = False
        RETURN_TYPES = ("IMAGE",)

        @classmethod
        def INPUT_TYPES(cls):
            return {"required": {"image": ("IMAGE",)}, "optional": {}}

        def execute(self, image):
            calls["n"] += 1
            return (image.clone(),)

    mappings = {"FastPath": FastPath}
    cfg = {
        "mode": "universal",
        "enabled": True,
        "budget_mb": 99999,
        "nodes": {},
    }
    ab.install_wrappers(mappings, cfg)
    FastPath().execute(torch.zeros(4, 2, 2, 3))
    assert calls["n"] == 1


def test_saved_mode_resolves_to_itself(config_dir):
    """Regression (live testbed, L4.15): saving {"mode": "universal"} stored enabled=false and resolved to off."""
    for mode in ("universal", "curated", "off"):
        saved = ab.save_config({"mode": mode, "budget_mb": 64})
        assert ab._resolve_mode(saved) == mode
        assert ab._resolve_mode(ab.load_config()) == mode
    legacy = {"enabled": True, "curated": False, "nodes": {"X": {"frames": ["image"]}}}
    assert ab._resolve_mode(legacy) == "explicit"


# ── Internal mode: C2C's own frame nodes batch by default (owner D0.14, A9 / L2.30) ─────────────────────────

def _frame_node(name, module):
    def run(self, image):
        return (image * 0.5,)
    cls = type(name, (), {
        "INPUT_TYPES": classmethod(lambda c: {"required": {"image": ("IMAGE",)}}),
        "RETURN_TYPES": ("IMAGE",), "FUNCTION": "run", "run": run, "CATEGORY": "test",
    })
    cls.RELATIVE_PYTHON_MODULE = module
    return cls


def test_internal_mode_wraps_c2c_nodes_and_the_core_list_only():
    mappings = {
        "MyC2CGrade": _frame_node("MyC2CGrade", "custom_nodes.ComfyUI-NukeMaxNodes"),
        "SomeoneElsesGrade": _frame_node("SomeoneElsesGrade", "custom_nodes.ComfyUI-OtherPack"),
        "ImageBlur": _frame_node("ImageBlur", "nodes"),
    }
    specs = ab._effective_wrap_specs({"mode": "internal"}, mappings)
    assert "MyC2CGrade" in specs and "ImageBlur" in specs        # C2C node + the measured core list
    assert "SomeoneElsesGrade" not in specs                      # another author's node needs Universal
    universal = ab._effective_wrap_specs({"mode": "universal"}, mappings)
    assert "SomeoneElsesGrade" in universal


def test_a_fresh_install_defaults_to_internal(config_dir):
    assert not config_dir.exists()
    assert ab._resolve_mode(ab.load_config()) == "internal"


def test_an_old_file_without_a_mode_keeps_its_own_switch(config_dir):
    """A file written before modes existed and switched off must stay off, not inherit the new default."""
    _write_config(config_dir, enabled=False)
    assert ab._resolve_mode(ab.load_config()) == "off"
    _write_config(config_dir, enabled=True, curated=True)
    assert ab._resolve_mode(ab.load_config()) == "curated"


def test_internal_round_trips_through_the_config_file(config_dir):
    saved = ab.save_config({"mode": "internal", "budget_mb": 64})
    assert ab._resolve_mode(saved) == "internal"
    assert ab._resolve_mode(ab.load_config()) == "internal"
