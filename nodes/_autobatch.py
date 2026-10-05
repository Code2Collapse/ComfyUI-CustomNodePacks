"""Automatic frame batching for ComfyUI nodes (owner amendment A5).

Provides ``run_chunked`` for memory-bounded per-frame execution, an
``@autobatch`` decorator for our own frame-independent nodes, and an
opt-in wrapper installed on the first queued prompt for listed core nodes.

Kill switch: ``C2C_AUTOBATCH=0`` disables chunking (single call).

Config file (``user/default/c2c/autobatch.json``, overridable via
``C2C_AUTOBATCH_CONFIG``) shape::

    {
      "mode": "off",
      "enabled": false,
      "curated": false,
      "max_frames": 0,
      "budget_mb": null,
      "allow": [],
      "never": [],
      "nodes": {
        "ImageScaleBy": {
          "frames": ["image"],
          "work_factor": 4.0,
          "max_frames": 0
        }
      }
    }

``mode``: ``off`` (default), ``curated``, or ``universal``. Legacy
``{"enabled": true, "curated": false, "nodes": {...}}`` without ``mode``
remains explicit-only (strict path for listed nodes).

``max_frames`` (global and per-node, default ``0`` = off): chunk when batch
size exceeds this limit. Per-node value wins; otherwise the global value is
used.

``budget_mb`` (global, optional): caps chunk size by memory; converted to
bytes as ``budget_mb * 2**20`` and passed to ``run_chunked`` as
``budget_bytes``. When unset, free RAM/VRAM heuristics apply.

Tests run without ComfyUI on ``sys.path``; every ``comfy`` import is lazy
with fallbacks (no progress bar, no interrupt check, RAM from psutil, VRAM
unknown).
"""
from __future__ import annotations

import asyncio
import collections
import functools
import inspect
import json
import logging
import os
import weakref
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Sequence

import torch

log = logging.getLogger("C2C.autobatch")

RAM_FRACTION = 0.35
VRAM_FRACTION = 0.6
CALIBRATION_CAP = 8
DEFAULT_WORK_FACTOR = 4.0

_ORIG_ATTR = "_c2c_autobatch_orig"
_WRAPPED: dict[str, tuple[type, Callable[..., Any], str]] = {}
_DEFINED_HERE: dict[str, bool] = {}          # was the wrapped attribute defined on the class itself
_PROMPT_HOOK_REGISTERED = False
_CONFIG_MTIME: float | None = None
_CONFIG_CACHE: dict[str, Any] | None = None
_DICT_RETURN_LABELS: set[str] = set()
_MACHINERY_FAILED: set[str] = set()
# Keyed by the function object itself, weakly: an id() key is reused once a function is collected, and a
# new function then got a dead function's signature (binding failed and calls silently ran unchunked).
_SIG_CACHE: "weakref.WeakKeyDictionary[Callable[..., Any], inspect.Signature]" = weakref.WeakKeyDictionary()
_ROUTES_REGISTERED = False
_WRAP_STRICT: dict[str, bool] = {}          # class_id -> True when cfg["nodes"] owns the spec
_SAFE: dict[str, str] = {}
_UNSAFE: dict[str, str] = {}
_REFUSAL_LOGGED: set[str] = set()
_MEASURED_BPF: dict[str, int] = {}
_CALL_HISTORY: collections.deque[dict[str, Any]] = collections.deque(maxlen=20)
_PROBE_TOLERANCE = 2e-3

# Cross-frame by name — chunking would produce visible seams or wrong semantics.
_NAME_EXCLUSION_WORDS: tuple[str, ...] = (
    "video", "temporal", "interpolat", "flow", "stabil", "deflicker",
    "propagat", "track", "batch", "list", "select", "repeat", "reverse",
    "merge", "combine", "concat", "split", "loop", "sequence", "frame",
    "rife", "film", "onion",
)
# Known ids — loaders/combiners/temporal editors manage the frame axis themselves.
_KNOWN_EXCLUDED_IDS: frozenset[str] = frozenset({
    # VHS — clip I/O and batch utilities, not per-frame filters
    "VHS_VideoCombine", "VHS_LoadVideo", "VHS_SelectImages", "VHS_SplitImages",
    "VHS_MergeImages", "VHS_DuplicateImages",
    # Core — explicit batch axis manipulation
    "ImageBatch", "RepeatImageBatch", "ImageFromBatch", "RebatchImages", "ImageStitch",
    # C2C temporal nodes
    "VideoStabilizerMEC", "VideoMaskEditorMEC", "MaskTrackerMEC",
})
_FRAME_TYPES: frozenset[str] = frozenset({"IMAGE", "MASK"})

# Measured peak RAM above baseline (48 frames @ 720p, CPU, peak_ram_core_nodes.json):
# ImageBlur 1147 -> 715 MB (chunking helps); ImageSharpen 657 -> 709 MB and
# ImageScaleBy 136 -> 151 MB (chunking does not help). ImageScale, ImageScaleBy,
# ImageScaleToTotalPixels, ImageInvert and ImageSharpen removed for that reason.
# ImageUpscaleWithModel is the VRAM case (moves the whole batch to the GPU); not
# measurable on a CPU testbed, so it is listed as experimental.
CURATED: dict[str, dict[str, Any]] = {
    "ImageBlur": {"frames": ["image"], "work_factor": 2.5},
    "ImageUpscaleWithModel": {"frames": ["image"], "work_factor": 6.0},
}


class AutobatchError(RuntimeError):
    """User-visible autobatch abort (not a wrapper machinery bug)."""


class AutobatchMachineryError(Exception):
    """Internal autobatch bug — wrapper unwraps the affected class."""


class _NodeRaised(Exception):
    """Private carrier: node raised ``original`` (never shown to callers)."""

    def __init__(self, original: BaseException):
        self.original = original
        super().__init__(original)


def _mm():
    try:
        import comfy.model_management as mm  # noqa: WPS433 — lazy for tests

        return mm
    except Exception:
        return None


def _progress_bar(total: int):
    try:
        from comfy.utils import ProgressBar  # noqa: WPS433

        return ProgressBar(int(total))
    except Exception:
        return _NoProgress()


class _NoProgress:
    def update(self, _n: int) -> None:
        return None


def _throw_interrupt() -> None:
    mm = _mm()
    if mm is None:
        return
    try:
        mm.throw_exception_if_processing_interrupted()
    except Exception:
        return


def _oom_types() -> tuple[type[BaseException], ...]:
    mm = _mm()
    if mm is not None:
        oom = getattr(mm, "OOM_EXCEPTION", None)
        if isinstance(oom, type):
            return (oom, MemoryError)
    return (MemoryError,)


def _is_oom(exc: BaseException) -> bool:
    if isinstance(exc, _oom_types()):
        return True
    if isinstance(exc, RuntimeError):
        return "out of memory" in str(exc).lower()
    return False


def _soft_empty_cache() -> None:
    mm = _mm()
    if mm is None:
        return
    try:
        mm.soft_empty_cache(force=False)
    except Exception:
        return


def _get_torch_device():
    mm = _mm()
    if mm is None:
        return torch.device("cpu")
    try:
        return mm.get_torch_device()
    except Exception:
        return torch.device("cpu")


def _format_bytes(n: int) -> str:
    n = max(0, int(n))
    if n >= 1024 ** 3:
        return f"{n / (1024 ** 3):.1f} GB"
    if n >= 1024 ** 2:
        return f"{n / (1024 ** 2):.1f} MB"
    if n >= 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n} B"


def _autobatch_enabled() -> bool:
    return os.environ.get("C2C_AUTOBATCH", "").strip() != "0"


def _is_ui_dict(out: Any) -> bool:
    return isinstance(out, dict) and "ui" in out and "result" in out


def _is_node_output(out: Any) -> bool:
    return (
        hasattr(out, "args")
        and isinstance(getattr(out, "args", None), tuple)
        and hasattr(out, "ui")
        and hasattr(out, "expand")
        and hasattr(out, "block_execution")
    )


def _node_output_non_chunkable(out: Any) -> bool:
    if not _is_node_output(out):
        return False
    if getattr(out, "ui", None) is not None:
        return True
    if getattr(out, "expand", None) is not None:
        return True
    if getattr(out, "block_execution", None) is not None:
        return True
    return False


def _decode_v3_chunk(out: Any) -> tuple[tuple[Any, ...] | None, bool]:
    """Return ``(args_tuple, needs_full_batch)`` for one V3 chunk result."""
    if _is_node_output(out):
        if _node_output_non_chunkable(out):
            return None, True
        return out.args, False
    if isinstance(out, tuple):
        return out, False
    return None, True


def _encode_v3_result(
    cal_out_raw: Any,
    buffers: Sequence[torch.Tensor],
    non_tensors: Sequence[Any],
) -> Any:
    args = tuple(buffers) + tuple(non_tensors)
    if _is_node_output(cal_out_raw):
        node_cls = type(cal_out_raw)
        return node_cls(*args)
    return args


def _folder_paths():
    try:
        import folder_paths  # noqa: WPS433

        return folder_paths
    except Exception:
        return None


def _config_path() -> Path | None:
    override = os.environ.get("C2C_AUTOBATCH_CONFIG", "").strip()
    if override:
        return Path(override)
    fp = _folder_paths()
    if fp is None:
        return None
    try:
        user_dir = fp.get_user_directory()
    except Exception:
        return None
    return Path(user_dir) / "default" / "c2c" / "autobatch.json"


def _default_config() -> dict[str, Any]:
    return {
        "mode": "off",
        "enabled": False,
        "curated": False,
        "max_frames": 0,
        "budget_mb": None,
        "allow": [],
        "never": [],
        "nodes": {},
    }


def _normalize_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if isinstance(item, str) and item]


def _resolve_mode(cfg: Mapping[str, Any]) -> Literal["off", "explicit", "curated", "universal"]:
    # An explicit curated/universal mode wins. "off" or no mode leaves the decision to the legacy switch, so an
    # old {"enabled": true, "nodes": {...}} file keeps working as explicit (or curated).
    mode = cfg.get("mode")
    lowered = mode.strip().lower() if isinstance(mode, str) else None
    if lowered in ("curated", "universal"):
        return lowered
    if not cfg.get("enabled"):
        return "off"
    return "curated" if cfg.get("curated") else "explicit"


def validate_config(patch: Mapping[str, Any]) -> None:
    if "mode" in patch:
        mode = patch.get("mode")
        if mode not in ("off", "curated", "universal"):
            raise ValueError('autobatch "mode" must be off, curated, or universal')
    for key in ("allow", "never"):
        if key in patch and not isinstance(patch.get(key), list):
            raise ValueError(f'autobatch "{key}" must be a list of class id strings')
        if key in patch:
            for item in patch.get(key) or []:
                if not isinstance(item, str):
                    raise ValueError(f'autobatch "{key}" must contain only strings')


def merge_config(current: Mapping[str, Any], patch: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(_default_config())
    out.update(current)
    for key, value in patch.items():
        if key == "nodes" and isinstance(value, dict):
            nodes = dict(out.get("nodes") or {})
            nodes.update(value)
            out["nodes"] = nodes
        else:
            out[key] = value
    return out


def save_config(cfg: Mapping[str, Any]) -> dict[str, Any]:
    validate_config(cfg)
    path = _config_path()
    if path is None:
        raise ValueError("autobatch config path is not available")
    normalized = merge_config(_default_config(), cfg)
    mode_field = cfg.get("mode")
    if isinstance(mode_field, str) and mode_field.strip().lower() in ("off", "curated", "universal"):
        m = mode_field.strip().lower()
        normalized["mode"] = m
        # Choosing a mode sets the legacy switches too, so "off" really turns a legacy file off.
        normalized["enabled"] = m != "off"
        normalized["curated"] = m == "curated"
    resolved = _resolve_mode(normalized)
    if resolved == "explicit":
        normalized["enabled"] = True
        normalized["curated"] = False
    else:
        normalized["enabled"] = resolved != "off"
        normalized["curated"] = resolved == "curated"
        if isinstance(mode_field, str) and mode_field.strip().lower() in ("off", "curated", "universal"):
            normalized["mode"] = mode_field.strip().lower()
    normalized["allow"] = _normalize_string_list(normalized.get("allow"))
    normalized["never"] = _normalize_string_list(normalized.get("never"))
    nodes = normalized.get("nodes")
    normalized["nodes"] = dict(nodes) if isinstance(nodes, dict) else {}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(normalized, indent=2), encoding="utf-8")
    os.replace(tmp, path)
    return normalized


def _budget_bytes_from_config(cfg: Mapping[str, Any]) -> int | None:
    budget_mb = cfg.get("budget_mb")
    if budget_mb is None:
        return None
    try:
        return int(float(budget_mb) * (2 ** 20))
    except (TypeError, ValueError):
        return None


def _max_frames_from_config(
    cfg: Mapping[str, Any],
    spec: Mapping[str, Any] | None = None,
) -> int | None:
    if spec is not None and "max_frames" in spec:
        value = int(spec.get("max_frames") or 0)
    else:
        value = int(cfg.get("max_frames") or 0)
    return value if value > 0 else None


def load_config() -> dict[str, Any]:
    path = _config_path()
    if path is None or not path.is_file():
        return _default_config()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        log.warning("[C2C autobatch] could not read %s: %s", path, exc)
        return _default_config()
    if not isinstance(data, dict):
        return _default_config()
    out = _default_config()
    out["enabled"] = bool(data.get("enabled", False))
    out["curated"] = bool(data.get("curated", False))
    out["max_frames"] = int(data.get("max_frames", 0) or 0)
    budget_mb = data.get("budget_mb")
    if budget_mb is None:
        out["budget_mb"] = None
    else:
        try:
            out["budget_mb"] = float(budget_mb)
        except (TypeError, ValueError):
            out["budget_mb"] = None
    nodes = data.get("nodes", {})
    out["nodes"] = dict(nodes) if isinstance(nodes, dict) else {}
    out["allow"] = _normalize_string_list(data.get("allow"))
    out["never"] = _normalize_string_list(data.get("never"))
    mode = data.get("mode")
    if isinstance(mode, str) and mode.strip().lower() in ("off", "curated", "universal"):
        out["mode"] = mode.strip().lower()
    else:
        resolved = _resolve_mode(out)
        out["mode"] = {
            "off": "off",
            "explicit": "off",
            "curated": "curated",
            "universal": "universal",
        }[resolved]
    return out


def _config_mtime() -> float | None:
    path = _config_path()
    if path is None or not path.is_file():
        return None
    try:
        return path.stat().st_mtime
    except OSError:
        return None


def _is_frame_type(type_value: Any) -> bool:
    return isinstance(type_value, str) and type_value in _FRAME_TYPES


def _get_node_info(cls: type) -> dict[str, Any] | None:
    if _is_v3_node(cls):
        try:
            info = cls.GET_NODE_INFO_V1()
        except Exception:
            return None
        return info if isinstance(info, dict) else None
    input_types_fn = getattr(cls, "INPUT_TYPES", None)
    if not callable(input_types_fn):
        return None
    try:
        inputs = input_types_fn()
    except Exception:
        return None
    if not isinstance(inputs, dict):
        return None
    required = inputs.get("required", {})
    optional = inputs.get("optional", {})
    merged_inputs: dict[str, Any] = {}
    if isinstance(required, dict):
        merged_inputs.update(required)
    if isinstance(optional, dict):
        merged_inputs.update(optional)
    output_is_list = getattr(cls, "OUTPUT_IS_LIST", False)
    if isinstance(output_is_list, (list, tuple)):
        output_is_list = any(output_is_list)
    return {
        "input": {"required": merged_inputs, "optional": {}},
        "output": list(getattr(cls, "RETURN_TYPES", ()) or ()),
        "output_node": bool(getattr(cls, "OUTPUT_NODE", False)),
        "output_is_list": output_is_list,
        "is_input_list": bool(getattr(cls, "INPUT_IS_LIST", False)),
    }


def _input_frame_names(info: Mapping[str, Any]) -> list[str]:
    names: list[str] = []
    inputs = info.get("input", {})
    if not isinstance(inputs, dict):
        return names
    for section in ("required", "optional"):
        section_inputs = inputs.get(section, {})
        if not isinstance(section_inputs, dict):
            continue
        for name, spec in section_inputs.items():
            if not isinstance(spec, (list, tuple)) or not spec:
                continue
            if _is_frame_type(spec[0]):
                names.append(str(name))
    return names


def _has_frame_tensor_output(info: Mapping[str, Any]) -> bool:
    outputs = info.get("output", ())
    if not isinstance(outputs, (list, tuple)):
        return False
    return any(_is_frame_type(item) for item in outputs)


def _output_is_list(info: Mapping[str, Any]) -> bool:
    value = info.get("output_is_list", False)
    if isinstance(value, (list, tuple)):
        return any(value)
    return bool(value)


def _is_input_list(info: Mapping[str, Any]) -> bool:
    return bool(info.get("is_input_list", False))


def _display_name(
    class_id: str,
    display_mappings: Mapping[str, str] | None,
) -> str:
    if display_mappings and class_id in display_mappings:
        return str(display_mappings[class_id])
    return class_id


def _name_excluded(class_id: str, display_name: str) -> bool:
    hay = f"{class_id} {display_name}".lower()
    return any(word in hay for word in _NAME_EXCLUSION_WORDS)


def _spec_from_class(cls: type, class_id: str) -> dict[str, Any] | None:
    info = _get_node_info(cls)
    if info is None:
        return None
    frames = _input_frame_names(info)
    if not frames:
        return None
    return {"frames": frames, "work_factor": DEFAULT_WORK_FACTOR}


def _discovery_refused(
    cls: type,
    class_id: str,
    display_mappings: Mapping[str, str] | None,
    cfg: Mapping[str, Any],
) -> str | None:
    never = set(_normalize_string_list(cfg.get("never")))
    if class_id in never:
        return "never list"
    if getattr(cls, "TEMPORAL", False):
        return "TEMPORAL"
    if class_id in _KNOWN_EXCLUDED_IDS:
        return "known excluded id"
    allow = set(_normalize_string_list(cfg.get("allow")))
    if class_id not in allow and _name_excluded(class_id, _display_name(class_id, display_mappings)):
        return "name exclusion"
    info = _get_node_info(cls)
    if info is None:
        return "no node info"
    if info.get("output_node"):
        return "OUTPUT_NODE"
    if _is_input_list(info):
        return "INPUT_IS_LIST"
    if _output_is_list(info):
        return "OUTPUT_IS_LIST"
    if not _input_frame_names(info):
        return "no IMAGE/MASK input"
    if not _has_frame_tensor_output(info):
        return "no IMAGE/MASK output"
    if _is_v3_node(cls):
        raw = inspect.getattr_static(cls, "execute", None)
        if raw is None or not isinstance(raw, classmethod):
            return "execute missing"
        if inspect.iscoroutinefunction(raw.__func__):
            return "async execute"
    else:
        fn_name = getattr(cls, "FUNCTION", None)
        if not fn_name or not isinstance(fn_name, str):
            return "FUNCTION missing"
        orig = getattr(cls, fn_name, None)
        if orig is None or not callable(orig):
            return "FUNCTION missing"
    return None


def discover_universal_candidates(
    node_class_mappings: Mapping[str, type],
    display_mappings: Mapping[str, str] | None,
    cfg: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for class_id, cls in node_class_mappings.items():
        if _discovery_refused(cls, class_id, display_mappings, cfg) is not None:
            continue
        spec = _spec_from_class(cls, class_id)
        if spec is not None:
            out[class_id] = dict(spec)
    return out


def _effective_wrap_specs(
    cfg: Mapping[str, Any],
    node_class_mappings: Mapping[str, type],
    display_mappings: Mapping[str, str] | None = None,
) -> dict[str, dict[str, Any]]:
    mode = _resolve_mode(cfg)
    if mode == "off":
        return {}
    specs: dict[str, dict[str, Any]] = {}
    never = set(_normalize_string_list(cfg.get("never")))
    nodes_cfg = cfg.get("nodes", {})
    nodes_dict = nodes_cfg if isinstance(nodes_cfg, dict) else {}
    allow = _normalize_string_list(cfg.get("allow"))

    if mode == "curated":
        for name, spec in CURATED.items():
            if name not in never:
                specs[name] = dict(spec)

    if mode == "universal":
        discovered = discover_universal_candidates(
            node_class_mappings, display_mappings, cfg,
        )
        for name, spec in discovered.items():
            if name not in never:
                specs[name] = dict(spec)

    for name, spec in nodes_dict.items():
        if name in never or not isinstance(spec, dict):
            continue
        merged = dict(specs.get(name, {}))
        merged.update(spec)
        specs[str(name)] = merged

    for class_id in allow:
        if class_id in never or class_id in specs:
            continue
        cls = node_class_mappings.get(class_id)
        if cls is None:
            continue
        reason = _discovery_refused(cls, class_id, display_mappings, cfg)
        if reason in (None, "name exclusion", "known excluded id"):
            spec = _spec_from_class(cls, class_id)
            if spec is not None:
                specs[class_id] = dict(spec)

    return specs


def _effective_node_specs(cfg: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Backward-compatible helper for tests that do not pass mappings."""
    specs: dict[str, dict[str, Any]] = {}
    if cfg.get("curated"):
        for name, spec in CURATED.items():
            specs[name] = dict(spec)
    nodes = cfg.get("nodes", {})
    if isinstance(nodes, dict):
        for name, spec in nodes.items():
            if isinstance(spec, dict):
                merged = dict(specs.get(name, {}))
                merged.update(spec)
                specs[str(name)] = merged
    return specs


def _batch_size(frames: Mapping[str, torch.Tensor]) -> int:
    if not frames:
        raise AutobatchError("Automatic batching requires at least one frame input tensor.")
    b = 0
    for tensor in frames.values():
        if not isinstance(tensor, torch.Tensor) or tensor.ndim < 1:
            raise AutobatchError("Frame inputs must be batched torch tensors.")
        dim0 = int(tensor.shape[0])
        if dim0 > 1:
            b = max(b, dim0)
    if b <= 0:
        b = 1
    for name, tensor in frames.items():
        dim0 = int(tensor.shape[0])
        if dim0 not in (1, b):
            raise AutobatchError(
                f"Automatic batching aborted: frame input '{name}' batch size "
                f"{dim0} does not match batch {b}."
            )
    return b


def _slice_frames(
    frames: Mapping[str, torch.Tensor],
    start: int,
    end: int,
) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for name, tensor in frames.items():
        if int(tensor.shape[0]) == 1:
            out[name] = tensor
        else:
            out[name] = tensor[start:end]
    return out


def _resolve_device(
    device: torch.device | str | None,
    frames: Mapping[str, torch.Tensor],
) -> torch.device:
    if device is not None:
        return torch.device(device)
    for tensor in frames.values():
        if isinstance(tensor, torch.Tensor) and tensor.is_cuda:
            return tensor.device
    return _get_torch_device()


def _available_ram_bytes() -> int | None:
    try:
        import psutil  # noqa: WPS433

        return int(psutil.virtual_memory().available)
    except Exception:
        return None


def _available_vram_bytes(device: torch.device) -> int | None:
    if device.type != "cuda":
        return None
    mm = _mm()
    if mm is None:
        return None
    try:
        free_mem, _torch_free = mm.get_free_memory(device, torch_free_too=True)
        return int(free_mem + _torch_free)
    except TypeError:
        try:
            return int(mm.get_free_memory(device))
        except Exception:
            return None
    except Exception:
        return None


def _read_budget(
    device: torch.device,
    *,
    budget_bytes: int | None = None,
) -> int | None:
    if budget_bytes is not None:
        return max(0, int(budget_bytes))
    parts: list[int] = []
    ram = _available_ram_bytes()
    if ram is not None:
        parts.append(int(ram * RAM_FRACTION))
    vram = _available_vram_bytes(device)
    if vram is not None:
        parts.append(int(vram * VRAM_FRACTION))
    if not parts:
        return None
    return min(parts)


def _input_bytes_per_frame(
    frames: Mapping[str, torch.Tensor],
    work_factor: float,
) -> int:
    total = 0
    for tensor in frames.values():
        one = tensor[0:1]
        total += one.numel() * one.element_size()
    return max(1, int(total * float(work_factor)))


def _would_chunk(
    batch_size: int,
    input_bpf: int,
    budget: int | None,
    max_frames: int | None,
) -> bool:
    if max_frames is not None and int(max_frames) > 0 and batch_size > int(max_frames):
        return True
    if budget is None:
        return False
    return batch_size * input_bpf > budget


def _mark_unsafe(class_name: str, reason: str) -> None:
    _SAFE.pop(class_name, None)
    _UNSAFE[class_name] = reason


def _mark_safe(class_name: str) -> None:
    _UNSAFE.pop(class_name, None)
    _SAFE[class_name] = "probe passed"


def _has_batched_non_frame_input(
    bound: Mapping[str, Any],
    frame_names: Sequence[str],
    batch_size: int,
) -> bool:
    frame_set = set(frame_names)
    for name, value in bound.items():
        if name in frame_set:
            continue
        if isinstance(value, torch.Tensor):
            if value.ndim >= 1 and int(value.shape[0]) == batch_size and batch_size > 1:
                return True
        elif isinstance(value, dict):
            samples = value.get("samples")
            if isinstance(samples, torch.Tensor):
                if samples.ndim >= 1 and int(samples.shape[0]) == batch_size and batch_size > 1:
                    return True
    return False


def _log_refusal_once(class_name: str, reason: str) -> None:
    if class_name in _REFUSAL_LOGGED:
        return
    _REFUSAL_LOGGED.add(class_name)
    log.info("[C2C autobatch] %s: %s — not chunked", class_name, reason)


def _tensor_outputs_from_result(out: Any, *, v3: bool) -> tuple[list[torch.Tensor], bool]:
    if _is_ui_dict(out):
        return [], True
    if v3:
        decoded, needs_full = _decode_v3_chunk(out)
        if needs_full or decoded is None:
            return [], True
        out_tuple = decoded
    else:
        if not isinstance(out, tuple):
            return [], True
        out_tuple = out
    tensors = [item for item in out_tuple if isinstance(item, torch.Tensor)]
    return tensors, False


def _probe_outputs_match(
    cal_tensors: Sequence[torch.Tensor],
    single_tensors: Sequence[torch.Tensor],
    cal_size: int,
) -> tuple[bool, float]:
    if len(cal_tensors) != len(single_tensors):
        return False, float("inf")
    max_diff = 0.0
    for cal_t, single_t in zip(cal_tensors, single_tensors):
        if not isinstance(cal_t, torch.Tensor) or not isinstance(single_t, torch.Tensor):
            return False, float("inf")
        if int(cal_t.shape[0]) < cal_size or int(single_t.shape[0]) < 1:
            return False, float("inf")
        ref = cal_t[cal_size - 1].detach().float().cpu()
        cand = single_t[0].detach().float().cpu()
        diff = (ref - cand).abs().max().item()
        scale = max(1.0, ref.abs().max().item())
        max_diff = max(max_diff, float(diff))
        if diff > _PROBE_TOLERANCE * scale:
            return False, max_diff
    return True, max_diff


def _measure_cuda_bpf(
    device: torch.device,
    class_name: str,
    cal_size: int,
    start_bytes: int,
) -> None:
    if device.type != "cuda" or cal_size <= 0:
        return
    try:
        peak = int(torch.cuda.max_memory_allocated(device))
        delta = max(0, peak - start_bytes)
        if delta > 0:
            _MEASURED_BPF[class_name] = max(1, delta // cal_size)
    except Exception:
        return


def _effective_work_factor(
    class_name: str,
    frames: Mapping[str, torch.Tensor],
    spec: Mapping[str, Any],
) -> float:
    measured = _MEASURED_BPF.get(class_name)
    if measured is not None:
        raw = _input_bytes_per_frame(frames, 1.0)
        if raw > 0:
            return max(1.0, float(measured) / float(raw))
    return float(spec.get("work_factor", DEFAULT_WORK_FACTOR))


def _record_chunked_call(
    label: str,
    batch_size: int,
    chunk_count: int,
    chunk_size: int,
    budget: int | None,
) -> None:
    _CALL_HISTORY.append({
        "label": label,
        "frames": int(batch_size),
        "chunks": int(chunk_count),
        "chunk_size": int(chunk_size),
        "budget": int(budget) if budget is not None else None,
    })


def _run_frame_probe(
    fn: Callable[..., Any],
    frames: Mapping[str, torch.Tensor],
    static: Mapping[str, Any],
    cal_size: int,
    *,
    class_name: str,
    v3: bool,
    device: torch.device,
) -> tuple[bool, Any | None, int, float, str]:
    c = max(2, min(8, int(cal_size), _batch_size(frames)))
    start_bytes = 0
    if device.type == "cuda":
        try:
            torch.cuda.reset_peak_memory_stats(device)
            start_bytes = int(torch.cuda.memory_allocated(device))
        except Exception:
            start_bytes = 0
    try:
        cal_out_raw, used_cal = _run_chunk_with_oom_retry(
            fn, frames, static, 0, c, label=class_name,
        )
    except (AutobatchError, _NodeRaised):
        return False, None, 0, float("inf"), "calibration failed"
    _measure_cuda_bpf(device, class_name, used_cal, start_bytes)
    cal_tensors, bad = _tensor_outputs_from_result(cal_out_raw, v3=v3)
    if bad:
        return False, None, used_cal, float("inf"), "non-chunkable calibration result"
    try:
        if v3:
            cal_out, needs_full = _decode_v3_chunk(cal_out_raw)
            if needs_full or cal_out is None:
                return False, None, used_cal, float("inf"), "non-chunkable calibration result"
            cal_valid, _ = _validate_chunk_outputs(
                cal_out, used_cal, reference_non_tensors=None, label=class_name,
            )
        else:
            cal_valid, _ = _validate_chunk_outputs(
                _normalize_node_result(cal_out_raw),
                used_cal,
                reference_non_tensors=None,
                label=class_name,
            )
        cal_tensors = [t for t in cal_valid if isinstance(t, torch.Tensor)]
    except AutobatchError:
        return False, None, used_cal, float("inf"), "calibration validation failed"
    try:
        single_out_raw, _ = _run_chunk_with_oom_retry(
            fn, frames, static, c - 1, 1, label=class_name,
        )
    except (AutobatchError, _NodeRaised):
        return False, None, used_cal, float("inf"), "single-frame probe failed"
    single_tensors, bad = _tensor_outputs_from_result(single_out_raw, v3=v3)
    if bad:
        return False, None, used_cal, float("inf"), "non-chunkable single-frame result"
    single_tensors = [
        t for t in single_tensors
        if isinstance(t, torch.Tensor) and int(t.shape[0]) >= 1
    ]
    passed, max_diff = _probe_outputs_match(cal_tensors, single_tensors, used_cal)
    if not passed:
        return False, None, used_cal, max_diff, f"probe max diff {max_diff:.6g}"
    return True, cal_out_raw, used_cal, max_diff, ""


def status() -> dict[str, Any]:
    return {
        "mode": _resolve_mode(_CONFIG_CACHE or load_config()),
        "kill_switch": not _autobatch_enabled(),
        "wrapped": sorted(_WRAPPED.keys()),
        "strict": sorted(name for name, strict in _WRAP_STRICT.items() if strict),
        "universal": sorted(name for name, strict in _WRAP_STRICT.items() if not strict),
        "safe": dict(_SAFE),
        "unsafe": dict(_UNSAFE),
        "history": list(_CALL_HISTORY),
    }


def _tensor_frame_bytes(tensor: torch.Tensor) -> int:
    if tensor.ndim < 1 or int(tensor.shape[0]) < 1:
        return max(tensor.element_size(), 1)
    one = tensor[0:1]
    return one.numel() * one.element_size()


def _normalize_node_result(out: Any) -> tuple[Any, ...]:
    if not isinstance(out, tuple):
        raise AutobatchError(
            "Automatic batching requires node methods to return a tuple of outputs."
        )
    return out


def _validate_chunk_outputs(
    out: tuple[Any, ...],
    chunk_len: int,
    *,
    reference_non_tensors: list[Any] | None,
    label: str,
) -> tuple[list[torch.Tensor], list[Any]]:
    tensor_outputs: list[torch.Tensor] = []
    non_tensors: list[Any] = []
    for item in out:
        if isinstance(item, torch.Tensor):
            if int(item.shape[0]) != chunk_len:
                raise AutobatchError(
                    f"Automatic batching aborted: output frame count "
                    f"({int(item.shape[0])}) does not match input chunk "
                    f"({chunk_len}). This node is not frame-independent."
                )
            tensor_outputs.append(item)
        else:
            non_tensors.append(item)
    if reference_non_tensors is not None:
        if len(non_tensors) != len(reference_non_tensors):
            raise AutobatchError(
                f"Automatic batching aborted: non-tensor output count changed "
                f"for {label}. This node is not frame-independent."
            )
        for idx, (ref, val) in enumerate(zip(reference_non_tensors, non_tensors)):
            if val != ref:
                raise AutobatchError(
                    f"Automatic batching aborted: non-tensor output #{idx} "
                    f"changed between chunks. This node is not frame-independent."
                )
    return tensor_outputs, non_tensors


def _copy_chunk_to_buffers(
    tensor_outputs: Sequence[torch.Tensor],
    buffers: Sequence[torch.Tensor],
    start: int,
    end: int,
) -> None:
    for buf, chunk_out in zip(buffers, tensor_outputs):
        buf[start:end].copy_(chunk_out.detach().cpu())


def _allocate_output_buffers(
    template_outputs: Sequence[torch.Tensor],
    batch_size: int,
) -> list[torch.Tensor]:
    buffers: list[torch.Tensor] = []
    for template in template_outputs:
        shape = (batch_size, *template.shape[1:])
        buffers.append(torch.empty(shape, dtype=template.dtype, device="cpu"))
    return buffers


def _call_node_fn(fn: Callable[..., Any], /, **kwargs: Any) -> Any:
    """Invoke a node function; node exceptions are wrapped in ``_NodeRaised``."""
    try:
        return fn(**kwargs)
    except BaseException as exc:
        raise _NodeRaised(exc) from None


def _run_chunk_with_oom_retry(
    fn: Callable[..., Any],
    frames: Mapping[str, torch.Tensor],
    static: Mapping[str, Any],
    start: int,
    chunk_len: int,
    *,
    label: str,
) -> tuple[Any, int]:
    current = max(1, int(chunk_len))
    while True:
        try:
            end = start + current
            frames_slice = _slice_frames(frames, start, end)
            out = _call_node_fn(fn, **frames_slice, **static)
            return out, current
        except _NodeRaised as carrier:
            exc = carrier.original
            if isinstance(exc, AutobatchError):
                raise exc
            if not _is_oom(exc) or current <= 1:
                if _is_oom(exc) and current <= 1:
                    raise AutobatchError(
                        "Automatic batching ran out of memory even on a single "
                        "frame. Lower resolution or frame count."
                    ) from exc
                raise carrier
            log.info(
                "[C2C autobatch] %s: out of memory at chunk %d, halving",
                label,
                current,
            )
            _soft_empty_cache()
            current = max(1, current // 2)
        except AutobatchError:
            raise


def _run_chunked_impl(
    fn: Callable[..., Any],
    frames: Mapping[str, torch.Tensor],
    static: Mapping[str, Any],
    *,
    label: str,
    device: torch.device | str | None,
    work_factor: float,
    max_frames: int | None,
    budget_bytes: int | None,
    strict: bool = True,
    cal_bootstrap: tuple[Any, int] | None = None,
) -> Any:
    """Inner chunking body (exceptions converted by ``run_chunked``)."""
    if not _autobatch_enabled():
        return _call_node_fn(fn, **dict(frames), **dict(static))

    batch_size = _batch_size(frames)
    if batch_size <= 1:
        return _call_node_fn(fn, **dict(frames), **dict(static))

    if label in _DICT_RETURN_LABELS:
        return _call_node_fn(fn, **dict(frames), **dict(static))

    resolved_device = _resolve_device(device, frames)
    input_bpf = _input_bytes_per_frame(frames, work_factor)
    budget = _read_budget(resolved_device, budget_bytes=budget_bytes)
    if not _would_chunk(batch_size, input_bpf, budget, max_frames):
        return _call_node_fn(fn, **dict(frames), **dict(static))

    if budget is None:
        if max_frames is not None and int(max_frames) > 0 and batch_size > int(max_frames):
            budget = (input_bpf * batch_size) + 1
        else:
            return _call_node_fn(fn, **dict(frames), **dict(static))

    provisional_chunk = max(1, min(batch_size, budget // input_bpf))
    if max_frames is not None and int(max_frames) > 0:          # max_frames also caps the chunk size
        provisional_chunk = min(provisional_chunk, int(max_frames))

    if cal_bootstrap is not None:
        cal_out_raw, used_cal = cal_bootstrap
    else:
        cal_size = min(CALIBRATION_CAP, provisional_chunk, batch_size)
        cal_out_raw, used_cal = _run_chunk_with_oom_retry(
            fn, frames, static, 0, cal_size, label=label,
        )
    if _is_ui_dict(cal_out_raw):
        if strict:
            _DICT_RETURN_LABELS.add(label)
            log.info("[C2C autobatch] %s returns a ui dict: not chunked", label)
            return _call_node_fn(fn, **dict(frames), **dict(static))
        raise AutobatchError(f"{label} returns a ui dict")

    cal_out = _normalize_node_result(cal_out_raw)
    cal_tensors, cal_non_tensors = _validate_chunk_outputs(
        cal_out, used_cal, reference_non_tensors=None, label=label,
    )
    output_bpf = sum(_tensor_frame_bytes(t) for t in cal_tensors)
    bytes_per_frame = max(1, input_bpf + output_bpf)
    full_chunk = max(1, min(batch_size, budget // bytes_per_frame))
    if max_frames is not None and int(max_frames) > 0:          # max_frames also caps the chunk size
        full_chunk = min(full_chunk, int(max_frames))

    buffers = _allocate_output_buffers(cal_tensors, batch_size)
    _copy_chunk_to_buffers(cal_tensors, buffers, 0, used_cal)

    pbar = _progress_bar(batch_size)
    if used_cal:
        pbar.update(used_cal)

    pos = used_cal
    chunk_count = 1 if used_cal else 0
    while pos < batch_size:
        _throw_interrupt()
        chunk_len = min(full_chunk, batch_size - pos)
        chunk_out_raw, used_chunk = _run_chunk_with_oom_retry(
            fn, frames, static, pos, chunk_len, label=label,
        )
        chunk_out = _normalize_node_result(chunk_out_raw)
        chunk_tensors, _ = _validate_chunk_outputs(
            chunk_out, used_chunk, reference_non_tensors=cal_non_tensors, label=label,
        )
        _copy_chunk_to_buffers(chunk_tensors, buffers, pos, pos + used_chunk)
        pbar.update(used_chunk)
        pos += used_chunk
        chunk_count += 1

    log.info(
        "[C2C autobatch] %s: %d frames in %d chunks of %d (budget %s)",
        label,
        batch_size,
        chunk_count,
        full_chunk,
        _format_bytes(budget),
    )
    _record_chunked_call(label, batch_size, chunk_count, full_chunk, budget)
    if cal_non_tensors:
        return tuple(buffers) + tuple(cal_non_tensors)
    return tuple(buffers)


def _run_chunked_v3_impl(
    fn: Callable[..., Any],
    frames: Mapping[str, torch.Tensor],
    static: Mapping[str, Any],
    *,
    label: str,
    device: torch.device | str | None,
    work_factor: float,
    max_frames: int | None,
    budget_bytes: int | None,
    strict: bool = True,
    cal_bootstrap: tuple[Any, int] | None = None,
) -> Any:
    """Inner V3 chunking body (NodeOutput or tuple returns)."""
    if not _autobatch_enabled():
        return _call_node_fn(fn, **dict(frames), **dict(static))

    batch_size = _batch_size(frames)
    if batch_size <= 1:
        return _call_node_fn(fn, **dict(frames), **dict(static))

    if label in _DICT_RETURN_LABELS:
        return _call_node_fn(fn, **dict(frames), **dict(static))

    resolved_device = _resolve_device(device, frames)
    input_bpf = _input_bytes_per_frame(frames, work_factor)
    budget = _read_budget(resolved_device, budget_bytes=budget_bytes)
    if not _would_chunk(batch_size, input_bpf, budget, max_frames):
        return _call_node_fn(fn, **dict(frames), **dict(static))

    if budget is None:
        if max_frames is not None and int(max_frames) > 0 and batch_size > int(max_frames):
            budget = (input_bpf * batch_size) + 1
        else:
            return _call_node_fn(fn, **dict(frames), **dict(static))

    provisional_chunk = max(1, min(batch_size, budget // input_bpf))
    if max_frames is not None and int(max_frames) > 0:          # max_frames also caps the chunk size
        provisional_chunk = min(provisional_chunk, int(max_frames))

    if cal_bootstrap is not None:
        cal_out_raw, used_cal = cal_bootstrap
    else:
        cal_size = min(CALIBRATION_CAP, provisional_chunk, batch_size)
        cal_out_raw, used_cal = _run_chunk_with_oom_retry(
            fn, frames, static, 0, cal_size, label=label,
        )
    if _is_ui_dict(cal_out_raw):
        if strict:
            _DICT_RETURN_LABELS.add(label)
            log.info("[C2C autobatch] %s returns a ui dict: not chunked", label)
            return _call_node_fn(fn, **dict(frames), **dict(static))
        raise AutobatchError(f"{label} returns a ui dict")

    cal_out, needs_full_batch = _decode_v3_chunk(cal_out_raw)
    if needs_full_batch or cal_out is None:
        if strict:
            _DICT_RETURN_LABELS.add(label)
            if _is_node_output(cal_out_raw) and _node_output_non_chunkable(cal_out_raw):
                log.info(
                    "[C2C autobatch] %s returns NodeOutput with "
                    "ui/expand/block_execution: not chunked",
                    label,
                )
            else:
                log.info("[C2C autobatch] %s: return type not chunkable", label)
            return _call_node_fn(fn, **dict(frames), **dict(static))
        raise AutobatchError(f"{label} returned a non-chunkable result")

    cal_tensors, cal_non_tensors = _validate_chunk_outputs(
        cal_out, used_cal, reference_non_tensors=None, label=label,
    )
    output_bpf = sum(_tensor_frame_bytes(t) for t in cal_tensors)
    bytes_per_frame = max(1, input_bpf + output_bpf)
    full_chunk = max(1, min(batch_size, budget // bytes_per_frame))
    if max_frames is not None and int(max_frames) > 0:          # max_frames also caps the chunk size
        full_chunk = min(full_chunk, int(max_frames))

    buffers = _allocate_output_buffers(cal_tensors, batch_size)
    _copy_chunk_to_buffers(cal_tensors, buffers, 0, used_cal)

    pbar = _progress_bar(batch_size)
    if used_cal:
        pbar.update(used_cal)

    pos = used_cal
    chunk_count = 1 if used_cal else 0
    while pos < batch_size:
        _throw_interrupt()
        chunk_len = min(full_chunk, batch_size - pos)
        chunk_out_raw, used_chunk = _run_chunk_with_oom_retry(
            fn, frames, static, pos, chunk_len, label=label,
        )
        chunk_out, chunk_full = _decode_v3_chunk(chunk_out_raw)
        if chunk_full or chunk_out is None:
            raise AutobatchError(
                f"Automatic batching aborted: {label} returned a non-chunkable "
                "result mid-run. This node is not frame-independent."
            )
        chunk_tensors, _ = _validate_chunk_outputs(
            chunk_out, used_chunk, reference_non_tensors=cal_non_tensors, label=label,
        )
        _copy_chunk_to_buffers(chunk_tensors, buffers, pos, pos + used_chunk)
        pbar.update(used_chunk)
        pos += used_chunk
        chunk_count += 1

    log.info(
        "[C2C autobatch] %s: %d frames in %d chunks of %d (budget %s)",
        label,
        batch_size,
        chunk_count,
        full_chunk,
        _format_bytes(budget),
    )
    _record_chunked_call(label, batch_size, chunk_count, full_chunk, budget)
    return _encode_v3_result(cal_out_raw, buffers, cal_non_tensors)


def run_chunked_v3(
    orig_func: Callable[..., Any],
    cls: type,
    frames: Mapping[str, torch.Tensor],
    static: Mapping[str, Any],
    *,
    label: str,
    device: torch.device | str | None = None,
    work_factor: float = DEFAULT_WORK_FACTOR,
    max_frames: int | None = None,
    budget_bytes: int | None = None,
    strict: bool = True,
    cal_bootstrap: tuple[Any, int] | None = None,
) -> Any:
    """Run a V3 ``execute`` classmethod in memory-bounded frame chunks."""
    fn = lambda **kw: orig_func(cls, **kw)
    try:
        return _run_chunked_v3_impl(
            fn, frames, static,
            label=label,
            device=device,
            work_factor=work_factor,
            max_frames=max_frames,
            budget_bytes=budget_bytes,
            strict=strict,
            cal_bootstrap=cal_bootstrap,
        )
    except _NodeRaised as carrier:
        raise carrier.original from None
    except AutobatchError:
        raise
    except AutobatchMachineryError:
        raise
    except Exception as exc:
        raise AutobatchMachineryError(str(exc)) from exc


def run_chunked(
    fn: Callable[..., Any],
    frames: Mapping[str, torch.Tensor],
    static: Mapping[str, Any],
    *,
    label: str,
    device: torch.device | str | None = None,
    work_factor: float = DEFAULT_WORK_FACTOR,
    max_frames: int | None = None,
    budget_bytes: int | None = None,
    strict: bool = True,
    cal_bootstrap: tuple[Any, int] | None = None,
) -> Any:
    """Run ``fn`` in memory-bounded frame chunks.

    ``frames`` maps argument names to batched tensors (dim 0 is the frame
    axis). A 1-frame tensor is broadcast to every chunk. ``static`` holds all
    other kwargs passed through unchanged.
    """
    try:
        return _run_chunked_impl(
            fn, frames, static,
            label=label,
            device=device,
            work_factor=work_factor,
            max_frames=max_frames,
            budget_bytes=budget_bytes,
            strict=strict,
            cal_bootstrap=cal_bootstrap,
        )
    except _NodeRaised as carrier:
        raise carrier.original from None
    except AutobatchError:
        raise
    except AutobatchMachineryError:
        raise
    except Exception as exc:
        # ComfyUI's interrupt (InterruptProcessingException) is a BaseException: it never reaches
        # this handler and propagates unchanged, like the node's own exceptions.
        raise AutobatchMachineryError(str(exc)) from exc


def autobatch(
    *,
    frames: Sequence[str] = ("image",),
    work_factor: float = DEFAULT_WORK_FACTOR,
):
    """Decorator for frame-independent node methods."""

    frame_names = tuple(frames)

    def decorator(method: Callable[..., Any]):
        @functools.wraps(method)
        def wrapper(self, *args, **kwargs):
            bound = _bind_node_kwargs(method, self, *args, **kwargs)
            if bound is None:
                return method(self, *args, **kwargs)
            frame_inputs, static, batch_size = _frame_inputs_from_bound(
                bound, frame_names,
            )
            if not frame_inputs or batch_size <= 1:
                return method(self, *args, **kwargs)
            label = f"{self.__class__.__name__}.{method.__name__}"
            return run_chunked(
                lambda **kw: method(self, **kw),
                frame_inputs,
                static,
                label=label,
                work_factor=work_factor,
            )

        return wrapper

    return decorator


def _is_v3_node(cls: type) -> bool:
    # getattr(cls, "execute") returns a BOUND method, never the classmethod object: read it statically.
    try:
        from comfy_api.internal import _ComfyNodeInternal       # core 0.36: the V3 base
        if isinstance(cls, type) and issubclass(cls, _ComfyNodeInternal):
            return True
    except Exception:
        pass
    raw = inspect.getattr_static(cls, "execute", None)
    return isinstance(raw, classmethod) and hasattr(cls, "define_schema")


def _restore_attr(cls: type, name: str, orig: Any, defined_here: bool) -> None:
    # An inherited method is restored by removing our override, not by shadowing it with a copy.
    if defined_here:
        setattr(cls, name, orig)
        return
    try:
        delattr(cls, name)
    except AttributeError:
        setattr(cls, name, orig)


def _signature_for(fn: Callable[..., Any]) -> inspect.Signature:
    try:
        cached = _SIG_CACHE.get(fn)
    except TypeError:                      # not weak-referenceable (rare builtins): no caching
        return inspect.signature(fn)
    if cached is None:
        cached = inspect.signature(fn)
        try:
            _SIG_CACHE[fn] = cached
        except TypeError:
            pass
    return cached


def _bind_node_kwargs(
    fn: Callable[..., Any],
    self: Any,
    *args: Any,
    **kwargs: Any,
) -> dict[str, Any] | None:
    """Bind positional/keyword args to a flat kwargs dict (no self, no defaults)."""
    try:
        sig = _signature_for(fn)
        bound = sig.bind(self, *args, **kwargs)
    except TypeError:
        return None
    out = dict(bound.arguments)
    param_names = list(sig.parameters.keys())
    if param_names:
        out.pop(param_names[0], None)
    for name, param in sig.parameters.items():
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            extra = out.pop(name, None)
            if isinstance(extra, dict):
                out.update(extra)
    return out


def _frame_inputs_from_bound(
    bound: Mapping[str, Any],
    frame_names: Sequence[str],
) -> tuple[dict[str, torch.Tensor], dict[str, Any], int]:
    frame_inputs: dict[str, torch.Tensor] = {}
    batch_size = 0
    for name in frame_names:
        value = bound.get(name)
        if isinstance(value, torch.Tensor):
            frame_inputs[name] = value
            if int(value.shape[0]) > 1:
                batch_size = max(batch_size, int(value.shape[0]))
    static = {k: v for k, v in bound.items() if k not in frame_inputs}
    return frame_inputs, static, batch_size


def _refuse_class(class_name: str, reason: str) -> None:
    log.warning("[C2C autobatch] skip %s: %s", class_name, reason)


def _is_strict_class(class_name: str, cfg: Mapping[str, Any]) -> bool:
    nodes = cfg.get("nodes", {})
    return isinstance(nodes, dict) and class_name in nodes


def _chunk_plan(
    frames: Mapping[str, torch.Tensor],
    work_factor: float,
    max_frames: int | None,
    budget_bytes: int | None,
    device: torch.device | str | None,
) -> tuple[int, int | None, int]:
    batch_size = _batch_size(frames)
    resolved_device = _resolve_device(device, frames)
    input_bpf = _input_bytes_per_frame(frames, work_factor)
    budget = _read_budget(resolved_device, budget_bytes=budget_bytes)
    if budget is None and max_frames is not None and batch_size > int(max_frames):
        budget = (input_bpf * batch_size) + 1
    provisional_chunk = 1
    if budget is not None:
        provisional_chunk = max(1, min(batch_size, budget // max(1, input_bpf)))
        if max_frames is not None and int(max_frames) > 0:
            provisional_chunk = min(provisional_chunk, int(max_frames))
    return batch_size, budget, provisional_chunk


def _run_universal_chunked(
    fn: Callable[..., Any],
    class_name: str,
    frame_inputs: Mapping[str, torch.Tensor],
    static: Mapping[str, Any],
    *,
    spec: Mapping[str, Any],
    v3: bool,
    v3_cls: type | None,
    v3_func: Callable[..., Any] | None,
    max_frames: int | None,
    budget_bytes: int | None,
    orig_call: Callable[[], Any],
) -> Any:
    batch_size, budget, provisional_chunk = _chunk_plan(
        frame_inputs,
        _effective_work_factor(class_name, frame_inputs, spec),
        max_frames,
        budget_bytes,
        None,
    )
    work_factor = _effective_work_factor(class_name, frame_inputs, spec)
    resolved_device = _resolve_device(None, frame_inputs)
    input_bpf = _input_bytes_per_frame(frame_inputs, work_factor)
    if not _would_chunk(batch_size, input_bpf, budget, max_frames):
        return orig_call()

    if class_name in _UNSAFE:
        return orig_call()

    cal_bootstrap: tuple[Any, int] | None = None
    if class_name not in _SAFE:
        passed, cal_out, used_cal, max_diff, reason = _run_frame_probe(
            fn,
            frame_inputs,
            static,
            provisional_chunk,
            class_name=class_name,
            v3=v3,
            device=resolved_device,
        )
        if not passed:
            _mark_unsafe(class_name, reason or f"probe max diff {max_diff:.6g}")
            log.info(
                "[C2C autobatch] %s: %s — running full batch unchunked",
                class_name,
                _UNSAFE[class_name],
            )
            return orig_call()
        _mark_safe(class_name)
        cal_bootstrap = (cal_out, used_cal)

    try:
        if v3 and v3_cls is not None and v3_func is not None:
            return run_chunked_v3(
                v3_func, v3_cls,
                frame_inputs, static,
                label=class_name,
                work_factor=work_factor,
                max_frames=max_frames,
                budget_bytes=budget_bytes,
                strict=False,
                cal_bootstrap=cal_bootstrap,
            )
        return run_chunked(
            fn,
            frame_inputs,
            static,
            label=class_name,
            work_factor=work_factor,
            max_frames=max_frames,
            budget_bytes=budget_bytes,
            strict=False,
            cal_bootstrap=cal_bootstrap,
        )
    except AutobatchError as exc:
        _mark_unsafe(class_name, str(exc))
        log.info(
            "[C2C autobatch] %s: %s — running full batch unchunked",
            class_name,
            exc,
        )
        return orig_call()
    except AutobatchMachineryError:
        raise


def _make_universal_wrapper(
    orig: Callable[..., Any],
    class_name: str,
    frame_names: Sequence[str],
    work_factor: float,
    max_frames: int | None,
    budget_bytes: int | None,
    spec: Mapping[str, Any],
):
    names = tuple(frame_names)

    @functools.wraps(orig)
    def wrapped(self, *args, **kwargs):
        if class_name in _MACHINERY_FAILED:
            return orig(self, *args, **kwargs)

        bound = _bind_node_kwargs(orig, self, *args, **kwargs)
        if bound is None:
            return orig(self, *args, **kwargs)

        frame_inputs, static, batch_size = _frame_inputs_from_bound(bound, names)
        if not frame_inputs or batch_size <= 1:
            return orig(self, *args, **kwargs)

        if _has_batched_non_frame_input(bound, names, batch_size):
            _log_refusal_once(
                class_name,
                "batched non-frame tensor or LATENT alongside frame inputs",
            )
            return orig(self, *args, **kwargs)

        try:
            return _run_universal_chunked(
                lambda **kw: orig(self, **kw),
                class_name,
                frame_inputs,
                static,
                spec=spec,
                v3=False,
                v3_cls=None,
                v3_func=None,
                max_frames=max_frames,
                budget_bytes=budget_bytes,
                orig_call=lambda: orig(self, *args, **kwargs),
            )
        except AutobatchMachineryError as exc:
            if _unwrap_class(class_name):
                log.warning(
                    "[C2C autobatch] machinery failure on %s: %s — unwrapped",
                    class_name,
                    exc,
                )
            return orig(self, *args, **kwargs)

    return wrapped


def _make_universal_v3_wrapper(
    orig_cm: classmethod,
    class_name: str,
    frame_names: Sequence[str],
    work_factor: float,
    max_frames: int | None,
    budget_bytes: int | None,
    spec: Mapping[str, Any],
):
    orig_func = orig_cm.__func__
    names = tuple(frame_names)

    def execute_wrapped(cls, *args, **kwargs):
        if class_name in _MACHINERY_FAILED:
            return orig_func(cls, *args, **kwargs)

        bound = _bind_node_kwargs(orig_func, cls, *args, **kwargs)
        if bound is None:
            return orig_func(cls, *args, **kwargs)

        frame_inputs, static, batch_size = _frame_inputs_from_bound(bound, names)
        if not frame_inputs or batch_size <= 1:
            return orig_func(cls, *args, **kwargs)

        if _has_batched_non_frame_input(bound, names, batch_size):
            _log_refusal_once(
                class_name,
                "batched non-frame tensor or LATENT alongside frame inputs",
            )
            return orig_func(cls, *args, **kwargs)

        try:
            return _run_universal_chunked(
                lambda **kw: orig_func(cls, **kw),
                class_name,
                frame_inputs,
                static,
                spec=spec,
                v3=True,
                v3_cls=cls,
                v3_func=orig_func,
                max_frames=max_frames,
                budget_bytes=budget_bytes,
                orig_call=lambda: orig_func(cls, *args, **kwargs),
            )
        except AutobatchMachineryError as exc:
            if _unwrap_class(class_name):
                log.warning(
                    "[C2C autobatch] machinery failure on %s: %s — unwrapped",
                    class_name,
                    exc,
                )
            return orig_func(cls, *args, **kwargs)

    return classmethod(execute_wrapped)


def _make_wrapper(
    orig: Callable[..., Any],
    class_name: str,
    frame_names: Sequence[str],
    work_factor: float,
    max_frames: int | None,
    budget_bytes: int | None,
):
    names = tuple(frame_names)

    @functools.wraps(orig)
    def wrapped(self, *args, **kwargs):
        if class_name in _MACHINERY_FAILED:
            return orig(self, *args, **kwargs)

        bound = _bind_node_kwargs(orig, self, *args, **kwargs)
        if bound is None:
            return orig(self, *args, **kwargs)

        frame_inputs, static, batch_size = _frame_inputs_from_bound(bound, names)
        if not frame_inputs or batch_size <= 1:
            return orig(self, *args, **kwargs)

        label = class_name
        try:
            return run_chunked(
                lambda **kw: orig(self, **kw),
                frame_inputs,
                static,
                label=label,
                work_factor=work_factor,
                max_frames=max_frames,
                budget_bytes=budget_bytes,
            )
        except AutobatchMachineryError as exc:
            if _unwrap_class(class_name):
                log.warning(
                    "[C2C autobatch] machinery failure on %s: %s — unwrapped",
                    class_name,
                    exc,
                )
            return orig(self, *args, **kwargs)

    return wrapped


def _make_v3_wrapper(
    orig_cm: classmethod,
    class_name: str,
    frame_names: Sequence[str],
    work_factor: float,
    max_frames: int | None,
    budget_bytes: int | None,
):
    orig_func = orig_cm.__func__
    names = tuple(frame_names)

    def execute_wrapped(cls, *args, **kwargs):
        if class_name in _MACHINERY_FAILED:
            return orig_func(cls, *args, **kwargs)

        bound = _bind_node_kwargs(orig_func, cls, *args, **kwargs)
        if bound is None:
            return orig_func(cls, *args, **kwargs)

        frame_inputs, static, batch_size = _frame_inputs_from_bound(bound, names)
        if not frame_inputs or batch_size <= 1:
            return orig_func(cls, *args, **kwargs)

        label = class_name
        try:
            return run_chunked_v3(
                orig_func, cls,
                frame_inputs, static,
                label=label,
                work_factor=work_factor,
                max_frames=max_frames,
                budget_bytes=budget_bytes,
            )
        except AutobatchMachineryError as exc:
            if _unwrap_class(class_name):
                log.warning(
                    "[C2C autobatch] machinery failure on %s: %s — unwrapped",
                    class_name,
                    exc,
                )
            return orig_func(cls, *args, **kwargs)

    return classmethod(execute_wrapped)


def _unwrap_class(class_name: str) -> bool:
    entry = _WRAPPED.pop(class_name, None)
    _MACHINERY_FAILED.add(class_name)
    if entry is None:
        return False
    cls, orig, fn_name = entry
    _restore_attr(cls, fn_name, orig, _DEFINED_HERE.pop(class_name, True))
    if hasattr(cls, _ORIG_ATTR):
        delattr(cls, _ORIG_ATTR)
    return True


def install_wrappers(
    node_class_mappings: Mapping[str, type],
    cfg: Mapping[str, Any] | None = None,
    display_mappings: Mapping[str, str] | None = None,
) -> list[str]:
    """Wrap listed node classes. Returns names actually wrapped."""
    config = dict(cfg) if cfg is not None else load_config()
    if _resolve_mode(config) == "off":
        return []

    wrapped_now: list[str] = []
    budget_bytes = _budget_bytes_from_config(config)
    specs = _effective_wrap_specs(config, node_class_mappings, display_mappings)
    for class_name, spec in specs.items():
        if class_name in _WRAPPED:
            continue
        cls = node_class_mappings.get(class_name)
        if cls is None:
            _refuse_class(class_name, "class not found")
            continue
        if getattr(cls, "OUTPUT_NODE", False):
            _refuse_class(class_name, "OUTPUT_NODE")
            continue
        if getattr(cls, "INPUT_IS_LIST", False):
            _refuse_class(class_name, "INPUT_IS_LIST")
            continue
        frames_spec = spec.get("frames", ["image"])
        if not isinstance(frames_spec, (list, tuple)) or not frames_spec:
            _refuse_class(class_name, "invalid frames spec")
            continue
        wf = float(spec.get("work_factor", DEFAULT_WORK_FACTOR))
        node_max_frames = _max_frames_from_config(config, spec)
        strict = _is_strict_class(class_name, config)
        if _is_v3_node(cls):
            orig_cm = inspect.getattr_static(cls, "execute", None)
            if orig_cm is None or not isinstance(orig_cm, classmethod):
                _refuse_class(class_name, "execute missing")
                continue
            orig_func = orig_cm.__func__
            if inspect.iscoroutinefunction(orig_func):
                _refuse_class(class_name, "async execute")
                continue
            if strict:
                wrapper = _make_v3_wrapper(
                    orig_cm, class_name, frames_spec, wf, node_max_frames, budget_bytes,
                )
            else:
                wrapper = _make_universal_v3_wrapper(
                    orig_cm, class_name, frames_spec, wf, node_max_frames, budget_bytes, spec,
                )
            _DEFINED_HERE[class_name] = "execute" in cls.__dict__
            cls.execute = wrapper
            setattr(cls, _ORIG_ATTR, orig_cm)
            _WRAPPED[class_name] = (cls, orig_cm, "execute")
            _WRAP_STRICT[class_name] = strict
            wrapped_now.append(class_name)
            continue
        fn_name = getattr(cls, "FUNCTION", None)
        if not fn_name or not isinstance(fn_name, str):
            _refuse_class(class_name, "FUNCTION missing")
            continue
        orig = getattr(cls, fn_name, None)
        if orig is None or not callable(orig):
            _refuse_class(class_name, "FUNCTION missing")
            continue
        if strict:
            wrapper = _make_wrapper(
                orig, class_name, frames_spec, wf, node_max_frames, budget_bytes,
            )
        else:
            wrapper = _make_universal_wrapper(
                orig, class_name, frames_spec, wf, node_max_frames, budget_bytes, spec,
            )
        _DEFINED_HERE[class_name] = fn_name in cls.__dict__
        setattr(cls, fn_name, wrapper)
        setattr(cls, _ORIG_ATTR, orig)
        _WRAPPED[class_name] = (cls, orig, fn_name)
        _WRAP_STRICT[class_name] = strict
        wrapped_now.append(class_name)

    if wrapped_now:
        log.info("[C2C autobatch] wrapped: %s", ", ".join(sorted(wrapped_now)))
    return wrapped_now


def uninstall_wrappers() -> None:
    for class_name, (cls, orig, fn_name) in list(_WRAPPED.items()):
        _restore_attr(cls, fn_name, orig, _DEFINED_HERE.pop(class_name, True))
        if hasattr(cls, _ORIG_ATTR):
            delattr(cls, _ORIG_ATTR)
    _WRAPPED.clear()
    _WRAP_STRICT.clear()


def _sync_wrappers_from_config() -> None:
    global _CONFIG_MTIME, _CONFIG_CACHE
    mtime = _config_mtime()
    if _CONFIG_CACHE is not None and mtime == _CONFIG_MTIME:
        return
    _CONFIG_MTIME = mtime
    _CONFIG_CACHE = load_config()
    uninstall_wrappers()
    if _resolve_mode(_CONFIG_CACHE) != "off":
        try:
            import nodes  # noqa: WPS433 — merged ComfyUI mappings

            display_mappings = getattr(nodes, "NODE_DISPLAY_NAME_MAPPINGS", None)
            install_wrappers(
                nodes.NODE_CLASS_MAPPINGS,
                _CONFIG_CACHE,
                display_mappings if isinstance(display_mappings, dict) else None,
            )
        except Exception as exc:
            log.warning("[C2C autobatch] install failed: %s", exc)


def on_prompt(json_data: dict[str, Any]) -> dict[str, Any]:
    """PromptServer hook: sync wrappers when config changes."""
    _sync_wrappers_from_config()
    return json_data


def register_routes() -> bool:
    """Idempotently register autobatch HTTP routes on PromptServer."""
    global _ROUTES_REGISTERED
    if _ROUTES_REGISTERED:
        return True
    try:
        from aiohttp import web
        import server  # noqa: WPS433
    except Exception as exc:
        log.debug("[C2C autobatch] route registration skipped: %s", exc)
        return False
    inst = getattr(server.PromptServer, "instance", None)
    routes = getattr(inst, "routes", None) if inst else None
    if routes is None or not hasattr(routes, "get"):
        return False

    @routes.get("/c2c/autobatch/status")
    async def _autobatch_status(_request):  # noqa: ANN001
        return web.json_response(status())

    @routes.get("/c2c/autobatch/config")
    async def _autobatch_get_config(_request):  # noqa: ANN001
        loop = asyncio.get_running_loop()
        cfg = await loop.run_in_executor(None, load_config)
        return web.json_response(cfg)

    @routes.post("/c2c/autobatch/config")
    async def _autobatch_post_config(request):  # noqa: ANN001
        try:
            patch = await request.json()
        except Exception as exc:
            return web.json_response(
                {"ok": False, "error": f"invalid JSON: {exc}"},
                status=400,
            )
        if not isinstance(patch, dict):
            return web.json_response(
                {"ok": False, "error": "payload must be a JSON object"},
                status=400,
            )
        loop = asyncio.get_running_loop()
        try:
            current = await loop.run_in_executor(None, load_config)
            merged = merge_config(current, patch)
            saved = await loop.run_in_executor(None, save_config, merged)
        except ValueError as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=500)
        global _CONFIG_MTIME, _CONFIG_CACHE
        _CONFIG_MTIME = None
        _CONFIG_CACHE = None
        _sync_wrappers_from_config()
        return web.json_response(saved)

    _ROUTES_REGISTERED = True
    log.info(
        "[C2C autobatch] routes registered: "
        "GET /c2c/autobatch/status, GET/POST /c2c/autobatch/config"
    )
    return True


def register_prompt_hook() -> None:
    """Register ``on_prompt`` with PromptServer (called from pack ``__init__``)."""
    global _PROMPT_HOOK_REGISTERED
    register_routes()
    if _PROMPT_HOOK_REGISTERED:
        return
    import server  # noqa: WPS433

    server.PromptServer.instance.add_on_prompt_handler(on_prompt)
    _PROMPT_HOOK_REGISTERED = True
