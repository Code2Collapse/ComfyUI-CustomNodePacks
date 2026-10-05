"""Automatic frame batching for ComfyUI nodes (owner amendment A5).

Provides ``run_chunked`` for memory-bounded per-frame execution, an
``@autobatch`` decorator for our own frame-independent nodes, and an
opt-in wrapper installed on the first queued prompt for listed core nodes.

Kill switch: ``C2C_AUTOBATCH=0`` disables chunking (single call).

Config file (``user/default/c2c/autobatch.json``, overridable via
``C2C_AUTOBATCH_CONFIG``) shape::

    {
      "enabled": false,
      "curated": false,
      "max_frames": 0,
      "budget_mb": null,
      "nodes": {
        "ImageScaleBy": {
          "frames": ["image"],
          "work_factor": 4.0,
          "max_frames": 0
        }
      }
    }

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

import functools
import inspect
import json
import logging
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

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
_SIG_CACHE: dict[int, inspect.Signature] = {}

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
        "enabled": False,
        "curated": False,
        "max_frames": 0,
        "budget_mb": None,
        "nodes": {},
    }


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
    return out


def _config_mtime() -> float | None:
    path = _config_path()
    if path is None or not path.is_file():
        return None
    try:
        return path.stat().st_mtime
    except OSError:
        return None


def _effective_node_specs(cfg: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
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
    cal_size = min(CALIBRATION_CAP, provisional_chunk, batch_size)

    cal_out_raw, used_cal = _run_chunk_with_oom_retry(
        fn, frames, static, 0, cal_size, label=label,
    )
    if _is_ui_dict(cal_out_raw):
        _DICT_RETURN_LABELS.add(label)
        log.info("[C2C autobatch] %s returns a ui dict: not chunked", label)
        return _call_node_fn(fn, **dict(frames), **dict(static))

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
    cal_size = min(CALIBRATION_CAP, provisional_chunk, batch_size)

    cal_out_raw, used_cal = _run_chunk_with_oom_retry(
        fn, frames, static, 0, cal_size, label=label,
    )
    if _is_ui_dict(cal_out_raw):
        _DICT_RETURN_LABELS.add(label)
        log.info("[C2C autobatch] %s returns a ui dict: not chunked", label)
        return _call_node_fn(fn, **dict(frames), **dict(static))

    cal_out, needs_full_batch = _decode_v3_chunk(cal_out_raw)
    if needs_full_batch or cal_out is None:
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
    key = id(fn)
    cached = _SIG_CACHE.get(key)
    if cached is None:
        cached = inspect.signature(fn)
        _SIG_CACHE[key] = cached
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
) -> list[str]:
    """Wrap listed node classes. Returns names actually wrapped."""
    config = dict(cfg) if cfg is not None else load_config()
    if not config.get("enabled"):
        return []

    wrapped_now: list[str] = []
    budget_bytes = _budget_bytes_from_config(config)
    for class_name, spec in _effective_node_specs(config).items():
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
        if _is_v3_node(cls):
            orig_cm = inspect.getattr_static(cls, "execute", None)
            if orig_cm is None or not isinstance(orig_cm, classmethod):
                _refuse_class(class_name, "execute missing")
                continue
            orig_func = orig_cm.__func__
            if inspect.iscoroutinefunction(orig_func):
                _refuse_class(class_name, "async execute")
                continue
            wrapper = _make_v3_wrapper(
                orig_cm, class_name, frames_spec, wf, node_max_frames, budget_bytes,
            )
            _DEFINED_HERE[class_name] = "execute" in cls.__dict__
            cls.execute = wrapper
            setattr(cls, _ORIG_ATTR, orig_cm)
            _WRAPPED[class_name] = (cls, orig_cm, "execute")
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
        wrapper = _make_wrapper(
            orig, class_name, frames_spec, wf, node_max_frames, budget_bytes,
        )
        _DEFINED_HERE[class_name] = fn_name in cls.__dict__
        setattr(cls, fn_name, wrapper)
        setattr(cls, _ORIG_ATTR, orig)
        _WRAPPED[class_name] = (cls, orig, fn_name)
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


def _sync_wrappers_from_config() -> None:
    global _CONFIG_MTIME, _CONFIG_CACHE
    mtime = _config_mtime()
    if _CONFIG_CACHE is not None and mtime == _CONFIG_MTIME:
        return
    _CONFIG_MTIME = mtime
    _CONFIG_CACHE = load_config()
    uninstall_wrappers()
    if _CONFIG_CACHE.get("enabled"):
        try:
            import nodes  # noqa: WPS433 — merged ComfyUI mappings

            install_wrappers(nodes.NODE_CLASS_MAPPINGS, _CONFIG_CACHE)
        except Exception as exc:
            log.warning("[C2C autobatch] install failed: %s", exc)


def on_prompt(json_data: dict[str, Any]) -> dict[str, Any]:
    """PromptServer hook: sync wrappers when config changes."""
    _sync_wrappers_from_config()
    return json_data


def register_prompt_hook() -> None:
    """Register ``on_prompt`` with PromptServer (called from pack ``__init__``)."""
    global _PROMPT_HOOK_REGISTERED
    if _PROMPT_HOOK_REGISTERED:
        return
    import server  # noqa: WPS433

    server.PromptServer.instance.add_on_prompt_handler(on_prompt)
    _PROMPT_HOOK_REGISTERED = True
