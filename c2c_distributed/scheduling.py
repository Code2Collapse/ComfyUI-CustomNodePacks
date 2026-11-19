"""L9.174 (A25): live resource monitor, workload suitability, memory-aware GPU selection, worker preflight.

Used by C2C GPU Inventory and C2C Multi-GPU Split (c2c_distributed/nodes.py). Every collector is a small module-level
function so tests can replace it; every collector fails soft (null + a note) - a monitor must never take the node down.
Nothing here changes how a model is split: it only decides how many GPUs C2C Multi-GPU Split asks core for, using the
same device order core's create_multigpu_deepclones uses (primary = model.load_device, then
comfy.model_management.get_all_torch_devices(exclude_current=False) minus the primary, in that order).
"""
from __future__ import annotations

import time
from typing import Callable, Dict, List, Optional

GB = float(2 ** 30)


# ── collectors (monkeypatchable) ─────────────────────────────────────────────────────────────────────────────────────

def cuda_memory(index: int) -> Optional[dict]:
    """Free / total device memory (driver view) and this process's allocated / reserved bytes. None without CUDA."""
    import torch
    if not torch.cuda.is_available() or index >= torch.cuda.device_count():
        return None
    free, total = torch.cuda.mem_get_info(index)
    return {"free_gb": round(free / GB, 3), "total_gb": round(total / GB, 3),
            "allocated_gb": round(torch.cuda.memory_allocated(index) / GB, 3),
            "reserved_gb": round(torch.cuda.memory_reserved(index) / GB, 3)}


def nvml_utilisation() -> Dict[int, dict]:
    """{nvml index: {gpu_pct, memory_pct}}; {} when pynvml is absent (utilisation is then reported as null)."""
    try:
        import pynvml
        pynvml.nvmlInit()
    except Exception:  # noqa: BLE001
        return {}
    out: Dict[int, dict] = {}
    try:
        for i in range(pynvml.nvmlDeviceGetCount()):
            try:
                u = pynvml.nvmlDeviceGetUtilizationRates(pynvml.nvmlDeviceGetHandleByIndex(i))
                out[i] = {"gpu_pct": int(u.gpu), "memory_pct": int(u.memory)}
            except Exception:  # noqa: BLE001
                continue
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:  # noqa: BLE001
            pass
    return out


def loaded_models() -> Optional[List[dict]]:
    """Models ComfyUI holds right now: comfy.model_management.current_loaded_models (LoadedModel.device,
    model_loaded_memory()). None outside ComfyUI."""
    try:
        import comfy.model_management as mm
    except Exception:  # noqa: BLE001
        return None
    out = []
    for lm in list(getattr(mm, "current_loaded_models", []) or []):
        entry = {"device": str(getattr(lm, "device", None)), "loaded_gb": None,
                 "model": type(getattr(getattr(lm, "model", None), "model", None)).__name__}
        try:
            entry["loaded_gb"] = round(float(lm.model_loaded_memory()) / GB, 3)
        except Exception:  # noqa: BLE001
            pass
        out.append(entry)
    return out


def queue_state() -> Optional[dict]:
    """ComfyUI's prompt queue: PromptQueue.get_current_queue_volatile() -> (running, queued). None outside ComfyUI."""
    try:
        from server import PromptServer
        q = PromptServer.instance.prompt_queue
        running, queued = q.get_current_queue_volatile()
        return {"active_jobs": len(running), "pending_jobs": len(queued)}
    except Exception:  # noqa: BLE001
        return None


def live_snapshot(gpu_list: List[dict]) -> dict:
    """Live per-device memory / utilisation / loaded models plus the ComfyUI queue."""
    notes = []
    util = nvml_utilisation()
    if not util:
        notes.append("utilisation: pynvml not available - reported as null")
    models = loaded_models()
    if models is None:
        notes.append("loaded models: comfy.model_management not importable - reported as null")
    devices = []
    for g in gpu_list:
        i = int(g.get("index", len(devices)))
        try:
            mem = cuda_memory(i)
        except Exception as exc:  # noqa: BLE001
            mem = None
            notes.append(f"cuda:{i} memory: {type(exc).__name__}: {str(exc)[:120]}")
        u = util.get(i) if util else None
        mine = None if models is None else [m for m in models if m["device"] == f"cuda:{i}"]
        devices.append({
            "index": i, "name": g.get("name"),
            "memory": mem,
            "utilisation_pct": None if u is None else u["gpu_pct"],
            "loaded_models": None if mine is None else len(mine),
            "loaded_models_gb": None if mine is None else round(sum(m["loaded_gb"] or 0.0 for m in mine), 3),
            "available_gb": None if not mem else mem["free_gb"],
        })
    q = queue_state()
    if q is None:
        notes.append("queue: PromptServer not available - reported as null")
    return {"time": time.time(), "devices": devices, "queue": q, "notes": notes}


# ── suitability ──────────────────────────────────────────────────────────────────────────────────────────────────────

PRECISIONS = ("any", "fp32", "fp16", "bf16", "fp8", "fp4")


def _precision(g: dict, precision: str):
    """-> (state, reason). States: native | emulated (no tensor-core support, slower) | unsupported."""
    if precision in ("any", "fp32", "fp16"):
        return "native", f"{precision} runs on every CUDA architecture"
    if precision == "bf16":
        return ("native", "bf16 (cc >= 8.0)") if g.get("bf16") else \
            ("emulated (no tensor-core support, slower)", "cc < 8.0: torch emulates bf16, much slower")
    if precision == "fp8":
        return ("native", "fp8 tensor cores (cc >= 8.9)") if g.get("fp8_tensor_cores") else \
            ("emulated (no tensor-core support, slower)", "no fp8 tensor cores (cc < 8.9): weights are upcast")
    if precision == "fp4":
        return ("native", "fp4 tensor cores (cc >= 10.0)") if g.get("fp4_tensor_cores") else \
            ("emulated (no tensor-core support, slower)", "no fp4 tensor cores (cc < 10.0): weights are upcast")
    return "unsupported", f"unknown precision {precision!r}"


def suitability(report: dict, need_gb: float, precision: str = "any", gpus_needed: int = 1,
                live: Optional[dict] = None, headroom: float = 1.1) -> dict:
    """Estimate whether a workload of need_gb (per GPU) fits the GPUs in an inventory report. An ESTIMATE from total /
    free memory and the architecture flags - not a measured run."""
    if precision not in PRECISIONS:
        raise ValueError(f"workload_precision must be one of {', '.join(PRECISIONS)}, got {precision!r}")
    if need_gb <= 0:
        raise ValueError("workload_vram_gb must be > 0 to estimate suitability")
    gpus_needed = max(1, int(gpus_needed))
    free_by_index = {}
    for d in (live or {}).get("devices", []) or []:
        if d.get("memory"):
            free_by_index[d["index"]] = d["memory"]["free_gb"]
    want = float(need_gb) * float(headroom)
    rows = []
    for g in report.get("gpus", []) or []:
        total = float(g.get("memory_gb") or 0.0)
        free = free_by_index.get(g.get("index"))
        p_state, p_reason = _precision(g, precision)
        kernels = g.get("kernels")
        kernels_ok = kernels in ("native", "compatible", "ptx-jit") if kernels is not None else None
        fits_total = want <= total
        fits_free = None if free is None else want <= free
        ok = fits_total and (fits_free is not False) and p_state != "unsupported" and kernels_ok is not False
        rows.append({"index": g.get("index"), "name": g.get("name"), "memory_gb": total, "free_gb": free,
                     "needed_gb_with_headroom": round(want, 3), "fits_total": fits_total, "fits_free": fits_free,
                     "precision": p_state, "precision_reason": p_reason, "kernels": kernels,
                     "kernels_ok": kernels_ok, "suitable": ok})
    if not rows:
        verdict = "no CUDA GPU: this workload cannot run on a GPU here"
    else:
        n = sum(1 for r in rows if r["suitable"])
        verdict = (f"fits: {n} of {len(rows)} GPU(s) qualify, {gpus_needed} needed" if n >= gpus_needed else
                   f"does not fit: {n} of {len(rows)} GPU(s) qualify, {gpus_needed} needed")
    return {"workload": {"vram_gb": float(need_gb), "precision": precision, "gpus": gpus_needed,
                         "headroom": headroom}, "gpus": rows, "verdict": verdict,
            "note": "estimate from total/free memory and architecture flags, not a measured run"}


# ── memory-aware selection + worker preflight (C2C Multi-GPU Split) ──────────────────────────────────────────────────

def candidate_devices(model) -> list:
    """The devices core's create_multigpu_deepclones would use, in its order: the primary first, then the others."""
    import comfy.model_management as mm
    primary = getattr(model, "load_device", None)
    others = [d for d in mm.get_all_torch_devices(exclude_current=False) if d != primary]
    return [primary] + others


def device_free_gb(device) -> Optional[float]:
    """Free memory of one torch device in GB; None for non-CUDA devices (no budget to check)."""
    import torch
    if getattr(device, "type", None) != "cuda" or not torch.cuda.is_available():
        return None
    free, _total = torch.cuda.mem_get_info(device)
    return free / GB


def model_size_gb(model) -> float:
    try:
        return float(model.model_size()) / GB
    except Exception:  # noqa: BLE001
        return 0.0


def kernel_check(device) -> Optional[str]:
    """A small real matmul on the device compared with the CPU result. None = passed, else the error in words."""
    import torch
    try:
        g = torch.Generator().manual_seed(7)
        a, b = torch.randn(64, 64, generator=g), torch.randn(64, 64, generator=g)
        ref = a @ b
        out = (a.to(device) @ b.to(device)).cpu()
        if getattr(device, "type", None) == "cuda":
            torch.cuda.synchronize(device)
        err = float((out - ref).abs().max() / ref.abs().max().clamp_min(1e-12))
        if not err < 1e-2:
            return f"wrong result (relative error {err:.3g})"
        return None
    except Exception as exc:  # noqa: BLE001 - no kernel image, OOM, driver error
        return f"{type(exc).__name__}: {str(exc)[:200]}"


def plan_devices(model, gpus: int, memory_policy: str = "off", headroom_gb: float = 1.0,
                 preflight: bool = False, devices: Optional[list] = None,
                 free_fn: Optional[Callable] = None, check_fn: Optional[Callable] = None):
    """-> (gpus to request from usp.apply, report lines). gpus=0 means "all", as for the node.

    memory_policy: off | report | fit. fit caps the GPU count to 1 + the longest prefix of extra devices that have
    room for one model copy (model_size + headroom). preflight runs kernel_check on every device that would be used;
    a failed extra device caps the count before it, a failed primary raises."""
    if memory_policy not in ("off", "report", "fit"):
        raise ValueError(f"memory_policy must be off, report or fit, got {memory_policy!r}")
    if memory_policy == "off" and not preflight:
        return gpus, []
    free_fn = free_fn or device_free_gb
    check_fn = check_fn or kernel_check
    devs = list(devices) if devices is not None else candidate_devices(model)
    use = len(devs) if int(gpus) <= 0 else min(int(gpus), len(devs))
    lines: List[str] = []
    if use < 2:
        lines.append("scheduling: single device - nothing to schedule")
    cap = use
    if memory_policy != "off" and use >= 2:
        need = model_size_gb(model) + float(headroom_gb)
        lines.append(f"memory: one model copy needs ~{need:.2f} GB per GPU (weights {model_size_gb(model):.2f} GB "
                     f"+ headroom {float(headroom_gb):.2f} GB)")
        fit_prefix = None
        for i, d in enumerate(devs[:use]):
            free = free_fn(d)
            ok = None if free is None else free >= need
            lines.append(f"  {d}: free {'?' if free is None else f'{free:.2f} GB'} -> "
                         f"{'fits' if ok else ('unknown (not a CUDA device)' if ok is None else 'DOES NOT FIT')}")
            if i == 0 and ok is False:
                lines.append(f"  warning: the primary device {d} lacks room; it is chosen by the model loader, not "
                             f"by this node")
            if i > 0 and ok is False and fit_prefix is None:
                fit_prefix = i
        if memory_policy == "fit" and fit_prefix is not None:
            cap = min(cap, fit_prefix)
            lines.append(f"memory fit: using {cap} GPU(s) - {devs[fit_prefix]} and later devices lack room")
    if preflight:
        for i, d in enumerate(devs[:cap]):
            err = check_fn(d)
            if err is None:
                lines.append(f"preflight: {d} ok")
                continue
            if i == 0:
                raise RuntimeError(f"C2C Multi-GPU Split: the primary device {d} failed its kernel check ({err}). "
                                   f"Fix the GPU / driver or load the model on another device.")
            lines.append(f"preflight: {d} FAILED ({err}) - excluded with every later device; using {i} GPU(s)")
            cap = i
            break
    out = cap if cap != use else gpus
    return out, lines
