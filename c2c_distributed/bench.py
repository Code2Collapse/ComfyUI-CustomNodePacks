"""Reproducible GPU benchmark harness (ledger L9.217, A26 4.8).

A benchmark is a list of CONFIGURATIONS of one workload; the first is the baseline (normally one GPU). For every
configuration it records: physical GPUs used (UUID / name), model load time, peak VRAM per GPU, peak system RAM, per-
step and total runtime, communication time, idle time per worker, scheduling overhead, output shape and numerical
difference from the baseline, PSNR vs the baseline, seam and temporal-flicker figures, failed / retried tasks, and
speed-up / scaling efficiency against the baseline.

A workload is ``fn(cfg, rec) -> output tensor``; it reports its own phases through ``rec``:
    with rec.phase("load"): ...           model loading (counted as load time)
    with rec.step(): ...                  one sampler step / one unit of work (per-step times)
    with rec.comm(): ...                  time spent exchanging tensors between devices
    rec.service(svc, job_info)            a TileService run: per-worker busy / idle, failures, retries

LABELS - never stronger than the hardware: MEASURED_ON_<n>_GPUS (n distinct physical GPUs), SINGLE_GPU_ONLY (one
physical GPU, whatever the worker count: several workers on one GPU say nothing about scaling), CPU_ONLY, SIMULATED
(C2C_DISTRIBUTED_SIMULATE). Speed-up and scaling efficiency are only reported for MEASURED_ON_<n>_GPUS rows.
"""
from __future__ import annotations

import contextlib
import json
import os
import platform
import threading
import time
from pathlib import Path
from typing import Callable, Optional

import torch


class Recorder:
    def __init__(self):
        self.phases: dict = {}
        self.steps: list = []
        self.comm_s = 0.0
        self.services: list = []

    @contextlib.contextmanager
    def phase(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.phases[name] = self.phases.get(name, 0.0) + time.perf_counter() - t0

    @contextlib.contextmanager
    def step(self):
        _sync_all()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            _sync_all()
            self.steps.append(time.perf_counter() - t0)

    @contextlib.contextmanager
    def comm(self):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.comm_s += time.perf_counter() - t0

    def service(self, svc, job_info: dict, wall_s: float) -> None:
        snap = svc.snapshot()
        workers = {}
        for wid, w in snap["workers"].items():
            slots = w.get("slots", 1)
            workers[wid] = {"device": w["device"], "busy_s": w.get("busy_s", 0.0),
                            "idle_s": max(0.0, wall_s * slots - w.get("busy_s", 0.0)), "tasks": w.get("completed", 0)}
        busiest = max((w["busy_s"] for w in workers.values()), default=0.0)
        self.services.append({"workers": workers, "failures": len(job_info.get("failures", [])),
                              "retries": job_info.get("retries", 0),
                              "scheduling_overhead_s": max(0.0, wall_s - busiest)})


def _sync_all() -> None:
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            torch.cuda.synchronize(i)


class _RamSampler:
    def __init__(self, every: float = 0.02):
        self.every, self.peak, self._stop = every, 0, threading.Event()
        self._t = threading.Thread(target=self._loop, daemon=True)

    def _loop(self):
        try:
            import psutil
            proc = psutil.Process()
        except Exception:  # noqa: BLE001
            return
        while not self._stop.is_set():
            self.peak = max(self.peak, proc.memory_info().rss)
            self._stop.wait(self.every)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        self._t.join(timeout=2)


def gpu_identity(devices) -> list:
    out = []
    for d in devices:
        d = torch.device(d)
        if d.type != "cuda":
            continue
        p = torch.cuda.get_device_properties(d)
        out.append({"device": str(d), "name": p.name, "uuid": str(getattr(p, "uuid", "")) or None})
    return out


def label(devices, simulated: bool = False) -> str:
    if simulated or os.environ.get("C2C_DISTRIBUTED_SIMULATE"):
        return "SIMULATED"
    ids = gpu_identity(devices)
    n = len({g["uuid"] or g["device"] for g in ids})
    if n == 0:
        return "CPU_ONLY"
    return "SINGLE_GPU_ONLY" if n == 1 else f"MEASURED_ON_{n}_GPUS"


def psnr(a: torch.Tensor, b: torch.Tensor) -> Optional[float]:
    a, b = a.double(), b.double()
    rng = float(b.max() - b.min()) or 1.0
    mse = float(((a - b) ** 2).mean())
    return None if mse == 0 else 10.0 * __import__("math").log10(rng * rng / mse)


def flicker(video: torch.Tensor) -> Optional[float]:
    """Std over time of the detrended per-frame mean (video [T, ...]); None for one frame."""
    if video.dim() < 2 or video.shape[0] < 3:
        return None
    m = video.double().reshape(video.shape[0], -1).mean(1)
    t = torch.arange(m.numel(), dtype=torch.float64)
    a = ((t - t.mean()) * (m - m.mean())).sum() / ((t - t.mean()) ** 2).sum()
    return float((m - (m.mean() + a * (t - t.mean()))).std())


def run(workload: Callable, configs: list, *, name: str, out: Optional[str] = None, seam: Optional[Callable] = None,
        is_video: bool = False) -> dict:
    """Run every config (the first = baseline), return the report (also written to ``out`` as JSON)."""
    rows, base_out, base_time = [], None, None
    for cfg in configs:
        devices = cfg.get("devices", [])
        rec = Recorder()
        if torch.cuda.is_available():
            torch.cuda.init()                                   # peak-memory calls need an initialised context
            for d in devices:
                dd = torch.device(d)
                if dd.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(dd.index if dd.index is not None else 0)
        with _RamSampler() as ram:
            t0 = time.perf_counter()
            output = workload(cfg, rec)
            _sync_all()
            total = time.perf_counter() - t0
        lab = label(devices, bool(cfg.get("simulated")))
        row = {"config": {k: v for k, v in cfg.items() if k not in ("devices",)},
               "devices": [str(d) for d in devices], "gpus": gpu_identity(devices), "label": lab,
               "load_s": round(rec.phases.get("load", 0.0), 4), "total_s": round(total, 4),
               "steps": len(rec.steps), "step_s_median": round(sorted(rec.steps)[len(rec.steps) // 2], 5)
               if rec.steps else None, "comm_s": round(rec.comm_s, 4), "peak_ram_bytes": ram.peak or None,
               "peak_vram_bytes": {str(d): int(torch.cuda.max_memory_allocated(torch.device(d).index or 0))
                                   for d in devices if torch.device(d).type == "cuda"},
               "service": rec.services, "output_shape": list(output.shape),
               "output_dtype": str(output.dtype).replace("torch.", "")}
        if base_out is None:
            base_out, base_time = output.detach().cpu(), total
            row["baseline"] = True
        else:
            o = output.detach().cpu()
            row["max_abs_diff_vs_baseline"] = float((o.double() - base_out.double()).abs().max()) \
                if o.shape == base_out.shape else None
            row["psnr_vs_baseline_db"] = psnr(o, base_out) if o.shape == base_out.shape else None
            if lab.startswith("MEASURED_ON_"):
                n = int(lab.split("_")[2])
                row["speed_up"] = round(base_time / total, 3)
                row["scaling_efficiency"] = round(base_time / total / n, 3)
            else:
                row["speed_up"] = row["scaling_efficiency"] = None
                row["why_no_speed_up"] = f"{lab}: speed-up is only reported for distinct physical GPUs"
        if seam is not None:
            row["seam_ratio"] = seam(output)
        if is_video:
            row["flicker"] = flicker(output)
        rows.append(row)
    rep = {"benchmark": name, "host": platform.platform(), "torch": torch.__version__,
           "cuda": torch.version.cuda, "rows": rows,
           "verdict": "PERFORMANCE_VERIFIED" if any(r["label"].startswith("MEASURED_ON_") and r.get("speed_up")
                                                    for r in rows) else "NOT VERIFIED (no multi-GPU measurement)"}
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(rep, indent=1, default=str), encoding="utf-8")
    return rep
