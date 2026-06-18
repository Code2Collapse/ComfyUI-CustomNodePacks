"""c2c_sys_metrics.py - backend metrics endpoint for the Stats Pill.

Exposes:
    GET /c2c/sys/metrics  ->  {
        "ok": bool,
        "ts": float,
        "gpu": {
            "available": bool,
            "util_pct": int|None,        # primary GPU utilisation %
            "temp_c":   int|None,        # primary GPU temperature C
            "vram_used_gb": float|None,
            "vram_total_gb": float|None,
            "name":     str|None,
            "devices":  [{name, util_pct, temp_c, vram_used_gb, vram_total_gb}],
        },
        "cpu": {
            "util_pct": float,           # 0.0 - 100.0
            "logical":  int,
        },
        "ram": {
            "used_gb":  float,
            "total_gb": float,
            "pct":      float,
        },
        "source": "pynvml" | "nvidia-smi" | "none",
    }

Data sources, in priority order:
    1. pynvml (Python NVIDIA Management Library) - lowest overhead.
    2. nvidia-smi subprocess (cached for SHELL_CACHE_S seconds).
    3. None - returns available=false.

psutil powers CPU and RAM. If psutil is missing both fields become 0 and
{"ok": True, "ram":{"total_gb":0, ...}} is returned so the JS can render
"--" rather than throw.

Cached at 1.0s to avoid hammering nvidia-smi when many UI panes poll.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import time
from typing import Any, Dict, List, Optional


_CACHE: Dict[str, Any] = {"ts": 0.0, "data": None}
_CACHE_TTL_S = 1.0
_SHELL_CACHE_S = 2.0
_SHELL_CACHE: Dict[str, Any] = {"ts": 0.0, "data": None}

# Probe pynvml lazily — some installs ship it but with a broken libnvidia-ml.
_PYNVML = None
_PYNVML_PROBED = False


def _try_pynvml():
    global _PYNVML, _PYNVML_PROBED
    if _PYNVML_PROBED:
        return _PYNVML
    _PYNVML_PROBED = True
    try:
        import pynvml  # type: ignore
        pynvml.nvmlInit()
        _PYNVML = pynvml
        return _PYNVML
    except Exception:
        _PYNVML = None
        return None


def _gpu_pynvml() -> Optional[Dict[str, Any]]:
    p = _try_pynvml()
    if p is None:
        return None
    try:
        count = p.nvmlDeviceGetCount()
        devs: List[Dict[str, Any]] = []
        for i in range(count):
            h = p.nvmlDeviceGetHandleByIndex(i)
            mem = p.nvmlDeviceGetMemoryInfo(h)
            try:
                util = p.nvmlDeviceGetUtilizationRates(h).gpu
            except Exception:
                util = None
            try:
                temp = p.nvmlDeviceGetTemperature(h, p.NVML_TEMPERATURE_GPU)
            except Exception:
                temp = None
            try:
                name = p.nvmlDeviceGetName(h)
                if isinstance(name, bytes):
                    name = name.decode("utf-8", "replace")
            except Exception:
                name = f"GPU{i}"
            devs.append({
                "name": name,
                "util_pct": int(util) if util is not None else None,
                "temp_c": int(temp) if temp is not None else None,
                "vram_used_gb": round((mem.total - mem.free) / 1073741824, 3),
                "vram_total_gb": round(mem.total / 1073741824, 3),
            })
        if not devs:
            return None
        head = devs[0]
        return {
            "available": True,
            "util_pct": head.get("util_pct"),
            "temp_c": head.get("temp_c"),
            "vram_used_gb": head.get("vram_used_gb"),
            "vram_total_gb": head.get("vram_total_gb"),
            "name": head.get("name"),
            "devices": devs,
            "_source": "pynvml",
        }
    except Exception:
        return None


def _gpu_nvidia_smi() -> Optional[Dict[str, Any]]:
    now = time.time()
    if _SHELL_CACHE["data"] is not None and (now - _SHELL_CACHE["ts"]) < _SHELL_CACHE_S:
        return _SHELL_CACHE["data"]
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.check_output(
            [exe, "--query-gpu=name,utilization.gpu,temperature.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            timeout=2.5,
            stderr=subprocess.DEVNULL,
        ).decode("utf-8", "replace")
    except Exception:
        return None
    devs: List[Dict[str, Any]] = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        try:
            name = parts[0]
            util = int(parts[1])
            temp = int(parts[2])
            used_mib = float(parts[3])
            total_mib = float(parts[4])
        except Exception:
            continue
        devs.append({
            "name": name,
            "util_pct": util,
            "temp_c": temp,
            "vram_used_gb": round(used_mib / 1024.0, 3),
            "vram_total_gb": round(total_mib / 1024.0, 3),
        })
    if not devs:
        return None
    head = devs[0]
    data = {
        "available": True,
        "util_pct": head["util_pct"],
        "temp_c": head["temp_c"],
        "vram_used_gb": head["vram_used_gb"],
        "vram_total_gb": head["vram_total_gb"],
        "name": head["name"],
        "devices": devs,
        "_source": "nvidia-smi",
    }
    _SHELL_CACHE["data"] = data
    _SHELL_CACHE["ts"] = now
    return data


def _read_cpu_ram() -> Dict[str, Any]:
    out = {
        "cpu": {"util_pct": 0.0, "logical": os.cpu_count() or 0},
        "ram": {"used_gb": 0.0, "total_gb": 0.0, "pct": 0.0},
    }
    try:
        import psutil  # type: ignore
    except Exception:
        return out
    try:
        # non-blocking sample; first call returns 0.0 — that's fine, we
        # poll every second from the frontend so subsequent calls are real.
        out["cpu"]["util_pct"] = float(psutil.cpu_percent(interval=None))
        out["cpu"]["logical"] = int(psutil.cpu_count(logical=True) or 0)
    except Exception:
        pass
    try:
        vm = psutil.virtual_memory()
        out["ram"]["used_gb"] = round((vm.total - vm.available) / 1073741824, 3)
        out["ram"]["total_gb"] = round(vm.total / 1073741824, 3)
        out["ram"]["pct"] = round(float(vm.percent), 1)
    except Exception:
        pass
    return out


def collect_metrics() -> Dict[str, Any]:
    now = time.time()
    if _CACHE["data"] is not None and (now - _CACHE["ts"]) < _CACHE_TTL_S:
        d = dict(_CACHE["data"])
        d["cached"] = True
        return d
    gpu = _gpu_pynvml() or _gpu_nvidia_smi()
    cpu_ram = _read_cpu_ram()
    data: Dict[str, Any] = {
        "ok": True,
        "ts": now,
        "gpu": gpu or {"available": False, "_source": "none"},
        "cpu": cpu_ram["cpu"],
        "ram": cpu_ram["ram"],
        "source": (gpu or {}).get("_source", "none"),
    }
    _CACHE["data"] = data
    _CACHE["ts"] = now
    return data


# ---------------------------------------------------------------------------
# Aiohttp route registration
# ---------------------------------------------------------------------------
_ROUTES_REGISTERED = False


def register_routes(server) -> None:
    global _ROUTES_REGISTERED
    if _ROUTES_REGISTERED or server is None:
        return
    try:
        from aiohttp import web
    except Exception:
        return
    routes = server.routes if hasattr(server, "routes") else server.app.router

    @routes.get("/c2c/sys/metrics")
    async def _metrics(_req):
        try:
            data = collect_metrics()
            return web.json_response(data)
        except Exception as e:
            return web.json_response({"ok": False, "error": str(e)}, status=200)

    _ROUTES_REGISTERED = True
