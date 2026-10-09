"""GPU spill detection (L7.59, owner 2026-10-09: "detect and tell me").

On Windows the NVIDIA driver's "CUDA - Sysmem Fallback Policy" (default since driver 536.40) lets allocations run past
the card's memory into shared system RAM instead of raising out-of-memory. Nothing fails, everything crawls, and
ComfyUI's own out-of-memory fallbacks (tiled VAE, offload) never start. Measured on an RTX 4060 Laptop 8 GB: a 2K Wan
VAE decode took 502 s untiled (12.4 GB "allocated") and 37 s tiled (docs/evidence/L7.59).

PyTorch's peak allocation can only pass the physical size when that spill happens, so: GET /c2c/gpu/peak returns the
peak since the last reset and the card's size, and resets when asked. The front end resets at the start of a run and
asks at the end (js/c2c_gpu_spill.js).
"""
from __future__ import annotations

import logging

log = logging.getLogger("C2C.gpu_spill")
_ROUTES = False


def peak_report(reset: bool = False) -> dict:
    try:
        import torch
    except Exception:
        return {"available": False}
    if not torch.cuda.is_available():
        return {"available": False}
    dev = torch.cuda.current_device()
    total = torch.cuda.get_device_properties(dev).total_memory
    peak = torch.cuda.max_memory_allocated(dev)
    out = {"available": True, "device": torch.cuda.get_device_name(dev), "total_gb": round(total / 2**30, 2),
           "peak_gb": round(peak / 2**30, 2), "spilled": peak > total}
    if reset:
        torch.cuda.reset_peak_memory_stats(dev)
    return out


def register_routes() -> bool:
    global _ROUTES
    if _ROUTES:
        return True
    try:
        from aiohttp import web
        from server import PromptServer
    except Exception as exc:
        log.debug("gpu spill route skipped: %s", exc)
        return False
    routes = getattr(getattr(PromptServer, "instance", None), "routes", None)
    if routes is None:
        return False

    async def _peak(request):
        reset = request.rel_url.query.get("reset", "0") in ("1", "true", "yes")
        try:
            return web.json_response(peak_report(reset))
        except Exception as exc:  # noqa: BLE001
            return web.json_response({"available": False, "error": str(exc)[:200]})

    routes.get("/c2c/gpu/peak")(_peak)
    _ROUTES = True
    return True
