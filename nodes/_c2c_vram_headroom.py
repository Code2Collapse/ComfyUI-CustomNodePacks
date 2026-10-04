"""Browser VRAM headroom — reserve GPU memory for the browser/desktop.

ComfyUI's model loader calls ``free_memory`` on every load; dynamic-VRAM mode
uses ``comfy_aimdo.control`` headroom, classic mode uses
``comfy.model_management.EXTRA_RESERVED_VRAM`` (read live by
``extra_reserved_memory()``). This route lets the front-end ask ComfyUI to keep
extra VRAM free so WebGL views (pose/gaze/3D) are less likely to lose their GPU
context.

Never raises — plain-English JSON errors only.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

log = logging.getLogger("c2c.vram_headroom")

_CORE_DEFAULT_BYTES: Optional[int] = None
_MODE: Optional[str] = None
_ROUTES_REGISTERED = False


def _detect_mode() -> str:
    global _MODE
    if _MODE is not None:
        return _MODE
    try:
        import comfy_aimdo.control as aimdo_ctrl
        aimdo_ctrl.get_simple_vram_headroom()
        _MODE = "dynamic"
        return _MODE
    except Exception:
        pass
    try:
        import comfy.model_management as mm
        dev = getattr(mm, "get_torch_device", lambda: None)()
        if dev is not None and str(getattr(dev, "type", dev)).lower() == "cpu":
            _MODE = "cpu"
            return _MODE
        if hasattr(mm, "EXTRA_RESERVED_VRAM") or hasattr(mm, "extra_reserved_memory"):
            _MODE = "classic"
            return _MODE
    except Exception:
        pass
    _MODE = "unknown"
    return _MODE


def _ensure_core_default_bytes() -> int:
    """Capture ComfyUI's starting headroom once — before any apply or mutation."""
    global _CORE_DEFAULT_BYTES
    if _CORE_DEFAULT_BYTES is not None:
        return _CORE_DEFAULT_BYTES
    mode = _detect_mode()
    default = 400 * 1024 * 1024
    try:
        if mode == "dynamic":
            import comfy_aimdo.control as aimdo_ctrl
            default = int(aimdo_ctrl.get_simple_vram_headroom())
        else:
            import comfy.model_management as mm
            erm = getattr(mm, "extra_reserved_memory", None)
            if callable(erm):
                default = int(erm())
            else:
                default = int(getattr(mm, "EXTRA_RESERVED_VRAM", default))
    except Exception as exc:
        log.debug("[c2c.vram_headroom] core default fallback: %s", exc)
    _CORE_DEFAULT_BYTES = max(0, default)
    return _CORE_DEFAULT_BYTES


def _applied_bytes() -> int:
    mode = _detect_mode()
    try:
        if mode == "dynamic":
            import comfy_aimdo.control as aimdo_ctrl
            return int(aimdo_ctrl.get_simple_vram_headroom())
        import comfy.model_management as mm
        erm = getattr(mm, "extra_reserved_memory", None)
        if callable(erm):
            return int(erm())
        return int(getattr(mm, "EXTRA_RESERVED_VRAM", _ensure_core_default_bytes()))
    except Exception:
        return _ensure_core_default_bytes()


def _vram_total_gb() -> Optional[float]:
    # On a CPU-only server core's get_total_memory() reports system RAM, which
    # is not VRAM; say "no GPU" instead of a misleading number.
    if _detect_mode() == "cpu":
        return 0.0
    try:
        import comfy.model_management as mm
        gtm = getattr(mm, "get_total_memory", None)
        dev = getattr(mm, "get_torch_device", lambda: None)()
        if callable(gtm) and dev is not None:
            total = gtm(dev)
            if isinstance(total, tuple):
                total = total[0]
            return round(float(total) / (1024 ** 3), 2)
        tv = getattr(mm, "total_vram", None)
        if tv:
            return round(float(tv) / 1024.0, 2)
    except Exception:
        pass
    return None


def build_status() -> Dict[str, Any]:
    try:
        core = _ensure_core_default_bytes()
        applied = _applied_bytes()
        return {
            "ok": True,
            "mode": _detect_mode(),
            "core_default_gb": round(core / (1024 ** 3), 3),
            "applied_gb": round(applied / (1024 ** 3), 3),
            "vram_total_gb": _vram_total_gb(),
        }
    except Exception as exc:
        return {"ok": False, "error": f"Could not read VRAM headroom status: {exc}"}


def apply_browser_headroom_gb(gb: float) -> Dict[str, Any]:
    try:
        gb = float(gb)
    except (TypeError, ValueError):
        return {"ok": False, "error": "gb must be a number between 0 and 4."}
    if gb < 0 or gb > 4:
        return {"ok": False, "error": "gb must be a number between 0 and 4."}

    core = _ensure_core_default_bytes()
    if gb == 0:
        target = core
    else:
        target = int(gb * (1024 ** 3))
    applied = max(core, target)

    try:
        mode = _detect_mode()
        if mode == "dynamic":
            import comfy_aimdo.control as aimdo_ctrl
            aimdo_ctrl.set_simple_vram_headroom(applied)
        else:
            import comfy.model_management as mm
            mm.EXTRA_RESERVED_VRAM = applied
    except Exception as exc:
        return {
            "ok": False,
            "error": f"Could not apply browser VRAM headroom: {exc}",
        }

    out = build_status()
    out["ok"] = True
    return out


def register_routes() -> bool:
    global _ROUTES_REGISTERED
    if _ROUTES_REGISTERED:
        return True
    try:
        from server import PromptServer
        from aiohttp import web
    except Exception as exc:
        log.debug("[c2c.vram_headroom] route registration skipped: %s", exc)
        return False
    inst = getattr(PromptServer, "instance", None)
    routes = getattr(inst, "routes", None) if inst else None
    if routes is None or not hasattr(routes, "get"):
        return False

    @routes.get("/c2c/memory/browser_headroom")
    async def _get_browser_headroom(_request):  # noqa: ANN001
        try:
            return web.json_response(build_status())
        except Exception as exc:  # noqa: BLE001
            return web.json_response(
                {"ok": False, "error": f"Could not read VRAM headroom: {exc}"},
                status=200,
            )

    @routes.post("/c2c/memory/browser_headroom")
    async def _post_browser_headroom(request):  # noqa: ANN001
        try:
            try:
                data = await request.json()
            except Exception:
                data = {}
            gb = data.get("gb", 0)
            result = apply_browser_headroom_gb(gb)
            return web.json_response(result, status=200)
        except Exception as exc:  # noqa: BLE001
            return web.json_response(
                {"ok": False, "error": f"Could not apply VRAM headroom: {exc}"},
                status=200,
            )

    _ROUTES_REGISTERED = True
    log.info("[c2c.vram_headroom] registered GET/POST /c2c/memory/browser_headroom")
    return True
