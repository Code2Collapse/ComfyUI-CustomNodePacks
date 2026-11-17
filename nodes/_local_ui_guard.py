"""Local-UI guard for routes that read or change secrets (L9.287, L3 audit P1).

refuse_remote(request) -> None when the request comes from the ComfyUI page on THIS machine (loopback socket peer +
loopback Host + no cross-site Origin / Sec-Fetch-Site), else a 403 JSON response to return. Through an SSH tunnel the
page still counts as local (the tunnel's end is loopback). The check is nodes/magnific/security.is_local_ui_origin,
loaded BY FILE: importing the magnific package would register its 15 nodes and /magnific/* routes (CNP-INV
2026-11-14). The peer is read from the socket, never from request.remote (a forwarded-header middleware could
rewrite that).
"""
from __future__ import annotations

from pathlib import Path

_SECURITY = None


def _security():
    global _SECURITY
    if _SECURITY is None:
        import importlib.util
        here = Path(__file__).resolve().parent / "magnific" / "security.py"
        spec = importlib.util.spec_from_file_location("_c2c_magnific_security_guard", here)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _SECURITY = mod
    return _SECURITY


def peer_ip(request):
    t = getattr(request, "transport", None)
    pn = t.get_extra_info("peername") if t else None
    return pn[0] if isinstance(pn, (tuple, list)) and pn else request.remote


def is_local(request) -> bool:
    try:
        return bool(_security().is_local_ui_origin(request.headers, peer_ip(request)))
    except Exception:  # noqa: BLE001 - a broken guard refuses (fail closed)
        return False


def refuse_remote(request, what: str = "This"):
    if is_local(request):
        return None
    from aiohttp import web
    return web.json_response(
        {"error": "LOCAL_UI_ONLY",
         "message": f"{what} can be used only from the ComfyUI page on this machine (http://127.0.0.1 or localhost)."},
        status=403)
