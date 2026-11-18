"""C2C Control Center report (L3.04, slice S7 of docs/research/L3_settings_secrets_audit.md).

``GET /c2c/control/report`` (this machine only) answers, as JSON:

* versions: ComfyUI, the frontend core requires, Python, torch, CUDA;
* devices: every torch device core reports (name, VRAM total / free) and system RAM - the same calls core's
  ``/system_stats`` makes (verified against core 0.39 server.py);
* packs: every loaded C2C pack - folder, version (its pyproject.toml), nodes it registered, and COLLISIONS: ids the
  pack declares whose registered class came from another module (core lets a later pack overwrite an id silently,
  ``nodes.py`` NODE_CLASS_MAPPINGS[name] = node_cls; the winner's ``RELATIVE_PYTHON_MODULE`` names it);
* failures: the C2C failure registry's records (already secret-redacted, L3 S6);
* settings: the C2C settings registry's effective values and env locks (L3.01).

The whole answer goes through the shared secret redactor once more. Cheap: pyproject versions are read once and cached;
no disk walk per request (performance rule).
"""
from __future__ import annotations

import json
import logging
import os
import platform
import sys
from typing import Any, Dict, List

LOG = logging.getLogger("C2C.ControlCenter")
C2C_PACKS = ("ComfyUI-CustomNodePacks", "ComfyUI-WanNodeExperiments", "ComfyUI-NukeMaxNodes", "ComfyUI-MiniMaxSuite",
             "ComfyUI-WanAnimatePreprocessV2", "ComfyUI-LTX-Enhanced", "ComfyUI-OmniScale", "ComfyUI-GLM_Image",
             "ComfyUI-WanAnimalPreprocessor")
_VERSION_CACHE: Dict[str, str] = {}


def _pyproject_version(folder: str) -> str:
    if folder in _VERSION_CACHE:
        return _VERSION_CACHE[folder]
    ver = ""
    try:
        with open(os.path.join(folder, "pyproject.toml"), encoding="utf-8") as fh:
            for line in fh:
                s = line.strip()
                if s.startswith("version") and "=" in s:
                    ver = s.split("=", 1)[1].strip().strip("\"'")
                    break
    except OSError:
        pass
    _VERSION_CACHE[folder] = ver
    return ver


def _versions() -> Dict[str, Any]:
    out: Dict[str, Any] = {"python": platform.python_version(), "os": f"{platform.system()} {platform.release()}"}
    try:
        import comfyui_version  # type: ignore
        out["comfyui"] = comfyui_version.__version__
    except Exception:  # noqa: BLE001
        out["comfyui"] = None
    try:
        from app.frontend_management import FrontendManager  # type: ignore
        out["frontend_required"] = FrontendManager.get_required_frontend_version()
    except Exception:  # noqa: BLE001
        out["frontend_required"] = None
    try:
        import torch
        out["torch"] = torch.__version__
        out["cuda"] = torch.version.cuda
    except Exception:  # noqa: BLE001
        out["torch"] = out["cuda"] = None
    return out


def _devices() -> Dict[str, Any]:
    try:
        import comfy.model_management as mm  # type: ignore
    except Exception:  # noqa: BLE001
        return {"error": "ComfyUI core not loaded"}
    out: Dict[str, Any] = {"devices": []}
    try:
        cpu = mm.torch.device("cpu")
        out["ram_total"] = int(mm.get_total_memory(cpu))
        out["ram_free"] = int(mm.get_free_memory(cpu))
        primary = mm.get_torch_device()
        devs = list(mm.get_all_torch_devices()) if hasattr(mm, "get_all_torch_devices") else [primary]
        if primary not in devs:
            devs = [primary] + devs
        for d in devs:
            total = mm.get_total_memory(d)
            free = mm.get_free_memory(d)
            out["devices"].append({"name": mm.get_torch_device_name(d), "type": d.type,
                                   "vram_total": int(total), "vram_free": int(free)})
    except Exception as exc:  # noqa: BLE001 - a device query must not break the report
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def pack_table(loaded_dirs: Dict[str, str], mappings: Dict[str, Any], pack_modules: Dict[str, Any]) -> List[dict]:
    """Per C2C pack: folder, version, registered node count and collisions. Pure (tests pass fakes).

    loaded_dirs: core's LOADED_MODULE_DIRS (module name -> folder); mappings: core's NODE_CLASS_MAPPINGS (the
    winners); pack_modules: module name -> the pack's own module (for its declared NODE_CLASS_MAPPINGS)."""
    rows = []
    for name, folder in sorted(loaded_dirs.items()):
        if name not in C2C_PACKS:
            continue
        mod_prefix = f"custom_nodes.{name}"
        own = [k for k, cls in mappings.items()
               if str(getattr(cls, "RELATIVE_PYTHON_MODULE", "")).startswith(mod_prefix)]
        declared = list(getattr(pack_modules.get(name), "NODE_CLASS_MAPPINGS", {}) or {})
        collisions = []
        for node_id in declared:
            winner = mappings.get(node_id)
            wmod = str(getattr(winner, "RELATIVE_PYTHON_MODULE", "")) if winner is not None else ""
            if winner is not None and not wmod.startswith(mod_prefix):
                collisions.append({"id": node_id, "won_by": wmod.split(".")[1] if wmod.count(".") >= 1 else wmod})
        rows.append({"pack": name, "folder": folder, "version": _pyproject_version(folder),
                     "nodes": len(own), "declared": len(declared), "collisions": collisions})
    return rows


def report() -> Dict[str, Any]:
    out: Dict[str, Any] = {"versions": _versions(), "memory": _devices()}
    try:
        import nodes as core_nodes  # type: ignore
        mods = {n: sys.modules.get(f"custom_nodes.{n}") for n in C2C_PACKS}
        out["packs"] = pack_table(dict(core_nodes.LOADED_MODULE_DIRS), dict(core_nodes.NODE_CLASS_MAPPINGS), mods)
    except Exception as exc:  # noqa: BLE001
        out["packs"] = {"error": f"{type(exc).__name__}: {exc}"}
    try:
        from . import _c2c_registry as REG
        out["failures"] = REG.summary()
    except Exception as exc:  # noqa: BLE001
        out["failures"] = {"error": f"{type(exc).__name__}: {exc}"}
    try:
        from . import _c2c_settings as S
        out["settings"] = S.effective()
    except Exception as exc:  # noqa: BLE001
        out["settings"] = {"error": f"{type(exc).__name__}: {exc}"}
    try:  # one more pass over everything (a key inside a failure text, a path holding a token, ...)
        from ._c2c_redact import redact
        out = json.loads(redact(json.dumps(out, default=str)))
    except Exception:  # noqa: BLE001
        pass
    return out


def register_routes(server) -> None:
    from aiohttp import web

    @server.routes.get("/c2c/control/report")
    async def _report(request):  # noqa: ANN001
        try:
            from ._local_ui_guard import refuse_remote
            denied = refuse_remote(request, "C2C Control Center")
            if denied is not None:
                return denied
        except Exception:  # noqa: BLE001 - no guard, no report
            return web.json_response({"error": "unavailable"}, status=503)
        return web.json_response(report())
