"""C2C Farm backend adapter registry/factory.

Adapters are cached per backend name (they hold websocket taps + object_info
caches). backends.json `type` selects the class; comfyui_native covers every
backend that is a full ComfyUI instance (AKS Docker, RunPod, vast.ai, LAN).
"""

from __future__ import annotations

import threading

from .base_adapter import BackendAdapter  # noqa: F401 (re-export)

_ADAPTERS: dict[str, "BackendAdapter"] = {}
_lock = threading.Lock()

_TYPES = {
    "comfyui_native": "comfyui_native_adapter.ComfyUINativeAdapter",
}


def get_adapter(backend_name: str) -> BackendAdapter:
    with _lock:
        if backend_name in _ADAPTERS:
            return _ADAPTERS[backend_name]
    from ..user_config import get_backend
    cfg = get_backend(backend_name)
    btype = cfg.get("type", "comfyui_native")
    if btype == "comfyui_native":
        from .comfyui_native_adapter import ComfyUINativeAdapter
        adapter = ComfyUINativeAdapter(cfg)
    else:
        raise RuntimeError(
            f"C2C Farm: backend '{backend_name}' has unknown type '{btype}'. "
            f"Supported types: {sorted(_TYPES)}."
        )
    with _lock:
        _ADAPTERS[backend_name] = adapter
    return adapter
