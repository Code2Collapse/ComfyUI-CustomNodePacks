"""Read a C2C setting from any pack, without CustomNodePacks' registry (L3.02, design: docs/research/
L3_settings_secrets_audit.md section S9). Canonical copy in ComfyUI-CustomNodePacks/nodes/; the other packs ship a
byte-identical copy (tools/sync_vendored.py group py-saved-setting) because each pack must work alone.

    get(setting_id, default, env=None, parse=None)

= the environment variable ``env`` when it is set (its old meaning: ``parse`` turns the raw string into a value; a
value that does not parse is ignored), else the user's saved value in ``user/default/comfy.settings.json`` (re-read
only when the file changes), else ``default``. Never raises: a missing ComfyUI, a broken file or a wrong type gives
the default.
"""
from __future__ import annotations

import json
import os
import threading
from typing import Any, Callable, Optional

_CACHE: dict = {"path": None, "mtime": None, "data": {}}
_LOCK = threading.Lock()


def _settings_path() -> Optional[str]:
    try:
        import folder_paths  # type: ignore
        base = folder_paths.get_user_directory()
    except Exception:  # noqa: BLE001 - outside ComfyUI
        return None
    return os.path.join(base, "default", "comfy.settings.json") if base else None


def _saved() -> dict:
    path = _settings_path()
    if not path:
        return {}
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return {}
    with _LOCK:
        if _CACHE["path"] == path and _CACHE["mtime"] == mtime:
            return _CACHE["data"]
        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
            data = data if isinstance(data, dict) else {}
        except Exception:  # noqa: BLE001 - a broken file means nothing saved
            data = {}
        _CACHE.update(path=path, mtime=mtime, data=data)
        return data


def get(setting_id: str, default: Any, env: Optional[str] = None,
        parse: Optional[Callable[[str], Any]] = None) -> Any:
    if env:
        raw = os.environ.get(env)
        if raw is not None and raw != "":
            try:
                return parse(raw) if parse else raw
            except Exception:  # noqa: BLE001 - an unparsable variable is ignored, as the old reads did
                pass
    val = _saved().get(setting_id, None)
    if val is None:
        return default
    if isinstance(default, bool):
        return val if isinstance(val, bool) else default
    if isinstance(default, (int, float)) and not isinstance(default, bool):
        try:
            return type(default)(val)
        except (TypeError, ValueError):
            return default
    return val if isinstance(val, type(default)) or default is None else default
