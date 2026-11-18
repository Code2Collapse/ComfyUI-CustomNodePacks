"""C2C settings registry (L3.01, slice S3 of docs/research/L3_settings_secrets_audit.md).

ONE typed declaration per setting a backend reads: id, type, default, limits / options, tooltip, panel group, pack,
the environment variable that overrides it (env-locked), the code that consumes it. ``get(id)`` resolves

    environment (when the setting names one, parsed to its type)  ->  the user's saved value (comfy.settings.json,
    validated: a bad value falls back to the default with one warning)  ->  the declared default

``GET /c2c/settings/schema`` publishes the declarations (no values, never a secret) for js/_c2c_settings_registry.js,
which registers every id the page has not registered yet (fe 1.53.10: extension ``init`` hooks are awaited before the
graph loads; verified in the audit, section 0). ``GET /c2c/settings/effective`` (this machine only) returns the
resolved values and which ones are env-locked. Setting ids are persistence keys: never rename one.

Secrets are NOT settings (S4, the secret hub): a declaration with secret=True is refused.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

LOG = logging.getLogger("C2C.Settings")
TYPES = ("boolean", "number", "slider", "combo", "text")


@dataclass(frozen=True)
class Setting:
    id: str
    type: str
    default: Any
    name: str
    group: Sequence[str]                      # panel path under "C2C", e.g. ("Video", "Colour Management")
    pack: str = "ComfyUI-CustomNodePacks"
    tooltip: str = ""
    options: Sequence[str] = ()
    min: Optional[float] = None
    max: Optional[float] = None
    step: Optional[float] = None
    env: Optional[str] = None                 # environment variable that overrides (locks) it
    restart: bool = False                     # takes effect after a restart only
    consumer: str = ""                        # "module:function" that reads it (lint: L3.05)
    secret: bool = False


_REG: Dict[str, Setting] = {}
_LOCK = threading.Lock()
_SAVED: Dict[str, Any] = {"path": None, "mtime": None, "data": {}}
_WARNED: set = set()


def register(*settings: Setting) -> None:
    """Declare settings. Re-declaring an id with the same declaration is a no-op; a different one is an error."""
    for s in settings:
        if s.secret:
            raise ValueError(f"{s.id}: secrets are not settings (use the secret hub)")
        if s.type not in TYPES:
            raise ValueError(f"{s.id}: unknown type {s.type!r}")
        if s.type == "combo" and s.default not in s.options:
            raise ValueError(f"{s.id}: default {s.default!r} is not one of its options")
        with _LOCK:
            old = _REG.get(s.id)
            if old is not None and old != s:
                raise ValueError(f"{s.id}: declared twice with different values")
            _REG[s.id] = s


def declared() -> List[Setting]:
    with _LOCK:
        return list(_REG.values())


def _settings_path(user: str = "default") -> Optional[str]:
    try:
        import folder_paths  # type: ignore
        base = folder_paths.get_user_directory()
    except Exception:  # noqa: BLE001 - outside ComfyUI
        return None
    return os.path.join(base, user, "comfy.settings.json") if base else None


def _saved(user: str = "default") -> Dict[str, Any]:
    """The user's comfy.settings.json, re-read only when its modification time changes."""
    path = _settings_path(user)
    if not path or not os.path.isfile(path):
        return {}
    try:
        mtime = os.path.getmtime(path)
        if _SAVED["path"] == path and _SAVED["mtime"] == mtime:
            return _SAVED["data"]
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        data = data if isinstance(data, dict) else {}
        _SAVED.update(path=path, mtime=mtime, data=data)
        return data
    except Exception as exc:  # noqa: BLE001 - a broken file means "nothing saved", said once
        if path not in _WARNED:
            _WARNED.add(path)
            LOG.warning("C2C settings: could not read %s (%s); using defaults", path, exc)
        return {}


def _coerce(s: Setting, value: Any) -> Any:
    """``value`` as the setting's type, or raise ValueError."""
    if s.type == "boolean":
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.strip().lower() in ("1", "true", "yes", "on", "0", "false", "no", "off"):
            return value.strip().lower() in ("1", "true", "yes", "on")
        raise ValueError("not a boolean")
    if s.type in ("number", "slider"):
        if isinstance(value, bool):
            raise ValueError("not a number")
        v = float(value)
        if v != v:
            raise ValueError("not a number")
        if (s.min is not None and v < s.min) or (s.max is not None and v > s.max):
            raise ValueError(f"outside [{s.min}, {s.max}]")
        return int(v) if isinstance(s.default, int) and v.is_integer() else v
    if s.type == "combo":
        v = str(value)
        if v not in s.options:
            raise ValueError(f"not one of {list(s.options)}")
        return v
    return "" if value is None else str(value)


def is_env_locked(setting_id: str) -> bool:
    s = _REG.get(setting_id)
    return bool(s and s.env and os.environ.get(s.env, "") != "")


def get(setting_id: str, user: str = "default") -> Any:
    """Environment -> saved -> default (see the module doc). An undeclared id raises KeyError."""
    s = _REG[setting_id]
    if s.env:
        raw = os.environ.get(s.env, "")
        if raw != "":
            try:
                return _coerce(s, raw)
            except ValueError as exc:
                _warn_once(f"env:{s.id}", "C2C settings: %s=%r is not valid for %s (%s); ignored", s.env, raw, s.id, exc)
    data = _saved(user)
    if s.id in data:
        try:
            return _coerce(s, data[s.id])
        except ValueError as exc:
            _warn_once(f"saved:{s.id}", "C2C settings: saved %s=%r is not valid (%s); using the default %r",
                       s.id, data[s.id], exc, s.default)
    return s.default


def _warn_once(key: str, msg: str, *args) -> None:
    if key not in _WARNED:
        _WARNED.add(key)
        LOG.warning(msg, *args)


def schema() -> List[Dict[str, Any]]:
    """The front end's view of every declaration (no values)."""
    out = []
    for s in sorted(declared(), key=lambda x: x.id):
        d = {"id": s.id, "name": s.name, "type": s.type, "defaultValue": s.default,
             "category": ["C2C", *s.group], "tooltip": s.tooltip, "pack": s.pack}
        if s.options:
            d["options"] = list(s.options)
        attrs = {k: v for k, v in (("min", s.min), ("max", s.max), ("step", s.step)) if v is not None}
        if attrs:
            d["attrs"] = attrs
        if s.env:
            d["env"] = s.env
        if s.restart:
            d["restart"] = True
        out.append(d)
    return out


def effective(user: str = "default") -> List[Dict[str, Any]]:
    return [{"id": s.id, "value": get(s.id, user), "env_locked": is_env_locked(s.id), "env": s.env}
            for s in sorted(declared(), key=lambda x: x.id)]


def register_routes(server) -> None:
    from aiohttp import web

    routes = server.routes

    @routes.get("/c2c/settings/schema")
    async def _schema(_request):  # noqa: ANN001 - declarations only, no values, no secrets
        return web.json_response({"settings": schema()})

    @routes.get("/c2c/settings/effective")
    async def _effective(request):  # noqa: ANN001
        try:
            from ._local_ui_guard import refuse_remote
            denied = refuse_remote(request, "C2C settings")
            if denied is not None:
                return denied
        except Exception:  # noqa: BLE001 - no guard, no values
            return web.json_response({"error": "unavailable"}, status=503)
        user = request.headers.get("comfy-user") or "default"
        user = user if user.replace("-", "").replace("_", "").isalnum() else "default"
        return web.json_response({"settings": effective(user)})


# ── declarations (slice 1: the settings a backend already reads; each mirrors its JS declaration exactly) ──────────
register(
    Setting("c2c.saveVideo.naming", "combo", "Folder Version Incrementer", "Save Video file naming",
            ("Video", "Save"), options=("Folder Version Incrementer", "ComfyUI counter"),
            consumer="nodes.c2c_video.save_encode:effective_naming"),
    Setting("c2c.video.cm.ocioConfig", "text", "", "OCIO config file (used when $OCIO is not set)",
            ("Video", "Colour Management"), consumer="nodes.c2c_video.save_ocio:_setting"),
    Setting("c2c.video.cm.builtinConfig", "combo", "studio (latest)", "Built-in OCIO config",
            ("Video", "Colour Management"),
            options=("studio (latest)", "cg (latest)", "studio ACES 1.3", "cg ACES 1.3"),
            consumer="nodes.c2c_video.save_ocio:_setting"),
    Setting("c2c.video.cm.processing", "combo", "Auto", "Processing", ("Video", "Colour Management"),
            options=("Auto", "Exact CPU", "OCIO GPU"), consumer="nodes.c2c_video.save_ocio:_setting"),
    Setting("c2c.video.cm.view", "text", "", "Display view (empty = the config's default)",
            ("Video", "Colour Management"), consumer="nodes.c2c_video.save_ocio:_setting"),
)

# slice 2: backend environment tunables that had no UI (audit section 3b, rec S); the variable still wins (env-locked)
register(
    Setting("c2c.models.neverDownload", "boolean", False, "Never download models",
            ("Models & Downloads",), env="C2C_NO_DOWNLOADS",
            tooltip="On: no C2C node downloads model weights, even with its own 'Allow Download' ticked; nodes use "
                    "installed copies or their labelled model-free method and say which model is missing. Downloads "
                    "you start yourself (Model Browser) are not affected.",
            consumer="nodes._download_policy:never_download"),
    Setting("c2c.video.uploadMaxMB", "number", 20000, "Largest upload (MB)", ("Video", "Upload"),
            min=0, max=1000000, step=1, env="C2C_UPLOAD_MAX_MB",   # min 0: the variable always took any size
            tooltip="Biggest single file Load Video (C2C) / Load Images accept from the upload button.",
            consumer="nodes.c2c_video.media_upload:max_bytes"),
)

# slice 3 (audit 3b rec S/R): more environment-only switches get a setting; each variable still wins
register(
    Setting("c2c.insight.enabled", "boolean", True, "Per-node time and memory (Insight)", ("Diagnostics",),
            env="C2C_INSIGHT", restart=True,
            tooltip="Measures each node's run time and memory and shows it under the node. Off: the executor hook "
                    "is not installed at all.", consumer="nodes.insight:install"),
    Setting("c2c.matte.viewer", "boolean", True, "Matte viewer", ("Masks",), env="C2C_VIEWER",
            tooltip="The interactive viewer on Mask + Matting and the SAM nodes. Off: their plain widgets (picks as "
                    "a text box). A saved selection still applies, so results never change.",
            consumer="nodes.mask_matting._picks:viewer_enabled"),
    Setting("c2c.models.fallbacks", "boolean", True, "Model-free fallbacks", ("Models & Downloads",),
            env="C2C_MODEL_FALLBACK",
            tooltip="When a mask / matting model is missing, run the labelled model-free method instead of stopping "
                    "with an error.", consumer="nodes.mask_matting._classic:fallback_allowed"),
    Setting("c2c.video.saveFfmpeg", "boolean", True, "Encode with FFmpeg", ("Video", "Save"), env="C2C_SAVE_FFMPEG",
            tooltip="Save Video (C2C) uses an installed FFmpeg for the codecs it supports (NVENC, ProRes, ...). Off: "
                    "always the built-in PyAV encoder.", consumer="nodes.c2c_video.save_ffmpeg:ffmpeg_for"),
    Setting("c2c.vault.ttlMinutes", "number", 480, "Unlock lasts (minutes)", ("Vault",), max=100000, step=1,   # no min: 0 or less always meant no expiry
            env="C2C_VAULT_TTL_MINUTES", restart=True,
            tooltip="How long an unlocked Vault stays unlocked. 0 = until ComfyUI restarts.",
            consumer="nodes.vault_node:_ttl_seconds"),
)
