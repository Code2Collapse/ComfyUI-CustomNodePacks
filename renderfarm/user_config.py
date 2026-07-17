"""RIB user/project + backend configuration.

`config/users.json`  — maps the RIB_USER_NAME env var to a role (admin/user),
                       project tags, and a per-user concurrency cap.
`config/backends.json` — the registered compute backends + compute profiles.

Fail-loud policy: a submit without RIB_USER_NAME set, an unknown user, an
unknown backend, or a disabled backend raises a RuntimeError that says
exactly what to fix — nothing is silently defaulted.
"""

from __future__ import annotations

import json
import os
import threading

CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config")
USERS_PATH = os.path.join(CONFIG_DIR, "users.json")
BACKENDS_PATH = os.path.join(CONFIG_DIR, "backends.json")

_cache_lock = threading.Lock()
_cache: dict = {}  # path -> (mtime, data)


def _load_json(path: str) -> dict:
    if not os.path.exists(path):
        raise RuntimeError(
            f"RIB: config file missing: {path}. It ships with the pack — restore it "
            f"from git or recreate it (see renderfarm/README section in docs)."
        )
    mtime = os.path.getmtime(path)
    with _cache_lock:
        hit = _cache.get(path)
        if hit and hit[0] == mtime:
            return hit[1]
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    with _cache_lock:
        _cache[path] = (mtime, data)
    return data


# ── users ────────────────────────────────────────────────────────────
def current_user_name() -> str:
    name = os.environ.get("RIB_USER_NAME", "").strip()
    if not name:
        raise RuntimeError(
            "RIB: the RIB_USER_NAME environment variable is not set. Set it to your "
            "farm user name (must exist in renderfarm/config/users.json) and restart "
            "ComfyUI. Example (PowerShell): $env:RIB_USER_NAME = 'varig'"
        )
    return name


def get_user(name: str | None = None) -> dict:
    name = name or current_user_name()
    users = _load_json(USERS_PATH).get("users", {})
    if name not in users:
        raise RuntimeError(
            f"RIB: user '{name}' is not declared in renderfarm/config/users.json. "
            f"Known users: {sorted(users)}. Add an entry with a role (admin/user), "
            f"projects, and max_concurrent_jobs."
        )
    u = dict(users[name])
    u.setdefault("role", "user")
    u.setdefault("projects", [])
    u.setdefault("max_concurrent_jobs", 2)
    u["name"] = name
    return u


def is_admin(name: str | None = None) -> bool:
    try:
        return get_user(name)["role"] == "admin"
    except RuntimeError:
        return False


def can_control(actor: str, job_owner: str) -> bool:
    """Admins can pause/cancel anyone's jobs; users only their own."""
    return actor == job_owner or is_admin(actor)


# ── backends / compute profiles ──────────────────────────────────────
def list_backends(enabled_only: bool = True) -> list[dict]:
    backends = _load_json(BACKENDS_PATH).get("backends", [])
    return [b for b in backends if (b.get("enabled", False) or not enabled_only)]


def get_backend(name: str) -> dict:
    for b in list_backends(enabled_only=False):
        if b.get("name") == name:
            if not b.get("enabled", False):
                raise RuntimeError(
                    f"RIB: backend '{name}' exists but is disabled in "
                    f"renderfarm/config/backends.json — set \"enabled\": true once the "
                    f"container/gateway is reachable."
                )
            return b
    raise RuntimeError(
        f"RIB: backend '{name}' is not registered in renderfarm/config/backends.json. "
        f"Registered: {[b.get('name') for b in list_backends(enabled_only=False)]}"
    )


def compute_profiles() -> dict:
    return _load_json(BACKENDS_PATH).get("compute_profiles", {})


def validate_profile(backend: dict, profile: str):
    allowed = backend.get("compute_profiles", [])
    if allowed and profile not in allowed:
        raise RuntimeError(
            f"RIB: backend '{backend.get('name')}' does not offer compute profile "
            f"'{profile}'. Offered: {allowed}. Pick one the gateway can route."
        )
