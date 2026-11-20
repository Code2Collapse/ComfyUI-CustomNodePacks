"""Node Identity admin policy (design docs/research/node_identity_design.md, section 4).

The admin is the ComfyUI page on THIS machine (nodes/_local_ui_guard.is_local - the same owner check as the Control
Center report, stored secrets and the AI routes). The policy lives in ``$C2C_HOME/node_identity_policy.json``
(default ``~/.c2c``), NOT in comfy.settings.json: core's ``/settings`` POST and ``/userdata`` writes are not
local-only, so a ``--listen`` client could change anything stored there.

Pure functions; routes.py only adds the HTTP layer.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
from pathlib import Path
from typing import Any, Dict, Optional

LOG = logging.getLogger("C2C.NodeIdentity")
FILENAME = "node_identity_policy.json"
SETTING_HIDE = "c2c.nodeIdentity.hideTitles"
SETTING_HOVER = "c2c.nodeIdentity.hoverReveal"

DEFAULTS: Dict[str, Any] = {
    "enabled": True,              # off: nobody renames, titles are always shown
    "template": "{stage} · {function}",
    "separator": " · ",
    "use_prompt_words": True,     # prompt words may serve as the detail of a prompt node's name
    "keep_manual": True,          # names a person typed are never replaced by the engine
    "excluded_types": [],         # node types the engine never renames
    "others_may_rename": False,   # clients other than the admin may rename / restore / edit names
    "others_may_toggle": True,    # clients other than the admin may hide / show titles
    "locked": False,              # names are read-only for everyone but the admin
    "ai_naming": False,           # optional AI namer hook (front end); deterministic rules stay the fallback
}
_BOOLS = ("enabled", "use_prompt_words", "keep_manual", "others_may_rename", "others_may_toggle", "locked",
          "ai_naming")
ACTIONS = ("rename", "restore", "edit", "toggle", "configure")
_LOCK = threading.Lock()


class PolicyError(ValueError):
    """A policy value the admin sent is not valid (the message says which and why)."""


def policy_path() -> Path:
    base = os.environ.get("C2C_HOME") or str(Path.home() / ".c2c")
    return Path(base) / FILENAME


def _bool(key: str, value: Any) -> bool:
    if isinstance(value, bool):
        return value
    raise PolicyError(f"{key} must be true or false")


def validate(data: Any) -> Dict[str, Any]:
    """A complete, valid policy from ``data`` (missing keys take the defaults, unknown keys are dropped).
    Raises PolicyError for a value of the wrong kind."""
    if not isinstance(data, dict):
        raise PolicyError("the policy must be a JSON object")
    out = dict(DEFAULTS)
    out["excluded_types"] = list(DEFAULTS["excluded_types"])
    for key in _BOOLS:
        if key in data:
            out[key] = _bool(key, data[key])
    if "template" in data:
        t = data["template"]
        if not isinstance(t, str) or not t.strip() or len(t) > 120:
            raise PolicyError("template must be text of 1-120 characters")
        if "{stage}" not in t and "{function}" not in t:
            raise PolicyError("template must contain {stage} or {function}")
        out["template"] = t
    if "separator" in data:
        s = data["separator"]
        if not isinstance(s, str) or not (1 <= len(s) <= 5):
            raise PolicyError("separator must be 1-5 characters")
        out["separator"] = s
    if "excluded_types" in data:
        ex = data["excluded_types"]
        if not isinstance(ex, list) or len(ex) > 200 or not all(isinstance(x, str) and 0 < len(x) <= 200 for x in ex):
            raise PolicyError("excluded_types must be a list of at most 200 node type names")
        out["excluded_types"] = sorted(set(ex))
    return out


def load_policy(path: Optional[Path] = None) -> Dict[str, Any]:
    """The saved policy, or the defaults (missing or broken file: defaults, said once in the log)."""
    p = Path(path) if path else policy_path()
    try:
        with open(p, encoding="utf-8") as fh:
            return validate(json.load(fh))
    except FileNotFoundError:
        return validate({})
    except (OSError, ValueError) as exc:
        LOG.warning("Node Identity: could not read %s (%s); using the default policy", p, exc)
        return validate({})


def save_policy(data: Any, path: Optional[Path] = None) -> Dict[str, Any]:
    """Validate and write atomically (temporary file + replace). Returns what was written."""
    pol = validate(data)
    p = Path(path) if path else policy_path()
    with _LOCK:
        p.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=".node_identity_", suffix=".tmp", dir=str(p.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(pol, fh, indent=1, ensure_ascii=False)
            os.replace(tmp, p)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
    return pol


def permissions(pol: Dict[str, Any], is_admin: bool) -> Dict[str, bool]:
    """What this client may do. The admin may always configure (to turn the feature back on, or unlock)."""
    if not pol.get("enabled", True):
        return {"rename": False, "toggle": False, "configure": bool(is_admin)}
    if is_admin:
        return {"rename": True, "toggle": True, "configure": True}
    return {"rename": bool(pol.get("others_may_rename")) and not pol.get("locked"),
            "toggle": bool(pol.get("others_may_toggle")),
            "configure": False}


def authorize(pol: Dict[str, Any], is_admin: bool, action: str):
    """(allowed, reason) for one write action of the page. restore / edit follow rename."""
    if action not in ACTIONS:
        return False, f"unknown action {action!r}"
    may = permissions(pol, is_admin)
    key = "rename" if action in ("rename", "restore", "edit") else action
    if may[key]:
        return True, ""
    if not pol.get("enabled", True) and action != "configure":
        return False, "Node Identity is turned off by the admin."
    if action == "configure":
        return False, "Only the admin (the ComfyUI page on this machine) can change the Node Identity rules."
    if key == "rename" and pol.get("locked"):
        return False, "Node names are locked by the admin."
    if key == "rename":
        return False, "The admin has not allowed other clients to rename nodes."
    return False, "The admin has not allowed other clients to hide or show titles."


def effective_state(is_admin: bool, user: str = "default") -> Dict[str, Any]:
    """Policy + permissions + this user's two title preferences (the settings c2c.nodeIdentity.*), with the policy
    applied: feature off -> titles are shown."""
    pol = load_policy()
    prefs = {"hideTitles": False, "hoverReveal": True}
    try:
        from .. import _c2c_settings as S
        prefs = {"hideTitles": bool(S.get(SETTING_HIDE, user)), "hoverReveal": bool(S.get(SETTING_HOVER, user))}
    except Exception as exc:  # noqa: BLE001 - outside ComfyUI the declared defaults apply
        LOG.debug("Node Identity: settings unavailable (%s)", exc)
    if not pol.get("enabled", True):
        prefs["hideTitles"] = False
    return {"policy": pol, "is_admin": bool(is_admin), "may": permissions(pol, is_admin), "prefs": prefs}
