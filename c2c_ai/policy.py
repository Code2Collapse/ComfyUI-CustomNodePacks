"""Per-feature routing policy.

Features declare a default policy at registration time; users override per-
feature in ``Settings → C2C → AI Backends``. Overrides are persisted to
``~/.c2c/policy.json``.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

from .types import Policy

log = logging.getLogger("c2c_ai.policy")


def _config_dir() -> Path:
    base = os.environ.get("C2C_HOME") or str(Path.home() / ".c2c")
    p = Path(base)
    p.mkdir(parents=True, exist_ok=True)
    return p


_POLICY_FILE = _config_dir() / "policy.json"


# Default policy per registered feature. Features are free to register their
# own defaults at import time via :func:`register_default`.
_DEFAULTS: dict[str, Policy] = {
    # P1 features
    "node_explainer":   Policy.AUTO,           # cheap, both work
    "error_translator": Policy.PREFER_LOCAL,   # may contain tracebacks with paths
    "prompt_wizard":    Policy.AUTO,
    "workflow_doctor":  Policy.PREFER_LOCAL,   # graph JSON can include local file widget values
    "macro_ai":         Policy.PREFER_CLOUD,   # benefits from stronger reasoning
    "workflow_translator": Policy.PREFER_CLOUD,
    "tensor_inspector": Policy.LOCAL_ONLY,     # numeric data, no benefit from cloud
}


def register_default(feature: str, policy: Policy) -> None:
    """Features call this at import time so the catalog stays accurate."""
    _DEFAULTS.setdefault(feature, policy)


# ---------- user overrides ---------------------------------------------------

def _load_overrides() -> dict[str, Policy]:
    if not _POLICY_FILE.is_file():
        return {}
    try:
        with open(_POLICY_FILE, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
    except Exception as exc:
        log.warning("policy load failed: %s", exc)
        return {}
    out: dict[str, Policy] = {}
    for k, v in raw.items():
        try:
            out[k] = Policy(v)
        except ValueError:
            log.warning("dropping unknown policy %r for feature %r", v, k)
    return out


def _save_overrides(overrides: dict[str, Policy]) -> None:
    with open(_POLICY_FILE, "w", encoding="utf-8") as fh:
        json.dump({k: v.value for k, v in overrides.items()}, fh, indent=2)


_OVERRIDES: dict[str, Policy] = _load_overrides()


def resolve(feature: str, explicit: Policy | None = None) -> Policy:
    """Return the effective policy: explicit > user override > registered default > AUTO."""
    if explicit is not None:
        return explicit
    if feature in _OVERRIDES:
        return _OVERRIDES[feature]
    return _DEFAULTS.get(feature, Policy.AUTO)


def set_override(feature: str, policy: Policy | None) -> None:
    """``policy=None`` clears the override (falls back to default)."""
    global _OVERRIDES
    if policy is None:
        _OVERRIDES.pop(feature, None)
    else:
        _OVERRIDES[feature] = policy
    _save_overrides(_OVERRIDES)


def all_known_features() -> list[str]:
    return sorted(set(_DEFAULTS.keys()) | set(_OVERRIDES.keys()))


@dataclass(frozen=True)
class PolicyEntry:
    feature: str
    default: Policy
    override: Policy | None
    effective: Policy


def listing() -> list[PolicyEntry]:
    out: list[PolicyEntry] = []
    for f in all_known_features():
        default = _DEFAULTS.get(f, Policy.AUTO)
        override = _OVERRIDES.get(f)
        out.append(PolicyEntry(
            feature=f,
            default=default,
            override=override,
            effective=override or default,
        ))
    return out
