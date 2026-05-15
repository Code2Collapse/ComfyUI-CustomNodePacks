"""Shared helpers for Wan Director nodes."""
from __future__ import annotations

import json
import logging
from typing import Any

log = logging.getLogger("MEC.WanDirector")

# ── Default shot template ────────────────────────────────────────────
DEFAULT_SHOT: dict[str, Any] = {
    "prompt": "",
    "negative_prompt": "",
    "length": 81,           # Wan native frames-per-shot default
    "seed": 0,
    "width": 832,
    "height": 480,
    "cfg": 5.0,
    "steps": 20,
}

DEFAULT_SHOTLIST_JSON = json.dumps(
    [
        {**DEFAULT_SHOT, "prompt": "A wide establishing shot of a sunlit valley"},
        {**DEFAULT_SHOT, "prompt": "Camera pushes in toward a lone figure"},
        {**DEFAULT_SHOT, "prompt": "Close-up on the figure's face, calm expression"},
    ],
    indent=2,
)


def _coerce_int(v: Any, default: int) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _coerce_float(v: Any, default: float) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def normalize_shot(raw: Any, idx: int = 0) -> dict[str, Any]:
    """Coerce a raw shot dict (from user JSON) to a fully-typed shot dict.

    Unknown keys are preserved verbatim so downstream nodes can extend.
    """
    if not isinstance(raw, dict):
        raise ValueError(f"shot #{idx} is not an object: {type(raw).__name__}")
    out = dict(DEFAULT_SHOT)
    out.update(raw)
    out["prompt"]          = str(out.get("prompt") or "").strip()
    out["negative_prompt"] = str(out.get("negative_prompt") or "").strip()
    out["length"]          = max(1, _coerce_int(out.get("length"), DEFAULT_SHOT["length"]))
    out["seed"]            = _coerce_int(out.get("seed"), 0)
    out["width"]           = max(64, _coerce_int(out.get("width"),  DEFAULT_SHOT["width"]))
    out["height"]          = max(64, _coerce_int(out.get("height"), DEFAULT_SHOT["height"]))
    out["cfg"]             = _coerce_float(out.get("cfg"),   DEFAULT_SHOT["cfg"])
    out["steps"]           = max(1, _coerce_int(out.get("steps"),  DEFAULT_SHOT["steps"]))
    return out


def parse_shotlist(text: str) -> list[dict[str, Any]]:
    """Parse the JSON text widget into a normalized list of shots.

    Raises ValueError on malformed input.
    """
    s = (text or "").strip()
    if not s:
        return []
    try:
        data = json.loads(s)
    except json.JSONDecodeError as e:
        raise ValueError(f"shotlist JSON parse error: {e}") from e
    if not isinstance(data, list):
        raise ValueError("shotlist must be a JSON array of shot objects")
    return [normalize_shot(shot, i) for i, shot in enumerate(data)]
