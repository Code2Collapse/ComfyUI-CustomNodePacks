"""WanShotListMEC — JSON shot list source node."""
from __future__ import annotations

import json
from ._common import DEFAULT_SHOTLIST_JSON, parse_shotlist, log


class WanShotListMEC:
    """
    Define a list of shots as a JSON array. Each entry is an object with:
        prompt, negative_prompt, length, seed, width, height, cfg, steps
    Outputs a SHOTLIST (Python list of dicts) plus a normalized
    pretty-printed JSON for debugging.
    """
    CATEGORY = "C2C/wan_director"
    FUNCTION = "build"
    RETURN_TYPES = ("SHOTLIST", "STRING", "INT")
    RETURN_NAMES = ("shotlist", "normalized_json", "num_shots")
    DESCRIPTION = (
        "Define a multi-shot script as a JSON array. Output SHOTLIST socket "
        "feeds Shot Picker, Prompt Schedule, etc. Each shot: prompt, "
        "negative_prompt, length, seed, width, height, cfg, steps."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "shots_json": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": DEFAULT_SHOTLIST_JSON,
                        "tooltip": "JSON array of shot objects. See node description.",
                    },
                ),
            },
        }

    def build(self, shots_json: str):
        try:
            shots = parse_shotlist(shots_json)
        except ValueError as e:
            log.warning("[WanShotList] %s", e)
            raise
        normalized = json.dumps(shots, indent=2)
        log.info("[WanShotList] parsed %d shots", len(shots))
        return (shots, normalized, len(shots))
