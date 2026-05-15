"""WanShotPickerMEC + WanShotCountMEC."""
from __future__ import annotations

from ._common import DEFAULT_SHOT, log


class WanShotPickerMEC:
    """
    Select one shot from a SHOTLIST by index. Fans the shot out into
    individual primitive widgets (prompt, neg, length, seed, w/h, cfg, steps)
    suitable for feeding a standard KSampler subgraph.

    When ``index`` exceeds the list length, the LAST shot is returned (with a
    warning) so partial graphs still execute cleanly during authoring.
    """
    CATEGORY = "C2C/wan_director"
    FUNCTION = "pick"
    RETURN_TYPES = ("STRING", "STRING", "INT", "INT", "INT", "INT", "FLOAT", "INT")
    RETURN_NAMES = ("prompt", "negative_prompt", "length",
                    "seed", "width", "height", "cfg", "steps")
    DESCRIPTION = (
        "Pick one shot from a SHOTLIST by zero-based index. Outputs all "
        "primitive parameters (prompt, neg, length, seed, w, h, cfg, steps). "
        "Out-of-range indices clamp to the last shot."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "shotlist": ("SHOTLIST",),
                "index":    ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
            },
        }

    def pick(self, shotlist, index: int):
        if not isinstance(shotlist, list) or not shotlist:
            log.warning("[WanShotPicker] empty shotlist; emitting defaults")
            s = dict(DEFAULT_SHOT)
        else:
            i = int(index)
            if i >= len(shotlist):
                log.warning(
                    "[WanShotPicker] index %d >= len %d, clamping",
                    i, len(shotlist),
                )
                i = len(shotlist) - 1
            elif i < 0:
                i = 0
            s = shotlist[i]
        return (
            s.get("prompt", ""),
            s.get("negative_prompt", ""),
            int(s.get("length", DEFAULT_SHOT["length"])),
            int(s.get("seed", 0)),
            int(s.get("width", DEFAULT_SHOT["width"])),
            int(s.get("height", DEFAULT_SHOT["height"])),
            float(s.get("cfg", DEFAULT_SHOT["cfg"])),
            int(s.get("steps", DEFAULT_SHOT["steps"])),
        )


class WanShotCountMEC:
    """Trivial helper: SHOTLIST -> length. Useful for batch loops."""
    CATEGORY = "C2C/wan_director"
    FUNCTION = "count"
    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("num_shots",)
    DESCRIPTION = "Return the number of shots in a SHOTLIST."

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"shotlist": ("SHOTLIST",)}}

    def count(self, shotlist):
        n = len(shotlist) if isinstance(shotlist, list) else 0
        return (int(n),)
