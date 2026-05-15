"""WanPromptScheduleMEC — emit a frame-indexed prompt schedule string."""
from __future__ import annotations

from ._common import log


class WanPromptScheduleMEC:
    """
    Convert a SHOTLIST into a frame-indexed prompt schedule string,
    compatible with the FizzNodes / batch prompt schedule conventions:

        "0": "first prompt",
        "16": "second prompt",
        ...

    Two output styles are supported:
      - ``fizz``   : the JSON-ish dict style above (used by Fizz BatchPromptSchedule)
      - ``simple`` : ``"<frame>: <prompt>"`` lines (one per shot)

    The total length is also returned so downstream samplers know how
    many frames to generate.
    """
    CATEGORY = "C2C/wan_director"
    FUNCTION = "build"
    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("schedule", "total_frames")
    DESCRIPTION = (
        "Build a frame-indexed prompt schedule from a SHOTLIST. Styles: "
        "fizz (BatchPromptSchedule format) or simple (one line per shot). "
        "Also returns total frame count."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "shotlist": ("SHOTLIST",),
                "style":    (["fizz", "simple"], {"default": "fizz"}),
                "use_negative": ("BOOLEAN", {"default": False,
                    "tooltip": "If true, emit negative_prompt instead of prompt."}),
            },
        }

    def build(self, shotlist, style: str, use_negative: bool):
        if not isinstance(shotlist, list):
            log.warning("[WanPromptSchedule] non-list SHOTLIST; emitting empty")
            return ("", 0)
        cursor = 0
        lines: list[str] = []
        for shot in shotlist:
            text = shot.get("negative_prompt" if use_negative else "prompt") or ""
            # Escape double-quotes for fizz style.
            safe = text.replace('"', '\\"')
            if style == "fizz":
                lines.append(f'"{cursor}": "{safe}"')
            else:
                lines.append(f"{cursor}: {safe}")
            cursor += int(shot.get("length", 0) or 0)
        if style == "fizz":
            sched = ",\n".join(lines)
        else:
            sched = "\n".join(lines)
        log.info("[WanPromptSchedule] %d shots -> total_frames=%d", len(shotlist), cursor)
        return (sched, int(cursor))
