"""The viewer's selection (`picks`) and what it does to a Mask + Matting run (L7.114, owner A14).

The viewer on the node (js/c2c_viewer/) writes one JSON string into MaskOps' appended `picks` input. Empty = the node
runs exactly as before (tests/test_matte_picks.py). Set = it replaces the prompt inputs of the chosen backend:

    subject point + add points        -> positive points
    excluded objects + cut points     -> negative points
    the subject's box (+ the adds)    -> the box prompt
    frame                             -> the frame the prompts sit on (frame_annotation)

Schema version 1, coordinates normalised to the plate (0..1, origin top left):

    {"v": 1, "frame": 12, "text": "person", "model": "sam3.1/sam3.1_multiplex_fp16.safetensors",
     "subject": {"x": 0.41, "y": 0.52, "box": [0.30, 0.20, 0.55, 0.90], "label": "person"},
     "excluded": [{"x": 0.72, "y": 0.50, "label": "person"}],
     "add": [{"x": 0.47, "y": 0.61}], "cut": [{"x": 0.40, "y": 0.30}],
     "plate": {"frames": 120, "w": 1920, "h": 1080},
     "range": {"start": 12, "count": 17}}

`range` is only written into the viewer's own helper prompts (the preview chunk); the saved widget never has it.
`text` is what the viewer's Find used; the run prompts with the points and the box, not the text (text plus points
would ask SAM for every match again). Every error is a plain sentence that says what to do.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

VERSION = 1
#: Margin added around the subject's box (share of the box size) so the box never crops the edge SAM has to find.
BOX_MARGIN = 0.04
#: The plate's aspect may change this much (resize, crop by a pixel) before the selection is called stale.
ASPECT_TOLERANCE = 0.02

_REPICK = "Open the viewer on the Mask + Matting node and pick again, or clear the selection."


def viewer_enabled() -> bool:
    """Kill switch C2C_VIEWER=0: the node shows its plain widgets (picks as a text box). Execution is the same
    either way - a saved selection still applies, so switching the viewer off never changes a result silently."""
    return os.environ.get("C2C_VIEWER", "").strip() != "0"


@dataclass
class Marker:
    x: float
    y: float
    box: Optional[Tuple[float, float, float, float]] = None
    label: str = ""


@dataclass
class Picks:
    frame: int
    subject: Marker
    excluded: List[Marker] = field(default_factory=list)
    add: List[Marker] = field(default_factory=list)
    cut: List[Marker] = field(default_factory=list)
    text: str = ""
    model: str = ""
    plate: Optional[Dict[str, int]] = None
    range_start: Optional[int] = None
    range_count: Optional[int] = None
    raw: str = ""

    @property
    def has_range(self) -> bool:
        return self.range_start is not None


@dataclass
class Prompts:
    """The selection in plate pixels, for one plate size."""
    positive: List[Tuple[float, float]]
    negative: List[Tuple[float, float]]
    box: Optional[List[int]]
    frame: int

    def positive_json(self) -> str:
        return json.dumps([[round(x, 2), round(y, 2)] for x, y in self.positive])

    def negative_json(self) -> str:
        return json.dumps([[round(x, 2), round(y, 2)] for x, y in self.negative]) if self.negative else ""

    def points_json(self) -> str:
        """The [{x, y, label}] list SAM + ViTMatte and SeC + MatAnyone2 read."""
        pts = [{"x": round(x, 2), "y": round(y, 2), "label": 1} for x, y in self.positive]
        pts += [{"x": round(x, 2), "y": round(y, 2), "label": 0} for x, y in self.negative]
        return json.dumps(pts)

    def bbox_json(self) -> str:
        return json.dumps(self.box) if self.box else ""


def _num(v: Any, what: str) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        raise ValueError(f"The viewer selection has a {what} that is not a number ({v!r}). {_REPICK}") from None
    if f != f:  # NaN
        raise ValueError(f"The viewer selection has a {what} that is not a number. {_REPICK}")
    return f


def _unit(v: Any, what: str) -> float:
    f = _num(v, what)
    if f < -1e-6 or f > 1 + 1e-6:
        raise ValueError(f"The viewer selection has a {what} outside the picture ({f:.3f}; 0..1 expected). {_REPICK}")
    return min(1.0, max(0.0, f))


def _marker(raw: Any, what: str) -> Marker:
    if not isinstance(raw, dict):
        raise ValueError(f"The viewer selection has a {what} that is not an object. {_REPICK}")
    m = Marker(x=_unit(raw.get("x"), f"{what} x"), y=_unit(raw.get("y"), f"{what} y"),
               label=str(raw.get("label") or ""))
    box = raw.get("box")
    if box is not None:
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            raise ValueError(f"The viewer selection has a {what} box that is not [x0, y0, x1, y1]. {_REPICK}")
        x0, y0, x1, y1 = (_unit(b, f"{what} box") for b in box)
        if x1 > x0 and y1 > y0:
            m.box = (x0, y0, x1, y1)
    return m


def _markers(raw: Any, what: str) -> List[Marker]:
    if raw in (None, ""):
        return []
    if not isinstance(raw, list):
        raise ValueError(f"The viewer selection's {what} list is not a list. {_REPICK}")
    return [_marker(r, what) for r in raw]


def parse(text: Any) -> Optional[Picks]:
    """The selection, or None when the widget is empty (the node then runs as it always did)."""
    if text is None:
        return None
    s = str(text).strip()
    if not s:
        return None
    try:
        data = json.loads(s)
    except json.JSONDecodeError:
        raise ValueError(f"The viewer selection (picks) is not valid JSON. {_REPICK}") from None
    if not isinstance(data, dict):
        raise ValueError(f"The viewer selection (picks) is not a JSON object. {_REPICK}")
    # nothing clicked yet (the viewer keeps the Find text before the first click): no selection, not an error
    if not any(data.get(k) for k in ("subject", "add", "excluded", "cut")):
        return None
    v = int(_num(data.get("v", VERSION), "version"))
    if v != VERSION:
        raise ValueError(f"The viewer selection was written by a newer version of this node (format {v}, this "
                         f"node reads {VERSION}). Update ComfyUI-CustomNodePacks, or pick again.")
    if data.get("subject") in (None, {}):
        raise ValueError(f"The viewer selection has no subject. Click the subject in the viewer first. {_REPICK}")
    frame = int(_num(data.get("frame", 0), "frame"))
    if frame < 0:
        raise ValueError(f"The viewer selection points at frame {frame}. {_REPICK}")
    plate = data.get("plate") if isinstance(data.get("plate"), dict) else None
    p = Picks(frame=frame, subject=_marker(data["subject"], "subject"),
              excluded=_markers(data.get("excluded"), "excluded object"),
              add=_markers(data.get("add"), "added part"), cut=_markers(data.get("cut"), "cut part"),
              text=str(data.get("text") or ""), model=str(data.get("model") or ""),
              plate={k: int(_num(plate[k], f"plate {k}")) for k in ("frames", "w", "h") if k in plate}
              if plate else None, raw=s)
    rng = data.get("range")
    if isinstance(rng, dict):
        p.range_start = max(0, int(_num(rng.get("start", frame), "range start")))
        p.range_count = max(1, int(_num(rng.get("count", 1), "range count")))
    return p


def check_plate(p: Picks, frames: int, width: int, height: int) -> List[str]:
    """Refuse a selection that cannot belong to this plate; return notes for the info output."""
    notes: List[str] = []
    if p.frame >= frames:
        raise ValueError(f"The selection was picked on frame {p.frame}, but the plate has {frames} frame"
                         f"{'s' if frames != 1 else ''}. {_REPICK}")
    if p.plate and p.plate.get("w") and p.plate.get("h"):
        a0 = p.plate["w"] / p.plate["h"]
        a1 = width / max(height, 1)
        if abs(a1 - a0) / a0 > ASPECT_TOLERANCE:
            raise ValueError(f"The plate changed shape since the selection was made ({p.plate['w']}x{p.plate['h']} "
                             f"then, {width}x{height} now). {_REPICK}")
        if (p.plate["w"], p.plate["h"]) != (width, height):
            notes.append(f"picked on a {p.plate['w']}x{p.plate['h']} plate, run on {width}x{height} "
                         "(points are relative, so they still land on the same spot)")
    if p.plate and p.plate.get("frames") and p.plate["frames"] != frames and not p.has_range:
        notes.append(f"picked on a {p.plate['frames']}-frame plate, run on {frames} frames")
    return notes


def frame_range(p: Picks, frames: int) -> Tuple[int, int]:
    """[start, stop) of the frames this run renders (the whole plate unless the helper asked for a chunk)."""
    if not p.has_range:
        return 0, frames
    start = min(p.range_start, frames - 1)
    stop = min(frames, start + int(p.range_count or 1))
    if not (start <= p.frame < stop):
        raise ValueError(f"The preview starts at frame {start} but the subject was picked on frame {p.frame}. "
                         "Preview from the pick frame, or pick on a frame inside the preview.")
    return start, stop


def slice_frames(t: Any, start: int, stop: int, frames: int):
    """A per-frame input cut to the rendered range; a one-frame (or other-length) input is left as it is."""
    try:
        import torch
    except Exception:  # pragma: no cover
        return t
    if not isinstance(t, torch.Tensor) or (start, stop) == (0, frames):
        return t
    if t.ndim >= 3 and int(t.shape[0]) == frames:
        return t[start:stop]
    return t


def prompts(p: Picks, width: int, height: int, frame_offset: int = 0) -> Prompts:
    """The selection in pixels of a ``width`` x ``height`` plate."""
    W, H = float(width), float(height)

    def px(m: Marker) -> Tuple[float, float]:
        # pixel centres: 0..W-1, never on the far edge
        return (min(W - 1.0, m.x * W), min(H - 1.0, m.y * H))

    pos = [px(p.subject)] + [px(m) for m in p.add]
    neg = [px(m) for m in p.excluded] + [px(m) for m in p.cut]
    box = None
    if p.subject.box is not None:
        x0, y0, x1, y1 = p.subject.box
        for m in p.add:                           # an added part outside the subject's box widens it
            x0, y0, x1, y1 = min(x0, m.x), min(y0, m.y), max(x1, m.x), max(y1, m.y)
        mx, my = (x1 - x0) * BOX_MARGIN, (y1 - y0) * BOX_MARGIN
        x0, y0, x1, y1 = max(0.0, x0 - mx), max(0.0, y0 - my), min(1.0, x1 + mx), min(1.0, y1 + my)
        box = [int(round(x0 * W)), int(round(y0 * H)), int(round(x1 * W)), int(round(y1 * H))]
        if box[2] <= box[0] + 1 or box[3] <= box[1] + 1:
            box = None
    return Prompts(positive=pos, negative=neg, box=box, frame=max(0, p.frame - frame_offset))


def summary(p: Picks) -> Dict[str, Any]:
    """What the info output records about the selection (counts, never the coordinates' meaning)."""
    return {"frame": p.frame, "subject": p.subject.label or "point", "excluded": len(p.excluded),
            "add": len(p.add), "cut": len(p.cut), "text": p.text, "model": p.model or "",
            "box": p.subject.box is not None}
