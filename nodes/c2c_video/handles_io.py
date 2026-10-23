"""Built-in AV handles for Load Video (C2C) and Save Video (C2C) (L9.40, docs/save-video-v2.md).

Load adds them: the count comes from AV Handles' own planner (fit / add, the model frame grids, the record it keeps),
run on a stand-in batch of the right length, and the frames are added LAZILY as repeated edge-frame indices of the
video handle - so IMAGE, mask and the VAE (LATENT) path all get them without decoding the clip first. Audio moves by
the same duration. The count travels in video_info["c2c_handles"].

Save trims them: auto = exactly what video_info says (fit's target length wins over the count, as in AV Handles);
manual = a count and a side; off = nothing. auto never reads AV Handles' remembered count: an implicit trim must not
act on a record from an unrelated earlier run.
"""

from __future__ import annotations

import os
from typing import Any

import torch

try:
    from .. import av_handles as AV
    from .._frame_grid import CUSTOM, GRID_NAMES, resolve, trim_to
except ImportError:  # tests: the pack root on sys.path, nodes imported as a package
    from nodes import av_handles as AV  # type: ignore
    from nodes._frame_grid import CUSTOM, GRID_NAMES, resolve, trim_to  # type: ignore

KILL_SWITCH = "C2C_VIDEO_HANDLES"
HANDLE_MODES = ("off", "fit", "add")
FROM_FORMAT = "from format"
GRID_CHOICES = (FROM_FORMAT,) + tuple(GRID_NAMES)
TRIM_MODES = ("auto", "manual", "off")
SIDES = AV.SIDES
INFO_KEY = "c2c_handles"


def disabled() -> bool:
    return os.environ.get(KILL_SWITCH, "").strip().lower() in ("0", "off", "false", "no")


def grid_for(handle_grid: str, format_preset: dict, format_name: str, step: int, plus: int) -> tuple[str, int, int, str]:
    """(AV Handles model name, custom step, custom plus, label). 'from format' uses Load Video's format preset grid
    (one grid, not two); a format without a frame rule means no grid."""
    if handle_grid == FROM_FORMAT:
        frames = (format_preset or {}).get("frames")
        if frames:
            s, p = int(frames[0]), int(frames[1])
            return CUSTOM, s, p, f"{format_name} ({s}n+{p}), from format"
        return "none (any frame count)", 1, 0, "no grid (format has no frame rule)"
    if handle_grid not in GRID_NAMES:
        raise AV.AVHandlesError(f"Unknown handle grid {handle_grid!r}; pick one of {', '.join(GRID_CHOICES)}.")
    g = resolve(handle_grid, int(step), int(plus))
    return handle_grid, int(step), int(plus), (g.note if handle_grid == CUSTOM else handle_grid)


def add_on_load(loaded, audio, *, mode: str, frames: int, target: int, side: str, model: str, step: int, plus: int,
                fps: float, memo_key: str = "") -> tuple[Any, Any, dict, str]:
    """(handle with the handles, audio moved to match, record, report). mode = fit | add."""
    if mode not in ("fit", "add"):
        raise AV.AVHandlesError(f"handles must be off, fit or add; got {mode!r}.")
    n = len(loaded)
    proxy = torch.zeros(n, 1, 1, 1)                      # AV Handles plans from the length only
    _img, out_aud, total, rec, info = AV.AVHandlesMEC().run(
        mode=mode, handle_frames=int(frames), target_frames=int(target), model=model, custom_step=int(step),
        custom_plus=int(plus), images=proxy, audio=audio, side=side,
        padding_mode=AV.USE_MODEL if mode == "add" else "disabled", manual_fps=float(fps), memo_key=memo_key)
    head, tail = int(rec["head"]), int(rec["tail"])
    sel = list(loaded.selected_indices())
    if head or tail:
        from .handle import TransformOp
        loaded = loaded._selected([sel[0]] * head + sel + [sel[-1]] * tail, TransformOp("handles", (head, tail)))
    assert len(loaded) == total == n + head + tail
    rec = dict(rec, fps=float(fps))
    return loaded, out_aud, rec, info


class FrameWindow:
    """frames [start, start + count) of a tensor batch or a FrameStream, read front to back like the encoders do."""

    def __init__(self, frames, start: int, count: int):
        self._f, self._start = frames, int(start)
        n, h, w, c = frames.shape
        self.shape = (int(count), int(h), int(w), int(c))
        self.ndim = 4
        self.dtype = getattr(frames, "dtype", torch.float32)

    def __getitem__(self, key):
        if isinstance(key, slice):
            a = 0 if key.start is None else int(key.start)
            b = self.shape[0] if key.stop is None else min(int(key.stop), self.shape[0])
            return self._f[self._start + a:self._start + b]
        if isinstance(key, int):
            return self._f[self._start + key]
        raise TypeError("frames are read by index or slice")

    def __getattr__(self, name):                       # last_frame, delivered, close, _transform ...
        return getattr(self._f, name)


def window(frames, start: int, count: int):
    if start == 0 and count == frames.shape[0]:
        return frames
    if torch.is_tensor(frames):
        return frames[start:start + count]
    return FrameWindow(frames, start, count)


def trim_count(n: int, video_info: Any, mode: str, frames: int, side: str) -> tuple[int, int, str]:
    """(head, tail, where it came from) for Save Video's trim."""
    if mode == "off" or disabled():
        return 0, 0, "trim off" if mode == "off" else f"handles switched off ({KILL_SWITCH}=0)"
    if mode == "manual":
        k = max(0, int(frames))
        head, tail = (k, k) if side == "both" else AV.split_ends(k, side)
        return head, tail, f"manual: {k} frame(s) on {'each end' if side == 'both' else 'the ' + side}"
    rec = video_info.get(INFO_KEY) if isinstance(video_info, dict) else None
    if not isinstance(rec, dict):
        return 0, 0, "no handles in video_info - nothing trimmed (wire Load Video's video_info, or trim manually)"
    head, tail = int(rec.get("head", 0)), int(rec.get("tail", 0))
    tgt = int(rec.get("target") or 0)
    if tgt:
        excess = trim_to(n, tgt)
        if excess != head + tail:
            head, tail = AV.split_ends(excess, str(rec.get("side") or "head"))
        return head, tail, f"auto: fit's target of {tgt} frames ({n} arrived)"
    return head, tail, f"auto: {head} head + {tail} tail frame(s) added by Load Video"


def trim_on_save(images, audio, video_info, *, mode: str, frames: int, side: str, fps: float):
    """(frames, audio, report). Refuses a trim that would leave nothing, with the numbers."""
    n = int(images.shape[0])
    head, tail, origin = trim_count(n, video_info, mode, frames, side)
    if head + tail >= n and head + tail > 0:
        raise AV.AVHandlesError(f"Trimming {head} + {tail} frames off a {n}-frame clip would leave nothing; "
                                f"the handles ({origin}) do not belong to these frames.")
    out = window(images, head, n - head - tail)
    out_aud = AV.shift_audio_ends(audio, head, tail, fps, add=False) if audio is not None else None
    return out, out_aud, {"head": head, "tail": tail, "from": origin, "frames_in": n, "frames_out": n - head - tail}


def export_range(images, audio, range_in: int, range_out: int, fps: float):
    """Frames [range_in, range_out] (inclusive, of the trimmed clip; 0 / 0 = all), clamped and reported."""
    n = int(images.shape[0])
    a, b = int(range_in or 0), int(range_out or 0)
    if a <= 0 and b <= 0:
        return images, audio, None, (0, n - 1)
    a = max(0, min(a, n - 1))
    b = n - 1 if b <= 0 else max(a, min(b, n - 1))
    note = f"frames {a}-{b} of {n}" + (" (clamped)" if (int(range_in) != a or (int(range_out) and int(range_out) != b))
                                        else "")
    out = window(images, a, b - a + 1)
    if audio is not None and fps > 0:
        head, tail = a, n - 1 - b
        audio = AV.shift_audio_ends(audio, head, tail, fps, add=False)
    return out, audio, note, (a, b)
