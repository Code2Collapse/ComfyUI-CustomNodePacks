"""AV Handles — add stabilisation frames, then trim exactly those back off.

PORTED AND COMBINED FROM: ComfyUI-AV-Handles (MIT, per its README). That pack
ships two nodes, Add and Trim, and the trim has its own `handle_frames` widget
you must retype to match what the add used. Two numbers that have to agree,
kept in two places, edited at two different times — so they drift, and the
symptom is a clip that is a few frames long or short with nothing to say why.

THIS IS ONE NODE, and in trim mode it REMEMBERS.

    mode = add    handle_frames = 12        ->  12 frames on the head
    mode = trim   trim_amount   = auto      ->  12 frames off the head

The trim finds its number in this order, and the report always says which one
it used, because "it trimmed the wrong amount" is otherwise unanswerable:

  1. trim_amount = manual   -> handle_frames, an explicit instruction
  2. a `handles` signal wired from the add node  -> exact, and it crosses
     workflows where the memory cannot
  3. what the add node remembered                -> the automatic path
  4. nothing: it trims 0 and SAYS so, rather than guessing

`trim_amount` is a separate control rather than "a non-zero number means you
typed an override", because handle_frames defaults to 8 — a sensible number
to ADD — and that default is indistinguishable from a deliberate 8. Reading it
as an override meant switching the mode to trim and touching nothing else
trimmed 8 instead of the remembered count: the node's whole promise, defeated
by its own default.

WHY A MEMORY AND NOT JUST A WIRE. Add usually sits at the top of a graph and
trim at the bottom, often far apart, and people build the trim end days later.
A wire is the better answer when it exists; the memory is what makes the node
work when it does not. It survives across runs on purpose — add in one queue,
generate, trim in the next — so it is module-level rather than run-scoped, and
the report distinguishes "added earlier in this same run" from "remembered
from a previous run", because the second one is worth a second look.

WHAT IT REFUSES. Trimming more frames than exist returns an empty batch, which
downstream reads as "no video" and reports nothing useful. That is refused with
the numbers in the message instead.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

import torch

try:
    from ._frame_grid import (
        CUSTOM, GRID_NAMES, FrameGridError, Grid, describe_fit, plan_fit,
        resolve, trim_to,
    )
except ImportError:  # loaded as a top-level module rather than a package
    from _frame_grid import (  # type: ignore
        CUSTOM, GRID_NAMES, FrameGridError, Grid, describe_fit, plan_fit,
        resolve, trim_to,
    )

# Kept for the legacy padding_mode widget. The real table now lives in
# _frame_grid.py, shared with anything else that has to know what a model
# will accept — two tables describing the same models is how they drift.
GRIDS = {
    "disabled": None,
    "WAN (4n+1)": (4, 1),
    "LTX2 (8n+1)": (8, 1),
    "H3 (17n+5)": (17, 5),
}
#: padding_mode choice that pads `add` onto the `model` dropdown's grid (or the custom step / plus). The default
#: for NEW nodes; a saved workflow keeps the padding_mode it stored, so its frame count does not change (L2.55).
USE_MODEL = "model preset"
PADDING_CHOICES = ("disabled", USE_MODEL) + tuple(k for k in GRIDS if k != "disabled")
MODES = ("add", "trim", "fit")
SIDES = ("head", "tail", "both")
TRIM_SOURCES = ("auto", "manual")


class AVHandlesError(ValueError):
    pass


# ── the memory ──────────────────────────────────────────────────────────────

@dataclass
class Handle:
    frames: int
    side: str
    fps: float
    source_frames: int      # what the clip was BEFORE the handles went on
    run: str | None
    #: The length to come back to, when `fit` set one. Obeyed in preference to
    #: `frames`: a sampler that returns one frame more or fewer than asked is
    #: common, and subtracting a remembered count from an unexpected length is
    #: how a clip ends up short with nothing to explain it.
    target: int = 0
    #: Frames on each end. `side = both` puts frames on the head AND the tail; -1 = derive from side/frames
    #: (records made before "both" existed).
    head: int = -1
    tail: int = -1
    #: L9.171 audio_sync "exact samples": the audio samples the add put on each end (-1 = not recorded), and their
    #: sample rate, so the trim removes exactly those samples whatever fps it would compute.
    audio_head_samples: int = -1
    audio_tail_samples: int = -1
    audio_sample_rate: int = 0

    def ends(self) -> tuple[int, int]:
        if self.head >= 0 and self.tail >= 0:
            return self.head, self.tail
        return split_ends(self.frames, self.side)

_MEM: dict[str, Handle] = {}
_LOCK = threading.Lock()
_LOADED = False


def _mem_file():
    """Where the memory survives a ComfyUI restart (add in one session, trim in the next). None outside ComfyUI."""
    try:
        import os
        import folder_paths  # type: ignore
        base = folder_paths.get_user_directory()
        if not base:
            return None
        return os.path.join(base, "c2c", "av_handles_memory.json")
    except Exception:  # noqa: BLE001 - tests / stubbed ComfyUI
        return None


def _load_disk() -> None:
    global _LOADED
    if _LOADED:
        return
    _LOADED = True
    path = _mem_file()
    if not path:
        return
    try:
        import json
        import os
        if not os.path.isfile(path):
            return
        with open(path, encoding="utf-8") as fh:
            raw = json.load(fh)
        for k, v in (raw or {}).items():
            if isinstance(v, dict) and k not in _MEM:
                _MEM[k] = Handle(**{f: v[f] for f in Handle.__dataclass_fields__ if f in v})
    except Exception:  # noqa: BLE001 - a broken file must never stop a run
        pass


def _save_disk() -> None:
    path = _mem_file()
    if not path:
        return
    try:
        import dataclasses
        import json
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump({k: dataclasses.asdict(v) for k, v in _MEM.items()}, fh)
        os.replace(tmp, path)
    except Exception:  # noqa: BLE001
        pass


def _run_id() -> str | None:
    """The current execution's id, when the host exposes one."""
    try:
        from comfy_execution.utils import get_executing_context
        ctx = get_executing_context()
        return getattr(ctx, "prompt_id", None) if ctx else None
    except Exception:  # noqa: BLE001 - older or stubbed ComfyUI
        return None


def remember(key: str, h: Handle) -> None:
    with _LOCK:
        _load_disk()
        _MEM[key or "default"] = h
        _save_disk()


def recall(key: str) -> Handle | None:
    with _LOCK:
        _load_disk()
        return _MEM.get(key or "default")


def forget_all() -> None:
    """Tests only."""
    global _LOADED
    with _LOCK:
        _MEM.clear()
        _LOADED = True     # and do not read a saved memory back in


# ── frame grids ─────────────────────────────────────────────────────────────

def on_grid(n: int, grid: str) -> bool:
    g = GRIDS.get(grid)
    if not g:
        return True
    step, off = g
    return n >= off and (n - off) % step == 0


def next_on_grid(n: int, grid: str) -> int:
    """The smallest legal count at or above n."""
    g = GRIDS.get(grid)
    if not g:
        return n
    step, off = g
    if n <= off:
        return off
    k = -(-(n - off) // step)          # ceil division
    return off + k * step


def add_grid(padding_mode: str, model: str, custom_step: int = 4,
             custom_plus: int = 1) -> tuple[Grid | None, str]:
    """The grid an `add` lands on, and where it came from (L2.55).

    `add` used to read only padding_mode's fixed table: the `model` preset and
    the custom step / plus did nothing in add mode, and with the widget-hiding
    front-end not loaded (owner's Linux box) every preset and custom formula
    looked broken. padding_mode = USE_MODEL now pads onto the model grid; the
    other padding_mode values keep their old result."""
    if GRIDS.get(padding_mode):
        return None, f"{padding_mode} (padding_mode)"
    if padding_mode != USE_MODEL:
        return None, (f"none - padding_mode is {padding_mode!r}; choose {USE_MODEL!r} "
                      "to land on the model / custom grid")
    g = resolve(model, int(custom_step), int(custom_plus))
    if g.step <= 1:
        return None, f"none ({model})"
    return g, (g.note if model == CUSTOM else model) + " (model)"


def plan_add(source: int, want: int, grid: str) -> int:
    """How many frames actually go on.

    `want = 0` with a grid means "just enough to land on it", which is the
    common case: the clip is 121 frames, the model wants 4n+1, add 0 and it
    stays 121 rather than being padded for no reason.
    """
    if not GRIDS.get(grid):
        return max(0, want)
    target = next_on_grid(source + max(0, want), grid)
    return max(0, target - source)


# ── the work ────────────────────────────────────────────────────────────────

def pad_images(images: torch.Tensor, count: int, side: str) -> torch.Tensor:
    """Repeat the edge frame. Repeating, not black: a black run at the head is
    content the model will try to continue out of."""
    if count <= 0:
        return images
    if side == "head":
        return torch.cat([images[0:1].repeat(count, 1, 1, 1), images], dim=0)
    return torch.cat([images, images[-1:].repeat(count, 1, 1, 1)], dim=0)


def trim_images(images: torch.Tensor, count: int, side: str) -> torch.Tensor:
    if count <= 0:
        return images
    return images[count:] if side == "head" else images[:-count]


def split_ends(count: int, side: str) -> tuple[int, int]:
    """(head, tail) for `count` frames on `side`. `both` splits a total, the extra frame going to the head."""
    count = max(0, int(count))
    if side == "head":
        return count, 0
    if side == "tail":
        return 0, count
    head = (count + 1) // 2
    return head, count - head


def pad_ends(images: torch.Tensor, head: int, tail: int) -> torch.Tensor:
    return pad_images(pad_images(images, head, "head"), tail, "tail")


def trim_ends(images: torch.Tensor, head: int, tail: int) -> torch.Tensor:
    return trim_images(trim_images(images, head, "head"), tail, "tail")


def _waveform_parts(audio: dict):
    """(2D waveform, restore fn) for 1D/2D/3D audio, so the shape that comes
    out is the shape that went in — a node that silently changes an audio
    tensor's rank breaks whatever is downstream."""
    w = audio["waveform"]
    shape = w.shape
    if w.ndim == 3:
        batch = shape[0]
        return w[0], lambda x: x.unsqueeze(0).repeat(batch, 1, 1)
    if w.ndim == 1:
        return w.unsqueeze(0), lambda x: x.squeeze(0)
    return w, lambda x: x


def shift_audio(audio: dict, frames: int, fps: float, side: str,
                add: bool) -> dict:
    """Keep sound in step with picture.

    The whole point of the pair: handles added to the picture without the
    matching silence slide the audio against it, and a lip-sync that is eight
    frames out is worse than no lip-sync.
    """
    if audio is None or frames <= 0 or fps <= 0:
        return audio
    w, restore = _waveform_parts(audio)
    sr = int(audio["sample_rate"])
    n = int(round(frames / float(fps) * sr))
    if n <= 0:
        return audio

    if add:
        pad = torch.zeros(w.shape[0], n, dtype=w.dtype, device=w.device)
        out = torch.cat([pad, w], dim=1) if side == "head" else torch.cat([w, pad], dim=1)
    else:
        if n >= w.shape[1]:
            raise AVHandlesError(
                f"Trimming {frames} frames means cutting {n} audio samples, but "
                f"the clip only has {w.shape[1]}. The handle count does not "
                "belong to this audio.")
        out = w[:, n:] if side == "head" else w[:, :-n]
    return {"waveform": restore(out), "sample_rate": sr}


def shift_audio_ends(audio: dict, head: int, tail: int, fps: float, add: bool) -> dict:
    """shift_audio on each end that has frames."""
    if head > 0:
        audio = shift_audio(audio, head, fps, "head", add)
    if tail > 0:
        audio = shift_audio(audio, tail, fps, "tail", add)
    return audio


AUDIO_SYNC = ("frame duration", "exact samples")     # L9.171: appended control; "frame duration" = shift_audio


def end_samples(audio: dict, head: int, tail: int, fps: float) -> tuple[int, int, int]:
    """(head samples, tail samples, sample rate) that shift_audio_ends moves for these frame counts."""
    sr = int(audio["sample_rate"])

    def n(k: int) -> int:
        return int(round(k / float(fps) * sr)) if k > 0 and fps > 0 else 0
    return n(head), n(tail), sr


def shift_audio_samples(audio: dict, head_n: int, tail_n: int, add: bool) -> dict:
    """shift_audio_ends by exact sample counts (L9.171, A25 spec item 70.2): silence put on / cut off each end."""
    if audio is None or (head_n <= 0 and tail_n <= 0):
        return audio
    w, restore = _waveform_parts(audio)
    sr = int(audio["sample_rate"])
    head_n, tail_n = max(0, int(head_n)), max(0, int(tail_n))
    if add:
        out = torch.cat([torch.zeros(w.shape[0], head_n, dtype=w.dtype, device=w.device), w,
                         torch.zeros(w.shape[0], tail_n, dtype=w.dtype, device=w.device)], dim=1)
    else:
        if head_n + tail_n >= w.shape[1]:
            raise AVHandlesError(
                f"Trimming the recorded {head_n} + {tail_n} audio samples would leave nothing of the "
                f"{w.shape[1]}-sample clip. The recorded handles do not belong to this audio.")
        out = w[:, head_n:w.shape[1] - tail_n]
    return {"waveform": restore(out), "sample_rate": sr}


def fade_audio(audio: dict, fade_in_at: int | None, fade_out_end: int | None, n: int) -> dict:
    """Linear fade-in of n samples starting at fade_in_at, linear fade-out of n samples ending at fade_out_end
    (None = no fade on that side). Clicks at a cut or at the edge of added silence go (L9.171)."""
    if audio is None or n <= 0 or (fade_in_at is None and fade_out_end is None):
        return audio
    w, restore = _waveform_parts(audio)
    w = w.clone()
    length = int(w.shape[1])
    if fade_in_at is not None:
        a = max(0, min(int(fade_in_at), length))
        k = min(n, length - a)
        if k > 0:
            w[:, a:a + k] *= torch.linspace(0.0, 1.0, k, dtype=w.dtype, device=w.device)
    if fade_out_end is not None:
        b = max(0, min(int(fade_out_end), length))
        k = min(n, b)
        if k > 0:
            w[:, b - k:b] *= torch.linspace(1.0, 0.0, k, dtype=w.dtype, device=w.device)
    return {"waveform": restore(w), "sample_rate": int(audio["sample_rate"])}


def resolve_fps(images, audio, manual_fps: float, source_frames: int) -> tuple[float, str]:
    """fps, and where it came from.

    Derived from the clip's own duration when possible, because a guessed 30
    on 24fps material puts the silence 25% out — which reads as a lip-sync
    problem, not a settings problem.
    """
    if manual_fps and manual_fps > 0:
        return float(manual_fps), "set by hand"
    if audio is not None and source_frames > 0:
        w, _ = _waveform_parts(audio)
        dur = w.shape[1] / float(audio["sample_rate"])
        if dur > 0.001:
            return source_frames / dur, "measured from this clip's own audio"
    return 30.0, "ASSUMED 30 — no audio to measure against"


def _where(side: str, head: int, tail: int) -> str:
    if side == "both":
        return f"head ({head}) and tail ({tail})"
    return f"the {side}"


def describe(mode: str, used: int, src: int, out: int, side: str, grid: str,
             fps: float, fps_from: str, origin: str, has_audio: bool,
             head: int | None = None, tail: int | None = None,
             model_grid: Grid | None = None, grid_label: str = "") -> str:
    lines = []
    if head is None or tail is None:
        head, tail = split_ends(used, side)
    if mode == "add":
        lines.append(f"Added {used} frame(s) at {_where(side, head, tail)}: {src} -> {out}.")
        if used == 0:
            lines.append("Nothing was added. With a grid selected that means "
                         "the clip already lands on it.")
    else:
        lines.append(f"Trimmed {used} frame(s) off {_where(side, head, tail)}: {src} -> {out}.")
        lines.append("Count came from: " + origin)
        if used == 0:
            lines.append(
                "NOTHING WAS TRIMMED. No handles were remembered and none were "
                "wired in, so there was no number to use. Either run the add "
                "side first, wire its `handles` output to this node, or set "
                "trim_amount to 'manual' and type the count in.")

    if mode == "add" and grid_label:
        lines.append(f"Grid: {grid_label}.")
    ok, name = None, ""
    if GRIDS.get(grid):
        ok, name = on_grid(out, grid), grid
    elif model_grid is not None:
        ok, name = model_grid.accepts(out), grid_label.replace(" (model)", "")
    if ok is not None:
        lines.append(
            f"{out} frames {'is' if ok else 'is NOT'} on the {name} grid." +
            ("" if ok else
             " A model expecting that grid will reinterpret the clip rather "
             "than refuse it, so the result comes out at the wrong speed."))
    if has_audio:
        lines.append(f"Audio moved with it at {fps:.3f} fps ({fps_from}).")
        if fps_from.startswith("ASSUMED"):
            lines.append(
                "That fps was a guess. On 24fps material a guessed 30 puts the "
                "silence 25% out, which reads as a lip-sync fault rather than "
                "a settings one — set manual_fps.")
    else:
        lines.append("No audio connected, so only the picture changed.")
    return "\n".join(lines)


# ── the node ────────────────────────────────────────────────────────────────

class AVHandlesMEC:
    """Add stabilisation handles, and trim exactly those back off."""

    CATEGORY = "C2C/Video"
    RETURN_TYPES = ("IMAGE", "AUDIO", "INT", "HANDLES", "STRING")
    RETURN_NAMES = ("images", "audio", "total_frames", "handles", "info")
    OUTPUT_TOOLTIPS = (
        "Frames, with handles added or removed.",
        "Audio moved by the same duration, so it stays in step.",
        "How many frames came out.",
        "The handle count, to wire into the trim node. Exact, and it does not "
        "depend on the memory — use it when the add and the trim are in "
        "different workflows.",
        "What happened, including WHERE the trim count came from.",
    )
    FUNCTION = "execute"
    DESCRIPTION = (
        "Add repeated frames at the head (or tail) to stabilise a video model, "
        "then trim exactly those back off afterwards — without retyping the "
        "number. In trim mode it uses the count the add recorded, so the two "
        "ends cannot drift apart. Audio is moved by the matching duration."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (list(MODES), {"default": "fit", "tooltip":
                    "fit  — say how many frames you want OUT. The node works "
                    "out what the model will accept, pads up to it, and "
                    "remembers the target so the trim lands exactly there. "
                    "This is the one to use.\n\n"
                    "add  — put a specific number of frames on, and remember "
                    "how many.\n\n"
                    "trim — take them back off, using the remembered target "
                    "or count."}),
                "handle_frames": ("INT", {
                    "default": 8, "min": 0, "max": 400, "step": 1, "tooltip":
                    "ADD: how many frames to put on. With a grid selected, 0 "
                    "means 'just enough to land on the grid'.\n\n"
                    "TRIM: ignored unless trim_amount is 'manual'."}),
                "trim_amount": (list(TRIM_SOURCES), {"default": "auto",
                    "tooltip":
                    "TRIM mode only.\n\n"
                    "auto   — use the count the add side recorded (or the one "
                    "wired into `handles`). This is the point of the node: the "
                    "two ends cannot drift apart.\n\n"
                    "manual — use handle_frames instead. Only for undoing "
                    "handles this node did not add.\n\n"
                    "It is a separate control because 'left at the default' "
                    "and 'deliberately typed 8' cannot be told apart in one "
                    "number, and guessing wrong silently trims the wrong "
                    "amount."}),
            },
            "optional": {
                "images": ("IMAGE", {"tooltip": "The clip to pad or trim (fit mode needs it). Wire images, audio or both."}),
                "audio": ("AUDIO", {"tooltip": "Audio moved by the same duration as the frames, so it stays in sync."}),
                "target_frames": ("INT", {
                    "default": 0, "min": 0, "max": 100000, "step": 1,
                    "tooltip":
                    "FIT mode: how many frames you want OUT. The model's own "
                    "grid is an implementation detail — ask for 64 and you "
                    "get 64, whatever the model needs to render to make "
                    "them.\n\n"
                    "0 in fit mode means 'use the clip's current length as "
                    "the target', which pads it to something the model "
                    "accepts and trims straight back."}),
                "model": (list(GRID_NAMES), {
                    "default": "MiniMax H3 (17n+5)", "tooltip":
                    "FIT, and ADD with padding_mode 'model preset': which "
                    "model's frame grid to land on. Every video model wants "
                    "step*n + plus frames and they disagree only on the "
                    "numbers: H3 is 17n+5, Wan is 4n+1, LTX is 8n+1.\n\n"
                    "Hand a model a count it does not accept and it does not "
                    "complain — it reinterprets the clip, and the result "
                    "comes out at the wrong speed."}),
                "custom_step": ("INT", {"default": 4, "min": 1, "max": 1000,
                    "tooltip": "Only used when model is 'custom'."}),
                "custom_plus": ("INT", {"default": 1, "min": 0, "max": 1000,
                    "tooltip": "Only used when model is 'custom'."}),
                "side": (list(SIDES), {"default": "head", "tooltip":
                    "Which end. Handles go on the head because that is the run "
                    "a video model uses to settle.\n\n"
                    "both: handle_frames on the head AND the tail (fit splits "
                    "its padding across the two ends).\n\n"
                    "TRIM takes a wired or remembered count off exactly the "
                    "ends it went on; this setting only matters for a manual "
                    "count."}),
                "padding_mode": (list(PADDING_CHOICES), {"default": USE_MODEL, "tooltip":
                    "ADD: round the total onto a model's frame grid. A model "
                    "that wants 4n+1 and is handed 4n does not complain — it "
                    "reinterprets the clip, and the result comes out at the "
                    "wrong speed.\n\n"
                    "model preset: the grid of the `model` dropdown (or the "
                    "custom step / plus). disabled: exactly handle_frames, no "
                    "rounding. The fixed WAN / LTX2 / H3 entries are kept for "
                    "workflows saved with them."}),
                "manual_fps": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 240.0, "step": 0.001,
                    "tooltip":
                    "0 measures it from the clip's own audio. Set it when "
                    "there is no audio to measure against — a guessed 30 on "
                    "24fps material puts the silence 25% out."}),
                "handles": ("HANDLES", {"tooltip":
                    "Wire the add node's `handles` output here for an exact, "
                    "explicit count. Takes priority over the remembered one, "
                    "and works across separate workflows where the memory "
                    "cannot."}),
                "memo_key": ("STRING", {"default": "", "tooltip":
                    "Only needed when two independent clips are being handled "
                    "in one workflow — give each its own name so their counts "
                    "do not overwrite each other. Empty is right for one clip."}),
                # L9.171 (A25 spec item 70) - appended; the defaults give exactly the old result
                "handle_align": ("BOOLEAN", {"default": False, "tooltip":
                    "ADD only. Round each end's handle count UP to a whole multiple of the `model` grid step (Wan 4, "
                    "LTX 8, H3 17, or custom_step), so the handles fill whole latent time blocks and a clip that was "
                    "on the grid is on it again after the trim. If a padding grid is also on, its rounding frames "
                    "go on the tail. Off: handle_frames exactly, as before."}),
                "audio_sync": (list(AUDIO_SYNC), {"default": "frame duration", "tooltip":
                    "frame duration: audio moves by frames / fps on each end (as before). exact samples: the add "
                    "records the exact samples it inserted (in `handles` and the memory), and the trim removes "
                    "exactly those when it undoes the same ends - the audio comes back sample for sample even when "
                    "the trim side measures a slightly different fps. Set it on the add AND the trim."}),
                "audio_fade_ms": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1000.0, "step": 1.0, "tooltip":
                    "0 = off. Linear fades of this length at each audio edge the node makes: add / fit fade the "
                    "original audio in after the head silence and out before the tail silence; trim fades in at "
                    "the new start and out at the new end (only on ends that were changed). Removes clicks."}),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kw):
        # Core passes only widget constants here (linked images / audio never
        # arrive), so this keys the cache on the settings and the mode. A cached
        # add does not re-record: the record from its last real run stays in the
        # memory, which is right unless another add with the same memo_key ran
        # in between - wire `handles` (or give each clip a memo_key) for that.
        import hashlib
        h = hashlib.md5()
        for k in sorted(kw):
            v = kw[k]
            if torch.is_tensor(v):
                h.update(str(tuple(v.shape)).encode())
            elif isinstance(v, dict) and torch.is_tensor(v.get("waveform")):
                h.update(str(tuple(v["waveform"].shape)).encode())
            else:
                h.update(f"{k}={v}".encode())
        if str(kw.get("mode", "")) == "add":
            h.update(b"|records-state")
        return h.hexdigest()

    def execute(self, **kw):
        """`run`, plus its report shown on the node after the run. A trim with nothing to go on passes the clip
        through unchanged, and with the report only on the `info` output nobody saw that it had done nothing
        (A9: "not adding or trimming at all")."""
        out = self.run(**kw)
        return {"ui": {"text": [out[4]]}, "result": out}

    def run(self, mode="fit", handle_frames=8, trim_amount="auto",
            target_frames=0, model="MiniMax H3 (17n+5)", custom_step=4,
            custom_plus=1, images=None, audio=None, side="head",
            padding_mode="disabled", manual_fps=0.0, handles=None,
            memo_key="", handle_align=False, audio_sync="frame duration",
            audio_fade_ms=0.0):
        if audio_sync not in AUDIO_SYNC:
            raise AVHandlesError(f"Unknown audio_sync {audio_sync!r}. Use 'frame duration' or 'exact samples'.")
        fade_ms = max(0.0, float(audio_fade_ms or 0.0))
        if mode not in MODES:
            raise AVHandlesError(
                f"Unknown mode {mode!r}. Use 'fit', 'add' or 'trim'.")
        if side not in SIDES:
            raise AVHandlesError(f"Unknown side {side!r}. Use 'head', 'tail' or 'both'.")
        if images is None and audio is None:
            raise AVHandlesError(
                "Connect at least an image batch or an audio clip — there is "
                "nothing to put handles on.")

        if images is not None and images.ndim == 3:
            images = images.unsqueeze(0)
        src_frames = int(images.shape[0]) if images is not None else 0
        key = (memo_key or "").strip() or "default"

        if mode == "fit":
            if images is None:
                raise AVHandlesError(
                    "fit mode needs images: it works from the clip's length. "
                    "For audio alone, use add with an explicit count.")
            grid = resolve(model, custom_step, custom_plus)
            want = int(target_frames) or src_frames
            plan = plan_fit(want, grid)
            # Pad the CLIP up to what the model accepts. The target is what
            # the trim will obey later, so it is recorded alongside the count.
            used = plan.generate - src_frames
            if used < 0:
                raise AVHandlesError(
                    f"This clip is already {src_frames} frames but the target "
                    f"is {want}. fit only pads — trim it down first, or set "
                    "the target to the length you actually want.")
            head, tail = split_ends(used, side)
            fps, fps_from = resolve_fps(images, audio, manual_fps, src_frames)
            out_img = pad_ends(images, head, tail)
            out_aud = shift_audio_ends(audio, head, tail, fps, add=True)
            out_aud, a_rec, a_notes = self._add_audio_extras(audio, out_aud, head, tail, fps, audio_sync, fade_ms)
            total = int(out_img.shape[0])
            remember(key, Handle(frames=used, side=side, fps=fps,
                                 source_frames=src_frames, run=_run_id(),
                                 target=plan.target, head=head, tail=tail, **a_rec))
            info = (describe_fit(plan, model) + "\n" +
                    describe("add", used, src_frames, total, side, "disabled",
                             fps, fps_from, "recorded for the trim side",
                             audio is not None, head, tail))
            if a_notes:
                info += "\n" + "\n".join(a_notes)
            return (out_img, out_aud, total,
                    dict({"frames": used, "side": side, "fps": fps,
                          "source_frames": src_frames, "target": plan.target,
                          "head": head, "tail": tail}, **a_rec),
                    info)

        model_grid, grid_label = None, ""
        extra_notes: list[str] = []          # L9.171 report lines, only when a new control is on
        a_rec: dict = {}
        if mode == "add":
            want = max(0, int(handle_frames))
            model_grid, grid_label = add_grid(padding_mode, model, custom_step, custom_plus)

            def planned(n_want: int) -> int:
                if images is None:
                    return n_want
                if GRIDS.get(padding_mode):                       # legacy table, unchanged result
                    return plan_add(src_frames, n_want, padding_mode)
                if model_grid is None:
                    return n_want
                return model_grid.next_at_or_above(src_frames + n_want) - src_frames

            if side == "both":
                # handle_frames on EACH end; a grid's rounding frames go on the tail, so the head (the run the
                # model settles on) is exactly what was asked for
                needed = planned(2 * want)
                head, tail = want, needed - want
            else:
                head, tail = split_ends(planned(want), side)
            if handle_align:
                head, tail, align_note = self._align_ends(head, tail, src_frames, images is not None, padding_mode,
                                                          model_grid, model, custom_step, custom_plus)
                extra_notes.append(align_note)
            used = head + tail
            fps, fps_from = resolve_fps(images, audio, manual_fps, src_frames)
            out_img = pad_ends(images, head, tail) if images is not None else None
            out_aud = shift_audio_ends(audio, head, tail, fps, add=True)
            out_aud, a_rec, a_notes = self._add_audio_extras(audio, out_aud, head, tail, fps, audio_sync, fade_ms)
            extra_notes.extend(a_notes)
            total = int(out_img.shape[0]) if out_img is not None else used + src_frames

            rec = Handle(frames=used, side=side, fps=fps,
                         source_frames=src_frames, run=_run_id(), target=0,
                         head=head, tail=tail, **a_rec)
            remember(key, rec)
            origin = "recorded for the trim side"
        else:
            used, origin, (head, tail), rec_side = self._trim_count(
                handle_frames, trim_amount, handles, key, side)
            # A recorded TARGET wins over a recorded count. `fit` set one, and
            # trimming to a length survives a sampler that returned one frame
            # more or fewer than it was asked for — subtracting a remembered
            # count from an unexpected length is how a clip ends up short.
            if trim_amount != "manual" and images is not None:
                tgt = 0
                if isinstance(handles, dict):
                    tgt = int(handles.get("target") or 0)
                if not tgt and not isinstance(handles, dict):
                    # audit (c2c_backend_audit av_handles.py:734): wired handles always win - a stale
                    # av_handles_memory.json target from another clip must not override them
                    rem = recall(key)
                    tgt = int(getattr(rem, "target", 0) or 0) if rem else 0
                if tgt:
                    excess = trim_to(src_frames, tgt)
                    if excess != head + tail:      # not what was added: split it the way it went on
                        head, tail = split_ends(excess, rec_side)
                    used = excess
                    origin = (f"the target of {tgt} frames recorded by fit "
                              f"({src_frames} arrived, so {used} come off)")
            side = rec_side
            if images is not None and used >= src_frames:
                raise AVHandlesError(
                    f"Trimming {used} frames off a {src_frames}-frame clip "
                    "would leave nothing. Downstream reads an empty batch as "
                    "'no video' and says nothing useful, so this stops here. "
                    "The remembered count does not belong to this clip — "
                    "check the add side ran on the same material.")
            remembered = recall(key)
            wired_fps = float(handles.get("fps") or 0.0) if isinstance(handles, dict) else 0.0
            fps, fps_from = resolve_fps(
                images, audio, manual_fps or wired_fps or (remembered.fps if remembered else 0.0),
                src_frames)
            if wired_fps and not manual_fps:
                fps_from = "from the wired handles"
            elif remembered and not manual_fps:
                fps_from = "carried from the add side"
            out_img = trim_ends(images, head, tail) if images is not None else None
            exact_aud, sync_note = None, None
            if audio is not None and audio_sync == "exact samples":
                exact_aud, sync_note = self._trim_exact_samples(audio, head, tail, trim_amount, handles, key)
                extra_notes.append(sync_note)
            out_aud = exact_aud if exact_aud is not None else shift_audio_ends(audio, head, tail, fps, add=False)
            if audio is not None and fade_ms > 0:
                n_fade = int(round(fade_ms / 1000.0 * int(audio["sample_rate"])))
                length = int(_waveform_parts(out_aud)[0].shape[1])
                out_aud = fade_audio(out_aud, 0 if head > 0 else None, length if tail > 0 else None, n_fade)
                extra_notes.append(f"Audio fades: {fade_ms:g} ms ({n_fade} samples)"
                                   + (" in at the new start" if head > 0 else "")
                                   + (" and" if head > 0 and tail > 0 else "")
                                   + (" out at the new end" if tail > 0 else "")
                                   + ("" if head > 0 or tail > 0 else " - nothing was cut, so none applied") + ".")
            total = int(out_img.shape[0]) if out_img is not None else max(0, src_frames - used)
            if used == 0:
                import logging
                logging.getLogger("C2C.AVHandles").warning(
                    "AV Handles trim: nothing was trimmed (%s). Wire the add node's `handles` output to this "
                    "node, or set trim_amount to 'manual'.", origin)

        info = describe(mode, used, src_frames, total, side, padding_mode,
                        fps, fps_from, origin, audio is not None, head, tail,
                        model_grid=model_grid, grid_label=grid_label)
        if extra_notes:
            info += "\n" + "\n".join(extra_notes)
        _rem = recall(key)
        return (out_img, out_aud, total,
                dict({"frames": used, "side": side, "fps": fps,
                      "source_frames": src_frames,
                      "target": int(getattr(_rem, "target", 0) or 0) if _rem else 0,
                      "head": head, "tail": tail}, **(a_rec if mode == "add" else {})),
                info)

    # ── L9.171 (A25 spec item 70) ───────────────────────────────────────────

    @staticmethod
    def _align_ends(head, tail, src_frames, has_images, padding_mode, model_grid, model, custom_step, custom_plus):
        """Each end with handles rounded UP to a multiple of the model grid's step; a padding grid's rounding frames
        then go on the tail so the padded clip still lands on that grid."""
        g = resolve(model, int(custom_step), int(custom_plus))
        step = int(getattr(g, "step", 1) or 1)
        name = g.note if model == CUSTOM else model
        if step <= 1:
            return head, tail, f"handle_align: {name} has no frame step, so nothing was aligned."

        def up(k: int) -> int:
            return -(-int(k) // step) * step if k > 0 else 0
        head, tail = up(head), up(tail)
        if has_images:
            total = src_frames + head + tail
            if GRIDS.get(padding_mode):
                tail += plan_add(total, 0, padding_mode)
            elif model_grid is not None and not model_grid.accepts(total):
                tail += model_grid.next_at_or_above(total) - total
        return head, tail, f"Handles aligned to the {name} step of {step}: head {head}, tail {tail}."

    @staticmethod
    def _add_audio_extras(audio_in, out_aud, head, tail, fps, audio_sync, fade_ms):
        """(audio, record for HANDLES / memory, report lines) after an add / fit put head + tail frames on."""
        if audio_in is None or (audio_sync != "exact samples" and fade_ms <= 0):
            return out_aud, {}, []
        hn, tn, sr = end_samples(audio_in, head, tail, fps)
        rec, notes = {}, []
        if audio_sync == "exact samples":
            rec = {"audio_head_samples": hn, "audio_tail_samples": tn, "audio_sample_rate": sr}
            notes.append(f"Audio: exact samples recorded for the trim - {hn} on the head, {tn} on the tail "
                         f"at {sr} Hz.")
        if fade_ms > 0:
            n = int(round(fade_ms / 1000.0 * sr))
            length = int(_waveform_parts(out_aud)[0].shape[1])
            out_aud = fade_audio(out_aud, hn if head > 0 else None, length - tn if tail > 0 else None, n)
            notes.append(f"Audio fades: {fade_ms:g} ms ({n} samples)"
                         + (" in after the head silence" if head > 0 else "")
                         + (" and" if head > 0 and tail > 0 else "")
                         + (" out before the tail silence" if tail > 0 else "")
                         + ("" if head > 0 or tail > 0 else " - no handles were added, so none applied") + ".")
        return out_aud, rec, notes

    @staticmethod
    def _trim_exact_samples(audio, head, tail, trim_amount, handles, key):
        """(trimmed audio or None, report line): the recorded samples off each end when this trim undoes exactly
        the recorded ends at the same sample rate; None = fall back to frame duration (the line says why)."""
        if trim_amount == "manual":
            return None, "Audio: exact samples not used - a manual count has no recorded samples (frame duration)."
        if isinstance(handles, dict) and "frames" in handles:
            src = handles if "audio_head_samples" in handles else None
            r_ends = (int(handles.get("head", -1)), int(handles.get("tail", -1)))
            where = "the `handles` wire"
        else:
            rem = recall(key)
            src = None
            if rem is not None and int(rem.audio_head_samples) >= 0:
                src = {"audio_head_samples": rem.audio_head_samples, "audio_tail_samples": rem.audio_tail_samples,
                       "audio_sample_rate": rem.audio_sample_rate}
            r_ends = rem.ends() if rem is not None else (-1, -1)
            where = "the memory"
        if src is None:
            return None, (f"Audio: exact samples not used - {where} has no recorded samples (set audio_sync to "
                          "'exact samples' on the add side too); trimmed by frame duration.")
        if (int(head), int(tail)) != tuple(r_ends):
            return None, (f"Audio: exact samples not used - this trim ({head} + {tail} frames) is not the recorded "
                          f"add ({r_ends[0]} + {r_ends[1]}); trimmed by frame duration.")
        if int(src["audio_sample_rate"]) != int(audio["sample_rate"]):
            return None, (f"Audio: exact samples not used - recorded at {src['audio_sample_rate']} Hz, this audio is "
                          f"{int(audio['sample_rate'])} Hz; trimmed by frame duration.")
        hn, tn = int(src["audio_head_samples"]), int(src["audio_tail_samples"])
        return (shift_audio_samples(audio, hn, tn, add=False),
                f"Audio: exactly the recorded {hn} head + {tn} tail samples removed ({where}).")

    @staticmethod
    def _trim_count(typed, source, wired, key, side="head"):
        """(count, where it came from, (head, tail), side) - in priority order.

        `manual` is checked FIRST because it is an explicit instruction; a
        wired signal would otherwise silently override a deliberate choice.
        A wired or remembered count comes off exactly the ends it went on: the
        trim's own `side` widget only matters for a manual count (left at
        `head` after an add at the tail, it used to cut the wrong end).
        """
        if source == "manual":
            n = int(typed or 0)
            if side == "both":
                return 2 * n, "handle_frames on each end, because trim_amount is 'manual'", (n, n), side
            return n, "handle_frames, because trim_amount is 'manual'", split_ends(n, side), side
        if isinstance(wired, dict) and "frames" in wired:
            n = int(wired["frames"])
            w_side = str(wired.get("side") or side)
            if "head" in wired and "tail" in wired and int(wired["head"]) + int(wired["tail"]) == n:
                ends = (int(wired["head"]), int(wired["tail"]))
            else:
                ends = split_ends(n, w_side)
            return n, "the `handles` wire from the add node", ends, w_side
        rec = recall(key)
        if rec:
            same = rec.run is not None and rec.run == _run_id()
            return rec.frames, (
                "remembered from the add node earlier in THIS run" if same else
                "remembered from the add node in an EARLIER run — worth a "
                "glance that it belongs to this clip"), rec.ends(), rec.side
        return 0, "nothing to go on", (0, 0), side

NODE_CLASS_MAPPINGS = {"AVHandlesMEC": AVHandlesMEC}
NODE_DISPLAY_NAME_MAPPINGS = {"AVHandlesMEC": "AV Handles — Add / Trim"}


# A28 (ORDERS 29, docs/audit/A28/PARAM_CNP_A.md "AVHandlesMEC"): which controls show when, read by js/_c2c_node_layout.js.
# Display only: every input keeps its name, order, type and default, is saved and reaches the node's function as before.
# fit: target + grid; add: count + padding (+ grid for model preset); trim: amount (+ count when manual).
try:
    from ._layout_decl import declare_layout, merge, sections
except ImportError:  # pragma: no cover - loaded outside the pack
    declare_layout = None
if declare_layout is not None:
    _AVH_CUSTOM = ["custom (step / plus)"]
    declare_layout(AVHandlesMEC, merge(sections(Advanced=["manual_fps", "memo_key"]), {
        "handle_frames": {"c2c_show_if_any": [{"mode": ["add"]}, {"mode": ["trim"], "trim_amount": ["manual"]}]},
        "trim_amount": {"c2c_show_if": {"mode": ["trim"]}},
        "target_frames": {"c2c_show_if": {"mode": ["fit"]}},
        "model": {"c2c_show_if_any": [{"mode": ["fit"]}, {"mode": ["add"], "padding_mode": ["model preset"]}]},
        "custom_step": {"c2c_show_if_any": [{"mode": ["fit"], "model": _AVH_CUSTOM},
                                            {"mode": ["add"], "padding_mode": ["model preset"], "model": _AVH_CUSTOM}]},
        "custom_plus": {"c2c_show_if_any": [{"mode": ["fit"], "model": _AVH_CUSTOM},
                                            {"mode": ["add"], "padding_mode": ["model preset"], "model": _AVH_CUSTOM}]},
        "padding_mode": {"c2c_show_if": {"mode": ["add"]}},
        # L9.171: appended controls
        "handle_align": {"c2c_show_if": {"mode": ["add"]}},
        "audio_sync": {"c2c_show_if_linked": ["audio"]},
        "audio_fade_ms": {"c2c_show_if_linked": ["audio"]},
    }))

#: L9.171 inputs appended after the A28 signature pin (tests/test_a28_cnp_a_layout.py leaves these out of its hash).
APPENDED_AFTER_A28 = ("handle_align", "audio_sync", "audio_fade_ms")
