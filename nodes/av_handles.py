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
        CUSTOM, GRID_NAMES, FrameGridError, describe_fit, plan_fit, resolve,
        trim_to,
    )
except ImportError:  # loaded as a top-level module rather than a package
    from _frame_grid import (  # type: ignore
        CUSTOM, GRID_NAMES, FrameGridError, describe_fit, plan_fit, resolve,
        trim_to,
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
MODES = ("add", "trim", "fit")
SIDES = ("head", "tail")
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

_MEM: dict[str, Handle] = {}
_LOCK = threading.Lock()


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
        _MEM[key or "default"] = h


def recall(key: str) -> Handle | None:
    with _LOCK:
        return _MEM.get(key or "default")


def forget_all() -> None:
    """Tests only."""
    with _LOCK:
        _MEM.clear()


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


def describe(mode: str, used: int, src: int, out: int, side: str, grid: str,
             fps: float, fps_from: str, origin: str, has_audio: bool) -> str:
    lines = []
    if mode == "add":
        lines.append(f"Added {used} frame(s) at the {side}: {src} -> {out}.")
        if used == 0:
            lines.append("Nothing was added. With a grid selected that means "
                         "the clip already lands on it.")
    else:
        lines.append(f"Trimmed {used} frame(s) off the {side}: {src} -> {out}.")
        lines.append("Count came from: " + origin)
        if used == 0:
            lines.append(
                "NOTHING WAS TRIMMED. No handles were remembered and none were "
                "wired in, so there was no number to use. Either run the add "
                "side first, wire its `handles` output to this node, or set "
                "trim_amount to 'manual' and type the count in.")

    if GRIDS.get(grid):
        lines.append(
            f"{out} frames {'is' if on_grid(out, grid) else 'is NOT'} on the "
            f"{grid} grid." +
            ("" if on_grid(out, grid) else
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
    FUNCTION = "run"
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
                "images": ("IMAGE",),
                "audio": ("AUDIO",),
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
                    "Which model's frame grid to fit. Every video model wants "
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
                    "a video model uses to settle; trim takes them off the "
                    "same end."}),
                "padding_mode": (list(GRIDS), {"default": "disabled", "tooltip":
                    "Round the total onto a model's frame grid. A model that "
                    "wants 4n+1 and is handed 4n does not complain — it "
                    "reinterprets the clip, and the result comes out at the "
                    "wrong speed."}),
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
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kw):
        # Add RECORDS state, so it must not be skipped by a cache hit: a
        # second run that reuses the cached output never refreshes the memory,
        # and the trim then works from a stale count.
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

    def run(self, mode="fit", handle_frames=8, trim_amount="auto",
            target_frames=0, model="MiniMax H3 (17n+5)", custom_step=4,
            custom_plus=1, images=None, audio=None, side="head",
            padding_mode="disabled", manual_fps=0.0, handles=None,
            memo_key=""):
        if mode not in MODES:
            raise AVHandlesError(
                f"Unknown mode {mode!r}. Use 'add' or 'trim'.")
        if side not in SIDES:
            raise AVHandlesError(f"Unknown side {side!r}. Use 'head' or 'tail'.")
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
            fps, fps_from = resolve_fps(images, audio, manual_fps, src_frames)
            out_img = pad_images(images, used, side)
            out_aud = shift_audio(audio, used, fps, side, add=True)
            total = int(out_img.shape[0])
            remember(key, Handle(frames=used, side=side, fps=fps,
                                 source_frames=src_frames, run=_run_id(),
                                 target=plan.target))
            info = (describe_fit(plan, model) + "\n" +
                    describe("add", used, src_frames, total, side, "disabled",
                             fps, fps_from, "recorded for the trim side",
                             audio is not None))
            return (out_img, out_aud, total,
                    {"frames": used, "side": side, "fps": fps,
                     "source_frames": src_frames, "target": plan.target},
                    info)

        if mode == "add":
            used = plan_add(src_frames, int(handle_frames), padding_mode) \
                if images is not None else max(0, int(handle_frames))
            fps, fps_from = resolve_fps(images, audio, manual_fps, src_frames)
            out_img = pad_images(images, used, side) if images is not None else None
            out_aud = shift_audio(audio, used, fps, side, add=True)
            total = int(out_img.shape[0]) if out_img is not None else used + src_frames

            rec = Handle(frames=used, side=side, fps=fps,
                         source_frames=src_frames, run=_run_id(), target=0)
            remember(key, rec)
            origin = "recorded for the trim side"
        else:
            used, origin = self._trim_count(
                handle_frames, trim_amount, handles, key)
            # A recorded TARGET wins over a recorded count. `fit` set one, and
            # trimming to a length survives a sampler that returned one frame
            # more or fewer than it was asked for — subtracting a remembered
            # count from an unexpected length is how a clip ends up short.
            if trim_amount != "manual" and images is not None:
                tgt = 0
                if isinstance(handles, dict):
                    tgt = int(handles.get("target") or 0)
                if not tgt:
                    rem = recall(key)
                    tgt = int(getattr(rem, "target", 0) or 0) if rem else 0
                if tgt:
                    used = trim_to(src_frames, tgt)
                    origin = (f"the target of {tgt} frames recorded by fit "
                              f"({src_frames} arrived, so {used} come off)")
            if images is not None and used >= src_frames:
                raise AVHandlesError(
                    f"Trimming {used} frames off a {src_frames}-frame clip "
                    "would leave nothing. Downstream reads an empty batch as "
                    "'no video' and says nothing useful, so this stops here. "
                    "The remembered count does not belong to this clip — "
                    "check the add side ran on the same material.")
            remembered = recall(key)
            fps, fps_from = resolve_fps(
                images, audio, manual_fps or (remembered.fps if remembered else 0.0),
                src_frames)
            if remembered and not manual_fps:
                fps_from = "carried from the add side"
            out_img = trim_images(images, used, side) if images is not None else None
            out_aud = shift_audio(audio, used, fps, side, add=False)
            total = int(out_img.shape[0]) if out_img is not None else max(0, src_frames - used)

        info = describe(mode, used, src_frames, total, side, padding_mode,
                        fps, fps_from, origin, audio is not None)
        _rem = recall(key)
        return (out_img, out_aud, total,
                {"frames": used, "side": side, "fps": fps,
                 "source_frames": src_frames,
                 "target": int(getattr(_rem, "target", 0) or 0) if _rem else 0},
                info)

    @staticmethod
    def _trim_count(typed, source, wired, key) -> tuple[int, str]:
        """The number, and where it came from — in priority order.

        `manual` is checked FIRST because it is an explicit instruction; a
        wired signal would otherwise silently override a deliberate choice.
        """
        if source == "manual":
            return int(typed or 0), "handle_frames, because trim_amount is 'manual'"
        if isinstance(wired, dict) and "frames" in wired:
            return int(wired["frames"]), "the `handles` wire from the add node"
        rec = recall(key)
        if rec:
            same = rec.run is not None and rec.run == _run_id()
            return rec.frames, (
                "remembered from the add node earlier in THIS run" if same else
                "remembered from the add node in an EARLIER run — worth a "
                "glance that it belongs to this clip")
        return 0, "nothing to go on"


NODE_CLASS_MAPPINGS = {"AVHandlesMEC": AVHandlesMEC}
NODE_DISPLAY_NAME_MAPPINGS = {"AVHandlesMEC": "AV Handles — Add / Trim"}
