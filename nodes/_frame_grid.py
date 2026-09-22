"""What frame counts each video model will actually accept.

Every video model wants the same shape and disagrees only on the numbers:

    frames = step * n + plus,   and never below min_frames

MiniMax H3 is 17n+5, Wan is 4n+1, LTX is 8n+1. That vocabulary — step, plus,
min — is taken from ComfyUI-Pixaroma's `_duration_helpers.py` (MIT), which
uses it to turn seconds into a frame count. Using the same three words here
means a number worked out in one place still means the same thing in the
other, instead of two tables that describe the same models differently.

WHAT THIS ADDS that computing the count does not. Knowing you need 124 frames
does not get you the 120 you asked for. The model hands back 124, and either
you live with four frames you did not want or you go and trim them by hand and
have to remember how many. So the useful primitive is not "what count is
legal" but:

    plan_fit(want=120, model="MiniMax H3")
      -> generate 124, then take the first 120

which is a pad and a trim that know about each other.

AND THE TRIM IS BY TARGET, NOT BY COUNT. That distinction is the whole
robustness of it. Pad 120 -> 124 and the obvious trim is "remove 4". But
samplers do not always return what was asked for — a pipeline that drops or
adds a frame is common — and removing 4 from 123 gives 119, silently. Trimming
TO 120 gives 120 from whatever arrives. The count is kept for reporting; the
target is what is obeyed.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Grid:
    """A model's accepted frame counts, as step*n + plus."""

    step: int
    plus: int
    min_frames: int
    fps: float
    note: str = ""

    def accepts(self, n: int) -> bool:
        if n < self.min_frames:
            return False
        if self.step <= 1:
            return True
        return (n - self.plus) >= 0 and (n - self.plus) % self.step == 0

    def next_at_or_above(self, n: int) -> int:
        """The smallest count this model accepts that is >= n."""
        n = max(int(n), self.min_frames)
        if self.step <= 1:
            return n
        if n <= self.plus:
            return self.plus if self.plus >= self.min_frames else \
                self._climb(self.min_frames)
        k = -(-(n - self.plus) // self.step)          # ceil division
        return self.plus + k * self.step

    def _climb(self, floor: int) -> int:
        k = 0
        while self.plus + k * self.step < floor:
            k += 1
        return self.plus + k * self.step


#: The models this pack knows. `step = 1` means "any count", which is the
#: honest entry for a model with no grid rather than leaving it out.
#:
#: H3's 17n+5 was verified against the model code in this workspace, not
#: taken on trust: video packs as 5 frames then 17-frame chunks. The rest
#: follow Pixaroma's table, which matches the published figures.
GRIDS: dict[str, Grid] = {
    "none (any frame count)": Grid(1, 0, 1, 24.0,
        "No grid. Anything goes, so nothing is padded."),
    "MiniMax H3 (17n+5)": Grid(17, 5, 5, 24.0,
        "Video packs as 5 frames then 17-frame chunks; 124 is the 5-second "
        "figure everyone quotes."),
    "Wan 2.x (4n+1)": Grid(4, 1, 5, 16.0,
        "Wan and Wan-Animate. 16 fps is Wan's own training rate."),
    "LTX / LTX-2 (8n+1)": Grid(8, 1, 9, 24.0, ""),
    "Hunyuan Video (4n+1)": Grid(4, 1, 5, 24.0, ""),
    "CogVideoX (8n+1)": Grid(8, 1, 9, 8.0,
        "49 frames is CogVideoX's usual length: 8*6+1."),
    "Mochi (6n+1)": Grid(6, 1, 7, 30.0, ""),
    "custom (step / plus)": Grid(1, 0, 1, 24.0,
        "Set step and plus by hand for a model not listed."),
}

GRID_NAMES = tuple(GRIDS)
CUSTOM = "custom (step / plus)"


class FrameGridError(ValueError):
    pass


def resolve(model: str, step: int = 1, plus: int = 0,
            min_frames: int = 1) -> Grid:
    """The grid for a named model, or one built from step/plus for `custom`."""
    if model == CUSTOM:
        if step < 1:
            raise FrameGridError(
                f"A custom step must be at least 1, got {step}. Step 1 means "
                "'any frame count', which is what 'none' already does.")
        if plus < 0:
            raise FrameGridError(f"A custom plus cannot be negative, got {plus}.")
        return Grid(int(step), int(plus), max(1, int(min_frames)), 24.0,
                    f"custom {step}n+{plus}")
    g = GRIDS.get(model)
    if g is None:
        raise FrameGridError(
            f"Unknown model {model!r}. Choose one of: " +
            ", ".join(GRID_NAMES) + ".")
    return g


@dataclass(frozen=True)
class FitPlan:
    """How to get exactly `target` frames out of a model that will not make
    exactly `target` frames."""

    target: int          # what the user actually wants out
    generate: int        # what to hand the model
    pad: int             # frames added to reach it
    grid: Grid
    already_fits: bool

    @property
    def wasted(self) -> int:
        """Frames the model renders that are thrown away. Worth seeing: at
        17n+5 an unlucky target can cost 16 frames of render time."""
        return self.pad


def plan_fit(want: int, grid: Grid) -> FitPlan:
    """Pad up to the next count the model accepts, and remember the target.

    Padding UP rather than down, always: rounding down silently delivers
    fewer frames than were asked for, and a clip that is short is a clip that
    has to be re-rendered.
    """
    want = int(want)
    if want < 1:
        raise FrameGridError(
            f"Asking for {want} frames does not mean anything. Set the target "
            "to the number of frames you want out.")
    gen = grid.next_at_or_above(want)
    return FitPlan(target=want, generate=gen, pad=gen - want, grid=grid,
                   already_fits=(gen == want))


def trim_to(n_available: int, target: int) -> int:
    """How many frames to remove to land on `target`.

    Takes what ARRIVED, not what was expected. A sampler that returns one
    frame more or fewer than asked is common, and removing a remembered
    count from an unexpected length is how a clip ends up one frame short
    with nothing to explain it.
    """
    if target < 1:
        raise FrameGridError(f"A target of {target} frames is not a length.")
    if n_available < target:
        raise FrameGridError(
            f"Only {n_available} frames came back but {target} were asked for. "
            "Trimming cannot add frames — the generation returned short, so "
            "look upstream rather than here.")
    return n_available - target


def describe_fit(plan: FitPlan, model: str) -> str:
    """The plan in the terms someone would check it in."""
    g = plan.grid
    if plan.already_fits:
        lines = [f"{plan.target} frames is already a length {model} accepts, "
                 "so nothing was padded."]
    else:
        lines = [
            f"{model} accepts {g.step}n+{g.plus} frames, and {plan.target} is "
            f"not one of those. Generating {plan.generate} and trimming back "
            f"to {plan.target}.",
            f"{plan.pad} extra frame(s) get rendered and thrown away — about "
            f"{plan.pad / max(1, plan.generate) * 100:.0f}% of the render.",
        ]
    if g.note:
        lines.append(g.note)
    lines.append(
        "The trim runs to the TARGET, not by a count: a sampler that returns "
        "one frame more or fewer than asked is common, and subtracting a "
        "remembered number from an unexpected length is how a clip ends up "
        "short with nothing to explain it.")
    return "\n".join(lines)


def seconds_to_frames(seconds: float, grid: Grid, fps: float = 0.0) -> int:
    """A duration in seconds as a count this model accepts.

    Same three steps Pixaroma's duration helper uses — round, floor, snap —
    so a number worked out there means the same thing here.
    """
    if fps and float(fps) < 0:
        # Not treated as "use the default": a negative rate is a typo, and
        # quietly substituting the model's rate hides it until the clip comes
        # out the wrong length.
        raise FrameGridError(
            f"fps must be positive, got {fps}. Leave it at 0 to use the "
            f"model's own rate ({grid.fps:g}).")
    rate = float(fps) if fps and fps > 0 else grid.fps
    if rate <= 0:
        raise FrameGridError(f"fps must be positive, got {rate}.")
    n = max(grid.min_frames, int(round(float(seconds) * rate)))
    return grid.next_at_or_above(n)
