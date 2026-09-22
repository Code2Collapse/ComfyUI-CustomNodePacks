"""Tile a frame, refine the tiles, put it back with no seam.

Written fresh against the published signal processing rather than adapted from
any pack: complementary raised-cosine windows are textbook, and this file has
to live in an Apache-licensed pack. The GPL implementation in
`third_party/comfyui-deno-custom-nodes/deno_ltx_tiling.py` was read as a
reference for the LTX-specific adapter only, which lands in a GPL pack.

THE THREE WAYS TILED VIDEO GOES WRONG, and what is done about each.

1. VISIBLE SEAMS, from windows that do not sum to 1 across an overlap.

   The fix is not "a Hann window" - a Hann window over the whole tile does NOT
   sum to 1 at 50% overlap, it sums to a ripple. What sums to 1 is a
   COMPLEMENTARY pair over the overlap region itself:

       rise(i) = ½(1 − cos(πi/L))
       fall(i) = ½(1 + cos(πi/L))
       rise(i) + fall(i) ≡ 1     for every i, exactly

   The tile entering the overlap fades out with `fall`; the tile leaving it
   fades in with `rise`; they are the same length over the same pixels, so the
   sum is exactly one everywhere and there is no seam and no brightness dip.
   A test pins that identity rather than trusting the algebra.

2. BRIGHTNESS BANDING, from blending in fp16. The accumulator is float32 and
   the result is divided by the accumulated weight, so even a plan whose
   windows do not quite sum to 1 - at the frame edges, where there is no
   neighbour to complement - comes out at the right level.

3. CRAWLING SEAMS, and this is the video-specific one.

   A seam that is static disappears into the picture. A seam that MOVES is the
   most distracting artefact in the frame - and it moves whenever the tile
   plan is recomputed per frame, because a different rounding gives a
   different boundary. So a plan is built ONCE for a clip and every frame uses
   it. That is why `TilePlan` is a value that gets passed around rather than
   something each node works out for itself.

Works on pixels or latents, stills or clips: a plan is pure geometry and does
not know or care what is in the tensor.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

BLEND_MODES = ("cosine", "linear", "none")


class TilingError(ValueError):
    """Raised where a silently wrong reconstruction would otherwise happen."""


@dataclass(frozen=True)
class Tile:
    """One tile's placement, and how far it fades on each side.

    The fades are stored per SIDE rather than as one overlap number because a
    tile at the frame edge has no neighbour there and must not fade into
    nothing - it has to carry full weight to the border or the frame darkens
    at the rim.
    """

    row: int
    col: int
    top: int
    left: int
    height: int
    width: int
    fade_top: int
    fade_bottom: int
    fade_left: int
    fade_right: int

    @property
    def bottom(self) -> int:
        return self.top + self.height

    @property
    def right(self) -> int:
        return self.left + self.width


@dataclass(frozen=True)
class TilePlan:
    """Every tile for one frame size. Built once, reused for every frame."""

    height: int
    width: int
    tiles: tuple[Tile, ...]
    rows: int
    cols: int
    overlap: int
    blend: str

    def __len__(self) -> int:
        return len(self.tiles)

    @property
    def redundancy(self) -> float:
        """How much more area is processed than the frame contains.

        0.0 means the tiles exactly cover it; 0.5 means half again as much
        work. Overlap is not free and the number should be visible before a
        render rather than inferred from the clock afterwards.
        """
        covered = sum(t.height * t.width for t in self.tiles)
        return covered / float(self.height * self.width) - 1.0


# ── the windows ─────────────────────────────────────────────────────────────

def rise(length: int, *, dtype=torch.float32, device="cpu") -> torch.Tensor:
    """Fade in: ½(1 − cos(πi/L)). Complements `fall` exactly."""
    if length <= 0:
        return torch.empty(0, dtype=dtype, device=device)
    i = torch.arange(length, dtype=dtype, device=device)
    return 0.5 * (1.0 - torch.cos(math.pi * i / length))


def fall(length: int, *, dtype=torch.float32, device="cpu") -> torch.Tensor:
    """Fade out: ½(1 + cos(πi/L)). Complements `rise` exactly."""
    if length <= 0:
        return torch.empty(0, dtype=dtype, device=device)
    i = torch.arange(length, dtype=dtype, device=device)
    return 0.5 * (1.0 + torch.cos(math.pi * i / length))


def window_1d(size: int, fade_before: int, fade_after: int, *,
              blend: str = "cosine", dtype=torch.float32,
              device="cpu") -> torch.Tensor:
    """Weight along one axis: fade in, flat, fade out."""
    if size < 1:
        raise TilingError(f"A tile cannot be {size} wide.")
    if blend not in BLEND_MODES:
        raise TilingError(
            f"Unknown blend mode {blend!r}. Use one of: {', '.join(BLEND_MODES)}.")

    fade_before = max(0, min(int(fade_before), size))
    fade_after = max(0, min(int(fade_after), size))
    w = torch.ones(size, dtype=dtype, device=device)
    if blend == "none":
        return w

    if blend == "linear":
        # Also complementary: t + (1-t) = 1. Sharper than cosine at the ends,
        # which shows as a faint crease on a gradient - offered because it is
        # marginally cheaper and some footage does not care.
        if fade_before:
            w[:fade_before] = torch.linspace(
                0.0, 1.0, fade_before + 1, dtype=dtype, device=device)[:-1]
        if fade_after:
            w[size - fade_after:] = torch.linspace(
                1.0, 0.0, fade_after + 1, dtype=dtype, device=device)[:-1]
        return w

    if fade_before:
        w[:fade_before] = rise(fade_before, dtype=dtype, device=device)
    if fade_after:
        w[size - fade_after:] = fall(fade_after, dtype=dtype, device=device)
    return w


def window_2d(tile: Tile, *, blend: str = "cosine", dtype=torch.float32,
              device="cpu") -> torch.Tensor:
    """The tile's weight map, as the outer product of its two axes."""
    v = window_1d(tile.height, tile.fade_top, tile.fade_bottom,
                  blend=blend, dtype=dtype, device=device)
    h = window_1d(tile.width, tile.fade_left, tile.fade_right,
                  blend=blend, dtype=dtype, device=device)
    return v[:, None] * h[None, :]


# ── the plan ────────────────────────────────────────────────────────────────

def _starts(total: int, count: int, overlap: int) -> tuple[list[int], int]:
    """Where each tile starts along one axis, and how long they are.

    Tiles are equal-sized and evenly spaced, with the last one ending exactly
    at the frame edge. Equal sizes matter: a short final tile is a different
    denoise problem from its neighbours and shows up as a band.
    """
    if count < 1:
        raise TilingError(f"A plan needs at least one tile, got {count}.")
    if count == 1:
        return [0], total

    # size chosen so that `count` tiles overlapping by `overlap` span `total`
    size = math.ceil((total + overlap * (count - 1)) / count)
    size = min(size, total)

    # An overlap at least as wide as the frame. The algebra makes this the only
    # way the tile size can come out no larger than the overlap:
    #   size = ceil((total + ov(n-1))/n) <= ov  <=>  total <= ov
    if size <= overlap:
        raise TilingError(
            f"A {overlap}px overlap is as wide as the {total}px frame, so every "
            f"tile would be {size}px - no larger than the overlap itself, and "
            "neighbouring tiles would cover the same pixels twice with nothing "
            "between them. Use a smaller overlap.")

    stride = (total - size) / float(count - 1)

    starts = [int(round(i * stride)) for i in range(count)]
    starts[-1] = total - size          # exact, not rounded, at the edge
    return starts, size


def build_plan(height: int, width: int, *, rows: int = 2, cols: int = 2,
               overlap: int = 64, blend: str = "cosine") -> TilePlan:
    """One plan for one frame size. Build it once per clip."""
    if height < 1 or width < 1:
        raise TilingError(f"Frame size must be positive, got {width}x{height}.")
    if overlap < 0:
        raise TilingError(f"Overlap cannot be negative, got {overlap}.")
    if blend not in BLEND_MODES:
        raise TilingError(
            f"Unknown blend mode {blend!r}. Use one of: {', '.join(BLEND_MODES)}.")

    row_starts, tile_h = _starts(height, rows, overlap)
    col_starts, tile_w = _starts(width, cols, overlap)

    tiles: list[Tile] = []
    for r, top in enumerate(row_starts):
        for c, left in enumerate(col_starts):
            # Fade only where there IS a neighbour. A tile at the frame edge
            # that fades there has no partner to make up the weight, so the
            # rim of the frame goes dark - the classic tiled-upscale vignette.
            fade_top = min(overlap, top) if r > 0 else 0
            fade_left = min(overlap, left) if c > 0 else 0
            fade_bottom = min(overlap, height - (top + tile_h)) if r < rows - 1 else 0
            fade_right = min(overlap, width - (left + tile_w)) if c < cols - 1 else 0
            # The real overlap with the next tile, which is what the fade must
            # match: two tiles whose fades differ in length do not sum to 1.
            if r < rows - 1:
                fade_bottom = max(0, (top + tile_h) - row_starts[r + 1])
            if c < cols - 1:
                fade_right = max(0, (left + tile_w) - col_starts[c + 1])
            if r > 0:
                fade_top = max(0, (row_starts[r - 1] + tile_h) - top)
            if c > 0:
                fade_left = max(0, (col_starts[c - 1] + tile_w) - left)

            tiles.append(Tile(
                row=r, col=c, top=top, left=left, height=tile_h, width=tile_w,
                fade_top=fade_top, fade_bottom=fade_bottom,
                fade_left=fade_left, fade_right=fade_right))

    plan = TilePlan(height=height, width=width, tiles=tuple(tiles),
                    rows=rows, cols=cols, overlap=overlap, blend=blend)
    validate_plan(plan)
    return plan


def plan_for_tile_size(height: int, width: int, *, tile: int = 1024,
                       overlap: int = 64, blend: str = "cosine") -> TilePlan:
    """A plan expressed as a tile SIZE rather than a tile count.

    `tile` is a MAXIMUM, not a target. Tiles are equal-sized and must tile the
    frame exactly, so the size that comes out is the largest one at or below
    `tile` that does: asking for 1024px across 2048px with 64px overlap gives
    three 726px tiles, because two would have to be 1056px and that is over
    the limit. The report states the size actually chosen, so it is not left
    to be assumed.
    """
    if tile <= overlap:
        raise TilingError(
            f"A {tile}px tile with {overlap}px overlap leaves "
            f"{tile - overlap}px of new picture per tile, which is not a tiling "
            "so much as a very slow blur. Make the tile larger or the overlap "
            "smaller.")
    rows = max(1, math.ceil((height - overlap) / float(tile - overlap)))
    cols = max(1, math.ceil((width - overlap) / float(tile - overlap)))
    return build_plan(height, width, rows=rows, cols=cols,
                      overlap=overlap, blend=blend)


def validate_plan(plan: TilePlan) -> None:
    """Refuse a plan that would leave a hole.

    A hole is not a crash - it is a black rectangle in the middle of a
    finished shot, found on playback.
    """
    if not plan.tiles:
        raise TilingError("The plan has no tiles in it.")
    covered = torch.zeros(plan.height, plan.width, dtype=torch.bool)
    for t in plan.tiles:
        if t.top < 0 or t.left < 0 or t.bottom > plan.height or t.right > plan.width:
            raise TilingError(
                f"Tile r{t.row}c{t.col} runs outside the frame: "
                f"({t.left},{t.top})-({t.right},{t.bottom}) in "
                f"{plan.width}x{plan.height}.")
        covered[t.top:t.bottom, t.left:t.right] = True
    if not bool(covered.all()):
        missing = int((~covered).sum())
        raise TilingError(
            f"{missing} pixel(s) are in no tile, which would come out black. "
            "This is a bug in the plan, not a setting - report the frame size, "
            f"{plan.rows}x{plan.cols} tiles and {plan.overlap}px overlap.")


def describe_plan(plan: TilePlan) -> str:
    """The plan in the terms that decide whether it is a good one."""
    t = plan.tiles[0]
    lines = [
        f"{len(plan.tiles)} tiles ({plan.rows}x{plan.cols}), each "
        f"{t.width}x{t.height}, overlapping {plan.overlap}px, "
        f"{plan.blend} blend.",
        f"{plan.redundancy * 100:.0f}% more area is processed than the frame "
        "contains - that is what the overlap costs.",
        "This plan is built ONCE and used for every frame. A plan recomputed "
        "per frame gives slightly different boundaries each time, and a seam "
        "that moves is far more visible than one that does not.",
    ]
    if plan.overlap < 32 and plan.blend != "none":
        lines.append(
            f"WARNING: {plan.overlap}px is a narrow overlap. The blend has "
            "little room to hide a difference between tiles, so a seam may "
            "show wherever two tiles disagree. 64px or more is usual.")
    if plan.redundancy > 1.0:
        lines.append(
            "WARNING: more than twice the frame area is being processed. "
            "Fewer, larger tiles would do the same work in less time.")
    return "\n".join(lines)


# ── split and merge ─────────────────────────────────────────────────────────

def _spatial_dims(x: torch.Tensor) -> tuple[int, int]:
    """(height, width) for BHWC images and BCHW / BCTHW latents.

    Guessing wrong here tiles the channel axis, which produces a confident
    result that is nonsense, so the ambiguous case is refused.
    """
    if x.ndim == 4:
        # BHWC (ComfyUI IMAGE) has a small LAST axis; BCHW (LATENT) has a small
        # SECOND axis. When only one of them is small the layout is decided.
        last_small = x.shape[-1] <= 4
        second_small = x.shape[1] <= 4
        if last_small and not second_small:
            return int(x.shape[1]), int(x.shape[2])          # BHWC
        if second_small and not last_small:
            return int(x.shape[2]), int(x.shape[3])          # BCHW
        if not last_small and not second_small:
            # Neither axis looks like channels. A latent with many channels
            # (16, 32) lands here and is BCHW.
            if x.shape[1] <= 64:
                return int(x.shape[2]), int(x.shape[3])
            raise TilingError(
                f"Cannot tell whether a tensor of shape {tuple(x.shape)} is "
                "BHWC or BCHW - neither axis looks like a channel count. Pass "
                "a standard ComfyUI IMAGE [B,H,W,C] or LATENT [B,C,H,W].")
        # BOTH are small, e.g. (1, 3, 8, 3): equally readable as 3 rows of
        # 8x3-channel pixels or 3 channels of 8x3. Guessing tiles the channel
        # axis and returns confident nonsense, so it is refused instead.
        raise TilingError(
            f"Cannot tell whether a tensor of shape {tuple(x.shape)} is BHWC "
            "or BCHW: both axes are small enough to be channels. Pass a "
            "standard ComfyUI IMAGE [B,H,W,C] or LATENT [B,C,H,W].")
    if x.ndim == 5:
        return int(x.shape[-2]), int(x.shape[-1])
    raise TilingError(
        f"Expected a 4D image/latent or a 5D video latent, got shape "
        f"{tuple(x.shape)}.")


def _is_bhwc(x: torch.Tensor) -> bool:
    """Must agree with _spatial_dims, or crop() and merge() read the same
    tensor two different ways."""
    return x.ndim == 4 and x.shape[-1] <= 4 and x.shape[1] > 4


def crop(x: torch.Tensor, tile: Tile) -> torch.Tensor:
    """One tile out of a frame or clip, whatever the layout."""
    if _is_bhwc(x):
        return x[:, tile.top:tile.bottom, tile.left:tile.right, :]
    if x.ndim == 4:
        return x[:, :, tile.top:tile.bottom, tile.left:tile.right]
    return x[..., tile.top:tile.bottom, tile.left:tile.right]


def split(x: torch.Tensor, plan: TilePlan) -> list[torch.Tensor]:
    """Every tile of a frame or clip, in plan order."""
    h, w = _spatial_dims(x)
    if (h, w) != (plan.height, plan.width):
        raise TilingError(
            f"This plan is for {plan.width}x{plan.height} but the tensor is "
            f"{w}x{h}. A plan belongs to one frame size - build a new one, or "
            "you will tile the wrong region of every frame.")
    return [crop(x, t) for t in plan.tiles]


def merge(tiles: list[torch.Tensor], plan: TilePlan, *,
          scale: float = 1.0) -> torch.Tensor:
    """Put the tiles back, weighted, in float32.

    `scale` is for the upscaling case: the tiles come back larger than they
    went out, and the plan's coordinates scale with them. Passed explicitly
    rather than inferred, because inferring it from one tile's size silently
    rounds a non-integer ratio and shifts every tile after the first.
    """
    if len(tiles) != len(plan.tiles):
        raise TilingError(
            f"The plan has {len(plan.tiles)} tiles but {len(tiles)} came back. "
            "Every tile has to be returned, in plan order - a missing one "
            "leaves a black rectangle.")
    if scale <= 0:
        raise TilingError(f"Scale must be positive, got {scale}.")

    out_h = int(round(plan.height * scale))
    out_w = int(round(plan.width * scale))
    first = tiles[0]
    bhwc = _is_bhwc(first)

    # float32 throughout: accumulating weighted tiles in fp16 bands visibly,
    # and the division at the end is what rescues the frame edges where the
    # windows legitimately do not sum to 1.
    if bhwc:
        acc = torch.zeros((first.shape[0], out_h, out_w, first.shape[-1]),
                          dtype=torch.float32, device=first.device)
        wacc = torch.zeros((1, out_h, out_w, 1), dtype=torch.float32,
                           device=first.device)
    else:
        lead = first.shape[:-2]
        acc = torch.zeros((*lead, out_h, out_w), dtype=torch.float32,
                          device=first.device)
        wacc = torch.zeros((1,) * (len(lead)) + (out_h, out_w),
                           dtype=torch.float32, device=first.device)

    for tile, piece in zip(plan.tiles, tiles):
        top = int(round(tile.top * scale))
        left = int(round(tile.left * scale))
        ph, pw = _spatial_dims(piece)
        scaled = Tile(
            row=tile.row, col=tile.col, top=top, left=left,
            height=ph, width=pw,
            fade_top=int(round(tile.fade_top * scale)),
            fade_bottom=int(round(tile.fade_bottom * scale)),
            fade_left=int(round(tile.fade_left * scale)),
            fade_right=int(round(tile.fade_right * scale)),
        )
        if top + ph > out_h or left + pw > out_w:
            raise TilingError(
                f"Tile r{tile.row}c{tile.col} came back {pw}x{ph} and does not "
                f"fit at ({left},{top}) in {out_w}x{out_h}. The scale passed to "
                f"merge() ({scale}) does not match what the tiles actually did.")

        w2 = window_2d(scaled, blend=plan.blend, dtype=torch.float32,
                       device=first.device)
        if bhwc:
            acc[:, top:top + ph, left:left + pw, :] += \
                piece.float() * w2[None, :, :, None]
            wacc[:, top:top + ph, left:left + pw, :] += w2[None, :, :, None]
        else:
            shape = (1,) * (piece.ndim - 2) + w2.shape
            acc[..., top:top + ph, left:left + pw] += piece.float() * w2.view(shape)
            wacc[..., top:top + ph, left:left + pw] += w2.view(shape)

    # A zero weight would be a hole; validate_plan() makes that impossible, so
    # the clamp is belt and braces rather than a silent repair.
    return (acc / wacc.clamp(min=1e-8)).to(first.dtype)
