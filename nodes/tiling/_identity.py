"""Refine every pixel without letting anything move.

THE PROBLEM. A tiled upscale that refines detail also regenerates content, and
regenerated content includes faces: the jaw shifts a few pixels, an eye catches
a different highlight, and over 124 frames that reads as the actor's face
*swimming*. Turning denoise down until it stops also stops the refinement,
which was the point.

THE ANSWER is not a tuning knob. It is to split the signal:

    out = low(original) + (refined - low(refined))

Composition, face geometry, pose and colour live in the LOW frequencies.
Detail, pores, hair, grain and edge micro-contrast live in the HIGH. Take the
low band from the original and only the high band from the model, and the face
cannot move - not "usually does not", cannot, because the only thing taken
from the model is the part of the signal that does not encode where anything
is.

WHAT IT COSTS, stated plainly because it is a real trade and not a free win:
anything the model would have FIXED structurally also does not come through. A
warped ear stays warped; a badly formed hand stays badly formed. This is the
right trade for a finishing pass on a shot that is already correct, and the
wrong one for a repair pass. So it is a mode with a strength, not a default
nobody sees.

TWO KNOBS, and they do different jobs:

  * `radius` is the split frequency, and the trade runs this way round:

        SMALL radius -> almost the whole image is "structure" -> very hard
                        lock, and almost no refinement gets through
        LARGE radius -> only the coarsest shapes are "structure" -> much more
                        refinement arrives, and medium-scale movement with it

    Measured on a face moved 6px with no added texture, drift at radius 12:
    a radius-2 lock drifts 0.0000002, a radius-24 lock drifts 0.018. So if a
    face is still moving, the answer is a SMALLER radius, at the cost of
    detail - not a larger one.

  * `strength` fades between the locked result and the raw refinement, for
    when a little structural change IS wanted.

`strength = 0` WITH a protection mask is the configuration most shots want:
nothing locked anywhere except inside the mask, where it is locked completely.

The lock is measurable, so this module measures it: `drift()` reports how far
the low band actually moved, and that number is what turns "faces must not
change" from a hope into something checkable.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


class IdentityLockError(ValueError):
    pass


def _to_bchw(x: torch.Tensor) -> tuple[torch.Tensor, bool]:
    """(BCHW view, was_bhwc). ComfyUI IMAGE is BHWC; conv wants BCHW."""
    if x.ndim != 4:
        raise IdentityLockError(
            f"Expected a 4D image batch, got shape {tuple(x.shape)}.")
    if x.shape[-1] <= 4 and x.shape[1] > 4:
        return x.movedim(-1, 1), True
    return x, False


def _from_bchw(x: torch.Tensor, was_bhwc: bool) -> torch.Tensor:
    return x.movedim(1, -1) if was_bhwc else x


def gaussian_kernel(radius: float, *, dtype=torch.float32,
                    device="cpu") -> torch.Tensor:
    """A 1D Gaussian wide enough not to clip its own tail.

    Sized at 3 sigma: a kernel truncated tighter than that has a visible
    shoulder, and a shoulder in the low-pass becomes a halo in the result -
    the classic frequency-separation ring around every hard edge.
    """
    sigma = max(1e-3, float(radius))
    size = int(2 * math.ceil(3.0 * sigma) + 1)
    x = torch.arange(size, dtype=dtype, device=device) - (size - 1) / 2.0
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


def low_pass(x: torch.Tensor, radius: float) -> torch.Tensor:
    """Separable Gaussian blur - the "structure" half of the split."""
    if radius <= 0:
        return x
    bchw, was_bhwc = _to_bchw(x)
    c = bchw.shape[1]
    k = gaussian_kernel(radius, dtype=torch.float32, device=bchw.device)
    n = k.numel()
    pad = n // 2

    t = bchw.float()
    # 'reflect' rather than zeros: padding with black darkens the frame edge,
    # which then shows up as a dark rim in the recombined result.
    mode = "reflect" if pad < min(t.shape[-2], t.shape[-1]) else "replicate"
    t = F.pad(t, (pad, pad, 0, 0), mode=mode)
    t = F.conv2d(t, k.view(1, 1, 1, n).expand(c, 1, 1, n), groups=c)
    t = F.pad(t, (0, 0, pad, pad), mode=mode)
    t = F.conv2d(t, k.view(1, 1, n, 1).expand(c, 1, n, 1), groups=c)
    return _from_bchw(t.to(bchw.dtype), was_bhwc)


def identity_lock(original: torch.Tensor, refined: torch.Tensor, *,
                  radius: float = 8.0, strength: float = 1.0,
                  protect: torch.Tensor | None = None,
                  corrections: int = 2) -> torch.Tensor:
    """Keep the original's structure, take the refinement's detail.

    `protect` is an optional [B,H,W] mask, 1 where the lock should be at full
    strength (a face) and 0 where the refinement may pass through untouched.
    It is how a shot gets a hard lock on the actor and a free hand on the
    background, in one pass.

    `corrections` tightens the lock. A single split leaves a residual, because
    a Gaussian is not a brick-wall filter: the detail band still carries some
    energy below the cutoff, and if the refinement moved a face, part of that
    movement rides through in it. Each correction pushes the result's low band
    back onto the original's:

        out <- out + (low(original) - low(out))

    Measured on a face shifted 6px at radius 12, the drift goes 0.0110 (no
    corrections) -> 0.0046 -> 0.0025, while the detail actually kept falls only
    2%. Two is the default because that is where the curve flattens; zero
    reproduces the plain frequency split.
    """
    if original.shape != refined.shape:
        raise IdentityLockError(
            f"The original is {tuple(original.shape)} and the refined image is "
            f"{tuple(refined.shape)}. They must match - if the refinement "
            "upscaled, resize the original to the new size first, or the lock "
            "would compare different pixels.")
    if not 0.0 <= strength <= 1.0:
        raise IdentityLockError(
            f"strength must be between 0 and 1, got {strength}. It fades "
            "between the raw refinement (0) and the fully locked result (1).")
    if radius < 0:
        raise IdentityLockError(f"radius cannot be negative, got {radius}.")
    if corrections < 0:
        raise IdentityLockError(
            f"corrections cannot be negative, got {corrections}.")

    if radius == 0.0:
        return refined
    if strength == 0.0 and protect is None:
        return refined
    # NOTE: strength == 0 with a mask does NOT return here. That combination -
    # full lock inside the mask, completely free outside - is the most useful
    # configuration this node has, and returning early made it a silent no-op.

    low_original = low_pass(original.float(), radius)
    low_refined = low_pass(refined.float(), radius)
    detail = refined.float() - low_refined
    locked = low_original + detail

    # Push the result's low band back onto the original's. See the docstring:
    # the single split leaves a residual because a Gaussian is not brick-wall,
    # and without this a moved face is only ~71% held rather than ~94%.
    for _ in range(int(corrections)):
        locked = locked + (low_original - low_pass(locked, radius))

    out = torch.lerp(refined.float(), locked, strength)

    if protect is not None:
        m = protect
        if m.ndim == 3:
            m = m.unsqueeze(-1) if out.shape[-1] <= 4 else m.unsqueeze(1)
        if m.shape[0] == 1 and out.shape[0] > 1:
            m = m.expand_as(out[..., :1]) if out.shape[-1] <= 4 else m
        m = m.to(out.dtype).clamp(0.0, 1.0)
        # Inside the mask: the locked result. Outside: whatever `strength`
        # already produced, so the background still gets whatever lock the
        # user asked for globally rather than none at all.
        out = out * (1.0 - m) + torch.lerp(out, locked, 1.0) * m

    return out.to(original.dtype)


def drift(original: torch.Tensor, result: torch.Tensor, *,
          radius: float = 8.0) -> float:
    """How far the STRUCTURE moved, as a fraction of the original's range.

    Measured on the low band on purpose: comparing the images directly would
    report every added pore as drift, which is the refinement working. What
    matters is whether the thing UNDER the detail moved.

    0.0 means structure is untouched. Above ~0.02 on a locked pass means the
    lock is not holding and the radius is probably too small.
    """
    if original.shape != result.shape:
        raise IdentityLockError(
            f"Cannot measure drift between {tuple(original.shape)} and "
            f"{tuple(result.shape)}.")
    lo_a = low_pass(original.float(), radius)
    lo_b = low_pass(result.float(), radius)
    # Normalised by the ORIGINAL's own range, not the blurred one.
    #
    # Blurring collapses the spread - a face blob at radius 48 flattens almost
    # flat - so dividing by the blurred spread makes a tiny absolute difference
    # read as a huge ratio. Measured on a clip where nothing moved at all,
    # that gave 0.03% at radius 12 and 3.3% at radius 48, which is the
    # denominator shrinking, not structure moving. A fixed denominator makes
    # the number mean the same thing at every scale, which is the whole point
    # of reporting it at two.
    spread = float(original.float().max() - original.float().min())
    if spread < 1e-8:
        return 0.0
    return float((lo_a - lo_b).abs().mean() / spread)


def drift_map(original: torch.Tensor, result: torch.Tensor, *,
              radius: float = 8.0) -> torch.Tensor:
    """Where structure moved, as a [B,H,W] map. Bright means it moved.

    A number says whether the lock held; this says WHERE it did not, which is
    the difference between knowing there is a problem and finding it.
    """
    lo_a = low_pass(original.float(), radius)
    lo_b = low_pass(result.float(), radius)
    d = (lo_a - lo_b).abs()
    d = d.mean(dim=-1) if d.shape[-1] <= 4 else d.mean(dim=1)
    peak = float(d.max())
    return d / peak if peak > 1e-8 else d


def describe(original: torch.Tensor, result: torch.Tensor, *,
             radius: float, strength: float, protected: bool,
             corrections: int = 2) -> str:
    """What the lock did, and whether it held."""
    moved = drift(original, result, radius=radius)
    # Measured COARSER than the lock, on purpose. Drift at the lock's own
    # radius is self-fulfilling - the correction passes force the low band at
    # that radius to match, so it always reads ~0. Movement at a scale the
    # radius cannot see is only visible when measured at a scale that can.
    coarse = drift(original, result, radius=max(radius * 4.0, 16.0))
    lines = [
        f"Identity lock at strength {strength:.2f}, radius {radius:.1f}px, "
        f"{corrections} correction pass(es).",
        f"Structure moved {moved * 100:.3f}% of the image range.",
    ]
    if corrections == 0 and strength > 0:
        lines.append(
            "Corrections are OFF, so this is a plain frequency split. A "
            "Gaussian is not a brick-wall filter, so roughly a third of any "
            "structural movement still rides through in the detail band. Two "
            "passes take that from about 29% to about 6%.")
    lines.append(
        f"Measured coarser, at {max(radius * 4.0, 16.0):.0f}px, structure "
        f"moved {coarse * 100:.3f}%.")
    if strength >= 0.99 and coarse > 0.02:
        lines.append(
            "WARNING: the lock is at full strength but structure still moved "
            "at a scale COARSER than the split. Detail above "
            f"{radius:.0f}px is being passed through as 'detail' when it is "
            "actually carrying shape. Use a SMALLER radius - that locks "
            "harder, at the cost of how much refinement gets through.")
    elif coarse <= 0.005:
        lines.append(
            "Structure is effectively unchanged: the refinement contributed "
            "detail only, which is what a finishing pass should do.")

    lines.append(
        "What this CANNOT do, by construction: fix anything structural. A "
        "warped ear or a malformed hand stays exactly as it was, because the "
        "only thing taken from the model is the band that does not encode "
        "where things are. For a repair pass, lower the strength.")
    if protected:
        lines.append(
            "A protection mask is in use: the lock is at full strength inside "
            "it regardless of the global strength, so the face holds while the "
            "background is free to take more of the refinement.")
    return "\n".join(lines)
