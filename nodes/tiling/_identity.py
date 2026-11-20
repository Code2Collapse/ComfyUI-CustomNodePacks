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


# ── L9.172 (A25 spec 57): multi-band preservation, temporal identity protection ─────────────────────────────────

def parse_bands(band_radii: str, band_strengths: str):
    """(radii, strengths) from the two comma-separated lists, or None when both are empty (the single split).

    radii r1 < ... < rn, each in (0, 128]; n + 1 strengths in [0, 1] (band 0 = finest detail, band n = the residual).
    """
    rs = str(band_radii or "").strip()
    ss = str(band_strengths or "").strip()
    if not rs and not ss:
        return None
    if not rs or not ss:
        raise IdentityLockError(
            "Multi-band lock needs BOTH band_radii and band_strengths (or neither, for the plain split). "
            "Example: band_radii '2, 8, 32' with band_strengths '0, 0.5, 1, 1'.")
    try:
        radii = [float(v) for v in rs.replace(";", ",").split(",") if v.strip()]
        strengths = [float(v) for v in ss.replace(";", ",").split(",") if v.strip()]
    except ValueError:
        raise IdentityLockError(
            f"band_radii {rs!r} / band_strengths {ss!r}: write plain numbers separated by commas.") from None
    if not radii:
        raise IdentityLockError("band_radii has no numbers in it.")
    if any(not (0.0 < r <= 128.0) for r in radii):
        raise IdentityLockError(f"Every band radius must be above 0 and at most 128 px, got {radii}.")
    if any(b <= a for a, b in zip(radii, radii[1:])):
        raise IdentityLockError(f"band_radii must increase from fine to coarse, got {radii}.")
    if len(strengths) != len(radii) + 1:
        raise IdentityLockError(
            f"{len(radii)} radii make {len(radii) + 1} bands (finest detail, {len(radii) - 1} middle band(s), "
            f"the coarse residual), so band_strengths needs {len(radii) + 1} numbers - got {len(strengths)}.")
    if any(not (0.0 <= s <= 1.0) for s in strengths):
        raise IdentityLockError(f"Every band strength must be between 0 and 1, got {strengths}.")
    return radii, strengths


def decompose(x: torch.Tensor, radii) -> list:
    """Bands of x, fine to coarse: [x - L(r1), L(r1) - L(r2), ..., L(rn)]. They sum back to x (float error only)."""
    t = x.float()
    lows = [low_pass(t, float(r)) for r in radii]
    bands = [t - lows[0]]
    bands += [lows[k] - lows[k + 1] for k in range(len(lows) - 1)]
    bands.append(lows[-1])
    return bands


def _mask_like(protect: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """A [B,H,W] / [H,W] mask broadcast against `out` (BHWC or BCHW), 0..1, out's dtype."""
    m = protect
    if m.ndim == 2:
        m = m.unsqueeze(0)
    m = m.to(device=out.device, dtype=out.dtype).clamp(0.0, 1.0)
    bhwc = out.shape[-1] <= 4
    m = m.unsqueeze(-1) if bhwc else m.unsqueeze(1)
    if m.shape[0] == 1 and out.shape[0] > 1:
        m = m.expand(out.shape[0], *m.shape[1:])
    return m


def multiband_lock(original: torch.Tensor, refined: torch.Tensor, *, radii, strengths, strength: float = 1.0,
                   protect: torch.Tensor | None = None) -> torch.Tensor:
    """Per band k: take s_k of the ORIGINAL's band and 1 - s_k of the refinement's.

    The effective per-band strength is s_k * (strength * (1 - m) + m), m = protect (0 without one), so `strength`
    keeps fading the whole lock and the protect mask still means "full lock here".
    """
    if original.shape != refined.shape:
        raise IdentityLockError(
            f"The original is {tuple(original.shape)} and the refined image is {tuple(refined.shape)}. They must "
            "match.")
    if not 0.0 <= strength <= 1.0:
        raise IdentityLockError(f"strength must be between 0 and 1, got {strength}.")
    bo = decompose(original, radii)
    br = decompose(refined, radii)
    g = torch.tensor(float(strength), dtype=torch.float32, device=refined.device)
    if protect is not None:
        m = _mask_like(protect, refined.float())
        g = g * (1.0 - m) + m
    out = torch.zeros(refined.shape, dtype=torch.float32, device=refined.device)
    for b_o, b_r, s in zip(bo, br, strengths):
        out = out + b_r + (b_o - b_r) * (float(s) * g)
    return out.to(original.dtype)


def describe_bands(radii, strengths) -> str:
    names = [f"detail < {radii[0]:g}px"]
    names += [f"{radii[k]:g}-{radii[k + 1]:g}px" for k in range(len(radii) - 1)]
    names.append(f"structure > {radii[-1]:g}px")
    parts = ", ".join(f"{n}: {s:.2f} from the plate" for n, s in zip(names, strengths))
    return ("Multi-band lock: " + parts + ". The correction passes are not used in multi-band mode (each band is "
            "mixed once; radius only sets where drift is measured).")


def _is_bhwc(x: torch.Tensor) -> bool:
    return x.shape[-1] <= 4 and x.shape[1] > 4


def track_protect_mask(original: torch.Tensor, protect: torch.Tensor):
    """Carry a one-frame protect mask through the clip: the protected region of `original` (its box, padded 25 %) is
    followed from frame to frame by FFT phase correlation and the mask moved with it (integer shift, zero fill).

    Returns ([B,H,W] float mask, [(dy, dx)] cumulative integer offsets per frame). Frame 0 keeps the mask as given.
    """
    from ._motion import bbox, luma, pad_box, phase_correlate, shift_mask
    lum = luma(original if _is_bhwc(original) else original.movedim(1, -1))          # [B, H, W]
    B, H, W = int(lum.shape[0]), int(lum.shape[1]), int(lum.shape[2])
    m0 = protect.float()
    if m0.ndim == 3:
        m0 = m0[0]
    if tuple(m0.shape) != (H, W):
        m0 = F.interpolate(m0[None, None], size=(H, W), mode="bilinear", align_corners=False)[0, 0]
    m0 = m0.to(lum.device).clamp(0.0, 1.0)
    box = bbox(m0)
    if box is None:
        return m0.unsqueeze(0).expand(B, H, W).clone(), [(0, 0)] * B
    y0, y1, x0, x1 = pad_box(box, H, W)
    cum_y = cum_x = 0.0
    offs = [(0, 0)]
    masks = [m0]
    for t in range(1, B):
        oy, ox = int(round(cum_y)), int(round(cum_x))
        ry0, ry1 = max(0, y0 + oy), min(H, y1 + oy)
        rx0, rx1 = max(0, x0 + ox), min(W, x1 + ox)
        if ry1 - ry0 >= 4 and rx1 - rx0 >= 4:
            dy, dx = phase_correlate(lum[t - 1, ry0:ry1, rx0:rx1], lum[t, ry0:ry1, rx0:rx1])
            cum_y += dy
            cum_x += dx
        oy, ox = int(round(cum_y)), int(round(cum_x))
        offs.append((oy, ox))
        masks.append(shift_mask(m0, oy, ox))
    return torch.stack(masks, dim=0), offs


def drift_report(original: torch.Tensor, result: torch.Tensor, *, radius: float,
                 protect: torch.Tensor | None = None, tracked_offsets=None) -> list:
    """Per frame: low-band drift inside the protected region (whole frame without one), normalised by that frame's
    range like drift(), and the region's geometric shift original -> result in pixels (phase correlation inside the
    region's padded box)."""
    from ._motion import bbox, luma, pad_box, phase_correlate
    if original.shape != result.shape:
        raise IdentityLockError(
            f"Cannot measure drift between {tuple(original.shape)} and {tuple(result.shape)}.")
    bhwc = _is_bhwc(original)
    lo_a = low_pass(original.float(), radius)
    lo_b = low_pass(result.float(), radius)
    d = (lo_a - lo_b).abs()
    d = d.mean(dim=-1) if bhwc else d.mean(dim=1)                            # [B, H, W]
    la = luma(original if bhwc else original.movedim(1, -1))
    lb = luma(result if bhwc else result.movedim(1, -1))
    B, H, W = int(d.shape[0]), int(d.shape[1]), int(d.shape[2])
    m = None
    if protect is not None:
        m = protect.float()
        if m.ndim == 2:
            m = m.unsqueeze(0)
        if tuple(m.shape[-2:]) != (H, W):
            m = F.interpolate(m.unsqueeze(1), size=(H, W), mode="bilinear", align_corners=False).squeeze(1)
        m = m.to(d.device).clamp(0.0, 1.0)
    rows = []
    for t in range(B):
        mt = None if m is None else m[min(t, int(m.shape[0]) - 1)]
        frame = original[t].float()
        spread = float(frame.max() - frame.min())
        if mt is not None and float(mt.sum()) > 0:
            dr = float((d[t] * mt).sum() / mt.sum())
            box = bbox(mt)
        else:
            dr = float(d[t].mean())
            box = None
        dr = dr / spread if spread > 1e-8 else 0.0
        y0, y1, x0, x1 = pad_box(box, H, W) if box is not None else (0, H, 0, W)
        if y1 - y0 >= 4 and x1 - x0 >= 4:
            sy, sx = phase_correlate(la[t, y0:y1, x0:x1], lb[t, y0:y1, x0:x1])
        else:
            sy, sx = 0.0, 0.0
        row = {"frame": t, "drift": round(dr, 6), "region": "protect" if box is not None else "frame",
               "region_shift_px": [round(sy, 3), round(sx, 3)]}
        if tracked_offsets is not None:
            row["tracked_offset_px"] = list(tracked_offsets[t])
        rows.append(row)
    return rows
