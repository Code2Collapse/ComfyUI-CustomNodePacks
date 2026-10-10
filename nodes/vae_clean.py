"""Diagnose and correct a decode that came back reddish, oversaturated or hard.

THE COMPLAINT this exists for: "the colours are more reddish or oversaturated
or too contrasty, and it is not the prompt - I think it is the model."

That is usually right, and it is usually not ONE fault. Four different things
produce that description and they need different fixes, so this measures
before it corrects and says which it found:

  1. CHANNEL IMBALANCE - the decoder's own bias. Most VAEs have a small,
     CONSTANT per-channel offset. It reads as a cast, it is the same on every
     frame, and it divides out exactly.

  2. OVERSATURATION - chroma scaled too far. On a diffusion model this is
     very often CFG rather than the decoder, and this node says so rather
     than quietly papering over a sampler setting.

  3. EXCESS CONTRAST - the black point crushed below zero and the white point
     pushed past one. Recoverable only where it has not clipped.

  4. CLIPPING - and this is the one that matters most, because it CANNOT be
     undone. A channel pinned at 1.0 has lost the values above it. Any tool
     that "fixes" a clipped highlight is inventing it. This measures how much
     is gone and refuses to pretend.

EVERYTHING HAPPENS IN LINEAR. Scaling chroma or contrast on sRGB values
applies the operation to a display curve rather than to light: pull saturation
down 20% in sRGB and the midtones shift as well, which is why a naive
saturation slider always seems to also change exposure. Convert, operate,
convert back.

WHAT IT WILL NOT DO. Grey-world balancing assumes the scene averages neutral.
A genuine sunset, a red-lit interior, a single-colour plate - all average
strongly non-neutral, and neutralising them destroys the shot. So grey-world
is offered but is not the default, and the report says when the measurement
looks like a real colour rather than a cast.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

CATEGORY = "C2C/Color"

# Rec.709 luma. Used for chroma/luma separation, so that a saturation change
# holds brightness instead of darkening the picture as it desaturates.
LUMA = (0.2126, 0.7152, 0.0722)

# A channel within this fraction of the others is not a cast worth correcting -
# it is measurement noise, and "correcting" it adds a cast of its own.
CAST_FLOOR = 0.005

# Above this fraction pinned at the extremes, the data is gone. Quoted in the
# report rather than silently worked around.
CLIP_NOTICE = 0.001

BALANCE_MODES = ("off", "grey world", "match reference", "white point")


class VAECleanError(ValueError):
    pass


# ── transfer ────────────────────────────────────────────────────────────────

def srgb_to_linear(x: torch.Tensor) -> torch.Tensor:
    """The real piecewise sRGB EOTF, not a 2.2 power.

    The linear toe below 0.04045 matters here: a 2.2 approximation shifts the
    darkest values, and the black point is exactly what this node measures.
    """
    return torch.where(x <= 0.04045, x / 12.92,
                       ((x.clamp(min=0.0) + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(min=0.0)
    return torch.where(x <= 0.0031308, x * 12.92,
                       1.055 * x ** (1.0 / 2.4) - 0.055)


def luma(lin: torch.Tensor) -> torch.Tensor:
    """Rec.709 luminance of a linear BHWC image, kept as [B,H,W,1]."""
    w = torch.tensor(LUMA, dtype=lin.dtype, device=lin.device)
    return (lin * w).sum(dim=-1, keepdim=True)


# ── measurement ─────────────────────────────────────────────────────────────

def channel_means(lin: torch.Tensor) -> tuple[float, float, float]:
    return tuple(float(lin[..., c].mean()) for c in range(3))


def clipping(srgb: torch.Tensor) -> dict:
    """How much is pinned at the ends, per channel. Unrecoverable either way."""
    out = {}
    for i, name in enumerate("rgb"):
        ch = srgb[..., i]
        out[name] = (float((ch <= 0.0005).float().mean()),
                     float((ch >= 0.9995).float().mean()))
    return out


def saturation(lin: torch.Tensor) -> float:
    """Mean chroma as a fraction of luma - scale-free, so exposure does not
    read as saturation."""
    y = luma(lin)
    chroma = (lin - y).abs().mean()
    return float(chroma / y.mean().clamp(min=1e-6))


def contrast(lin: torch.Tensor) -> tuple[float, float]:
    """(black point, white point) as robust percentiles of luma.

    Percentiles, not min/max: one hot pixel or one dead pixel would otherwise
    define the whole range and every correction would be driven by noise.
    """
    y = luma(lin).flatten()
    if y.numel() > 1_000_000:                 # a percentile does not need 8M samples
        y = y[torch.randperm(y.numel(), device=y.device)[:1_000_000]]
    lo = float(torch.quantile(y, 0.001))
    hi = float(torch.quantile(y, 0.999))
    return lo, hi


def decode_noise(lin: torch.Tensor) -> float:
    """High-frequency CHROMA energy - the signature of a VAE decode artifact.

    Measured on chroma alone because luma high-frequency is detail you want to
    keep; chroma high-frequency at this scale is almost always the decoder,
    not the picture.
    """
    y = luma(lin)
    chroma = lin - y
    bchw = chroma.movedim(-1, 1)
    k = torch.tensor([[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]],
                     dtype=bchw.dtype, device=bchw.device).view(1, 1, 3, 3)
    lap = F.conv2d(F.pad(bchw, (1, 1, 1, 1), mode="replicate"),
                   k.expand(3, 1, 3, 3), groups=3)
    return float(lap.abs().mean())


def measure(srgb: torch.Tensor) -> dict:
    lin = srgb_to_linear(srgb.float().clamp(0.0, 1.0))
    r, g, b = channel_means(lin)
    grey = (r + g + b) / 3.0
    lo, hi = contrast(lin)
    return {
        "means": (r, g, b),
        "grey": grey,
        "cast": ((r - grey) / max(grey, 1e-6),
                 (g - grey) / max(grey, 1e-6),
                 (b - grey) / max(grey, 1e-6)),
        "saturation": saturation(lin),
        "black": lo,
        "white": hi,
        "clip": clipping(srgb),
        "chroma_noise": decode_noise(lin),
    }


# ── correction, all in linear ───────────────────────────────────────────────

def balance(lin: torch.Tensor, mode: str, *, reference: torch.Tensor | None,
            strength: float) -> tuple[torch.Tensor, str]:
    """Per-channel gain. Returns the image and what it did."""
    if mode == "off" or strength <= 0:
        return lin, "Channel balance off."

    r, g, b = channel_means(lin)
    if mode == "grey world":
        target = (r + g + b) / 3.0
        want = (target, target, target)
        why = "grey world (assumes the scene averages neutral)"
    elif mode == "white point":
        # Anchor on the BRIGHTEST neutral-ish region rather than the average,
        # which is what survives a scene that is genuinely one colour.
        y = luma(lin)
        thr = torch.quantile(y.flatten()[:1_000_000], 0.99)
        m = (y >= thr).expand_as(lin)
        if m.any():
            # Measure the bright region's channels and drive them to their own
            # average - i.e. make the brightest thing in frame neutral. The
            # gains are then computed against those same bright values, not
            # against the whole-frame means, which is the entire point: a shot
            # that is genuinely one colour still has a neutral highlight.
            hi = [float(lin[..., c][m[..., c]].mean()) for c in range(3)]
            target = sum(hi) / 3.0
            want = (target, target, target)
            r, g, b = hi
            why = "white point (the brightest 1% driven neutral)"
        else:
            return lin, "White point: no bright region found; left alone."
    elif mode == "match reference":
        if reference is None:
            raise VAECleanError(
                "Balance mode is 'match reference' but no reference image is "
                "connected. Wire the plate you want the colour to match, or "
                "choose a different mode.")
        ref = srgb_to_linear(reference.float().clamp(0.0, 1.0))
        want = channel_means(ref)
        why = "matched to the reference image's channel means"
    else:
        raise VAECleanError(f"Unknown balance mode {mode!r}.")

    gains = [float(w) / max(float(c), 1e-6)
             for w, c in zip(want, (r, g, b))]
    gains = [1.0 + (gn - 1.0) * strength for gn in gains]
    out = lin * torch.tensor(gains, dtype=lin.dtype, device=lin.device)
    return out, (f"Channel balance by {why}: gains "
                 f"R {gains[0]:.3f} G {gains[1]:.3f} B {gains[2]:.3f}.")


def desaturate(lin: torch.Tensor, amount: float) -> torch.Tensor:
    """Scale chroma about luma. amount 1.0 leaves it alone.

    About LUMA, so pulling saturation down does not also pull brightness down -
    the thing a naive RGB lerp toward grey always does.
    """
    if amount == 1.0:
        return lin
    y = luma(lin)
    return y + (lin - y) * amount


def soften_contrast(lin: torch.Tensor, amount: float) -> torch.Tensor:
    """Lift crushed shadows. This REDUCES contrast, which is the complaint.

    The first version of this normalised the range to [0,1] - which INCREASES
    contrast, the exact opposite of what "too contrasty" asks for. A test
    caught it by measuring the black point going the wrong way.

    What a colourist does instead is raise the floor: a pedestal, so the
    crushed region stops being pure black, with white left where it is. That
    genuinely lowers contrast and, unlike a range stretch, cannot clip -
    everything moves toward white, never past it.

    It does NOT bring back detail that clipped to zero. Shadows pinned at
    black have no values left to separate; lifting them makes a flat dark grey
    rather than recovered shadow, and the report says how much is in that
    state.
    """
    if amount <= 0:
        return lin
    pedestal = amount * 0.05            # up to 5% of full scale, in linear
    return lin * (1.0 - pedestal) + pedestal


# ── round-trip metering (R5) and original-pixel restore (R4) ───────────────

def _align_reference(image: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Broadcast a 1-frame reference or require matching batch/H/W."""
    if reference.shape[-3:] != image.shape[-3:]:
        raise VAECleanError(
            f"Reference size {tuple(reference.shape)} does not match image "
            f"size {tuple(image.shape)}. Height and width must match.")
    if reference.shape[0] == image.shape[0]:
        return reference
    if reference.shape[0] == 1:
        return reference.expand(image.shape[0], -1, -1, -1)
    raise VAECleanError(
        f"Reference has {reference.shape[0]} frames but the image has "
        f"{image.shape[0]}. Connect a single-frame plate or match the count.")


def _box_mean(x: torch.Tensor, radius: int) -> torch.Tensor:
    """Mean over a (2r+1)^2 window with replicate padding. x: [B,H,W]."""
    if radius < 1:
        return x
    k = 2 * radius + 1
    x4 = x.unsqueeze(1)
    w = torch.ones(1, 1, k, k, dtype=x.dtype, device=x.device) / (k * k)
    return F.conv2d(
        F.pad(x4, (radius, radius, radius, radius), mode="replicate"), w,
    ).squeeze(1)


def _psnr_vs_ref(a: torch.Tensor, b: torch.Tensor, peak: float = 1.0) -> float:
    mse = float(torch.mean((a.float() - b.float()) ** 2))
    if mse == 0.0:
        return 99.0
    return 10.0 * math.log10((peak * peak) / mse)


def _psnr_masked(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor,
                 peak: float = 1.0) -> float:
    """PSNR over pixels where mask is True. a,b: [B,H,W,C]; mask: [B,H,W]."""
    m = mask.unsqueeze(-1).expand_as(a)
    if not m.any():
        return 99.0
    diff = (a[m] - b[m]).float()
    mse = float(torch.mean(diff ** 2))
    if mse == 0.0:
        return 99.0
    return 10.0 * math.log10((peak * peak) / mse)


def write_back_unchanged(
    image: torch.Tensor,
    reference: torch.Tensor,
    threshold: float,
    radius: int,
) -> tuple[torch.Tensor, dict]:
    """Write back unchanged pixels from reference (R4). Experimental."""
    img = image.float()
    ref = _align_reference(img, reference.float())
    radius = max(1, int(radius))

    d = (img - ref).abs().mean(dim=-1)
    d_mean = _box_mean(d, radius)
    changed = d_mean > threshold

    k = 2 * radius + 1
    dilated = F.max_pool2d(
        changed.float().unsqueeze(1),
        kernel_size=k, stride=1, padding=radius,
    ).squeeze(1) > 0.5

    feather = _box_mean(dilated.float(), radius)
    # A box mean of ones sums to 0.99999994, not 1: snap float noise at both ends so pixels the write-back does
    # not touch (well inside an edit, or far from one) come out bit-exact.
    feather = torch.where(feather > 1.0 - 1e-6, torch.ones_like(feather),
                          torch.where(feather < 1e-6, torch.zeros_like(feather), feather))
    fully_restored = feather == 0
    psnr_before = _psnr_masked(img, ref, fully_restored)
    m = feather.unsqueeze(-1)
    out = m * img + (1.0 - m) * ref
    psnr_after = _psnr_masked(out, ref, fully_restored)

    return out, {
        "restored_share": float(fully_restored.float().mean()),
        "psnr_unchanged_before": psnr_before,
        "psnr_unchanged_after": psnr_after,
    }


def srgb_to_lab(t: torch.Tensor) -> torch.Tensor:
    """sRGB (D65) to CIE Lab using the exact piecewise EOTF."""
    lin = srgb_to_linear(t.clamp(min=0.0))
    # sRGB D65 -> XYZ (IEC 61966-2-1)
    r, g, b = lin[..., 0], lin[..., 1], lin[..., 2]
    x = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b
    y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b
    z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b
    # D65 reference white
    xn, yn, zn = 0.95047, 1.0, 1.08883
    xr, yr, zr = x / xn, y / yn, z / zn
    delta = 6.0 / 29.0
    delta3 = delta ** 3
    inv3d2 = 1.0 / (3.0 * delta ** 2)
    fx = torch.where(xr > delta3, xr.pow(1.0 / 3.0), xr * inv3d2 + 4.0 / 29.0)
    fy = torch.where(yr > delta3, yr.pow(1.0 / 3.0), yr * inv3d2 + 4.0 / 29.0)
    fz = torch.where(zr > delta3, zr.pow(1.0 / 3.0), zr * inv3d2 + 4.0 / 29.0)
    l = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)
    return torch.stack((l, a, b), dim=-1)


def ciede2000(lab1: torch.Tensor, lab2: torch.Tensor) -> torch.Tensor:
    """CIEDE2000 colour difference (Sharma et al. 2005)."""
    l1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
    l2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]

    c1 = torch.sqrt(a1 * a1 + b1 * b1)
    c2 = torch.sqrt(a2 * a2 + b2 * b2)
    c_bar = (c1 + c2) * 0.5
    c_bar7 = c_bar ** 7
    g = 0.5 * (1.0 - torch.sqrt(c_bar7 / (c_bar7 + 25.0 ** 7)))

    a1p = (1.0 + g) * a1
    a2p = (1.0 + g) * a2
    c1p = torch.sqrt(a1p * a1p + b1 * b1)
    c2p = torch.sqrt(a2p * a2p + b2 * b2)

    h1p = torch.atan2(b1, a1p)
    h2p = torch.atan2(b2, a2p)
    h1p = torch.where(h1p < 0, h1p + 2.0 * torch.pi, h1p)
    h2p = torch.where(h2p < 0, h2p + 2.0 * torch.pi, h2p)

    dlp = l2 - l1
    dcp = c2p - c1p

    dhp = h2p - h1p
    dhp = torch.where(dhp > torch.pi, dhp - 2.0 * torch.pi, dhp)
    dhp = torch.where(dhp < -torch.pi, dhp + 2.0 * torch.pi, dhp)
    dhp = torch.where((c1p * c2p) == 0, torch.zeros_like(dhp), dhp)
    dhp = 2.0 * torch.sqrt(c1p * c2p) * torch.sin(dhp * 0.5)

    l_bar_p = (l1 + l2) * 0.5
    c_bar_p = (c1p + c2p) * 0.5

    hp_sum = h1p + h2p
    hp_diff = torch.abs(h1p - h2p)
    h_bar_p = torch.where(
        (c1p * c2p) == 0,
        hp_sum,
        torch.where(
            hp_diff <= torch.pi,
            hp_sum * 0.5,
            torch.where(
                hp_sum < 2.0 * torch.pi,
                (hp_sum + 2.0 * torch.pi) * 0.5,
                (hp_sum - 2.0 * torch.pi) * 0.5,
            ),
        ),
    )

    t = (1.0
         - 0.17 * torch.cos(h_bar_p - torch.pi / 6.0)
         + 0.24 * torch.cos(2.0 * h_bar_p)
         + 0.32 * torch.cos(3.0 * h_bar_p + torch.pi / 30.0)
         - 0.20 * torch.cos(4.0 * h_bar_p - 63.0 * torch.pi / 180.0))

    d_theta = 30.0 * torch.pi / 180.0 * torch.exp(
        -(((h_bar_p * 180.0 / torch.pi - 275.0) / 25.0) ** 2)
    )
    rc = 2.0 * torch.sqrt(c_bar_p ** 7 / (c_bar_p ** 7 + 25.0 ** 7))
    sl = 1.0 + 0.015 * (l_bar_p - 50.0) ** 2 / torch.sqrt(20.0 + (l_bar_p - 50.0) ** 2)
    sc = 1.0 + 0.045 * c_bar_p
    sh = 1.0 + 0.015 * c_bar_p * t
    rt = -torch.sin(2.0 * d_theta) * rc

    return torch.sqrt(
        (dlp / sl) ** 2
        + (dcp / sc) ** 2
        + (dhp / sh) ** 2
        + rt * (dcp / sc) * (dhp / sh)
    )


def _spatial_subsample_stride(frames: int, height: int, width: int,
                              limit: int = 2_000_000) -> int:
    n_pixels = frames * height * width
    return max(1, int(math.ceil(math.sqrt(n_pixels / limit))))


def roundtrip_report(image: torch.Tensor, reference: torch.Tensor) -> dict:
    """Round-trip loss vs reference (R5). PSNR/max_err on raw; dE2000 on clamped."""
    img = image.float()
    ref = reference.float()
    n_frames_img = img.shape[0]
    n_frames_ref = ref.shape[0]

    if img.shape[1:3] != ref.shape[1:3]:
        ih, iw = img.shape[1], img.shape[2]
        rh, rw = ref.shape[1], ref.shape[2]
        line = (
            f"Round trip not measured: the reference is {rw}x{rh}, "
            f"the image {iw}x{ih}."
        )
        return {
            "psnr": None,
            "max_err": None,
            "mean_de2000": None,
            "frames_image": n_frames_img,
            "frames_reference": n_frames_ref,
            "frame_note": None,
            "outside_01_share": None,
            "lines": [line],
        }

    if n_frames_ref == 1 and n_frames_img > 1:
        ref = ref.expand(n_frames_img, -1, -1, -1)
    elif n_frames_ref != n_frames_img:
        common = min(n_frames_img, n_frames_ref)
        img = img[:common]
        ref = ref[:common]
    else:
        common = n_frames_img

    outside = ((img < 0.0) | (img > 1.0)).any(dim=-1)
    outside_share = float(outside.float().mean())

    max_err = float((img - ref).abs().max())
    psnr = _psnr_vs_ref(img, ref, peak=1.0)

    stride = _spatial_subsample_stride(img.shape[0], img.shape[1], img.shape[2])
    img_s = img[:, ::stride, ::stride, :].clamp(0.0, 1.0)
    ref_s = ref[:, ::stride, ::stride, :].clamp(0.0, 1.0)
    mean_de = float(ciede2000(srgb_to_lab(img_s), srgb_to_lab(ref_s)).mean())

    lines = [
        f"Round-trip vs reference: PSNR {psnr:.2f} dB (raw, peak 1.0), "
        f"max error {max_err:.4f}, mean ΔE2000 {mean_de:.3f} (clamped 0..1).",
        f"Pixels outside 0..1: {outside_share * 100:.2f}%.",
    ]
    frame_note = None
    if n_frames_ref > 1 and n_frames_img > 1 and n_frames_img != n_frames_ref:
        frame_note = (
            f"Frame count: image {n_frames_img}, reference {n_frames_ref} "
            f"(compared first {common}). A Wan VAE encode rounds down to 4n+1 "
            f"frames and drops the rest without a message."
        )
        lines.append(frame_note)

    return {
        "psnr": psnr,
        "max_err": max_err,
        "mean_de2000": mean_de,
        "frames_image": n_frames_img,
        "frames_reference": n_frames_ref,
        "frame_note": frame_note,
        "outside_01_share": outside_share,
        "lines": lines,
    }


def clean_chroma(lin: torch.Tensor, strength: float) -> torch.Tensor:
    """Blur CHROMA only, leaving luma untouched.

    This is why it works on decode artifacts without softening the picture:
    the eye carries detail in luma, so chroma can be smoothed hard before
    anything looks soft. It is the same reason every video codec subsamples
    chroma and not luma.
    """
    if strength <= 0:
        return lin
    y = luma(lin)
    chroma = (lin - y).movedim(-1, 1)
    radius = max(1, int(round(strength * 3)))
    k = 2 * radius + 1
    w = torch.ones(1, 1, k, 1, dtype=chroma.dtype, device=chroma.device) / k
    c = F.conv2d(F.pad(chroma, (0, 0, radius, radius), mode="replicate"),
                 w.expand(3, 1, k, 1), groups=3)
    c = F.conv2d(F.pad(c, (radius, radius, 0, 0), mode="replicate"),
                 w.view(1, 1, 1, k).expand(3, 1, 1, k), groups=3)
    return y + c.movedim(1, -1)


# ── the report ──────────────────────────────────────────────────────────────

def describe(before: dict, after: dict, actions: list[str]) -> str:
    lines = []
    r, g, b = before["cast"]
    worst = max(abs(r), abs(g), abs(b))

    if worst < CAST_FLOOR:
        lines.append(
            f"No channel cast worth correcting: the strongest deviation is "
            f"{worst * 100:.2f}%, which is measurement noise. If the picture "
            "still looks wrong the problem is not a cast - look at saturation "
            "and contrast below.")
    else:
        names = ("red", "green", "blue")
        lead = names[max(range(3), key=lambda i: before["cast"][i])]
        lines.append(
            f"Channel cast: R {r * 100:+.1f}%  G {g * 100:+.1f}%  "
            f"B {b * 100:+.1f}% against neutral - a {lead} lean.")
        if worst > 0.12:
            lines.append(
                "That is a LARGE deviation. Before treating it as a decoder "
                "cast, check it is not simply what the shot is: a sunset or a "
                "red-lit interior measures exactly like this, and neutralising "
                "it throws the lighting away. A decoder cast is the same on "
                "EVERY shot; a scene colour is not.")

    sat_before, sat_after = before["saturation"], after["saturation"]
    lines.append(
        f"Saturation (mean chroma / luma): {sat_before:.3f} -> {sat_after:.3f}.")
    # 0.40 from measurement, not taste. On a neutral-but-colourful plate the
    # metric reads 0.13; the same plate with its chroma scaled 2x reads 0.25
    # and 3.5x reads 0.42. So 0.40 sits above "vivid and fine" and below
    # "clearly pushed", which is where the warning is useful rather than
    # constant.
    if sat_before > 0.40:
        lines.append(
            "That is high. On a diffusion model, oversaturation together with "
            "hard contrast is the classic signature of CFG set too high - not "
            "a decoder fault. Pulling it down here treats the symptom; "
            "lowering CFG treats the cause, and keeps the detail that "
            "oversaturation is currently burning out.")

    lines.append(
        f"Black point {before['black']:.4f} -> {after['black']:.4f}, "
        f"white point {before['white']:.4f} -> {after['white']:.4f} (linear).")

    clipped = []
    for ch in "rgb":
        lo, hi = before["clip"][ch]
        if lo > CLIP_NOTICE or hi > CLIP_NOTICE:
            clipped.append(f"{ch.upper()} {lo * 100:.2f}% at black, "
                           f"{hi * 100:.2f}% at white")
    if clipped:
        lines.append("CLIPPED, and this cannot be undone: " + "; ".join(clipped)
                     + ". Those pixels have lost the values beyond the rail. "
                     "Nothing here reconstructs them - a tool that claims to "
                     "is inventing detail. Re-render with less contrast if it "
                     "matters.")
    else:
        lines.append("Nothing is clipped; the full range survived the decode.")

    noise = before["chroma_noise"]
    # 0.006 from measurement: a clean plate reads 0.0018, and visible decode
    # speckle pushes it several times higher.
    lines.append(f"High-frequency chroma energy {noise:.5f}"
                 + (" - high for a clean decode; chroma cleanup will help "
                    "without softening the picture, because detail lives in "
                    "luma." if noise > 0.006 else " - normal."))

    if actions:
        lines.append("")
        lines.extend(actions)
    else:
        lines.append("")
        lines.append("Measured only; nothing was changed.")
    return "\n".join(lines)


# ── the node ────────────────────────────────────────────────────────────────

class VAECleanMEC:
    """Measure a decode, then correct what is actually wrong with it."""

    VRAM_TIER = 1

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "balance_mode": (list(BALANCE_MODES), {
                    "default": "off",
                    "tooltip": "off = measure only.\n"
                               "grey world = drive the AVERAGE neutral. Fast "
                               "and wrong on any shot that is genuinely one "
                               "colour - a sunset averages red and this will "
                               "throw the sunset away.\n"
                               "white point = drive the BRIGHTEST 1% neutral. "
                               "Safer on coloured scenes, because a red room "
                               "still has a neutral highlight.\n"
                               "match reference = copy a plate's channel "
                               "means. Use when you have the real footage."}),
                "balance_strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "How far toward neutral. 1.0 is full "
                               "correction; back it off when the cast is "
                               "partly the lighting you wanted."}),
                "saturation": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01,
                    "tooltip": "Chroma scale about luma, so pulling it down "
                               "does not also darken the picture. 1.0 leaves "
                               "it alone. If the render is oversaturated AND "
                               "contrasty, read the report first - that "
                               "combination is usually CFG, and lowering CFG "
                               "keeps detail that desaturating here cannot "
                               "bring back."}),
                "contrast_restore": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Lift crushed shadows, which brings contrast "
                               "DOWN - the fix for a render that came back too "
                               "hard. Raises the floor rather than stretching "
                               "the range, so nothing can clip. It does NOT "
                               "recover detail already crushed to black: those "
                               "pixels just stop being pure black. The report "
                               "says how much is in that state."}),
                "chroma_cleanup": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Smooth colour noise without touching detail. "
                               "Blurs CHROMA only - the eye carries detail in "
                               "luma, which is why every codec subsamples "
                               "chroma and not luma, and why this can go quite "
                               "hard before anything looks soft. The fix for "
                               "VAE decode speckle."}),
            },
            "optional": {
                "reference": ("IMAGE", {
                    "tooltip": "Required by 'match reference'. The plate, or "
                               "any frame whose colour you want copied."}),
                "restore_unchanged": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 0.5, "step": 0.005,
                    "tooltip": "Experimental. 0 = off. Every VAE round trip "
                               "loses detail everywhere, even where nothing was "
                               "meant to change. With the original plate wired "
                               "to 'reference', this writes the original pixels "
                               "back wherever the picture did not really change, "
                               "and keeps the edit.\n"
                               "The value is how far the local average may move "
                               "and still count as unchanged. Measured on the "
                               "Flux VAE: one round trip moves it by up to 0.037, "
                               "three by up to 0.050 - so start near 0.05-0.07. "
                               "Too low leaves VAE loss in place; too high also "
                               "reverts subtle intended changes."}),
                "restore_radius": ("INT", {
                    "default": 8, "min": 1, "max": 64,
                    "tooltip": "Window for the write-back, in pixels (8 = one "
                               "latent cell of an 8x VAE). An edit's influence "
                               "reaches about 3x this far past its edge, so the "
                               "boundary blends rather than cuts. Experimental."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "FLOAT", "FLOAT")
    RETURN_NAMES = ("image", "report", "cast_strength", "saturation")
    FUNCTION = "clean"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "Diagnose a decode that came back reddish, oversaturated or too "
        "contrasty, then correct what is actually wrong. Measures the channel "
        "cast, saturation, black and white points, clipping and chroma noise, "
        "and says WHICH of those is the problem - they need different fixes "
        "and only some are the decoder's fault. All corrections happen in "
        "LINEAR, because scaling chroma on sRGB values moves the midtones too."
    )

    @classmethod
    def IS_CHANGED(cls, image, **kw):
        import hashlib
        h = hashlib.md5(image.detach().cpu().numpy().tobytes())
        h.update(repr(sorted((k, str(v)) for k, v in kw.items())).encode())
        return h.hexdigest()

    def clean(self, image, balance_mode, balance_strength, saturation,
              contrast_restore, chroma_cleanup, reference=None,
              restore_unchanged=0.0, restore_radius=8):
        if image.ndim != 4 or image.shape[-1] < 3:
            raise VAECleanError(
                f"Expected an IMAGE batch [B,H,W,C], got {tuple(image.shape)}.")

        srgb = image.float().clamp(0.0, 1.0)
        before = measure(srgb)

        roundtrip_lines: list[str] = []
        if reference is not None:
            roundtrip_lines = roundtrip_report(image.float(), reference)["lines"]

        working = image.float()
        actions: list[str] = []
        if restore_unchanged > 0:
            if reference is None:
                raise VAECleanError(
                    "restore_unchanged is set but no reference image is "
                    "connected. Wire the original plate, or set "
                    "restore_unchanged to 0.")
            working, rinfo = write_back_unchanged(
                working, reference, float(restore_unchanged), int(restore_radius))
            actions.append(
                f"Experimental restore: {rinfo['restored_share'] * 100:.1f}% of "
                f"pixels written back from reference. PSNR on unchanged pixels "
                f"{rinfo['psnr_unchanged_before']:.1f} -> "
                f"{rinfo['psnr_unchanged_after']:.1f} dB.")

        srgb = working.clamp(0.0, 1.0)
        lin = srgb_to_linear(srgb)

        lin, why = balance(lin, balance_mode, reference=reference,
                           strength=float(balance_strength))
        if balance_mode != "off" and balance_strength > 0:
            actions.append(why)

        if saturation != 1.0:
            lin = desaturate(lin, float(saturation))
            actions.append(f"Saturation scaled to {saturation:.2f} about luma.")

        if contrast_restore > 0:
            lin = soften_contrast(lin, float(contrast_restore))
            actions.append(
                f"Shadows lifted at {contrast_restore:.2f} - a pedestal of "
                f"{contrast_restore * 0.05:.3f} in linear, so contrast comes "
                "DOWN and nothing can clip. Detail already crushed to black "
                "is not recovered by this; it just stops being pure black.")

        if chroma_cleanup > 0:
            lin = clean_chroma(lin, float(chroma_cleanup))
            actions.append(
                f"Chroma smoothed at {chroma_cleanup:.2f}; luma untouched, so "
                "detail is unaffected.")

        out = linear_to_srgb(lin).clamp(0.0, 1.0)
        after = measure(out)
        cast = max(abs(c) for c in before["cast"])
        report_body = describe(before, after, actions)
        if roundtrip_lines:
            report = "\n".join(roundtrip_lines + ["", report_body])
        else:
            report = report_body
        return (out.to(image.dtype), report,
                float(cast), float(before["saturation"]))


# L7.65 P15: merged into VAE Decode (C2C) (hdr_color_science.C2CVAEQualityDecode, which runs this class's
# clean()); saved workflows migrate there. Not registered on its own any more.
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
