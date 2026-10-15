"""L7.114: the Matte viewer's QC numbers (nodes/c2c_viewer/qc.py) on synthetic clips with a KNOWN true alpha.

The clip: a textured ellipse with 24 thin hair strands moving 2 px per frame over a textured, static background,
rendered at 4x supersampling (true alpha = exact coverage). The coarse mask is what a segmenter gives: the ellipse
only, binary. Each case damages the true alpha in one known way; the thresholds in qc.THRESHOLDS were chosen from
these measurements (docs/design/matte_viewer.md, "QC thresholds") and this file pins every case to its level, so a
change to the maths that moves a case to another colour fails here.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from nodes.c2c_viewer import qc

N, H, W, SS = 10, 180, 320, 4


def _supersampled(fn, t):
    yy, xx = np.mgrid[0:H * SS, 0:W * SS].astype(np.float32)
    m = fn((xx + 0.5) / SS, (yy + 0.5) / SS, t).astype(np.float32)
    return m.reshape(H, SS, W, SS).mean(axis=(1, 3))


def _ellipse(x, y, t, cx0=110.0, cy=95.0, rx=48.0, ry=60.0, speed=2.0):
    cx = cx0 + speed * t
    return ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 <= 1.0


def _strands(x, y, t, cx0=110.0, cy=95.0, ry=60.0, speed=2.0):
    cx = cx0 + speed * t
    out = np.zeros_like(x, dtype=bool)
    rng = np.random.default_rng(11)
    for _ in range(24):
        ang = rng.uniform(-1.2, 1.2)                     # around straight up
        length = rng.uniform(18, 34)
        x0, y0 = cx + 40 * np.sin(ang), cy - ry * np.cos(ang) * 0.92
        dx, dy = np.sin(ang), -np.cos(ang)
        # distance from each point to the segment
        px, py = x - x0, y - y0
        s = np.clip(px * dx + py * dy, 0, length)
        d = np.hypot(px - s * dx, py - s * dy)
        out |= d <= 0.6                                  # ~1.2 px wide
    return out


def make_clip(n=N, seed=0):
    """(plate B,H,W,3, true alpha B,H,W, coarse B,H,W) as float tensors."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    bg = np.stack([0.30 + 0.2 * xx / W, 0.35 + 0.1 * yy / H, np.full_like(xx, 0.40)], -1)
    bg += 0.08 * F.avg_pool2d(torch.from_numpy(rng.standard_normal((1, 3, H, W)).astype(np.float32)), 3, 1, 1
                              )[0].permute(1, 2, 0).numpy()
    tex = 0.10 * F.avg_pool2d(torch.from_numpy(rng.standard_normal((1, 3, H, W + 4 * n)).astype(np.float32)), 3, 1,
                              1)[0].permute(1, 2, 0).numpy()
    plates, alphas, coarses = [], [], []
    for t in range(n):
        a = _supersampled(lambda x, y, tt: _ellipse(x, y, tt) | _strands(x, y, tt), t)
        c = (_supersampled(lambda x, y, tt: _ellipse(x, y, tt), t) > 0.5).astype(np.float32)
        shift = int(round(2.0 * t))
        fg = np.array([0.80, 0.55, 0.35], np.float32) + tex[:, 4 * n - shift: 4 * n - shift + W]
        p = a[..., None] * fg + (1 - a[..., None]) * bg
        plates.append(p.clip(0, 1))
        alphas.append(a)
        coarses.append(c)
    t = lambda v: torch.from_numpy(np.stack(v)).float()   # noqa: E731
    return t(plates), t(alphas), t(coarses)


def blur(a: torch.Tensor, sigma: float) -> torch.Tensor:
    r = int(3 * sigma)
    x = torch.arange(-r, r + 1, dtype=torch.float32)
    k = torch.exp(-x ** 2 / (2 * sigma ** 2))
    k = k / k.sum()
    y = F.conv2d(a.unsqueeze(1), k.view(1, 1, 1, -1), padding=(0, r))
    y = F.conv2d(y, k.view(1, 1, -1, 1), padding=(r, 0))
    return y[:, 0].clamp(0, 1)


def jitter(a: torch.Tensor, seed=1) -> torch.Tensor:
    """Each frame's edge randomly grown or shrunk by one pixel: a matte that crawls from frame to frame."""
    rng = np.random.default_rng(seed)
    out = []
    for f in a:
        m = f[None, None]
        if rng.random() < 0.5:
            m = F.max_pool2d(m, 3, 1, 1)
        else:
            m = -F.max_pool2d(-m, 3, 1, 1)
        out.append(m[0, 0])
    return torch.stack(out)


def noisy(a: torch.Tensor, amp=0.15, seed=2) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    edge = (a > 0.02) & (a < 0.98)
    n = (torch.rand(a.shape, generator=g) * 2 - 1) * amp
    return torch.where(edge | (blur(a, 1.5) - a).abs().gt(0.02), (a + n).clamp(0, 1), a)


def cut(a: torch.Tensor, coarse: torch.Tensor, share: float) -> torch.Tensor:
    """Remove the right-most `share` of the subject (per frame): a matte that lost part of the selection."""
    out = a.clone()
    for i in range(a.shape[0]):
        xs = torch.nonzero(coarse[i] > 0.5)[:, 1]
        cols = torch.sort(xs).values
        x_cut = int(cols[int(len(cols) * (1 - share))])
        out[i, :, x_cut:] = 0
    return out


CASES = {
    "true alpha":  lambda a, c: a,
    "coarse only": lambda a, c: c,
    "blur 1.5 px": lambda a, c: blur(a, 1.5),
    "blur 3 px":   lambda a, c: blur(a, 3.0),
    "blur 6 px":   lambda a, c: blur(a, 6.0),
    "edge jitter": lambda a, c: jitter(a),
    "noise 0.07":  lambda a, c: noisy(a, amp=0.07),
    "noise 0.15":  lambda a, c: noisy(a),
    "lost 5 %":    lambda a, c: cut(a, c, 0.05),
    "lost 15 %":   lambda a, c: cut(a, c, 0.15),
}

#: the level each case must land in (coverage, edge detail, flicker)
EXPECT = {
    "true alpha":  ("green", "green", "green"),
    "coarse only": ("green", "green", "green"),
    "blur 1.5 px": ("green", "amber", "green"),
    "blur 3 px":   ("green", "red", "green"),
    "blur 6 px":   ("green", "red", "green"),
    "edge jitter": ("green", "green", "red"),
    "noise 0.07":  ("green", "green", "amber"),
    "noise 0.15":  ("green", "green", "red"),
    "lost 5 %":    ("amber", "green", "green"),
    "lost 15 %":   ("red", "green", "green"),
}


@pytest.fixture(scope="module")
def clip():
    pytest.importorskip("cv2")
    return make_clip()


def measure_all(clip):
    plate, alpha, coarse = clip
    return {name: qc.measure(plate, fn(alpha, coarse), coarse) for name, fn in CASES.items()}


@pytest.fixture(scope="module")
def results(clip):
    return measure_all(clip)


@pytest.mark.parametrize("name", list(CASES))
def test_each_damage_lands_in_its_level(results, name):
    r = results[name]
    got = (r["coverage"]["level"], r["edge_detail"]["level"], r["flicker"]["level"])
    assert got == EXPECT[name], (name, {k: r[k]["value"] for k in ("coverage", "edge_detail", "flicker")})


def test_the_flicker_is_motion_compensated(results):
    """The subject moves 2 px a frame; without the flow warp the true alpha would read as flicker."""
    assert results["true alpha"]["flicker"]["flow"] == "farneback"
    assert results["true alpha"]["flicker"]["pairs"] == N - 1


def test_one_frame_has_no_flicker_and_no_coarse_has_no_coverage(clip):
    plate, alpha, coarse = clip
    one = qc.measure(plate[:1], alpha[:1], coarse[:1])
    assert one["flicker"]["value"] is None and one["flicker"]["level"] == "n/a"
    bare = qc.measure(plate, alpha, None)
    assert bare["coverage"]["level"] == "n/a" and bare["edge_detail"]["level"] == "n/a"


def test_long_clips_measure_a_bounded_number_of_pairs():
    ts = qc.pairs(1000)
    assert len(ts) == qc.MAX_PAIRS and ts[0] == 1 and ts[-1] == 999


def test_levels():
    assert qc.level("coverage", 0.99) == "green" and qc.level("coverage", 0.5) == "red"
    assert qc.level("flicker", 0.001) == "green" and qc.level("flicker", 0.5) == "red"
    assert qc.level("edge_detail", None) == "n/a"
