"""Golden tests for the Image Mask Editor's raster kernels (js/image_mask_editor/tools.js, undo.js).

The kernels run in Node on synthetic inputs; the references are OpenCV (fillPoly, ellipse, floodFill) or
plain numpy, never a copy of the JS arithmetic.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="node not on PATH")
cv2 = pytest.importorskip("cv2")

JS = Path(__file__).resolve().parents[1] / "js" / "image_mask_editor"


def _run(tmp_path: Path, body: str, inputs: dict) -> dict:
    probe = tmp_path / "probe.mjs"
    probe.write_text(
        f"const T = await import({json.dumps((JS / 'tools.js').resolve().as_uri())});\n"
        f"const U = await import({json.dumps((JS / 'undo.js').resolve().as_uri())});\n"
        f"const IN = {json.dumps(inputs)};\n"
        "const out = {};\n" + body + "\nconsole.log(JSON.stringify(out));\n",
        encoding="utf-8")
    r = subprocess.run([NODE, str(probe)], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr[-2000:]
    return json.loads(r.stdout.strip().splitlines()[-1])


def _run_colour(tmp_path: Path, body: str, inputs: dict) -> dict:
    probe = tmp_path / "probe_colour.mjs"
    probe.write_text(
        f"const C = await import({json.dumps((JS / 'colour.js').resolve().as_uri())});\n"
        f"const T = await import({json.dumps((JS / 'tools.js').resolve().as_uri())});\n"
        f"const IN = {json.dumps(inputs)};\n"
        "const out = {};\n" + body + "\nconsole.log(JSON.stringify(out));\n",
        encoding="utf-8")
    r = subprocess.run([NODE, str(probe)], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr[-2000:]
    return json.loads(r.stdout.strip().splitlines()[-1])


def _mask(out_list, w, h) -> np.ndarray:
    return np.array(out_list, dtype=np.uint8).reshape(h, w)


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a > 127, b > 127
    return float((a & b).sum()) / max(1, int((a | b).sum()))


def test_fill_rect_is_the_exact_pixel_range(tmp_path):
    w, h = 64, 48
    out = _run(tmp_path, """
      const m = new Uint8Array(IN.w * IN.h), f = new Uint8Array(IN.w * IN.h);
      out.b = T.fillRect(m, IN.w, IN.h, 10, 7, 40, 30, 255, false);
      T.fillRect(f, IN.w, IN.h, 10.6, 7.2, 40.4, 30.5, 255, false);    // fractional: pixel CENTRES decide
      out.m = Array.from(m); out.f = Array.from(f);
    """, {"w": w, "h": h})
    ref = np.zeros((h, w), np.uint8)
    ref[7:30, 10:40] = 255
    assert np.array_equal(_mask(out["m"], w, h), ref)
    assert out["b"] == {"x0": 10, "y0": 7, "x1": 39, "y1": 29}
    reff = np.zeros((h, w), np.uint8)
    reff[7:30, 11:40] = 255                       # centres 11.5..39.5 and 7.5..29.5 lie inside
    assert np.array_equal(_mask(out["f"], w, h), reff)


def test_fill_ellipse_matches_opencv(tmp_path):
    w, h = 160, 120
    out = _run(tmp_path, """
      const m = new Uint8Array(IN.w * IN.h);
      T.fillEllipse(m, IN.w, IN.h, 20, 15, 140, 105, 255, false);
      out.m = Array.from(m);
    """, {"w": w, "h": h})
    # OpenCV puts pixel centres on integers; the box 20..140 x 15..105 (pixel corners) has its centre at
    # (79.5, 59.5) there - drawn with 4 fractional bits through cv2's shift argument.
    ref = np.zeros((h, w), np.uint8)
    cv2.ellipse(ref, (int(79.5 * 16), int(59.5 * 16)), (60 * 16, 45 * 16), 0, 0, 360, 255, -1, cv2.LINE_8, 4)
    got = _mask(out["m"], w, h)
    assert _iou(got, ref) > 0.98
    # centred where it was drawn: the box 20..140 x 15..105 mirrors onto itself
    box = got[15:105, 20:140]
    assert np.array_equal(box, box[:, ::-1]) and np.array_equal(box, box[::-1, :])


def test_fill_polygon_matches_opencv(tmp_path):
    w, h = 160, 120
    pts = [[20, 10], [140, 30], [100, 110], [30, 90]]
    out = _run(tmp_path, """
      const m = new Uint8Array(IN.w * IN.h);
      T.fillPolygon(m, IN.w, IN.h, IN.pts.map(([x, y]) => ({ x, y })), 255);
      out.m = Array.from(m);
    """, {"w": w, "h": h, "pts": pts})
    # Independent reference: OpenCV's point-in-polygon test at every pixel CENTRE (no centre of this
    # integer-vertex quad lies exactly on an edge, so the comparison is exact).
    contour = np.array(pts, np.float32).reshape(-1, 1, 2)
    ref = np.zeros((h, w), np.uint8)
    for y in range(h):
        for x in range(w):
            if cv2.pointPolygonTest(contour, (x + 0.5, y + 0.5), False) > 0:
                ref[y, x] = 255
    got = _mask(out["m"], w, h)
    assert np.array_equal(got, ref), f"{int((got != ref).sum())} pixels differ"
    filled = np.zeros((h, w), np.uint8)
    cv2.fillPoly(filled, [np.array(pts, np.int32)], 255)
    assert _iou(got, filled) > 0.96                     # and close to OpenCV's own rasteriser


def test_flood_fill_matches_opencv_fixed_range(tmp_path):
    rng = np.random.default_rng(7)
    w, h, tol = 96, 72, 20
    img = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    img[10:50, 20:70] = (200, 40, 40) + rng.integers(-15, 16, (40, 50, 3))     # a noisy red patch, within tol
    img[30:34, 20:70] = (20, 200, 20)                                          # a green bar cutting it in two
    rgba = np.dstack([img, np.full((h, w), 255, np.uint8)]).reshape(-1).tolist()
    out = _run(tmp_path, """
      const m = new Uint8Array(IN.w * IN.h);
      out.b = T.floodFill(m, Uint8ClampedArray.from(IN.rgba), IN.w, IN.h, 25, 15, IN.tol, 255);
      out.m = Array.from(m);
    """, {"w": w, "h": h, "tol": tol, "rgba": rgba})
    ff = np.zeros((h + 2, w + 2), np.uint8)
    seed = img[15, 25].astype(int)
    work = img.copy()
    cv2.floodFill(work, ff, (25, 15), (0, 0, 0), (tol,) * 3, (tol,) * 3,
                  4 | cv2.FLOODFILL_FIXED_RANGE | cv2.FLOODFILL_MASK_ONLY | (255 << 8))
    ref = ff[1:-1, 1:-1]
    got = _mask(out["m"], w, h)
    assert np.array_equal(got, ref), f"seed {seed.tolist()}: {int((got != ref).sum())} pixels differ"
    assert got[40:50, 20:70].max() == 0          # the bar stops the fill: the lower half is not reached


def test_brush_stamp_profile(tmp_path):
    w, h = 64, 64
    out = _run(tmp_path, """
      const m = new Uint8Array(IN.w * IN.h), start = new Uint8Array(IN.w * IN.h), cov = new Uint8Array(IN.w * IN.h);
      T.stampBrushCoverage(m, start, cov, IN.w, IN.h, 32, 32, 20, 0.5, 1, false);
      out.m = Array.from(m);
    """, {"w": w, "h": h})
    m = _mask(out["m"], w, h).astype(int)
    row = m[32, 32:]
    assert m[32, 32] == 255 and m[32, 52:].max() == 0 and m[0:10, :].max() == 0
    # hard core out to radius * hardness = 10: pixel 32+k has its centre at distance ~ k + 0.5
    assert (row[:10] == 255).all() and row[10] < 255
    assert all(a >= b for a, b in zip(row, row[1:]))    # falloff never rises
    assert np.array_equal(m, m.T)                       # round, not square
    assert np.array_equal(m[:, :32][:, ::-1], m[:, 32:])  # centred on the pointer at x = 32.0


def test_hardness_one_is_a_hard_disc_and_zero_a_linear_cone(tmp_path):
    w, h = 64, 64
    out = _run(tmp_path, """
      const n = IN.w * IN.h, z = () => new Uint8Array(n);
      const hard = z(), soft = z();
      T.stampBrushCoverage(hard, z(), z(), IN.w, IN.h, 32, 32, 20, 1, 1, false);
      T.stampBrushCoverage(soft, z(), z(), IN.w, IN.h, 32, 32, 20, 0, 1, false);
      out.hard = Array.from(hard); out.soft = Array.from(soft);
    """, {"w": w, "h": h})
    yy, xx = np.mgrid[0:h, 0:w]
    d = np.hypot(xx + 0.5 - 32, yy + 0.5 - 32)          # distance of every pixel CENTRE to the pointer
    hard, soft = _mask(out["hard"], w, h).astype(int), _mask(out["soft"], w, h).astype(int)
    assert np.array_equal(hard, np.where(d <= 20, 255, 0))
    ref = np.where(d <= 20, np.round(255 * (1 - d / 20)), 0)
    assert np.abs(soft - ref).max() <= 1


def test_eraser_stamp_lowers_toward_zero(tmp_path):
    w, h = 32, 32
    out = _run(tmp_path, """
      const n = IN.w * IN.h;
      const start = new Uint8Array(n).fill(255), m = new Uint8Array(n).fill(255), cov = new Uint8Array(n);
      T.stampBrushCoverage(m, start, cov, IN.w, IN.h, 16, 16, 8, 1, 1, true);
      out.m = Array.from(m);
    """, {"w": w, "h": h})
    m = _mask(out["m"], w, h)
    assert m[16, 16] == 0 and m[16, 25] == 255 and m[0, 0] == 255


def test_undo_stack_holds_its_byte_cap(tmp_path):
    out = _run(tmp_path, """
      const s = new U.UndoStack();
      const big = 20 * 1024 * 1024;                      // 40 MB per entry (before + after)
      for (let i = 0; i < 10; i++) s.push({ x: 0, y: 0, w: 1, h: 1, before: new Uint8Array(big), after: new Uint8Array(big) });
      out.entries = s.undo.length; out.bytes = s.bytes;
      const e = s.popUndo(); s.pushRedo(e);
      out.afterUndo = { undo: s.undo.length, redo: s.redo.length, bytes: s.bytes };
    """, {})
    cap = 128 * 1024 * 1024
    assert out["bytes"] <= cap and out["entries"] == 3
    assert out["afterUndo"]["bytes"] <= cap and out["afterUndo"]["undo"] + out["afterUndo"]["redo"] <= 3


def test_shift_square_constrains_rect_and_ellipse(tmp_path):
    w, h = 64, 48
    out = _run(tmp_path, """
      const r = new Uint8Array(IN.w * IN.h), e = new Uint8Array(IN.w * IN.h);
      T.fillRect(r, IN.w, IN.h, 10, 10, 40, 25, 255, true);      // Shift: the shorter side wins
      T.fillEllipse(e, IN.w, IN.h, 10, 10, 40, 25, 255, true);
      out.r = Array.from(r); out.e = Array.from(e);
    """, {"w": w, "h": h})
    ref = np.zeros((h, w), np.uint8)
    ref[10:25, 10:25] = 255
    assert np.array_equal(_mask(out["r"], w, h), ref)
    e = _mask(out["e"], w, h)
    ys, xs = np.nonzero(e)
    assert xs.min() >= 10 and xs.max() <= 24 and ys.min() >= 10 and ys.max() <= 24
    sub = e[:40, :40]
    assert np.array_equal(sub, sub.T)                            # a circle: symmetric about the diagonal


def test_compose_selection_add_subtract_intersect(tmp_path):
    w, h = 8, 8
    out = _run(tmp_path, """
      const mask = new Uint8Array(IN.mask);
      const sel = new Uint8Array(IN.sel);
      const b = { x0: 2, y0: 2, x1: 5, y1: 5 };
      out.add = Array.from((() => {
        const m = new Uint8Array(mask); T.composeSelection(m, sel, IN.w, IN.h, "add", b); return m;
      })());
      out.sub = Array.from((() => {
        const m = new Uint8Array(mask); T.composeSelection(m, sel, IN.w, IN.h, "subtract", b); return m;
      })());
      out.inter = Array.from((() => {
        const m = new Uint8Array(mask); T.composeSelection(m, sel, IN.w, IN.h, "intersect", b); return m;
      })());
    """, {
        "w": w, "h": h,
        "mask": [100] * (w * h),
        "sel": [0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 255, 255, 255, 0, 0, 0,
                0, 0, 255, 200, 255, 0, 0, 0,
                0, 0, 255, 255, 255, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0],
    })
    m = np.full((h, w), 100, np.uint8)
    s = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 255, 255, 255, 0, 0, 0],
        [0, 0, 255, 200, 255, 0, 0, 0],
        [0, 0, 255, 255, 255, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ], np.uint8)
    b = np.s_[2:6, 2:6]
    ref_add = m.copy()
    ref_add[b] = np.maximum(m[b], s[b])
    ref_sub = m.copy()
    ref_sub[b] = np.minimum(m[b], 255 - s[b])
    ref_inter = np.minimum(m, s)
    assert np.array_equal(_mask(out["add"], w, h), ref_add)
    assert np.array_equal(_mask(out["sub"], w, h), ref_sub)
    assert np.array_equal(_mask(out["inter"], w, h), ref_inter)
    assert ref_inter[0, 0] == 0                                        # outside sel zeroed


def test_selection_from_distance_ramp(tmp_path):
    out = _run_colour(tmp_path, """
      const dist = new Float32Array(IN.dist);
      out.s = Array.from(C.selectionFromDistance(dist, IN.tol, IN.soft));
    """, {"dist": [0, 16, 32, 48, 80], "tol": 32, "soft": 32})
    s = out["s"]
    assert s[0] == 255 and s[1] == 255 and s[2] == 255                 # d <= tol
    assert s[3] == 128                                                # d == tol + softness/2
    assert s[4] == 0                                                  # beyond tol + softness


def test_keep_connected_keeps_seeded_blob(tmp_path):
    w, h = 6, 4
    sel = [
        0, 0, 255, 255, 0, 0,
        0, 0, 255, 255, 0, 0,
        255, 255, 0, 0, 255, 255,
        255, 255, 0, 0, 255, 255,
    ]
    out = _run_colour(tmp_path, """
      const sel = new Uint8Array(IN.sel);
      out.k = Array.from(C.keepConnected(sel, IN.w, IN.h, [{ x: 2, y: 0 }]));
    """, {"w": w, "h": h, "sel": sel})
    k = _mask(out["k"], w, h)
    assert k[0, 2] == 255 and k[0, 3] == 255
    assert k[2:, :].max() == 0


def _hsv_dist_ref(rgb: np.ndarray, samples: np.ndarray) -> np.ndarray:
    """Reference HSV distance matching the JS contract (OpenCV 8-bit convention)."""
    f = rgb.astype(np.float32) / 255.0
    hsv = cv2.cvtColor(f.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3)
    h = hsv[:, 0] * 0.5                                               # degrees -> 0..180
    s = hsv[:, 1] * 255.0
    v = hsv[:, 2] * 255.0
    out = np.empty(len(rgb), np.float32)
    for i in range(len(rgb)):
        dmin = np.inf
        for samp in samples:
            sf = samp.astype(np.float32) / 255.0
            shsv = cv2.cvtColor(sf.reshape(1, 1, 3), cv2.COLOR_RGB2HSV).reshape(3)
            sh, ss, sv = shsv[0] * 0.5, shsv[1] * 255.0, shsv[2] * 255.0
            dh = abs(h[i] - sh)
            if dh > 90:
                dh = 180 - dh
            dh_w = dh * (255.0 / 90.0) * (min(s[i], ss) / 255.0)
            ds = abs(s[i] - ss)
            dv = abs(v[i] - sv)
            dmin = min(dmin, max(dh_w, ds, dv))
        out[i] = dmin
    return out


def _lab_of(rgb_u8: np.ndarray, exact: bool) -> np.ndarray:
    f = rgb_u8.astype(np.float64).reshape(-1, 1, 3) / 255.0
    if exact:      # scikit-image: the CIE formulas with an exact sRGB curve, D65
        from skimage.color import rgb2lab
        return rgb2lab(f).reshape(-1, 3)
    # OpenCV's float path interpolates the sRGB curve from a table: a looser, second opinion
    return cv2.cvtColor(f.astype(np.float32), cv2.COLOR_RGB2Lab).reshape(-1, 3).astype(np.float64)


def _lab_dist_ref(rgb: np.ndarray, samples: np.ndarray, exact: bool = True) -> np.ndarray:
    lab, slab = _lab_of(rgb, exact), _lab_of(samples, exact)
    d = np.linalg.norm(lab[:, None, :] - slab[None, :, :], axis=2).min(axis=1)
    return (2.55 * d).astype(np.float32)


def test_distance_map_rgb_hsv_lab(tmp_path):
    rng = np.random.default_rng(42)
    n = 2000
    rgb = rng.integers(0, 256, (n, 3), dtype=np.uint8)
    samples = rng.integers(0, 256, (3, 3), dtype=np.uint8)
    rgba = np.zeros((n, 4), np.uint8)
    rgba[:, :3] = rgb
    rgba[:, 3] = 255
    for space in ("rgb", "hsv", "lab"):
        out = _run_colour(tmp_path, """
          const rgba = new Uint8ClampedArray(IN.rgba);
          out.d = Array.from(C.distanceMap(rgba, IN.n, 1, IN.samples, IN.space));
        """, {"rgba": rgba.reshape(-1).tolist(), "n": n, "samples": samples.tolist(), "space": space})
        got = np.array(out["d"], np.float32)
        if space == "rgb":
            ref = np.empty(n, np.float32)
            for i, px in enumerate(rgb):
                ref[i] = min(np.linalg.norm(px.astype(float) - s.astype(float)) for s in samples)
        elif space == "hsv":
            ref = _hsv_dist_ref(rgb, samples)
        else:
            ref = _lab_dist_ref(rgb, samples)
            assert float(np.max(np.abs(got - _lab_dist_ref(rgb, samples, exact=False)))) <= 1.5, "lab vs OpenCV"
        tol = 0.1 if space == "lab" else 0.75
        assert float(np.max(np.abs(got - ref))) <= tol, space
