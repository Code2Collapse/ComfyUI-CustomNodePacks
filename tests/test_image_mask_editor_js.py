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
