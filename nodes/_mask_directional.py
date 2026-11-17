"""Directional mask extend, shared by Face Landmarks (face_mask) and Mask Tools (L9.279 / L9.281, owner 2026-11-17).

extend(mask, up, down, left, right, angle_deg, falloff_px): grow a 0..1 mask toward each side of a frame that is
rotated by angle_deg (0 = the image's own up; positive = counter-clockwise, the way a head tilts to its left in the
picture). Growing is a one-sided max filter, so only that side moves; a soft fall-off ramps the added band from the
original edge (1) to the new edge (0). Negative amounts shrink that side. Pure numpy + OpenCV; no model.
"""
from __future__ import annotations

import math

import numpy as np


def _one_side(m: np.ndarray, n: int, axis: int, toward_low: bool) -> np.ndarray:
    """Grow (n > 0) or shrink (n < 0) a float mask by |n| px toward one side of one axis (the Mask Tools method)."""
    import cv2
    k = abs(int(n))
    if not k:
        return m
    grow = n > 0
    ksize = (k + 1, 1) if axis == 1 else (1, k + 1)                  # cv2 sizes are (width, height)
    # a max filter whose kernel sits on one side of the anchor moves the edge only toward the other side
    anchor_end = (grow and toward_low) or (not grow and not toward_low)
    anchor = (0, 0) if anchor_end else ((k, 0) if axis == 1 else (0, k))
    kernel = np.ones((ksize[1], ksize[0]), np.uint8)
    if grow:
        return cv2.dilate(m, kernel, anchor=anchor, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    return cv2.erode(m, kernel, anchor=anchor, borderType=cv2.BORDER_REPLICATE)


def extend(mask: np.ndarray, up: float = 0, down: float = 0, left: float = 0, right: float = 0,
           angle_deg: float = 0.0, falloff_px: float = 0.0) -> np.ndarray:
    """mask: HxW float32 0..1 (or uint8, returned as float32 0..1). Amounts in pixels of the mask."""
    import cv2
    m = mask.astype(np.float32) / (255.0 if mask.dtype == np.uint8 else 1.0)
    if not any(int(round(v)) for v in (up, down, left, right)):
        return m
    h, w = m.shape
    a = float(angle_deg) % 360.0
    rotated = abs(a) > 1e-3 and abs(a - 360.0) > 1e-3
    if rotated:                                                        # work in the tilted frame, on a padded canvas
        pad = int(math.ceil(math.hypot(h, w) / 2 - min(h, w) / 2)) + int(max(abs(up), abs(down), abs(left),
                                                                             abs(right))) + 2
        big = cv2.copyMakeBorder(m, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
        c = (big.shape[1] / 2.0, big.shape[0] / 2.0)
        rot = cv2.getRotationMatrix2D(c, -a, 1.0)                      # undo the tilt: the frame's up -> image up
        work = cv2.warpAffine(big, rot, (big.shape[1], big.shape[0]), flags=cv2.INTER_LINEAR, borderValue=0)
    else:
        work = m
    base = work.copy()
    for n, axis, low in ((up, 0, True), (down, 0, False), (left, 1, True), (right, 1, False)):
        work = _one_side(work, int(round(n)), axis, low)
    if falloff_px > 0:
        # the added band ramps from 1 at the original edge to 0 at the new edge: distance from the original mask
        added = (work > 0.5) & (base <= 0.5)
        if added.any():
            d = cv2.distanceTransform((base <= 0.5).astype(np.uint8), cv2.DIST_L2, 5)
            ramp = np.clip(1.0 - d / float(falloff_px), 0.0, 1.0)
            work = np.where(added, np.minimum(work, ramp), work)
    if rotated:
        back = cv2.getRotationMatrix2D(c, a, 1.0)
        work = cv2.warpAffine(work, back, (work.shape[1], work.shape[0]), flags=cv2.INTER_LINEAR, borderValue=0)
        work = work[pad:pad + h, pad:pad + w]
    return np.clip(work, 0.0, 1.0).astype(np.float32)
