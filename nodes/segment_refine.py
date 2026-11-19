"""SAMRefiner-style mask refinement on ComfyUI's own SAM 3 (L7.123 slice 3, opt-in, EXPERIMENTAL until measured).

Method: Lin et al., "SAMRefiner: Taming Segment Anything Model for Universal Mask Refinement", ICLR 2025,
arXiv 2502.06756 - implemented here from the paper's text (section 3.2, appendix A.2 / C.2), no code copied (the
authors' repository was not opened; its licence is unchecked). From a coarse mask, three prompts are mined and SAM
re-decodes:
  * distance-guided points: the positive point is the foreground pixel farthest from the background (distance
    transform maximum); the negative point is the background pixel farthest from the foreground INSIDE the mask's
    tight box (none when the box holds no background);
  * the box: the tight box of the coarse mask. The paper's context-aware elastic box (CEBox) grows it where SAM's
    image features keep matching the object; it needs the image embedding, which this path does not expose, so this
    is the paper's lambda = 0 baseline (its own ablation: box AP 36.7 -> 38.2 with CEBox - small);
  * a Gaussian-style mask prompt: GM(x, y) = omega * exp(-((x - x0)^2 + (y - y0)^2) / (|FG| * gamma)) on the
    foreground, centred on the positive point, omega = 15, gamma = 4 (paper defaults). The paper leaves the
    background value open; here it is -omega (a confident "not the object" logit) - a choice, recorded.
Cascade: `iterations` passes; each pass rebuilds the points, box and Gaussian from the previous pass's mask
(appendix B.2: the Gaussian is the mask prompt of every iteration).

Apache-2.0, C2C.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch

OMEGA, GAMMA = 15.0, 4.0


def mine_prompts(mask: np.ndarray) -> Optional[dict]:
    """Points, box and Gaussian mask prompt from a binary HxW mask (frame pixels). None for an empty mask."""
    import cv2
    fg = (np.asarray(mask) > 0.5).astype(np.uint8)
    n = int(fg.sum())
    if n == 0:
        return None
    H, W = fg.shape
    dist_fg = cv2.distanceTransform(fg, cv2.DIST_L2, 5)
    py, px = np.unravel_index(int(np.argmax(dist_fg)), dist_fg.shape)
    ys, xs = np.nonzero(fg)
    x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
    neg = None
    inside_bg = np.zeros_like(fg)
    inside_bg[y0:y1, x0:x1] = 1 - fg[y0:y1, x0:x1]
    if inside_bg.any():
        dist_bg = cv2.distanceTransform((1 - fg).astype(np.uint8), cv2.DIST_L2, 5) * inside_bg
        ny, nx = np.unravel_index(int(np.argmax(dist_bg)), dist_bg.shape)
        neg = (float(nx), float(ny))
    yy, xx = np.mgrid[0:H, 0:W]
    g = OMEGA * np.exp(-((xx - px) ** 2 + (yy - py) ** 2) / (n * GAMMA))
    gm = np.where(fg > 0, g, -OMEGA).astype(np.float32)
    return {"pos": (float(px), float(py)), "neg": neg, "box": (x0, y0, x1, y1), "gaussian": gm}


def refine(core, frame_hwc: np.ndarray, mask: np.ndarray, iterations: int = 2) -> Tuple[np.ndarray, dict]:
    """Refine one object mask with SAM 3's decoder. `core` is the CoreSAM3 wrapper (its _prepare / SIDE)."""
    import torch.nn.functional as F
    H, W = int(frame_hwc.shape[0]), int(frame_hwc.shape[1])
    cur = (np.asarray(mask, np.float32) > 0.5).astype(np.float32)
    if not cur.any():
        return cur, {"skipped": "empty mask"}
    net, frame, _img, device, dtype = core._prepare(frame_hwc)
    s = float(core.SIDE)
    log = []
    for it in range(max(1, int(iterations))):
        pr = mine_prompts(cur)
        if pr is None:
            break
        pts = [pr["pos"]] + ([pr["neg"]] if pr["neg"] else [])
        labels = [1] + ([0] if pr["neg"] else [])
        point_inputs = {"point_coords": torch.tensor([[[x / W * s, y / H * s] for x, y in pts]], dtype=dtype,
                                                     device=device),
                        "point_labels": torch.tensor([labels], dtype=torch.int32, device=device)}
        x0, y0, x1, y1 = pr["box"]
        box = torch.tensor([[[x0 / W * s, y0 / H * s], [x1 / W * s, y1 / H * s]]], dtype=dtype, device=device)
        gm = torch.from_numpy(pr["gaussian"])[None, None]
        gm = F.interpolate(gm, size=(int(s), int(s)), mode="bilinear", align_corners=False).to(device, dtype)
        logit = net.forward_segment(frame, point_inputs=point_inputs, box_inputs=box, mask_inputs=gm)
        logit = F.interpolate(logit.float(), size=(H, W), mode="bilinear", align_corners=False)[0, 0]
        new = (logit > 0).float().cpu().numpy()
        if not new.any():                                  # SAM lost the object: keep the last mask
            log.append({"iteration": it, "kept_previous": True})
            break
        log.append({"iteration": it, "pos": pr["pos"], "neg": pr["neg"], "box": list(pr["box"]),
                    "area_before": int(cur.sum()), "area_after": int(new.sum())})
        cur = new
    return cur, {"method": "SAMRefiner (arXiv 2502.06756), tight box (no CEBox)", "passes": log}
