# PORTED FROM: https://huggingface.co/myned-ai/facepoints :: detect.py @ 6d51ab48af940aca03819840c856984222002e3d
# Copyright 2026 Myned AI (opensource@myned.ai). Licensed under the Apache License, Version 2.0
# (THIRD_PARTY_LICENSES/facepoints-LICENSE.txt).
# Changes for ComfyUI-CustomNodePacks (L9.199): several faces go through the network as ONE batch
# (forward_many / detect_many; the per-face maths is upstream's, the single-face API is kept); the device and the
# model are handed in by the node (ComfyUI's device, a cached release); to(device) for offloading.
"""Inference: the released model on faces -> 620 GNM points with std, and the 68 / 98 sets."""

# The deployment contract: a square box (x0, y0, side) in image px -> warpAffine to res x res
# (INTER_AREA, black border) -> ImageNet-normalised RGB -> Net -> (x, y, log scale) per point in
# crop px -> image px (x side / res). std = exp(log scale) x the likelihood's std factor
# (config.json): one standard deviation per point, in image px, valid for either coordinate.
# detect() runs `passes` forwards: the given box first, then the box of the predicted mesh points
# x SELF_BOX, because the face finder's box is only a start and the output moves with the box.
import cv2
import numpy as np
import torch

from .model import load_release, to_tensor

SELF_BOX = 1.25  # side of the self-box = the mesh points' extent x this
MESH_CLASSES = ("lmk", "skin", "lid")  # the points on the skin surface frame the self-box


def square_box(x, y, w, h, factor=1.1):
    """Return the square (x0, y0, side) around a face box (x, y, w, h), side = factor x max."""
    side = factor * max(w, h)
    return (x + w / 2 - side / 2, y + h / 2 - side / 2, side)


def box_of_points(uv, factor=SELF_BOX):
    """Return the square (x0, y0, side) around points uv (n, 2): side = their extent x factor."""
    lo, hi = uv.min(0), uv.max(0)
    side = float((hi - lo).max() * factor)
    c = (lo + hi) / 2
    return (float(c[0] - side / 2), float(c[1] - side / 2), side)


class Detector:
    """The released model, loaded once; detect(image, box) returns the points of one face."""

    def __init__(self, model_dir, device=None, loaded=None):
        self.device = str(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.net, self.side, self.side_rf, self.config = loaded or load_release(model_dir, self.device)
        self.cls = np.array(self.config["classes"])
        self.res = int(self.config["input_px"])
        self.std_factor = float(self.config["std_factor"])
        self.log_scale_range = tuple(self.config["log_scale_range"])
        self.mesh = np.isin(self.cls, MESH_CLASSES)

    def to(self, device):
        """Move the network (and its side heads) to `device`; returns self."""
        self.device = str(device)
        self.net.to(device)
        for m in (*self.side.values(), *self.side_rf.values()):
            m.to(device)
        return self

    def crop(self, image_bgr, box):
        """Return the model's input tensor (3, res, res) for the box."""
        x0, y0, side = box
        s = self.res / side
        M = np.array([[s, 0, -x0 * s], [0, s, -y0 * s]])
        crop = cv2.warpAffine(
            image_bgr, M, (self.res, self.res), flags=cv2.INTER_AREA, borderValue=(0, 0, 0)
        )
        return to_tensor(crop)

    def forward_many(self, images_bgr, boxes):
        """One batched pass: a list of {"points": (620, 2), "std": (620,), "<set>": (K, 2)} in image px."""
        if not boxes:
            return []
        x = torch.stack([self.crop(im, b) for im, b in zip(images_bgr, boxes, strict=True)]).to(self.device)
        kind = self.device.split(":")[0]
        outs = []
        with torch.no_grad(), torch.autocast(kind, dtype=torch.float16, enabled=kind == "cuda"):
            points, shared, maps = self.net.run(x)
            o = points.float().cpu().numpy()
            sides = {}
            for name, head in self.side.items():
                so = head(shared).view(len(boxes), -1, 3)
                if name in self.side_rf:
                    so = self.side_rf[name]([m.float() for m in maps], so.float())
                sides[name] = so.float()[:, :, :2].cpu().numpy()
        for i, (x0, y0, side) in enumerate(boxes):
            s = side / self.res
            out = {
                "points": o[i, :, :2] * s + [x0, y0],
                "std": np.exp(np.clip(o[i, :, 2], *self.log_scale_range)) * s * self.std_factor,
            }
            for name, so in sides.items():
                out[name] = so[i] * s + [x0, y0]
            outs.append(out)
        return outs

    def forward(self, image_bgr, box):
        """Return one pass: {"points": (620, 2), "std": (620,), "<set>": (K, 2) ...} in image px."""
        return self.forward_many([image_bgr], [box])[0]

    def detect_many(self, images_bgr, boxes, passes=3):
        """detect() for several faces at once (batched): each re-framed on its own previous points."""
        outs = self.forward_many(images_bgr, boxes)
        for _ in range(passes - 1):
            outs = self.forward_many(images_bgr, [box_of_points(o["points"][self.mesh]) for o in outs])
        for o, b in zip(outs, boxes, strict=True):
            o["box"] = b
        return outs

    def detect(self, image_bgr, box, passes=3):
        """Return the last of `passes` forwards, each re-framed on the previous points, plus box."""
        return self.detect_many([image_bgr], [box], passes)[0]
