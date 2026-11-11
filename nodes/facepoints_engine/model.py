# PORTED FROM: https://huggingface.co/myned-ai/facepoints :: model.py @ 6d51ab48af940aca03819840c856984222002e3d
# Copyright 2026 Myned AI (opensource@myned.ai). Licensed under the Apache License, Version 2.0
# (THIRD_PARTY_LICENSES/facepoints-LICENSE.txt). The WEIGHTS this code loads are CC BY-NC 4.0 (non-commercial) and are
# not shipped with this pack.
# Changes for ComfyUI-CustomNodePacks (L9.199): the architecture print became a debug log; load_release takes an
# explicit device and checks the weight file's presence with a plain message; the training-checkpoint loader
# (load_net) was dropped (not needed for inference). The network, the refinement stage and the tensor naming are
# unchanged, so the released model.safetensors loads strictly.
"""The network (timm backbone, 1x1 neck, flattened MLP head, side heads) and its release loader."""

# Every head predicts (x, y, log scale) per point. The third channel is a log scale, clamped to
# LOG_SCALE_RANGE; std_factor() turns the scale into the per-coordinate standard deviation of the
# likelihood the model was trained with.
import json
import logging
import math
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

LOG = logging.getLogger("C2C.Facepoints")
LOG_SCALE_RANGE = (-3.0, 6.0)
WEIGHTS, CONFIG = "model.safetensors", "config.json"  # the two release files
FORMAT = "facepoints-1"  # config.json "format"
MEAN = np.array([0.485, 0.456, 0.406], np.float32)  # ImageNet input normalisation, RGB
STD = np.array([0.229, 0.224, 0.225], np.float32)


def to_tensor(im_bgr):
    """Return a uint8 BGR image as a float (3, H, W) tensor: RGB, ImageNet-normalised."""
    x = (im_bgr[:, :, ::-1].astype(np.float32) / 255.0 - MEAN) / STD
    return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))


def std_factor(likelihood, nu=10.0):
    """Return the factor from the predicted scale to the per-coordinate standard deviation."""
    if likelihood == "laplace":
        return math.sqrt(2.0)
    if likelihood == "studentt":
        return math.sqrt(nu / (nu - 2.0))
    return 1.0


class Refine(nn.Module):
    """Local refinement of a point set: features around each coarse point -> a correction."""

    def __init__(self, n_points, channels, res, ch=32, emb=32, hidden=256):
        super().__init__()
        self.res = res
        self.proj = nn.ModuleList([nn.Conv2d(c, ch, 1) for c in channels])  # one per feature map

        def grid(n, step):
            a = torch.arange(n, dtype=torch.float32) - (n - 1) / 2
            return torch.stack(torch.meshgrid(a, a, indexing="xy"), -1).view(-1, 2) * step

        self.register_buffer("off4", grid(5, 3.0))  # sample offsets in crop px, stride-4 map
        self.register_buffer("off8", grid(3, 6.0))  # and stride-8 map
        self.emb = nn.Embedding(n_points, emb)
        n_in = ch * (len(self.off4) + len(self.off8)) + emb + 1
        self.mlp = nn.Sequential(
            nn.Linear(n_in, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 3),
        )
        nn.init.zeros_(self.mlp[-1].weight)  # refined == coarse at initialisation
        nn.init.zeros_(self.mlp[-1].bias)

    def sample(self, feature_map, points, offsets):
        """Return bilinear samples (B, P, Q * C) of one map at each point plus Q offsets."""
        B, P = points.shape[:2]
        g = (points[:, :, None] + offsets[None, None]) * (2.0 / self.res) - 1.0  # (B, P, Q, 2)
        s = F.grid_sample(
            feature_map, g, mode="bilinear", padding_mode="zeros", align_corners=False
        )
        return s.permute(0, 2, 3, 1).reshape(B, P, -1)

    def forward(self, maps, coarse):
        """Return the refined points (B, P, 3); gradients do not reach the coarse points."""
        c = coarse.detach().float()
        f4, f8 = (p(f).float() for p, f in zip(self.proj, maps, strict=True))
        x = torch.cat(
            [
                self.sample(f4, c[..., :2], self.off4),
                self.sample(f8, c[..., :2], self.off8),
                self.emb.weight[None].expand(len(c), -1, -1),
                c[..., 2:3],
            ],
            -1,
        )
        return c + self.mlp(x)


class Net(nn.Module):
    """Backbone -> 1x1 neck on the stride-`stride` grid -> flatten -> Linear 1024 -> Linear 3P."""

    def __init__(
        self, backbone, res, n_points, pretrained=False, neck=32, stride=8, side_maps=False
    ):
        super().__init__()
        import timm

        self.n_points = n_points
        probe = timm.create_model(backbone, pretrained=False, features_only=True)
        red = probe.feature_info.reduction()  # hrnet_w48: strides 2, 4, 8, 16, 32
        del probe
        # side_maps: also return the stride-4 map (the side heads' refinement reads it)
        idx = (red.index(4), red.index(stride)) if side_maps else (red.index(stride),)
        self.b = timm.create_model(
            backbone, pretrained=pretrained, features_only=True, out_indices=idx
        )
        D = self.b.feature_info.channels()[-1]
        with torch.no_grad():
            g = self.b(torch.zeros(1, 3, res, res))[-1].shape[2:]
        self.neck = nn.Sequential(nn.Conv2d(D, neck, 1), nn.GELU())
        self.h = nn.Sequential(
            nn.Flatten(),
            nn.Linear(neck * g[0] * g[1], 1024),
            nn.GELU(),
            nn.Linear(1024, n_points * 3),
        )
        LOG.debug("facepoints feature grid %s x %s -> neck %s", tuple(g), D, neck)

    @property
    def shared_features(self):
        """The width of the shared layer the point head and the side heads read."""
        return self.h[-1].in_features

    def shared(self, x):
        """Return (shared layer (B, 1024), backbone feature maps) for a batch of crops."""
        maps = self.b(x)
        return self.h[:-1](self.neck(maps[-1])), maps

    def run(self, x, n=None):
        """Return (points (n, P, 3), shared (B, 1024), maps); the point head reads the first n."""
        z, maps = self.shared(x)
        n = len(x) if n is None else n
        return self.h[-1](z[:n]).view(-1, self.n_points, 3), z, maps

    def forward(self, x):
        """Return the points (B, P, 3) in crop px with their log scale."""
        return self.h[-1](self.shared(x)[0]).view(-1, self.n_points, 3)


def side_head(net, n_points):
    """Return a new side head: one Linear on the shared layer -> n_points x (x, y, log scale)."""
    return nn.Linear(net.shared_features, n_points * 3)


def side_refine(net, n_points, res):
    """Return a new refinement stage for a side head of n_points points."""
    return Refine(n_points, net.b.feature_info.channels(), res)


def read_config(path):
    """config.json of a release folder (raises ValueError on an unknown format)."""
    d = path if os.path.isdir(path) else os.path.dirname(os.path.abspath(path))
    with open(os.path.join(d, CONFIG), encoding="utf-8") as fh:
        config = json.load(fh)
    if config.get("format") != FORMAT:
        raise ValueError(f"{d}: unknown facepoints config format {config.get('format')!r} (expected {FORMAT!r})")
    return config


def load_release(path, device="cpu"):
    """Return (net, side heads, refinement stages, config) from a release folder, in eval mode."""
    # The release = model.safetensors + config.json. Tensors are stored flat under the prefixes
    # net. / side.<set>. / side_rf.<set>.; config.json holds the architecture, the point classes in
    # output order, the likelihood and its std factor and the side heads.
    from safetensors.torch import load_file

    d = path if os.path.isdir(path) else os.path.dirname(os.path.abspath(path))
    config = read_config(d)
    t = load_file(os.path.join(d, WEIGHTS), device="cpu")

    def sub(prefix):
        return {k[len(prefix):]: v for k, v in t.items() if k.startswith(prefix)}

    net = Net(config["backbone"], config["input_px"], config["n_points"], False, config["neck"],
              config["stride"], config["side_refine"])
    net.load_state_dict(sub("net."))
    side, side_rf = {}, {}
    for name, K in config["side_heads"].items():
        side[name] = side_head(net, K)
        side[name].load_state_dict(sub(f"side.{name}."))
        side[name].to(device).eval()
        if config["side_refine"]:
            side_rf[name] = side_refine(net, K, config["input_px"])
            side_rf[name].load_state_dict(sub(f"side_rf.{name}."))
            side_rf[name].to(device).eval()
    return net.to(device).eval(), side, side_rf, config
