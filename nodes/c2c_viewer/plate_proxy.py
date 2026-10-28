"""Viewer helper "Plate Proxy" (L7.114): the plate as small JPEG frames for the Matte viewer's frame view.

Hidden (DEV_ONLY): it only ever runs inside a helper prompt the viewer builds - the nodes feeding Mask + Matting's
`image` input plus this node - so ComfyUI's cache reuses the decode when the user later queues the workflow.
"""
from __future__ import annotations

import os

import torch

from . import _common as C


class C2CViewerPlateProxy:
    DEV_ONLY = True
    CATEGORY = C.HIDDEN_CATEGORY
    DESCRIPTION = ("Viewer helper (hidden): writes the plate as proxy frames (at most 960 px, at most 240 frames) "
                   "for the Mask + Matting viewer. Not meant to be added by hand.")
    FUNCTION = "execute"
    RETURN_TYPES = ()
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"image": ("IMAGE", {"tooltip": "The plate (what feeds Mask + Matting's image input)."})},
            "optional": {
                "max_side": ("INT", {"default": C.MAX_SIDE, "min": 64, "max": 2048, "step": 16,
                                     "tooltip": "Long side of the proxy frames in pixels."}),
                "max_frames": ("INT", {"default": C.MAX_FRAMES, "min": 1, "max": 2000,
                                       "tooltip": "At most this many frames; a longer clip shows every k-th."}),
            },
        }

    def execute(self, image, max_side=C.MAX_SIDE, max_frames=C.MAX_FRAMES):
        if not isinstance(image, torch.Tensor) or image.ndim != 4:
            raise ValueError("The viewer's plate is not an image batch. Connect an IMAGE to Mask + Matting's image.")
        return self._run(image, int(max_side), int(max_frames))

    @staticmethod
    def _run(image, max_side, max_frames):
        from ..mask_matting._stage_cache import fingerprint
        B, H, W = int(image.shape[0]), int(image.shape[1]), int(image.shape[2])
        fp = fingerprint(image)
        pw, ph = C.proxy_size(W, H, max_side)
        idx, step = C.sample_indices(B, max_frames)
        tag = f"plate_{fp[:16]}_{pw}x{ph}_{step}"
        d = C.out_dir(tag)
        frames = []
        for i in idx:
            name = f"f{i:06d}.jpg"
            path = os.path.join(d, name)
            if not os.path.isfile(path):                 # the same plate again: the files are already there
                C.save_jpeg(C.resize_image(image[i], (pw, ph)), path)
            frames.append({**C.file_ref(tag, name), "index": i})
        return C.ui({"kind": "plate", "count": B, "width": W, "height": H, "proxy": [pw, ph], "step": step,
                     "fingerprint": fp, "frames": frames})
