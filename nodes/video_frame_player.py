"""VideoFramePlayerMEC - lightweight in-graph video scrubber + drag-crop + resize.

Originally a frame scrubber. Extended with:
  - Aspect ratio presets (free / 1:1 / 4:3 / 3:4 / 16:9 / 9:16 / 2:1 / 21:9 / custom / original)
  - Drag-crop overlay on the player canvas (8 handles + interior drag,
    aspect-locked snap, dim-overlay, rule-of-thirds guides)
  - Integrated resize with lanczos / bicubic / bilinear / area / nearest-exact
  - Optional upscale factor on top of the resize
  - Output modes: 'current_frame' (single frame) or 'all_frames' (whole batch
    processed with the same crop + resize so it can chain into samplers).

All operations are pure tensor ops. Crop is normalized in [0,1] so values
survive resolution swaps. Aspect-lock is enforced both client-side (drag)
and server-side (snap final crop to preset).

UX inspiration (no code copied — clean-room implementation, see NOTICE.md):
  - Olm DragCrop (Olli Sorjonen, https://github.com/o-l-l-i/ComfyUI-Olm-DragCrop)
    popularised the in-node drag-rectangle crop UX. Its licence is
    source-available / not OSS, so its code cannot be vendored; this
    implementation was written independently using standard HTML5 canvas
    drag-handle patterns.
  - WhatDreamsCost-ComfyUI 'Load Video UI' (Jonathan Watkins,
    https://github.com/WhatDreamsCost/WhatDreamsCost-ComfyUI) is GPL-3.0;
    its widget layout inspired this node's widget set. No GPL source was
    copied — both the widget definitions and the canvas overlay are original.
"""

from __future__ import annotations

from . import _interrupt_check as _IC

import os
import json
import hashlib
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import folder_paths

from ._is_changed_util import hash_args_and_kwargs


from . import _progress as _PB
_PREVIEW_QUALITY = 80

# Browser preview format. PNG = lossless, exact colors at 8-bit (browser
# can't display deeper than 8-bit anyway). JPEG @ q=95 is provided as a
# fast fallback for huge batches where preview I/O dominates.
_PREVIEW_FORMAT_DEFAULT = "png"


# ----------------------------------------------------------------------
#  Constants
# ----------------------------------------------------------------------

ASPECT_PRESETS = {
    "free":        None,           # user-drag any rectangle
    "original":    None,           # locked to source W/H (computed at runtime)
    "1:1":         1.0,
    "4:3":         4.0 / 3.0,
    "3:4":         3.0 / 4.0,
    "16:9":        16.0 / 9.0,
    "9:16":        9.0 / 16.0,
    "2:1":         2.0,
    "21:9":        21.0 / 9.0,
    "custom":      None,           # uses custom_aspect_w / custom_aspect_h
}

RESIZE_METHODS = ["none", "lanczos", "bicubic", "bilinear", "area", "nearest-exact"]

OUTPUT_MODES = ["current_frame", "all_frames"]

LOOP_MODES = ["once", "loop", "ping-pong"]


# ----------------------------------------------------------------------
#  Resize helper
# ----------------------------------------------------------------------

def _keep_range(out: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
    """Clamp to 0..1 only when the source was SDR; HDR / scene-linear values outside 0..1 are kept (agy review:
    every resize clipped them)."""
    if bool(src.min() >= 0.0) and bool(src.max() <= 1.0):
        return out.clamp(0.0, 1.0)
    return out


def _resize_lanczos_bhwc(image: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    """Lanczos resize for (B,H,W,C). GPU antialiased bicubic for B>4
    (much faster than CPU PIL Lanczos, visually identical at moderate
    scale factors); single frames go through PIL Lanczos for max fidelity."""
    B, H, W, C = image.shape
    if H == target_h and W == target_w:
        return image
    if B > 4:
        img_bchw = image.permute(0, 3, 1, 2)
        try:
            resized = F.interpolate(
                img_bchw, size=(target_h, target_w),
                mode="bicubic", align_corners=False, antialias=True,
            )
        except TypeError:
            resized = F.interpolate(
                img_bchw, size=(target_h, target_w),
                mode="bicubic", align_corners=False,
            )
        return _keep_range(resized.permute(0, 2, 3, 1), image)
    out = []
    for i in range(B):
        _IC.check()
        # agy review: each channel as a 32-bit float image (PIL mode "F" supports Lanczos) - the old 8-bit round
        # trip banded gradients and clipped HDR
        arr = image[i].detach().float().cpu().numpy()
        chans = [np.asarray(Image.fromarray(arr[:, :, c], mode="F").resize((target_w, target_h), Image.LANCZOS))
                 for c in range(C)]
        out.append(torch.from_numpy(np.stack(chans, -1).astype("float32")))
    return _keep_range(torch.stack(out).to(image.device), image)


_TORCH_INTERP = {
    "bicubic": "bicubic",
    "bilinear": "bilinear",
    "area": "area",
    "nearest-exact": "nearest-exact",
}


def _resize_bhwc(image: torch.Tensor, target_h: int, target_w: int, method: str) -> torch.Tensor:
    if method == "none":
        return image
    if image.shape[1] == target_h and image.shape[2] == target_w:
        return image
    if method == "lanczos":
        return _resize_lanczos_bhwc(image, target_h, target_w)
    mode = _TORCH_INTERP.get(method, "bilinear")
    img_bchw = image.permute(0, 3, 1, 2)
    kwargs = {"size": (target_h, target_w), "mode": mode}
    if mode in ("bilinear", "bicubic"):
        kwargs["align_corners"] = False
    return _keep_range(F.interpolate(img_bchw, **kwargs).permute(0, 2, 3, 1), image)


def _aspect_for(preset: str, src_w: int, src_h: int,
                custom_w: float, custom_h: float):
    if preset == "original":
        return (src_w / src_h) if src_h > 0 else None
    if preset == "custom":
        if custom_h <= 0 or custom_w <= 0:
            return None
        return float(custom_w) / float(custom_h)
    return ASPECT_PRESETS.get(preset)


def _snap_crop_to_aspect(cx: float, cy: float, cw: float, ch: float,
                         aspect, src_w: int = 1, src_h: int = 1) -> Tuple[float, float, float, float]:
    """Adjust normalized crop (cx,cy,cw,ch) to the target PIXEL aspect, keeping
    the centre fixed and shrinking the longer dimension. All values in [0,1].
    agy review / c2c_backend_audit: the ratio was taken on the normalised rect, so a 16:9 lock on a 1920x1080 frame
    gave a 1:1-shaped pixel crop; the target is converted to normalised units with the frame size."""
    if aspect is None or aspect <= 0:
        return (cx, cy, cw, ch)
    aspect = float(aspect) * float(src_h) / float(max(src_w, 1))
    cur_ar = cw / max(ch, 1e-6)
    centre_x = cx + cw / 2.0
    centre_y = cy + ch / 2.0
    if cur_ar > aspect:
        cw = ch * aspect
    else:
        ch = cw / aspect
    cx = max(0.0, min(1.0 - cw, centre_x - cw / 2.0))
    cy = max(0.0, min(1.0 - ch, centre_y - ch / 2.0))
    return (cx, cy, cw, ch)


def _apply_crop_bhwc(image: torch.Tensor, cx: float, cy: float, cw: float, ch: float) -> torch.Tensor:
    """Crop (B,H,W,C) by normalized rect. Clamps to image bounds."""
    B, H, W, C = image.shape
    x0 = max(0, min(W - 1, int(round(cx * W))))
    y0 = max(0, min(H - 1, int(round(cy * H))))
    x1 = max(x0 + 1, min(W, int(round((cx + cw) * W))))
    y1 = max(y0 + 1, min(H, int(round((cy + ch) * H))))
    return image[:, y0:y1, x0:x1, :].contiguous()


# ----------------------------------------------------------------------
#  L9.173: rotate + pad (AusBoss Video Crop + Rotate + Pad, mapped onto this node; own implementation) and the
#  source map. Positive degrees = clockwise, as AusBoss; here the rotation turns the picture inside the crop window
#  (what the player's crop rectangle shows), so the drawn crop stays what you get.
# ----------------------------------------------------------------------
PAD_MODES = ("color", "edge", "mirror")
L9173_APPENDED_INPUTS = ("rotate_degrees", "pad_left", "pad_top", "pad_right", "pad_bottom", "pad_mode", "pad_color")
APPENDED_AFTER_A28 = L9173_APPENDED_INPUTS          # read by tests/test_a28_cnp_a_layout.py


def _parse_rgb(text) -> Tuple[float, float, float]:
    t = str(text or "").strip()
    try:
        if t.startswith("#") and len(t) == 7:
            vals = [int(t[i:i + 2], 16) for i in (1, 3, 5)]
        else:
            vals = [float(v) for v in t.replace(";", ",").split(",") if v.strip()]
        if len(vals) != 3 or not all(0 <= v <= 255 for v in vals):
            raise ValueError("need three values 0-255")
    except ValueError as exc:
        raise ValueError(f"VideoFramePlayerMEC: pad_color {text!r} is not 'R, G, B' (0-255) or #RRGGBB ({exc}).") \
            from exc
    return tuple(float(v) / 255.0 for v in vals)


def _rotation_grid(box, deg: float, W: int, H: int, device) -> torch.Tensor:
    """grid_sample grid (1, h, w, 2): output pixel centres of the crop window, rotated by -deg about its centre,
    in the source frame's normalised coordinates (align_corners=False)."""
    x0, y0, w, h = box
    th = np.radians(deg)
    c, s = float(np.cos(th)), float(np.sin(th))
    ys, xs = torch.meshgrid(torch.arange(h, dtype=torch.float32, device=device) + 0.5 - h / 2.0,
                            torch.arange(w, dtype=torch.float32, device=device) + 0.5 - w / 2.0, indexing="ij")
    sx = x0 + w / 2.0 + c * xs + s * ys
    sy = y0 + h / 2.0 - s * xs + c * ys
    return torch.stack([2.0 * sx / W - 1.0, 2.0 * sy / H - 1.0], dim=-1)[None]


def _rotate_pad(src: torch.Tensor, box, deg: float, pads, mode: str, fill):
    """(frames, mask): the crop window rotated (sampled from the whole frame) then padded; mask = 1 where filled."""
    n, H, W, C = src.shape
    x0, y0, w, h = box
    dev = src.device
    fill_t = torch.zeros(C, dtype=torch.float32, device=dev)
    fill_t[:min(3, C)] = torch.tensor(fill[:min(3, C)], dtype=torch.float32, device=dev)
    if abs(deg) > 1e-9:
        grid = _rotation_grid(box, deg, W, H, dev)
        pm = {"color": "zeros", "edge": "border", "mirror": "reflection"}[mode]
        outs, valids = [], []
        for s0 in range(0, n, 16):
            _IC.check()
            x = src[s0:s0 + 16].permute(0, 3, 1, 2).float()
            g = grid.expand(x.shape[0], -1, -1, -1)
            o = F.grid_sample(x, g, mode="bilinear", padding_mode=pm, align_corners=False)
            v = F.grid_sample(torch.ones_like(x[:, :1]), g, mode="bilinear", padding_mode="zeros",
                              align_corners=False)
            if mode == "color":
                o = o + (1.0 - v) * fill_t.view(1, C, 1, 1)
            outs.append(o.permute(0, 2, 3, 1))
            valids.append(v[:, 0])
        img = torch.cat(outs).to(src.dtype)
        mask = (1.0 - torch.cat(valids)).clamp(0.0, 1.0)
        mask = torch.where(mask < 1e-4, torch.zeros_like(mask), mask)
    else:
        img = src[:, y0:y0 + h, x0:x0 + w, :]
        mask = torch.zeros((n, h, w), dtype=torch.float32, device=dev)
    l, t, r, b = pads
    if any(pads):
        if mode == "mirror" and (max(l, r) >= w or max(t, b) >= h):
            raise ValueError(f"VideoFramePlayerMEC: mirror padding needs each pad smaller than the picture "
                             f"({w}x{h}); got left/top/right/bottom {pads}. Use edge or color for wider pads.")
        x = img.permute(0, 3, 1, 2)
        if mode == "color":
            canvas = fill_t.view(1, C, 1, 1).to(x.dtype).expand(n, C, h + t + b, w + l + r).clone()
            canvas[:, :, t:t + h, l:l + w] = x
            x = canvas
        else:
            x = F.pad(x.float(), (l, r, t, b), mode="replicate" if mode == "edge" else "reflect").to(img.dtype)
        img = x.permute(0, 2, 3, 1)
        mask = F.pad(mask, (l, r, t, b), value=1.0)
    return img.contiguous(), mask


def _timecode(frame: int, fps: float) -> str:
    nominal = max(1, int(round(fps)))
    ff = frame % nominal
    secs = frame // nominal
    return f"{secs // 3600:02d}:{(secs // 60) % 60:02d}:{secs % 60:02d}:{ff:02d}"


def _source_map(src_frames, fps: float, box, deg: float, pads, pad_mode: str, pre_size, out_size, W: int, H: int):
    """JSON: per output frame the source frame / seconds / timecode, plus the out -> source pixel matrix."""
    x0, y0, w, h = box
    l, t, r, b = pads
    pw, ph = pre_size                       # size after rotate + pad, before resize
    ow, oh = out_size
    sx, sy = ow / float(max(1, pw)), oh / float(max(1, ph))
    th = np.radians(deg)
    c, s = float(np.cos(th)), float(np.sin(th))
    # out (u, v) -> (u / sx - l - w/2, v / sy - t - h/2) = (dx, dy) -> source centre + R(-deg) (dx, dy)
    cxs, cys = x0 + w / 2.0, y0 + h / 2.0
    a = np.array([[c / sx, s / sy, cxs - c * (l + w / 2.0) - s * (t + h / 2.0)],
                  [-s / sx, c / sy, cys + s * (l + w / 2.0) - c * (t + h / 2.0)]])
    fps = float(fps) if fps and fps > 0 else 24.0
    return json.dumps({
        "fps": fps, "timecode": f"non-drop, {max(1, int(round(fps)))} frames per second, from 00:00:00:00",
        "frames": [{"out": k, "source_frame": int(i), "seconds": round(i / fps, 6), "timecode": _timecode(int(i), fps)}
                   for k, i in enumerate(src_frames)],
        "transform": {"order": "crop -> rotate -> pad -> resize", "source_size": [int(W), int(H)],
                      "crop_px": [int(x0), int(y0), int(w), int(h)], "rotate_degrees": float(deg),
                      "pad_ltrb": [int(l), int(t), int(r), int(b)], "pad_mode": pad_mode,
                      "pre_resize_size": [int(pw), int(ph)], "output_size": [int(ow), int(oh)],
                      "scale": [round(sx, 6), round(sy, 6)],
                      "out_to_source_affine": np.round(a, 8).tolist(),
                      "convention": "continuous pixel coordinates (pixel centres at +0.5)"},
    })


# ----------------------------------------------------------------------
#  Node
# ----------------------------------------------------------------------


def _preview_batch_id(frames, prefix: str) -> str:
    """Content key of a clip's preview thumbnails. A22 L9.136: it hashed the top-left 8x8 pixels of three frames,
    so two letterboxed clips of one size (black corners) shared a key and the player showed the old clip. Now a
    strided grid (about 32 x 32 samples) of EVERY frame - cheap, and any visible change moves the key."""
    import hashlib as _hl
    f = frames.detach()
    H, W = int(f.shape[1]), int(f.shape[2])
    sy, sx = max(1, H // 32), max(1, W // 32)
    grid = f[:, sy // 2::sy, sx // 2::sx, :].float().clamp(-8.0, 8.0)
    q = (grid * 1024.0).round().to(torch.int32).cpu().numpy()
    return _hl.sha256(prefix.encode() + q.tobytes()).hexdigest()[:12]


# L9.262 (design docs/research/L9.262_frame_player_proxy_design.md): above this many frames the browser gets ONE
# all-intra H.264 proxy instead of a still per frame (disk and browser memory bounded by one file / one <video>).
_PROXY_ABOVE_DEFAULT = 120
_PROXY_CRF = "14"          # preview-size all-intra: measured >= 35 dB PSNR on the unit clip (tests/test_l9262_vfp_proxy.py)


def _proxy_above() -> int:
    """Frame-count threshold: env C2C_FRAME_PLAYER_PROXY_ABOVE, else the C2C setting c2c.framePlayer.proxyAbove,
    else 120. 0 (or less) = always stills."""
    try:
        try:
            from . import _c2c_saved_setting as _SS
        except ImportError:  # loaded top-level (tests)
            import _c2c_saved_setting as _SS  # type: ignore
        v = _SS.get("c2c.framePlayer.proxyAbove", _PROXY_ABOVE_DEFAULT,
                    env="C2C_FRAME_PLAYER_PROXY_ABOVE", parse=lambda s: int(float(s)))
        return int(v)
    except Exception:  # noqa: BLE001 - a broken setting never breaks the preview
        raw = os.environ.get("C2C_FRAME_PLAYER_PROXY_ABOVE", "").strip()
        try:
            return int(float(raw)) if raw else _PROXY_ABOVE_DEFAULT
        except ValueError:
            return _PROXY_ABOVE_DEFAULT


def _write_preview_proxy(frames: torch.Tensor, path: str, size: Tuple[int, int], fps: float) -> None:
    """One all-intra (every frame a keyframe, so every browser seek lands on the exact frame) H.264 yuv420p mp4 of
    the whole batch at `size` (w, h, both even), BT.709-tagged, written in chunks to `path` (via a .part file, so an
    interrupted run never leaves a half file that a re-run would reuse). Raises on any failure (caller falls back)."""
    from fractions import Fraction

    import av  # noqa: F401 - ImportError -> caller falls back to stills
    try:
        from ._c2c_preview_encode import preview_frames
    except ImportError:  # loaded top-level (tests)
        from _c2c_preview_encode import preview_frames
    try:
        from .c2c_video.testmedia import _rgb_to_yuv, _tag_bt709
    except (ImportError, ValueError):
        from c2c_core.video.testmedia import _rgb_to_yuv, _tag_bt709  # type: ignore

    B = int(frames.shape[0])
    tw, th = int(size[0]), int(size[1])
    tmp = path + ".part"
    ok = False
    container = av.open(tmp, mode="w", format="mp4", options={"movflags": "+faststart"})
    try:
        stream = container.add_stream("libx264", rate=Fraction(float(fps or 24.0)).limit_denominator(1_000_000))
        stream.width, stream.height, stream.pix_fmt = tw, th, "yuv420p"
        _tag_bt709(stream)
        try:
            stream.codec_context.gop_size = 1
        except Exception:  # noqa: BLE001 - the x264 options below say the same
            pass
        stream.options = {"crf": _PROXY_CRF, "preset": "veryfast", "g": "1", "keyint_min": "1", "bf": "0",
                          "x264-params": "keyint=1:min-keyint=1:scenecut=0"}
        k = 0
        starts = list(range(0, B, 16))
        for s0 in _PB.track(starts, len(starts), "FramePlayer: proxy"):
            _IC.check()
            arrs, why = preview_frames(frames[..., :3], indices=list(range(s0, min(B, s0 + 16))), size=(tw, th))
            if arrs is None:
                raise RuntimeError(f"Frame Player: {why}")
            for arr in arrs:
                vf = _rgb_to_yuv(np.ascontiguousarray(arr[..., :3]), "yuv420p")
                vf.pts = k
                k += 1
                for packet in stream.encode(vf):
                    container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)
        ok = True
    finally:
        container.close()
        if ok:
            os.replace(tmp, path)
        else:
            try:
                os.remove(tmp)
            except OSError:
                pass


class VideoFramePlayerMEC:
    """Video frame player + drag-crop + integrated resize.

    Pipeline:
        frames -> pick (current_frame | all_frames)
              -> optional crop (drag rect, aspect-locked)
              -> optional resize (lanczos / bicubic / bilinear / area / nearest)
              -> optional upscale (multiplier)
              -> output

    Existing widgets keep their v1 behaviour. New widgets default to no-op
    so existing graphs produce identical output.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE", {"tooltip": "Frame batch (B,H,W,C). B=number of frames."}),
                "frame_index": ("INT", {
                    "default": 0, "min": 0, "max": 99999, "step": 1,
                    "tooltip": "Current frame to emit on the IMAGE output. Drag the timeline to scrub."}),
                "output_mode": (OUTPUT_MODES, {
                    "default": "current_frame",
                    "tooltip": "current_frame: emit only the selected frame. all_frames: apply trim+stride+crop+resize to every frame."}),
                # trim / stride / playback
                "frame_start": ("INT", {
                    "default": 0, "min": 0, "max": 99999, "step": 1,
                    "tooltip": "First frame of the trim range. Drag the green marker on the timeline."}),
                "frame_end": ("INT", {
                    "default": -1, "min": -1, "max": 99999, "step": 1,
                    "tooltip": "Last frame of the trim range (inclusive). -1 = last frame. Drag the red marker on the timeline."}),
                "frame_stride": ("INT", {
                    "default": 1, "min": 1, "max": 64, "step": 1,
                    "tooltip": "In 'all_frames' mode, output every Nth frame within the trim range."}),
                "playback_fps": ("FLOAT", {
                    "default": 24.0, "min": 0.1, "max": 240.0, "step": 0.5,
                    "tooltip": "Preview playback speed (frames per second)."}),
                "loop_mode": (LOOP_MODES, {
                    "default": "loop",
                    "tooltip": "Preview playback at end of trim range: once / loop / ping-pong."}),
                # crop
                "crop_enabled": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable the drag-crop rectangle on the preview."}),
                "crop_locked": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Lock the crop rect to prevent accidental drags. Press R on the canvas to reset."}),
                "aspect_ratio": (list(ASPECT_PRESETS.keys()), {
                    "default": "free",
                    "tooltip": "Aspect lock. 'original' = source W:H. 'custom' = custom_aspect_w:custom_aspect_h."}),
                "custom_aspect_w": ("FLOAT", {
                    "default": 16.0, "min": 0.0, "max": 999.0, "step": 0.1,
                    "tooltip": "Custom aspect width (used when aspect_ratio = custom)."}),
                "custom_aspect_h": ("FLOAT", {
                    "default": 9.0, "min": 0.0, "max": 999.0, "step": 0.1,
                    "tooltip": "Custom aspect height (used when aspect_ratio = custom)."}),
                "crop_x": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001,
                    "tooltip": "Crop left edge as fraction of source width [0..1]. Set by dragging the rectangle on the preview."}),
                "crop_y": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001,
                    "tooltip": "Crop top edge as fraction of source height [0..1]."}),
                "crop_w": ("FLOAT", {
                    "default": 1.0, "min": 0.001, "max": 1.0, "step": 0.001,
                    "tooltip": "Crop width as fraction of source width (0..1]."}),
                "crop_h": ("FLOAT", {
                    "default": 1.0, "min": 0.001, "max": 1.0, "step": 0.001,
                    "tooltip": "Crop height as fraction of source height (0..1]."}),
                # resize
                "resize_method": (RESIZE_METHODS, {
                    "default": "none",
                    "tooltip": "Post-crop resize. 'lanczos' = high-quality."}),
                "target_width": ("INT", {
                    "default": 0, "min": 0, "max": 8192, "step": 8,
                    "tooltip": "Target width after crop (0 = keep crop width)."}),
                "target_height": ("INT", {
                    "default": 0, "min": 0, "max": 8192, "step": 8,
                    "tooltip": "Target height after crop (0 = keep crop height)."}),
                "upscale_factor": ("FLOAT", {
                    "default": 1.0, "min": 0.1, "max": 8.0, "step": 0.05,
                    "tooltip": "Multiplier applied AFTER target_width/target_height (1.0 = no upscale)."}),
                # preview
                "preview_width": ("INT", {
                    "default": 480, "min": 96, "max": 1920, "step": 16,
                    "tooltip": "Width (px) of preview frames sent to the browser. Lower = lighter UI; full-resolution IMAGE outputs are unaffected."}),
                "preview_format": (["png", "jpeg"], {
                    "default": _PREVIEW_FORMAT_DEFAULT,
                    "tooltip": "png = lossless, exact colors (recommended). jpeg = smaller payload, slight chroma loss. The IMAGE outputs are always the full-precision source tensor regardless of this setting."}),
                "preview_quality": ("INT", {
                    "default": 95, "min": 30, "max": 100, "step": 5,
                    "tooltip": "JPEG quality (ignored when preview_format=png)."}),
            },
            # L9.173 (A25: AusBoss Video Crop + Rotate + Pad -> this node; spec 47): appended, optional. The
            # defaults (0 / no pad) leave every old output unchanged.
            "optional": {
                "rotate_degrees": ("FLOAT", {
                    "default": 0.0, "min": -180.0, "max": 180.0, "step": 0.1,
                    "tooltip": "Rotate the picture inside the crop window by this many degrees (positive = "
                               "clockwise), about the window's centre. Pixels come from the whole source frame, "
                               "so the corners fill with real picture where there is any; the rest is filled by "
                               "pad_mode and marked in pad_mask. Order: crop -> rotate -> pad -> resize. 0 = off."}),
                "pad_left": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 1,
                             "tooltip": "Pixels added on the left after crop and rotate (e.g. for outpainting); "
                                        "marked 1 in pad_mask."}),
                "pad_top": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 1,
                            "tooltip": "Pixels added at the top after crop and rotate; marked 1 in pad_mask."}),
                "pad_right": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 1,
                              "tooltip": "Pixels added on the right after crop and rotate; marked 1 in pad_mask."}),
                "pad_bottom": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 1,
                               "tooltip": "Pixels added at the bottom after crop and rotate; marked 1 in pad_mask."}),
                "pad_mode": (list(PAD_MODES), {
                    "default": "color",
                    "tooltip": "How padding and empty rotation corners are filled: color = pad_color; edge = the "
                               "nearest edge pixel repeated; mirror = the picture reflected (a pad must then be "
                               "smaller than the picture)."}),
                "pad_color": ("STRING", {
                    "default": "127, 127, 127",
                    "tooltip": "pad_mode color: 'R, G, B' (0-255) or #RRGGBB. Extra channels (alpha) are 0."}),
            },
        }

    RETURN_TYPES = (
        "IMAGE", "INT", "INT",
        "IMAGE", "INT", "INT",
        "INT", "INT", "INT", "INT",
        "INT", "FLOAT",
        # --- new in v2 (May 2026): video-info outputs for downstream automation ---
        "IMAGE", "INT", "INT", "INT", "INT", "FLOAT", "STRING",
        # --- L9.173: rotate / pad mask + source mapping ---
        "MASK", "STRING",
    )
    RETURN_NAMES = (
        "frame", "frame_index", "frame_count",
        "processed", "out_width", "out_height",
        "crop_x_px", "crop_y_px", "crop_w_px", "crop_h_px",
        "trimmed_count", "playback_fps",
        # --- new in v2 ---
        "frames_trimmed", "trim_start_idx", "trim_end_idx",
        "width", "height", "duration", "video_info",
        "pad_mask", "source_map",
    )
    OUTPUT_TOOLTIPS = (
        "Single full-resolution frame at the slider position (1,H,W,C) - pre-crop, pre-resize. Full source precision (no JPEG re-encode).",
        "Echo of the selected frame_index (clamped to trim range).",
        "Total number of frames in the input batch.",
        "Processed output: the selected frame OR the trimmed/strided batch (per output_mode), with crop + resize + upscale applied. Full source precision.",
        "Width (px) of the processed output.",
        "Height (px) of the processed output.",
        "Crop left edge in source pixels.",
        "Crop top edge in source pixels.",
        "Crop width in source pixels.",
        "Crop height in source pixels.",
        "Number of frames emitted on 'processed' after trim+stride (1 in current_frame mode).",
        "Echo of playback_fps (handy as input to video savers).",
        "Trimmed source batch (frame_start..frame_end inclusive, with stride). NO crop/resize applied. Use this to feed downstream video savers without losing source resolution.",
        "Resolved trim start frame index.",
        "Resolved trim end frame index (inclusive).",
        "Source width in pixels.",
        "Source height in pixels.",
        "Total duration of the trimmed range in seconds (trimmed_count / playback_fps).",
        "JSON string with all video metadata: width, height, fps, duration, frame_count, trim_*, crop_*, etc. Wire to a Show Text node or any downstream automation.",
        "Mask matching 'processed': 1 where padding or an empty rotation corner was filled (the area to outpaint / inpaint), 0 on real picture. All 0 without rotate / pad.",
        "Source mapping as JSON: for every frame on 'processed' its source frame number, seconds and timecode (non-drop, at playback_fps), plus the transform (crop px, rotation, padding, resize scale) and a 2x3 matrix taking a 'processed' pixel back to source pixels - for exact downstream reconstruction.",
    )
    FUNCTION = "play"
    CATEGORY = "C2C/Preview"
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Video scrubber + drag-crop + integrated resize. Drag the timeline to scrub frames. "
        "Toggle crop_enabled and drag the rectangle on the preview to crop (aspect-locked when a preset is set). "
        "resize_method + target_width/target_height + upscale_factor produce the final output. "
        "Set output_mode = all_frames to process the whole batch."
    )

    @classmethod
    def IS_CHANGED(cls, frames, frame_index=0, output_mode="current_frame",
                   frame_start=0, frame_end=-1, frame_stride=1,
                   playback_fps=24.0, loop_mode="loop", crop_enabled=False,
                   crop_locked=False, aspect_ratio="free", custom_aspect_w=16.0,
                   custom_aspect_h=9.0, crop_x=0.0, crop_y=0.0, crop_w=1.0,
                   crop_h=1.0, resize_method="none", target_width=0,
                   target_height=0, upscale_factor=1.0, preview_width=480,
                   preview_format=_PREVIEW_FORMAT_DEFAULT, preview_quality=95,
                   **kwargs):
        return hash_args_and_kwargs(
            frames, frame_index, output_mode, frame_start, frame_end,
            frame_stride, playback_fps, loop_mode, crop_enabled, crop_locked,
            aspect_ratio, custom_aspect_w, custom_aspect_h, crop_x, crop_y,
            crop_w, crop_h, resize_method, target_width, target_height,
            upscale_factor, preview_width, preview_format, preview_quality,
            **kwargs,
        )

    @staticmethod
    def _compute_target_size(crop_w_px: int, crop_h_px: int,
                             target_w: int, target_h: int,
                             upscale: float) -> Tuple[int, int]:
        if target_w <= 0 and target_h <= 0:
            tw, th = crop_w_px, crop_h_px
        elif target_w > 0 and target_h <= 0:
            tw = target_w
            th = max(1, int(round(target_w * crop_h_px / max(crop_w_px, 1))))
        elif target_h > 0 and target_w <= 0:
            th = target_h
            tw = max(1, int(round(target_h * crop_w_px / max(crop_h_px, 1))))
        else:
            tw, th = target_w, target_h
        if upscale != 1.0:
            tw = max(1, int(round(tw * upscale)))
            th = max(1, int(round(th * upscale)))
        return (tw, th)

    def play(
        self,
        frames: torch.Tensor,
        frame_index: int = 0,
        output_mode: str = "current_frame",
        frame_start: int = 0,
        frame_end: int = -1,
        frame_stride: int = 1,
        playback_fps: float = 24.0,
        loop_mode: str = "loop",
        crop_enabled: bool = False,
        crop_locked: bool = False,
        aspect_ratio: str = "free",
        custom_aspect_w: float = 16.0,
        custom_aspect_h: float = 9.0,
        crop_x: float = 0.0,
        crop_y: float = 0.0,
        crop_w: float = 1.0,
        crop_h: float = 1.0,
        resize_method: str = "none",
        target_width: int = 0,
        target_height: int = 0,
        upscale_factor: float = 1.0,
        preview_width: int = 480,
        preview_format: str = _PREVIEW_FORMAT_DEFAULT,
        preview_quality: int = 95,
        rotate_degrees: float = 0.0,
        pad_left: int = 0,
        pad_top: int = 0,
        pad_right: int = 0,
        pad_bottom: int = 0,
        pad_mode: str = "color",
        pad_color: str = "127, 127, 127",
    ):
        if frames is None or not hasattr(frames, "shape"):
            raise ValueError("VideoFramePlayerMEC: 'frames' input is missing or invalid (expected IMAGE B,H,W,C).")
        if not isinstance(frames, torch.Tensor) or frames.ndim != 4:
            raise ValueError(f"VideoFramePlayerMEC: expected 4D IMAGE (B,H,W,C), got shape {tuple(frames.shape)}.")
        with torch.no_grad():
            return self._play_impl(
                frames, frame_index, output_mode, frame_start, frame_end,
                frame_stride, playback_fps, loop_mode, crop_enabled,
                crop_locked, aspect_ratio, custom_aspect_w, custom_aspect_h,
                crop_x, crop_y, crop_w, crop_h, resize_method, target_width,
                target_height, upscale_factor, preview_width, preview_format,
                preview_quality, rotate_degrees=rotate_degrees,
                pads=(pad_left, pad_top, pad_right, pad_bottom), pad_mode=pad_mode, pad_color=pad_color,
            )

    def _play_impl(
        self,
        frames: torch.Tensor,
        frame_index: int = 0,
        output_mode: str = "current_frame",
        frame_start: int = 0,
        frame_end: int = -1,
        frame_stride: int = 1,
        playback_fps: float = 24.0,
        loop_mode: str = "loop",
        crop_enabled: bool = False,
        crop_locked: bool = False,
        aspect_ratio: str = "free",
        custom_aspect_w: float = 16.0,
        custom_aspect_h: float = 9.0,
        crop_x: float = 0.0,
        crop_y: float = 0.0,
        crop_w: float = 1.0,
        crop_h: float = 1.0,
        resize_method: str = "none",
        target_width: int = 0,
        target_height: int = 0,
        upscale_factor: float = 1.0,
        preview_width: int = 480,
        preview_format: str = _PREVIEW_FORMAT_DEFAULT,
        preview_quality: int = 95,
        rotate_degrees: float = 0.0,
        pads=(0, 0, 0, 0),
        pad_mode: str = "color",
        pad_color: str = "127, 127, 127",
    ):
        if frames.dim() != 4:
            raise ValueError(f"VideoFramePlayerMEC: expected 4D IMAGE (B,H,W,C), got shape {tuple(frames.shape)}.")

        B, H, W, C = frames.shape
        rot = float(rotate_degrees or 0.0)
        pads = tuple(int(v or 0) for v in pads)
        if any(v < 0 for v in pads):
            raise ValueError(f"VideoFramePlayerMEC: padding must be >= 0, got {pads}.")
        if pad_mode not in PAD_MODES:
            raise ValueError(f"VideoFramePlayerMEC: pad_mode must be one of {list(PAD_MODES)}, got {pad_mode!r}.")
        fill = _parse_rgb(pad_color)
        transform_on = abs(rot) > 1e-9 or any(pads)

        # trim range (inclusive). frame_end < 0 means "last frame".
        fs = max(0, min(B - 1, int(frame_start)))
        fe = (B - 1) if int(frame_end) < 0 else max(0, min(B - 1, int(frame_end)))
        if fe < fs:
            fe = fs
        stride = max(1, int(frame_stride))
        idx = max(fs, min(int(frame_index), fe))

        # preview thumbnails (cached on disk by content digest)
        # Format: PNG = lossless (production default), JPEG = fast.
        fmt = "png" if str(preview_format).lower() == "png" else "jpeg"
        ext = "png" if fmt == "png" else "jpg"
        batch_id = _preview_batch_id(frames, f"{B}x{H}x{W}x{C}_{preview_width}_{fmt}_{preview_quality}_")

        temp_dir = folder_paths.get_temp_directory()
        thumb_h = max(1, int(round(H * preview_width / max(W, 1))))
        names = [f"vfp_{batch_id}_{i:05d}.{ext}" for i in range(B)]
        previews = [{"filename": n, "subfolder": "", "type": "temp"} for n in names]
        # L9.262: a long clip gets ONE all-intra proxy video instead of B stills; short clips are unchanged.
        proxy_above = _proxy_above()
        proxy = None
        proxy_note = None
        still_set = range(B)
        if proxy_above > 0 and B > proxy_above:
            pw2 = max(2, (int(preview_width) // 2) * 2)
            ph2 = max(2, (thumb_h // 2) * 2)
            fps_tag = f"{float(playback_fps):g}".replace(".", "p")
            pid = _preview_batch_id(frames, f"{B}x{H}x{W}x{C}_{pw2}x{ph2}_proxy_")
            pname = f"vfp_{pid}_{fps_tag}fps.mp4"
            ppath = os.path.join(temp_dir, pname)
            try:
                if not os.path.exists(ppath):           # identical clip + size + fps: reuse, no rewrite
                    os.makedirs(temp_dir, exist_ok=True)
                    _write_preview_proxy(frames, ppath, (pw2, ph2), float(playback_fps))
                proxy = [{"filename": pname, "subfolder": "", "type": "temp",
                          "fps": float(playback_fps), "frames": int(B), "width": pw2, "height": ph2}]
            except Exception as e:  # noqa: BLE001 - no PyAV / libx264: capped stills instead
                try:
                    from comfy.model_management import InterruptProcessingException as _IPE
                except Exception:  # noqa: BLE001
                    _IPE = ()
                if _IPE and isinstance(e, _IPE):
                    raise
                # capped, evenly spaced stills; every frame index maps to its nearest kept still
                keep = sorted({int(round(j * (B - 1) / max(1, proxy_above - 1))) for j in range(proxy_above)})
                still_set = keep
                near = np.searchsorted(np.asarray(keep), np.arange(B))
                near = np.clip(near, 0, len(keep) - 1)
                prev = np.clip(near - 1, 0, len(keep) - 1)
                ka = np.asarray(keep)
                pick = np.where(np.abs(ka[prev] - np.arange(B)) <= np.abs(ka[near] - np.arange(B)), prev, near)
                previews = [previews[int(ka[j])] for j in pick]
                proxy_note = (f"Proxy video unavailable ({type(e).__name__}: {e}); showing {len(keep)} of {B} frames "
                              f"as stills (scrubbing shows the nearest kept frame).")
        todo = [] if proxy else [i for i in still_set if not os.path.exists(os.path.join(temp_dir, names[i]))]
        if todo:
            # L9.282: the shared chunked encoder (area downscale on the tensor's device, a few frames at a time,
            # rounded - the old cast truncated, half a code value too dark) + parallel encoding (Pillow releases
            # the GIL). Colour only: an RGBA batch made the JPEG save raise.
            try:
                from ._c2c_preview_encode import preview_frames
            except ImportError:  # loaded top-level (tests)
                from _c2c_preview_encode import preview_frames
            from concurrent.futures import ThreadPoolExecutor

            def _save(i, arr):
                path = os.path.join(temp_dir, names[i])
                tmp = path + ".part"
                pil = Image.fromarray(arr)
                if fmt == "png":
                    pil.save(tmp, "PNG", optimize=False, compress_level=1)
                else:
                    pil.save(tmp, "JPEG", quality=int(preview_quality), optimize=False, subsampling=0)
                os.replace(tmp, path)

            starts = list(range(0, len(todo), 8))
            with ThreadPoolExecutor(max_workers=max(1, min(4, (os.cpu_count() or 2) - 1))) as pool:
                for s0 in _PB.track(starts, len(starts), "FramePlayer: previews"):
                    _IC.check()
                    chunk = todo[s0:s0 + 8]
                    arrs, why = preview_frames(frames[..., :3], indices=chunk, size=(int(preview_width), thumb_h))
                    if arrs is None:
                        raise RuntimeError(f"Frame Player: {why}")
                    list(pool.map(lambda ia: _save(*ia), zip(chunk, arrs)))

        # backward-compat output: untouched current frame
        out_frame = frames[idx:idx + 1].clone()

        # Build trimmed/strided source for 'all_frames' mode
        trimmed = frames[fs:fe + 1:stride] if (fs > 0 or fe < B - 1 or stride > 1) else frames

        # crop
        cx, cy, cw, ch = float(crop_x), float(crop_y), float(crop_w), float(crop_h)
        if not crop_enabled:
            cx, cy, cw, ch = 0.0, 0.0, 1.0, 1.0
        else:
            aspect = _aspect_for(aspect_ratio, W, H, custom_aspect_w, custom_aspect_h)
            cx, cy, cw, ch = _snap_crop_to_aspect(cx, cy, cw, ch, aspect, W, H)
            # clamp
            cx = max(0.0, min(1.0, cx))
            cy = max(0.0, min(1.0, cy))
            cw = max(0.001, min(1.0 - cx, cw))
            ch = max(0.001, min(1.0 - cy, ch))

        src = trimmed if output_mode == "all_frames" else out_frame

        cropping = crop_enabled and (cx > 0.0 or cy > 0.0 or cw < 1.0 or ch < 1.0)
        if cropping:
            processed = _apply_crop_bhwc(src, cx, cy, cw, ch)
        else:
            processed = src.clone() if src is out_frame else src

        # crop pixel coords (post-crop dims)
        crop_x_px = max(0, min(W - 1, int(round(cx * W))))
        crop_y_px = max(0, min(H - 1, int(round(cy * H))))
        crop_w_px = int(processed.shape[2])
        crop_h_px = int(processed.shape[1])

        # L9.173: rotate inside the crop window (sampling the whole source) and pad, with a mask of filled pixels
        box = (crop_x_px, crop_y_px, crop_w_px, crop_h_px) if cropping else (0, 0, W, H)
        pad_mask = None
        if transform_on:
            processed, pad_mask = _rotate_pad(src, box, rot, pads, pad_mode, fill)

        # resize
        cur_h = processed.shape[1]
        cur_w = processed.shape[2]
        tw, th = self._compute_target_size(cur_w, cur_h, int(target_width), int(target_height), float(upscale_factor))
        if (tw, th) != (cur_w, cur_h):
            method = resize_method if resize_method != "none" else "bilinear"
            processed = _resize_bhwc(processed, th, tw, method)
            if pad_mask is not None:
                pad_mask = F.interpolate(pad_mask[:, None], size=(th, tw), mode="bilinear",
                                         align_corners=False)[:, 0].clamp(0.0, 1.0)

        out_h, out_w = int(processed.shape[1]), int(processed.shape[2])
        if pad_mask is None:
            pad_mask = torch.zeros((int(processed.shape[0]), out_h, out_w), dtype=torch.float32,
                                   device=processed.device)

        ui_payload = {
            "frames": previews,
            "frame_count": [B],
            "current_index": [idx],
            "width": [W],
            "height": [H],
            "frame_start": [int(fs)],
            "frame_end": [int(fe)],
            "frame_stride": [int(stride)],
            "playback_fps": [float(playback_fps)],
            "loop_mode": [str(loop_mode)],
            "crop_enabled": [bool(crop_enabled)],
            "crop_locked": [bool(crop_locked)],
            "aspect_ratio": [aspect_ratio],
            "custom_aspect_w": [float(custom_aspect_w)],
            "custom_aspect_h": [float(custom_aspect_h)],
            "crop_x": [cx],
            "crop_y": [cy],
            "crop_w": [cw],
            "crop_h": [ch],
            "out_w": [out_w],
            "out_h": [out_h],
            "crop_x_px": [int(crop_x_px)],
            "crop_y_px": [int(crop_y_px)],
            "crop_w_px": [int(crop_w_px)],
            "crop_h_px": [int(crop_h_px)],
        }
        if proxy:                                       # L9.262: one video replaces the per-frame stills
            del ui_payload["frames"]
            ui_payload["proxy"] = proxy
        if proxy_note:
            ui_payload["proxy_note"] = [proxy_note]

        trimmed_count = int(processed.shape[0])

        # ── v2 outputs (May 2026): video metadata + raw trimmed batch ──
        # frames_trimmed = source slice with no crop / resize, preserving
        # full source resolution and dtype. Useful for feeding into video
        # savers (CombineVideo, VideoHelperSuite Save, etc.) without losing
        # source pixels.
        frames_trimmed = trimmed if trimmed.shape[0] > 0 else frames[idx:idx + 1]
        trim_count_for_dur = max(1, int(frames_trimmed.shape[0]))
        duration_sec = trim_count_for_dur / max(1e-6, float(playback_fps))

        video_info = {
            "width": int(W), "height": int(H),
            "frame_count": int(B),
            "trim_start_idx": int(fs), "trim_end_idx": int(fe),
            "trim_stride": int(stride),
            "trimmed_count": int(frames_trimmed.shape[0]),
            "fps": float(playback_fps),
            "duration_sec": float(duration_sec),
            "loop_mode": str(loop_mode),
            "crop_enabled": bool(crop_enabled),
            "crop_norm": [float(cx), float(cy), float(cw), float(ch)],
            "crop_px":   [int(crop_x_px), int(crop_y_px), int(crop_w_px), int(crop_h_px)],
            "out_width": int(out_w), "out_height": int(out_h),
            "aspect_ratio": str(aspect_ratio),
            "dtype": str(frames.dtype),
            "channels": int(C),
            "preview_format": fmt,
        }
        video_info_json = json.dumps(video_info, indent=2, sort_keys=True)
        src_frames = list(range(fs, fe + 1, stride)) if output_mode == "all_frames" else [idx]
        if output_mode == "all_frames" and not (fs > 0 or fe < B - 1 or stride > 1):
            src_frames = list(range(B))
        source_map = _source_map(src_frames, float(playback_fps), box, rot, pads, pad_mode, (cur_w, cur_h),
                                 (out_w, out_h), W, H)
        if transform_on:
            ui_payload["rotate_degrees"] = [rot]
            ui_payload["pads"] = [list(pads)]

        return {
            "ui": ui_payload,
            "result": (
                out_frame, idx, B,
                processed, out_w, out_h,
                int(crop_x_px), int(crop_y_px),
                int(crop_w_px), int(crop_h_px),
                trimmed_count, float(playback_fps),
                # --- v2 ---
                frames_trimmed, int(fs), int(fe),
                int(W), int(H), float(duration_sec), video_info_json,
                # --- L9.173 ---
                pad_mask, source_map,
            ),
        }


# A28 (ORDERS 29, docs/audit/A28/PARAM_CNP_A.md "VideoFramePlayerMEC"): which controls show when, read by js/_c2c_node_layout.js.
# Display only: every input keeps its name, order, type and default, is saved and reaches the node's function as before.
# The player toolbar owns loop_mode / crop_locked; crop details only with crop on; JPEG quality only for JPEG.
try:
    from ._layout_decl import declare_layout, merge, sections
except ImportError:  # pragma: no cover - loaded outside the pack
    declare_layout = None
if declare_layout is not None:
    declare_layout(VideoFramePlayerMEC, merge(
        sections(Range=["frame_start", "frame_end", "frame_stride"], Player=["playback_fps"],
                 Crop=["crop_enabled", "aspect_ratio", "custom_aspect_w", "custom_aspect_h", "crop_x", "crop_y",
                       "crop_w", "crop_h"],
                 Resize=["resize_method", "target_width", "target_height", "upscale_factor"],
                 Transform=list(L9173_APPENDED_INPUTS),
                 Preview=["preview_width", "preview_format", "preview_quality"]),
        {"loop_mode": {"c2c_hidden": True},
         "crop_locked": {"c2c_hidden": True},
         "aspect_ratio": {"c2c_show_if": {"crop_enabled": [True]}},
         "custom_aspect_w": {"c2c_show_if": {"crop_enabled": [True], "aspect_ratio": ["custom"]}},
         "custom_aspect_h": {"c2c_show_if": {"crop_enabled": [True], "aspect_ratio": ["custom"]}},
         "crop_x": {"c2c_show_if": {"crop_enabled": [True]}},
         "crop_y": {"c2c_show_if": {"crop_enabled": [True]}},
         "crop_w": {"c2c_show_if": {"crop_enabled": [True]}},
         "crop_h": {"c2c_show_if": {"crop_enabled": [True]}},
         "preview_quality": {"c2c_show_if": {"preview_format": ["jpeg"]}},
         "pad_color": {"c2c_show_if": {"pad_mode": ["color"]}}},
    ))
