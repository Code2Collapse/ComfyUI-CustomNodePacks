"""Synthetic test media for c2c_video — no external downloads."""

from __future__ import annotations

import math
import os
from fractions import Fraction
from typing import Literal

import numpy as np

# Frame identity pattern: 48 bits (40-bit frame index + CRC-8), ONE BIT PER
# 16x16 BLOCK, 16 blocks per row. Whole blocks (not single pixels) survive
# lossy H.264 and 4:2:0 chroma; the decoder reads each block's centre only.
_BLOCK = 16
_COLS = 16
_INDEX_BITS = 40
_BITS = _INDEX_BITS + 8
_MIN_W = _COLS * _BLOCK                                   # 256
_MIN_H = -(-_BITS // _COLS) * _BLOCK                      # 48

# PyAV 17 colour tags are integer enums (AVColorPrimaries / TRC / Space / Range)
_BT709 = 1
_RANGE_TV = 1
_RANGE_PC = 2
# BICUBIC | ACCURATE_RND | FULL_CHR_H_INT | FULL_CHR_H_INP (see reader._SWS_FLAGS)
_SWS_FLAGS = 0x4 | 0x40000 | 0x2000 | 0x4000


def _rgb_to_yuv(rgb: np.ndarray, pix_fmt: str):
    """RGB(A) uint8 -> a VideoFrame in ``pix_fmt``, converted with the
    BT.709 limited-range matrix the clip is TAGGED with. A bare
    ``reformat(format=...)`` uses swscale's default BT.601 and would make
    the file lie about its own colour."""
    import av

    frame = av.VideoFrame.from_ndarray(rgb, format="rgba" if rgb.shape[2] == 4 else "rgb24")
    return frame.reformat(format=pix_fmt, dst_colorspace=_BT709, dst_color_range=_RANGE_TV,
                          interpolation=_SWS_FLAGS)


def _tag_bt709(stream) -> None:
    cc = stream.codec_context
    cc.color_primaries = _BT709
    cc.color_trc = _BT709
    cc.colorspace = _BT709
    cc.color_range = _RANGE_TV


def _crc8(value: int, poly: int = 0x07) -> int:
    crc = 0
    for shift in range(_INDEX_BITS - 8, -1, -8):
        crc ^= (value >> shift) & 0xFF
        for _ in range(8):
            crc = ((crc << 1) ^ poly) & 0xFF if crc & 0x80 else (crc << 1) & 0xFF
    return crc


def encode_index_pattern(frame_idx: int, width: int, height: int) -> np.ndarray:
    """RGB uint8 canvas carrying ``frame_idx`` as whole-block bits + CRC-8."""
    if width < _MIN_W or height < _MIN_H:
        raise ValueError(f"canvas must be at least {_MIN_W}x{_MIN_H} for the index pattern")
    idx = int(frame_idx) & ((1 << _INDEX_BITS) - 1)
    payload = (idx << 8) | _crc8(idx)
    img = np.full((height, width, 3), 96, dtype=np.uint8)
    for k in range(_BITS):
        bit = (payload >> (_BITS - 1 - k)) & 1
        r, c = divmod(k, _COLS)
        img[r * _BLOCK:(r + 1) * _BLOCK, c * _BLOCK:(c + 1) * _BLOCK] = 255 if bit else 0
    return img


def decode_index(frame: np.ndarray) -> int:
    """Recover the frame index from a decoded frame (uint8, uint16 or float
    0..1, RGB or RGBA). Raises ValueError when the CRC does not match."""
    arr = np.asarray(frame)
    rgb = arr[..., :3] if arr.ndim == 3 else arr[..., None].repeat(3, axis=-1)
    if rgb.dtype == np.uint16:
        rgb = rgb.astype(np.float32) / 257.0
    elif np.issubdtype(rgb.dtype, np.floating):
        rgb = np.clip(rgb.astype(np.float32), 0.0, 1.0) * 255.0
    gray = rgb.astype(np.float32).mean(axis=-1)
    h, w = gray.shape[:2]
    if h < _MIN_H or w < _MIN_W:
        raise ValueError("frame too small for the index pattern")
    q = _BLOCK // 4
    payload = 0
    for k in range(_BITS):
        r, c = divmod(k, _COLS)
        cell = gray[r * _BLOCK + q:(r + 1) * _BLOCK - q, c * _BLOCK + q:(c + 1) * _BLOCK - q]
        payload = (payload << 1) | (1 if float(cell.mean()) >= 127.5 else 0)
    idx, crc = payload >> 8, payload & 0xFF
    if _crc8(idx) != crc:
        raise ValueError("index pattern CRC mismatch")
    return int(idx)


def flat_colour_patch(width: int, height: int, rgb: tuple[float, float, float]) -> np.ndarray:
    """BT.709 limited-range 8-bit patch values (0..255)."""
    levels = tuple(max(0, min(255, int(round(c * 255)))) for c in rgb)
    patch = np.empty((height, width, 3), dtype=np.uint8)
    patch[..., 0] = levels[0]
    patch[..., 1] = levels[1]
    patch[..., 2] = levels[2]
    return patch


def make_colour_test_clip(path: str, *, width: int = 320, height: int = 240, frames: int = 4) -> str:
    """CFR H.264 clip with flat colour patches tagged bt709 limited range."""
    import av

    patches = [
        (0.75, 0.0, 0.0),
        (0.0, 0.75, 0.0),
        (0.0, 0.0, 0.75),
        (0.75, 0.75, 0.75),
    ]
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    container = av.open(path, mode="w")
    try:
        stream = container.add_stream("libx264", rate=Fraction(24, 1))
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        _tag_bt709(stream)
        stream.options = {"crf": "18", "g": "30", "bf": "2"}
        for i in range(frames):
            rgb = flat_colour_patch(width, height, patches[i % len(patches)])
            frame = _rgb_to_yuv(rgb, "yuv420p")
            frame.pts = i
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)
    finally:
        container.close()
    return path


def make_test_clip(
    variant: Literal["h264_cfr_b", "h264_vfr", "prores4444", "h264_10bit", "ffv1", "h264_cfr_mislabel"],
    path: str,
    *,
    frames: int = 60,
    width: int = 256,
    height: int = 256,
) -> str:
    import av

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    container = av.open(path, mode="w")
    try:
        if variant == "prores4444":
            stream = container.add_stream("prores_ks", rate=Fraction(24, 1))
            stream.pix_fmt = "yuva444p10le"
            stream.options = {"profile": "4444"}
        elif variant == "h264_10bit":
            stream = container.add_stream("libx264", rate=Fraction(24, 1))
            stream.pix_fmt = "yuv420p10le"
            stream.options = {"crf": "18", "profile": "high10"}
        elif variant == "ffv1":
            stream = container.add_stream("ffv1", rate=Fraction(24, 1))
            stream.pix_fmt = "yuv420p"
        else:
            stream = container.add_stream("libx264", rate=Fraction(24, 1))
            stream.pix_fmt = "yuv420p"
            stream.options = {"crf": "18", "g": "30", "bf": "2"}

        stream.width = width
        stream.height = height
        if variant in ("h264_vfr", "h264_cfr_mislabel"):
            # timestamps in milliseconds so frames can sit at irregular times
            stream.time_base = Fraction(1, 1000)
            # the ENCODER's time base too: MP4/MOV reject a stream whose codec
            # context is still 1/24 while frames arrive in milliseconds
            stream.codec_context.time_base = Fraction(1, 1000)
        _tag_bt709(stream)

        # irregular, strictly increasing ms timestamps (deltas 33/50/41/67 ms)
        vfr_deltas = (33, 50, 41, 67, 33, 42)
        vfr_t = 0
        for i in range(frames):
            rgb = encode_index_pattern(i, width, height)
            if variant == "prores4444":
                rgba = np.dstack([rgb, np.full((height, width), 128, dtype=np.uint8)])
                frame = _rgb_to_yuv(rgba, "yuva444p10le")
            elif variant == "h264_10bit":
                frame = _rgb_to_yuv(rgb, "yuv420p10le")
            else:
                frame = _rgb_to_yuv(rgb, "yuv420p")

            if variant in ("h264_vfr", "h264_cfr_mislabel"):
                frame.pts = vfr_t
                frame.time_base = Fraction(1, 1000)
                vfr_t += vfr_deltas[i % len(vfr_deltas)]
            else:
                frame.pts = i

            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)
    finally:
        container.close()
    return path


def make_test_sequence(path_dir: str, *, frames: int = 10, width: int = 256, height: int = 64) -> str:
    """Write EXR float sequence with HDR values and index pattern."""
    import OpenImageIO as oiio  # type: ignore[import-not-found]

    os.makedirs(path_dir, exist_ok=True)
    for i in range(frames):
        rgb = encode_index_pattern(i, width, height).astype(np.float32) / 255.0
        rgb[..., 0] *= 2.5  # HDR red channel > 1.0
        out_path = os.path.join(path_dir, f"frame_{i:04d}.exr")
        spec = oiio.ImageSpec(width, height, 3, oiio.FLOAT)
        out = oiio.ImageOutput.create(out_path)
        if out is None:
            raise RuntimeError(f"cannot write {out_path}: {oiio.geterror()}")
        try:
            if not out.open(out_path, spec):
                raise RuntimeError(out.geterror())
            if not out.write_image(rgb):
                raise RuntimeError(out.geterror())
        finally:
            out.close()
    return path_dir


def expected_flat_rgb_patch(frame_index: int) -> tuple[int, int, int]:
    patches = [
        (0.75, 0.0, 0.0),
        (0.0, 0.75, 0.0),
        (0.0, 0.0, 0.75),
        (0.75, 0.75, 0.75),
    ]
    rgb = patches[frame_index % len(patches)]
    return tuple(max(0, min(255, int(round(c * 255)))) for c in rgb)
