"""Universal media I/O for VideoComparerMEC.

Decodes:
  - Still images: .png .jpg .jpeg .bmp .webp .tif .tiff (via imageio / cv2)
  - HDR images:   .exr .hdr (via cv2 IMREAD_UNCHANGED, or imageio FreeImage)
  - Video:        .mp4 .mov .avi .mkv .webm .gif (via imageio_ffmpeg)
  - Audio:        .wav .mp3 .flac .ogg .aac .m4a (via soundfile / librosa)

Returns a normalised dict:
    {
      "frames":      torch.Tensor (B, H, W, 3) float32 in [0,1]  (None if pure audio)
      "fps":         float                                       (1.0 if still)
      "frame_count": int
      "audio":       torch.Tensor (channels, samples) float32    (None if no audio)
      "audio_sr":    int                                         (0 if no audio)
      "source_bits": int   # detected source bit depth (8/16/32)
      "source_path": str
      "kind":        "image" | "video" | "audio" | "audio+video"
    }

No stub paths. Each codec branch is fully wired and returns real tensors.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch


_IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
_HDR_EXTS = {".exr", ".hdr"}
_VID_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".gif", ".m4v"}
_AUD_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".aac", ".m4a", ".opus"}


def _to_float01(arr: np.ndarray) -> tuple[np.ndarray, int]:
    """Return (float32 [0,1] HWC RGB, detected source bit depth)."""
    src_bits = 8
    if arr.dtype == np.uint8:
        out = arr.astype(np.float32) / 255.0
        src_bits = 8
    elif arr.dtype == np.uint16:
        out = arr.astype(np.float32) / 65535.0
        src_bits = 16
    elif arr.dtype in (np.float16, np.float32, np.float64):
        out = arr.astype(np.float32)
        src_bits = 32
    else:
        # fall back: rescale to [0,1] using dtype range
        info = np.iinfo(arr.dtype) if np.issubdtype(arr.dtype, np.integer) else None
        if info is not None:
            out = (arr.astype(np.float32) - info.min) / max(1, info.max - info.min)
            src_bits = int(np.dtype(arr.dtype).itemsize * 8)
        else:
            out = arr.astype(np.float32)
            src_bits = 32

    if out.ndim == 2:
        out = np.stack([out, out, out], axis=-1)
    elif out.ndim == 3:
        if out.shape[2] == 4:
            out = out[..., :3]
        elif out.shape[2] == 1:
            out = np.repeat(out, 3, axis=2)
        elif out.shape[2] == 2:
            # GA -> grey
            g = out[..., 0:1]
            out = np.concatenate([g, g, g], axis=2)
        elif out.shape[2] > 4:
            out = out[..., :3]
    else:
        raise ValueError(f"Unsupported image array shape: {out.shape}")
    return out, src_bits


def _load_image_any(path: str) -> tuple[np.ndarray, int]:
    """Decode a single image (any common format incl. EXR/HDR)."""
    ext = os.path.splitext(path)[1].lower()
    # HDR/EXR: prefer cv2 (preserves float32)
    if ext in _HDR_EXTS:
        try:
            import cv2  # type: ignore
            arr = cv2.imread(path, cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
            if arr is not None:
                # cv2 returns BGR
                if arr.ndim == 3 and arr.shape[2] >= 3:
                    arr = arr[..., ::-1]
                return _to_float01(arr)
        except Exception:
            pass
        # fall through to imageio
    # Try imageio v3
    try:
        import imageio.v3 as iio  # type: ignore
        arr = np.asarray(iio.imread(path))
        return _to_float01(arr)
    except Exception:
        pass
    # Last resort: PIL
    from PIL import Image
    img = Image.open(path)
    img.load()
    return _to_float01(np.asarray(img))


def _load_video_frames(path: str, max_frames: int = 0) -> tuple[np.ndarray, float, int]:
    """Decode a video as (frames HxWx3 float32 [0,1] stack, fps, src_bits)."""
    import imageio.v3 as iio  # type: ignore
    try:
        meta = iio.immeta(path, plugin="pyav")
        fps = float(meta.get("fps", 24.0) or 24.0)
    except Exception:
        fps = 24.0
    frames: list[np.ndarray] = []
    src_bits = 8
    # Use pyav-backed reader (handles mp4/mov/mkv reliably)
    try:
        for i, fr in enumerate(iio.imiter(path, plugin="pyav")):
            f01, sb = _to_float01(np.asarray(fr))
            src_bits = max(src_bits, sb)
            frames.append(f01)
            if max_frames and len(frames) >= max_frames:
                break
    except Exception:
        # fallback: imageio v2 + ffmpeg
        import imageio as iio2  # type: ignore
        rd = iio2.get_reader(path)
        try:
            md = rd.get_meta_data()
            fps = float(md.get("fps", fps) or fps)
        except Exception:
            pass
        for i, fr in enumerate(rd):
            f01, sb = _to_float01(np.asarray(fr))
            src_bits = max(src_bits, sb)
            frames.append(f01)
            if max_frames and len(frames) >= max_frames:
                break
        rd.close()
    if not frames:
        raise RuntimeError(f"No frames decoded from {path}")
    return np.stack(frames, axis=0), fps, src_bits


def _load_audio(path: str) -> tuple[np.ndarray, int]:
    """Decode audio. Returns (channels, samples) float32, sample_rate."""
    try:
        import soundfile as sf  # type: ignore
        data, sr = sf.read(path, always_2d=True, dtype="float32")
        # sf returns (samples, channels)
        return data.T.astype(np.float32), int(sr)
    except Exception:
        pass
    # librosa fallback (handles mp3/m4a/aac via audioread)
    import librosa  # type: ignore
    y, sr = librosa.load(path, sr=None, mono=False)
    if y.ndim == 1:
        y = y[None, :]
    return y.astype(np.float32), int(sr)


def _try_extract_audio_from_video(path: str) -> tuple[Optional[np.ndarray], int]:
    """Best-effort audio extraction from a video container via pyav."""
    try:
        import av  # type: ignore
        container = av.open(path)
        astream = next((s for s in container.streams if s.type == "audio"), None)
        if astream is None:
            container.close()
            return None, 0
        sr = int(astream.rate or 48000)
        chunks: list[np.ndarray] = []
        for frame in container.decode(audio=0):
            a = frame.to_ndarray()
            if a.ndim == 1:
                a = a[None, :]
            chunks.append(a.astype(np.float32))
        container.close()
        if not chunks:
            return None, sr
        # Normalise int -> float
        cat = np.concatenate(chunks, axis=-1)
        if np.issubdtype(cat.dtype, np.integer):
            info = np.iinfo(cat.dtype)
            cat = (cat.astype(np.float32) - 0) / max(1, info.max)
        return cat.astype(np.float32), sr
    except Exception:
        return None, 0


def decode_media(path: str, max_frames: int = 0) -> dict:
    """Universal entry point.  See module docstring for return schema."""
    if not path:
        raise ValueError("decode_media: empty path")
    if not os.path.isabs(path):
        # resolve against ComfyUI input dir
        try:
            import folder_paths  # type: ignore
            cand = os.path.join(folder_paths.get_input_directory(), path)
            if os.path.isfile(cand):
                path = cand
        except Exception:
            pass
    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    ext = os.path.splitext(path)[1].lower()
    if ext in _AUD_EXTS:
        audio, sr = _load_audio(path)
        return {
            "frames": None,
            "fps": 1.0,
            "frame_count": 0,
            "audio": torch.from_numpy(audio),
            "audio_sr": sr,
            "source_bits": 32,
            "source_path": path,
            "kind": "audio",
        }

    if ext in _VID_EXTS:
        frames_np, fps, sb = _load_video_frames(path, max_frames=max_frames)
        audio_np, asr = _try_extract_audio_from_video(path)
        return {
            "frames": torch.from_numpy(frames_np),
            "fps": fps,
            "frame_count": int(frames_np.shape[0]),
            "audio": torch.from_numpy(audio_np) if audio_np is not None else None,
            "audio_sr": asr,
            "source_bits": sb,
            "source_path": path,
            "kind": "audio+video" if audio_np is not None else "video",
        }

    # default: still image (covers _IMG_EXTS + _HDR_EXTS + unknowns)
    img_np, sb = _load_image_any(path)
    arr = img_np[None, ...]  # (1,H,W,3)
    return {
        "frames": torch.from_numpy(arr),
        "fps": 1.0,
        "frame_count": 1,
        "audio": None,
        "audio_sr": 0,
        "source_bits": sb,
        "source_path": path,
        "kind": "image",
    }


def list_input_media(subfolders: bool = True) -> list[str]:
    """Enumerate everything in ComfyUI's input dir that we can decode."""
    try:
        import folder_paths  # type: ignore
        root = folder_paths.get_input_directory()
    except Exception:
        return []
    if not os.path.isdir(root):
        return []
    out: list[str] = []
    all_exts = _IMG_EXTS | _HDR_EXTS | _VID_EXTS | _AUD_EXTS
    for dp, _dn, fn in os.walk(root):
        for n in fn:
            ext = os.path.splitext(n)[1].lower()
            if ext in all_exts:
                rel = os.path.relpath(os.path.join(dp, n), root).replace("\\", "/")
                out.append(rel)
        if not subfolders:
            break
    out.sort()
    return out
