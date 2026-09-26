"""ComfyUI media conversions: IMAGE tensors ([B, H, W, C] float32 0-1) ↔ PNG
bytes / remote URLs, plus the native AUDIO (waveform dict) and VIDEO
(comfy_api VideoFromFile) types for the audio/video nodes."""

import base64
import io
import os
import tempfile
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image

from . import api, config
from .mcp import McpError

# Tried in order once the lossless encode overflows the upload ceiling. Below
# 80 the recompression starts showing on flat gradients, so a frame that still
# doesn't fit is reported instead of degraded further.
_JPEG_QUALITIES = (95, 90, 85, 80)

# A MASK value at or above this counts as selected when flattening to the pure
# black-and-white mask the inpainting provider demands.
MASK_SELECTED_THRESHOLD = 0.5


def _first_frame_pil(image: torch.Tensor) -> Image.Image:
    frame = image[0] if image.dim() == 4 else image
    array = (frame.cpu().numpy().clip(0.0, 1.0) * 255.0).round().astype(np.uint8)
    return Image.fromarray(array)


def _encode(pil: Image.Image, image_format: str, **options) -> bytes:
    buffer = io.BytesIO()
    pil.save(buffer, format=image_format, **options)
    return buffer.getvalue()


def _encode_jpeg(pil: Image.Image, quality: int) -> bytes:
    """4:4:4 chroma so the recompression doesn't smear coloured edges. No
    `optimize`: it buys ~3% and makes Pillow buffer the whole scan, which a
    large noisy frame overflows ("broken data stream when writing image file")."""
    return _encode(pil, "JPEG", quality=quality, subsampling=0)


def image_upload_payload(image: torch.Tensor) -> Tuple[bytes, str]:
    """Uploadable (bytes, mime) for an IMAGE tensor. PNG is preferred — it is
    lossless and keeps alpha — but a lossless encode of a 6K+ photograph runs
    past the server's 25MB ceiling, and rejecting an otherwise valid graph over
    the container choice is worse than re-encoding the same pixels as JPEG.
    Dimensions are never touched: a retouch mask is validated against them."""
    pil = _first_frame_pil(image)
    png = _encode(pil, "PNG")
    if len(png) <= config.MAX_IMAGE_UPLOAD_BYTES:
        return png, "image/png"
    if "A" in pil.getbands():
        # JPEG has no alpha — flattening a cutout silently is not a fallback.
        raise McpError(_too_large_message(pil, len(png)))
    for quality in _JPEG_QUALITIES:
        data = _encode_jpeg(pil, quality)
        if len(data) <= config.MAX_IMAGE_UPLOAD_BYTES:
            return data, "image/jpeg"
    raise McpError(_too_large_message(pil, len(png)))


def _too_large_message(pil: Image.Image, size: int) -> str:
    limit_mb = config.MAX_IMAGE_UPLOAD_BYTES // (1024 * 1024)
    return (
        f"This {pil.width}x{pil.height} image encodes to {size / (1024 * 1024):.1f}MB and Magnific "
        f"accepts at most {limit_mb}MB — put an Upscale Image (ImageScale) node before this one to "
        "reduce its pixel size"
    )


def mask_frame(mask: torch.Tensor) -> torch.Tensor:
    """First MASK of the batch as [H, W]. The MASK convention is [B, H, W],
    but some nodes emit [B, 1, H, W] (an extra channel dim) — peeling leading
    dims until 2D handles both, plus a bare [H, W]."""
    frame = mask
    while frame.dim() > 2:
        frame = frame[0]
    return frame


def _mask_array(mask: torch.Tensor) -> np.ndarray:
    """First MASK of the batch as a strictly binary [H, W] uint8 array.

    The inpainting provider refuses anything in between ("a mask must ONLY
    contain black and white pixels"), and grey is the norm on the way in: a
    painted mask is antialiased at its edges, MASK inputs derived from an image
    channel are continuous, and fit_for_retouch's own resampling reintroduces
    edge grey even on a clean mask. ComfyUI's 1.0 = selected matches the
    server's white = change, so the split is at half."""
    frame = mask_frame(mask)
    values = frame.cpu().numpy().clip(0.0, 1.0)
    return np.where(values >= MASK_SELECTED_THRESHOLD, 255, 0).astype(np.uint8)


def mask_to_png_bytes(mask: torch.Tensor) -> bytes:
    return _encode(Image.fromarray(_mask_array(mask), mode="L"), "PNG")


def mask_has_selection(mask: torch.Tensor) -> bool:
    """Whether anything survives the binary split — a mask that is entirely
    below the threshold reads as painted to a user but uploads as pure black."""
    return bool(_mask_array(mask).any())


def _floor_to_8(value: float) -> int:
    return max(8, int(value) // 8 * 8)


def fit_for_retouch(image: torch.Tensor, mask: torch.Tensor):
    """Image + mask resized together to what the retouch pipeline accepts, or
    returned untouched when they already comply. Two constraints:

    Longest edge at most config.RETOUCH_MAX_EDGE. Retouch renders inside the
    HTTP request (the renderer fetches both assets, inverts the mask and base64s
    the pair to the provider), so a full-resolution photo doesn't fail on
    quality — it hits the 30s max_execution_time and comes back as an opaque
    500. The web editor never sends the full image either: it patches at 1024,
    or 2048-4096 for the full-output models.

    Both sides a multiple of 8, the same rule the web client applies to its own
    seed/mask exports ("the render API rejects non-multiple-of-8 dimensions").
    Off-grid dimensions get realigned downstream, and that resampling reaches
    the provider as a mask with grey in it — which it refuses outright.

    Image and mask take the same factor: the server rejects a misaligned pair."""
    frame = image[0] if image.dim() == 4 else image
    height, width = int(frame.shape[0]), int(frame.shape[1])
    ratio = min(1.0, config.RETOUCH_MAX_EDGE / max(height, width))
    size = (_floor_to_8(width * ratio), _floor_to_8(height * ratio))
    if size == (width, height):
        return image, mask
    # Band count preserved, unlike _pil_to_tensor's RGB conversion: dropping
    # alpha here would flatten a cutout source silently AND hide it from
    # image_upload_payload, whose whole point is to refuse that trade.
    scaled_image = _keep_bands_to_tensor(_first_frame_pil(image).resize(size, Image.LANCZOS)).unsqueeze(0)
    mask_pil = Image.fromarray(
        (mask_frame(mask).cpu().numpy().clip(0.0, 1.0) * 255.0).round().astype(np.uint8), mode="L"
    )
    scaled_mask = torch.from_numpy(
        np.asarray(mask_pil.resize(size, Image.LANCZOS), dtype=np.float32) / 255.0
    ).unsqueeze(0)
    return scaled_image, scaled_mask


def _keep_bands_to_tensor(pil: Image.Image) -> torch.Tensor:
    if pil.mode not in ("RGB", "RGBA"):
        pil = pil.convert("RGB")
    array = np.asarray(pil, dtype=np.float32) / 255.0
    return torch.from_numpy(array)


def _pil_to_tensor(pil: Image.Image) -> torch.Tensor:
    array = np.asarray(pil.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array)


def url_to_image_and_mask(url: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """RGB batch + alpha-derived MASK — for cutout results (remove background),
    whose PNG alpha would be silently dropped by the RGB batch conversion."""
    pil = Image.open(io.BytesIO(api.download_bytes(url)))
    rgb = _pil_to_tensor(pil).unsqueeze(0)
    if "A" in pil.getbands():
        alpha = np.asarray(pil.getchannel("A"), dtype=np.float32) / 255.0
        mask = torch.from_numpy(alpha).unsqueeze(0)
    else:
        mask = torch.ones((1, pil.height, pil.width))
    return rgb, mask


def _media_dir() -> str:
    try:
        import folder_paths  # ComfyUI runtime

        directory = folder_paths.get_temp_directory()
    except ImportError:
        directory = tempfile.gettempdir()
    os.makedirs(directory, exist_ok=True)
    return directory


def _decode_audio_torchaudio(data: bytes):
    import torchaudio

    return torchaudio.load(io.BytesIO(data))


def _decode_audio_soundfile(data: bytes):
    import soundfile

    # soundfile returns [T, C]; ComfyUI wants [C, T].
    array, sample_rate = soundfile.read(io.BytesIO(data), dtype="float32", always_2d=True)
    return torch.from_numpy(array.T.copy()), sample_rate


def _decode_audio_av(data: bytes):
    import av

    with av.open(io.BytesIO(data)) as container:
        stream = container.streams.audio[0]
        # Planar float keeps to_ndarray at [C, T] per frame regardless of the
        # source codec's sample packing. The list-returning resample() (and the
        # None flush) needs PyAV >= 9 — on older ones this decoder just drops
        # out of the fallback chain via the caller's failure aggregation.
        resampler = av.AudioResampler(format="fltp", layout=stream.layout, rate=stream.rate)
        chunks = [
            converted.to_ndarray()
            for frame in container.decode(stream)
            for converted in resampler.resample(frame)
        ]
        chunks.extend(flushed.to_ndarray() for flushed in resampler.resample(None))
        if not chunks:
            raise ValueError("no audio frames decoded")
        return torch.from_numpy(np.concatenate(chunks, axis=1)), int(stream.rate)


# torchaudio is no longer guaranteed by every ComfyUI build (desktop/portable
# installs may lack it, or lack an MP3-capable backend for it), so decoding
# falls through soundfile and av — both regular ComfyUI requirements.
_AUDIO_DECODERS = (
    ("torchaudio", _decode_audio_torchaudio),
    ("soundfile", _decode_audio_soundfile),
    ("av", _decode_audio_av),
)

AUDIO_DECODER_HINT = (
    "No MP3-capable audio decoder is available in this ComfyUI install — "
    "run 'pip install av' (or soundfile / torchaudio) in ComfyUI's Python environment and restart"
)

# ~50ms of MP3 silence (ffmpeg anullsrc, 8kHz mono, 477 bytes). The pre-flight
# decodes it for real: importability alone lies — torchaudio can be present
# without an MP3-capable backend, which is exactly the failure being guarded.
_MP3_PROBE = base64.b64decode(
    "SUQzBAAAAAAAI1RTU0UAAAAPAAADTGF2ZjYyLjEyLjEwMgAAAAAAAAAAAAAA/+M4wAAAAAAAAAAA"
    "AEluZm8AAAAPAAAAAwAAAbAAqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq1dXV1dXV"
    "1dXV1dXV1dXV1dXV1dXV1dXV1dXV1dXV1dXV////////////////////////////////////////"
    "////AAAAAExhdmM2Mi4yOAAAAAAAAAAAAAAAACQC8AAAAAAAAAGw9wpEpwAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA/+MYxAAAAANIAAAAAExBTUUzLjEwMFVV"
    "VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV/+MYxDsAAANI"
    "AAAAAFVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV"
    "VVVVVVVV/+MYxHYAAANIAAAAAFVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV"
    "VVVVVVVVVVVVVVVVVVVVVVVVVVVV"
)

_mp3_capable = None  # cached — the environment can't gain a backend mid-process


def _available_audio_decoders():
    available = []
    for name, decode in _AUDIO_DECODERS:
        try:
            __import__(name)
        except Exception:
            continue
        available.append((name, decode))
    return available


def _mp3_capable_decoders():
    global _mp3_capable
    if _mp3_capable is None:
        _mp3_capable = tuple(
            (name, decode)
            for name, decode in _available_audio_decoders()
            if _decodes_probe(decode)
        )
    return _mp3_capable


def _decodes_probe(decode) -> bool:
    try:
        decode(_MP3_PROBE)
    except Exception:
        return False
    return True


def assert_audio_decodable() -> None:
    """Audio nodes call this BEFORE dispatching: the generation charges credits
    server-side even when the local result conversion is doomed to fail."""
    if not _mp3_capable_decoders():
        raise McpError(AUDIO_DECODER_HINT)


def audio_bytes_to_comfy(data: bytes) -> dict:
    """ComfyUI AUDIO: {"waveform": [B, C, T], "sample_rate": int}."""
    # Every importable decoder is tried, not just the MP3-capable ones — a
    # backend that fails the MP3 probe can still decode a WAV/OGG result.
    decoders = _available_audio_decoders()
    if not decoders:
        raise McpError(AUDIO_DECODER_HINT)
    failures = []
    for name, decode in decoders:
        try:
            waveform, sample_rate = decode(data)
        except Exception as error:
            failures.append(f"{name}: {error}")
            continue
        return {"waveform": waveform.unsqueeze(0), "sample_rate": int(sample_rate)}
    raise McpError("Could not decode the generated audio — " + "; ".join(failures))


def video_bytes_to_comfy(data: bytes, name: str) -> object:
    """ComfyUI VIDEO: the file lands in ComfyUI's temp dir and VideoFromFile
    holds the path (downstream Save/Preview nodes read it lazily)."""
    from comfy_api.input_impl import VideoFromFile  # ComfyUI runtime

    path = os.path.join(_media_dir(), name)
    with open(path, "wb") as handle:
        handle.write(data)
    return VideoFromFile(path)


def video_to_mp4_bytes(video) -> bytes:
    """A VIDEO input (any container) re-saved as an uploadable MP4 file."""
    path = os.path.join(_media_dir(), f"magnific-upload-{id(video)}.mp4")
    video.save_to(path)
    try:
        with open(path, "rb") as handle:
            return handle.read()
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


def urls_to_batch(urls: List[str]) -> torch.Tensor:
    """Download and stack into one batch. Heterogeneous sizes are resized to the
    first image's dimensions — a batch tensor must be rectangular."""
    if not urls:
        raise ValueError("No result URLs to download")
    frames: List[torch.Tensor] = []
    size = None
    for url in urls:
        pil = Image.open(io.BytesIO(api.download_bytes(url)))
        if size is None:
            size = pil.size
        elif pil.size != size:
            pil = pil.resize(size, Image.LANCZOS)
        frames.append(_pil_to_tensor(pil))
    return torch.stack(frames)
