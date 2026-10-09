"""Format definitions and availability for Save Video (C2C)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

QualityParam = Literal["crf", "qscale", "prores_profile", "gif_quality", None]


@dataclass(frozen=True)
class FormatSpec:
    label: str
    container: str
    codec: str
    pix_fmt: str
    profile: str | None
    bit_depth: int
    supports_alpha: bool
    audio_codec: str | None
    sequence: bool
    yuv420: bool
    browser_playable: bool
    quality_param: QualityParam
    ext: str
    file_ext: str | None = None  # sequence frame ext override

    def pix_fmt_for(self, has_alpha: bool) -> str:
        if self.codec == "ffv1" and has_alpha:
            return "gbrap16le"
        if self.codec == "libvpx-vp9" and has_alpha:
            return "yuva420p"
        return self.pix_fmt


class SaveVideoError(ValueError):
    pass


_OIIO_FORMATS = frozenset({"oiio_png16", "oiio_exr_half", "oiio_exr_float"})


def _base_formats() -> dict[str, FormatSpec]:
    return {
        "MP4 H.264": FormatSpec(
            label="MP4 H.264",
            container="mp4",
            codec="libx264",
            pix_fmt="yuv420p",
            profile=None,
            bit_depth=8,
            supports_alpha=False,
            audio_codec="aac",
            sequence=False,
            yuv420=True,
            browser_playable=True,
            quality_param="crf",
            ext="mp4",
        ),
        "MP4 H.265 10-bit": FormatSpec(
            label="MP4 H.265 10-bit",
            container="mp4",
            codec="libx265",
            pix_fmt="yuv420p10le",
            profile=None,
            bit_depth=10,
            supports_alpha=False,
            audio_codec="aac",
            sequence=False,
            yuv420=True,
            browser_playable=False,
            quality_param="crf",
            ext="mp4",
        ),
        "MOV ProRes 422": FormatSpec(
            label="MOV ProRes 422",
            container="mov",
            codec="prores_ks",
            pix_fmt="yuv422p10le",
            profile="2",
            bit_depth=10,
            supports_alpha=False,
            audio_codec="pcm_s24le",
            sequence=False,
            yuv420=False,
            browser_playable=False,
            quality_param="prores_profile",
            ext="mov",
        ),
        "MOV ProRes 422 HQ": FormatSpec(
            label="MOV ProRes 422 HQ",
            container="mov",
            codec="prores_ks",
            pix_fmt="yuv422p10le",
            profile="3",
            bit_depth=10,
            supports_alpha=False,
            audio_codec="pcm_s24le",
            sequence=False,
            yuv420=False,
            browser_playable=False,
            quality_param="prores_profile",
            ext="mov",
        ),
        "MOV ProRes 4444": FormatSpec(
            label="MOV ProRes 4444",
            container="mov",
            codec="prores_ks",
            pix_fmt="yuva444p10le",
            profile="4",
            bit_depth=10,
            supports_alpha=True,
            audio_codec="pcm_s24le",
            sequence=False,
            yuv420=False,
            browser_playable=False,
            quality_param="prores_profile",
            ext="mov",
        ),
        "MOV ProRes 4444 XQ": FormatSpec(
            label="MOV ProRes 4444 XQ",
            container="mov",
            codec="prores_ks",
            pix_fmt="yuva444p10le",
            profile="5",
            bit_depth=10,
            supports_alpha=True,
            audio_codec="pcm_s24le",
            sequence=False,
            yuv420=False,
            browser_playable=False,
            quality_param="prores_profile",
            ext="mov",
        ),
        "MOV DNxHR HQ": FormatSpec(
            label="MOV DNxHR HQ",
            container="mov",
            codec="dnxhd",
            pix_fmt="yuv422p",
            profile="dnxhr_hq",
            bit_depth=8,
            supports_alpha=False,
            audio_codec="pcm_s24le",
            sequence=False,
            yuv420=False,
            browser_playable=False,
            quality_param=None,
            ext="mov",
        ),
        "MOV DNxHR HQX": FormatSpec(
            label="MOV DNxHR HQX",
            container="mov",
            codec="dnxhd",
            pix_fmt="yuv422p10le",
            profile="dnxhr_hqx",
            bit_depth=10,
            supports_alpha=False,
            audio_codec="pcm_s24le",
            sequence=False,
            yuv420=False,
            browser_playable=False,
            quality_param=None,
            ext="mov",
        ),
        "MOV DNxHR 444": FormatSpec(
            label="MOV DNxHR 444",
            container="mov",
            codec="dnxhd",
            pix_fmt="yuv444p10le",
            profile="dnxhr_444",
            bit_depth=10,
            supports_alpha=False,
            audio_codec="pcm_s24le",
            sequence=False,
            yuv420=False,
            browser_playable=False,
            quality_param=None,
            ext="mov",
        ),
        "MKV FFV1 (lossless)": FormatSpec(
            label="MKV FFV1 (lossless)",
            container="mkv",
            codec="ffv1",
            pix_fmt="yuv444p16le",
            profile=None,
            bit_depth=16,
            supports_alpha=True,
            audio_codec="flac",
            sequence=False,
            yuv420=False,
            browser_playable=False,
            quality_param=None,
            ext="mkv",
        ),
        "WebM VP9": FormatSpec(
            label="WebM VP9",
            container="webm",
            codec="libvpx-vp9",
            pix_fmt="yuv420p",
            profile=None,
            bit_depth=8,
            supports_alpha=True,
            audio_codec="libopus",
            sequence=False,
            yuv420=True,
            browser_playable=True,
            quality_param="crf",
            ext="webm",
        ),
        "GIF": FormatSpec(
            label="GIF",
            container="gif",
            codec="pil_gif",
            pix_fmt="pal1",
            profile=None,
            bit_depth=1,
            supports_alpha=False,
            audio_codec=None,
            sequence=False,
            yuv420=False,
            browser_playable=True,
            quality_param="gif_quality",
            ext="gif",
        ),
        "WebP (animated)": FormatSpec(
            label="WebP (animated)",
            container="webp",
            # Pillow, not PyAV: PyAV's libwebp encoder segfaulted the whole process on an RGBA batch (measured
            # 2026-10-09, PyAV 17.0.1) - inside ComfyUI that kills the server.
            codec="pil_webp",
            pix_fmt="yuva420p",
            profile=None,
            bit_depth=8,
            supports_alpha=True,
            audio_codec=None,
            sequence=False,
            yuv420=False,
            browser_playable=True,
            quality_param="qscale",
            ext="webp",
        ),
        "PNG sequence 16-bit": FormatSpec(
            label="PNG sequence 16-bit",
            container="dir",
            codec="oiio_png16",
            pix_fmt="uint16",
            profile=None,
            bit_depth=16,
            supports_alpha=True,
            audio_codec="wav_sidecar",
            sequence=True,
            yuv420=False,
            browser_playable=False,
            quality_param=None,
            ext="dir",
            file_ext="png",
        ),
        "EXR sequence (half)": FormatSpec(
            label="EXR sequence (half)",
            container="dir",
            codec="oiio_exr_half",
            pix_fmt="half",
            profile=None,
            bit_depth=16,
            supports_alpha=True,
            audio_codec="wav_sidecar",
            sequence=True,
            yuv420=False,
            browser_playable=False,
            quality_param=None,
            ext="dir",
            file_ext="exr",
        ),
        "EXR sequence (float)": FormatSpec(
            label="EXR sequence (float)",
            container="dir",
            codec="oiio_exr_float",
            pix_fmt="float",
            profile=None,
            bit_depth=32,
            supports_alpha=True,
            audio_codec="wav_sidecar",
            sequence=True,
            yuv420=False,
            browser_playable=False,
            quality_param=None,
            ext="dir",
            file_ext="exr",
        ),
    }


FORMATS: dict[str, FormatSpec] = _base_formats()

_OIIO_OK: bool | None = None


def _oiio_available() -> bool:
    global _OIIO_OK
    if _OIIO_OK is not None:
        return _OIIO_OK
    try:
        import OpenImageIO  # noqa: F401
        _OIIO_OK = True
    except Exception:
        _OIIO_OK = False
    return _OIIO_OK


def _codec_writable(name: str) -> bool:
    if name in _OIIO_FORMATS:
        return _oiio_available()
    if name in ("pil_gif", "pil_webp"):
        try:
            from PIL import Image, features  # noqa: F401
            return name == "pil_gif" or bool(features.check("webp"))
        except Exception:
            return False
    try:
        import av
        av.Codec(name, "w")
        return True
    except Exception:
        return False


def _audio_writable(name: str | None) -> bool:
    if not name or name == "wav_sidecar":
        return True
    return _codec_writable(name)


def _unavailable_reason(spec: FormatSpec) -> str | None:
    if not _codec_writable(spec.codec):
        if spec.codec in _OIIO_FORMATS:
            return "pip install OpenImageIO"
        if spec.codec in ("pil_gif", "pil_webp"):
            return "pip install Pillow (with WebP support for animated WebP)"
        return f"pip install av with {spec.codec} support (rebuild ffmpeg with the codec enabled)"
    if not _audio_writable(spec.audio_codec):
        return f"audio codec {spec.audio_codec} is not available — install ffmpeg with that encoder"
    return None


def available_formats() -> list[str]:
    out: list[str] = []
    for label, spec in FORMATS.items():
        reason = _unavailable_reason(spec)
        out.append(f"{label}  [unavailable]" if reason else label)
    return out


def resolve_format(label: str) -> FormatSpec:
    base = label.replace("  [unavailable]", "").strip()
    spec = FORMATS.get(base)
    if spec is None:
        raise SaveVideoError(f"Unknown format {label!r}.")
    reason = _unavailable_reason(spec)
    if reason:
        raise SaveVideoError(
            f"Format {base!r} is not available on this machine ({spec.codec}). "
            f"Fix: {reason}. Pick another format from the list."
        )
    return spec


def format_has_quality(spec: FormatSpec) -> bool:
    return spec.quality_param is not None
