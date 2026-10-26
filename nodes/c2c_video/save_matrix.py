"""Save Video (C2C) v2 format matrix (L9.40, docs/save-video-v2.md): container -> codec -> profile -> bit depth ->
chroma, plus colour tags. Builds the FormatSpec the encoders already consume, and maps the old single-list labels onto
the same choices, so both roads produce the same file.

Availability is per encoder AND pixel format, per backend: the ffmpeg CLI (when it is found and new enough) and PyAV
are probed once; a choice neither can write is listed with the reason, never replaced by another one silently.
"""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass, field

from .save_formats import FORMATS, FormatSpec, SaveVideoError, _nvidia_gpu

CUSTOM = "custom (container / codec)"
CONTAINERS = ("MP4", "MOV", "MKV", "WebM", "GIF", "WebP", "Image sequence")
CHROMAS = ("4:2:0", "4:2:2", "4:4:4", "RGB")
BIT_DEPTHS = ("8", "10", "12", "16", "32 (float)")
PRIMARIES = ("auto", "BT.709", "BT.2020", "Display P3", "BT.601")
TRANSFERS = ("auto", "BT.709", "sRGB", "linear", "PQ (ST 2084)", "HLG", "gamma 2.2")
RANGES = ("auto", "limited", "full")
KILL_SWITCH = "C2C_SAVE_V2_MATRIX"

_EXT = {"MP4": "mp4", "MOV": "mov", "MKV": "mkv", "WebM": "webm", "GIF": "gif", "WebP": "webp", "Image sequence": "dir"}
_DEPTH_SUFFIX = {8: "", 9: "9le", 10: "10le", 12: "12le", 14: "14le", 16: "16le"}


def disabled() -> bool:
    return os.environ.get(KILL_SWITCH, "").strip().lower() in ("0", "off", "false", "no")


def bits_of(bit_depth: str | int) -> int:
    s = str(bit_depth).strip()
    return 32 if s.startswith("32") else int(re.match(r"\d+", s).group(0))


@dataclass(frozen=True)
class Profile:
    opt: str | None                         # the encoder's profile option value (None: none passed)
    modes: tuple[tuple[int, str], ...]      # allowed (bits, chroma)


@dataclass(frozen=True)
class Codec:
    name: str
    enc: str
    containers: tuple[str, ...]
    profiles: dict[str, Profile]
    family: str                             # yuv | rgb | nvenc | pil | seq
    quality: str | None = None              # crf | cq | vp9 | svt | cfhd | qscale | jpeg | gif_quality
    alpha: bool = False
    hw: str | None = None
    tags: dict[str, str] = field(default_factory=dict)   # container -> codec_tag
    note: str = ""


def _m(bits, chromas):
    return tuple((b, c) for b in bits for c in chromas)


_C420, _C422, _C444, _RGB = ("4:2:0",), ("4:2:2",), ("4:4:4",), ("RGB",)

CODECS: dict[str, Codec] = {c.name: c for c in (
    Codec("H.264", "libx264", ("MP4", "MOV", "MKV"), {
        "High": Profile("high", _m((8,), _C420)),
        "Main": Profile("main", _m((8,), _C420)),
        "Baseline": Profile("baseline", _m((8,), _C420)),
        "High 10": Profile("high10", _m((10,), _C420)),
        "High 4:2:2": Profile("high422", _m((8, 10), _C422)),
        "High 4:4:4": Profile("high444", _m((8, 10), _C444)),
    }, "yuv", "crf", note="plays everywhere at High 8-bit 4:2:0"),
    Codec("H.265", "libx265", ("MP4", "MOV", "MKV"), {
        "Main": Profile("main", _m((8,), _C420)),
        "Main 10": Profile("main10", _m((10,), _C420)),
        "Main 12": Profile("main12", _m((12,), _C420)),
        "Main 4:2:2 10": Profile("main422-10", _m((10,), _C422)),
        "Main 4:2:2 12": Profile("main422-12", _m((12,), _C422)),
        "Main 4:4:4": Profile("main444-8", _m((8,), _C444)),
        "Main 4:4:4 10": Profile("main444-10", _m((10,), _C444)),
        "Main 4:4:4 12": Profile("main444-12", _m((12,), _C444)),
    }, "yuv", "crf", tags={"MP4": "hvc1", "MOV": "hvc1"}),
    Codec("AV1 (SVT)", "libsvtav1", ("MP4", "MKV", "WebM"), {
        "Main": Profile(None, _m((8, 10), _C420)),
    }, "yuv", "svt"),
    Codec("H.264 (NVENC)", "h264_nvenc", ("MP4", "MOV", "MKV"), {
        "High": Profile("high", _m((8,), _C420)),
        "High 4:4:4": Profile("high444p", _m((8,), _C444)),
    }, "nvenc", "cq", hw="nvenc"),
    Codec("H.265 (NVENC)", "hevc_nvenc", ("MP4", "MOV", "MKV"), {
        "Main": Profile("main", _m((8,), _C420)),
        "Main 10": Profile("main10", _m((10,), _C420)),
        "Main 4:4:4": Profile("rext", _m((8,), _C444)),
    }, "nvenc", "cq", hw="nvenc", tags={"MP4": "hvc1", "MOV": "hvc1"}),
    Codec("AV1 (NVENC)", "av1_nvenc", ("MP4", "MKV", "WebM"), {
        "Main": Profile(None, _m((8, 10), _C420)),       # ffmpeg 8.1 av1_nvenc refuses profile=main (measured)
    }, "nvenc", "cq", hw="nvenc", note="needs an RTX 40-series or newer"),
    Codec("ProRes", "prores_ks", ("MOV", "MKV"), {
        "422 HQ": Profile("3", _m((10,), _C422)),
        "422": Profile("2", _m((10,), _C422)),
        "422 LT": Profile("1", _m((10,), _C422)),
        "422 Proxy": Profile("0", _m((10,), _C422)),
        "4444": Profile("4", _m((10,), _C444)),
        "4444 XQ": Profile("5", _m((10,), _C444)),
    }, "yuv", alpha=True, note="alpha with 4444 / 4444 XQ"),
    Codec("DNxHR", "dnxhd", ("MOV", "MKV"), {
        "HQ": Profile("dnxhr_hq", _m((8,), _C422)),
        "SQ": Profile("dnxhr_sq", _m((8,), _C422)),
        "LB": Profile("dnxhr_lb", _m((8,), _C422)),
        "HQX": Profile("dnxhr_hqx", _m((10,), _C422)),
        "444": Profile("dnxhr_444", _m((10,), _C444)),
    }, "yuv", note="frames at least 256 x 120"),
    Codec("CineForm", "cfhd", ("MOV",), {
        "YUV 4:2:2 10-bit": Profile(None, _m((10,), _C422)),
        "RGB 12-bit": Profile(None, _m((12,), _RGB)),
    }, "yuv", "cfhd", alpha=True, note="alpha with RGB 12-bit"),
    Codec("FFV1 (lossless)", "ffv1", ("MKV",), {
        "Lossless": Profile(None, _m((8, 10, 12, 16), _C420 + _C422 + _C444 + _RGB)),
    }, "yuv", alpha=True),
    Codec("UT Video (lossless)", "utvideo", ("MKV", "MOV"), {
        "Lossless": Profile(None, _m((8,), _C420 + _C422 + _C444 + _RGB)),
    }, "yuv", alpha=True, note="alpha with RGB"),
    Codec("VP9", "libvpx-vp9", ("WebM", "MKV"), {
        "Profile 0": Profile("0", _m((8,), _C420)),
        "Profile 1": Profile("1", _m((8,), _C422 + _C444)),
        "Profile 2": Profile("2", _m((10, 12), _C420)),
        "Profile 3": Profile("3", _m((10, 12), _C422 + _C444)),
    }, "yuv", "vp9", alpha=True),
    Codec("PNG (lossless)", "png", ("MOV", "MKV"), {
        "RGB": Profile(None, _m((8, 16), _RGB)),
    }, "rgb", alpha=True),
    Codec("QuickTime Animation", "qtrle", ("MOV",), {
        "RGB": Profile(None, _m((8,), _RGB)),
    }, "rgb", alpha=True),
    Codec("Uncompressed 10-bit", "v210", ("MOV",), {
        "4:2:2": Profile(None, _m((10,), _C422)),
    }, "yuv"),
    Codec("GIF", "pil_gif", ("GIF",), {"Palette": Profile(None, _m((8,), _RGB))}, "pil", "gif_quality"),
    Codec("WebP", "pil_webp", ("WebP",), {"Animated": Profile(None, _m((8,), _RGB))}, "pil", "qscale", alpha=True),
    Codec("PNG", "oiio_png", ("Image sequence",), {"RGB": Profile(None, _m((8, 16), _RGB))}, "seq", alpha=True),
    Codec("TIFF", "oiio_tiff", ("Image sequence",), {
        "ZIP": Profile("zip", _m((8, 16, 32), _RGB)),
        "LZW": Profile("lzw", _m((8, 16), _RGB)),
        "None": Profile("none", _m((8, 16, 32), _RGB)),
    }, "seq", alpha=True),
    Codec("DPX", "oiio_dpx", ("Image sequence",), {"RGB": Profile(None, _m((10, 12, 16), _RGB))}, "seq"),
    Codec("EXR", "oiio_exr", ("Image sequence",), {
        "ZIP": Profile("zip", _m((16, 32), _RGB)),
        "PIZ": Profile("piz", _m((16, 32), _RGB)),
        "ZIPS": Profile("zips", _m((16, 32), _RGB)),
        "DWAA": Profile("dwaa", _m((16, 32), _RGB)),
        "DWAB": Profile("dwab", _m((16, 32), _RGB)),
        "None": Profile("none", _m((16, 32), _RGB)),
    }, "seq", alpha=True, note="16 = half float, 32 = float"),
    Codec("JPEG", "oiio_jpeg", ("Image sequence",), {
        "4:2:0": Profile("4:2:0", _m((8,), _C420)),
        "4:4:4": Profile("4:4:4", _m((8,), _C444)),
    }, "seq", "jpeg"),
)}

_AUDIO = {"MP4": "aac", "MKV": "flac", "WebM": "libopus"}
_PRO_AUDIO = {"prores_ks", "dnxhd", "cfhd", "v210", "png", "qtrle", "utvideo", "ffv1"}


def pix_fmt(codec: Codec, bits: int, chroma: str, alpha: bool) -> str | None:
    """The encoder pixel format for (bits, chroma, alpha), or None for writers that are not ffmpeg encoders."""
    if codec.family in ("pil", "seq"):
        return None
    a = alpha and codec.alpha
    if codec.enc == "png":
        return ("rgba" if a else "rgb24") if bits <= 8 else ("rgba64be" if a else "rgb48be")
    if codec.enc == "qtrle":
        return "argb" if a else "rgb24"
    if codec.family == "nvenc":
        if chroma == "4:2:0":
            return "p010le" if bits > 8 else "yuv420p"
        return "yuv444p16le" if bits > 8 else "yuv444p"
    if codec.enc == "prores_ks" and chroma == "4:4:4":
        return "yuva444p10le" if a else "yuv444p10le"
    if codec.enc == "cfhd" and chroma == "RGB":
        return "gbrap12le" if a else "gbrp12le"
    if codec.enc == "ffv1" and chroma == "RGB" and bits <= 8:
        return "bgra" if a else "bgr0"
    if chroma == "RGB":
        base = "gbrap" if a else "gbrp"
        return base + _DEPTH_SUFFIX[bits] if bits > 8 else base
    if a and codec.enc in ("ffv1", "libvpx-vp9"):
        base = {"4:2:0": "yuva420p", "4:2:2": "yuva422p", "4:4:4": "yuva444p"}[chroma]
    else:
        base = {"4:2:0": "yuv420p", "4:2:2": "yuv422p", "4:4:4": "yuv444p"}[chroma]
    return base + _DEPTH_SUFFIX[bits] if bits > 8 else base


# ---- availability ---------------------------------------------------------------------------------------------------
_PIX_CACHE: dict[tuple[str, str], frozenset[str] | None] = {}


def _cli_pix(exe: str, enc: str) -> frozenset[str] | None:
    key = ("cli", enc)
    if key not in _PIX_CACHE:
        try:
            from .fast_decode import _NO_WINDOW
            out = subprocess.run([exe, "-hide_banner", "-h", f"encoder={enc}"], capture_output=True, text=True,
                                 timeout=20, creationflags=_NO_WINDOW).stdout
            m = re.search(r"Supported pixel formats:\s*(.+)", out)
            _PIX_CACHE[key] = frozenset(m.group(1).split()) if m else (frozenset() if "Encoder" in out else None)
        except Exception:  # noqa: BLE001
            _PIX_CACHE[key] = None
    return _PIX_CACHE[key]


def _pyav_pix(enc: str) -> frozenset[str] | None:
    key = ("pyav", enc)
    if key not in _PIX_CACHE:
        try:
            import av
            _PIX_CACHE[key] = frozenset(f.name for f in (av.Codec(enc, "w").video_formats or ()))
        except Exception:  # noqa: BLE001
            _PIX_CACHE[key] = None
    return _PIX_CACHE[key]


def _cli_exe() -> str | None:
    if os.environ.get("C2C_SAVE_FFMPEG", "").strip() == "0":
        return None
    try:
        from .fast_decode import ffmpeg_version, find_ffmpeg
        exe = find_ffmpeg()
        ver = ffmpeg_version(exe) if exe else None
        return exe if ver is not None and ver >= (5, 1) else None
    except Exception:  # noqa: BLE001
        return None


def backend_for(enc: str, pix: str | None) -> str | None:
    """'ffmpeg' or 'pyav': who can write `enc` in `pix` (the CLI first: it is faster), else None."""
    exe = _cli_exe()
    if exe:
        cli = _cli_pix(exe, enc)
        if cli is not None and (pix is None or pix in cli):
            return "ffmpeg"
    py = _pyav_pix(enc)
    if py is not None and (pix is None or pix in py):
        return "pyav"
    return None


def _writer_ok(codec: Codec) -> str | None:
    """None when the codec's writer exists at all, else the reason."""
    if codec.hw == "nvenc" and not _nvidia_gpu():
        return "NVENC needs an NVIDIA GPU with its driver"
    if codec.family == "seq":
        try:
            import OpenImageIO  # noqa: F401
        except Exception:  # noqa: BLE001
            return "pip install OpenImageIO"
        return None
    if codec.family == "pil":
        try:
            from PIL import features
            return None if codec.enc == "pil_gif" or features.check("webp") else "Pillow has no WebP support"
        except Exception:  # noqa: BLE001
            return "pip install Pillow"
    exe = _cli_exe()
    if (exe is None or _cli_pix(exe, codec.enc) is None) and _pyav_pix(codec.enc) is None:
        return f"neither ffmpeg nor PyAV here can encode {codec.enc}"
    return None


def mode_reason(codec: Codec, bits: int, chroma: str, alpha: bool = False) -> str | None:
    """None when (codec, bits, chroma) can be written here, else why not."""
    r = _writer_ok(codec)
    if r:
        return r
    pix = pix_fmt(codec, bits, chroma, alpha)
    if pix is not None and backend_for(codec.enc, pix) is None:
        return f"{codec.enc} here cannot write {pix} (an ffmpeg / libvpx / x265 build with that bit depth is needed)"
    return None


# ---- the matrix the front-end filters with --------------------------------------------------------------------------
def matrix() -> dict:
    out = {"containers": {}, "codecs": {}, "chromas": list(CHROMAS), "bit_depths": list(BIT_DEPTHS),
           "primaries": list(PRIMARIES), "transfers": list(TRANSFERS), "ranges": list(RANGES), "custom": CUSTOM}
    for cname in CONTAINERS:
        out["containers"][cname] = [c.name for c in CODECS.values() if cname in c.containers]
    for c in CODECS.values():
        profs = {}
        for pname, p in c.profiles.items():
            modes = []
            for bits, chroma in p.modes:
                modes.append({"bits": str(bits) if bits < 32 else "32 (float)", "chroma": chroma,
                              "reason": mode_reason(c, bits, chroma), "alpha": alpha_ok(c, bits, chroma)})
            profs[pname] = modes
        out["codecs"][c.name] = {"profiles": profs, "quality": c.quality is not None, "alpha": c.alpha, "enc": c.enc,
                                 "colour": "none" if c.family in ("pil", "seq") else ("rgb" if c.family == "rgb"
                                                                                       else "yuv"),
                                 "reason": _writer_ok(c), "note": c.note}
    # the old single-list labels and the five controls they mean (the front-end converts saved workflows)
    out["legacy"] = {label: [ch.container, ch.codec, ch.profile, ch.bit_depth, ch.chroma] for label, ch in LEGACY.items()}
    out["families"] = list(FAMILIES)
    out["from_controls"] = FROM_CONTROLS
    out["subformats"] = {lab: {"family": d["family"], "five": [d["choice"].container, d["choice"].codec,
                                                               d["choice"].profile, d["choice"].bit_depth,
                                                               d["choice"].chroma],
                               "options": list(d["options"]), "labels": OPTION_LABELS.get(lab, {}),
                               "reason": _writer_ok(CODECS[d["choice"].codec])}
                         for lab, d in SUBFORMATS.items()}
    return out


_MATRIX_CACHE: dict | None = None


def cached_matrix() -> dict:
    global _MATRIX_CACHE
    if _MATRIX_CACHE is None:
        _MATRIX_CACHE = matrix()
    return _MATRIX_CACHE


def all_profiles() -> list[str]:
    seen: list[str] = []
    for c in CODECS.values():
        for p in c.profiles:
            if p not in seen:
                seen.append(p)
    return seen


# ---- building a FormatSpec ------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Choice:
    container: str
    codec: str
    profile: str
    bit_depth: str
    chroma: str


def check_choice(ch: Choice, alpha: bool = False) -> tuple[Codec, Profile, int]:
    if ch.container not in CONTAINERS:
        raise SaveVideoError(f"Unknown container {ch.container!r}.")
    codec = CODECS.get(ch.codec)
    if codec is None:
        raise SaveVideoError(f"Unknown codec {ch.codec!r}.")
    if ch.container not in codec.containers:
        raise SaveVideoError(f"{ch.codec} does not go in {ch.container}; it goes in {', '.join(codec.containers)}.")
    prof = codec.profiles.get(ch.profile)
    if prof is None:
        raise SaveVideoError(f"{ch.codec} has no profile {ch.profile!r}; pick one of {', '.join(codec.profiles)}.")
    bits = bits_of(ch.bit_depth)
    if (bits, ch.chroma) not in prof.modes:
        allowed = ", ".join(f"{b if b < 32 else '32 float'}-bit {c}" for b, c in prof.modes)
        raise SaveVideoError(f"{ch.codec} {ch.profile} writes {allowed}; not {ch.bit_depth}-bit {ch.chroma}.")
    reason = mode_reason(codec, bits, ch.chroma, alpha)
    if reason:
        raise SaveVideoError(f"{ch.codec} {ch.profile} {ch.bit_depth}-bit {ch.chroma} is not available here: {reason}.")
    return codec, prof, bits


def alpha_ok(codec: Codec, bits: int, chroma: str) -> bool:
    """Whether this mode writes an alpha channel - the same rule as FormatSpec.supports_alpha (the UI shows the
    channels option only then: ProRes 4444 yes, ProRes 422 no)."""
    if not codec.alpha:
        return False
    if codec.family in ("pil", "seq"):
        return True
    apix = pix_fmt(codec, bits, chroma, True)
    return apix is not None and apix != pix_fmt(codec, bits, chroma, False)


def build_spec(ch: Choice, alpha: bool = False) -> FormatSpec:
    codec, prof, bits = check_choice(ch, alpha)
    pix = pix_fmt(codec, bits, ch.chroma, False)
    apix = pix_fmt(codec, bits, ch.chroma, True) if codec.alpha else None
    seq = codec.family == "seq"
    if codec.family == "pil":
        legacy = FORMATS["GIF" if codec.enc == "pil_gif" else "WebP (animated)"]
        return legacy
    if seq:
        audio = "wav_sidecar"
    elif codec.enc in _PRO_AUDIO and ch.container == "MOV":
        audio = "pcm_s24le"
    else:
        audio = _AUDIO.get(ch.container, "aac")
    playable = (ch.container == "MP4" and codec.enc in ("libx264", "h264_nvenc") and bits == 8 and ch.chroma == "4:2:0"
                and ch.profile in ("High", "Main", "Baseline")) or \
               (ch.container in ("MP4", "WebM") and codec.enc in ("libsvtav1", "av1_nvenc") and ch.chroma == "4:2:0") or \
               (ch.container == "WebM" and codec.enc == "libvpx-vp9" and ch.profile == "Profile 0")
    file_ext = {"oiio_png": "png", "oiio_tiff": "tif", "oiio_dpx": "dpx", "oiio_exr": "exr", "oiio_jpeg": "jpg"}
    return FormatSpec(
        label=f"{ch.container} {ch.codec} {ch.profile} {bits if bits < 32 else '32f'}-bit {ch.chroma}",
        container=_EXT[ch.container], codec=codec.enc, pix_fmt=pix or ("float" if bits == 32 else f"uint{bits}"),
        profile=prof.opt, bit_depth=bits,
        supports_alpha=codec.alpha if seq else (apix is not None and apix != pix),
        audio_codec=audio, sequence=seq, yuv420=ch.chroma == "4:2:0" and not seq,
        browser_playable=playable, quality_param=codec.quality, ext="dir" if seq else _EXT[ch.container],
        file_ext=file_ext.get(codec.enc), hw=codec.hw, codec_tag=codec.tags.get(ch.container),
        alpha_pix_fmt=apix if apix != pix else None, matrix=True,
    )


# ---- the old single list ---------------------------------------------------------------------------------------------
LEGACY: dict[str, Choice] = {
    "MP4 H.264": Choice("MP4", "H.264", "High", "8", "4:2:0"),
    "MP4 H.265 10-bit": Choice("MP4", "H.265", "Main 10", "10", "4:2:0"),
    "MOV ProRes 422": Choice("MOV", "ProRes", "422", "10", "4:2:2"),
    "MOV ProRes 422 HQ": Choice("MOV", "ProRes", "422 HQ", "10", "4:2:2"),
    "MOV ProRes 4444": Choice("MOV", "ProRes", "4444", "10", "4:4:4"),
    "MOV ProRes 4444 XQ": Choice("MOV", "ProRes", "4444 XQ", "10", "4:4:4"),
    "MOV DNxHR HQ": Choice("MOV", "DNxHR", "HQ", "8", "4:2:2"),
    "MOV DNxHR HQX": Choice("MOV", "DNxHR", "HQX", "10", "4:2:2"),
    "MOV DNxHR 444": Choice("MOV", "DNxHR", "444", "10", "4:4:4"),
    "MKV FFV1 (lossless)": Choice("MKV", "FFV1 (lossless)", "Lossless", "16", "4:4:4"),
    "WebM VP9": Choice("WebM", "VP9", "Profile 0", "8", "4:2:0"),
    "GIF": Choice("GIF", "GIF", "Palette", "8", "RGB"),
    "WebP (animated)": Choice("WebP", "WebP", "Animated", "8", "RGB"),
    "PNG sequence 16-bit": Choice("Image sequence", "PNG", "RGB", "16", "RGB"),
    "EXR sequence (half)": Choice("Image sequence", "EXR", "ZIP", "16", "RGB"),
    "EXR sequence (float)": Choice("Image sequence", "EXR", "ZIP", "32 (float)", "RGB"),
    "MP4 H.265": Choice("MP4", "H.265", "Main", "8", "4:2:0"),
    "MP4 H.264 (NVENC)": Choice("MP4", "H.264 (NVENC)", "High", "8", "4:2:0"),
    "MP4 H.265 (NVENC)": Choice("MP4", "H.265 (NVENC)", "Main", "8", "4:2:0"),
    "MP4 H.265 10-bit (NVENC)": Choice("MP4", "H.265 (NVENC)", "Main 10", "10", "4:2:0"),
    "MP4 AV1 (NVENC)": Choice("MP4", "AV1 (NVENC)", "Main", "8", "4:2:0"),
}


def legacy_choice(label: str) -> Choice | None:
    return LEGACY.get(str(label or "").replace("  [unavailable]", "").strip())


DEFAULT_CHOICE = LEGACY["MP4 H.264"]


# ---- Format -> Subformat -> options (L9.54, owner 2026-10-25) -----------------------------------------------------------
# A subformat fixes the container and the codec, starts from a sensible profile / depth / chroma, and names the option
# widgets that are relevant to it (the front end shows only those). The five controls stay the encode's truth, so a
# subformat never encodes differently from the same five values chosen by hand. "Custom" = the five-control cascade.
FROM_CONTROLS = "from the controls"
FAMILIES = ("Video", "Image Sequence", "Custom")
_VIDEO_OPTS = ("quality", "encoder_preset", "profile", "bit_depth", "chroma", "gop", "bitrate_kbps")
_SUB = (
    ("Video", "H.264 / MP4", Choice("MP4", "H.264", "High", "8", "4:2:0"), _VIDEO_OPTS),
    ("Video", "H.265 / HEVC / MP4", Choice("MP4", "H.265", "Main", "8", "4:2:0"), _VIDEO_OPTS),
    ("Video", "ProRes 422 Proxy", Choice("MOV", "ProRes", "422 Proxy", "10", "4:2:2"), ("profile", "channels")),
    ("Video", "ProRes 422 LT", Choice("MOV", "ProRes", "422 LT", "10", "4:2:2"), ("profile", "channels")),
    ("Video", "ProRes 422", Choice("MOV", "ProRes", "422", "10", "4:2:2"), ("profile", "channels")),
    ("Video", "ProRes 422 HQ", Choice("MOV", "ProRes", "422 HQ", "10", "4:2:2"), ("profile", "channels")),
    ("Video", "ProRes 4444", Choice("MOV", "ProRes", "4444", "10", "4:4:4"), ("profile", "channels")),
    ("Video", "ProRes 4444 XQ", Choice("MOV", "ProRes", "4444 XQ", "10", "4:4:4"), ("profile", "channels")),
    ("Video", "DNxHR LB / SQ / HQ / HQX / 444", Choice("MOV", "DNxHR", "HQ", "8", "4:2:2"), ("profile",)),
    ("Video", "FFV1 / MKV", Choice("MKV", "FFV1 (lossless)", "Lossless", "10", "4:4:4"),
     ("bit_depth", "chroma", "channels")),
    ("Image Sequence", "OpenEXR", Choice("Image sequence", "EXR", "ZIP", "16", "RGB"),
     ("profile", "bit_depth", "channels")),
    ("Image Sequence", "DPX", Choice("Image sequence", "DPX", "RGB", "10", "RGB"), ("bit_depth",)),
    ("Image Sequence", "PNG", Choice("Image sequence", "PNG", "RGB", "16", "RGB"),
     ("bit_depth", "channels", "png_compression")),
    ("Image Sequence", "TIFF", Choice("Image sequence", "TIFF", "ZIP", "16", "RGB"),
     ("profile", "bit_depth", "channels")),
    ("Image Sequence", "JPEG", Choice("Image sequence", "JPEG", "4:2:0", "8", "4:2:0"), ("quality", "profile")),
)
SUBFORMATS: dict[str, dict] = {label: {"family": fam, "choice": ch, "options": opts} for fam, label, ch, opts in _SUB}
# what the option widgets are called in each subformat (the controls keep their names; only the label changes)
OPTION_LABELS = {"OpenEXR": {"profile": "compression", "bit_depth": "precision (16 = half, 32 = float)"},
                 "TIFF": {"profile": "compression"}, "JPEG": {"profile": "chroma subsampling"},
                 "DPX": {"bit_depth": "bit depth"}, "PNG": {"bit_depth": "bit depth"}}


def subformat_of(ch: Choice) -> str | None:
    """The subformat a five-control choice belongs to (old workflows open on the right subformat), or None (Custom)."""
    exact = [lab for lab, d in SUBFORMATS.items() if d["choice"] == ch]
    if exact:
        return exact[0]
    same = [lab for lab, d in SUBFORMATS.items() if (d["choice"].container, d["choice"].codec) == (ch.container, ch.codec)]
    return same[0] if same else None


def choice_for_subformat(subformat: str, ch: Choice | None) -> Choice:
    """A subformat fixes container + codec; profile / depth / chroma come from the controls when the codec takes them
    (the UI writes them), else the subformat's own. FROM_CONTROLS (old workflows, API prompts without it) = `ch`."""
    if not subformat or subformat == FROM_CONTROLS or subformat not in SUBFORMATS:
        return ch or DEFAULT_CHOICE
    base = SUBFORMATS[subformat]["choice"]
    if ch is None:
        return base
    codec = CODECS[base.codec]
    prof = ch.profile if ch.profile in codec.profiles else base.profile
    modes = codec.profiles[prof].modes
    bits = bits_of(ch.bit_depth)
    bit_depth, chroma = ch.bit_depth, ch.chroma
    if (bits, chroma) not in modes:
        bit_depth, chroma = base.bit_depth, base.chroma
        if (bits_of(bit_depth), chroma) not in modes:
            b0, c0 = modes[0]
            bit_depth, chroma = (str(b0) if b0 < 32 else "32 (float)"), c0
    return Choice(base.container, base.codec, prof, bit_depth, chroma)


# ---- colour tags ------------------------------------------------------------------------------------------------------
# (ffmpeg name, libavutil enum) - pixfmt.h AVColorPrimaries / AVColorTransferCharacteristic / AVColorSpace
_PRIM = {"BT.709": ("bt709", 1), "BT.2020": ("bt2020", 9), "Display P3": ("smpte432", 12), "BT.601": ("smpte170m", 6)}
_TRC = {"BT.709": ("bt709", 1), "sRGB": ("iec61966-2-1", 13), "linear": ("linear", 8),
        "PQ (ST 2084)": ("smpte2084", 16), "HLG": ("arib-std-b67", 18), "gamma 2.2": ("gamma22", 4)}
# YUV matrix that goes with the primaries: (colorspace tag, enum, ffmpeg scale out_color_matrix, swscale SWS_CS_*)
_MATRIX = {"BT.709": ("bt709", 1, "bt709", 1), "Display P3": ("bt709", 1, "bt709", 1),
           "BT.2020": ("bt2020nc", 9, "bt2020", 9), "BT.601": ("smpte170m", 6, "bt601", 5)}
TAG_NUM = {name: num for name, num in list(_PRIM.values()) + list(_TRC.values())} | {
    "bt2020nc": 9, "tv": 1, "pc": 2}


@dataclass(frozen=True)
class Colour:
    primaries: str = "auto"
    transfer: str = "auto"
    range: str = "auto"

    def is_auto(self) -> bool:
        return self.primaries == "auto" and self.transfer == "auto" and self.range == "auto"


def check_colour(col: Colour) -> None:
    for value, allowed, what in ((col.primaries, PRIMARIES, "primaries"), (col.transfer, TRANSFERS, "transfer"),
                                 (col.range, RANGES, "range")):
        if value not in allowed:
            raise SaveVideoError(f"Unknown colour {what} {value!r}; pick one of {', '.join(allowed)}.")


def conversion(col: Colour | None) -> tuple[str, int, str, int]:
    """(ffmpeg out_color_matrix, swscale colourspace, ffmpeg out_range, swscale range) for the RGB -> YUV step."""
    prim = (col.primaries if col else "auto")
    mat = _MATRIX.get(prim, _MATRIX["BT.709"])
    full = col is not None and col.range == "full"
    return mat[2], mat[3], ("pc" if full else "tv"), (2 if full else 1)


def apply_colour(tags: dict[str, str] | None, col: Colour | None, yuv: bool) -> dict[str, str] | None:
    """Today's automatic tags with the explicit choices on top. YUV: matrix follows the primaries, range drives the
    conversion. RGB: primaries / transfer only (an RGB file has no matrix and is always full range)."""
    if col is None or col.is_auto():
        return tags
    out = dict(tags or {})
    if col.primaries != "auto":
        out["color_primaries"] = _PRIM[col.primaries][0]
        if yuv:
            out["colorspace"] = _MATRIX[col.primaries][0]
    if col.transfer != "auto":
        out["color_trc"] = _TRC[col.transfer][0]
    if yuv:
        out.setdefault("colorspace", "bt709")
        out["color_range"] = "pc" if col.range == "full" else "tv"
    else:
        out.pop("colorspace", None)
        out.pop("color_range", None)
    return out or None


def resolve(format_label: str, choice: Choice | None, alpha: bool = False) -> tuple[FormatSpec, str]:
    """(spec, how): a legacy label keeps its exact old spec; custom builds from the five controls."""
    base = str(format_label or "").replace("  [unavailable]", "").strip()
    if base and base != CUSTOM:
        from .save_formats import resolve_format
        return resolve_format(format_label), f"format {base!r}"
    if disabled():
        raise SaveVideoError(f"The five format controls are switched off ({KILL_SWITCH}=0); pick a format from the "
                             "list.")
    ch = choice or DEFAULT_CHOICE
    return build_spec(ch, alpha), f"{ch.container} / {ch.codec} / {ch.profile} / {ch.bit_depth}-bit / {ch.chroma}"
