import hashlib
import os
import re
import sys
from datetime import datetime
from pathlib import Path

try:
    from .nodes._is_changed_util import dir_version_fingerprint, hash_kwargs
except ImportError:  # standalone / test import
    from nodes._is_changed_util import dir_version_fingerprint, hash_kwargs


# Date format selector → strftime mapping
DATE_FORMAT_MAP = {
    "MM-DD-YYYY": "%m-%d-%Y",
    "DD-MM-YYYY": "%d-%m-%Y",
    "YYYY-MM-DD": "%Y-%m-%d",
}
DATE_FORMAT_CHOICES = list(DATE_FORMAT_MAP.keys())

# Path separator styles — different OSes expect different separators
# when paths are passed to external tools or displayed to users.
# ComfyUI internally handles "/" on all OSes, but users may need
# native separators for downstream scripts or other tools.
PATH_STYLE_CHOICES = ["auto", "windows", "linux", "macos"]

# Source-choice widget options.  Drives the JS companion's graph
# traversal.  Python only consumes `source_filename` (already filled
# by JS), so this widget is effectively a routing hint for the frontend.
# MANUAL bug-fix (Apr 2026): added 'custom' so users can hand-type a
# name in the new ``custom_name`` widget instead of being forced to
# wire up a loader trigger.
# VFX (2026-08-13): added 'exr' so a VFX plate (EXR/DPX/CIN sequence or
# single EXR) can be selected explicitly. Works WITHOUT a trigger: wire
# the plate path STRING into `source_path` (or any loader's path), pick
# 'exr', and the node derives the stem — no trigger_image/trigger_video
# required. 'exr' is not special-cased in increment() (only 'custom' is),
# so it flows through the normal source_path > source_filename path.
SOURCE_CHOICE_CHOICES = ["auto", "image", "video", "exr", "custom"]

# Name-format choices applied to the derived source filename.
#   basename       — strip extension only           ("clip_2160_25fps")
#   strip_tags     — strip ext + trailing res/fps    ("clip")
#   first_segment  — keep only first chunk before    ("clip")
#                    a `.` or `_`
# The JS companion mirrors this so the on-node status preview matches
# what Python writes to disk.
NAME_FORMAT_CHOICES = ["basename", "strip_tags", "first_segment"]

# Where the `suffix` lands. Lets a user route mask/wan/etc outputs into
# SEPARATE folders instead of only tagging the filename — the
# "ATG_..._v002/mask" vs "ATG_..._mask/date/version" request.
#   filename  — append to the output basename  (.../v001/clip_mask.mov)  [legacy default]
#   subfolder — nest a folder AFTER the version (.../v001/mask/clip.mov)
#   folder    — append to the TOP folder name  (ATG_..._mask/date/v001/clip.mov)
SUFFIX_MODE_CHOICES = ["filename", "subfolder", "folder"]

_TRAILING_TAG_RE = re.compile(
    r"[._\-](\d{3,4}p?|\d{2,3}fps|[248]k|uhd|hd|sd|sdr|hdr|raw|proxy|final|wip)$",
    re.IGNORECASE,
)


def _format_source_name(name_no_ext: str, name_format: str) -> str:
    """Apply the user-selected name_format to a basename (no extension).

    Returns the transformed string, or *name_no_ext* unchanged on any
    edge case (empty input, unrecognized format, transformation produced
    empty result).
    """
    if not name_no_ext:
        return name_no_ext
    if name_format == "first_segment":
        parts = re.split(r"[._]", name_no_ext, maxsplit=1)
        return parts[0] if parts and parts[0] else name_no_ext
    if name_format == "strip_tags":
        cleaned = name_no_ext
        # Strip up to 4 trailing tag segments (e.g. `_2160_25fps_proxy`).
        for _ in range(4):
            new = _TRAILING_TAG_RE.sub("", cleaned)
            if new == cleaned:
                break
            cleaned = new
        return cleaned or name_no_ext
    # "basename" (default) — nothing to do, *name_no_ext* is already ext-less.
    return name_no_ext


def _get_path_sep(style: str) -> str:
    """Return the path separator for the selected style."""
    if style == "windows":
        return "\\"
    elif style in ("linux", "macos"):
        return "/"
    else:  # auto — detect from current OS
        return os.sep


def _get_output_dir():
    """Return ComfyUI output directory, with fallback for standalone use."""
    try:
        import folder_paths
        return folder_paths.get_output_directory()
    except Exception:
        return str(Path(__file__).resolve().parent / "output")


def _get_current_os() -> str:
    """Return the detected OS name for display."""
    if sys.platform == "win32":
        return "Windows"
    elif sys.platform == "darwin":
        return "macOS"
    else:
        return "Linux"


# Common input file patterns that should not be used as output folder names.
# If source_filename looks like one of these, fall back to the label instead.
_INPUT_FILE_PATTERNS = re.compile(
    r"^(ComfyUI_temp_|input_|ref_|reference_)"
    r"|^\d{5,}_\.png$"  # ComfyUI temp uploads like 00001_.png
    r"|^clipspace/",
    re.IGNORECASE,
)


_KNOWN_EXT_RE = re.compile(
    r"\.(mp4|mov|webm|mkv|avi|m4v|flv|wmv|mpeg|mpg|ts|"
    r"png|jpe?g|gif|webp|bmp|tiff?|tga|exr|dpx|cin|hdr|heic|avif|"
    r"wav|mp3|aac|flac|pdf|zip)$",
    re.IGNORECASE,
)


#: Still-image extensions that a FRAME SEQUENCE uses. Kept separate from
#: _KNOWN_EXT_RE because only these carry a frame token worth stripping — a
#: movie container never does, and stripping digits off 'take_002.mov' would
#: destroy a real name.
_SEQ_EXT_RE = __import__("re").compile(r"^\.(exr|dpx|cin|tiff?|tga|png|jpe?g|hdr)$", __import__("re").IGNORECASE)
#: The trailing frame token: .1001  _0042  .####  .%04d  (end of stem only).
#:
#: THREE variants, because one rule cannot serve both conventions (2026-08-29).
#: The 2026-08-01 fix stripped any 2-8 digit trailing token so a numbered
#: sequence groups under one folder. That also ate SHOT NUMBERS: shot_010.png,
#: shot_020.png and shot_030.png all collapsed to "shot", silently mixing
#: unrelated plates into one sequence. It also contradicted _SOURCE_VERSION_RE
#: below, which was deliberately anchored so a shot number is NOT eaten.
#:
#: The two failure modes cost very different amounts:
#:   * false GROUPING mixes unrelated plates — destructive and invisible;
#:   * false SPLITTING puts a sequence in per-frame folders — visible, harmless.
#: So the default errs toward preserving identity.
#:
#: Explicit frame syntax (#### runs, %0Nd printf tokens) is ALWAYS stripped —
#: it never appears in a shot-named still. A bare digit run is judged by width:
#: >=4 digits reads as a frame index (plate.0001.png), <=3 as a shot number
#: (shot_010.png). A convention, not a law — shot_0100 and plate.001 still
#: misfire, which is what `numbered_still_mode` is for.
_SEQ_FRAME_RE_AUTO = __import__("re").compile(r"[._-](\d{4,8}|#{2,8}|%0?\d*d)$")
#: `sequence` — the exact pre-2026-08-29 behaviour, for bare-numbered stills.
_SEQ_FRAME_RE_SEQUENCE = __import__("re").compile(r"[._-](\d{2,8}|#{2,8}|%0?\d*d)$")
#: `identity` — explicit frame syntax only; never touch bare digits.
_SEQ_FRAME_RE_IDENTITY = __import__("re").compile(r"[._-](#{2,8}|%0?\d*d)$")

NUMBERED_STILL_MODES = ("auto", "sequence", "identity")

_SEQ_FRAME_RE_BY_MODE = {
    "auto": _SEQ_FRAME_RE_AUTO,
    "sequence": _SEQ_FRAME_RE_SEQUENCE,
    "identity": _SEQ_FRAME_RE_IDENTITY,
}

#: Back-compat alias for any external caller that imported the old name.
_SEQ_FRAME_RE = _SEQ_FRAME_RE_AUTO


# Trailing version token inside a source filename stem, e.g. the "_v001"
# in "shot_0010_v001". Recognised prefixes: v / ver / version / rev, case
# insensitive, separated by . _ or - and followed by 1-8 digits at the end.
# Anchored at $ so a shot number like "shot_0010" is NOT mistaken for a
# version — only a *trailing* prefixed token counts.
_SOURCE_VERSION_RE = re.compile(
    r"(?P<sep>[._\-])(?P<prefix>v(?:er(?:sion)?)?|rev)(?P<digits>\d{1,8})$",
    re.IGNORECASE,
)


def _extract_source_version(stem: str):
    """Find a trailing version token in *stem*.

    Returns a dict (sep, prefix, digits, width, full_token, current_token)
    or ``None`` when no version token is present.

    Example: ``"B_0151C002_260527_134258_a1IE7_v001"``
        → sep="_", prefix="v", digits="001", width=3,
          full_token="_v001", current_token="v001"
    """
    if not stem:
        return None
    m = _SOURCE_VERSION_RE.search(stem)
    if not m:
        return None
    sep = m.group("sep")
    prefix = m.group("prefix")
    digits = m.group("digits")
    return {
        "sep": sep,
        "prefix": prefix,
        "digits": digits,
        "width": len(digits),
        "full_token": m.group(0),
        "current_token": prefix + digits,
    }


def _resolve_source_path(source_path: str) -> str:
    """Return the basename of *source_path* (a full file path), or "".

    Used when the user wires a full media path (e.g. an OCIORead source
    STRING holding ``.../shot_v001/shot_v001.1001.exr``) into the node.
    The downstream ``_resolve_stem_and_ext`` already calls ``Path(...).name``
    so it accepts full paths directly — this helper just normalises and trims.
    """
    sp = (source_path or "").strip()
    if not sp:
        return ""
    return sp


def _resolve_stem_and_ext(
    raw: str, fallback_ext: str = "", numbered_still_mode: str = "auto"
) -> tuple[str, str]:
    """Split a filename or path into (stem, extension).

    *raw* may be a bare stem (``C1799.MP4 Comp 1``), a filename with a
    real trailing extension (``clip.mp4``), or a path.  When the basename
    contains interior dots (``C1799.MP4 Comp 1``), ``pathlib`` suffix
    detection is unreliable — in that case *fallback_ext* from the JS
    companion (``source_extension``) is authoritative.
    """
    if not raw or not str(raw).strip():
        return "", ""
    basename = Path(str(raw).strip().replace("\\", "/")).name

    fb = str(fallback_ext or "").strip()
    if fb and not fb.startswith("."):
        fb = f".{fb}"

    # Basename already ends with the companion extension → strip once.
    if fb and len(fb) > 1 and basename.lower().endswith(fb.lower()):
        return basename[: -len(fb)], fb

    # Real trailing extension on the basename (e.g. clip_2160_25fps.mp4).
    if _KNOWN_EXT_RE.search(basename):
        stem, suffix = Path(basename).stem, Path(basename).suffix
        # FRAME-SEQUENCE FIX (2026-08-01). A VFX moving-image source is a
        # NUMBERED sequence — shot_0010.1001.exr, shot.####.exr,
        # shot.%04d.exr. Left alone, the frame number rides along in the stem,
        # so every frame derives a DIFFERENT folder name and the version
        # counter never groups the sequence. Strip the trailing frame token so
        # the whole sequence resolves to one name.
        #
        # Only stripped when the extension is a still-image one, because that
        # is what a sequence uses; a movie container never carries a frame
        # token, and stripping digits off e.g. "take_002.mov" would be wrong.
        if _SEQ_EXT_RE.match(suffix):
            frame_re = _SEQ_FRAME_RE_BY_MODE.get(
                str(numbered_still_mode or "auto").strip().lower(), _SEQ_FRAME_RE_AUTO
            )
            stem = frame_re.sub("", stem)
        return stem, suffix

    # Interior dots only — keep full basename; extension from companion.
    return basename, fb


def _looks_like_input_file(filename: str) -> bool:
    """Return True if filename appears to be a ComfyUI input/temp file
    rather than a meaningful output name."""
    if not filename:
        return False
    return bool(_INPUT_FILE_PATTERNS.search(filename))


# Characters illegal on Windows file systems (also covers Linux/macOS reserved
# slash). NUL and other control chars are stripped too.
_ILLEGAL_FOLDER_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_RESERVED_WINDOWS_NAMES = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}


def _sanitize_folder_name(name: str, max_length: int = 100, fallback: str = "output") -> str:
    """Make *name* safe to use as a folder component on Windows/Linux/macOS.

    Removes illegal characters, strips leading/trailing dots and whitespace,
    rejects reserved Windows names (CON, PRN, ...) and clamps length.
    Returns *fallback* if the result is empty.
    """
    if not name:
        return fallback
    cleaned = _ILLEGAL_FOLDER_CHARS.sub("_", str(name))
    cleaned = cleaned.strip(" .")
    if len(cleaned) > max_length:
        cleaned = cleaned[:max_length].rstrip(" .")
    if not cleaned:
        return fallback
    if cleaned.upper() in _RESERVED_WINDOWS_NAMES:
        cleaned = f"_{cleaned}"
    return cleaned


def _scan_next_version(scan_dir, prefix, padding):
    """
    Scan *scan_dir* for existing sub-directories that match the version
    pattern (e.g. v001, v002 …) and return the **next** version number.
    If no version folders exist yet → returns 1.
    Purely filesystem-based: cancelling a run cannot "waste" a number.
    """
    scan_path = Path(scan_dir)
    if not scan_path.is_dir():
        return 1
    pattern = re.compile(rf"^{re.escape(prefix)}(\d{{{padding},}})$")
    max_ver = 0
    for entry in scan_path.iterdir():
        if entry.is_dir():
            m = pattern.match(entry.name)
            if m:
                max_ver = max(max_ver, int(m.group(1)))
    return max_ver + 1


class FolderIncrementer:
    """
    Automatic dynamic file output management node.

    How it works
    ────────────
    1. Reads the input filename from whatever is connected (image, video,
       or any file type) via the JS companion that auto-fills
       ``source_filename``.
    2. Creates folder structure:
       ``output/{base_name}/{MM-DD-YYYY}/v###/{original_filename}``
    3. Version scanning happens inside the **date folder**, so each day
       starts fresh at v001.
    4. The version folder is created on execution to "claim" the number.
       Cancelled / stopped runs that never reach this node do NOT waste
       a version number.

    Example (source_filename = ``SC_30_SHT50.mp4``, today = 02-22-2026)
    ────────────────────────────────────────────────────────────────────
    output/
      SC_30_SHT50/
        02-22-2026/
          v001/
            SC_30_SHT50.mp4
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prefix": ("STRING", {"default": "v",
                    "tooltip": "Prefix before the version number (e.g. 'v' → v001)"}),
                "padding": ("INT", {"default": 3, "min": 1, "max": 10,
                    "tooltip": "Zero-pad width (3 → 001)"}),
                "suffix": ("STRING", {"default": "",
                    "tooltip": "Optional tag like '_mask' or '_wan'. WHERE it lands is "
                               "controlled by suffix_mode:\n"
                               "  filename  → '.../v001/clip_mask.mov'\n"
                               "  subfolder → '.../v001/mask/clip.mov'  (separate folder per run)\n"
                               "  folder    → 'ATG_..._mask/<date>/v001/clip.mov'  (separate top folder)\n"
                               "Leave empty to disable. Sanitized for cross-platform safety."}),
                "suffix_mode": (SUFFIX_MODE_CHOICES, {
                    "default": "filename",
                    "tooltip": "Where the `suffix` is applied:\n"
                               "  filename  — append to the output basename (legacy behaviour)\n"
                               "  subfolder — nest a folder AFTER the version, e.g. v001/mask/ — "
                               "keeps mask vs wan outputs of the SAME run side by side, same version\n"
                               "  folder    — append to the TOP folder name, e.g. ATG_..._mask/ — "
                               "fully separate folder tree with its own versioning.\n"
                               "For subfolder/folder a leading '_' or '-' is stripped from the folder "
                               "name (so '_mask' → 'mask') and the basename is left clean."}),
                "label": ("STRING", {"default": "default",
                    "tooltip": "Fallback folder name (used only when no source file is connected)"}),
                "date_format": (DATE_FORMAT_CHOICES, {
                    "default": "MM-DD-YYYY",
                    "tooltip": "Date format for the date subfolder (e.g. 02-22-2026 or 2026-02-22)",
                }),
                "path_style": (PATH_STYLE_CHOICES, {
                    "default": "auto",
                    "tooltip": "Path separator style for output strings. "
                               "auto=detect from current OS, windows=backslash, "
                               "linux/macos=forward slash. Use 'auto' unless you "
                               "design workflows on one OS and run on another.",
                }),
                "source_choice": (SOURCE_CHOICE_CHOICES, {
                    "default": "auto",
                    "tooltip": "Where the source name comes from. "
                               "'image' → trigger_image, 'video' → trigger_video, "
                               "'auto' → prefer video if connected, else image, "
                               "else legacy `trigger`. "
                               "'exr' → prefer an EXR/DPX/CIN plate (sequence or single); "
                               "works WITHOUT a trigger — wire the plate path STRING into "
                               "`source_path` and the stem is derived from it. "
                               "'custom' → use the ``custom_name`` widget verbatim "
                               "and ignore all triggers.",
                }),
                "name_format": (NAME_FORMAT_CHOICES, {
                    "default": "basename",
                    "tooltip": "How to format the detected filename for folder + prefix:\n"
                               "  basename      — strip extension only (e.g. clip_2160_25fps)\n"
                               "  strip_tags    — also strip trailing res/fps tags (clip)\n"
                               "  first_segment — keep only the first chunk before . or _ (clip)\n"
                               "The original file extension is preserved on output_filename.",
                }),
                "numbered_still_mode": (NUMBERED_STILL_MODES, {
                    "default": "auto",
                    "tooltip": "What a trailing number on a STILL means (png/exr/dpx/tif...):\n"
                               "  auto     — >=4 digits is a frame index and is stripped\n"
                               "             (plate.0001.png -> plate); <=3 digits is a shot\n"
                               "             number and is KEPT (shot_010.png -> shot_010).\n"
                               "  sequence — always strip a trailing 2-8 digit number. Use when\n"
                               "             your frames are bare-numbered (plate_01.png ...).\n"
                               "  identity — never strip bare digits; only #### and %04d go.\n"
                               "#### and %04d are stripped in every mode — they are never part\n"
                               "of a real name. auto errs toward KEEPING the number, because\n"
                               "over-grouping silently mixes unrelated plates into one sequence,\n"
                               "while over-splitting is merely inconvenient and visible.",
                }),
            },
            "optional": {
                "trigger": ("*", {
                    "tooltip": "Legacy generic trigger – connect any output here."}),
                "trigger_image": ("IMAGE", {
                    "tooltip": "Connect a LoadImage / image source here. "
                               "Used when source_choice = 'image' or 'auto'."}),
                "trigger_video": ("*", {
                    "tooltip": "Connect a LoadVideo / VHS_LoadVideo / video source here. "
                               "Used when source_choice = 'video' or 'auto' (preferred)."}),
                "source_filename": ("STRING", {"default": "",
                    "tooltip": "Auto-filled from the connected loader (basename only, no extension). "
                               "Drives folder name + output basename. Extension is stored separately "
                               "in source_extension for output_filename. "
                               "May also be a FULL PATH — the basename is extracted automatically."}),
                "source_extension": ("STRING", {"default": "",
                    "tooltip": "Auto-filled file extension from the connected loader (e.g. .mp4, .mov). "
                               "Used only for output_filename — not part of the folder name."}),
                "source_path": ("STRING", {"default": "",
                    "tooltip": "Full path to a source media file (image / video / EXR / DPX sequence). "
                               "When provided, the node extracts the filename stem from this path and "
                               "uses it as the source name — this takes PRECEDENCE over source_filename "
                               "and custom_name. Connect a STRING output here (e.g. an OCIORead "
                               "'source' widget, or any loader that exposes its file path as STRING). "
                               "The extracted stem is returned on the `source_stem` output, and a "
                               "trailing version token (e.g. _v001) is incremented to _v002 on the "
                               "`next_version_token` / `next_version_stem` outputs."}),
                "custom_name": ("STRING", {"default": "",
                    "tooltip": "Manual source name. Only used when source_choice='custom'. "
                               "May include an extension (e.g. 'my_shot.mp4'); if no "
                               "extension is given, output_filename will have none either. "
                               "Sanitized for cross-platform safety."}),
                "base_path": ("STRING", {"default": "",
                    "tooltip": "Override base output directory.  Leave empty → ComfyUI output dir."}),
                "folder_name_override": ("STRING", {"default": "",
                    "tooltip": "Force a specific folder name instead of deriving from the input filename. "
                               "Sanitized for cross-platform safety."}),
                "reserve_version": ("BOOLEAN", {"default": False,
                    "tooltip": "If True, create the version directory and write a `.reserved` marker file "
                               "to claim the version number atomically. Prevents collisions in batch/render-farm "
                               "workflows. Leave False for normal use (the directory will be created by "
                               "ComfyUI's Save node when output is actually written)."}),
            },
        }

    RETURN_TYPES  = ("STRING", "INT", "STRING", "STRING", "STRING", "STRING",
                     "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES  = ("version_string", "version_number", "folder_name",
                     "subfolder_path", "filename_prefix", "output_filename",
                     "source_stem", "current_version_token",
                     "next_version_token", "next_version_stem")
    OUTPUT_TOOLTIPS = (
        "Zero-padded version string, e.g. `v001`.",
        "Raw integer version number.",
        "Sanitized folder name derived from the input source (or label).",
        "Full subfolder path: `<base>/<folder>/<date>/<version>`.",
        "Filename prefix combining folder + version (for SaveImage).",
        "Final output filename including extension.",
        "Filename stem extracted from the connected source (extension and "
        "frame token stripped). e.g. `B_0151C002_260527_134258_a1IE7_v001`.",
        "Version token found at the end of the source stem, e.g. `v001`. "
        "Empty when the source name has no trailing version token.",
        "Source version token incremented by one, e.g. `v002`. "
        "Empty when the source name has no trailing version token.",
        "Source stem with its version token incremented, e.g. "
        "`B_0151C002_260527_134258_a1IE7_v002`. Empty when the source name "
        "has no trailing version token.",
    )
    FUNCTION = "increment"
    CATEGORY = "utils"
    OUTPUT_NODE = True   # ensure the node always executes
    DESCRIPTION = (
        "Auto-incrementing per-label / per-date version counter. Scans the output "
        "directory for existing `vNNN` subfolders and emits the next one, plus "
        "folder name, subfolder path, filename prefix, and full output filename."
    )

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Assumption: inputs + on-disk version dirs determine the next counter.
        h = hashlib.md5(hash_kwargs(**kwargs).encode())
        label = kwargs.get("label", "default")
        prefix = kwargs.get("prefix", "v")
        padding = int(kwargs.get("padding", 3))
        date_format = kwargs.get("date_format", "MM-DD-YYYY")
        base_path = (kwargs.get("base_path") or "").strip()
        base_dir = Path(base_path) if base_path else Path(_get_output_dir())
        fmt = DATE_FORMAT_MAP.get(date_format, "%m-%d-%Y")
        today_date = datetime.now().strftime(fmt)
        folder_name_override = (kwargs.get("folder_name_override") or "").strip()
        source_filename = (kwargs.get("source_filename") or "").strip()
        source_path = (kwargs.get("source_path") or "").strip()
        # source_path (full media path) takes precedence over source_filename.
        if source_path:
            source_filename = source_path
        name_format = kwargs.get("name_format", "basename")
        if folder_name_override:
            folder_name = _sanitize_folder_name(folder_name_override, fallback=label or "output")
        elif source_filename and not _looks_like_input_file(source_filename):
            raw_stem, _ext = _resolve_stem_and_ext(
                source_filename, kwargs.get("source_extension", ""),
                kwargs.get("numbered_still_mode", "auto"),
            )
            folder_name = _sanitize_folder_name(_format_source_name(raw_stem, name_format), fallback=label or "output")
        else:
            folder_name = _sanitize_folder_name(label, fallback="default")
        # `folder` suffix_mode versions in a SEPARATE top folder — fingerprint
        # that one so the cache key tracks the right v### counter.
        suffix = str(kwargs.get("suffix", "") or "").strip()
        smode = str(kwargs.get("suffix_mode", "filename") or "filename").strip().lower()
        if smode == "folder" and suffix:
            safe_suffix = _sanitize_folder_name(suffix, fallback="")
            if safe_suffix:
                folder_name = _sanitize_folder_name(folder_name + safe_suffix, fallback=label or "output")
        scan_dir = base_dir / folder_name / today_date
        h.update(dir_version_fingerprint(scan_dir, prefix, padding).encode())
        return h.hexdigest()

    def increment(self, prefix="v", padding=3, label="default",
                  date_format="MM-DD-YYYY", path_style="auto",
                  source_choice="auto", name_format="basename",
                  numbered_still_mode="auto",
                  trigger=None, trigger_image=None, trigger_video=None,
                  source_filename="", custom_name="", base_path="",
                  folder_name_override="", reserve_version=False,
                  suffix="", suffix_mode="filename", source_extension="",
                  source_path=""):

        sep = _get_path_sep(path_style)
        detected_os = _get_current_os()

        # MANUAL bug-fix (Apr 2026): when source_choice='custom', the
        # user-provided custom_name takes precedence over any auto-
        # detected source_filename. We funnel it through the same
        # downstream pipeline as a real filename so name_format,
        # extension preservation, and folder derivation all behave
        # identically.
        #
        # UX fix (May 2026): if custom_name is empty, fall through to
        # whatever is currently in source_filename — that field is
        # editable in the UI, so a user who types directly into it
        # while source_choice='custom' should NOT have their value
        # wiped to the "default" label. custom_name is now only a
        # *higher-priority override*, not the sole input.
        if str(source_choice).lower() == "custom":
            cn = (custom_name or "").strip()
            if cn:
                source_filename = cn
            # else: keep source_filename as-is (manual entry honoured).

        # ── 1. Derive names from source file ──────────────────────────
        # source_path (a full media file path, e.g. from OCIORead) takes
        # the HIGHEST precedence — it is the explicit "I wired a path in"
        # signal. _resolve_stem_and_ext extracts the basename, so passing
        # a full path here is safe.
        sp = _resolve_source_path(source_path)
        if sp:
            source_filename = sp

        raw_source = (source_filename or "").strip()
        if raw_source and _looks_like_input_file(raw_source):
            raw_source = ""

        if raw_source:
            raw_stem, ext = _resolve_stem_and_ext(raw_source, source_extension, numbered_still_mode)
            name_no_ext = _format_source_name(raw_stem, name_format)
            derived_folder = name_no_ext
        else:
            name_no_ext = ""
            ext = ""
            derived_folder = label

        # Explicit override wins; otherwise sanitize the derived name.
        if folder_name_override and folder_name_override.strip():
            folder_name = _sanitize_folder_name(folder_name_override.strip(), fallback=label or "output")
        else:
            folder_name = _sanitize_folder_name(derived_folder, fallback=label or "output")

        # ── 1b. Resolve the suffix + where it lands ───────────────────
        #    `suffix` (e.g. "_mask"/"_wan") is routed by `suffix_mode`:
        #      filename  → appended to the output basename (legacy)
        #      subfolder → a folder nested AFTER the version (v001/mask/)
        #      folder    → appended to the TOP folder name (..._mask/)
        #    `folder` must be applied BEFORE the version scan so the
        #    separate tree gets its own independent v### counter.
        safe_suffix = ""
        if suffix and str(suffix).strip():
            safe_suffix = _sanitize_folder_name(str(suffix).strip(), fallback="")
        smode = str(suffix_mode or "filename").strip().lower()
        if smode not in SUFFIX_MODE_CHOICES:
            smode = "filename"
        # For folder/subfolder a leading separator reads as a join hint,
        # not part of the name → "_mask" becomes the folder "mask".
        suffix_as_folder = safe_suffix.lstrip("_-. ") if safe_suffix else ""
        if smode == "folder" and safe_suffix:
            folder_name = _sanitize_folder_name(folder_name + safe_suffix, fallback=label or "output")

        # ── 2. Resolve base directory ─────────────────────────────────
        if base_path and base_path.strip():
            base_dir = Path(base_path.strip())
        else:
            base_dir = Path(_get_output_dir())

        # ── 3. Build date folder ─────────────────────────────────────
        fmt = DATE_FORMAT_MAP.get(date_format, "%m-%d-%Y")
        today_date = datetime.now().strftime(fmt)

        # ── 4. Scan for next version INSIDE the date folder ───────────
        #    Structure: base_dir / folder_name / today_date / v###
        date_dir = base_dir / folder_name / today_date
        version_num = _scan_next_version(date_dir, prefix, padding)
        version_string = f"{prefix}{str(version_num).zfill(padding)}"

        # ── 5. Optional atomic reservation ────────────────────────────
        #    Default behaviour: do NOT create the directory here.
        #    ComfyUI's get_save_image_path() will create it when the
        #    downstream Save node actually writes output. With
        #    reserve_version=True, we claim the version slot up-front
        #    by creating the directory and dropping a marker file —
        #    safe for parallel batch / render-farm workflows.
        if reserve_version:
            try:
                ver_dir = date_dir / version_string
                ver_dir.mkdir(parents=True, exist_ok=True)
                marker = ver_dir / ".reserved"
                if not marker.exists():
                    marker.write_text(
                        f"reserved={datetime.now().isoformat()}\n"
                        f"folder={folder_name}\n"
                        f"version={version_string}\n",
                        encoding="utf-8",
                    )
            except OSError as exc:
                # Reservation is best-effort; never crash the workflow.
                print(f"[MEC] FolderIncrementer: reserve_version failed: {exc}")

        # ── 6. Build output paths using chosen separator ──────────────
        #    subfolder mode nests `suffix_as_folder` right after the version
        #    dir, so masks/wan outputs of the SAME run sit side by side under
        #    one shared version (…/v001/mask/, …/v001/wan/).
        path_parts = [folder_name, today_date, version_string]
        if smode == "subfolder" and suffix_as_folder:
            path_parts.append(suffix_as_folder)
        subfolder_path = sep.join(path_parts)

        # In `filename` mode the suffix tags the basename; in subfolder/folder
        # mode it has already shaped the path, so the basename stays clean.
        basename_no_ext = (name_no_ext or version_string)
        if smode == "filename" and safe_suffix:
            basename_no_ext = basename_no_ext + safe_suffix

        filename_prefix = sep.join([subfolder_path, basename_no_ext])

        if name_no_ext and ext:
            output_filename = sep.join([subfolder_path, f"{basename_no_ext}{ext}"])
        else:
            output_filename = filename_prefix

        # ── 7. Source-stem + version-token outputs ─────────────────────
        #    `name_no_ext` is the filename stem with extension AND frame
        #    token already stripped (e.g. "shot_v001" from
        #    "shot_v001.1001.exr"). A trailing version token (v/ver/version/
        #    rev + digits) is detected and incremented by one so the user
        #    can build the NEXT version's name from the current source.
        source_stem = name_no_ext or ""
        sv = _extract_source_version(source_stem)
        if sv:
            current_version_token = sv["current_token"]
            next_num = int(sv["digits"]) + 1
            next_version_token = sv["prefix"] + str(next_num).zfill(sv["width"])
            next_version_stem = source_stem[:-len(sv["full_token"])] + sv["sep"] + next_version_token
        else:
            current_version_token = ""
            next_version_token = ""
            next_version_stem = ""

        return (version_string, version_num, folder_name,
                subfolder_path, filename_prefix, output_filename,
                source_stem, current_version_token,
                next_version_token, next_version_stem)


class FolderIncrementerReset:
    """
    Report the current version state for a folder (today's date).

    Scans ``<output>/<label>/<MM-DD-YYYY>/`` for version folders and
    reports how many exist.  To truly "reset", delete the version
    directories from disk.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "label": ("STRING", {"default": "default",
                    "tooltip": "Folder name to inspect"}),
                "date_format": (DATE_FORMAT_CHOICES, {
                    "default": "MM-DD-YYYY",
                    "tooltip": "Date format (must match what FolderIncrementer uses)",
                }),
            },
            "optional": {
                "trigger": ("*", {"tooltip": "Optional any-type trigger input. Connect any upstream output here to force this node to re-run after that node finishes (e.g. wire it to a SaveImage filename to recheck the version state after a save)."}),
                "base_path": ("STRING", {"default": "",
                    "tooltip": "Override base directory.  Leave empty → ComfyUI output dir."}),
            },
        }

    DESCRIPTION = "Report the current version state for a label/date folder. Scans <base>/<label>/<MM-DD-YYYY>/ and returns how many vNNN folders already exist plus the highest version number. To truly 'reset' a label, delete its date subfolder from disk."
    RETURN_TYPES  = ("STRING", "INT")
    RETURN_NAMES  = ("status", "current_version")
    OUTPUT_TOOLTIPS = (
        "Human-readable status describing how many versions exist for this label/date.",
        "Highest existing version number (0 if none yet).",
    )
    FUNCTION = "check"
    CATEGORY = "utils"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        h = hashlib.md5(hash_kwargs(**kwargs).encode())
        label = kwargs.get("label", "default")
        date_format = kwargs.get("date_format", "MM-DD-YYYY")
        base_path = (kwargs.get("base_path") or "").strip()
        base_dir = Path(base_path) if base_path else Path(_get_output_dir())
        fmt = DATE_FORMAT_MAP.get(date_format, "%m-%d-%Y")
        today_date = datetime.now().strftime(fmt)
        safe_label = _sanitize_folder_name(label, fallback="default")
        h.update(dir_version_fingerprint(base_dir / safe_label / today_date, "v", 3).encode())
        return h.hexdigest()

    def check(self, label="default", date_format="MM-DD-YYYY",
               trigger=None, base_path=""):
        base_dir = Path(base_path.strip()) if base_path and base_path.strip() else Path(_get_output_dir())
        fmt = DATE_FORMAT_MAP.get(date_format, "%m-%d-%Y")
        today_date = datetime.now().strftime(fmt)
        safe_label = _sanitize_folder_name(label, fallback="default")
        scan_dir = base_dir / safe_label / today_date
        next_ver = _scan_next_version(scan_dir, "v", 3)
        current  = next_ver - 1
        if current < 1:
            return (f"'{safe_label}/{today_date}': no versions yet – next will be v001", 0)
        return (f"'{safe_label}/{today_date}': {current} version(s) exist – next will be v{str(next_ver).zfill(3)}", current)


class FolderIncrementerSet:
    """
    Reserve version slots by creating empty directories (inside today's
    date folder).

    Creates ``<output>/<label>/<MM-DD-YYYY>/v001`` … ``v{value}`` so that
    the next FolderIncrementer run will output v{value+1}.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "label": ("STRING", {"default": "default",
                    "tooltip": "Folder name under the output directory"}),
                "value": ("INT", {"default": 1, "min": 1, "max": 999999,
                    "tooltip": "Create placeholder dirs up to this version number"}),
            },
            "optional": {
                "trigger": ("*", {"tooltip": "Optional any-type trigger input. Wire any upstream output here to control when this node runs in the graph."}),
                "prefix": ("STRING", {"default": "v", "tooltip": "Version-folder prefix. Default 'v' produces v001, v002, ... Must match what FolderIncrementer is using."}),
                "padding": ("INT", {"default": 3, "min": 1, "max": 10, "tooltip": "Zero-pad width for the version number (3 → v001, 4 → v0001). Must match what FolderIncrementer is using."}),
                "base_path": ("STRING", {"default": "", "tooltip": "Override base directory. Leave empty → ComfyUI output dir."}),
                "date_format": (DATE_FORMAT_CHOICES, {
                    "default": "MM-DD-YYYY",
                    "tooltip": "Date format (must match what FolderIncrementer uses)",
                }),
            },
        }

    DESCRIPTION = "Reserve version slots by creating empty placeholder directories under <base>/<label>/<MM-DD-YYYY>/. Creates v001 ... v{value} so the next FolderIncrementer run will produce v{value+1}. Useful for skipping ahead or reserving a known version range."
    RETURN_TYPES  = ("STRING", "INT")
    RETURN_NAMES  = ("status", "next_version")
    OUTPUT_TOOLTIPS = (
        "Status message confirming how many placeholder version dirs were created.",
        "The version number the *next* FolderIncrementer run will produce (value+1).",
    )
    FUNCTION = "set_version"
    CATEGORY = "utils"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        h = hashlib.md5(hash_kwargs(**kwargs).encode())
        label = kwargs.get("label", "default")
        prefix = kwargs.get("prefix", "v")
        padding = int(kwargs.get("padding", 3))
        date_format = kwargs.get("date_format", "MM-DD-YYYY")
        base_path = (kwargs.get("base_path") or "").strip()
        base_dir = Path(base_path) if base_path else Path(_get_output_dir())
        fmt = DATE_FORMAT_MAP.get(date_format, "%m-%d-%Y")
        today_date = datetime.now().strftime(fmt)
        safe_label = _sanitize_folder_name(label, fallback="default")
        h.update(dir_version_fingerprint(base_dir / safe_label / today_date, prefix, padding).encode())
        return h.hexdigest()

    def set_version(self, label="default", value=1, trigger=None,
                    prefix="v", padding=3, base_path="",
                    date_format="MM-DD-YYYY"):
        base_dir = Path(base_path.strip()) if base_path and base_path.strip() else Path(_get_output_dir())
        fmt = DATE_FORMAT_MAP.get(date_format, "%m-%d-%Y")
        today_date = datetime.now().strftime(fmt)
        safe_label = _sanitize_folder_name(label, fallback="default")
        folder = base_dir / safe_label / today_date
        for i in range(1, value + 1):
            ver_dir = folder / f"{prefix}{str(i).zfill(padding)}"
            ver_dir.mkdir(parents=True, exist_ok=True)
        next_ver = value + 1
        return (f"Reserved v001–v{str(value).zfill(padding)} for '{safe_label}/{today_date}'. "
                f"Next = v{str(next_ver).zfill(padding)}",
                next_ver)


# ----- Registration maps consumed by __init__.py -----
NODE_CLASS_MAPPINGS = {
    "FolderIncrementer": FolderIncrementer,
    "FolderIncrementerReset": FolderIncrementerReset,
    "FolderIncrementerSet": FolderIncrementerSet,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FolderIncrementer": "Folder Version Incrementer",
    "FolderIncrementerReset": "Folder Version Check",
    "FolderIncrementerSet": "Folder Version Set",
}
