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
SOURCE_CHOICE_CHOICES = ["auto", "image", "video", "custom"]

# Name-format choices applied to the derived source filename.
#   basename       — strip extension only           ("clip_2160_25fps")
#   strip_tags     — strip ext + trailing res/fps    ("clip")
#   first_segment  — keep only first chunk before    ("clip")
#                    a `.` or `_`
# The JS companion mirrors this so the on-node status preview matches
# what Python writes to disk.
NAME_FORMAT_CHOICES = ["basename", "strip_tags", "first_segment"]

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
    r"png|jpe?g|gif|webp|bmp|tiff?|tga|exr|hdr|heic|avif|"
    r"wav|mp3|aac|flac|pdf|zip)$",
    re.IGNORECASE,
)


def _resolve_stem_and_ext(raw: str, fallback_ext: str = "") -> tuple[str, str]:
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
        return Path(basename).stem, Path(basename).suffix

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
                    "tooltip": "Optional suffix appended to the BASENAME of the "
                               "filename_prefix and output_filename outputs. Example: "
                               "suffix='_Inpaint' → '.../v001/clip_Inpaint.mov'. "
                               "Leave empty to disable. Sanitized for cross-platform "
                               "safety. Does NOT affect the version_string or folder_name."}),
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
                               "in source_extension for output_filename."}),
                "source_extension": ("STRING", {"default": "",
                    "tooltip": "Auto-filled file extension from the connected loader (e.g. .mp4, .mov). "
                               "Used only for output_filename — not part of the folder name."}),
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

    RETURN_TYPES  = ("STRING", "INT", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES  = ("version_string", "version_number", "folder_name",
                     "subfolder_path", "filename_prefix", "output_filename")
    OUTPUT_TOOLTIPS = (
        "Zero-padded version string, e.g. `v001`.",
        "Raw integer version number.",
        "Sanitized folder name derived from the input source (or label).",
        "Full subfolder path: `<base>/<folder>/<date>/<version>`.",
        "Filename prefix combining folder + version (for SaveImage).",
        "Final output filename including extension.",
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
        name_format = kwargs.get("name_format", "basename")
        if folder_name_override:
            folder_name = _sanitize_folder_name(folder_name_override, fallback=label or "output")
        elif source_filename and not _looks_like_input_file(source_filename):
            raw_stem, _ext = _resolve_stem_and_ext(source_filename, kwargs.get("source_extension", ""))
            folder_name = _sanitize_folder_name(_format_source_name(raw_stem, name_format), fallback=label or "output")
        else:
            folder_name = _sanitize_folder_name(label, fallback="default")
        scan_dir = base_dir / folder_name / today_date
        h.update(dir_version_fingerprint(scan_dir, prefix, padding).encode())
        return h.hexdigest()

    def increment(self, prefix="v", padding=3, label="default",
                  date_format="MM-DD-YYYY", path_style="auto",
                  source_choice="auto", name_format="basename",
                  trigger=None, trigger_image=None, trigger_video=None,
                  source_filename="", custom_name="", base_path="",
                  folder_name_override="", reserve_version=False,
                  suffix="", source_extension=""):

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
        raw_source = (source_filename or "").strip()
        if raw_source and _looks_like_input_file(raw_source):
            raw_source = ""

        if raw_source:
            raw_stem, ext = _resolve_stem_and_ext(raw_source, source_extension)
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
        subfolder_path = sep.join([folder_name, today_date, version_string])

        # Suffix is appended to the BASENAME (not the folder). Sanitized so it
        # cannot escape the directory or break the file system. Empty string
        # (the default) is a no-op.
        safe_suffix = ""
        if suffix and str(suffix).strip():
            safe_suffix = _sanitize_folder_name(str(suffix).strip(), fallback="")
        basename_no_ext = (name_no_ext or version_string) + safe_suffix

        filename_prefix = sep.join([subfolder_path, basename_no_ext])

        if name_no_ext and ext:
            output_filename = sep.join([subfolder_path, f"{basename_no_ext}{ext}"])
        else:
            output_filename = filename_prefix

        return (version_string, version_num, folder_name,
                subfolder_path, filename_prefix, output_filename)


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
