"""ComfyUI loader nodes for the C2C video lazy-handle core (S1a)."""

from __future__ import annotations

import hashlib
import itertools
import os
import re
from dataclasses import dataclass, replace
from fractions import Fraction
from typing import Any

from . import colour_cm as CM
from . import handles_io as HIO
from . import media_roots as ROOTS
from . import source_meta as SMETA
from .audio import LazyAudio, lazy_audio_for_handle
from .fast_decode import (
    DECODER_CHOICES,
    DECODER_TIP,
    DecoderPlan,
    FastDecodeError,
    device_is_cuda,
    iter_chunks_for,
    plan_decoder,
)
from .handle import C2CVideo, stat_key_for
from .probe import get_index_table, probe_file, probe_sequence, resolve_sequence
from .reader import iter_chunks

try:                                   # L9.89: the unified media engine (c2c_core, the vendored shared core)
    from ...c2c_core.timeline import ledger as LEDGER
    from ...c2c_core.video import media as MEDIA
    from ...c2c_core.video.passes import ExrPasses
    from ...c2c_core.video.sequence import EDGE_POLICIES, parse_frame_range, scan_directory
except (ImportError, ValueError):      # tests: the pack root is on sys.path
    from c2c_core.timeline import ledger as LEDGER
    from c2c_core.video import media as MEDIA
    from c2c_core.video.passes import ExrPasses
    from c2c_core.video.sequence import EDGE_POLICIES, parse_frame_range, scan_directory

CATEGORY = "C2C/Video"
DIMMAX = 16384
BIGMAX = 1_000_000
_CHUNK = 16                         # frames per VAE-encode batch
# IMAGE decode staging per chunk, by bytes: 16 float32 frames are 1.6 GB at 4K on top of the output batch (L7.59:
# the loader alone peaked at 1.93x the batch at 2K). Same frames, same order - only the staging is smaller.
_CHUNK_BYTES = 64 * 1024 * 1024


def _image_chunk_frames(h: C2CVideo) -> int:
    per_frame = int(h.width) * int(h.height) * (4 if h.has_alpha else 3) * 4
    return max(1, min(_CHUNK, _CHUNK_BYTES // max(1, per_frame)))
_RAM_FRACTION = 0.5

VIDEO_EXTENSIONS: tuple[str, ...] = MEDIA.VIDEO_EXTENSIONS
SEQUENCE_IMAGE_EXTENSIONS = frozenset({
    ".exr", ".hdr", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".dpx",
})
_SEQUENCE_PATTERN_RE = re.compile(r"(####|@@@@|%0?\d+d|\$F\d+|\[\d+-\d+\])")

# Fixed decision #2: rate is UI-only; frames rule uses div/mod via capped().
FORMAT_PRESETS: dict[str, dict[str, Any]] = {
    "None": {},
    "AnimateDiff": {"rate": 8, "dim_multiple": 8},
    "Mochi": {"rate": 24, "dim_multiple": 16, "frames": (6, 1)},
    "LTXV": {"rate": 24, "dim_multiple": 32, "frames": (8, 1)},
    "Hunyuan": {"rate": 24, "dim_multiple": 16, "frames": (4, 1)},
    "Cosmos": {"rate": 24, "dim_multiple": 16, "frames": (8, 1)},
    "Wan": {"rate": 16, "dim_multiple": 8, "frames": (4, 1)},
    # L2.53 (owner A14): MiniMax H3 = 24 fps, 17n+5 frames (5, 22, 39, 56 ...), sizes in multiples of 32 - the same
    # grid as ComfyUI-MiniMaxSuite/mmx_utils/h3_constants.py (FPS, CANVAS_MULTIPLE, align_frame_count) and the 4K pack.
    "MiniMax H3": {"rate": 24, "dim_multiple": 32, "frames": (17, 5)},
}
FORMAT_CHOICES: list[str] = list(FORMAT_PRESETS.keys())

# L2.53: output size by its LONGEST side, keeping the aspect (a portrait clip gets the same class of size as a
# landscape one). "custom" = custom_width / custom_height as before (0 = keep the source size or follow the aspect),
# so it is the default and old workflows load exactly as they did. Values as in the 4K pack (README "resolution").
RESOLUTION_PRESETS: dict[str, int | None] = {
    "source": None,
    "2160p (4K)": 3840,
    "1440p": 2560,
    "1080p": 1920,
    "720p": 1280,
    "480p": 854,
    "2048 wide": 2048,
    "1920 wide": 1920,
    "1024 wide": 1024,
    "custom": None,
}
RESOLUTION_CHOICES: list[str] = list(RESOLUTION_PRESETS.keys())
DIVISIBLE_CHOICES: list[int] = [1, 2, 8, 16, 32, 64]
AUDIO_MISSING_CHOICES: list[str] = ["silence", "none"]


def _import_folder_paths():
    import folder_paths  # noqa: WPS433 — lazy for tests

    return folder_paths


def _progress_and_interrupt():
    try:
        from comfy.utils import ProgressBar
        import comfy.model_management as mm

        return ProgressBar, mm.throw_exception_if_processing_interrupted
    except ImportError:
        return None, lambda: None


def _import_torch():
    import torch

    return torch


def strip_path(path: str) -> str:
    path = (path or "").strip()
    if path.startswith('"'):
        path = path[1:]
    if path.endswith('"'):
        path = path[:-1]
    return path.strip()


def _list_input_videos() -> list[str]:
    fp = _import_folder_paths()
    input_dir = fp.get_input_directory()
    out: list[str] = []
    for name in os.listdir(input_dir):
        full = os.path.join(input_dir, name)
        if not os.path.isfile(full):
            continue
        parts = name.rsplit(".", 1)
        if len(parts) == 2 and parts[1].lower() in VIDEO_EXTENSIONS:
            out.append(name)
    return sorted(out)


# L9.89 (owner A21, Rev 3 Part 3.4): the combo lists every source the node loads - movies, stills, EXRs and each
# sequence ONCE, as its pattern (plates/shot.####.exr) - from input/ and one level of subfolders, only in formats an
# installed backend reads. A folder's listing is cached by its mtime (adding or removing a file changes it).
# C2C_MEDIA_LIST=videos is the kill switch back to the old movie-only list.
_LIST_CACHE: dict[str, tuple[int, list[str], list[str]]] = {}
_LIST_LIMIT = 50_000
_LIST_SUBDIRS = 200
_LIST_SKIP = frozenset({"clipspace", "3d", "webcam", "__pycache__"})


def _folder_media(folder: str, rel: str) -> tuple[list[str], list[str]]:
    """(entries, subfolder names) of one folder; entries are prefixed with ``rel`` ("" or "sub/")."""
    try:
        key = int(os.stat(folder).st_mtime_ns)
    except OSError:
        return [], []
    hit = _LIST_CACHE.get(folder)
    if hit is not None and hit[0] == key:
        return hit[1], hit[2]
    files: list[str] = []
    dirs: list[str] = []
    try:
        with os.scandir(folder) as it:
            for i, entry in enumerate(it):
                if i >= _LIST_LIMIT:
                    break
                try:
                    if entry.is_dir():
                        dirs.append(entry.name)
                        continue
                except OSError:
                    continue
                files.append(entry.name)
    except OSError:
        return [], []
    images = MEDIA.image_extensions()
    out = [rel + n for n in files if os.path.splitext(n)[1].lower().lstrip(".") in VIDEO_EXTENSIONS]
    seqs, others = scan_directory(folder, exts=images, names=files)
    for sq in seqs:
        if sq.count > 1:
            out.append(rel + sq.pattern)
        else:
            out.extend(rel + os.path.basename(f) for f in sq.files)
    out.extend(rel + n for n in others)
    _LIST_CACHE[folder] = (key, out, dirs)
    return out, dirs


def _list_input_media() -> list[str]:
    """INPUT_TYPES calls this on every /object_info: it never raises (an unreadable input folder lists nothing)."""
    try:
        if os.environ.get("C2C_MEDIA_LIST", "").strip().lower() == "videos":
            return _list_input_videos()
        input_dir = _import_folder_paths().get_input_directory()
    except Exception:  # noqa: BLE001
        return []
    out, dirs = _folder_media(input_dir, "")
    for d in sorted(dirs)[:_LIST_SUBDIRS]:
        if d.startswith(".") or d.lower() in _LIST_SKIP:
            continue
        sub_out, _ = _folder_media(os.path.join(input_dir, d), d + "/")
        out = out + sub_out
    return sorted(out)


def is_sequence_path(path: str) -> bool:
    path = strip_path(path)
    if os.path.isdir(path):
        return True
    base = os.path.basename(path)
    ext = os.path.splitext(base)[1].lower()
    if ext in SEQUENCE_IMAGE_EXTENSIONS:
        return True
    return bool(_SEQUENCE_PATTERN_RE.search(base))


def resolve_probe(path: str, *, sequence_fps: float = 24) -> C2CVideo:
    """The lazy handle of whatever ``path`` is (a movie, a sequence, a pattern, a folder, one still). ``sequence_fps``
    > 0 sets a sequence's rate (what Load Video Path always did); 0 = the file's own rate, else 24 (L9.89)."""
    return probe_for_load(path, sequence_fps=sequence_fps)[0]


MISSING_CHOICES: list[str] = ["error", "black", "hold previous", "nearest", "skip (as before)"]
_MISSING_POLICY = {"hold previous": "hold", "skip (as before)": "skip"}
LOCATION_CHOICES: list[str] = ["input folder", "server path"]


def missing_policy(choice: str) -> str:
    return _MISSING_POLICY.get(str(choice or "error"), str(choice or "error"))


def probe_for_load(path: str, *, source_mode: str = "auto", exr_layer: str = "auto", sequence_fps: float = 24,
                   frame_range: str = ""):
    """(raw handle, MediaFacts) - c2c_core.video.media.probe_media with the loader's widgets."""
    path = strip_path(path)
    hint = None
    if str(source_mode).startswith("single"):
        try:
            hint = parse_frame_range(frame_range).first
        except ValueError:
            hint = None
    return MEDIA.probe_media(path, mode=source_mode or "auto", layer=exr_layer or "auto",
                             sequence_fps=float(sequence_fps or 0), frame_hint=hint)


def source_times_for_handle(h: C2CVideo) -> list[float]:
    if h.source.kind == "file":                         # by kind: a handle from another pack's vendored core
        tbl = get_index_table(h.source.stat_key, stream_index=h.stream_index)
        return [float(tbl.pts_of(i) * tbl.time_base) for i in range(h.frame_count)]
    times = MEDIA.source_times(h)                  # L9.89: by frame NUMBER - a gap is time that passed
    if times is not None and len(times) == int(h.frame_count):
        return times
    fps = float(h.fps)
    return [i / fps for i in range(h.frame_count)]


def apply_format_frames(h: C2CVideo, preset: dict[str, Any], preset_name: str) -> C2CVideo:
    frames = preset.get("frames")
    if not frames:
        return h
    div, mod = int(frames[0]), int(frames[1])
    n = len(h)
    if n < mod:
        raise RuntimeError(
            f'Format "{preset_name}": need at least {mod} frames after trimming, but only {n} remain. '
            f"Lower skip_first_frames, raise select_every_nth, or increase frame_load_cap."
        )
    k = ((n - mod) // div) * div + mod
    if k <= 0:
        raise RuntimeError(
            f'Format "{preset_name}": no valid frame count (must satisfy count % {div} == {mod}). '
            f"Adjust skip, nth, or cap."
        )
    return h.capped(k) if k < n else h


def vae_dim_multiple(vae: Any) -> int:
    enc = getattr(vae, "spacial_compression_encode", None)
    if callable(enc):
        return int(enc())
    dr = getattr(vae, "downscale_ratio", 8)
    if isinstance(dr, int):
        return dr
    return 8


def compute_target_size(
    src_w: int,
    src_h: int,
    custom_w: int,
    custom_h: int,
    dim_multiple: int,
) -> tuple[int, int]:
    # Fixed decision #3
    w, h = int(src_w), int(src_h)
    cw, ch = int(custom_w), int(custom_h)
    m = max(1, int(dim_multiple))
    if cw == 0 and ch == 0:
        pass
    elif ch == 0:
        h = max(1, int(round(h * cw / w)))
        w = cw
    elif cw == 0:
        w = max(1, int(round(w * ch / h)))
        h = ch
    else:
        w, h = cw, ch
    w = max(m, int(w / m + 0.5) * m)
    h = max(m, int(h / m + 0.5) * m)
    return w, h


def _round_to(value: float, multiple: int) -> int:
    # PORTED FROM: HallettVisual/ComfyUI-4K-Video-Save-Load@7ae9ab9 video4k/nodes.py round_to (MIT)
    # Halves round up, as Math.round does in the node's preview of this size.
    return max(multiple, int(value / multiple + 0.5) * multiple)


def resolution_target_size(
    src_w: int,
    src_h: int,
    resolution: str,
    custom_w: int,
    custom_h: int,
    dim_multiple: int,
) -> tuple[int, int]:
    """The loaded size for a ``resolution`` choice. "custom" (and anything unknown) is compute_target_size, unchanged;
    "source" ignores custom_width / custom_height; the presets set the longest side and keep the aspect."""
    m = max(1, int(dim_multiple))
    if resolution not in RESOLUTION_PRESETS or resolution == "custom":
        return compute_target_size(src_w, src_h, custom_w, custom_h, m)
    longest = RESOLUTION_PRESETS[resolution]
    if longest is None:                                   # "source"
        return compute_target_size(src_w, src_h, 0, 0, m)
    # PORTED FROM: HallettVisual/ComfyUI-4K-Video-Save-Load@7ae9ab9 video4k/nodes.py target_size (MIT)
    scale = longest / max(int(src_w), int(src_h))
    return _round_to(src_w * scale, m), _round_to(src_h * scale, m)


def seconds_to_frames(seconds: float, fps: float) -> int:
    """Frames that ``seconds`` of footage at ``fps`` hold (the 4K pack's int(seconds * rate + 1e-6))."""
    if float(seconds) <= 0 or float(fps) <= 0:
        return 0
    return int(float(seconds) * float(fps) + 1e-6)


def make_video_info(source: C2CVideo, loaded: C2CVideo) -> dict[str, float | int]:
    src_fps = float(source.fps)
    src_count = int(source.frame_count)
    return {
        "source_fps": src_fps,
        "source_frame_count": src_count,
        "source_duration": src_count / src_fps if src_fps > 0 else 0.0,
        "source_width": int(source.width),
        "source_height": int(source.height),
        "loaded_fps": float(loaded.fps),
        "loaded_frame_count": len(loaded),
        "loaded_duration": len(loaded) / float(loaded.fps) if len(loaded) and float(loaded.fps) > 0 else 0.0,
        "loaded_width": int(loaded.width),
        "loaded_height": int(loaded.height),
    }


def build_loaded_handle(
    raw: C2CVideo,
    *,
    force_rate: float,
    skip_first_frames: int,
    select_every_nth: int,
    frame_load_cap: int,
    format_name: str,
    custom_width: int,
    custom_height: int,
    vae: Any | None,
    crop: tuple[float, float, float, float] | None = None,
    seconds_cap: float = 0.0,
    resolution: str = "custom",
    divisible_by: int = 1,
    snap_frames: bool = True,
) -> tuple[C2CVideo, C2CVideo]:
    """Fixed decision #1: probe → retime → trim → seconds → every → cap → format → crop → size → scale.
    ``crop`` = (x, y, w, h) as fractions of the frame (L2.53); None or (0, 0, 1, 1) keeps the whole frame.
    ``seconds_cap`` limits the footage after the skip to that many seconds at the (retimed) rate, before every-nth -
    as the 4K pack does it. ``resolution`` / ``divisible_by``: see resolution_target_size; the defaults change
    nothing."""
    preset = FORMAT_PRESETS.get(format_name or "None", {})
    h = raw
    if float(force_rate) > 0:
        h = h.retimed(float(force_rate), source_times_for_handle(raw))
    h = h.trimmed(int(skip_first_frames))
    if float(seconds_cap or 0) > 0:
        h = h.trimmed(0, seconds_to_frames(float(seconds_cap), float(h.fps)))
    h = h.every(int(select_every_nth))
    h = h.capped(int(frame_load_cap))
    if snap_frames:
        h = apply_format_frames(h, preset, format_name or "None")
    if crop is not None:
        h = h.cropped(*crop)
    if vae is not None:
        dim_multiple = vae_dim_multiple(vae)
    else:
        dim_multiple = int(preset.get("dim_multiple", 1) or 1)
    dim_multiple = max(dim_multiple, int(divisible_by or 1))     # the preset (or VAE) may raise divisible_by
    tw, th = resolution_target_size(h.width, h.height, resolution or "custom", custom_width, custom_height,
                                    dim_multiple)
    h = h.scaled(tw, th)
    return raw, h


@dataclass(frozen=True)
class LoadPlan:
    raw: C2CVideo                 # the source as probed (all its frames)
    facts: Any                    # c2c_core.video.media.MediaFacts
    timeline: Any                 # c2c_core.video.sequence.Timeline
    loaded: C2CVideo              # what loads, before any handle is added
    n_before_skip: int


def load_plan(
    path: str,
    *,
    source_mode: str = "auto",
    exr_layer: str = "auto",
    sequence_fps: float = 24,
    frame_range: str = "",
    missing_frames: str = "skip (as before)",
    edge_policy: str = "hold",
    reverse: bool = False,
    force_rate: float = 0.0,
    skip_first_frames: int = 0,
    select_every_nth: int = 1,
    frame_load_cap: int = 0,
    format_name: str = "None",
    custom_width: int = 0,
    custom_height: int = 0,
    vae: Any | None = None,
    crop: tuple[float, float, float, float] | None = None,
    seconds_cap: float = 0.0,
    resolution: str = "custom",
    divisible_by: int = 1,
    snap_frames: bool = True,
) -> LoadPlan:
    """The whole selection a run makes - probe, timeline (frame range / missing frames / edges), retime, skip,
    seconds, every-nth, cap, format, crop, size, reverse - in ONE place, so the run, the probe route and the frame
    planner always agree (L9.89). The defaults reproduce the old loader exactly."""
    raw, facts = probe_for_load(path, source_mode=source_mode, exr_layer=exr_layer, sequence_fps=sequence_fps,
                                frame_range=frame_range)
    single = str(source_mode or "").startswith("single")
    base, timeline = MEDIA.apply_timeline(raw, frame_range="" if single else frame_range,
                                          missing=missing_policy(missing_frames), edge=str(edge_policy or "hold"))
    h = base
    if float(force_rate) > 0:
        h = h.retimed(float(force_rate), source_times_for_handle(raw))
    n_before = len(h)
    _, loaded = build_loaded_handle(
        base,
        force_rate=force_rate,
        skip_first_frames=skip_first_frames,
        select_every_nth=select_every_nth,
        frame_load_cap=frame_load_cap,
        format_name=format_name or "None",
        custom_width=custom_width,
        custom_height=custom_height,
        vae=vae,
        crop=crop,
        seconds_cap=seconds_cap,
        resolution=resolution,
        divisible_by=divisible_by,
        snap_frames=snap_frames,
    )
    if reverse and len(loaded):
        loaded = loaded.reversed()                    # after the selection: the chosen frames, last first
    return LoadPlan(raw, facts, timeline, loaded, n_before)


def build_images_handle(
    directory: str,
    *,
    skip_first_images: int,
    select_every_nth: int,
    image_load_cap: int,
) -> C2CVideo:
    h = probe_sequence(strip_path(directory))
    return h.trimmed(int(skip_first_images)).every(int(select_every_nth)).capped(int(image_load_cap))


def image_ram_bytes(n: int, w: int, h: int, has_alpha: bool) -> int:
    base = int(n) * int(w) * int(h) * 3 * 4
    if has_alpha:
        base += int(n) * int(w) * int(h) * 4
    return base


def frames_that_fit_budget(w: int, h: int, has_alpha: bool, available: int) -> int:
    bpf = w * h * 3 * 4 + (w * h * 4 if has_alpha else 0)
    if bpf <= 0:
        return 0
    budget = int(available * _RAM_FRACTION)
    return max(0, budget // bpf)


def check_image_budget(node_name: str, n: int, w: int, h: int, has_alpha: bool) -> None:
    # Fixed decision #4
    import psutil

    available = int(psutil.virtual_memory().available)
    need = image_ram_bytes(n, w, h, has_alpha)
    limit = int(available * _RAM_FRACTION)
    if need <= limit:
        return
    fits = frames_that_fit_budget(w, h, has_alpha, available)
    need_gb = need / (1024 ** 3)
    free_gb = limit / (1024 ** 3)
    raise RuntimeError(
        f"{node_name}: this selection is {n} frames of {w}x{h}, {need_gb:.1f} GB as an IMAGE, "
        f"and only {free_gb:.1f} GB of RAM is free. Lower frame_load_cap (at most {fits} frames fit), "
        f"raise select_every_nth, or set custom_width/custom_height."
    )


def _raise_if_empty_selection(
    node_name: str,
    *,
    n_frames: int,
    skip: int,
    force_rate: float = 0,
    skip_label: str = "skip_first_frames",
) -> None:
    retime_note = " after retiming" if float(force_rate) > 0 else ""
    raise RuntimeError(
        f"{node_name}: no frames selected - the clip has {n_frames} frames{retime_note} "
        f"and {skip_label}={skip} leaves none."
    )


def is_output_linked(prompt: dict | None, unique_id: str | None, slot: int = 0) -> bool:
    # Fixed decision #5: prompt None/unknown → decode
    if prompt is None or unique_id is None:
        return True
    uid = str(unique_id)
    for node in prompt.values():
        if not isinstance(node, dict):
            continue
        inputs = node.get("inputs") or {}
        if not isinstance(inputs, dict):
            continue
        for value in inputs.values():
            if isinstance(value, (list, tuple)) and len(value) == 2:
                if str(value[0]) == uid and int(value[1]) == int(slot):
                    return True
    return False


# L9.34: a run decodes only the outputs something reads, and returns None for the others. ComfyUI's cache signature
# holds a node's own inputs, not who reads its outputs, so that None was cached and handed out after IMAGE / MASK got
# linked with unchanged widgets (owner 2026-10-23: AVHandlesAdd received images=None, the next resize crashed).
# IS_CHANGED therefore folds in which decode outputs are linked in the prompt being run; if it cannot see that prompt,
# a run that skipped decoding is never reused.
_SKIPPED: dict[str, tuple[int, ...]] = {}
_RERUN = itertools.count()


def _running_prompt(unique_id: str) -> dict | None:
    try:
        import server  # ComfyUI's PromptServer; absent in unit tests
        queue = server.PromptServer.instance.prompt_queue
        with queue.mutex:
            items = list(getattr(queue, "currently_running", {}).values())
    except Exception:  # noqa: BLE001
        return None
    for item in items:
        p = item[2] if isinstance(item, (list, tuple)) and len(item) > 2 else None
        if isinstance(p, dict) and unique_id in p:
            return p
    return None


def decode_links_token(unique_id: str | None, slots: tuple[int, ...]) -> str:
    """Part of IS_CHANGED: which of `slots` (the outputs that need decoding) the running prompt links."""
    if unique_id is None:
        return "links:unknown"
    uid = str(unique_id)
    p = _running_prompt(uid)
    if p is not None:
        return "links:" + ",".join(str(s) for s in slots if is_output_linked(p, uid, s))
    if _SKIPPED.get(uid):
        return f"links:unseen-after-skip:{next(_RERUN)}"
    return "links:unseen"


def note_skipped(unique_id: str | None, skipped: tuple[int, ...]) -> None:
    if unique_id is not None:
        _SKIPPED[str(unique_id)] = tuple(skipped)


_EXACT = DecoderPlan("exact", "exact")


def _with_fallback(h: C2CVideo, plan: DecoderPlan | None, work):
    """Run ``work(chunks)`` on the plan's reader; when the fast path cannot deliver exactly the selected frames, run it
    again on the exact reader. Returns (result, plan actually used)."""
    plan = plan or _EXACT
    if plan.mode == "exact":
        return work(lambda size: iter_chunks(h, size, fmt="float32")), plan
    try:
        return work(lambda size: iter_chunks_for(h, size, plan)), plan
    except FastDecodeError as exc:
        import logging

        logging.getLogger("c2c_video").warning("[c2c_video] fast decode of %s fell back to the exact reader: %s",
                                               os.path.basename(getattr(h.source, "path", "") or "?"), exc)
        used = DecoderPlan(plan.asked, "exact", f"fast path stopped ({exc}); read with the exact reader")
        return work(lambda size: iter_chunks(h, size, fmt="float32")), used


def decode_to_image_and_mask(h: C2CVideo, plan: DecoderPlan | None = None) -> tuple[Any, Any]:
    # Fixed decision #6
    image, mask, _used = decode_with_plan(h, plan)
    return image, mask


def _coloured(chunks, colour: CM.Colour | None):
    """L9.53: the source -> working colour transform on each decoded chunk (none when colour management is off)."""
    return chunks if colour is None else colour.chunks(chunks)


def decode_with_plan(h: C2CVideo, plan: DecoderPlan | None = None,
                     colour: CM.Colour | None = None) -> tuple[Any, Any, DecoderPlan]:
    """IMAGE + MASK of the selection, through ``plan``'s reader (falls back to the exact reader); and the plan used."""
    if len(h) == 0:
        raise RuntimeError("No frames to decode.")
    (image, mask), used = _with_fallback(
        h, plan, lambda chunks: _decode_chunks(h, _coloured(chunks(_image_chunk_frames(h)), colour)))
    return image, mask, used


def _decode_chunks(h: C2CVideo, chunks) -> tuple[Any, Any]:
    torch = _import_torch()
    n = len(h)
    ProgressBar, interrupt = _progress_and_interrupt()
    pbar = ProgressBar(n) if ProgressBar is not None else None
    image = torch.empty((n, h.height, h.width, 3), dtype=torch.float32)
    if h.has_alpha:
        mask = torch.empty((n, h.height, h.width), dtype=torch.float32)
    else:
        mask = torch.zeros((n, 64, 64), dtype=torch.float32)
    offset = 0
    for chunk in chunks:
        interrupt()
        bn = chunk.shape[0]
        if h.has_alpha and chunk.shape[-1] >= 4:
            image[offset:offset + bn] = torch.from_numpy(chunk[..., :3])
            mask[offset:offset + bn] = 1.0 - torch.from_numpy(chunk[..., 3])
        else:
            rgb = chunk[..., :3] if chunk.shape[-1] > 3 else chunk
            image[offset:offset + bn] = torch.from_numpy(rgb)
        offset += bn
        if pbar is not None:
            pbar.update_absolute(offset, n)
    return image, mask


def encode_via_vae(h: C2CVideo, vae: Any, plan: DecoderPlan | None = None) -> dict[str, Any]:
    # Fixed decision #7
    result, _used = encode_with_plan(h, vae, plan)
    return result


def encode_with_plan(h: C2CVideo, vae: Any, plan: DecoderPlan | None = None,
                     colour: CM.Colour | None = None) -> tuple[dict[str, Any], DecoderPlan]:
    return _with_fallback(h, plan, lambda chunks: _encode_chunks(h, vae, _coloured(chunks(_CHUNK), colour)))


def _encode_chunks(h: C2CVideo, vae: Any, chunks) -> dict[str, Any]:
    torch = _import_torch()
    latents: list[Any] = []
    ProgressBar, interrupt = _progress_and_interrupt()
    n = len(h)
    pbar = ProgressBar(n) if ProgressBar is not None else None
    done = 0
    for chunk in chunks:
        interrupt()
        bn = chunk.shape[0]
        rgb = chunk[..., :3] if chunk.shape[-1] > 3 else chunk
        batch = torch.from_numpy(rgb)
        encoded = vae.encode(batch)
        samples = encoded["samples"] if isinstance(encoded, dict) else encoded
        if hasattr(samples, "dim") and samples.dim() == 5:
            raise RuntimeError(
                "This VAE compresses time (a video VAE): use its own encode node on the IMAGE output."
            )
        latents.append(samples)
        done += bn
        if pbar is not None:
            pbar.update_absolute(done, n)
    if not latents:
        raise RuntimeError("No frames to encode.")
    return {"samples": torch.cat(latents, dim=0)}


def is_changed_file(path: str) -> str:
    # Fixed decision #10
    sk = stat_key_for(path)
    return f"{sk[0]}|{sk[1]}|{sk[2]}"


def is_changed_sequence(path: str) -> str:
    try:
        files = resolve_sequence(path)
        if not files:
            return f"missing:{os.path.abspath(strip_path(path))}"
        names = {os.path.basename(f) for f in files}
        dirpath = os.path.dirname(files[0]) or os.path.abspath(strip_path(path))
        rows: list[tuple[str, int, int]] = []
        with os.scandir(dirpath) as it:
            for entry in it:
                if entry.name not in names:
                    continue
                st = entry.stat()
                mtime_ns = getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9))
                rows.append((entry.name, int(st.st_size), int(mtime_ns)))
        rows.sort()
        return hashlib.sha1(repr(rows).encode("utf-8")).hexdigest()
    except (OSError, FileNotFoundError, ValueError):
        return f"missing:{os.path.abspath(strip_path(path))}"


def is_changed_path(path: str) -> str:
    path = strip_path(path)
    if not path:
        return "missing:"
    try:
        abspath = os.path.abspath(path)
        if is_sequence_path(abspath):
            return is_changed_sequence(abspath)
        if not os.path.isfile(abspath):
            return f"missing:{abspath}"
        return is_changed_file(abspath)
    except (OSError, FileNotFoundError, ValueError):
        return f"missing:{os.path.abspath(path)}"


def colour_token(path: str, colorspace_in: str | None) -> str:
    """Part of IS_CHANGED with colorspace_in auto: a colour sidecar added or edited next to the file re-runs Load."""
    if str(colorspace_in or "") != CM.CS_AUTO:
        return ""
    try:
        abspath = os.path.abspath(path)
        if os.path.isfile(abspath):
            return "|" + CM.sidecar_token(abspath, True)
        files = resolve_sequence(abspath)
        return "|" + CM.sidecar_token(files[0], False) if files else ""
    except (OSError, FileNotFoundError, ValueError):
        return ""


def validate_video_path(video, node_name: str = "Load Video (C2C)") -> bool | str:
    if video is None:
        return True
    path = strip_path(video)
    if not path:
        return (
            f"{node_name}: server path mode - enter a video file, an image, a folder of images, "
            "or a sequence pattern like shot.####.exr."
        )
    abspath = os.path.abspath(path)
    if os.path.isfile(abspath) or os.path.isdir(abspath):
        return True
    if is_sequence_path(abspath):
        try:
            if resolve_sequence(abspath):
                return True
        except (OSError, FileNotFoundError, ValueError):
            pass
    return f"{node_name}: nothing found at {abspath}."


def node_path(video, location: str | None = "input folder", path: str | None = "") -> str:
    """The absolute source path a Load Video (C2C) node reads: its server path, or the input-folder pick."""
    if str(location or "") == "server path":
        return os.path.abspath(strip_path(path or ""))
    fp = _import_folder_paths()
    return fp.get_annotated_filepath(strip_path(video or ""))


def is_changed_source(path: str, source_mode: str | None = "auto") -> str:
    """A single-frame mode reads one file: only that file's change re-runs the node, not a sibling's."""
    if str(source_mode or "").startswith("single") and os.path.isfile(path):
        return is_changed_file(path)
    return is_changed_path(path)


def validate_images_directory(directory) -> bool | str:
    if directory is None:
        return True
    path = strip_path(directory)
    if not path:
        return "Load Images Path (C2C): enter a directory of sequential images."
    abspath = os.path.abspath(path)
    if os.path.isdir(abspath):
        return True
    try:
        if resolve_sequence(abspath):
            return True
    except (OSError, FileNotFoundError, ValueError):
        pass
    return f"Load Images Path (C2C): nothing found at {abspath}."


def execute_video_load(
    *,
    node_name: str,
    path: str,
    force_rate: float,
    custom_width: int,
    custom_height: int,
    frame_load_cap: int,
    skip_first_frames: int,
    select_every_nth: int,
    format: str = "None",
    vae: Any | None = None,
    sequence_fps: float = 24,
    prompt: dict | None = None,
    unique_id: str | None = None,
    crop: tuple[float, float, float, float] | None = None,
    decoder: str = "auto",
    seconds_cap: float = 0.0,
    resolution: str = "custom",
    divisible_by: int = 1,
    audio_when_missing: str = "silence",
    handles: dict | None = None,
    colorspace_in: str = CM.CS_OFF,
    working_space: str = CM.WORK_DEFAULT,
    source_mode: str = "auto",
    exr_layer: str = "auto",
    frame_range: str = "",
    reverse: bool = False,
    frame_offset: int = 0,
    missing_frames: str = "skip (as before)",
    edge_policy: str = "hold",
) -> tuple[Any, int, LazyAudio | None, dict, C2CVideo, Any | None, str, str, Any, str]:
    """(IMAGE or LATENT, frame_count, audio, video_info, video, mask, info, frame_map, exr_passes, metadata). The
    defaults of every input added after the first release reproduce the old behaviour (decoder "auto" only switches
    readers where the frames are identical; missing frames "skip" as the old sequence reader did)."""
    import json

    hopt = dict(handles or {})
    hmode = str(hopt.get("mode") or "off")
    if HIO.disabled():
        hmode = "off"
    lp = load_plan(
        path,
        source_mode=source_mode,
        exr_layer=exr_layer,
        sequence_fps=sequence_fps,
        frame_range=frame_range,
        missing_frames=missing_frames,
        edge_policy=edge_policy,
        reverse=reverse,
        force_rate=force_rate,
        skip_first_frames=skip_first_frames,
        select_every_nth=select_every_nth,
        frame_load_cap=frame_load_cap,
        format_name=format or "None",
        custom_width=custom_width,
        custom_height=custom_height,
        vae=vae,
        crop=crop,
        seconds_cap=seconds_cap,
        resolution=resolution,
        divisible_by=divisible_by,
        snap_frames=hmode != "fit",              # fit pads UP to the grid; the format's own snap would cut first
    )
    raw, facts, timeline, loaded, n_before_skip = lp.raw, lp.facts, lp.timeline, lp.loaded, lp.n_before_skip
    if len(loaded) == 0:
        _raise_if_empty_selection(
            node_name,
            n_frames=n_before_skip,
            skip=int(skip_first_frames),
            force_rate=force_rate,
        )
    audio_mode = audio_output_mode(loaded, audio_when_missing)
    audio = lazy_audio_for_handle(loaded) if audio_mode != "none" else None
    handles_rec, handles_report = None, None
    selected = loaded                                   # L9.55: the source frames, before any handle is added
    if hmode in ("fit", "add"):
        preset = FORMAT_PRESETS.get(format or "None", {})
        model, step, plus, glabel = HIO.grid_for(str(hopt.get("grid") or HIO.FROM_FORMAT), preset, format or "None",
                                                 int(hopt.get("step") or 4), int(hopt.get("plus") or 0))
        loaded, audio, handles_rec, handles_report = HIO.add_on_load(
            loaded, audio, mode=hmode, frames=int(hopt.get("frames") or 0), target=int(hopt.get("target") or 0),
            side=str(hopt.get("side") or "head"), model=model, step=step, plus=plus, fps=float(loaded.fps),
            memo_key=f"load_video_c2c:{unique_id}", extra_steps=int(hopt.get("extra") or 0))
        handles_rec = dict(handles_rec, grid=glabel, mode=hmode)
    video_info = make_video_info(raw, loaded)
    if handles_rec is not None:
        video_info[HIO.INFO_KEY] = handles_rec
    try:
        video_info[SMETA.INFO_KEY] = SMETA.record(raw, selected)     # numbers, timecode, pixel aspect, reel (L9.55)
    except Exception:  # noqa: BLE001 - source metadata is a bonus, never a reason not to load
        pass
    data_layer = facts.layer_kind not in ("", "colour", "alpha")
    colour = None if data_layer else CM.plan(raw, colorspace_in, working_space)   # depth / normals are values
    if colour is not None:
        video_info[CM.INFO_KEY] = colour.record          # Save Video (C2C) reverses exactly this (L9.53)
    HIO.remember_load(unique_id, prompt, video_info)      # Save Video finds it without a video_info wire (L9.52)
    count = len(loaded)
    needs_image = is_output_linked(prompt, unique_id, 0)
    needs_mask = is_output_linked(prompt, unique_id, 5)
    plan = plan_decoder(loaded, decoder, cuda=device_is_cuda()) if (needs_image or needs_mask) else None
    numbers = MEDIA.source_numbers(raw)
    frame_map = LEDGER.new(list(loaded.selected_indices()), fps=Fraction(loaded.fps).limit_denominator(1_000_000),
                           path=str(getattr(raw.source, "path", "") or getattr(raw.source, "pattern", "")),
                           total_source=int(raw.frame_count), numbers=numbers, offset=int(frame_offset or 0))
    frame_map["ops"] = [f"source {facts.kind}"] + list(timeline.notes) + (["reverse"] if reverse else [])
    passes = (ExrPasses(loaded, getattr(facts.image, "layers", ()), layer_used=facts.layer) if facts.is_exr
              else ExrPasses.empty())
    metadata = json.dumps(media_metadata(raw, facts, video_info), indent=2, default=str)

    def info(used: DecoderPlan | None) -> str:
        summary = load_summary(raw, loaded, format_name=format or "None", plan=used, audio_mode=audio_mode)
        summary["media"] = media_summary(raw, loaded, facts, timeline, data_layer=data_layer)
        if handles_rec is not None:
            summary["handles"] = {"head": handles_rec["head"], "tail": handles_rec["tail"],
                                  "target": handles_rec.get("target", 0), "grid": handles_rec["grid"],
                                  "report": handles_report}
        if colour is not None:
            summary["colour"] = colour.record
        return json.dumps(summary, indent=2)

    note_skipped(unique_id, tuple(s for s, needed in ((0, needs_image), (5, needs_mask)) if not needed))
    tail = (json.dumps(frame_map), passes, metadata)
    if not (needs_image or needs_mask):
        return (None, count, audio, video_info, loaded, None, info(None)) + tail
    if vae is not None and needs_image:
        latent, used = encode_with_plan(loaded, vae, plan, colour)
        return (latent, count, audio, video_info, loaded, None, info(used)) + tail
    check_image_budget(node_name, count, loaded.width, loaded.height, loaded.has_alpha)
    image, mask, used = decode_with_plan(loaded, plan, colour)
    if not needs_image:
        image = None
    if not needs_mask:
        mask = None
    return (image, count, audio, video_info, loaded, mask, info(used)) + tail


def media_summary(raw: C2CVideo, loaded: C2CVideo, facts, timeline, *, data_layer: bool = False) -> dict[str, Any]:
    """MEDIA_INFO (Rev 3 Part 3.11), inside the info output: what the source is and what the timeline did."""
    out = facts.to_dict()
    nums = MEDIA.source_numbers(raw)
    out["numbers"] = {"first": int(nums[0]), "last": int(nums[-1])} if nums else None
    out["timeline"] = {"first": int(timeline.numbers[0]) if timeline.numbers else None,
                       "last": int(timeline.numbers[-1]) if timeline.numbers else None,
                       "step": int(timeline.step), "missing": list(timeline.missing[:200]),
                       "missing_count": len(timeline.missing), "outside": len(timeline.outside),
                       "notes": list(timeline.notes)}
    out["data_layer"] = bool(data_layer)
    if data_layer:
        out["warnings"] = list(out.get("warnings") or []) + [
            f"layer {facts.layer!r} is {facts.layer_kind} data: loaded as raw values, no colour management"]
    px = int(loaded.width) * int(loaded.height)
    out["estimate"] = {"image_bytes": image_ram_bytes(len(loaded), int(loaded.width), int(loaded.height),
                                                      bool(loaded.has_alpha)),
                       "frame_bytes": px * (4 if loaded.has_alpha else 3) * 4}
    return out


def media_metadata(raw: C2CVideo, facts, video_info: dict) -> dict[str, Any]:
    """METADATA (Rev 3 Part 3.11): the first frame's header (EXR / DPX / TIFF / PNG attributes) or the movie's
    container and stream tags, plus the source facts."""
    meta: dict[str, Any] = {"source": facts.to_dict(), "header": dict(getattr(facts.image, "header", None) or {})}
    if facts.kind == "video":
        meta["container"] = _movie_metadata(str(getattr(raw.source, "path", "")))
    if SMETA.INFO_KEY in video_info:
        meta["source_meta"] = video_info[SMETA.INFO_KEY]
    return meta


def _movie_metadata(path: str) -> dict[str, Any]:
    try:
        import av

        with av.open(path, metadata_errors="ignore") as c:
            return {"format": c.format.name, "tags": dict(c.metadata or {}),
                    "streams": [{"index": st.index, "type": st.type,
                                 "codec": getattr(getattr(st, "codec_context", None), "name", None),
                                 "tags": dict(st.metadata or {})} for st in c.streams]}
    except Exception as exc:  # noqa: BLE001 - metadata is a bonus, never a reason not to load
        return {"error": str(exc)}


def audio_output_mode(h: C2CVideo, audio_when_missing: str = "silence") -> str:
    """"file" when the clip has an audio stream, else "silence" (a silent track of the clip's length - what the loader
    always did) or "none" (no AUDIO output)."""
    if h.source.kind == "file" and h.audio is not None:
        return "file"
    return "none" if str(audio_when_missing or "silence") == "none" else "silence"


def selection_seconds(raw: C2CVideo, loaded: C2CVideo) -> tuple[float, float]:
    """(start, end) of the loaded part in the clip's own seconds, from its first frame: start = the first selected
    frame's timestamp, end = start + frames / loaded fps."""
    sel = [int(i) for i in loaded.selected_indices() if int(i) >= 0]     # black frames have no time of their own
    if not len(sel):
        return 0.0, 0.0
    first = min(sel)                                                    # a reversed selection starts at its end
    if raw.source.kind == "file":
        tbl = get_index_table(raw.source.stat_key, stream_index=raw.stream_index)
        start = float((tbl.pts_of(first) - tbl.start_pts) * tbl.time_base)
    else:
        start = first / float(raw.fps) if float(raw.fps) > 0 else 0.0
    fps = float(loaded.fps)
    return start, start + (len(loaded) / fps if fps > 0 else 0.0)


def load_summary(raw: C2CVideo, loaded: C2CVideo, *, format_name: str = "None", plan: DecoderPlan | None = None,
                 audio_mode: str = "silence") -> dict[str, Any]:
    """The ``info`` output (JSON) and the probe route's numbers: what the file is and exactly what loads."""
    sel = loaded.selected_indices()
    src_fps = float(raw.fps)
    start, end = selection_seconds(raw, loaded)
    name = raw.source.path if raw.source.kind == "file" else raw.source.pattern
    out: dict[str, Any] = {
        "file": os.path.basename(str(name).rstrip("/\\")) or str(name),
        "source": {
            "width": int(raw.width), "height": int(raw.height), "fps": round(src_fps, 6),
            "frames": int(raw.frame_count),
            "duration": round(int(raw.frame_count) / src_fps, 6) if src_fps > 0 else 0.0,
            "pix_fmt": raw.pix_fmt, "bit_depth": int(raw.bit_depth), "has_alpha": bool(raw.has_alpha),
            "rotation": int(raw.rotation), "has_audio": raw.audio is not None,
        },
        "loaded": {
            "width": int(loaded.width), "height": int(loaded.height), "fps": round(float(loaded.fps), 6),
            "frames": len(loaded), "first_frame": int(sel[0]) if len(sel) else 0,
            "last_frame": int(sel[-1]) if len(sel) else 0,      # -1 = a black frame (L9.89)
            "start_seconds": round(start, 6), "end_seconds": round(end, 6), "duration": round(end - start, 6),
        },
        "format": format_name or "None",
        "audio": audio_mode,
    }
    if plan is not None:
        out["decoder"] = {"asked": plan.asked, "used": plan.label, "note": plan.note}
    return out


def execute_images_load(
    *,
    node_name: str,
    directory: str,
    image_load_cap: int,
    skip_first_images: int,
    select_every_nth: int,
    prompt: dict | None = None,
    unique_id: str | None = None,
) -> tuple[Any | None, Any | None, int, C2CVideo]:
    raw = probe_sequence(strip_path(directory))
    n_before_skip = len(raw)
    loaded = build_images_handle(
        directory,
        skip_first_images=skip_first_images,
        select_every_nth=select_every_nth,
        image_load_cap=image_load_cap,
    )
    if len(loaded) == 0:
        _raise_if_empty_selection(
            node_name,
            n_frames=n_before_skip,
            skip=int(skip_first_images),
            skip_label="skip_first_images",
        )
    count = len(loaded)
    needs_image = is_output_linked(prompt, unique_id, 0)
    needs_mask = is_output_linked(prompt, unique_id, 1)
    note_skipped(unique_id, tuple(s for s, needed in ((0, needs_image), (1, needs_mask)) if not needed))
    if not (needs_image or needs_mask):
        return None, None, count, loaded
    check_image_budget(node_name, count, loaded.width, loaded.height, loaded.has_alpha)
    image, mask = decode_to_image_and_mask(loaded)
    return image, mask, count, loaded


SECONDS_TIP = ("Load at most this many seconds after skip_first_frames (0 = no limit), at the force_rate rate when "
               "set. Works with frame_load_cap: the smaller one wins, and a format preset then snaps the count.")
RESOLUTION_TIP = ("Output size by its LONGEST side, keeping the aspect: 2160p is 3840, 1080p is 1920. custom uses "
                  "custom_width / custom_height (0 = keep the source size or follow the aspect). source ignores them.")
DIVISIBLE_TIP = "Round both output dimensions to a multiple of this. A format preset or a VAE may raise it."
AUDIO_MISSING_TIP = ("When the file has no audio: silence outputs a silent track as long as the loaded frames (what the "
                     "loader always did, so downstream nodes keep working); none outputs nothing.")
HANDLES_TIP = ("Handles = extra frames for a video model to settle on, added here and trimmed off again by Save Video "
               "(C2C) (it reads them from video_info). fit: pad the clip UP to the model's frame grid (handle_target "
               "= frames you want out, 0 = this clip's length). add: put handle_frames on, rounded onto the grid. "
               "Repeated edge frames, never black; the audio moves with them.")
HANDLE_FRAMES_TIP = "add: frames to put on (side both: on EACH end). With a grid, 0 = just enough to land on it."
HANDLE_TARGET_TIP = "fit: how many frames you want out after the trim. 0 = this clip's own length."
HANDLE_SIDE_TIP = "Which end gets the handles. head is where a video model settles; both splits them."
HANDLE_GRID_TIP = ("The model's frame grid: 'from format' uses the format preset above (MiniMax H3 17n+5, Wan 4n+1, "
                   "LTXV 8n+1 ...), so there is one grid, not two; or pick one here.")
COLORSPACE_IN_TIP = ("The colour space this file was rendered in. auto: read it from the file (a .colorspace / .c2c.json "
                     "sidecar, the EXR colorInteropID or chromaticities, the DPX or movie colour tags). An EXR with none "
                     "is the OCIO config's scene_linear role (Nuke's working space); an untagged movie or image is "
                     "display. Pick a space to override. off: no colour management (what the loader always did).")
WORKING_TIP = ("What the frames are converted to for the models. sRGB display: a Rec.709 / sRGB source stays as it is; "
               "anything else goes through the config's view (tone-mapped: ACES 2.0 SDR in the built-in config, what "
               "Nuke's viewer shows). Save Video (C2C) reverses exactly this. preserve source: no conversion.")
EXTRA_HANDLES_TIP = ("More handles on top of fit / add, in whole steps of the model's frame grid (MiniMax H3 17, Wan "
                     "4, LTX 8 ...), so the clip stays on the grid; on handle_side (both splits them). The details line "
                     "shows k x step = frames and the total. Save Video (C2C) trims them with the rest.")
NAME_TIP = ("Save Video (C2C) on 'auto (from Load)' names its file name_prefix + this clip's name + name_suffix, e.g. "
            "proj_ + shot010 + _comp; with the version folder the name carries the folder's version. Empty = nothing "
            "added. Typing a name on Save wins.")
CROP_TIP = ("Crop, as a fraction of the frame (x, y, width, height; 0, 0, 1, 1 = no crop). Set it with the Crop "
            "tool in the player; custom_width / custom_height then resize the cropped frame.")


LOCATION_TIP = ("Where the source is. input folder: pick or upload it in 'video'. server path: type any path in 'path' "
                "(a clip, an image, a folder or a pattern like shot.####.exr) - what Load Video Path (C2C) did.")
PATH_TIP = ("server path mode: a clip, an image, an EXR, a folder of images, or a sequence pattern (shot.####.exr, "
            "shot.%04d.exr, shot.$F4.exr, shot.[1001-1100].exr). One frame of a sequence finds the whole sequence.")
SOURCE_MODE_TIP = ("auto: a movie is a movie; a numbered image finds its sequence; a lone image is one frame. Force it "
                   "when that guess is wrong: single image / single EXR loads exactly one file even when it has "
                   "siblings (frame_range picks which frame of a sequence).")
SEQUENCE_FPS_TIP = ("Rate of an image sequence or still. 0 = the file's own rate (EXR FramesPerSecond, DPX frame rate), "
                    "else 24. A still is one frame at this rate - never a longer clip.")
FRAME_RANGE_TIP = ("Source frame NUMBERS (Nuke style): 1001-1100, 1001-1100x2 (every 2nd), or one frame like 1050. "
                   "Empty = all. For a movie, frame numbers count from 0.")
FRAME_OFFSET_TIP = ("Shifts the frame numbers frame_map reports for the output (where frame 0 lands on your timeline). "
                    "The pixels do not move.")
MISSING_TIP = ("A frame inside the sequence's range with no file. error (default) stops and names them; black, hold "
               "previous and nearest fill the gap and say so in info; skip (as before) drops it - every later frame "
               "then plays early.")
EDGE_TIP = ("frame_range beyond the first / last frame: hold repeats the edge frame, black adds black frames, loop "
            "wraps, bounce plays back and forth, error stops.")
EXR_LAYER_TIP = ("The EXR layer the IMAGE shows: auto (the beauty / rgba), or a layer name from info - diffuse, depth, "
                 "N, P, crypto00 ... Data layers (depth, normals, position, motion) load as raw values, never "
                 "colour-converted. exr_passes carries every layer.")


def _widget(tooltip: str, **extra: Any) -> dict[str, Any]:
    return {"tooltip": tooltip, **extra}


class LoadVideoC2C:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": (
                    _list_input_media(),
                    # no video_upload (L9.89): core's button takes webm / mp4 only and EXR has no browser type; the
                    # node's own upload bar (js/_c2c_loader_source.js, POST /c2c/media/upload) takes every format
                    _widget("A video, an image, an EXR or an image sequence (shot.####.exr) from the ComfyUI input "
                            "folder - upload with the node's bar or drop files on it. Server path mode (location) "
                            "reads any folder instead."),
                ),
                "force_rate": ("FLOAT", _widget("Output FPS; 0 keeps source.", default=0, min=0, max=240, step=1)),
                "custom_width": ("INT", _widget("Output width; 0 keeps source or derives from height.", default=0, min=0, max=DIMMAX)),
                "custom_height": ("INT", _widget("Output height; 0 keeps source or derives from width.", default=0, min=0, max=DIMMAX)),
                "frame_load_cap": ("INT", _widget("Max frames; 0 = no cap.", default=0, min=0, max=BIGMAX)),
                "skip_first_frames": ("INT", _widget("Skip this many frames first.", default=0, min=0, max=BIGMAX)),
                "select_every_nth": ("INT", _widget("Keep every Nth frame.", default=1, min=1, max=100_000)),
            },
            "optional": {
                "vae": ("VAE", _widget("Encode to LATENT instead of IMAGE.")),
                "format": (FORMAT_CHOICES, _widget("Model preset (frame grid / dim multiple).", default="None")),
                "crop_x": ("FLOAT", _widget(CROP_TIP, default=0.0, min=0.0, max=1.0, step=0.001)),
                "crop_y": ("FLOAT", _widget(CROP_TIP, default=0.0, min=0.0, max=1.0, step=0.001)),
                "crop_w": ("FLOAT", _widget(CROP_TIP, default=1.0, min=0.0, max=1.0, step=0.001)),
                "crop_h": ("FLOAT", _widget(CROP_TIP, default=1.0, min=0.0, max=1.0, step=0.001)),
                # L2.53 (owner A14, from the 4K pack) - appended, so saved workflows keep their widget order
                "decoder": (DECODER_CHOICES, _widget(DECODER_TIP, default="auto")),
                "seconds_cap": ("FLOAT", _widget(SECONDS_TIP, default=0.0, min=0.0, max=86400.0, step=0.1)),
                "resolution": (RESOLUTION_CHOICES, _widget(RESOLUTION_TIP, default="custom")),
                "divisible_by": (DIVISIBLE_CHOICES, _widget(DIVISIBLE_TIP, default=1)),
                "audio_when_missing": (AUDIO_MISSING_CHOICES, _widget(AUDIO_MISSING_TIP, default="silence")),
                # L9.40 (owner 2026-10-23: "need the av handles add/trim built in") - appended, saved workflows keep
                # their widget order. Save Video (C2C) trims exactly these when video_info is wired.
                "handles": (list(HIO.HANDLE_MODES), _widget(HANDLES_TIP, default="off")),
                "handle_frames": ("INT", _widget(HANDLE_FRAMES_TIP, default=8, min=0, max=400)),
                "handle_target": ("INT", _widget(HANDLE_TARGET_TIP, default=0, min=0, max=BIGMAX)),
                "handle_side": (list(HIO.SIDES), _widget(HANDLE_SIDE_TIP, default="head")),
                "handle_grid": (list(HIO.GRID_CHOICES), _widget(HANDLE_GRID_TIP, default=HIO.FROM_FORMAT)),
                "handle_custom_step": ("INT", _widget("custom grid: step (frames = step * n + plus).", default=4,
                                                      min=1, max=1000)),
                "handle_custom_plus": ("INT", _widget("custom grid: plus.", default=1, min=0, max=1000)),
                # L9.53 (owner 2026-10-25 / 26): colour management, Nuke / OCIO style - appended. "off (as before)" is
                # the default here, so saved workflows and API prompts keep their pixels; a new node starts on auto.
                "colorspace_in": (CM.colorspace_in_choices(), _widget(COLORSPACE_IN_TIP, default=CM.CS_OFF)),
                "working_space": (CM.working_choices(), _widget(WORKING_TIP, default=CM.WORK_DEFAULT)),
                # L9.55 (owner 2026-10-25): Save Video names its file name_prefix + this source's name + name_suffix
                # (read from the prompt - no effect on the frames). Empty by default: no suffix unless you add one.
                "name_prefix": ("STRING", _widget(NAME_TIP, default="")),
                "name_suffix": ("STRING", _widget(NAME_TIP, default="")),
                # L9.56 (owner 2026-10-25): more handles on top of the automatic ones, in whole model steps
                "extra_handles": ("INT", _widget(EXTRA_HANDLES_TIP, default=0, min=0, max=100)),
                # L9.89 (owner A21, Rev 3 Part 3): the unified loader - appended, saved workflows keep their widget
                # order, and every default loads a saved movie exactly as before
                "location": (LOCATION_CHOICES, _widget(LOCATION_TIP, default="input folder")),
                "path": ("STRING", _widget(PATH_TIP, default="",
                                           placeholder="D:/shots/sh010/plate.####.exr  or  a clip / folder")),
                "source_mode": (list(MEDIA.SOURCE_MODES), _widget(SOURCE_MODE_TIP, default="auto")),
                "sequence_fps": ("FLOAT", _widget(SEQUENCE_FPS_TIP, default=0.0, min=0.0, max=240.0, step=0.001)),
                "frame_range": ("STRING", _widget(FRAME_RANGE_TIP, default="", placeholder="1001-1100  or  1001-1100x2")),
                "reverse": ("BOOLEAN", _widget("Play the selected frames last to first (the audio too).",
                                               default=False)),
                "frame_offset": ("INT", _widget(FRAME_OFFSET_TIP, default=0, min=-10_000_000, max=10_000_000)),
                "missing_frames": (MISSING_CHOICES, _widget(MISSING_TIP, default="error")),
                "edge_policy": (list(EDGE_POLICIES), _widget(EDGE_TIP, default="hold")),
                "exr_layer": ("STRING", _widget(EXR_LAYER_TIP, default="auto")),
            },
            "hidden": {"prompt": "PROMPT", "unique_id": "UNIQUE_ID"},
        }

    # outputs 7-9 appended (L9.89): saved links to 0-6 keep their meaning; the info output is the MEDIA_INFO
    RETURN_TYPES = ("IMAGE,LATENT", "INT", "AUDIO", "VHS_VIDEOINFO", "C2C_VIDEO", "MASK", "STRING", "STRING",
                    "C2C_EXR_PASSES", "STRING")
    RETURN_NAMES = ("IMAGE", "frame_count", "audio", "video_info", "video", "mask", "info", "frame_map", "exr_passes",
                    "metadata")
    OUTPUT_TOOLTIPS = ("The frames (a LATENT when a VAE is connected).", "Frames loaded.", "The audio of the loaded "
                       "frames.", "Source and loaded numbers (VHS-compatible).", "The lazy media handle: nothing "
                       "decoded until a node reads a window.", "1 - alpha, when the source has alpha.", "MEDIA_INFO: "
                       "what the source is and exactly what loaded (JSON).", "Output frame -> source frame and its "
                       "frame NUMBER (h3.frame_ledger v1.1 JSON; -1 = a black frame).", "Every EXR layer, lazily "
                       "(empty for other sources).", "The source's header / container metadata (JSON).")
    FUNCTION = "load_video"
    CATEGORY = CATEGORY

    def load_video(
        self,
        video,
        force_rate,
        custom_width,
        custom_height,
        frame_load_cap,
        skip_first_frames,
        select_every_nth,
        vae=None,
        format="None",
        crop_x=0.0,
        crop_y=0.0,
        crop_w=1.0,
        crop_h=1.0,
        decoder="auto",
        seconds_cap=0.0,
        resolution="custom",
        divisible_by=1,
        audio_when_missing="silence",
        handles="off",
        handle_frames=8,
        handle_target=0,
        handle_side="head",
        handle_grid=HIO.FROM_FORMAT,
        handle_custom_step=4,
        handle_custom_plus=1,
        colorspace_in=CM.CS_OFF,
        working_space=CM.WORK_DEFAULT,
        name_prefix="",                     # read by Save Video from the prompt (L9.55)
        name_suffix="",
        extra_handles=0,
        location="input folder",
        path="",
        source_mode="auto",
        sequence_fps=0.0,
        frame_range="",
        reverse=False,
        frame_offset=0,
        missing_frames="error",
        edge_policy="hold",
        exr_layer="auto",
        prompt=None,
        unique_id=None,
    ):
        src = node_path(video, location, path)
        if location == "server path":
            if not strip_path(path):
                raise ValueError("Load Video (C2C): server path mode - enter a path in 'path' (a clip, an image, a "
                                 "folder, or a pattern like shot.####.exr).")
            blocked = ROOTS.check_run_path(src)
            if blocked:
                raise PermissionError(f"Load Video (C2C): {blocked}")
        return execute_video_load(
            node_name="Load Video (C2C)",
            path=src,
            force_rate=force_rate,
            custom_width=custom_width,
            custom_height=custom_height,
            frame_load_cap=frame_load_cap,
            skip_first_frames=skip_first_frames,
            select_every_nth=select_every_nth,
            format=format,
            vae=vae,
            prompt=prompt,
            unique_id=unique_id,
            crop=(crop_x, crop_y, crop_w, crop_h),
            decoder=decoder,
            seconds_cap=seconds_cap,
            resolution=resolution,
            divisible_by=divisible_by,
            audio_when_missing=audio_when_missing,
            handles=dict(mode=handles, frames=handle_frames, target=handle_target, side=handle_side,
                         grid=handle_grid, step=handle_custom_step, plus=handle_custom_plus, extra=extra_handles),
            colorspace_in=colorspace_in,
            working_space=working_space,
            sequence_fps=sequence_fps,
            source_mode=source_mode,
            exr_layer=exr_layer,
            frame_range=frame_range,
            reverse=reverse,
            frame_offset=frame_offset,
            missing_frames=missing_frames,
            edge_policy=edge_policy,
        )

    @classmethod
    def IS_CHANGED(cls, video, **kwargs):
        path = node_path(video, kwargs.get("location"), kwargs.get("path"))
        return (is_changed_source(path, kwargs.get("source_mode")) + "|"
                + decode_links_token(kwargs.get("unique_id"), (0, 5)) + colour_token(path, kwargs.get("colorspace_in")))

    @classmethod
    def VALIDATE_INPUTS(cls, video, location="input folder", path="", source_mode="auto"):
        if str(location or "") == "server path":
            ok = validate_video_path(path if path is not None else None)
            if ok is not True:
                return ok
            blocked = ROOTS.check_run_path(os.path.abspath(strip_path(path or "")))
            return f"Load Video (C2C): {blocked}" if blocked else True
        fp = _import_folder_paths()
        if fp.exists_annotated_filepath(video):
            return True
        full = node_path(video)
        if is_sequence_path(full):                   # a listed sequence pattern (plates/shot.####.exr)
            try:
                if resolve_sequence(full):
                    return True
            except (OSError, FileNotFoundError, ValueError):
                pass
        return f"Load Video (C2C): {video} is not in the input folder (pick another, or upload it)."


# Load Video Path (C2C) was merged into Load Video (C2C) as its server-path mode (owner 2026-10-31, D0.34 / L9.89);
# saved Path nodes and API prompts migrate automatically (nodes/_legacy_replacements.json).


class LoadImagesPathC2C:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "directory": (
                    "STRING",
                    _widget("Directory of sequential images.", placeholder="X://path/to/images/"),
                ),
            },
            "optional": {
                "image_load_cap": ("INT", _widget("Max images; 0 = no cap.", default=0, min=0, max=BIGMAX)),
                "skip_first_images": ("INT", _widget("Skip this many images first.", default=0, min=0, max=BIGMAX)),
                "select_every_nth": ("INT", _widget("Keep every Nth image.", default=1, min=1, max=100_000)),
            },
            "hidden": {"prompt": "PROMPT", "unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "C2C_VIDEO")
    RETURN_NAMES = ("IMAGE", "MASK", "frame_count", "video")
    FUNCTION = "load_images"
    CATEGORY = CATEGORY

    def load_images(
        self,
        directory,
        image_load_cap=0,
        skip_first_images=0,
        select_every_nth=1,
        prompt=None,
        unique_id=None,
    ):
        return execute_images_load(
            node_name="Load Images Path (C2C)",
            directory=directory,
            image_load_cap=image_load_cap,
            skip_first_images=skip_first_images,
            select_every_nth=select_every_nth,
            prompt=prompt,
            unique_id=unique_id,
        )

    @classmethod
    def IS_CHANGED(cls, directory, **kwargs):
        return is_changed_sequence(strip_path(directory)) + "|" + decode_links_token(kwargs.get("unique_id"), (0, 1))

    @classmethod
    def VALIDATE_INPUTS(cls, directory):
        return validate_images_directory(directory)


class VideoInfoC2C:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_info": ("VHS_VIDEOINFO", _widget("Metadata from a C2C or VHS loader.")),
            },
        }

    RETURN_TYPES = ("FLOAT", "INT", "FLOAT", "INT", "INT", "FLOAT", "INT", "FLOAT", "INT", "INT")
    RETURN_NAMES = (
        "source_fps",
        "source_frame_count",
        "source_duration",
        "source_width",
        "source_height",
        "loaded_fps",
        "loaded_frame_count",
        "loaded_duration",
        "loaded_width",
        "loaded_height",
    )
    FUNCTION = "get_video_info"
    CATEGORY = CATEGORY

    def get_video_info(self, video_info):
        keys = ("fps", "frame_count", "duration", "width", "height")
        return tuple(video_info[f"source_{k}"] for k in keys) + tuple(video_info[f"loaded_{k}"] for k in keys)


NODE_CLASS_MAPPINGS = {
    "LoadVideoC2C": LoadVideoC2C,
    "LoadImagesPathC2C": LoadImagesPathC2C,
    "VideoInfoC2C": VideoInfoC2C,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadVideoC2C": "Load Video (C2C)",
    "LoadImagesPathC2C": "Load Images Path (C2C)",
    "VideoInfoC2C": "Video Info (C2C)",
}
