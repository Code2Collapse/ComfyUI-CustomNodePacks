"""ComfyUI loader nodes for the C2C video lazy-handle core (S1a)."""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import replace
from fractions import Fraction
from typing import Any

from .audio import LazyAudio, lazy_audio_for_handle
from .handle import C2CVideo, FileSource, stat_key_for
from .probe import get_index_table, probe_file, probe_sequence, resolve_sequence
from .reader import iter_chunks

CATEGORY = "C2C/Video"
DIMMAX = 16384
BIGMAX = 1_000_000
_CHUNK = 16
_RAM_FRACTION = 0.5

VIDEO_EXTENSIONS: tuple[str, ...] = (
    "mp4", "mov", "mkv", "webm", "avi", "m4v", "mxf", "gif", "ts", "mts",
    "m2ts", "wmv", "flv", "mpg", "mpeg", "3gp", "ogv", "y4m",
)
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
}
FORMAT_CHOICES: list[str] = list(FORMAT_PRESETS.keys())


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
    path = os.path.abspath(strip_path(path))
    if is_sequence_path(path):
        h = probe_sequence(path)
        fps = Fraction(sequence_fps).limit_denominator(1_000_000)
        if fps <= 0:
            fps = Fraction(24, 1)
        return replace(h, fps=fps, time_base=Fraction(fps.denominator, fps.numerator))
    return probe_file(path)


def source_times_for_handle(h: C2CVideo) -> list[float]:
    if isinstance(h.source, FileSource):
        tbl = get_index_table(h.source.stat_key, stream_index=h.stream_index)
        return [float(tbl.pts_of(i) * tbl.time_base) for i in range(h.frame_count)]
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
) -> tuple[C2CVideo, C2CVideo]:
    """Fixed decision #1: probe → retime → trim → every → cap → format → size → scale."""
    preset = FORMAT_PRESETS.get(format_name or "None", {})
    h = raw
    if float(force_rate) > 0:
        h = h.retimed(float(force_rate), source_times_for_handle(raw))
    h = h.trimmed(int(skip_first_frames))
    h = h.every(int(select_every_nth))
    h = h.capped(int(frame_load_cap))
    h = apply_format_frames(h, preset, format_name or "None")
    if vae is not None:
        dim_multiple = vae_dim_multiple(vae)
    else:
        dim_multiple = int(preset.get("dim_multiple", 1) or 1)
    tw, th = compute_target_size(h.width, h.height, custom_width, custom_height, dim_multiple)
    h = h.scaled(tw, th)
    return raw, h


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


def decode_to_image_and_mask(h: C2CVideo) -> tuple[Any, Any]:
    # Fixed decision #6
    torch = _import_torch()
    n = len(h)
    if n == 0:
        raise RuntimeError("No frames to decode.")
    ProgressBar, interrupt = _progress_and_interrupt()
    pbar = ProgressBar(n) if ProgressBar is not None else None
    image = torch.empty((n, h.height, h.width, 3), dtype=torch.float32)
    if h.has_alpha:
        mask = torch.empty((n, h.height, h.width), dtype=torch.float32)
    else:
        mask = torch.zeros((n, 64, 64), dtype=torch.float32)
    offset = 0
    for chunk in iter_chunks(h, _CHUNK, fmt="float32"):
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


def encode_via_vae(h: C2CVideo, vae: Any) -> dict[str, Any]:
    # Fixed decision #7
    torch = _import_torch()
    latents: list[Any] = []
    ProgressBar, interrupt = _progress_and_interrupt()
    n = len(h)
    pbar = ProgressBar(n) if ProgressBar is not None else None
    done = 0
    for chunk in iter_chunks(h, _CHUNK, fmt="float32"):
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


def validate_video_path(video) -> bool | str:
    if video is None:
        return True
    path = strip_path(video)
    if not path:
        return (
            "Load Video Path (C2C): enter a video file, a folder of images, "
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
    return f"Load Video Path (C2C): nothing found at {abspath}."


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
) -> tuple[Any, int, LazyAudio, dict, C2CVideo, Any | None]:
    raw = resolve_probe(path, sequence_fps=sequence_fps)
    h = raw
    if float(force_rate) > 0:
        h = h.retimed(float(force_rate), source_times_for_handle(raw))
    n_before_skip = len(h)
    _, loaded = build_loaded_handle(
        raw,
        force_rate=force_rate,
        skip_first_frames=skip_first_frames,
        select_every_nth=select_every_nth,
        frame_load_cap=frame_load_cap,
        format_name=format or "None",
        custom_width=custom_width,
        custom_height=custom_height,
        vae=vae,
    )
    if len(loaded) == 0:
        _raise_if_empty_selection(
            node_name,
            n_frames=n_before_skip,
            skip=int(skip_first_frames),
            force_rate=force_rate,
        )
    video_info = make_video_info(raw, loaded)
    audio = lazy_audio_for_handle(loaded)
    count = len(loaded)
    needs_image = is_output_linked(prompt, unique_id, 0)
    needs_mask = is_output_linked(prompt, unique_id, 5)
    if not (needs_image or needs_mask):
        return None, count, audio, video_info, loaded, None
    if vae is not None and needs_image:
        return encode_via_vae(loaded, vae), count, audio, video_info, loaded, None
    check_image_budget(node_name, count, loaded.width, loaded.height, loaded.has_alpha)
    image, mask = decode_to_image_and_mask(loaded)
    if not needs_image:
        image = None
    if not needs_mask:
        mask = None
    return image, count, audio, video_info, loaded, mask


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
    if not (needs_image or needs_mask):
        return None, None, count, loaded
    check_image_budget(node_name, count, loaded.width, loaded.height, loaded.has_alpha)
    image, mask = decode_to_image_and_mask(loaded)
    return image, mask, count, loaded


def _widget(tooltip: str, **extra: Any) -> dict[str, Any]:
    return {"tooltip": tooltip, **extra}


class LoadVideoC2C:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": (
                    _list_input_videos(),
                    _widget("Video file from the ComfyUI input folder.", video_upload=True),
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
            },
            "hidden": {"prompt": "PROMPT", "unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("IMAGE,LATENT", "INT", "AUDIO", "VHS_VIDEOINFO", "C2C_VIDEO", "MASK")
    RETURN_NAMES = ("IMAGE", "frame_count", "audio", "video_info", "video", "mask")
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
        prompt=None,
        unique_id=None,
    ):
        fp = _import_folder_paths()
        path = fp.get_annotated_filepath(strip_path(video))
        return execute_video_load(
            node_name="Load Video (C2C)",
            path=path,
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
        )

    @classmethod
    def IS_CHANGED(cls, video, **kwargs):
        fp = _import_folder_paths()
        path = fp.get_annotated_filepath(video)
        return is_changed_path(path)

    @classmethod
    def VALIDATE_INPUTS(cls, video):
        fp = _import_folder_paths()
        if not fp.exists_annotated_filepath(video):
            return f"Invalid video file: {video}"
        return True


class LoadVideoPathC2C:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": (
                    "STRING",
                    _widget(
                        "Path to a video file or image sequence.",
                        placeholder="X://path/clip.mov  or  shot.####.exr",
                    ),
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
                "sequence_fps": ("FLOAT", _widget("FPS for image sequences only.", default=24, min=1, max=240)),
            },
            "hidden": {"prompt": "PROMPT", "unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("IMAGE,LATENT", "INT", "AUDIO", "VHS_VIDEOINFO", "C2C_VIDEO", "MASK")
    RETURN_NAMES = ("IMAGE", "frame_count", "audio", "video_info", "video", "mask")
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
        sequence_fps=24,
        prompt=None,
        unique_id=None,
    ):
        return execute_video_load(
            node_name="Load Video Path (C2C)",
            path=strip_path(video),
            force_rate=force_rate,
            custom_width=custom_width,
            custom_height=custom_height,
            frame_load_cap=frame_load_cap,
            skip_first_frames=skip_first_frames,
            select_every_nth=select_every_nth,
            format=format,
            vae=vae,
            sequence_fps=sequence_fps,
            prompt=prompt,
            unique_id=unique_id,
        )

    @classmethod
    def IS_CHANGED(cls, video, **kwargs):
        return is_changed_path(strip_path(video))

    @classmethod
    def VALIDATE_INPUTS(cls, video):
        return validate_video_path(video)


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
        return is_changed_sequence(strip_path(directory))

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
    "LoadVideoPathC2C": LoadVideoPathC2C,
    "LoadImagesPathC2C": LoadImagesPathC2C,
    "VideoInfoC2C": VideoInfoC2C,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadVideoC2C": "Load Video (C2C)",
    "LoadVideoPathC2C": "Load Video Path (C2C)",
    "LoadImagesPathC2C": "Load Images Path (C2C)",
    "VideoInfoC2C": "Video Info (C2C)",
}
