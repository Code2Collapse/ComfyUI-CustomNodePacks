"""Stream a video through its VAE ENCODER, frames in order, keeping the encoder's causal state (L7.75).

The decode twin is stream_decode.py. Load Video (C2C) with a Wan 2.1 / Wan 2.2 VAE wired used to refuse ("a video
VAE"): the user had to go Load -> IMAGE -> core VAE Encode (Tiled), holding the whole decoded clip in RAM.

What core does (ComfyUI, read-only references - no core code is copied; core is GPL-3.0, this pack Apache-2.0):
  * comfy/sd.py VAE.encode: vae_encode_crop_pixels -> [N, C, H, W] -> [1, C, N, H, W] -> process_input ->
    vae_dtype -> device -> first_stage_model.encode -> output_device / vae_output_dtype() -> format_encoded.
  * comfy/ldm/wan/vae.py WanVAE.encode (2.1): t = 1 + 4 ((T - 1) // 4) frames are used; frame 0 alone, then 2-frame
    steps, ONE feat_cache for the clip (none when there is a single step), `final=` on the last step, a step may
    return None (its frames are carried inside the cache); then conv1 over all step outputs, mu = first half.
  * comfy/ldm/wan/vae2_2.py WanVAE.encode (2.2): patchify, frame 0 alone, then 4-frame steps, one feat_cache.

Here the same encoder calls run in the same order on the same values, but the frames arrive chunk by chunk from the
loader and leave as soon as their step is encoded, so the decoded clip is never held whole:
  * untiled: the latent equals core's own untiled VAE.encode bit for bit (tests/test_l775_stream_encode.py);
  * spatial tiles (auto when one untiled step would not fit the device, else off): the clip is read ONCE and each
    tile's pixels are written tile-major to its own temp file; then each tile is encoded over the WHOLE clip in one
    causal pass with its own cache (never split in time) and blended in latent space with the feathered overlap the
    decode path uses (nodes/_vae_tiled.py). Tiles approximate the untiled encode; they are not exact.
Anything else (other video VAEs) is not streamed here: the caller keeps today's behaviour and its refusal.
"""
from __future__ import annotations

import logging
import math
import os
import shutil
import uuid
from dataclasses import dataclass, field
from typing import Callable, Iterable

import numpy as np
import torch

from .stream_decode import _family

log = logging.getLogger("c2c_video.stream_encode")

_AUTO_TILES = (1024, 768, 512, 384, 256)      # px; auto picks the largest whose step fits
_FIT = 0.8                                    # fraction of free device memory one step may take


class StreamEncodeError(RuntimeError):
    pass


@dataclass
class EncodeInfo:
    method: str = ""
    exact: bool = True
    tiled: bool = False
    tile_px: int = 0
    frames_in: int = 0
    frames_used: int = 0
    step_frames: list = field(default_factory=list)     # frames per encoder call (first tile only when tiled)
    chunk_frames: list = field(default_factory=list)    # frames per loader chunk read
    tile_files: list = field(default_factory=list)      # (name, bytes) present after the clip was written
    leftover: list = field(default_factory=list)


last_info: EncodeInfo | None = None


def stream_family(vae) -> str | None:
    """'wan21' / 'wan22' when this VAE encodes video with core's causal Wan encoder; None otherwise."""
    fsm = getattr(vae, "first_stage_model", None)
    if fsm is None or getattr(vae, "latent_dim", 2) != 3 or getattr(vae, "not_video", False):
        return None
    fam = _family(fsm)
    return fam if fam in ("wan21", "wan22") else None


def frames_used(n: int) -> int:
    """Frames core's Wan encoders read of n: 1 + 4 ((n - 1) // 4) (both 2.1 and 2.2)."""
    return 1 + 4 * ((int(n) - 1) // 4)


# ── input preparation and step cutting ──────────────────────────────────────────────────────────────────────────
def _prepare(vae, chunk) -> torch.Tensor:
    """One loader chunk [k, H, W, C] -> [1, C', k, H', W'] in vae_dtype on CPU, as core's VAE.encode prepares the
    whole clip (every operation here is per pixel / per frame, so chunk by chunk gives the same values)."""
    arr = chunk[..., :3] if chunk.shape[-1] > 3 else chunk
    t = torch.from_numpy(np.ascontiguousarray(arr)) if isinstance(arr, np.ndarray) else arr
    t = vae.vae_encode_crop_pixels(t)
    t = t.movedim(-1, 1).movedim(1, 0).unsqueeze(0)
    return vae.process_input(t).to(vae.vae_dtype)


def _step_sizes(fam: str, used: int) -> list[int]:
    step = 2 if fam == "wan21" else 4
    return [1] + [step] * ((used - 1) // step)


def _steps_from_chunks(prepared: Iterable[torch.Tensor], sizes: list[int]):
    """Cut a stream of [1, C, k, H, W] chunks into the encoder's step tensors; holds one chunk + one step at most."""
    buf: torch.Tensor | None = None
    it = iter(prepared)
    for need in sizes:
        while buf is None or buf.shape[2] < need:
            try:
                nxt = next(it)
            except StopIteration:
                have = 0 if buf is None else buf.shape[2]
                raise StreamEncodeError(f"the video ended {need - have} frame(s) early: the loader delivered fewer "
                                        f"frames than it announced, so the clip cannot be encoded") from None
            buf = nxt if buf is None else torch.cat([buf, nxt], 2)
        yield buf[:, :, :need]
        buf = buf[:, :, need:]


# ── the causal encoder, step by step ────────────────────────────────────────────────────────────────────────────
def _encode_steps(vae, fam: str, steps: Iterable[torch.Tensor], n_steps: int, record: list | None) -> torch.Tensor:
    """Core's WanVAE.encode loop over externally supplied steps; returns the latent on vae.output_device in
    vae_output_dtype() (format_encoded NOT applied)."""
    import importlib

    fsm = vae.first_stage_model
    m = importlib.import_module(type(fsm).__module__)
    if fam == "wan21":
        cache = [None] * m.count_cache_layers(fsm.encoder) if n_steps > 1 else None
    else:
        cache = [None] * m.count_conv3d(fsm.encoder)
        patch = int(getattr(fsm, "patch_size", 2))
    outs = []
    i = -1
    for i, part in enumerate(steps):
        if record is not None:
            record.append(int(part.shape[2]))
        x = part.to(vae.device)
        if fam == "wan21":
            if i == 0:
                out = fsm.encoder(x, feat_cache=cache, feat_idx=[0])
            else:
                out = fsm.encoder(x, feat_cache=cache, feat_idx=[0], final=(i == n_steps - 1))
        else:
            out = fsm.encoder(m.patchify(x, patch_size=patch), feat_cache=cache, feat_idx=[0])
        if out is not None:
            outs.append(out)
    if i + 1 != n_steps:
        raise StreamEncodeError(f"{n_steps} encoder steps were expected, {i + 1} ran")
    mu = fsm.conv1(torch.cat(outs, 2) if len(outs) > 1 else outs[0]).chunk(2, dim=1)[0]
    odev = getattr(vae, "output_device", torch.device("cpu"))
    odt = vae.vae_output_dtype() if hasattr(vae, "vae_output_dtype") else torch.float32
    return mu.to(odev).to(dtype=odt)


def _finish(vae, samples: torch.Tensor) -> dict:
    fmt = getattr(vae, "format_encoded", None)
    return {"samples": fmt(samples) if fmt is not None else samples}


# ── memory decisions ────────────────────────────────────────────────────────────────────────────────────────────
def _mem(vae, shape) -> float:
    try:
        return float(vae.memory_used_encode(tuple(shape), vae.vae_dtype))
    except Exception:  # noqa: BLE001
        return 0.0


def _auto_tile(vae, c: int, step: int, h: int, w: int) -> int:
    import comfy.model_management as mm

    try:
        free = float(mm.get_free_memory(vae.device))
    except Exception:  # noqa: BLE001
        return 0
    if _mem(vae, (1, c, step, h, w)) <= _FIT * free:
        return 0
    for tp in _AUTO_TILES:
        if tp < max(h, w) and _mem(vae, (1, c, step, min(tp, h), min(tp, w))) <= _FIT * free:
            return tp
    return _AUTO_TILES[-1]


def _tmp_dir() -> str:
    try:
        import folder_paths
        return folder_paths.get_temp_directory()
    except Exception:  # noqa: BLE001
        import tempfile
        return tempfile.gettempdir()


# ── entry point ─────────────────────────────────────────────────────────────────────────────────────────────────
def stream_encode(vae, chunks: Iterable, n_frames: int, *, tile_px: int | None = None, tmp_dir: str | None = None,
                  interrupt: Callable[[], None] | None = None,
                  progress: Callable[[int, int], None] | None = None) -> dict:
    """LATENT dict of the clip `chunks` ([k, H, W, C] float32 0..1 numpy or tensors, in order, read once).
    tile_px: None = auto (tiles only when one untiled step would not fit), 0 = untiled, > 0 = that tile size."""
    global last_info
    fam = stream_family(vae)
    if fam is None:
        raise StreamEncodeError(f"{type(getattr(vae, 'first_stage_model', vae)).__name__}: no exact streaming "
                                f"encode for this VAE (only the Wan 2.1 / Wan 2.2 video VAEs)")
    n = int(n_frames)
    if n < 1:
        raise StreamEncodeError("No frames to encode.")
    used = frames_used(n)
    sizes = _step_sizes(fam, used)
    info = EncodeInfo(frames_in=n, frames_used=used)
    last_info = info
    done = [0]

    def prepared():
        seen = 0
        for chunk in chunks:
            if interrupt is not None:
                interrupt()
            k = int(chunk.shape[0])
            info.chunk_frames.append(k)
            take = min(k, used - seen)
            seen += k
            if take > 0:
                yield _prepare(vae, chunk[:take])
            done[0] = min(seen, n)
            if progress is not None:
                progress(done[0], n)
            if seen >= used:
                break                                # the rest of the clip is not read (core drops it too)

    import comfy.model_management as mm

    gen = prepared()
    try:
        first = next(gen)
    except StopIteration:
        raise StreamEncodeError("No frames to encode.") from None
    c, h, w = int(first.shape[1]), int(first.shape[3]), int(first.shape[4])
    step = sizes[1] if len(sizes) > 1 else 1
    step_mem = _mem(vae, (1, c, step, h, w))
    mm.load_models_gpu([vae.patcher], memory_required=step_mem, force_full_load=getattr(vae, "disable_offload", False))
    if tile_px is None:
        tile_px = _auto_tile(vae, c, step, h, w)
    tiled = int(tile_px) > 0 and max(h, w) > int(tile_px)

    def all_prepared():
        yield first
        yield from gen

    ctx = getattr(mm, "cuda_device_context", None)
    import contextlib
    with (ctx(vae.device) if ctx is not None else contextlib.nullcontext()), torch.no_grad():
        if not tiled:
            info.method = (f"{'Wan 2.1' if fam == 'wan21' else 'Wan 2.2'} VAE, encoded step by step with one causal "
                           f"cache while the frames stream in")
            samples = _encode_steps(vae, fam, _steps_from_chunks(all_prepared(), sizes), len(sizes), info.step_frames)
            log.info("[c2c_video] %s (%d of %d frames used)", info.method, used, n)
            return _finish(vae, samples)
        info.tiled, info.exact, info.tile_px = True, False, int(tile_px)
        info.method = (f"{'Wan 2.1' if fam == 'wan21' else 'Wan 2.2'} VAE, {int(tile_px)} px tiles, each encoded "
                       f"over the whole clip in one causal pass, blended in latent space")
        samples = _encode_tiled(vae, fam, all_prepared(), sizes, c, h, w, int(tile_px), tmp_dir or _tmp_dir(), info)
        log.info("[c2c_video] %s (%d of %d frames used)", info.method, used, n)
        return _finish(vae, samples)


def _encode_tiled(vae, fam, prepared, sizes, c, h, w, tile_px, tmp_dir, info: EncodeInfo) -> torch.Tensor:
    from .._vae_tiled import _build_feather_mask, _plan_tile_spans

    fsm = vae.first_stage_model
    factor = int(vae.spacial_compression_encode()) if hasattr(vae, "spacial_compression_encode") else 8
    if h % factor or w % factor:
        raise StreamEncodeError(f"{type(fsm).__name__}: the frames are {w}x{h} px after cropping, not a multiple of "
                                f"the VAE's scale {factor}; cannot place tiles.")
    hl, wl = h // factor, w // factor
    tile_lat = max(1, tile_px // factor)
    eff = min(max(0, tile_lat // 4), (tile_lat - 1) // 2)
    tiles = [(hs, he, ws, we) for hs, he in _plan_tile_spans(hl, tile_lat, eff)
             for ws, we in _plan_tile_spans(wl, tile_lat, eff)]
    used = sum(sizes)
    dt = vae.vae_dtype
    esize = torch.empty((), dtype=dt).element_size()
    need = sum(c * used * (he - hs) * factor * (we - ws) * factor * esize for hs, he, ws, we in tiles)
    os.makedirs(tmp_dir, exist_ok=True)
    free = shutil.disk_usage(tmp_dir).free
    if free < 1.05 * need:
        raise StreamEncodeError(f"Tiled video encode needs {need / 2**30:.2f} GB of temp disk space in {tmp_dir}, "
                                f"only {free / 2**30:.2f} GB are free; free some space or load fewer frames.")
    tag = uuid.uuid4().hex[:12]
    paths = [os.path.join(tmp_dir, f"c2c_streamenc_{tag}_{i}.bin") for i in range(len(tiles))]
    try:
        # pass 1: the clip, read once, written tile-major (one file per tile)
        handles = [open(p, "wb") for p in paths]
        try:
            for part in prepared:
                for fh, (hs, he, ws, we) in zip(handles, tiles):
                    # frame-major [k, C, th, tw]: chunks of any length append into one frame sequence
                    crop = part[0, :, :, hs * factor:he * factor, ws * factor:we * factor].movedim(1, 0).contiguous()
                    crop.view(torch.uint8).numpy().tofile(fh)
        finally:
            for fh in handles:
                fh.close()
        info.tile_files = [(os.path.basename(p), os.path.getsize(p)) for p in paths]
        # pass 2: each tile over the whole clip, its own cache; blended in latent space
        acc = weight = None
        for idx, (path, (hs, he, ws, we)) in enumerate(zip(paths, tiles)):
            th, tw = (he - hs) * factor, (we - ws) * factor
            per = c * th * tw * esize

            def steps(path=path, th=th, tw=tw, per=per):
                at = 0
                for k in sizes:
                    raw = np.fromfile(path, dtype=np.uint8, count=k * per, offset=at * per)
                    if raw.size != k * per:
                        raise StreamEncodeError("a tile file is shorter than the clip; cannot place tiles.")
                    at += k
                    yield torch.from_numpy(raw).view(dt).reshape(k, c, th, tw).movedim(0, 1).unsqueeze(0)

            mu = _encode_steps(vae, fam, steps(), len(sizes), info.step_frames if idx == 0 else None)
            mu = mu.to(device="cpu", dtype=torch.float32)
            if tuple(mu.shape[-2:]) != (he - hs, we - ws):
                raise StreamEncodeError(
                    f"{type(fsm).__name__}: a {tw}x{th} px tile encoded to {mu.shape[-1]}x{mu.shape[-2]} latent px, "
                    f"but the VAE reports a scale of {factor} ({we - ws}x{he - hs} expected); cannot place tiles.")
            if acc is None:
                acc = torch.zeros(mu.shape[:3] + (hl, wl), dtype=torch.float32)
                weight = torch.zeros((1, 1, 1, hl, wl), dtype=torch.float32)
            mask = _build_feather_mask(he - hs, we - ws, eff, hs == 0, he == hl, ws == 0, we == wl,
                                       torch.float32, torch.device("cpu")).reshape(1, 1, 1, he - hs, we - ws)
            acc[..., hs:he, ws:we] += mu * mask
            weight[..., hs:he, ws:we] += mask
            os.remove(path)
        odev = getattr(vae, "output_device", torch.device("cpu"))
        odt = vae.vae_output_dtype() if hasattr(vae, "vae_output_dtype") else torch.float32
        return (acc / weight.clamp(min=1e-6)).to(odev).to(dtype=odt)
    finally:
        for p in paths:
            try:
                os.remove(p)
            except OSError:
                pass
        info.leftover = [os.path.basename(p) for p in paths if os.path.exists(p)]
