"""Stream a video latent through its VAE, frame range by frame range, in order, keeping the VAE's causal state.

Owner, 2026-10-10: "decode the complete latent sequence in its original temporal order, carrying causal state across
frames. Stream decoded frames directly to the encoder ... Chunking must be for I/O and memory management only, never
independent temporal sampling or decoding ... If the VAE cannot support exact stateful streaming for a given model, make
that limitation explicit." Measured why: splitting a causal video VAE in time and decoding the pieces independently
costs ~21 dB (Wan 2.1, docs/evidence/L7.59).

Engines - each produces exactly what core's own `VAE.decode` produces, only without holding every frame at once:
  * Wan 2.1 / Wan 2.2: the decoder is run one latent step at a time with ONE feature cache for the whole clip (the
    same steps core takes in `WanVAE.decode`), and each step's frames are handed on as soon as they exist.
  * VAEs with core's chunked-output protocol (`comfy_has_chunked_io`: LTX, MiniMax H3): core's decoder writes its
    finished frame ranges, in order, into the buffer we pass - a sink that hands them on. Runs in a worker thread
    with a 2-chunk queue so the decoder never gets ahead of the encoder by more than that.
  * Anything else: decoded whole by core (exact, but NOT bounded in RAM) and reported as such.
Only core's public classes and methods are called; no core code is copied (core is GPL-3.0, this pack Apache-2.0).
"""
from __future__ import annotations

import logging
import queue
import threading
from dataclasses import dataclass, field
from typing import Callable, Iterator

import torch

log = logging.getLogger("c2c_video.stream_decode")


class StreamDecodeError(RuntimeError):
    pass


# ── a forward-only frame batch ───────────────────────────────────────────────────────────────────────────────────
class FrameStream:
    """Frames that arrive in order, presented like a [N, H, W, C] tensor to code that reads them front to back
    (Save Video's encoders): `.shape`, `.ndim`, `stream[i]`, `stream[a:b]`. Each frame is handed out once; asking for
    an earlier frame again is an error, not a silent re-decode."""

    def __init__(self, chunks: Iterator[torch.Tensor], n: int, h: int, w: int, c: int,
                 transform: Callable[[torch.Tensor], torch.Tensor] | None = None):
        self._chunks = iter(chunks)
        self.shape = (int(n), int(h), int(w), int(c))
        self.ndim = 4
        self.dtype = torch.float32
        self._buf: torch.Tensor | None = None     # frames [self._at, self._at + len(buf))
        self._at = 0
        self._transform = transform
        self.last_frame: torch.Tensor | None = None
        self.delivered = 0

    def _fill(self, upto: int) -> None:
        while (self._buf is None or self._at + self._buf.shape[0] < upto):
            try:
                nxt = next(self._chunks)
            except StopIteration:
                raise StreamDecodeError(
                    f"the decoder produced {self._at + (0 if self._buf is None else self._buf.shape[0])} frames, "
                    f"{self.shape[0]} were expected") from None
            if self._transform is not None:
                nxt = self._transform(nxt)
            self._buf = nxt if self._buf is None else torch.cat([self._buf, nxt], 0)

    def _take(self, start: int, stop: int) -> torch.Tensor:
        if start < self._at:
            raise StreamDecodeError(f"frame {start} was already handed on (streamed frames are read once, in order)")
        stop = min(stop, self.shape[0])
        self._fill(stop)
        lo, hi = start - self._at, stop - self._at
        out = self._buf[lo:hi]
        self._buf = self._buf[hi:]                  # drop everything up to `stop`
        self._at = stop
        if out.shape[0]:
            self.last_frame = out[-1:].clone()
            self.delivered = stop
        return out

    def __getitem__(self, key):
        if isinstance(key, slice):
            if key.step not in (None, 1):
                raise StreamDecodeError("a streamed batch is read front to back")
            start = 0 if key.start is None else int(key.start)
            stop = self.shape[0] if key.stop is None else int(key.stop)
            return self._take(start, stop)
        if isinstance(key, int):
            return self._take(key, key + 1)[0]
        raise StreamDecodeError("a streamed batch supports frame indices and slices only")

    def close(self) -> None:
        close = getattr(self._chunks, "close", None)
        if close is not None:
            close()


# ── engines ─────────────────────────────────────────────────────────────────────────────────────────────────────
@dataclass
class StreamInfo:
    method: str
    exact: bool
    bounded: bool
    frames: int
    height: int
    width: int
    warnings: list = field(default_factory=list)


def _family(fsm) -> str:
    mod = type(fsm).__module__
    if type(fsm).__name__ == "WanVAE" and mod.endswith("wan.vae"):
        return "wan21"
    if type(fsm).__name__ == "WanVAE" and mod.endswith("wan.vae2_2"):
        return "wan22"
    if getattr(fsm, "comfy_has_chunked_io", False):
        return "chunked_io"
    return "whole"


def _wan_steps(fsm, z: torch.Tensor, family: str) -> Iterator[torch.Tensor]:
    """One latent step at a time with one cache for the clip: the step sequence of core's WanVAE.decode."""
    import importlib

    if family == "wan21":
        m = importlib.import_module(type(fsm).__module__)
        cache = [None] * m.count_cache_layers(fsm.decoder) if (1 + z.shape[2] // 2) > 1 else None
        x = fsm.conv2(z)
        steps = 1 + z.shape[2] // 2
        for i in range(steps):
            part = x[:, :, 0:1] if i == 0 else x[:, :, 1 + 2 * (i - 1):1 + 2 * i]
            if part.shape[2] == 0:
                continue
            out = fsm.decoder(part, feat_cache=cache, feat_idx=[0])
            yield torch.cat(out, 2) if isinstance(out, (list, tuple)) else out
    else:  # wan22
        m = importlib.import_module(type(fsm).__module__)
        cache = [None] * m.count_conv3d(fsm.decoder)
        x = fsm.conv2(z)
        for i in range(z.shape[2]):
            kw = {"first_chunk": True} if i == 0 else {}
            out = fsm.decoder(x[:, :, i:i + 1], feat_cache=cache, feat_idx=[0], **kw)
            yield m.unpatchify(out, patch_size=2)


class _Sink:
    """Stands in for core's output buffer: every finished frame range core writes goes straight to the queue."""

    def __init__(self, shape, put):
        self.shape = tuple(shape)
        self._put = put
        self._next = 0

    def _write(self, t0: int, part: torch.Tensor) -> None:
        if t0 != self._next:
            raise StreamDecodeError(f"the decoder wrote frames from {t0}, expected {self._next}: it does not write "
                                    f"in order, so it cannot be streamed exactly")
        self._next = t0 + part.shape[2]
        self._put(part)

    def copy_(self, src):                           # whole-buffer write (a decoder that did not chunk)
        self._write(0, src)
        return self

    def __getitem__(self, key):
        if not (isinstance(key, tuple) and len(key) >= 3 and isinstance(key[2], slice)):
            raise StreamDecodeError("unexpected write pattern into the stream buffer")
        sl = key[2]
        sink = self

        class _Range:
            def copy_(self, src):
                sink._write(0 if sl.start is None else int(sl.start), src)
                return self
        return _Range()


def _chunked_steps(fsm, z: torch.Tensor) -> Iterator[torch.Tensor]:
    shape = fsm.decode_output_shape(z.shape)
    q: queue.Queue = queue.Queue(maxsize=2)
    done = object()
    stop = threading.Event()

    def put(part):
        while not stop.is_set():
            try:
                q.put(part.detach(), timeout=0.5)
                return
            except queue.Full:
                continue
        raise StreamDecodeError("stream closed")

    def work():
        try:
            # no_grad, not inference_mode: inference tensors handed to the next node break its in-place ops
            # (tests/test_no_inference_tensors.py)
            with torch.no_grad():
                fsm.decode(z, output_buffer=_Sink(shape, put))
            q.put(done)
        except BaseException as exc:  # noqa: BLE001 - handed to the reading side
            q.put(exc)

    th = threading.Thread(target=work, name="c2c-stream-decode", daemon=True)
    th.start()
    try:
        while True:
            item = q.get()
            if item is done:
                return
            if isinstance(item, BaseException):
                raise item
            yield item
    finally:
        stop.set()
        th.join(timeout=60)


def _wan_tiles_via_disk(vae, fsm, zb: torch.Tensor, fam: str, tile_px: int, tmp_dir: str) -> Iterator[torch.Tensor]:
    """Tile by tile, each tile decoded over the WHOLE clip in one causal pass and written to its own file; then the
    frames are assembled chunk by chunk from those files and handed on in order.

    Why not every tile at once, step by step: the Wan 2.1 causal cache is ~1.2 GB per 512 px tile (bf16, measured),
    so all tiles of a 2K frame hold ~10-12 GB of state. One tile at a time needs one tile's cache. Plain sequential
    file writes and reads (not a memory map, whose touched pages count as resident RAM: measured 2.8 GB at 2K x 81).
    The arithmetic is decode_wan_spatial_tiled's - same tiles, same masks, tiles added in the same order - so the frames
    are identical to it."""
    import os
    import uuid

    import numpy as np

    from .._vae_tiled import _build_feather_mask, _plan_tile_spans

    _B, _C, T, H, W = zb.shape
    factor = vae.spacial_compression_decode() if hasattr(vae, "spacial_compression_decode") else 8
    tile_lat = max(1, tile_px // factor)
    eff = min(max(0, tile_lat // 4), (tile_lat - 1) // 2)
    h_tiles, w_tiles = _plan_tile_spans(H, tile_lat, eff), _plan_tile_spans(W, tile_lat, eff)
    out_h, out_w, ov_px = H * factor, W * factor, eff * factor
    t_out = 1 + 4 * (T - 1)
    os.makedirs(tmp_dir, exist_ok=True)
    tag = uuid.uuid4().hex[:12]
    tiles = []                                                # (path, oh0, oh1, ow0, ow1, mask)
    weight = torch.zeros((1, out_h, out_w, 1), dtype=torch.float32)
    try:
        for hs, he in h_tiles:
            for ws, we in w_tiles:
                oh0, oh1, ow0, ow1 = hs * factor, he * factor, ws * factor, we * factor
                mask = _build_feather_mask(oh1 - oh0, ow1 - ow0, ov_px, hs == 0, he == H, ws == 0, we == W,
                                           torch.float32, torch.device("cpu"))[0]          # [1, h, w, 1]
                path = os.path.join(tmp_dir, f"c2c_stream_{tag}_{len(tiles)}.f32")
                written = 0
                with open(path, "wb") as fh:
                    for raw in _wan_steps(fsm, zb[:, :, :, hs:he, ws:we], fam):
                        px = vae.process_output(raw.to(device="cpu", dtype=torch.float32, copy=True))[0].movedim(0, -1)
                        n = min(px.shape[0], t_out - written)
                        px[:n].contiguous().numpy().tofile(fh)
                        written += n
                tiles.append((path, oh0, oh1, ow0, ow1, mask))
                weight[:, oh0:oh1, ow0:ow1, :] += mask
        weight = weight.clamp(min=1e-6)
        step = max(1, int(64 * 2**20 // max(1, out_h * out_w * 3 * 4)))
        for t0 in range(0, t_out, step):
            k = min(step, t_out - t0)
            out = torch.zeros((k, out_h, out_w, 3), dtype=torch.float32)
            for path, oh0, oh1, ow0, ow1, mask in tiles:
                th, tw = oh1 - oh0, ow1 - ow0
                per = th * tw * 3
                part = np.fromfile(path, dtype=np.float32, count=k * per, offset=t0 * per * 4).reshape(k, th, tw, 3)
                out[:, oh0:oh1, ow0:ow1, :] += torch.from_numpy(part) * mask
            yield out / weight
    finally:
        for path, *_ in tiles:
            try:
                os.remove(path)
            except OSError:
                pass


def stream_decode(vae, latent: dict, tile_px: int = 0) -> tuple[FrameStream, StreamInfo]:
    """FrameStream of [k, H, W, 3] float32 chunks in 0..1 (core's process_output applied), and how it was made."""
    z = latent["samples"] if isinstance(latent, dict) else latent
    if not torch.is_tensor(z) or z.ndim != 5:
        raise StreamDecodeError("Stream save needs a video latent (5-D: batch, channels, time, height, width). "
                                "Use the images input for still-image latents.")
    fsm = vae.first_stage_model
    fam = _family(fsm)
    import comfy.model_management as mm

    step_shape = list(z.shape)
    step_shape[2] = min(step_shape[2], 2)
    try:
        mem = vae.memory_used_decode(tuple(step_shape) if fam != "whole" else tuple(z.shape), vae.vae_dtype)
    except Exception:  # noqa: BLE001
        mem = None
    mm.load_models_gpu([vae.patcher], memory_required=mem or 0,
                       force_full_load=getattr(vae, "disable_offload", False))
    out_shape = None
    if hasattr(fsm, "decode_output_shape"):
        try:
            out_shape = tuple(fsm.decode_output_shape(z.shape))
        except Exception:  # noqa: BLE001
            out_shape = None

    sf = vae.spacial_compression_decode() if hasattr(vae, "spacial_compression_decode") else 8
    tiled = fam == "wan21" and tile_px > 0 and max(z.shape[3], z.shape[4]) * sf > tile_px
    tmp_dir = ""
    if tiled:
        try:
            import folder_paths
            tmp_dir = folder_paths.get_temp_directory()
        except Exception:  # noqa: BLE001
            import tempfile
            tmp_dir = tempfile.gettempdir()

    def per_item(b: int) -> Iterator[torch.Tensor]:
        if fam == "whole":                      # core's full decode: already processed, [B, T, H, W, C]
            px = vae.decode(z[b:b + 1])
            yield px.reshape(-1, *px.shape[-3:]).to(device="cpu", dtype=torch.float32)
            return
        zb = z[b:b + 1].to(device=vae.device, dtype=vae.vae_dtype)
        if fam == "wan21" and tiled:
            yield from _wan_tiles_via_disk(vae, fsm, zb, fam, tile_px, tmp_dir)
            return
        steps = _wan_steps(fsm, zb, fam) if fam in ("wan21", "wan22") else _chunked_steps(fsm, zb)
        for raw in steps:
            # a copy: the chunk may be a view into the decoder's buffer, and core's process_output works in place
            px = vae.process_output(raw.to(device="cpu", dtype=torch.float32, copy=True))
            yield px[0].movedim(0, -1).contiguous()                      # [C, T, H, W] -> [T, H, W, C]

    def chunks() -> Iterator[torch.Tensor]:
        for b in range(z.shape[0]):
            yield from per_item(b)

    # frame count: core's own output-shape function when the VAE has one, else the causal rule (1 + 4 (T - 1))
    if out_shape is not None and len(out_shape) == 5:
        frames, h, w = out_shape[2], out_shape[3], out_shape[4]
    else:
        tf = 4
        frames = 1 + tf * (z.shape[2] - 1)
        sf = vae.spacial_compression_decode() if hasattr(vae, "spacial_compression_decode") else 8
        h, w = z.shape[3] * sf, z.shape[4] * sf
    frames *= z.shape[0]
    info = StreamInfo(
        method={"wan21": (f"Wan 2.1 VAE, {tile_px} px tiles, each decoded over the whole clip in one causal pass, "
                          f"blended on disk, streamed in order") if tiled
                else "Wan 2.1 VAE, streamed step by step with one causal cache",
                "wan22": "Wan 2.2 VAE, streamed step by step with one causal cache",
                "chunked_io": f"{type(fsm).__name__}: core's chunked output, streamed in order",
                "whole": f"{type(fsm).__name__}: decoded whole (no exact streaming for this VAE)"}[fam],
        exact=True, bounded=fam != "whole", frames=int(frames), height=int(h), width=int(w))
    if fam == "whole":
        info.warnings.append(f"{type(fsm).__name__} has no exact streaming decode, so the whole clip was decoded at "
                             f"once (not bounded in RAM). The frames are exact.")
    return FrameStream(chunks(), frames, h, w, 3), info
