"""Refinement mode A (ledger L9.214 S2, A26 4.4): independent tiles / temporal windows of one image or video, planned
deterministically, processed by GPU workers (c2c_distributed.tasks), stitched and checked before publishing.

- A tile's CORE is the area it owns (neighbouring cores overlap by ``overlap`` for blending); its READ area adds a
  ``halo`` of context on every side (clipped at the image edge) that the worker sees but that is cut away again -
  so a model that looks at neighbours does not invent edges. The halo is model-specific: measure it, do not guess.
- Temporal windows own ``window`` frames, overlap by ``overlap`` frames and read ``context`` frames on each side.
  Only plan windows when the model's temporal receptive field is known; never split a temporally attending model
  into independent frames.
- Stitch: every expected tile exactly once (else an error), feathered linear ramps over the overlaps, accumulated in
  float32 and normalised by the summed weights. Pixels outside every tile (outside ``roi``) are copied from the
  source unchanged and verified so.
- QC: coverage, protected region, finiteness, alpha range, and a seam ratio (mean absolute gradient across tile
  boundaries / inside tiles; ~1 means invisible seams).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch

from .tasks import TaskManifest, TileService, new_job_id


@dataclass(frozen=True)
class Tile:
    order: int
    tile_id: str
    core: tuple            # (y0, y1, x0, x1)
    read: tuple            # (y0, y1, x0, x1) = core + halo, clipped
    halo: tuple            # (top, bottom, left, right) actually available

    @property
    def core_in_read(self) -> tuple:
        return (self.core[0] - self.read[0], self.core[1] - self.read[0],
                self.core[2] - self.read[2], self.core[3] - self.read[2])


def _starts(length: int, size: int, overlap: int) -> list:
    if size >= length:
        return [0]
    step = size - overlap
    if step <= 0:
        raise ValueError("overlap must be smaller than the tile size")
    s = list(range(0, length - size, step))
    s.append(length - size)                                   # the last tile is aligned to the end
    return sorted(set(s))


def plan_tiles(height: int, width: int, tile: int, overlap: int, halo: int = 0, roi: Optional[tuple] = None) -> list:
    """Deterministic tile plan over ``roi`` (y0, y1, x0, x1; default the whole image)."""
    if tile <= 0 or overlap < 0 or halo < 0:
        raise ValueError("tile must be positive, overlap and halo non-negative")
    y0, y1, x0, x1 = roi or (0, height, 0, width)
    if not (0 <= y0 < y1 <= height and 0 <= x0 < x1 <= width):
        raise ValueError(f"roi {roi} is outside the {height}x{width} image")
    out = []
    for ty in _starts(y1 - y0, tile, overlap):
        for tx in _starts(x1 - x0, tile, overlap):
            cy0, cx0 = y0 + ty, x0 + tx
            cy1, cx1 = min(cy0 + tile, y1), min(cx0 + tile, x1)
            ry0, ry1 = max(0, cy0 - halo), min(height, cy1 + halo)
            rx0, rx1 = max(0, cx0 - halo), min(width, cx1 + halo)
            out.append(Tile(len(out), f"y{cy0}x{cx0}", (cy0, cy1, cx0, cx1), (ry0, ry1, rx0, rx1),
                            (cy0 - ry0, ry1 - cy1, cx0 - rx0, rx1 - cx1)))
    return out


def plan_windows(frames: int, window: int, overlap: int, context: int = 0) -> list:
    """Temporal windows as Tiles on the time axis: core / read are (t0, t1, 0, 0)."""
    out = []
    for t in _starts(frames, window, overlap):
        c0, c1 = t, min(t + window, frames)
        r0, r1 = max(0, c0 - context), min(frames, c1 + context)
        out.append(Tile(len(out), f"t{c0}", (c0, c1, 0, 0), (r0, r1, 0, 0), (c0 - r0, r1 - c1, 0, 0)))
    return out


def _ramp(n: int, lo_blend: int, hi_blend: int) -> torch.Tensor:
    """Weights along one axis of a core: rises over the first lo_blend samples, falls over the last hi_blend."""
    w = torch.ones(n, dtype=torch.float32)
    if lo_blend > 0:
        w[:lo_blend] = (torch.arange(lo_blend, dtype=torch.float32) + 0.5) / lo_blend
    if hi_blend > 0:
        w[n - hi_blend:] = torch.minimum(w[n - hi_blend:], (torch.arange(hi_blend, 0, -1, dtype=torch.float32) - 0.5)
                                         / hi_blend)
    return w


def feather(plan: list, overlap: int) -> dict:
    """Per tile: its [h, w] blend weight - ramps only on sides that overlap another tile's core."""
    out = {}
    cores = [t.core for t in plan]
    for t in plan:
        y0, y1, x0, x1 = t.core
        h, w = y1 - y0, x1 - x0

        top = max((c[1] - y0 for c in cores if c != t.core and c[0] < y0 < c[1] and _overlap_x(c, t.core)), default=0)
        bottom = max((y1 - c[0] for c in cores if c != t.core and c[0] < y1 < c[1] and _overlap_x(c, t.core)), default=0)
        left = max((c[3] - x0 for c in cores if c != t.core and c[2] < x0 < c[3] and _overlap_y(c, t.core)), default=0)
        right = max((x1 - c[2] for c in cores if c != t.core and c[2] < x1 < c[3] and _overlap_y(c, t.core)), default=0)
        out[t.tile_id] = _ramp(h, min(top, h), min(bottom, h))[:, None] * _ramp(w, min(left, w), min(right, w))[None, :]
    return out


def _overlap_x(a, b) -> bool:
    return a[2] < b[3] and b[2] < a[3]


def _overlap_y(a, b) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def stitch(source: torch.Tensor, plan: list, cores: dict, overlap: int) -> torch.Tensor:
    """source [..., H, W, C] (or [H, W, C]); cores: tile_id -> processed core [..., h, w, C]. Exactly-once checked."""
    ids = [t.tile_id for t in plan]
    missing, extra = sorted(set(ids) - set(cores)), sorted(set(cores) - set(ids))
    if missing or extra:
        raise ValueError(f"tile results do not match the plan: missing {missing[:5]}, unexpected {extra[:5]}")
    weights = feather(plan, overlap)
    acc = torch.zeros(source.shape, dtype=torch.float32)
    wsum = torch.zeros(source.shape[:-1] + (1,), dtype=torch.float32)
    for t in plan:
        y0, y1, x0, x1 = t.core
        c = cores[t.tile_id].to(torch.float32)
        if c.shape[-3:-1] != (y1 - y0, x1 - x0):
            raise ValueError(f"tile {t.tile_id}: core {tuple(c.shape)} does not fit {(y1 - y0, x1 - x0)}")
        wt = weights[t.tile_id][..., None]
        acc[..., y0:y1, x0:x1, :] += c * wt
        wsum[..., y0:y1, x0:x1, :] += wt
    covered = wsum > 0
    out = torch.where(covered, acc / wsum.clamp_min(1e-12), source.to(torch.float32))
    return out.to(source.dtype)


def seam_ratio(img: torch.Tensor, plan: list) -> Optional[float]:
    """Mean |horizontal + vertical gradient| on tile-core boundaries / inside cores (~1: no visible seams)."""
    x = img.to(torch.float32)
    while x.dim() > 3:
        x = x[0]
    gy = (x[1:, :, :] - x[:-1, :, :]).abs().mean(-1)
    gx = (x[:, 1:, :] - x[:, :-1, :]).abs().mean(-1)
    H, W = x.shape[:2]
    by = torch.zeros(H - 1, dtype=torch.bool)
    bx = torch.zeros(W - 1, dtype=torch.bool)
    for t in plan:
        y0, y1, x0, x1 = t.core
        for v in (y0, y1):
            if 0 < v < H:
                by[v - 1] = True
        for v in (x0, x1):
            if 0 < v < W:
                bx[v - 1] = True
    if not by.any() and not bx.any():
        return None
    seam = torch.cat([gy[by].flatten(), gx[:, bx].flatten()])
    inner = torch.cat([gy[~by].flatten(), gx[:, ~bx].flatten()])
    return float(seam.mean() / inner.mean().clamp_min(1e-12)) if inner.numel() else None


def qc(source: torch.Tensor, out: torch.Tensor, plan: list, alpha_channel: Optional[int] = None,
       seam_limit: float = 1.5) -> dict:
    cov = torch.zeros(source.shape[-3:-1], dtype=torch.bool)
    for t in plan:
        cov[t.core[0]:t.core[1], t.core[2]:t.core[3]] = True
    outside_same = bool(torch.equal(out[..., ~cov, :], source[..., ~cov, :])) if (~cov).any() else True
    rep = {"tiles": len(plan), "finite": bool(torch.isfinite(out.float()).all()),
           "protected_unchanged": outside_same, "covered_fraction": float(cov.float().mean()),
           "seam_ratio": seam_ratio(out, plan)}
    if alpha_channel is not None:
        a = out[..., alpha_channel].float()
        rep["alpha_in_range"] = bool(((a >= -1e-4) & (a <= 1 + 1e-4)).all())
    rep["ok"] = rep["finite"] and outside_same and rep.get("alpha_in_range", True) and \
        (rep["seam_ratio"] is None or rep["seam_ratio"] <= seam_limit)
    return rep


def run_tiled(service: TileService, image: torch.Tensor, fn: Callable, *, tile: int, overlap: int, halo: int = 0,
              roi: Optional[tuple] = None, model_fingerprint: str = "", vram_bytes: int = 0, seed: Optional[int] = None,
              colour_space: str = "", alpha_semantics: str = "", timeout: Optional[float] = None,
              task_type: str = "c2c.tile") -> tuple:
    """Process ``image`` [B, H, W, C] tile by tile on the service's workers. ``fn(read_crop, ctx) -> processed crop``
    (same shape as the read crop: halo included). Returns (stitched image, qc report, job info)."""
    if task_type not in service._executors:
        def ex(task, ctx):
            out = task.payload["fn"](task.payload["read"], ctx)
            cy0, cy1, cx0, cx1 = task.payload["core_in_read"]
            return out[..., cy0:cy1, cx0:cx1, :].contiguous()
        service.register_executor(task_type, ex)
    plan = plan_tiles(image.shape[-3], image.shape[-2], tile, overlap, halo, roi)
    job = new_job_id("tiles")
    manifests = []
    for t in plan:
        ry0, ry1, rx0, rx1 = t.read
        read = image[..., ry0:ry1, rx0:rx1, :].contiguous()
        cy0, cy1, cx0, cx1 = t.core_in_read
        manifests.append(TaskManifest(
            job_id=job, task_id=t.tile_id, order=t.order, task_type=task_type, model_fingerprint=model_fingerprint,
            tile_or_window_id=t.tile_id, spatial_bounds=t.core, halo_bounds=t.halo + (0, 0), seed=seed,
            expected_output_shape=tuple(image.shape[:-3]) + (cy1 - cy0, cx1 - cx0, image.shape[-1]),
            expected_dtype=str(image.dtype).replace("torch.", ""), colour_space=colour_space,
            alpha_semantics=alpha_semantics, vram_bytes=vram_bytes,
            payload={"fn": fn, "read": read, "core_in_read": t.core_in_read}))
    service.submit(manifests)
    state = service.wait(job, timeout)
    info = service.job(job)
    if state != "DONE":
        service.cancel(job)
        raise RuntimeError(f"tiled job ended {state}: {info.get('error')}")
    res = service.results(job)
    out = stitch(image, plan, {r.task_id: r.output for r in res}, overlap)
    report = qc(image, out, plan)
    info["workers_used"] = sorted({r.worker_id for r in res})
    service.close(job)
    return out, report, info


def stitch_windows(video: torch.Tensor, plan: list, cores: dict, overlap: int) -> torch.Tensor:
    """Temporal windows: video [T, ...]; cores: window id -> processed core frames [t1 - t0, ...]. Same exactly-once
    check and float32 feathered blend as ``stitch``, along time."""
    T = video.shape[0]
    flat = video.reshape(T, -1, 1)
    n = flat.shape[1]
    tiles = [Tile(t.order, t.tile_id, (t.core[0], t.core[1], 0, n), (t.read[0], t.read[1], 0, n), t.halo)
             for t in plan]
    flat_cores = {k: v.reshape(v.shape[0], -1, 1) for k, v in cores.items()}
    return stitch(flat, tiles, flat_cores, overlap).reshape(video.shape)
