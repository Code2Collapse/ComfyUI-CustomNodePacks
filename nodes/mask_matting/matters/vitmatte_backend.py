"""ViTMatte matter backend — uses HuggingFace ``transformers``.

Loads ViTMatte weights from ``ComfyUI/models/vitmatte/``. If the user only
has a HF model id (e.g. ``hustvl/vitmatte-small-composition-1k``), this
backend will pull from the HF cache transparently.
"""
from __future__ import annotations

import logging
import math
import os
import time
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from ..utils import (
    backend_first_root,
    free_vram,
    interruptible_range,
    list_backend_files,
    mask_to_trimap,
    resolve_backend_weight,
    to_bhwc,
    to_mask,
)
from . import BaseMatter, register

logger = logging.getLogger("MEC.MaskMatting.ViTMatte")

_DEFAULT_TILE = 512
_DEFAULT_TILE_OVERLAP = 64


def unpad_alpha(alpha: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """Map ViTMatte's output alpha back onto the (H, W) frame.

    The HF processor PADS the frame bottom/right to a multiple of 32 - it
    never resizes it (1080x1920 -> 1088x1920). The padding is cropped off.
    This used to resize the padded alpha to (H, W), which STRETCHED it and
    slid the matte off the true edges, increasingly towards the bottom/right
    (~8 px at the bottom of a 1080p frame). A processor that does resize is
    still mapped back by interpolation."""
    if alpha.shape[-2] >= H and alpha.shape[-1] >= W:
        return alpha[..., :H, :W]
    return F.interpolate(alpha, size=(H, W), mode="bilinear", align_corners=False)


def _have_transformers() -> bool:
    try:
        import transformers  # noqa: F401
        return True
    except ImportError:
        return False


def _trimap_float_to_uint8(tri: np.ndarray) -> np.ndarray:
    """Map float trimap values to ViTMatte uint8 0 / 127 / 255."""
    t = np.asarray(tri, dtype=np.float32)
    out = np.zeros(t.shape, dtype=np.uint8)
    out[t > 0.75] = 255
    unk = (t > 0.25) & (t <= 0.75)
    out[unk] = 127
    out[(t > 0.0) & (t <= 0.25)] = 0
    # values in (0, 0.25] that are not exactly 0 still count as bg
    return out


def _safe_pad_2d(
    arr: np.ndarray,
    pad_h: int,
    pad_w: int,
    *,
    is_image: bool = False,
) -> np.ndarray:
    """Pad bottom/right; use edge/constant modes that work for 1×1 inputs."""
    if pad_h <= 0 and pad_w <= 0:
        return arr
    h, w = arr.shape[:2]
    mode = "edge"
    if h < 2 or w < 2:
        mode = "edge"
    elif pad_h >= h or pad_w >= w:
        mode = "edge"
    if is_image:
        return np.pad(arr, ((0, pad_h), (0, pad_w), (0, 0)), mode=mode)
    return np.pad(arr, ((0, pad_h), (0, pad_w)), mode=mode)


def _pad_to_tile(
    patch_img: np.ndarray,
    merged_tri: np.ndarray,
    th: int,
    tw: int,
    tile_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if th >= tile_size and tw >= tile_size:
        return patch_img, merged_tri
    pad_h = max(0, tile_size - th)
    pad_w = max(0, tile_size - tw)
    patch_img = _safe_pad_2d(patch_img, pad_h, pad_w, is_image=True)
    merged_tri = _safe_pad_2d(merged_tri, pad_h, pad_w, is_image=False)
    return patch_img, merged_tri


def _trimaps_overlap_in_tile(
    tri_i: np.ndarray,
    tri_j: np.ndarray,
    ty: int, ty1: int, tx: int, tx1: int,
) -> bool:
    a = tri_i[ty:ty1, tx:tx1] > 0
    b = tri_j[ty:ty1, tx:tx1] > 0
    return bool(np.any(a & b))


def _group_objects_for_tile(
    obj_idx: List[int],
    trimaps: List[np.ndarray],
    ty: int, ty1: int, tx: int, tx1: int,
) -> List[List[int]]:
    groups: List[List[int]] = []
    for i in obj_idx:
        placed = False
        for g in groups:
            if all(
                not _trimaps_overlap_in_tile(trimaps[i], trimaps[j], ty, ty1, tx, tx1)
                for j in g
            ):
                g.append(i)
                placed = True
                break
        if not placed:
            groups.append([i])
    return groups


def _bbox_from_support(
    support: np.ndarray,
    H: int, W: int,
    pad: int,
) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(support)
    if not len(xs):
        return None
    x0 = max(0, int(xs.min()) - pad)
    y0 = max(0, int(ys.min()) - pad)
    x1 = min(W, int(xs.max()) + 1 + pad)
    y1 = min(H, int(ys.max()) + 1 + pad)
    return x0, y0, x1, y1


@register
class ViTMatteMatter(BaseMatter):
    KEY = "vitmatte"
    #: Widgets this matter reads; the node shows only these.
    PARAMS = (
        "trimap_dilate", "trimap_erode", "matte_resolution",
        "matte_tile", "matte_overlap", "matte_tile_batch",
    )
    #: Carries state between frames, so it does not flicker on video.
    TEMPORAL = False
    DISPLAY = "ViTMatte"
    MODELS_KEY = "vitmatte"
    STATUS = "ready" if _have_transformers() else "missing-deps"

    def load(self) -> None:
        if self._model is not None:
            return
        if not _have_transformers():
            raise RuntimeError("transformers not installed. `pip install transformers`.")
        from transformers import VitMatteForImageMatting, VitMatteImageProcessor

        def _is_hf_model_dir(p: str) -> bool:
            return (
                isinstance(p, str)
                and os.path.isdir(p)
                and os.path.isfile(os.path.join(p, "preprocessor_config.json"))
                and os.path.isfile(os.path.join(p, "config.json"))
            )

        candidates: list[str] = []
        path = resolve_backend_weight(self.MODELS_KEY, self.model_name) if self.model_name else None
        if path and os.path.isdir(path):
            candidates.append(path)
        elif path and os.path.isfile(path):
            candidates.append(os.path.dirname(path))

        try:
            root = backend_first_root(self.MODELS_KEY)
            if root and os.path.isdir(root):
                if _is_hf_model_dir(root):
                    candidates.append(root)
                for entry in sorted(os.listdir(root)):
                    sub = os.path.join(root, entry)
                    if _is_hf_model_dir(sub):
                        candidates.append(sub)
        except Exception:
            pass

        src: Optional[str] = None
        for c in candidates:
            if _is_hf_model_dir(c):
                src = c
                break

        if src is None:
            src = self.model_name or "hustvl/vitmatte-small-composition-1k"
            if "/" not in src and not os.path.isabs(src):
                src = "hustvl/vitmatte-small-composition-1k"
            logger.info(
                "[ViTMatte] no local HF model dir found under "
                "ComfyUI/models/vitmatte/ \u2014 falling back to HF Hub: %s", src,
            )

        dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}.get(
            self.precision, torch.float16,
        )
        self._processor = VitMatteImageProcessor.from_pretrained(src)
        self._model = VitMatteForImageMatting.from_pretrained(
            src, torch_dtype=dtype,
        ).to(self.device).eval()
        self._dtype = dtype

    # ── single-object path ──────────────────────────────────────────────

    def matte(
        self,
        image_bhwc,
        coarse_mask,
        *,
        trimap=None,
        edge_radius=4,
        memory_size=8,
        tile_size=_DEFAULT_TILE,
        tile_overlap=_DEFAULT_TILE_OVERLAP,
        tile_batch=4,
    ):
        """Single-object matte via per-object optimised tiling.

        When ``trimap`` is supplied, bbox and trimap bands come from it;
        otherwise they are derived from ``coarse_mask`` and ``edge_radius``.
        """
        try:
            t0 = time.perf_counter()
            t1 = t0
            img = to_bhwc(image_bhwc)
            B, H, W, _ = img.shape
            coarse = to_mask(coarse_mask)
            tri_all = to_mask(trimap) if trimap is not None else None
            alphas = []
            loaded = False
            for i in interruptible_range(B, label="vitmatte"):
                frame = (img[i].cpu().numpy() * 255).astype(np.uint8)
                m_np = coarse[i].cpu().numpy().astype(np.float32)
                ext_tri: Optional[np.ndarray] = None
                if tri_all is not None:
                    tri_slice = tri_all[i] if tri_all.ndim == 3 else tri_all
                    ext_tri = _trimap_float_to_uint8(tri_slice.cpu().numpy())
                    support = ext_tri > 0
                else:
                    support = m_np > 0.5
                pad = int(edge_radius) * 2
                bbox = _bbox_from_support(support, H, W, pad)
                if bbox is None:
                    alphas.append(torch.zeros(H, W, dtype=torch.float32))
                    continue
                if not loaded:
                    self.load()
                    t1 = time.perf_counter()
                    loaded = True
                ext_trimaps = [ext_tri] if ext_tri is not None else None
                result = self._matte_multi_frame(
                    frame, [m_np], [bbox], H, W, int(edge_radius),
                    tile_size=int(tile_size),
                    tile_overlap=int(tile_overlap),
                    tile_batch=int(tile_batch),
                    external_trimaps=ext_trimaps,
                )
                alphas.append(result[0])
            t2 = time.perf_counter()
            logger.debug(
                "[ViTMatte] matte (optimised) B=%d  load=%.2fs infer=%.2fs total=%.2fs",
                B, t1 - t0, t2 - t1, t2 - t0,
            )
            return {"alpha": torch.stack(alphas, 0).clamp(0, 1),
                    "info": {"backend": self.KEY}}
        except Exception:
            free_vram()
            raise

    # ── multi-object path ────────────────────────────────────────────────

    def matte_multi(
        self,
        image_bhwc: torch.Tensor,
        object_masks_list: List[torch.Tensor],
        object_bboxes_list: List[Tuple],
        *,
        edge_radius: int = 4,
        memory_size: int = 8,
        tile_size: int = _DEFAULT_TILE,
        tile_overlap: int = _DEFAULT_TILE_OVERLAP,
        tile_batch: int = 4,
    ) -> dict:
        """Run VitMatte for N objects with per-object optimised tile placement."""
        try:
            t0 = time.perf_counter()
            t1 = t0
            img = to_bhwc(image_bhwc)
            B, H, W, _ = img.shape
            N = len(object_masks_list)

            if N == 0:
                zero = torch.zeros(B, H, W)
                return {"alpha": zero, "object_alphas": [], "object_boxes": [],
                        "info": {"backend": self.KEY, "n_objects": 0}}

            per_frame: List[List[torch.Tensor]] = []
            loaded = False
            for b in interruptible_range(B, label="vitmatte-multi"):
                frame_hwc = (img[b].cpu().numpy() * 255).astype(np.uint8)
                frame_masks = []
                any_support = False
                for mt in object_masks_list:
                    m = mt[b] if mt.ndim == 3 else mt
                    m_np = m.cpu().numpy().astype(np.float32)
                    frame_masks.append(m_np)
                    if np.any(m_np > 0.5):
                        any_support = True
                if not any_support:
                    per_frame.append(
                        [torch.zeros(H, W, dtype=torch.float32) for _ in range(N)]
                    )
                    continue
                if not loaded:
                    self.load()
                    t1 = time.perf_counter()
                    loaded = True
                per_frame.append(
                    self._matte_multi_frame(
                        frame_hwc, frame_masks, object_bboxes_list, H, W,
                        int(edge_radius),
                        tile_size=int(tile_size),
                        tile_overlap=int(tile_overlap),
                        tile_batch=int(tile_batch),
                    )
                )

            object_alphas: List[torch.Tensor] = []
            for i in range(N):
                frames = [per_frame[b][i] for b in range(B)]
                object_alphas.append(torch.stack(frames, 0).clamp(0, 1))

            merged = torch.stack(object_alphas, 0).max(dim=0).values.clamp(0, 1)
            t2 = time.perf_counter()
            logger.debug(
                "[ViTMatte] matte_multi B=%d N=%d  load=%.2fs infer=%.2fs total=%.2fs",
                B, N, t1 - t0, t2 - t1, t2 - t0,
            )
            return {
                "alpha": merged,
                "object_alphas": object_alphas,
                "object_boxes": object_bboxes_list,
                "info": {"backend": self.KEY, "n_objects": N},
            }
        except Exception:
            free_vram()
            raise

    # Object-aware tiling and matte_multi: John Brisbin (PR #10)
    # ── per-frame multi-object implementation ───────────────────────────

    def _matte_multi_frame(
        self,
        frame_hwc: np.ndarray,
        object_masks: List[np.ndarray],
        object_bboxes: List[Tuple],
        H: int, W: int,
        edge_radius: int,
        *,
        tile_size: int = _DEFAULT_TILE,
        tile_overlap: int = _DEFAULT_TILE_OVERLAP,
        tile_batch: int = 4,
        external_trimaps: Optional[List[Optional[np.ndarray]]] = None,
    ) -> List[torch.Tensor]:
        _t0_frame = time.perf_counter()
        N = len(object_masks)
        if N == 0:
            return []

        tile_size = max(1, int(tile_size))
        tile_overlap = max(1, int(tile_overlap))
        stride = max(1, tile_size - tile_overlap)

        dilate = edge_radius * 2
        erode = edge_radius
        pad = dilate

        trimaps: List[np.ndarray] = []
        for idx, m_np in enumerate(object_masks):
            if (
                external_trimaps is not None
                and idx < len(external_trimaps)
                and external_trimaps[idx] is not None
            ):
                tri_u8 = external_trimaps[idx]
                if tri_u8.shape != (H, W):
                    tri_u8 = tri_u8[:H, :W]
                trimaps.append(tri_u8.astype(np.uint8, copy=False))
            else:
                m_t = torch.from_numpy(m_np).unsqueeze(0)
                tri_t = mask_to_trimap(m_t, dilate=dilate, erode=erode)
                trimaps.append((tri_t[0].cpu().numpy() * 255).astype(np.uint8))

        tile_sets: List[List[Tuple[int, int]]] = []
        for i, bbox in enumerate(object_bboxes):
            positions = self._object_tile_positions(
                bbox, H, W, pad, tile_size=tile_size, tile_overlap=tile_overlap,
            )
            tile_sets.append(positions)
            logger.debug("[ViTMatte] obj %d bbox=%s → %d tile(s)", i, bbox, len(positions))

        tile_map: dict = {}
        for i, tset in enumerate(tile_sets):
            for pos in tset:
                tile_map.setdefault(pos, []).append(i)

        global_full = (
            len(self._global_tile_starts_1d(W, tile_size, stride))
            * len(self._global_tile_starts_1d(H, tile_size, stride))
        )
        logger.debug(
            "[ViTMatte] multi-frame N=%d unique_tiles=%d  full-image_tiles=%d",
            N, len(tile_map), global_full,
        )

        alpha_acc = np.zeros((N, H, W), dtype=np.float64)
        weight_acc = np.zeros((N, H, W), dtype=np.float64)

        jobs: List[dict] = []
        for (tx, ty), obj_idx in tile_map.items():
            ty1, tx1 = min(ty + tile_size, H), min(tx + tile_size, W)
            th, tw = ty1 - ty, tx1 - tx
            groups = _group_objects_for_tile(obj_idx, trimaps, ty, ty1, tx, tx1)
            for group in groups:
                patch_img = frame_hwc[ty:ty1, tx:tx1].copy()
                merged_tri = np.zeros((th, tw), dtype=np.uint8)
                for i in group:
                    merged_tri = np.maximum(merged_tri, trimaps[i][ty:ty1, tx:tx1])
                patch_img, merged_tri = _pad_to_tile(
                    patch_img, merged_tri, th, tw, tile_size,
                )
                jobs.append({
                    "tx": tx, "ty": ty, "th": th, "tw": tw,
                    "group": group,
                    "patch_img": patch_img,
                    "merged_tri": merged_tri,
                })

        self._run_tile_jobs_batched(
            jobs, tile_batch, tile_overlap, trimaps, alpha_acc, weight_acc,
        )

        result: List[torch.Tensor] = []
        for i in range(N):
            wgt = weight_acc[i]
            with np.errstate(invalid="ignore", divide="ignore"):
                alpha = np.where(wgt > 1e-8, alpha_acc[i] / wgt, 0.0).astype(np.float32)
            alpha *= (trimaps[i] > 0).astype(np.float32)
            if N > 1:
                # Instance ownership: object i's dilated band reaches into touching neighbours,
                # and the model cannot tell one object's foreground from another's, so zero i
                # on pixels that belong ONLY to another object. Pixels both masks claim (true
                # overlap) keep i's alpha - zeroing them for both would cut a hole in the merge.
                own = object_masks[i] > 0.5
                for j in range(N):
                    if j != i:
                        alpha[(object_masks[j] > 0.5) & ~own] = 0.0
            result.append(torch.from_numpy(alpha))
        elapsed_frame = time.perf_counter() - _t0_frame
        logger.debug(
            "[ViTMatte] _matte_multi_frame N=%d unique_tiles=%d  %.2fs",
            N, len(tile_map), elapsed_frame,
        )
        return result

    def _forward_batch(self, jobs: List[dict]) -> List[np.ndarray]:
        """Run up to len(jobs) tiles in one model forward; return (th,tw) alphas."""
        if not jobs:
            return []
        batch_inputs = []
        for job in jobs:
            inputs = self._processor(
                images=job["patch_img"],
                trimaps=job["merged_tri"],
                return_tensors="pt",
            )
            batch_inputs.append(inputs)

        keys = batch_inputs[0].keys()
        merged: dict = {}
        for k in keys:
            merged[k] = torch.cat([bi[k] for bi in batch_inputs], dim=0)
        merged = {
            k: v.to(
                self.device,
                dtype=self._dtype if v.dtype.is_floating_point else v.dtype,
            )
            for k, v in merged.items()
        }
        with torch.no_grad(), torch.autocast(
            self.device, dtype=self._dtype, enabled=(self.device == "cuda"),
        ):
            out = self._model(**merged)

        alphas_out: List[np.ndarray] = []
        for j, job in enumerate(jobs):
            th, tw = job["th"], job["tw"]
            patch_alpha = out.alphas[j, 0].float().cpu().numpy()[:th, :tw]
            alphas_out.append(patch_alpha)
        return alphas_out

    def _run_tile_jobs_batched(
        self,
        jobs: List[dict],
        tile_batch: int,
        tile_overlap: int,
        trimaps: List[np.ndarray],
        alpha_acc: np.ndarray,
        weight_acc: np.ndarray,
    ) -> None:
        max_batch = max(1, int(tile_batch))
        idx = 0
        while idx < len(jobs):
            bs = min(max_batch, len(jobs) - idx)
            while bs >= 1:
                chunk = jobs[idx: idx + bs]
                try:
                    patch_alphas = self._forward_batch(chunk)
                except torch.cuda.OutOfMemoryError:
                    free_vram()
                    if bs <= 1:
                        raise RuntimeError(
                            "ViTMatte ran out of GPU memory even one tile at a time — "
                            "lower matte_tile or use the CPU."
                        )
                    bs //= 2
                    continue
                except RuntimeError as exc:
                    if "out of memory" not in str(exc).lower():
                        raise
                    free_vram()
                    if bs <= 1:
                        raise RuntimeError(
                            "ViTMatte ran out of GPU memory even one tile at a time — "
                            "lower matte_tile or use the CPU."
                        )
                    bs //= 2
                    continue
                for job, patch_alpha in zip(chunk, patch_alphas):
                    ty, tx = job["ty"], job["tx"]
                    th, tw = job["th"], job["tw"]
                    w = self._blend_weight(th, tw, tile_overlap)
                    for i in job["group"]:
                        tri_m = (trimaps[i][ty:ty + th, tx:tx + tw] > 0).astype(np.float64)
                        ow = w * tri_m
                        alpha_acc[i, ty:ty + th, tx:tx + tw] += patch_alpha * ow
                        weight_acc[i, ty:ty + th, tx:tx + tw] += ow
                idx += len(chunk)
                break

    # ── tile-placement helpers ───────────────────────────────────────────

    @staticmethod
    def _global_tile_starts_1d(length: int, tile: int, stride: int) -> List[int]:
        if length <= tile:
            return [0]
        starts = list(range(0, length - tile, stride))
        last = length - tile
        if not starts or starts[-1] != last:
            starts.append(last)
        return starts

    @staticmethod
    def _optimal_tile_starts_1d(
        region_start: int, region_end: int,
        img_length: int, tile: int, min_overlap: int,
    ) -> List[int]:
        """Minimum evenly-spaced tiles covering [region_start, region_end]."""
        length = region_end - region_start
        if length <= 0:
            return []
        max_stride = tile - min_overlap
        if length <= tile:
            centre = region_start + length // 2
            s = max(0, min(img_length - tile, centre - tile // 2))
            return [s]
        n = 1 + math.ceil((length - tile) / max_stride)
        first = max(0, region_start)
        last = min(img_length - tile, region_end - tile)
        if last < first:
            last = first
        if n == 1:
            return [first]
        return [round(first + i * (last - first) / (n - 1)) for i in range(n)]

    @classmethod
    def _best_tile_starts_1d(
        cls,
        px0: int, px1: int,
        img_length: int, tile: int, stride: int, min_overlap: int,
    ) -> List[int]:
        """Per-object optimal when fewer tiles than global grid; else global."""
        global_all = cls._global_tile_starts_1d(img_length, tile, stride)
        global_bbox = [t for t in global_all if t + tile > px0 and t < px1]
        optimal = cls._optimal_tile_starts_1d(px0, px1, img_length, tile, min_overlap)
        return optimal if len(optimal) < len(global_bbox) else global_bbox

    @classmethod
    def _object_tile_positions(
        cls,
        bbox: Tuple,
        H: int, W: int,
        pad: int,
        *,
        tile_size: int = _DEFAULT_TILE,
        tile_overlap: int = _DEFAULT_TILE_OVERLAP,
    ) -> List[Tuple[int, int]]:
        stride = max(1, tile_size - tile_overlap)
        x0, y0, x1, y1 = (int(v) for v in bbox)
        px0, py0 = max(0, x0 - pad), max(0, y0 - pad)
        px1, py1 = min(W, x1 + pad), min(H, y1 + pad)
        xs = cls._best_tile_starts_1d(px0, px1, W, tile_size, stride, tile_overlap)
        ys = cls._best_tile_starts_1d(py0, py1, H, tile_size, stride, tile_overlap)
        return [(x, y) for y in ys for x in xs]

    # ── blend weight ─────────────────────────────────────────────────────

    @staticmethod
    def _blend_weight(h: int, w: int, overlap: int) -> np.ndarray:
        wy = np.ones(h, dtype=np.float32)
        wx = np.ones(w, dtype=np.float32)
        for i in range(min(overlap, h // 2)):
            v = 0.5 * (1.0 - np.cos(np.pi * (i + 1) / (overlap + 1)))
            wy[i] = min(wy[i], v)
            wy[h - 1 - i] = min(wy[h - 1 - i], v)
        for i in range(min(overlap, w // 2)):
            v = 0.5 * (1.0 - np.cos(np.pi * (i + 1) / (overlap + 1)))
            wx[i] = min(wx[i], v)
            wx[w - 1 - i] = min(wx[w - 1 - i], v)
        return np.outer(wy, wx)

    # ── self-verification ────────────────────────────────────────────────

    @classmethod
    def _verify_tile_logic(cls) -> None:
        """Smoke-test tile placement. Raises AssertionError on failure."""
        tile, min_ov = _DEFAULT_TILE, _DEFAULT_TILE_OVERLAP
        stride = tile - min_ov
        cases = [
            (0, 512, 4080, 1, None),
            (0, 511, 4080, 1, None),
            (0, 960, 4080, 2, 64),
            (0, 961, 4080, 3, 64),
            (200, 1113, 4080, 2, 64),
            (0, 1000, 4080, 3, 64),
        ]
        for rs, re, L, n_exp, ov_exp in cases:
            pos = cls._optimal_tile_starts_1d(rs, re, L, tile, min_ov)
            assert len(pos) == n_exp, (
                f"_optimal({rs},{re},{L}): expected {n_exp} got {len(pos)}: {pos}"
            )
            assert pos[0] >= 0 and pos[-1] + tile <= L, f"out of image: {pos}"
            assert pos[0] <= max(0, rs), (
                f"first tile misses region start: {pos[0]} > {rs}"
            )
            assert pos[-1] + tile >= re, (
                f"last tile misses region end: {pos[-1] + tile} < {re}"
            )
            if ov_exp:
                for a, b in zip(pos, pos[1:]):
                    ov = (a + tile) - b
                    assert ov >= ov_exp, (
                        f"overlap {ov} < {ov_exp} between tiles {a} and {b}"
                    )

        best = cls._best_tile_starts_1d(200, 1113, 4080, tile, stride, min_ov)
        assert len(best) == 2, (
            f"expected 2 best tiles for [200,1113], got {len(best)}: {best}"
        )

        tiles = cls._object_tile_positions(
            (200, 100, 1113, 800), 3072, 4080, pad=8,
            tile_size=tile, tile_overlap=min_ov,
        )
        assert len(tiles) == 4, (
            f"expected 4 tiles for 913×700, got {len(tiles)}: {tiles}"
        )
        logger.debug("[ViTMatte] _verify_tile_logic: all checks passed")
