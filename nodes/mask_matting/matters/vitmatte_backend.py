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

_TILE   = 512
_MIN_OV = 64
_STRIDE = _TILE - _MIN_OV   # 448


def _have_transformers() -> bool:
    try:
        import transformers  # noqa: F401
        return True
    except ImportError:
        return False


@register
class ViTMatteMatter(BaseMatter):
    KEY = "vitmatte"
    DISPLAY = "ViTMatte"
    MODELS_KEY = "vitmatte"
    STATUS = "ready" if _have_transformers() else "missing-deps"

    def load(self) -> None:
        if self._model is not None:
            return
        if not _have_transformers():
            raise RuntimeError("transformers not installed. `pip install transformers`.")
        from transformers import VitMatteForImageMatting, VitMatteImageProcessor

        # ── Resolve a usable HF model directory ──────────────────────────
        # Acceptable layouts under ComfyUI/models/vitmatte/:
        #   * <root>/<repo-name>/{preprocessor_config.json, config.json, *.safetensors}
        #   * <root>/{preprocessor_config.json, config.json, *.safetensors}    (flat)
        #   * a single .safetensors file with a sibling preprocessor_config.json
        # Anything else → fall back to the HF Hub repo id so the user gets a
        # real model instead of OSError("Can't load image processor for ...").
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

        # Walk the backend root to find any HF-style sub-folder the user
        # might have downloaded (e.g. via huggingface-cli or git-lfs).
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
            # No usable local layout → use HF id (or a sensible default).
            src = self.model_name or "hustvl/vitmatte-small-composition-1k"
            # If the resolved name still looks like a bare filename (no '/'),
            # prepend the canonical HF org so transformers can fetch it.
            if "/" not in src and not os.path.isabs(src):
                src = "hustvl/vitmatte-small-composition-1k"
            logger.info(
                "[ViTMatte] no local HF model dir found under "
                "ComfyUI/models/vitmatte/ \u2014 falling back to HF Hub: %s", src,
            )

        dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}.get(self.precision, torch.float16)
        self._processor = VitMatteImageProcessor.from_pretrained(src)
        self._model = VitMatteForImageMatting.from_pretrained(src, torch_dtype=dtype).to(self.device).eval()
        self._dtype = dtype

    # ── single-object path ──────────────────────────────────────────────

    def matte(self, image_bhwc, coarse_mask, *, trimap=None, edge_radius=4, memory_size=8):
        """Single-object matte via per-object optimised tiling.

        ``trimap`` is accepted for API compatibility but ignored — the trimap
        is derived internally from ``coarse_mask`` and ``edge_radius``.
        """
        try:
            t0 = time.perf_counter()
            self.load()
            t1 = time.perf_counter()
            img    = to_bhwc(image_bhwc)
            B, H, W, _ = img.shape
            coarse = to_mask(coarse_mask)
            alphas = []
            for i in interruptible_range(B, label="vitmatte"):
                frame = (img[i].cpu().numpy() * 255).astype(np.uint8)
                m_np  = coarse[i].cpu().numpy().astype(np.float32)
                ys, xs = np.where(m_np > 0.5)
                if not len(xs):
                    alphas.append(torch.zeros(H, W, dtype=torch.float32))
                    continue
                bbox = (int(xs.min()), int(ys.min()),
                        int(xs.max()) + 1, int(ys.max()) + 1)
                result = self._matte_multi_frame(
                    frame, [m_np], [bbox], H, W, int(edge_radius))
                alphas.append(result[0])
            t2 = time.perf_counter()
            logger.warning("[ViTMatte] matte (optimised) B=%d  load=%.2fs infer=%.2fs total=%.2fs",
                           B, t1-t0, t2-t1, t2-t0)
            return {"alpha": torch.stack(alphas, 0).clamp(0, 1),
                    "info": {"backend": self.KEY}}
        except Exception:
            free_vram()
            raise

    # ── multi-object path ────────────────────────────────────────────────

    def matte_multi(
        self,
        image_bhwc: torch.Tensor,
        object_masks_list: List[torch.Tensor],  # N × [B,H,W] or [H,W]
        object_bboxes_list: List[Tuple],         # N × (x0,y0,x1,y1)
        *,
        edge_radius: int = 4,
        memory_size: int = 8,
    ) -> dict:
        """Run VitMatte for N objects with per-object optimised tile placement.

        Tiles shared between non-intersecting objects are processed once
        (merged trimap); per-object alphas are separated by trimap-presence.

        Returns {"alpha", "object_alphas", "object_boxes", "info"}.
        """
        try:
            t0 = time.perf_counter()
            self.load()
            t1 = time.perf_counter()
            img = to_bhwc(image_bhwc)
            B, H, W, _ = img.shape
            N = len(object_masks_list)

            if N == 0:
                zero = torch.zeros(B, H, W)
                return {"alpha": zero, "object_alphas": [], "object_boxes": [],
                        "info": {"backend": self.KEY, "n_objects": 0}}

            per_frame: List[List[torch.Tensor]] = []
            for b in interruptible_range(B, label="vitmatte-multi"):
                frame_hwc  = (img[b].cpu().numpy() * 255).astype(np.uint8)
                frame_masks = []
                for mt in object_masks_list:
                    m = mt[b] if mt.ndim == 3 else mt
                    frame_masks.append(m.cpu().numpy().astype(np.float32))
                per_frame.append(
                    self._matte_multi_frame(
                        frame_hwc, frame_masks, object_bboxes_list, H, W, edge_radius)
                )

            object_alphas: List[torch.Tensor] = []
            for i in range(N):
                frames = [per_frame[b][i] for b in range(B)]
                object_alphas.append(torch.stack(frames, 0).clamp(0, 1))

            merged = torch.stack(object_alphas, 0).max(dim=0).values.clamp(0, 1)
            t2 = time.perf_counter()
            logger.warning(
                "[ViTMatte] matte_multi B=%d N=%d  load=%.2fs infer=%.2fs total=%.2fs",
                B, N, t1-t0, t2-t1, t2-t0)
            return {
                "alpha": merged,
                "object_alphas": object_alphas,
                "object_boxes": object_bboxes_list,
                "info": {"backend": self.KEY, "n_objects": N},
            }
        except Exception:
            free_vram()
            raise

    # ── per-frame multi-object implementation ───────────────────────────

    def _matte_multi_frame(
        self,
        frame_hwc: np.ndarray,
        object_masks: List[np.ndarray],
        object_bboxes: List[Tuple],
        H: int, W: int,
        edge_radius: int,
    ) -> List[torch.Tensor]:
        _t0_frame = time.perf_counter()
        N = len(object_masks)
        if N == 0:
            return []

        dilate = edge_radius * 2
        erode  = edge_radius
        pad    = dilate

        # 1. Per-object trimaps (uint8 0/127/255)
        trimaps: List[np.ndarray] = []
        for m_np in object_masks:
            m_t   = torch.from_numpy(m_np).unsqueeze(0)
            tri_t = mask_to_trimap(m_t, dilate=dilate, erode=erode)
            trimaps.append((tri_t[0].cpu().numpy() * 255).astype(np.uint8))

        # 2. Per-object tile positions
        tile_sets: List[List[Tuple[int, int]]] = []
        for i, bbox in enumerate(object_bboxes):
            positions = self._object_tile_positions(bbox, H, W, pad)
            tile_sets.append(positions)
            logger.debug("[ViTMatte] obj %d bbox=%s → %d tile(s)", i, bbox, len(positions))

        # 3. Tile → object-index mapping
        tile_map: dict = {}
        for i, tset in enumerate(tile_sets):
            for pos in tset:
                tile_map.setdefault(pos, []).append(i)

        global_full = (len(self._global_tile_starts_1d(W, _TILE, _STRIDE)) *
                       len(self._global_tile_starts_1d(H, _TILE, _STRIDE)))
        logger.warning(
            "[ViTMatte] multi-frame N=%d unique_tiles=%d  full-image_tiles=%d",
            N, len(tile_map), global_full,
        )

        # 4. Per-object accumulators
        alpha_acc  = np.zeros((N, H, W), dtype=np.float64)
        weight_acc = np.zeros((N, H, W), dtype=np.float64)

        for (tx, ty), obj_idx in tile_map.items():
            ty1, tx1 = min(ty + _TILE, H), min(tx + _TILE, W)
            th, tw   = ty1 - ty, tx1 - tx

            patch_img  = frame_hwc[ty:ty1, tx:tx1]
            merged_tri = np.zeros((th, tw), dtype=np.uint8)
            for i in obj_idx:
                merged_tri = np.maximum(merged_tri, trimaps[i][ty:ty1, tx:tx1])

            if th < _TILE or tw < _TILE:
                patch_img  = np.pad(patch_img,
                                    ((0, _TILE-th), (0, _TILE-tw), (0, 0)), mode="reflect")
                merged_tri = np.pad(merged_tri,
                                    ((0, _TILE-th), (0, _TILE-tw)),           mode="edge")

            inputs = self._processor(images=patch_img, trimaps=merged_tri, return_tensors="pt")
            inputs = {
                k: v.to(self.device,
                        dtype=self._dtype if v.dtype.is_floating_point else v.dtype)
                for k, v in inputs.items()
            }
            with torch.inference_mode(), \
                 torch.autocast(self.device, dtype=self._dtype,
                                enabled=(self.device == "cuda")):
                out = self._model(**inputs)

            patch_alpha = out.alphas[0, 0].float().cpu().numpy()[:th, :tw]
            w = self._blend_weight(th, tw, _MIN_OV)

            for i in obj_idx:
                alpha_acc [i, ty:ty1, tx:tx1] += patch_alpha * w
                weight_acc[i, ty:ty1, tx:tx1] += w

        # 5. Normalise and apply trimap-presence mask
        result: List[torch.Tensor] = []
        for i in range(N):
            wgt = weight_acc[i]
            with np.errstate(invalid="ignore", divide="ignore"):
                alpha = np.where(wgt > 1e-8, alpha_acc[i] / wgt, 0.0).astype(np.float32)
            alpha *= (trimaps[i] > 0).astype(np.float32)
            result.append(torch.from_numpy(alpha))
        elapsed_frame = time.perf_counter() - _t0_frame
        logger.warning(
            "[ViTMatte] _matte_multi_frame N=%d unique_tiles=%d  %.2fs",
            N, len(tile_map), elapsed_frame,
        )
        return result

    # ── tile-placement helpers ───────────────────────────────────────────

    @staticmethod
    def _global_tile_starts_1d(length: int, tile: int, stride: int) -> List[int]:
        if length <= tile:
            return [0]
        starts = list(range(0, length - tile, stride))
        last   = length - tile
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
        n     = 1 + math.ceil((length - tile) / max_stride)
        first = max(0, region_start)
        last  = min(img_length - tile, region_end - tile)
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
        global_all  = cls._global_tile_starts_1d(img_length, tile, stride)
        global_bbox = [t for t in global_all if t + tile > px0 and t < px1]
        optimal     = cls._optimal_tile_starts_1d(px0, px1, img_length, tile, min_overlap)
        return optimal if len(optimal) < len(global_bbox) else global_bbox

    @classmethod
    def _object_tile_positions(
        cls, bbox: Tuple, H: int, W: int, pad: int = _MIN_OV,
    ) -> List[Tuple[int, int]]:
        x0, y0, x1, y1 = (int(v) for v in bbox)
        px0, py0 = max(0, x0 - pad), max(0, y0 - pad)
        px1, py1 = min(W, x1 + pad), min(H, y1 + pad)
        xs = cls._best_tile_starts_1d(px0, px1, W, _TILE, _STRIDE, _MIN_OV)
        ys = cls._best_tile_starts_1d(py0, py1, H, _TILE, _STRIDE, _MIN_OV)
        return [(x, y) for y in ys for x in xs]

    # ── blend weight ─────────────────────────────────────────────────────

    @staticmethod
    def _blend_weight(h: int, w: int, overlap: int) -> np.ndarray:
        wy = np.ones(h, dtype=np.float32)
        wx = np.ones(w, dtype=np.float32)
        for i in range(min(overlap, h // 2)):
            v = 0.5 * (1.0 - np.cos(np.pi * (i + 1) / (overlap + 1)))
            wy[i] = min(wy[i], v);  wy[h-1-i] = min(wy[h-1-i], v)
        for i in range(min(overlap, w // 2)):
            v = 0.5 * (1.0 - np.cos(np.pi * (i + 1) / (overlap + 1)))
            wx[i] = min(wx[i], v);  wx[w-1-i] = min(wx[w-1-i], v)
        return np.outer(wy, wx)

    # ── self-verification ────────────────────────────────────────────────

    @classmethod
    def _verify_tile_logic(cls) -> None:
        """Smoke-test tile placement. Raises AssertionError on failure."""
        tile, min_ov, stride = _TILE, _MIN_OV, _STRIDE
        cases = [
            #  (rs,  re, img_L, exp_n, min_ov_check)
            (0,   512, 4080, 1, None),
            (0,   511, 4080, 1, None),
            (0,   960, 4080, 2, 64),
            (0,   961, 4080, 3, 64),
            (200, 1113, 4080, 2, 64),   # 913 px — key case
            (0,  1000, 4080, 3, 64),
        ]
        for rs, re, L, n_exp, ov_exp in cases:
            pos = cls._optimal_tile_starts_1d(rs, re, L, tile, min_ov)
            assert len(pos) == n_exp, \
                f"_optimal({rs},{re},{L}): expected {n_exp} got {len(pos)}: {pos}"
            assert pos[0] >= 0 and pos[-1] + tile <= L, \
                f"out of image: {pos}"
            assert pos[0] <= max(0, rs), \
                f"first tile misses region start: {pos[0]} > {rs}"
            assert pos[-1] + tile >= re, \
                f"last tile misses region end: {pos[-1]+tile} < {re}"
            if ov_exp:
                for a, b in zip(pos, pos[1:]):
                    ov = (a + tile) - b
                    assert ov >= ov_exp, \
                        f"overlap {ov} < {ov_exp} between tiles {a} and {b}"

        # _best_tile_starts_1d: 913-px region → 2 optimal vs 3 global
        best = cls._best_tile_starts_1d(200, 1113, 4080, tile, stride, min_ov)
        assert len(best) == 2, f"expected 2 best tiles for [200,1113], got {len(best)}: {best}"

        # _object_tile_positions: 913×700 bbox with pad=8
        tiles = cls._object_tile_positions((200, 100, 1113, 800), 3072, 4080, pad=8)
        # x: [192,1121] = 929 px → 2 tiles; y: [92,808] = 716 px → 2 tiles → 4
        assert len(tiles) == 4, f"expected 4 tiles for 913×700, got {len(tiles)}: {tiles}"
        logger.warning("[ViTMatte] _verify_tile_logic: all checks passed")
