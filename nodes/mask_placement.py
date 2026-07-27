"""MaskPlacementMEC — single-node universal mask placement (+ tracking, staged).

The user's tool for "give a ref or a prompt, get that thing's alpha, drag it
onto a frame, and it follows the video": one node that
  1. sources an alpha mask (BYO mask socket, or text prompt via the SAME
     GroundingDINO+SAM pipeline SAMMaskGeneratorMEC runs — delegated, not
     duplicated),
  2. places it with a 4-corner perspective quad (position/scale/rotate/warp)
     authored in the companion editor widget (js/mask_placement_editor.js),
  3. propagates the placement across all frames.

Build slices (design doc: MASK_PLACEMENT_NODE_DESIGN.md at workspace root):
  Slice 1 (THIS): static propagation — same homography every frame. Fully
      usable for stills and locked-off shots. track_mode object_track /
      landmark_lock fall back to static with a clear note in `info`.
  Slice 2: Cutie (MIT, third_party/Cutie -> nodes_extras port) object
      tracking driving a per-frame quad update.
  Slice 3: landmark-locked barycentric binding for face/body attachments.

Apache-2.0, C2C/MEC.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

log = logging.getLogger("C2C.MaskPlacement")

try:
    import cv2
    _CV2_OK = True
except Exception:  # noqa: BLE001
    cv2 = None  # type: ignore
    _CV2_OK = False


def _to_np_image(t) -> np.ndarray:
    """(B,H,W,3) float 0..1 numpy from an IMAGE tensor/array."""
    if t is None:
        raise ValueError("image input is None")
    arr = t.detach().cpu().numpy() if hasattr(t, "detach") else np.asarray(t)
    if arr.ndim == 3:
        arr = arr[None]
    if arr.ndim != 4 or arr.shape[-1] != 3:
        raise ValueError(f"expected (B,H,W,3) image, got {arr.shape}")
    return np.clip(arr.astype(np.float32), 0.0, 1.0)


def _to_np_mask(m) -> np.ndarray:
    """(B,H,W) float 0..1 numpy from a MASK tensor/array."""
    arr = m.detach().cpu().numpy() if hasattr(m, "detach") else np.asarray(m)
    if arr.ndim == 2:
        arr = arr[None]
    if arr.ndim != 3:
        raise ValueError(f"expected (B,H,W) mask, got {arr.shape}")
    return np.clip(arr.astype(np.float32), 0.0, 1.0)


def _parse_placement(blob: str, W: int, H: int) -> Tuple[np.ndarray, float]:
    """placement_json -> (4,2) corner array (TL,TR,BR,BL pixel coords) + feather px.

    Empty/invalid json -> a centered quad at 40% of the frame so the node
    produces a sensible result before the editor has ever written anything.
    """
    corners = None
    feather = 6.0
    if blob and blob.strip():
        try:
            data = json.loads(blob)
            if isinstance(data, dict):
                c = data.get("corners")
                if (isinstance(c, list) and len(c) == 4
                        and all(isinstance(p, (list, tuple)) and len(p) >= 2 for p in c)):
                    corners = np.asarray([[float(p[0]), float(p[1])] for p in c],
                                         dtype=np.float32)
                feather = float(data.get("feather", feather))
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            log.warning("placement_json ignored — %s", exc)
    if corners is None:
        w4, h4 = W * 0.30, H * 0.30
        cx, cy = W * 0.5, H * 0.5
        corners = np.asarray([
            [cx - w4, cy - h4], [cx + w4, cy - h4],
            [cx + w4, cy + h4], [cx - w4, cy + h4],
        ], dtype=np.float32)
    return corners, max(0.0, feather)


def _tight_bbox(alpha: np.ndarray, thresh: float = 0.02) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(alpha > thresh)
    if len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


class MaskPlacementMEC:
    DESCRIPTION = (
        "Universal mask placement: source an alpha (wire a mask, or give a text "
        "prompt + SAM model and it segments the object for you), place it with a "
        "draggable 4-corner perspective quad, and propagate across the video. "
        "Slice 1 ships static propagation; object tracking (Cutie) and "
        "landmark-locked follow are staged next."
    )

    @classmethod
    def INPUT_TYPES(cls):
        gdino_models = ["none"]
        try:
            from .sam_mask_generator import MODEL_REGISTRY  # type: ignore
            for name, reg in MODEL_REGISTRY.items():
                if reg.get("family") == "groundingdino":
                    gdino_models.append(name)
        except Exception:  # noqa: BLE001 — SAM stack optional for BYO-mask use
            pass
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "Video frames or a single still (B,H,W,3)."}),
                "prompt": ("STRING", {"default": "", "multiline": False,
                    "tooltip": "What to cut out, e.g. 'dog', 'car', 'mouth'. Needs sam_model "
                               "(+ a GroundingDINO model) wired. Ignored when source_mask is wired."}),
                "grounding_model": (gdino_models, {"default": gdino_models[-1] if len(gdino_models) > 1 else "none",
                    "tooltip": "GroundingDINO model for text->box grounding (same list as SAM Mask Generator)."}),
                "track_mode": (["static", "object_track", "landmark_lock", "auto"], {"default": "static",
                    "tooltip": "How the placement follows the video. Slice 1: 'static' is live; "
                               "'object_track' (Cutie) and 'landmark_lock' fall back to static for now "
                               "and say so in `info`."}),
                "anchor_frame": ("INT", {"default": 0, "min": 0, "max": 99999,
                    "tooltip": "The frame the quad was placed on (and segmented from, when using prompt "
                               "without ref_image)."}),
                "placement_json": ("STRING", {"default": "", "multiline": True,
                    "tooltip": "Editor-owned: {\"corners\":[[x,y]x4 TL,TR,BR,BL], \"feather\":px} in "
                               "anchor-frame pixel coords. Empty = centered 60%-size quad."}),
                "feather_px": ("INT", {"default": 6, "min": 0, "max": 128,
                    "tooltip": "Edge feather (Gaussian, px) applied to the placed alpha. "
                               "placement_json's feather wins when present."}),
            },
            "optional": {
                "source_mask": ("MASK", {"tooltip": "BYO alpha (e.g. SAM Mask Generator / matting output). "
                                                     "Skips prompt segmentation entirely."}),
                "source_image": ("IMAGE", {"tooltip": "RGB that source_mask cuts out of (for the placed_rgb "
                                                       "output). When absent, the mask is placed without pixels."}),
                "ref_image": ("IMAGE", {"tooltip": "Segment the prompt from THIS image instead of the video's "
                                                    "anchor frame (e.g. a photo of the dog you want)."}),
                "sam_model": ("SAM_MODEL", {"tooltip": "From SAM Model Loader — needed only for prompt-based "
                                                        "segmentation."}),
            },
        }

    RETURN_TYPES = ("MASK", "IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("masks", "placed_rgb", "overlay_preview", "info")
    OUTPUT_TOOLTIPS = (
        "Per-frame placed alpha (B,H,W float 0..1). Feed to inpaint/composite.",
        "Per-frame RGB of the placed source pixels over black (use with `masks` to composite).",
        "Frames with the placement tinted + quad drawn — visual verification.",
        "JSON: source mode, quad, per-frame status, and which tracking actually ran.",
    )
    FUNCTION = "execute"
    CATEGORY = "MEC/Masking"

    # ── mask sourcing ────────────────────────────────────────────────
    def _source_alpha(self, images_np, prompt, grounding_model, anchor_frame,
                      source_mask, source_image, ref_image, sam_model):
        """Returns (alpha HxW float, rgb HxWx3 float or None, mode string)."""
        if source_mask is not None:
            alpha = _to_np_mask(source_mask)[0]
            rgb = None
            if source_image is not None:
                si = _to_np_image(source_image)[0]
                if si.shape[:2] == alpha.shape:
                    rgb = si
                else:
                    log.warning("source_image size %s != source_mask %s — placing mask only.",
                                si.shape[:2], alpha.shape)
            return alpha, rgb, "byo_mask"

        if prompt and prompt.strip():
            if sam_model is None:
                raise ValueError(
                    "MaskPlacementMEC: a text prompt needs the 'sam_model' input wired "
                    "(SAM Model Loader). Or wire 'source_mask' directly."
                )
            from .sam_mask_generator import SAMMaskGeneratorMEC  # delegated, not duplicated
            if ref_image is not None:
                seg_src = _to_np_image(ref_image)[0]
                seg_tensor = torch.from_numpy(seg_src[None])
            else:
                idx = min(int(anchor_frame), images_np.shape[0] - 1)
                seg_src = images_np[idx]
                seg_tensor = torch.from_numpy(seg_src[None])
            gen = SAMMaskGeneratorMEC()
            result = gen.generate(
                sam_model=sam_model, image=seg_tensor,
                points_json="[]", bbox_json="",
                text_prompt=str(prompt), negative_text_prompt="",
                grounding_model=str(grounding_model),
                text_threshold=0.25, text_box_threshold=0.3,
                multimask_output=True, mask_index=0, score_threshold=0.0,
                apply_bbox_crop=False, refine_iterations=2,
                auto_negative_points=False, edge_refine="guided", edge_radius=8,
            )
            alpha = _to_np_mask(result[0])[0]
            return alpha, seg_src, "prompt_segmentation"

        raise ValueError(
            "MaskPlacementMEC has no mask source: wire 'source_mask', or give a "
            "text 'prompt' + 'sam_model' (+ optionally 'ref_image')."
        )

    # ── main ─────────────────────────────────────────────────────────
    def execute(self, images, prompt, grounding_model, track_mode, anchor_frame,
                placement_json, feather_px,
                source_mask=None, source_image=None, ref_image=None, sam_model=None):
        if not _CV2_OK:
            raise RuntimeError("MaskPlacementMEC needs opencv-python (cv2) installed.")
        with torch.inference_mode():
            return self._execute_impl(
                images, prompt, grounding_model, track_mode, anchor_frame,
                placement_json, feather_px, source_mask, source_image,
                ref_image, sam_model,
            )

    def _execute_impl(self, images, prompt, grounding_model, track_mode, anchor_frame,
                      placement_json, feather_px,
                      source_mask, source_image, ref_image, sam_model):
        images_np = _to_np_image(images)
        B, H, W = images_np.shape[0], images_np.shape[1], images_np.shape[2]

        alpha_src, rgb_src, source_mode = self._source_alpha(
            images_np, prompt, grounding_model, anchor_frame,
            source_mask, source_image, ref_image, sam_model,
        )

        # Tight-crop the source so the quad maps the OBJECT, not dead space.
        bb = _tight_bbox(alpha_src)
        if bb is None:
            raise ValueError(
                "MaskPlacementMEC: the source mask is empty — nothing to place. "
                "Check the prompt/threshold or the wired mask."
            )
        sx1, sy1, sx2, sy2 = bb
        alpha_crop = alpha_src[sy1:sy2, sx1:sx2]
        rgb_crop = rgb_src[sy1:sy2, sx1:sx2] if rgb_src is not None else None
        sh, sw = alpha_crop.shape

        corners, feather = _parse_placement(placement_json, W, H)
        if feather <= 0:
            feather = float(feather_px)

        # Homography: source rect -> user quad.
        src_rect = np.asarray([[0, 0], [sw, 0], [sw, sh], [0, sh]], dtype=np.float32)
        Hm = cv2.getPerspectiveTransform(src_rect, corners.astype(np.float32))

        warped_alpha = cv2.warpPerspective(
            alpha_crop.astype(np.float32), Hm, (W, H),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0,
        )
        if feather > 0.5:
            k = max(1, int(round(feather)) * 2 + 1)
            warped_alpha = cv2.GaussianBlur(warped_alpha, (k, k), feather * 0.5)
        warped_alpha = np.clip(warped_alpha, 0.0, 1.0)

        if rgb_crop is not None:
            warped_rgb = cv2.warpPerspective(
                rgb_crop.astype(np.float32), Hm, (W, H),
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0,
            )
        else:
            warped_rgb = np.zeros((H, W, 3), np.float32)

        requested = str(track_mode)
        effective = "static"
        note = ""
        if requested in ("object_track", "landmark_lock", "auto") and B > 1:
            note = (f"track_mode='{requested}' is not built yet (Slice 2/3) — "
                    "placement propagated statically to all frames.")
            log.info("MaskPlacementMEC: %s", note)

        # Slice 1: same placement on every frame.
        masks = np.repeat(warped_alpha[None], B, axis=0)
        placed = np.repeat(warped_rgb[None], B, axis=0)

        # Overlay preview: original frames + green tint + quad outline.
        overlay = images_np.copy()
        tint = np.zeros_like(overlay[0])
        tint[..., 1] = 1.0
        a3 = warped_alpha[..., None] * 0.45
        quad_i = corners.astype(np.int32).reshape(-1, 1, 2)
        for i in range(B):
            overlay[i] = overlay[i] * (1.0 - a3) + tint * a3
            frame_u8 = (overlay[i] * 255).astype(np.uint8)
            cv2.polylines(frame_u8, [quad_i], True, (60, 180, 255), 2, cv2.LINE_AA)
            for cx, cy in corners:
                cv2.circle(frame_u8, (int(cx), int(cy)), 5, (60, 180, 255), -1, cv2.LINE_AA)
            overlay[i] = frame_u8.astype(np.float32) / 255.0

        info = json.dumps({
            "source_mode": source_mode,
            "source_bbox": [sx1, sy1, sx2, sy2],
            "corners": corners.tolist(),
            "feather_px": feather,
            "frames": int(B),
            "track_mode_requested": requested,
            "track_mode_effective": effective,
            "note": note,
        })

        # Anchor-frame preview for the editor backdrop (b64, downscaled).
        ui_payload: Dict[str, Any] = {}
        try:
            import base64
            idx = min(int(anchor_frame), B - 1)
            frame_u8 = (images_np[idx] * 255).astype(np.uint8)
            scale = min(1.0, 640.0 / max(H, W))
            if scale < 1.0:
                frame_u8 = cv2.resize(frame_u8, (int(W * scale), int(H * scale)),
                                      interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode(".jpg", frame_u8[:, :, ::-1],
                                   [int(cv2.IMWRITE_JPEG_QUALITY), 88])
            if ok:
                ui_payload = {
                    "mp_preview": ["data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")],
                    "mp_size": [[int(W), int(H)]],
                    "mp_anchor": [int(idx)],
                }
        except Exception as exc:  # noqa: BLE001 — preview is best-effort
            log.debug("editor preview skipped: %s", exc)

        return {
            "ui": ui_payload,
            "result": (
                torch.from_numpy(masks).float(),
                torch.from_numpy(placed).float(),
                torch.from_numpy(overlay).float(),
                info,
            ),
        }


NODE_CLASS_MAPPINGS = {"MaskPlacementMEC": MaskPlacementMEC}
NODE_DISPLAY_NAME_MAPPINGS = {"MaskPlacementMEC": "Mask Placement — Prompt/Ref → Place → Track"}
