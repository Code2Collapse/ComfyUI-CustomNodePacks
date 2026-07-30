"""
MaskPropagateVideo – Draw/define a mask on one frame and propagate it
across a video sequence.  Supports static copy, motion-compensated
propagation (optical flow), and SAM2 video-propagation mode.
"""

from . import _interrupt_check as _IC
from ._is_changed_util import hash_args_and_kwargs

import json
import logging
import torch
import torch.nn.functional as F
import numpy as np


from . import _progress as _PB
class MaskPropagateVideo:
    """Take a mask defined on a single frame and apply / propagate it
    to every frame in an image batch (video sequence)."""

    PROPAGATION_MODES = [
        "static",           # Same mask on every frame
        "optical_flow",     # Warp mask using dense optical flow
        "sam2_video",       # Use SAM2's video propagator
        "fade",             # Fade mask in/out over time
        "scale_linear",     # Linearly scale mask over frames
    ]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "Video frames as image batch (B, H, W, C)"}),
                "mask": ("MASK", {"tooltip": "Source mask (single frame or batch)"}),
                "source_frame": ("INT", {"default": 0, "min": 0, "max": 99999,
                                          "tooltip": "Frame index where the mask was drawn"}),
                "mode": (cls.PROPAGATION_MODES, {"default": "static", "tooltip": "Propagation method across frames"}),
                "flow_threshold": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 50.0, "step": 0.5,
                                              "tooltip": "Optical flow magnitude threshold for mask warping"}),
                "fade_start": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                                          "tooltip": "Mask opacity at source frame (for fade mode)"}),
                "fade_end": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                                        "tooltip": "Mask opacity at last frame (for fade mode)"}),
                "bidirectional": ("BOOLEAN", {"default": True,
                                               "tooltip": "Propagate both forward and backward from source frame"}),
            },
            "optional": {
                "sam_model": ("SAM_MODEL", {"tooltip": "SAM2 model for video propagation mode"}),
                "points_json": ("STRING", {"default": "", "multiline": True,
                                            "tooltip": "Point prompts for SAM2 video mode"}),
            },
        }

    RETURN_TYPES = ("MASK", "IMAGE",)
    RETURN_NAMES = ("masks", "preview",)
    OUTPUT_TOOLTIPS = (
        "Per-frame mask batch propagated across the video sequence.",
        "RGB preview with the mask overlaid on each frame.",
    )
    FUNCTION = "propagate"
    CATEGORY = "C2C/Video"
    DESCRIPTION = "Propagate a single-frame mask across all video frames using static copy, optical flow, SAM2 video, or fade modes."

    @classmethod
    def IS_CHANGED(cls, images, mask, source_frame, mode, flow_threshold,
                   fade_start, fade_end, bidirectional, sam_model=None,
                   points_json="", **kwargs):
        return hash_args_and_kwargs(
            images, mask, source_frame, mode, flow_threshold,
            fade_start, fade_end, bidirectional, sam_model, points_json, **kwargs,
        )

    def propagate(self, images, mask, source_frame, mode, flow_threshold,
                  fade_start, fade_end, bidirectional, sam_model=None, points_json=""):
        with _PB.session("MaskPropagate"):
            return self._propagate_impl(images, mask, source_frame, mode,
                                        flow_threshold, fade_start, fade_end,
                                        bidirectional, sam_model, points_json)

    def _propagate_impl(self, images, mask, source_frame, mode, flow_threshold,
                  fade_start, fade_end, bidirectional, sam_model=None, points_json=""):

        if not isinstance(images, torch.Tensor) or images.ndim != 4:
            raise ValueError("MaskPropagateVideo expects IMAGE tensor [B,H,W,C]")
        if not isinstance(mask, torch.Tensor) or mask.ndim not in (2, 3):
            raise ValueError("MaskPropagateVideo expects MASK tensor [H,W] or [B,H,W]")

        with torch.inference_mode():
            return self._propagate_core(
                images, mask, source_frame, mode, flow_threshold,
                fade_start, fade_end, bidirectional, sam_model, points_json,
            )

    def _propagate_core(self, images, mask, source_frame, mode, flow_threshold,
                  fade_start, fade_end, bidirectional, sam_model=None, points_json=""):

        B, H, W, C = images.shape
        source_frame = min(source_frame, B - 1)

        # Ensure mask is 2D (single frame)
        src_mask = mask
        if src_mask.dim() == 3:
            idx = min(source_frame, src_mask.shape[0] - 1)
            src_mask = src_mask[idx]
        while src_mask.dim() > 2:
            src_mask = src_mask.squeeze(0)

        # Resize mask if dimensions don't match
        if src_mask.shape[0] != H or src_mask.shape[1] != W:
            src_mask = F.interpolate(
                src_mask.unsqueeze(0).unsqueeze(0),
                size=(H, W), mode="bilinear", align_corners=False
            ).squeeze(0).squeeze(0)

        # ── Dispatch by mode ───────────────────────────────────────────
        if mode == "static":
            out_masks = self._static(src_mask, B)
        elif mode == "optical_flow":
            out_masks = self._optical_flow(images, src_mask, source_frame,
                                           flow_threshold, bidirectional)
        elif mode == "sam2_video":
            out_masks = self._sam2_video(images, src_mask, source_frame,
                                         sam_model, points_json)
        elif mode == "fade":
            out_masks = self._fade(src_mask, B, source_frame, fade_start, fade_end)
        elif mode == "scale_linear":
            out_masks = self._scale_linear(src_mask, B, source_frame, fade_start, fade_end)
        else:
            out_masks = self._static(src_mask, B)

        # Build preview: overlay mask on images
        preview = self._overlay_preview(images, out_masks)

        return (out_masks, preview)

    # ── Mode implementations ─────────────────────────────────────────

    @staticmethod
    def _static(mask, num_frames):
        """Copy the same mask to every frame."""
        return mask.unsqueeze(0).expand(num_frames, -1, -1).clone()

    @staticmethod
    def _fade(mask, num_frames, source_frame, start_opacity, end_opacity):
        """Linearly fade opacity across frames."""
        masks = mask.unsqueeze(0).expand(num_frames, -1, -1).clone()
        for i in _PB.track(range(num_frames), num_frames, "MaskPropagate"):
            _IC.check()
            if num_frames == 1:
                alpha = start_opacity
            else:
                t = abs(i - source_frame) / max(1, num_frames - 1)
                alpha = start_opacity + (end_opacity - start_opacity) * t
            masks[i] = masks[i] * alpha
        return masks

    @staticmethod
    def _scale_linear(mask, num_frames, source_frame, start_scale, end_scale):
        """Linearly scale mask spatially across frames (zoom in/out effect)."""
        masks = torch.zeros(num_frames, mask.shape[0], mask.shape[1],
                            dtype=mask.dtype, device=mask.device)
        H, W = mask.shape
        for i in _PB.track(range(num_frames), num_frames, "MaskPropagate"):
            _IC.check()
            if num_frames == 1:
                scale = start_scale
            else:
                t = abs(i - source_frame) / max(1, num_frames - 1)
                scale = start_scale + (end_scale - start_scale) * t
            if scale <= 0:
                continue
            new_h = max(1, int(H * scale))
            new_w = max(1, int(W * scale))
            scaled = F.interpolate(
                mask.unsqueeze(0).unsqueeze(0),
                size=(new_h, new_w), mode="bilinear", align_corners=False
            ).squeeze(0).squeeze(0)
            # Center-paste
            y_off = (H - new_h) // 2
            x_off = (W - new_w) // 2
            src_y0 = max(0, -y_off)
            src_x0 = max(0, -x_off)
            dst_y0 = max(0, y_off)
            dst_x0 = max(0, x_off)
            copy_h = min(new_h - src_y0, H - dst_y0)
            copy_w = min(new_w - src_x0, W - dst_x0)
            if copy_h > 0 and copy_w > 0:
                masks[i, dst_y0:dst_y0+copy_h, dst_x0:dst_x0+copy_w] = \
                    scaled[src_y0:src_y0+copy_h, src_x0:src_x0+copy_w]
        return masks

    def _optical_flow(self, images, src_mask, source_frame, threshold, bidirectional):
        """Warp mask using Farneback optical flow between consecutive frames."""
        B, H, W, C = images.shape
        masks = torch.zeros(B, H, W, dtype=src_mask.dtype, device=src_mask.device)
        masks[source_frame] = src_mask

        try:
            import cv2
        except ImportError:
            # Fallback to static if cv2 not available
            return self._static(src_mask, B)

        imgs_gray = []
        for i in _PB.track(range(B), B, "MaskPropagate"):
            _IC.check()
            frame = (images[i].cpu().numpy() * 255).astype(np.uint8)
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            imgs_gray.append(gray)

        # Forward propagation
        current_mask = src_mask.cpu().numpy()
        for i in _PB.track(range(source_frame + 1, B), None, "MaskPropagate"):
            flow = cv2.calcOpticalFlowFarneback(
                imgs_gray[i-1], imgs_gray[i],
                None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            h, w = flow.shape[:2]
            flow_map = np.zeros_like(flow)
            flow_map[:, :, 0] = np.arange(w) + flow[:, :, 0]
            flow_map[:, :, 1] = np.arange(h).reshape(-1, 1) + flow[:, :, 1]
            warped = cv2.remap(current_mask, flow_map[:, :, 0].astype(np.float32),
                               flow_map[:, :, 1].astype(np.float32),
                               cv2.INTER_LINEAR, borderValue=0)
            # Apply threshold
            if threshold > 0:
                mag = np.sqrt(flow[:, :, 0]**2 + flow[:, :, 1]**2)
                warped[mag < threshold * 0.1] = current_mask[mag < threshold * 0.1]
            current_mask = warped
            masks[i] = torch.from_numpy(current_mask).to(src_mask.device)

        # Backward propagation
        if bidirectional and source_frame > 0:
            current_mask = src_mask.cpu().numpy()
            for i in _PB.track(range(source_frame - 1, -1, -1), None, "MaskPropagate"):
                flow = cv2.calcOpticalFlowFarneback(
                    imgs_gray[i+1], imgs_gray[i],
                    None, 0.5, 3, 15, 3, 5, 1.2, 0
                )
                h, w = flow.shape[:2]
                flow_map = np.zeros_like(flow)
                flow_map[:, :, 0] = np.arange(w) + flow[:, :, 0]
                flow_map[:, :, 1] = np.arange(h).reshape(-1, 1) + flow[:, :, 1]
                warped = cv2.remap(current_mask, flow_map[:, :, 0].astype(np.float32),
                                   flow_map[:, :, 1].astype(np.float32),
                                   cv2.INTER_LINEAR, borderValue=0)
                current_mask = warped
                masks[i] = torch.from_numpy(current_mask).to(src_mask.device)

        return masks

    def _sam2_video(self, images, src_mask, source_frame, sam_model, points_json):
        """Propagate a seed mask across the clip with SAM2's video memory.

        This is the path that survives motion blur, occlusion and a subject
        flipping upside-down: SAM2 carries a memory of the object across
        frames instead of re-detecting it per frame, so a few unreadable
        frames do not break the track.

        It had never actually run. Four defects, all fixed here:

        1. init_state was called as init_state(video_path=tmp). This fork's
           signature is init_state(images, video_height, video_width, ...) and
           takes the frame TENSOR directly, so every call raised TypeError,
           was swallowed by a bare `except (ImportError, Exception)` and fell
           back to optical flow. SAM2 was never involved in any result.
        2. The frames were being written out as JPEG quality=95 into a temp
           directory first. Lossy compression, on every frame, with the
           artefacts concentrated exactly on the edges we are trying to cut.
           Gone - the tensor goes straight in, which is also much faster.
        3. The seed was reduced to the mask's BOUNDING BOX. That throws away
           the shape and hands SAM2 a rectangle full of background to guess
           from; on a subject mid-flip the box is nearly the whole frame.
           add_new_mask() takes the real mask, so use it.
        4. Propagation ran forward only, so with a seed anywhere but frame 0
           every earlier frame came back empty. Now it runs both directions.

        Edges come back as a soft alpha from the logits rather than a hard
        `> 0` threshold, which is what makes the boundary sub-pixel instead of
        stair-stepped.
        """
        B, H, W, C = images.shape
        _log = logging.getLogger(__name__)

        if sam_model is None:
            raise ValueError(
                "mode='sam2_video' needs a SAM2 model on the sam_model input. "
                "Connect SAMModelLoaderMEC with a sam2.1 checkpoint, or pick a "
                "different mode."
            )
        model = sam_model["model"]
        if sam_model.get("model_type") != "sam2.1":
            raise ValueError(
                f"mode='sam2_video' needs a sam2.1 model; got "
                f"{sam_model.get('model_type')!r}. Video propagation is a SAM2 "
                f"feature - SAM1 checkpoints cannot track across frames."
            )

        from sam2.sam2_video_predictor import SAM2VideoPredictor
        from comfy.utils import common_upscale

        if not isinstance(model, SAM2VideoPredictor):
            model.__class__ = SAM2VideoPredictor
            model.fill_hole_area = 8
            model.non_overlap_masks = False
            model.clear_non_cond_mem_around_input = False
            model.add_all_frames_to_correct_as_cond = False
        predictor = model

        device = next(predictor.parameters()).device
        sz = predictor.image_size
        # Same preparation the SAM2 pack's own video node uses: square-resize
        # to the model's input size, keep the ORIGINAL H/W for output scaling.
        frames = common_upscale(
            images.movedim(-1, 1), sz, sz, "bilinear", "disabled",
        ).contiguous().to(device)

        if getattr(self, "_sam2_state", None) is not None:
            try:
                predictor.reset_state(self._sam2_state)
            except Exception:      # noqa: BLE001 - a stale state must not block a new run
                pass
        state = predictor.init_state(frames, H, W, device=device)
        self._sam2_state = state

        seed = int(max(0, min(B - 1, source_frame)))
        m = src_mask
        if m.dim() == 3:
            m = m[0] if m.shape[0] == 1 else m[seed] if m.shape[0] > seed else m[0]
        m = m.to(torch.float32)
        if m.shape != (H, W):
            m = F.interpolate(m[None, None], size=(H, W), mode="bilinear",
                              align_corners=False)[0, 0]
        if float(m.max()) <= 0.0:
            raise ValueError(
                f"the seed mask on frame {seed} is empty, so there is nothing "
                f"for SAM2 to track. Check source_frame points at a frame where "
                f"the subject is actually masked."
            )
        predictor.add_new_mask(inference_state=state, frame_idx=seed,
                               obj_id=1, mask=(m > 0.5))

        # Extra clicks stay supported and are additive to the mask seed.
        if points_json and points_json.strip():
            try:
                pts = json.loads(points_json)
                if pts:
                    predictor.add_new_points_or_box(
                        inference_state=state, frame_idx=seed, obj_id=1,
                        points=np.array([[p["x"], p["y"]] for p in pts], np.float32),
                        labels=np.array([p.get("label", 1) for p in pts], np.int32),
                    )
            except Exception as exc:      # noqa: BLE001 - bad JSON must not kill the track
                _log.warning("MaskPropagateVideo: ignoring points_json (%s)", exc)

        masks = torch.zeros(B, H, W, dtype=torch.float32)
        covered = torch.zeros(B, dtype=torch.bool)
        # Both directions: forward from the seed, then backward, so a seed in
        # the middle of the clip fills the whole clip instead of half of it.
        for reverse in (False, True):
            for f_idx, _obj_ids, logits in predictor.propagate_in_video(
                state, start_frame_idx=seed, reverse=reverse,
            ):
                _IC.check()
                if logits.shape[0] == 0:
                    continue
                # Soft alpha, not a hard threshold: the logit carries
                # sub-pixel boundary information and binarising here is what
                # makes SAM output look stair-stepped.
                a = torch.sigmoid(logits[0, 0].float()).cpu()
                if a.shape != (H, W):
                    a = F.interpolate(a[None, None], size=(H, W), mode="bilinear",
                                      align_corners=False)[0, 0]
                masks[f_idx] = a.clamp(0, 1)
                covered[f_idx] = True

        n_cov = int(covered.sum())
        if n_cov < B:
            _log.warning(
                "MaskPropagateVideo: SAM2 returned no mask on %d/%d frames "
                "(seed frame %d). Those frames are left empty rather than "
                "filled with a stale copy - add a click on one of them if the "
                "subject is present.", B - n_cov, B, seed,
            )
        else:
            _log.info("MaskPropagateVideo: SAM2 tracked all %d frames from seed "
                      "frame %d (bidirectional, soft alpha).", B, seed)
        return masks


    @staticmethod
    def _overlay_preview(images, masks):
        """Create a preview with green mask overlay on images."""
        B, H, W, C = images.shape
        preview = images.clone()
        color = torch.tensor([0.0, 1.0, 0.0], device=images.device)  # green
        alpha = 0.35
        for i in _PB.track(range(B), B, "MaskPropagate"):
            _IC.check()
            m = masks[i]
            if m.shape[0] != H or m.shape[1] != W:
                m = F.interpolate(
                    m.unsqueeze(0).unsqueeze(0),
                    size=(H, W), mode="bilinear", align_corners=False
                ).squeeze()
            mask_3d = m.unsqueeze(-1).expand(-1, -1, 3)
            overlay = color.unsqueeze(0).unsqueeze(0).expand(H, W, 3)
            preview[i] = preview[i] * (1 - mask_3d * alpha) + overlay * mask_3d * alpha
        return preview.clamp(0, 1)
