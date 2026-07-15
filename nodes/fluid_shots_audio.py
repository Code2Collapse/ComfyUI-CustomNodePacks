"""Fluid Shots & Audio FX — temporal normalizer pair + audio reverser.

FluidShotEncoderMEC  — measures optical-flow motion per frame pair (Farneback)
    and inserts flow-warped in-between frames wherever pixel displacement
    exceeds `max_px_per_frame`, producing a constant-speed video that AI video
    models (Wan 2.1/2.2) can track without face-morphing. Emits a TIMEMAP.
FluidShotDecoderMEC  — selects exactly the frames of the AI output that
    correspond to the ORIGINAL input frames (index-list retiming, the same
    `images[indexes]` pattern as VHS issue #389's Rerate snippet), restoring
    the original frame count and speed-ramp pacing.
AudioReverserMEC     — reverses AUDIO ({"waveform": [B,C,S], "sample_rate"})
    via torch.flip on the sample dim; emits a TIMEMAP with the timestamp map.

TIMEMAP is a plain dict passed natively between nodes (custom type string).

Design notes (research-grounded):
  * Per-pair insert counts mirror ComfyUI-Frame-Interpolation's list-multiplier
    scheduling (generic_frame_loop pads/uses one multiplier per frame pair).
  * Interpolation is flow-warp + cross-fade (cv2.remap along ±t·flow); falls
    back to linear blend, then frame duplication, if OpenCV/flow fails.
  * AUDIO dict contract verified against comfy_extras/nodes_audio.py and VHS.
"""

import math
import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)

_MAX_INSERTS_PER_PAIR = 16   # safety cap so one cut can't explode the batch
_FLOW_PROC_LONG_SIDE = 384   # downscale for Farneback speed; magnitudes rescaled


def _to_gray_u8(frame_t, scale):
    """IMAGE frame [H,W,C] float 0-1 -> downscaled gray uint8 numpy."""
    import cv2
    a = (frame_t.detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    if scale < 1.0:
        a = cv2.resize(a, (max(8, int(a.shape[1] * scale)), max(8, int(a.shape[0] * scale))),
                       interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)


def _flow_and_motion(g0, g1, rescale):
    """Farneback flow between gray frames -> (flow [h,w,2], motion_px @ full res).

    Motion metric = 95th-percentile flow magnitude (robust to static borders),
    rescaled back to full-resolution pixels.
    """
    import cv2
    flow = cv2.calcOpticalFlowFarneback(g0, g1, None,
                                        0.5, 3, 21, 3, 5, 1.2, 0)
    mag = np.linalg.norm(flow, axis=2)
    motion = float(np.percentile(mag, 95.0)) * rescale
    return flow, motion


def _warp_mid(a_u8, b_u8, flow, t):
    """Flow-warp intermediate at fraction t between full-res RGB uint8 frames.

    Backward-warp approximation: A sampled at p - t*F, B at p + (1-t)*F,
    cross-faded. flow is at processing scale; rescaled to full res here.
    """
    import cv2
    h, w = a_u8.shape[:2]
    fh, fw = flow.shape[:2]
    fx = cv2.resize(flow[..., 0], (w, h), interpolation=cv2.INTER_LINEAR) * (w / fw)
    fy = cv2.resize(flow[..., 1], (w, h), interpolation=cv2.INTER_LINEAR) * (h / fh)
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    a_w = cv2.remap(a_u8, gx - t * fx, gy - t * fy, cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REPLICATE)
    b_w = cv2.remap(b_u8, gx + (1.0 - t) * fx, gy + (1.0 - t) * fy, cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REPLICATE)
    return ((1.0 - t) * a_w.astype(np.float32) + t * b_w.astype(np.float32))


class FluidShotEncoderMEC:
    """Normalize speed ramps: insert flow-warped frames where motion is fast."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "Video frames [B,H,W,C] 0-1. Speed-ramped source."}),
                "max_px_per_frame": ("FLOAT", {
                    "default": 5.0, "min": 0.5, "max": 100.0, "step": 0.5,
                    "tooltip": "Max allowed pixel displacement between consecutive output frames "
                               "(95th-percentile optical-flow magnitude). Fast sections get "
                               "flow-warped in-betweens until each step is under this."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "TIMEMAP")
    RETURN_NAMES = ("normalized_images", "time_map")
    FUNCTION = "execute"
    CATEGORY = "MEC/Temporal"
    DESCRIPTION = ("Analyzes optical-flow speed and inserts in-between frames on fast "
                   "sections so motion per frame is constant — safe for Wan-style AI video. "
                   "Pair with Fluid Shot Decoder to restore the original timing exactly.")

    def execute(self, images, max_px_per_frame):
        if images.ndim == 3:
            images = images.unsqueeze(0)
        B = images.shape[0]
        if B < 2:
            tm = {"kind": "fluid_shot", "src_count": int(B), "out_count": int(B),
                  "original_positions": list(range(int(B))),
                  "src_of_output": [float(i) for i in range(int(B))]}
            return (images, tm)

        H, W = int(images.shape[1]), int(images.shape[2])
        scale = min(1.0, _FLOW_PROC_LONG_SIDE / float(max(H, W)))
        rescale = 1.0 / scale if scale > 0 else 1.0
        max_px = max(0.5, float(max_px_per_frame))

        out_frames = [images[0]]
        original_positions = [0]
        src_of_output = [0.0]

        prev_gray = None
        for i in range(B - 1):
            a_t, b_t = images[i], images[i + 1]
            n_insert, flow = 0, None
            try:
                g0 = prev_gray if prev_gray is not None else _to_gray_u8(a_t, scale)
                g1 = _to_gray_u8(b_t, scale)
                prev_gray = g1
                flow, motion = _flow_and_motion(g0, g1, rescale)
                if motion > max_px:
                    n_insert = min(_MAX_INSERTS_PER_PAIR,
                                   int(math.ceil(motion / max_px)) - 1)
            except Exception as exc:  # noqa: BLE001 — flow failure = no inserts
                logger.warning("FluidShotEncoder: flow failed on pair %d (%s); "
                               "passing through.", i, exc)

            if n_insert > 0:
                a_u8 = (a_t.detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
                b_u8 = (b_t.detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
                for k in range(1, n_insert + 1):
                    t = k / float(n_insert + 1)
                    try:
                        mid = _warp_mid(a_u8, b_u8, flow, t) / 255.0
                        mid_t = torch.from_numpy(mid.astype(np.float32))
                    except Exception:  # noqa: BLE001 — warp failed: linear blend
                        try:
                            mid_t = (1.0 - t) * a_t + t * b_t
                        except Exception:  # noqa: BLE001 — last resort: duplicate
                            mid_t = a_t
                    out_frames.append(mid_t.to(images.dtype))
                    src_of_output.append(float(i) + t)

            out_frames.append(b_t)
            original_positions.append(len(out_frames) - 1)
            src_of_output.append(float(i + 1))

        normalized = torch.stack(out_frames, dim=0)
        tm = {"kind": "fluid_shot", "src_count": int(B),
              "out_count": int(normalized.shape[0]),
              "original_positions": original_positions,
              "src_of_output": src_of_output}
        logger.info("FluidShotEncoder: %d -> %d frames (max %.1fpx/frame).",
                    B, normalized.shape[0], max_px)
        return (normalized, tm)


class FluidShotDecoderMEC:
    """Restore the original frame count + speed ramp from a normalized batch."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "AI-processed frames (the expanded, normalized batch)."}),
                "time_map": ("TIMEMAP", {"tooltip": "From Fluid Shot Encoder."}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "execute"
    CATEGORY = "MEC/Temporal"
    DESCRIPTION = ("Selects exactly the frames that correspond to the ORIGINAL input "
                   "frames using the encoder's TIMEMAP — restores the original frame "
                   "count and slow-to-fast pacing.")

    def execute(self, images, time_map):
        if images.ndim == 3:
            images = images.unsqueeze(0)
        if not (isinstance(time_map, dict) and time_map.get("kind") == "fluid_shot"):
            raise ValueError("FluidShotDecoder: time_map is not a Fluid Shot TIMEMAP "
                             "(wire it from Fluid Shot Encoder).")
        B = int(images.shape[0])
        positions = [int(p) for p in time_map.get("original_positions", [])]
        if not positions:
            return (images,)
        expected = int(time_map.get("out_count", 0))
        if expected and B != expected:
            # AI models may trim/pad the batch (e.g. Wan's 4n+1). Scale the map
            # proportionally, then clamp — degrades gracefully instead of erroring.
            logger.warning("FluidShotDecoder: batch is %d frames but TIMEMAP expects %d; "
                           "rescaling map.", B, expected)
            ratio = (B - 1) / float(max(1, expected - 1))
            positions = [int(round(p * ratio)) for p in positions]
        idx = torch.tensor([min(max(p, 0), B - 1) for p in positions],
                           dtype=torch.long, device=images.device)
        return (images.index_select(0, idx),)


class AudioReverserMEC:
    """Reverse an AUDIO waveform; emit a timestamp TIMEMAP for video syncing."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {"tooltip": "Standard ComfyUI audio "
                                    "({'waveform': [B,C,S], 'sample_rate': int})."}),
            },
        }

    RETURN_TYPES = ("AUDIO", "TIMEMAP")
    RETURN_NAMES = ("audio", "time_map")
    FUNCTION = "execute"
    CATEGORY = "MEC/Audio"
    DESCRIPTION = ("Reverses the audio waveform (torch.flip on the sample dim), "
                   "preserving sample rate and channels. TIMEMAP maps original to "
                   "reversed timestamps (t_new = duration - t_orig).")

    def execute(self, audio):
        if not (isinstance(audio, dict) and "waveform" in audio):
            raise ValueError("AudioReverser: expected AUDIO dict with a 'waveform' tensor.")
        w = audio["waveform"]
        sr = int(audio.get("sample_rate", 44100))
        if not torch.is_tensor(w):
            raise ValueError("AudioReverser: 'waveform' is not a tensor.")
        rev = torch.flip(w, dims=[-1])
        n = int(w.shape[-1])
        dur = n / float(sr) if sr else 0.0
        out = dict(audio)
        out["waveform"] = rev
        out["sample_rate"] = sr
        tm = {"kind": "audio_reverse", "sample_rate": sr, "num_samples": n,
              "duration_s": dur, "formula": "t_new = duration_s - t_orig",
              "map_endpoints": [[0.0, dur], [dur, 0.0]]}
        return (out, tm)


NODE_CLASS_MAPPINGS = {
    "FluidShotEncoderMEC": FluidShotEncoderMEC,
    "FluidShotDecoderMEC": FluidShotDecoderMEC,
    "AudioReverserMEC": AudioReverserMEC,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "FluidShotEncoderMEC": "Fluid Shot Encoder (Temporal Normalizer)",
    "FluidShotDecoderMEC": "Fluid Shot Decoder (Restore Timing)",
    "AudioReverserMEC": "Audio Reverser",
}
