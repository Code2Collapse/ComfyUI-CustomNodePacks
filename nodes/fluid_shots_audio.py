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

import json
import math
import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)

_MAX_INSERTS_PER_PAIR = 16   # safety cap so one cut can't explode the batch
_MAX_OUT_FACTOR = 4          # L9.264: the whole clip may grow to at most 4x its frames (+16), or RAM runs out
_FLOW_PROC_LONG_SIDE = 384   # downscale for Farneback speed; magnitudes rescaled
# Frame grids (audit 2026-11-27): the encoder's output count is B + the in-betweens, which is generally NOT on a
# model's frame grid (Wan 4n+1, MiniMax H3 17n+5). A model that trims or pads the batch to its grid makes the
# decoder fall back to its proportional, approximate mapping (its info output says by how much). For exact timing,
# keep the frame count unchanged between encoder and decoder.


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


MOTION_METRICS = ("global p95", "local tiles")   # L9.171: appended choice; "global p95" = the metric above
_TILES = 4


def _motion_from_flow(flow, rescale, metric):
    """Motion of one pair in full-resolution pixels. "global p95": the 95th percentile of the whole flow magnitude
    (the metric of _flow_and_motion). "local tiles": the frame cut into a 4x4 grid, the largest of the tiles' 95th
    percentiles - a small fast subject on a still background then counts as fast (A25 spec item 90.1)."""
    mag = np.linalg.norm(flow, axis=2)
    if metric != "local tiles":
        return float(np.percentile(mag, 95.0)) * rescale
    h, w = mag.shape[:2]
    best = 0.0
    for ty in range(_TILES):
        for tx in range(_TILES):
            tile = mag[ty * h // _TILES:(ty + 1) * h // _TILES, tx * w // _TILES:(tx + 1) * w // _TILES]
            if tile.size:
                best = max(best, float(np.percentile(tile, 95.0)))
    return best * rescale


def _cut_distance(g0, g1):
    """Bhattacharyya distance (0 = same, 1 = unrelated) of two gray frames' 64-bin histograms (A25 spec item 90.2)."""
    import cv2
    h0 = cv2.calcHist([g0], [0], None, [64], [0, 256])
    h1 = cv2.calcHist([g1], [0], None, [64], [0, 256])
    return float(cv2.compareHist(h0, h1, cv2.HISTCMP_BHATTACHARYYA))


def _audio_parts(audio):
    """(waveform as [B,C,S], restore fn) so retimed audio keeps the rank it came in with."""
    w = audio["waveform"]
    if w.ndim == 3:
        return w, lambda x: x
    if w.ndim == 2:
        return w.unsqueeze(0), lambda x: x.squeeze(0)
    return w.reshape(1, 1, -1), lambda x: x.reshape(-1)


def retime_audio(audio, positions, batch):
    """Audio made on the expanded timeline (`batch` frames) brought back onto the source frames (A25 spec item 89.1).

    Source frame k sat at expanded position positions[k]; between those knots the map is linear, after the last knot
    it runs on at speed 1. Each output sample is linearly interpolated from the expanded waveform."""
    if audio is None:
        return None
    if not (isinstance(audio, dict) and torch.is_tensor(audio.get("waveform"))):
        raise ValueError("FluidShotDecoder: audio must be an AUDIO dict with a 'waveform' tensor.")
    w, restore = _audio_parts(audio)
    n_src = len(positions)
    S = int(w.shape[-1])
    if n_src < 2 or batch < 1 or S < 2:
        return audio
    spf = S / float(batch)
    s_out = int(round(n_src * spf))
    u = np.arange(s_out, dtype=np.float64) / spf
    knots = np.arange(n_src, dtype=np.float64)
    pos = np.asarray(positions, dtype=np.float64)
    v = np.interp(u, knots, pos)
    after = u > knots[-1]
    v[after] = pos[-1] + (u[after] - knots[-1])
    x = np.clip(v * spf, 0.0, S - 1.0)
    i0 = np.floor(x).astype(np.int64)
    i1 = np.minimum(i0 + 1, S - 1)
    frac = torch.from_numpy((x - i0).astype(np.float32)).to(device=w.device, dtype=w.dtype)
    out = (w.index_select(-1, torch.from_numpy(i0).to(w.device)) * (1.0 - frac)
           + w.index_select(-1, torch.from_numpy(i1).to(w.device)) * frac)
    res = dict(audio)
    res["waveform"] = restore(out)
    return res


def integrity_report(images, selected, count_mismatch, missing):
    """Missing / repeated / out-of-order / identical-content frames of the decoder's pick (A25 spec item 89.2).
    Compared one frame pair at a time (no copy of the batch)."""
    repeated, non_mono, dup = [], [], []
    with torch.no_grad():
        for k in range(1, len(selected)):
            if selected[k] == selected[k - 1]:
                repeated.append(k)
                continue
            if selected[k] < selected[k - 1]:
                non_mono.append(k)
            if float((images[selected[k]] - images[selected[k - 1]]).abs().max()) <= 1e-6:
                dup.append(k)
    return {"ok": not (count_mismatch or missing or repeated or non_mono or dup),
            "count_mismatch": bool(count_mismatch), "missing": list(missing), "repeated": repeated,
            "non_monotonic": non_mono, "content_duplicates": dup}


def _warp_mid(a_u8, b_u8, flow, t):
    """Flow-warp intermediate at fraction t between full-res RGB frames (uint8 or float32; the result is float32
    in the same value range).

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
            "optional": {
                # L9.171 (A25 spec item 90) - appended; at the defaults the frames and the TIMEMAP are as before
                "scene_cut_threshold": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "0 = off. Above 0: a pair of frames whose gray histograms differ by at least this "
                               "(Bhattacharyya distance, 0 = same .. 1 = unrelated; hard cuts are usually above "
                               "0.4) is a scene cut and gets no in-betweens, so two shots are never blended. The cut "
                               "pairs are listed in the TIMEMAP (scene_cuts)."}),
                "motion_metric": (list(MOTION_METRICS), {
                    "default": "global p95",
                    "tooltip": "global p95: motion = 95th percentile of the whole frame's flow (as before). local "
                               "tiles: the frame is cut into a 4x4 grid and the fastest tile counts, so a small fast "
                               "subject (a hand, a ball) on a still background also gets in-betweens."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "TIMEMAP")
    RETURN_NAMES = ("normalized_images", "time_map")
    OUTPUT_TOOLTIPS = ("The frames with flow-warped in-betweens on fast sections (more frames than the input).",
                       "Where each original frame sits in the new batch - wire to Fluid Shot Decoder.")
    FUNCTION = "execute"
    CATEGORY = "MEC/Temporal"
    DESCRIPTION = ("Analyzes optical-flow speed and inserts in-between frames on fast "
                   "sections so motion per frame is constant — safe for Wan-style AI video. "
                   "Pair with Fluid Shot Decoder to restore the original timing exactly.")

    def execute(self, images, max_px_per_frame, scene_cut_threshold=0.0, motion_metric="global p95"):
        if motion_metric not in MOTION_METRICS:
            raise ValueError(f"FluidShotEncoder: unknown motion_metric {motion_metric!r}; use one of "
                             f"{', '.join(MOTION_METRICS)}.")
        cut_thr = float(scene_cut_threshold or 0.0)
        extended = cut_thr > 0.0 or motion_metric != "global p95"
        scene_cuts, pair_motion = [], []
        if images.ndim == 3:
            images = images.unsqueeze(0)
        B = images.shape[0]
        if B < 2:
            tm = {"kind": "fluid_shot", "src_count": int(B), "out_count": int(B),
                  "original_positions": list(range(int(B))),
                  "src_of_output": [float(i) for i in range(int(B))]}
            return (images, tm)

        try:
            import cv2  # noqa: F401
        except ImportError:
            # L9.264 (CNP-NODES-2 audit): every pair's flow failed silently and the clip came back unchanged
            raise RuntimeError("Fluid Shot Encoder needs OpenCV (opencv-python) for optical flow, and it is not "
                               "installed in ComfyUI's Python. Install opencv-python and run again.") from None
        max_out = _MAX_OUT_FACTOR * B + 16
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
                if motion_metric != "global p95":
                    motion = _motion_from_flow(flow, rescale, motion_metric)
                if extended:
                    pair_motion.append(round(float(motion), 4))
                if cut_thr > 0.0 and _cut_distance(g0, g1) >= cut_thr:
                    scene_cuts.append(i)          # L9.171: a scene cut gets no in-betweens
                    motion = 0.0
                if motion > max_px:
                    n_insert = min(_MAX_INSERTS_PER_PAIR,
                                   int(math.ceil(motion / max_px)) - 1)
            except Exception as exc:  # noqa: BLE001 — flow failure = no inserts
                logger.warning("FluidShotEncoder: flow failed on pair %d (%s); "
                               "passing through.", i, exc)

            if n_insert > 0 and len(out_frames) + n_insert + (B - 1 - i) > max_out:
                raise RuntimeError(
                    f"Fluid Shot Encoder would make more than {max_out} frames from {B} (pair {i}: {n_insert} "
                    f"in-betweens). Raise max_px_per_frame (now {max_px:g}) or split the clip at the fast cut.")
            if n_insert > 0:
                # Audit 2026-11-27: the in-betweens are warped in float32 at the plate's own values. They used to
                # be made from 8-bit copies (clipped to 0..1 and quantised to 1/255: HDR highlights flattened to
                # 1.0, shadows below 1/255 went black, banding next to the untouched float source frames).
                a_f = a_t.detach().cpu().float().numpy()
                b_f = b_t.detach().cpu().float().numpy()
                for k in range(1, n_insert + 1):
                    t = k / float(n_insert + 1)
                    try:
                        mid = _warp_mid(a_f, b_f, flow, t)
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
        if extended:                              # L9.171: only when a new feature is on (old TIMEMAPs unchanged)
            tm.update(motion_metric=motion_metric, pair_motion=pair_motion)
            if cut_thr > 0.0:
                tm["scene_cuts"] = scene_cuts
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
            "optional": {
                # appended (L9.264): saved workflows keep their behaviour (on = the old proportional remap)
                "allow_approximate": ("BOOLEAN", {"default": True, "tooltip":
                    "When the processed batch has a different frame count than the encoder made (e.g. a model "
                    "trimmed it to 4n+1), map the original frames proportionally - their timing is then only "
                    "approximate and the info output says by how much. Off: stop with an error instead."}),
                # L9.171 (A25 spec item 89) - appended
                "audio": ("AUDIO", {"tooltip":
                    "Audio made on the expanded (normalized) timeline, e.g. by an audio-video model. It is retimed "
                    "onto the original frames with the same map as the pictures (linear between the original "
                    "frames' positions, speed 1 after the last one), so it stays in sync with the restored clip."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "AUDIO", "STRING", "STRING")
    RETURN_NAMES = ("images", "info", "audio", "frame_map", "integrity")
    OUTPUT_TOOLTIPS = ("The processed frames at the original count and pacing.",
                       "How the frames were picked; says how approximate the timing is when the count changed.",
                       "The wired audio retimed onto the original frames (nothing when no audio is wired).",
                       "JSON: for every original frame its position on the expanded timeline and the batch frame "
                       "that was picked.",
                       "JSON check of the pick: missing (outside the batch), repeated, out-of-order and identical-"
                       "content frames, and whether the frame count changed; ok is true only when all are empty.")
    FUNCTION = "execute"
    CATEGORY = "MEC/Temporal"
    DESCRIPTION = ("Selects exactly the frames that correspond to the ORIGINAL input "
                   "frames using the encoder's TIMEMAP — restores the original frame "
                   "count and slow-to-fast pacing.")

    def execute(self, images, time_map, allow_approximate=True, audio=None):
        if images.ndim == 3:
            images = images.unsqueeze(0)
        if not (isinstance(time_map, dict) and time_map.get("kind") == "fluid_shot"):
            raise ValueError("FluidShotDecoder: time_map is not a Fluid Shot TIMEMAP "
                             "(wire it from Fluid Shot Encoder).")
        B = int(images.shape[0])
        positions = [int(p) for p in time_map.get("original_positions", [])]
        if not positions:
            empty = json.dumps({"src_count": 0, "batch": B, "expected": int(time_map.get("out_count", 0) or 0),
                                "exact": False, "frames": []})
            bad = json.dumps({"ok": False, "count_mismatch": False, "missing": [], "repeated": [],
                              "non_monotonic": [], "content_duplicates": [],
                              "note": "the TIMEMAP has no original positions"})
            return (images, "TIMEMAP has no original positions: images passed through unchanged.", audio, empty, bad)
        expected = int(time_map.get("out_count", 0))
        info = f"exact: {len(positions)} original frames selected from {B}."
        positions_f = [float(p) for p in positions]
        if expected and B != expected:
            if not allow_approximate:
                raise ValueError(
                    f"FluidShotDecoder: the batch has {B} frames but the encoder made {expected}. The model changed "
                    f"the frame count, so the original timing cannot be restored exactly. Turn on allow_approximate "
                    f"to map it proportionally, or keep the frame count unchanged between encoder and decoder.")
            # AI models may trim/pad the batch (e.g. Wan's 4n+1): scale the map proportionally, then clamp
            ratio = (B - 1) / float(max(1, expected - 1))
            scaled = [p * ratio for p in positions]
            positions = [int(round(p)) for p in scaled]
            positions_f = scaled
            worst = max((abs(p - s) for p, s in zip(positions, scaled)), default=0.0)
            info = (f"APPROXIMATE: batch has {B} frames, the encoder made {expected}; original frames mapped "
                    f"proportionally (rounding error up to {worst:.2f} frame).")
            logger.warning("FluidShotDecoder: %s", info)
        selected = [min(max(p, 0), B - 1) for p in positions]
        idx = torch.tensor(selected, dtype=torch.long, device=images.device)
        # L9.171 (A25 spec item 89): explicit source -> output correspondence, integrity check, retimed audio
        mismatch = bool(expected and B != expected)
        missing = [k for k, p in enumerate(positions) if p < 0 or p > B - 1]
        frame_map = json.dumps({"src_count": len(positions), "batch": B, "expected": expected, "exact": not mismatch,
                                "frames": [{"src": k, "out": o, "pos": round(float(pf), 4)}
                                           for k, (o, pf) in enumerate(zip(selected, positions_f))]})
        integrity = json.dumps(integrity_report(images, selected, mismatch, missing))
        out_audio = retime_audio(audio, positions_f, B) if audio is not None else None
        return (images.index_select(0, idx), info, out_audio, frame_map, integrity)


def _grain_reverse(seg: torch.Tensor, sr: int, grain_ms: float) -> torch.Tensor:
    """Reverse the ORDER of short grains while each grain still plays forwards (L9.173 / A25 spec 79.1).

    Hann grains at 50 % overlap (they sum to exactly 1 at that hop, so a steady signal keeps its level); the
    grain that started at s is written at len - s - G. Notes keep their pitch AND their attack-then-decay shape (plain
    reversal turns every decay into a swell); only the order of events runs backwards. A segment shorter than two
    grains falls back to plain reversal."""
    n = int(seg.shape[-1])
    g = int(round(max(1.0, float(grain_ms)) * sr / 1000.0))
    g = max(4, g + (g % 2))                                    # even, so hop = g / 2 is exact
    if n < 2 * g:
        return torch.flip(seg, dims=[-1])
    hop = g // 2
    # half-sample-shifted periodic Hann: still sums to exactly 1 at hop g/2, and never 0, so an edge sample covered
    # by a single grain is still recovered exactly after normalising by the summed weights
    k = torch.arange(g, dtype=torch.float32, device=seg.device)
    win = (0.5 - 0.5 * torch.cos(2.0 * math.pi * (k + 0.5) / g)).to(seg.dtype)
    pad = (-(n - g)) % hop                                     # grains tile [0, n + pad) exactly
    x = torch.nn.functional.pad(seg, (0, pad))
    real = torch.nn.functional.pad(torch.ones(n, dtype=seg.dtype, device=seg.device), (0, pad))
    m = int(x.shape[-1])
    out = torch.zeros_like(x)
    wsum = torch.zeros(m, dtype=seg.dtype, device=seg.device)
    for s0 in range(0, m - g + 1, hop):
        d0 = m - s0 - g
        out[..., d0:d0 + g] += x[..., s0:s0 + g] * win
        wsum[d0:d0 + g] += win * real[s0:s0 + g]          # the zero tail carries no weight: edges keep their level
    out = out / wsum.clamp_min(1e-12)
    return out[..., pad:pad + n]                               # where plain reversal puts the real samples


_REVERSE_MODES = ("reverse", "grain_reverse (keeps note attacks)")


class AudioReverserMEC:
    """Reverse an AUDIO waveform; emit a timestamp TIMEMAP for video syncing."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {"tooltip": "Standard ComfyUI audio "
                                    "({'waveform': [B,C,S], 'sample_rate': int})."}),
            },
            # L9.173 (A25 spec 79): appended, optional; the defaults reverse the whole clip exactly as before
            "optional": {
                "start_seconds": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 86400.0, "step": 0.01,
                    "tooltip": "Reverse only from this time on (seconds); audio before it plays unchanged. "
                               "0 with end_seconds -1 = the whole clip (as before)."}),
                "end_seconds": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 86400.0, "step": 0.01,
                    "tooltip": "Reverse only up to this time (seconds); audio after it plays unchanged. -1 = the "
                               "end of the clip. The clip keeps its length, and time_map lists the reversed and the "
                               "untouched ranges as segments."}),
                "reverse_mode": (list(_REVERSE_MODES), {"default": _REVERSE_MODES[0],
                    "tooltip": "reverse = every sample backwards (as before; notes swell instead of decaying). "
                               "grain_reverse = the order of short grains runs backwards while each grain plays "
                               "forwards, so notes keep their pitch and attack; only the order of events is "
                               "reversed."}),
                "grain_ms": ("FLOAT", {"default": 80.0, "min": 10.0, "max": 1000.0, "step": 1.0,
                    "tooltip": "grain_reverse only: grain length in milliseconds (50 % overlap). Shorter = finer "
                               "time order, longer = smoother sound. A range shorter than two grains is reversed "
                               "plainly."}),
            },
        }

    RETURN_TYPES = ("AUDIO", "TIMEMAP")
    RETURN_NAMES = ("audio", "time_map")
    OUTPUT_TOOLTIPS = ("The audio played backwards (same sample rate and channels).",
                       "Maps original to reversed timestamps (t_new = duration - t_orig) for syncing video; with a "
                       "time range it also lists each range as a segment.")
    FUNCTION = "execute"
    CATEGORY = "MEC/Audio"
    DESCRIPTION = ("Reverses the audio waveform (torch.flip on the sample dim), "
                   "preserving sample rate and channels. TIMEMAP maps original to "
                   "reversed timestamps (t_new = duration - t_orig). Optional: reverse only a time range, or reverse "
                   "the grain order so notes keep their attacks.")

    def execute(self, audio, start_seconds=0.0, end_seconds=-1.0, reverse_mode=_REVERSE_MODES[0], grain_ms=80.0):
        if not (isinstance(audio, dict) and "waveform" in audio):
            raise ValueError("AudioReverser: expected AUDIO dict with a 'waveform' tensor.")
        w = audio["waveform"]
        sr = int(audio.get("sample_rate", 44100))
        if not torch.is_tensor(w):
            raise ValueError("AudioReverser: 'waveform' is not a tensor.")
        if reverse_mode not in _REVERSE_MODES:
            raise ValueError(f"AudioReverser: reverse_mode must be one of {list(_REVERSE_MODES)}, got {reverse_mode!r}.")
        n = int(w.shape[-1])
        dur = n / float(sr) if sr else 0.0
        s0 = 0 if float(start_seconds) <= 0 else int(round(float(start_seconds) * sr))
        s1 = n if float(end_seconds) < 0 else min(n, int(round(float(end_seconds) * sr)))
        whole = s0 == 0 and s1 == n
        if not whole and s1 - s0 < 2:
            raise ValueError(f"AudioReverser: the range {float(start_seconds):g} s to {float(end_seconds):g} s holds no "
                             f"audio (the clip is {dur:.3f} s long; end_seconds must be after start_seconds, or -1).")
        grain = reverse_mode != _REVERSE_MODES[0]
        seg = w[..., s0:s1]
        seg_rev = _grain_reverse(seg, sr, grain_ms) if grain else torch.flip(seg, dims=[-1])
        rev = seg_rev if whole else torch.cat([w[..., :s0], seg_rev, w[..., s1:]], dim=-1)
        out = dict(audio)
        out["waveform"] = rev
        out["sample_rate"] = sr
        tm = {"kind": "audio_reverse", "sample_rate": sr, "num_samples": n,
              "duration_s": dur, "formula": "t_new = duration_s - t_orig",
              "map_endpoints": [[0.0, dur], [dur, 0.0]]}
        if not whole:                                   # L9.173: piecewise map (the old keys stay, per range)
            a, b = s0 / float(sr), s1 / float(sr)
            segs = []
            if s0 > 0:
                segs.append({"src": [0.0, a], "dst": [0.0, a], "reversed": False})
            segs.append({"src": [a, b], "dst": [b, a], "reversed": True})
            if s1 < n:
                segs.append({"src": [b, dur], "dst": [b, dur], "reversed": False})
            tm.update({"formula": "inside [range_start_s, range_end_s]: t_new = range_start_s + range_end_s - t_orig;"
                                  " elsewhere t_new = t_orig",
                       "range_start_s": a, "range_end_s": b, "segments": segs, "map_endpoints": [[a, b], [b, a]]})
        if grain:
            tm.update({"reverse_mode": "grain_reverse", "grain_ms": float(grain_ms),
                       "note": "grain order reversed, each grain plays forwards: the map holds to +/- grain_ms / 2"})
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
