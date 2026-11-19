"""Save Video (C2C) delivery helpers - L9.171 (A25 spec item 73.2 + AusBoss Save Video parity).

pingpong_frames / pingpong_audio: a seamless forward-backward loop (AusBoss Save Video's pingpong), with the audio
bounced the same way so it stays in sync. manifest_path / write_manifest / validate_output / frame_runs: the delivery
manifest written next to the file - source frame mapping, audio, codec settings, OCIO transforms, colour tags and a
check of the written file read back from disk.
"""

from __future__ import annotations

import json
import os
from typing import Any

import torch


# PORTED FROM: https://github.com/ausboss/ComfyUI-AusBoss nodes/_video_save_helpers.py pingpong_frames (MIT)
def pingpong_frames(frames):
    """Frames followed by the same frames reversed, endpoints not repeated.

    Dropping the first and last of the reversed half is what makes the seam
    invisible: keeping them would show the turnaround frames twice each and
    the loop would visibly hitch at both ends. Fewer than three frames have no
    interior to bounce through, so they come back untouched.
    """
    if not isinstance(frames, torch.Tensor) or frames.shape[0] < 3:
        return frames
    return torch.cat([frames, frames.flip(0)[1:-1]], dim=0)


def _wave2d(audio: dict):
    w = audio["waveform"]
    w = w if torch.is_tensor(w) else torch.as_tensor(w, dtype=torch.float32)
    if w.ndim == 3:
        return w[0], (lambda x: x.unsqueeze(0))
    if w.ndim == 1:
        return w.unsqueeze(0), (lambda x: x.squeeze(0))
    return w, (lambda x: x)


def pingpong_audio(audio: dict | None, n_frames: int, fps: float) -> dict | None:
    """The audio of the bounced clip: the forward part (exactly n frames long, padded with silence or cut), then the
    same audio reversed without the first and last frame's duration - the picture's frames n-2 .. 1 - so the sound
    plays backwards in step with the picture. Fewer than three frames: unchanged (the picture is not bounced)."""
    if audio is None or n_frames < 3 or fps <= 0:
        return audio
    w, restore = _wave2d(audio)
    sr = int(audio["sample_rate"])
    fwd_n = int(round(n_frames / float(fps) * sr))
    if w.shape[1] >= fwd_n:
        fwd = w[:, :fwd_n]
    else:
        fwd = torch.cat([w, torch.zeros(w.shape[0], fwd_n - w.shape[1], dtype=w.dtype, device=w.device)], dim=1)
    total_n = int(round((2 * n_frames - 2) / float(fps) * sr))
    one = int(round(sr / float(fps)))                      # one frame's duration, dropped from each end of the bounce
    back = fwd.flip(1)[:, one:]
    back = back[:, :max(0, total_n - fwd_n)]
    out = torch.cat([fwd, back], dim=1)
    return {"waveform": restore(out), "sample_rate": sr}


def manifest_path(master_path: str, is_sequence: bool) -> str:
    """'<name>.manifest.json' next to a video file; '<folder>.manifest.json' beside an image-sequence folder."""
    if is_sequence:
        return str(master_path).rstrip("/\\") + ".manifest.json"
    return os.path.splitext(str(master_path))[0] + ".manifest.json"


def frame_runs(indices) -> list[list[int]]:
    """[[start, stop (inclusive), step], ...] - a frame list written compactly (0..99 = one run)."""
    idx = [int(i) for i in (indices or [])]
    runs: list[list[int]] = []
    i = 0
    while i < len(idx):
        start = idx[i]
        if i + 1 < len(idx):
            step = idx[i + 1] - idx[i]
            j = i + 1
            while j + 1 < len(idx) and idx[j + 1] - idx[j] == step:
                j += 1
            if step != 0 and j > i:
                runs.append([start, idx[j], step])
                i = j + 1
                continue
        runs.append([start, start, 1])
        i += 1
    return runs


def write_manifest(path: str, data: dict) -> str:
    """JSON, UTF-8, written to a temporary name and moved into place (never a half-written manifest)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, default=str)
    os.replace(tmp, path)
    return path


def validate_output(paths: list[str], spec: Any, *, frames: int, width: int, height: int, fps: float,
                    audio_expected: bool) -> dict:
    """Read the written output back and compare it with what was meant to be written. Never raises: a failure to
    read is itself a problem in the report."""
    problems: list[str] = []
    checked: dict[str, Any] = {}
    master = paths[0] if paths else ""
    try:
        if getattr(spec, "sequence", False):
            ext = "." + str(getattr(spec, "file_ext", None) or getattr(spec, "ext", "")).lower().lstrip(".")
            files = [f for f in os.listdir(master) if f.lower().endswith(ext)] if os.path.isdir(master) else []
            checked.update(kind="sequence", folder=master, files=len(files))
            if len(files) != int(frames):
                problems.append(f"{len(files)} {ext} files in the folder, {frames} frames were meant to be written")
        elif getattr(spec, "codec", "") in ("pil_gif", "pil_webp"):
            from PIL import Image
            with Image.open(master) as im:
                n = int(getattr(im, "n_frames", 1))
                checked.update(kind="image animation", frames=n, width=im.size[0], height=im.size[1])
                if n != int(frames):
                    problems.append(f"{n} frames in the file, {frames} meant")
                if tuple(im.size) != (int(width), int(height)):
                    problems.append(f"size {im.size[0]}x{im.size[1]}, {width}x{height} meant")
        else:
            import av
            with av.open(master) as c:
                vs = [s for s in c.streams if s.type == "video"]
                aus = [s for s in c.streams if s.type == "audio"]
                if not vs:
                    problems.append("no video stream in the file")
                else:
                    v = vs[0]
                    w, h = int(v.codec_context.width), int(v.codec_context.height)
                    rate = float(v.average_rate) if v.average_rate else 0.0
                    n = sum(1 for p in c.demux(v) if p.pts is not None)
                    checked.update(kind="container", frames=n, width=w, height=h, fps=round(rate, 4),
                                   audio_streams=len(aus))
                    if n != int(frames):
                        problems.append(f"{n} video frames in the file, {frames} meant")
                    if (w, h) != (int(width), int(height)):
                        problems.append(f"size {w}x{h}, {width}x{height} meant")
                    if rate and abs(rate - float(fps)) > 0.01:
                        problems.append(f"frame rate {rate:.4f}, {float(fps):.4f} meant")
                if audio_expected and getattr(spec, "audio_codec", None) and not aus:
                    problems.append("audio was wired but the file has no audio stream")
                if not audio_expected and aus:
                    problems.append("the file has an audio stream although no audio was wired")
    except Exception as exc:  # noqa: BLE001 - reported, never raised
        problems.append(f"could not read the output back: {exc}")
    return {"ok": not problems, "checked": checked, "problems": problems}
