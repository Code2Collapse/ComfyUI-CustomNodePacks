"""Streaming video writer + remux concat - memory bounded by one write() call (owner A20 Part 21.4, ledger L9.74).

``StreamWriter`` encodes each ``write(frames)`` call and keeps nothing, so a long job never holds more than one chunk.
``concat`` joins chunk files of the same codec / size / rate by remuxing packets (no decode, no re-encode).

PyAV only (ComfyUI core requires ``av``). Colour follows the conventions CustomNodePacks' Save Video measured
(L9.40): 8-bit sources start from rgb24, 10-bit from full-scale rgb48le, the RGB -> YUV conversion uses the BT.709
matrix in TV range with accurate swscale rounding, and the stream is tagged BT.709 so file and pixels agree.
"""
from __future__ import annotations

import os
from fractions import Fraction

import numpy as np

_BT709 = 1
_RANGE_TV = 1
_SWS_FLAGS = 0x4 | 0x40000 | 0x2000 | 0x4000        # bicubic + accurate rounding + full chroma interpolation / input

CODECS = {
    # name: (encoder, container, pix_fmt 8-bit, pix_fmt 10-bit, even size needed)
    "h264": ("libx264", "mp4", "yuv420p", "yuv420p10le", True),
    "prores": ("prores_ks", "mov", "yuv422p10le", "yuv422p10le", True),
    "ffv1": ("ffv1", "matroska", "yuv444p", "yuv444p10le", False),
}
EXTENSIONS = {"mp4": ".mp4", "mov": ".mov", "matroska": ".mkv"}


class WriterError(ValueError):
    """A writer refusal in plain words."""


def _frac(fps) -> Fraction:
    return fps if isinstance(fps, Fraction) else Fraction(fps).limit_denominator(1001000)


class StreamWriter:
    """Open once, ``write`` frames in any number of calls ([N, H, W, 3] float 0..1, torch or numpy), ``close``."""

    def __init__(self, path: str, fps, width: int, height: int, *, codec: str = "h264", quality: int = 85,
                 ten_bit: bool = False, audio_rate: int = 0, audio_channels: int = 0):
        """``audio_rate`` / ``audio_channels`` > 0 add an audio track (AAC in mp4 / mov, FLAC in mkv) that
        ``write_audio`` fills; without them the file is video only, exactly as before (L9.74 slice 7)."""
        import av

        if codec not in CODECS:
            raise WriterError(f"unknown codec {codec!r}: use one of {', '.join(CODECS)}")
        encoder, container, pix8, pix10, even = CODECS[codec]
        if even and (int(width) % 2 or int(height) % 2):
            raise WriterError(f"{codec} needs an even width and height (got {width}x{height}): use an even size, or "
                              f"codec ffv1")
        self.path, self.codec, self.width, self.height = path, codec, int(width), int(height)
        self.fps = _frac(fps)
        self.ten_bit = bool(ten_bit) or codec == "prores"
        self.pix_fmt = pix10 if self.ten_bit else pix8
        self.frames = 0
        self._av = av
        self._out = av.open(path, mode="w", format=container,
                            options={"movflags": "+faststart"} if container in ("mp4", "mov") else {})
        self._stream = self._out.add_stream(encoder, rate=self.fps)
        self._stream.width, self._stream.height, self._stream.pix_fmt = self.width, self.height, self.pix_fmt
        cc = self._stream.codec_context
        cc.colorspace, cc.color_range, cc.color_primaries, cc.color_trc = _BT709, _RANGE_TV, 1, 1
        q = max(0, min(100, int(quality)))
        if codec == "h264":
            self._stream.options = {"crf": str(round(51 - q * 0.41)), "preset": "medium"}
        elif codec == "prores":
            self._stream.options = {"profile": "3"}                  # 422 HQ
        self._astream = None
        self.audio_rate, self.audio_channels, self.audio_samples = int(audio_rate or 0), int(audio_channels or 0), 0
        if self.audio_rate > 0 and self.audio_channels > 0:
            self.audio_channels = min(2, self.audio_channels)        # mono / stereo; more channels keep the first two
            acodec = "flac" if container == "matroska" else "aac"
            self._alayout = "mono" if self.audio_channels == 1 else "stereo"
            self._astream = self._out.add_stream(acodec, rate=self.audio_rate)
            self._astream.layout = self._alayout
            afmt = getattr(getattr(self._astream.codec_context, "format", None), "name", None) or (
                "fltp" if acodec == "aac" else "s16")
            self._resampler = av.AudioResampler(format=afmt, layout=self._alayout, rate=self.audio_rate)
            self._fifo = av.AudioFifo()
            self._apts = 0
        self._closed = False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        if not self._closed:
            self.close()

    def _frame(self, rgb: np.ndarray):
        x = np.clip(np.asarray(rgb, dtype=np.float32)[..., :3], 0.0, 1.0)
        if self.ten_bit:
            q, src = np.rint(x * 65535.0).astype(np.uint16), "rgb48le"
        else:
            q, src = np.rint(x * 255.0).astype(np.uint8), "rgb24"
        frame = self._av.VideoFrame.from_ndarray(np.ascontiguousarray(q), format=src)
        return frame.reformat(format=self.pix_fmt, dst_colorspace=_BT709, dst_color_range=_RANGE_TV,
                              interpolation=_SWS_FLAGS)

    def write(self, frames) -> int:
        """Encode these frames now. Returns the total written so far."""
        if self._closed:
            raise WriterError("the writer is closed")
        arr = frames.detach().cpu().numpy() if hasattr(frames, "detach") else np.asarray(frames)
        if arr.ndim == 3:
            arr = arr[None]
        if arr.shape[1:3] != (self.height, self.width):
            raise WriterError(f"frames are {arr.shape[2]}x{arr.shape[1]}, the writer was opened at "
                              f"{self.width}x{self.height}")
        for rgb in arr:
            frame = self._frame(rgb)
            frame.pts = self.frames
            frame.time_base = 1 / self.fps
            for packet in self._stream.encode(frame):
                self._out.mux(packet)
            self.frames += 1
        return self.frames

    def write_audio(self, samples) -> int:
        """Encode audio now: [channels, samples] (or [samples]) float32 at ``audio_rate``. Returns samples so far."""
        if self._astream is None:
            raise WriterError("this writer was opened without audio (pass audio_rate and audio_channels)")
        if self._closed:
            raise WriterError("the writer is closed")
        arr = samples.detach().cpu().numpy() if hasattr(samples, "detach") else np.asarray(samples)
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[None]
        if arr.shape[0] < self.audio_channels:
            arr = np.repeat(arr[:1], self.audio_channels, axis=0)
        arr = np.ascontiguousarray(arr[: self.audio_channels])
        if not arr.shape[1]:
            return self.audio_samples
        frame = self._av.AudioFrame.from_ndarray(arr, format="fltp", layout=self._alayout)
        frame.sample_rate = self.audio_rate
        frame.pts = self.audio_samples
        frame.time_base = Fraction(1, self.audio_rate)
        self.audio_samples += arr.shape[1]
        for f in self._resampler.resample(frame):
            f.pts = None                                     # the fifo re-times; frames go out with our own pts
            self._fifo.write(f)
        self._drain_audio(final=False)
        return self.audio_samples

    def _drain_audio(self, final: bool) -> None:
        size = self._astream.codec_context.frame_size or 1024
        while self._fifo.samples >= size or (final and self._fifo.samples):
            f = self._fifo.read(min(size, self._fifo.samples))
            f.pts, f.time_base = self._apts, Fraction(1, self.audio_rate)
            self._apts += f.samples
            for packet in self._astream.encode(f):
                self._out.mux(packet)
        if final:
            for packet in self._astream.encode(None):
                self._out.mux(packet)

    def close(self) -> dict:
        if not self._closed:
            for packet in self._stream.encode():
                self._out.mux(packet)
            if self._astream is not None:
                self._drain_audio(final=True)
            self._out.close()
            self._closed = True
        out = {"path": self.path, "frames": self.frames, "codec": self.codec, "pix_fmt": self.pix_fmt,
               "fps": f"{self.fps.numerator}/{self.fps.denominator}", "width": self.width, "height": self.height}
        if self._astream is not None:
            out["audio"] = {"rate": self.audio_rate, "channels": self.audio_channels, "samples": self.audio_samples,
                            "codec": self._astream.codec_context.name}
        return out


def _video_stream(container):
    vs = [s for s in container.streams if s.type == "video"]
    if not vs:
        raise WriterError(f"no video stream in {container.name}")
    return vs[0]


def concat(chunks: list[str], out_path: str) -> dict:
    """Join chunk files by remuxing their video packets, in the order given. Same codec / size / rate required."""
    import av

    if not chunks:
        raise WriterError("nothing to join: no chunk files")
    ext = os.path.splitext(out_path)[1].lower()
    fmt = {".mp4": "mp4", ".mov": "mov", ".mkv": "matroska"}.get(ext)
    if fmt is None:
        raise WriterError(f"the output must be .mp4, .mov or .mkv (got {ext or 'no extension'})")
    first = av.open(chunks[0])
    try:
        ref = _video_stream(first)
        ref_sig = (ref.codec_context.name, ref.codec_context.width, ref.codec_context.height, ref.average_rate)
        out = av.open(out_path, mode="w", format=fmt)
        try:
            ostream = out.add_stream_from_template(ref)
        except AttributeError:                                   # older PyAV
            ostream = out.add_stream(template=ref)
    finally:
        first.close()
    offset_s = Fraction(0)                                       # where the next chunk starts, in exact seconds
    frames, packets = 0, 0
    try:
        for path in chunks:
            src = av.open(path)
            try:
                st = _video_stream(src)
                sig = (st.codec_context.name, st.codec_context.width, st.codec_context.height, st.average_rate)
                if sig != ref_sig:
                    raise WriterError(f"{os.path.basename(path)} does not match the first chunk "
                                      f"({sig[0]} {sig[1]}x{sig[2]} @ {sig[3]} vs {ref_sig[0]} {ref_sig[1]}x{ref_sig[2]} @ "
                                      f"{ref_sig[3]})")
                tb = Fraction(st.time_base)
                first, last_end = None, 0
                for pkt in src.demux(st):
                    if pkt.dts is None and pkt.pts is None:
                        continue                                  # the demuxer's flush packet
                    if first is None:
                        first = pkt.dts if pkt.dts is not None else pkt.pts
                    shift = round(offset_s / tb) - first          # this chunk's ticks, whatever its time base
                    if pkt.pts is not None:
                        pkt.pts += shift
                    if pkt.dts is not None:
                        pkt.dts += shift
                    end = (pkt.pts if pkt.pts is not None else pkt.dts) + (pkt.duration or 0)
                    last_end = max(last_end, end)
                    pkt.stream = ostream
                    out.mux(pkt)
                    packets += 1
                frames += st.frames or 0
                offset_s = last_end * tb
            finally:
                src.close()
    finally:
        out.close()
    return {"path": out_path, "chunks": len(chunks), "frames": frames, "packets": packets,
            "codec": ref_sig[0], "width": ref_sig[1], "height": ref_sig[2], "rate": str(ref_sig[3])}


__all__ = ["CODECS", "StreamWriter", "WriterError", "concat"]
