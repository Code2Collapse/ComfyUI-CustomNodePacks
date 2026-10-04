"""Lazy AUDIO output for C2C video loaders."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any

from .handle import C2CVideo, FileSource
from .probe import get_index_table


class LazyAudio(Mapping[str, Any]):
    def __init__(
        self,
        *,
        path: str | None,
        stream_index: int,
        start_sec: float,
        duration_sec: float,
        sample_rate: int = 48000,
        channels: int = 1,
    ) -> None:
        self._path = path
        self._stream_index = int(stream_index)
        self._start_sec = float(start_sec)
        self._duration_sec = max(0.0, float(duration_sec))
        self._default_sr = int(sample_rate)
        self._default_ch = int(channels)
        self._cache: dict[str, Any] | None = None

    def _decode(self) -> dict[str, Any]:
        if self._cache is not None:
            return self._cache
        import torch

        sr = self._default_sr
        ch = self._default_ch
        want = max(1, int(round(self._duration_sec * sr)))
        if self._path is None:
            out = {"waveform": torch.zeros(1, ch, want, dtype=torch.float32), "sample_rate": sr}
            self._cache = out
            return out

        import av
        import numpy as np

        container = av.open(self._path)
        try:
            streams = [s for s in container.streams.audio if s.index == self._stream_index]
            if not streams:
                streams = list(container.streams.audio)
            if not streams:
                out = {"waveform": torch.zeros(1, ch, want, dtype=torch.float32), "sample_rate": sr}
                self._cache = out
                return out
            stream = streams[0]
            sr = int(stream.codec_context.sample_rate or sr)
            ch = int(stream.codec_context.channels or ch)
            tb = float(stream.time_base)
            seek_pts = int(self._start_sec / tb) if tb > 0 else 0
            container.seek(max(0, seek_pts), stream=stream, backward=True)
            resampler = av.AudioResampler(format="fltp", layout=stream.layout.name)
            chunks: list[Any] = []
            t_first: float | None = None
            for frame in container.decode(stream):
                if frame.pts is not None:
                    t = float(frame.pts * stream.time_base)
                    if t_first is None:
                        t_first = t
                    if t > self._start_sec + self._duration_sec + 1.0:
                        break
                for out_frame in resampler.resample(frame):
                    arr = out_frame.to_ndarray()
                    if arr.ndim == 2:
                        arr = arr.reshape(1, arr.shape[0], -1)
                    chunks.append(arr)
            if not chunks or t_first is None:
                out = {"waveform": torch.zeros(1, ch, want, dtype=torch.float32), "sample_rate": sr}
                self._cache = out
                return out

            pcm = np.concatenate(chunks, axis=-1).astype(np.float32, copy=False)
            if pcm.shape[1] != ch:
                ch = pcm.shape[1]
            offset = int(round((self._start_sec - t_first) * sr))
            if offset < 0:
                pcm = np.concatenate(
                    [np.zeros((1, ch, -offset), dtype=np.float32), pcm],
                    axis=-1,
                )
                offset = 0
            want = int(round(self._duration_sec * sr))
            segment = pcm[:, :, offset:offset + want]
            if segment.shape[-1] < want:
                pad = np.zeros((1, ch, want - segment.shape[-1]), dtype=np.float32)
                segment = np.concatenate([segment, pad], axis=-1)
            waveform = torch.from_numpy(segment)
            out = {"waveform": waveform, "sample_rate": sr}
            self._cache = out
            return out
        finally:
            container.close()

    def __getitem__(self, key: str) -> Any:
        return self._decode()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._decode())

    def __len__(self) -> int:
        return len(self._decode())


def lazy_audio_for_handle(h: C2CVideo) -> LazyAudio:
    """Fixed decision #8: trim audio to [start, start + len/fps]; sequences → silence."""
    duration_sec = len(h) / float(h.fps) if len(h) and float(h.fps) > 0 else 0.0
    if not len(h):
        return LazyAudio(path=None, stream_index=0, start_sec=0.0, duration_sec=0.0)

    sel = list(h.selected_indices())
    first = sel[0]
    if isinstance(h.source, FileSource) and h.audio is not None:
        tbl = get_index_table(h.source.stat_key, stream_index=h.stream_index)
        start_sec = float(tbl.pts_of(first) * tbl.time_base)
        return LazyAudio(
            path=h.source.path,
            stream_index=h.audio.stream_index,
            start_sec=start_sec,
            duration_sec=duration_sec,
        )
    start_sec = first / float(h.fps)
    return LazyAudio(path=None, stream_index=0, start_sec=start_sec, duration_sec=duration_sec)
