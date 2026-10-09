"""Save Video (C2C) — CPU-only encode/decode round-trips."""

from __future__ import annotations

import math
import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

PACK = Path(__file__).resolve().parents[1]
if str(PACK) not in sys.path:
    sys.path.insert(0, str(PACK))

from nodes.c2c_video.save_formats import (  # noqa: E402
    SaveVideoError,
    resolve_format,
)
from nodes.c2c_video.save_encode import (  # noqa: E402
    effective_naming,
    encode_video,
    pad_edge_replicate,
    prepare_audio,
    resolve_output_path,
)
from nodes.c2c_video.save_ocio import (  # noqa: E402
    ocio_available,
    transform_chunk,
)
from nodes.save_video import SaveVideoC2C  # noqa: E402

av = pytest.importorskip("av")


def gradient_batch(n=24, w=64, h=48, alpha=False):
    t = torch.linspace(0, 1, n).view(n, 1, 1)
    y = torch.linspace(0, 1, h).view(1, h, 1)
    x = torch.linspace(0, 1, w).view(1, 1, w)
    rgb = torch.stack([
        t.expand(n, h, w),
        y.expand(n, h, w),
        x.expand(n, h, w),
    ], dim=-1)
    if alpha:
        a = torch.linspace(0.2, 1.0, n).view(n, 1, 1, 1).expand(n, h, w, 1)
        rgb = torch.cat([rgb, a], dim=-1)
    return rgb.float()


def tone_audio(seconds=1.0, sr=48000, ch=2):
    n = int(seconds * sr)
    t = torch.arange(n, dtype=torch.float32) / sr
    wave = torch.sin(2 * math.pi * 440.0 * t)
    w = wave.unsqueeze(0).repeat(ch, 1).unsqueeze(0)
    return {"waveform": w, "sample_rate": sr}


@pytest.fixture
def fp_dirs(tmp_path, monkeypatch):
    out = tmp_path / "output"
    temp = tmp_path / "temp"
    user = tmp_path / "user"
    out.mkdir()
    temp.mkdir()
    (user / "default").mkdir(parents=True)
    fp = types.SimpleNamespace(
        get_output_directory=lambda: str(out),
        get_temp_directory=lambda: str(temp),
        get_user_directory=lambda: str(user),
        # same contract as core folder_paths.get_save_image_path: (output/<dir of prefix>, basename, counter,
        # <dir of prefix>, prefix)
        get_save_image_path=lambda prefix, odir, w, h: (
            os.path.join(odir, os.path.dirname(os.path.normpath(prefix))),
            os.path.basename(os.path.normpath(prefix)),
            1,
            os.path.dirname(os.path.normpath(prefix)),
            prefix,
        ),
    )
    monkeypatch.setitem(sys.modules, "folder_paths", fp)
    return out, temp, user


def _codec_ok(name: str) -> bool:
    try:
        av.Codec(name, "w")
        return True
    except Exception:
        return False


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    if mse <= 0:
        return 99.0
    return 10.0 * math.log10(1.0 / mse)


@pytest.fixture(autouse=True)
def _patch_folder_paths(fp_dirs):
    return fp_dirs


def test_pad_odd_size():
    img = gradient_batch(2, 65, 47)
    out, note = pad_edge_replicate(img)
    assert out.shape[1:3] == (48, 66)
    assert note is not None


def test_effective_naming_default_reads_setting(fp_dirs, monkeypatch):
    _out, _temp, user = fp_dirs
    settings = user / "default" / "comfy.settings.json"
    settings.write_text('{"c2c.saveVideo.naming": "Folder Version"}', encoding="utf-8")
    assert effective_naming("default", "") == "Folder Version"
    assert effective_naming("ComfyUI counter", "") == "ComfyUI counter"


def test_folder_version_refuses_overwrite(fp_dirs):
    out, _temp, _user = fp_dirs
    spec = resolve_format("MP4 H.264")
    path_info = resolve_output_path(
        spec=spec,
        naming="Folder Version",
        filename_prefix="shot",
        subfolder="v001",
        save_output=True,
        width=64,
        height=48,
    )
    open(path_info[0], "wb").close()
    with pytest.raises(SaveVideoError, match="already exists"):
        resolve_output_path(
            spec=spec,
            naming="Folder Version",
            filename_prefix="shot",
            subfolder="v001",
            save_output=True,
            width=64,
            height=48,
        )


@pytest.mark.skipif(not _codec_ok("libx264"), reason="libx264 unavailable")
def test_h264_roundtrip(fp_dirs):
    images = gradient_batch(24, 64, 48)
    spec = resolve_format("MP4 H.264")
    path_info = resolve_output_path(
        spec=spec, naming="ComfyUI counter", filename_prefix="test/h264",
        subfolder="", save_output=True, width=64, height=48,
    )
    result = encode_video(
        images, spec, path_info=path_info, fps=24.0, quality=80,
        colorspace_in="sRGB - Display", colorspace_out="same as input",
        audio=None, pad_note=None,
    )
    assert len(result["paths"]) == 1
    assert os.path.isfile(result["paths"][0])
    with av.open(result["paths"][0]) as c:
        frames = list(c.decode(c.streams.video[0]))
    assert len(frames) == 24
    assert frames[0].width == 64 and frames[0].height == 48
    assert frames[0].format.name == "yuv420p"
    # decoded as BT.709 limited, like the C2C reader: to_ndarray(format="rgb24") assumes BT.601 (gave 28 dB)
    dec, _ = _decode_rgb(result["paths"][0], False)
    assert _psnr(images[..., :3].numpy().astype(np.float64), dec) >= 40.0


@pytest.mark.skipif(not _codec_ok("libx264"), reason="libx264 unavailable")
def test_h264_odd_size_padded(fp_dirs):
    images = gradient_batch(8, 65, 47)
    images, note = pad_edge_replicate(images)
    spec = resolve_format("MP4 H.264")
    path_info = resolve_output_path(
        spec=spec, naming="ComfyUI counter", filename_prefix="test/odd",
        subfolder="", save_output=True, width=66, height=48,
    )
    result = encode_video(
        images, spec, path_info=path_info, fps=24.0, quality=80,
        colorspace_in="sRGB - Display", colorspace_out="same as input",
        audio=None, pad_note=note,
    )
    assert any("Padded" in w for w in result.get("warnings", []))
    with av.open(result["paths"][0]) as c:
        f = next(c.decode(c.streams.video[0]))
    assert f.width % 2 == 0 and f.height % 2 == 0


@pytest.mark.skipif(not _codec_ok("libx264"), reason="libx264 unavailable")
def test_audio_duration(fp_dirs):
    images = gradient_batch(24, 64, 48)
    audio = tone_audio(1.0)
    spec = resolve_format("MP4 H.264")
    path_info = resolve_output_path(
        spec=spec, naming="ComfyUI counter", filename_prefix="test/audio",
        subfolder="", save_output=True, width=64, height=48,
    )
    encode_video(
        images, spec, path_info=path_info, fps=24.0, quality=80,
        colorspace_in="sRGB - Display", colorspace_out="same as input",
        audio=audio, pad_note=None,
    )
    with av.open(path_info[0]) as c:
        ast = c.streams.audio[0]
        dur = float(ast.duration * ast.time_base) if ast.duration else 0.0
    video_dur = 24 / 24.0
    assert abs(dur - video_dur) <= 1 / 24.0 + 0.05


@pytest.mark.skipif(not _codec_ok("ffv1"), reason="ffv1 unavailable")
def test_ffv1_lossless(fp_dirs):
    images = gradient_batch(12, 64, 48)
    spec = resolve_format("MKV FFV1 (lossless)")
    path_info = resolve_output_path(
        spec=spec, naming="ComfyUI counter", filename_prefix="test/ffv1",
        subfolder="", save_output=True, width=64, height=48,
    )
    result = encode_video(
        images, spec, path_info=path_info, fps=24.0, quality=80,
        colorspace_in="sRGB - Display", colorspace_out="same as input",
        audio=None, pad_note=None,
    )
    with av.open(result["paths"][0]) as c:
        vs = c.streams.video[0]
        frames = list(c.decode(vs))
    assert len(frames) == 12
    assert frames[0].width == 64 and frames[0].height == 48
    assert "16" in frames[0].format.name or frames[0].format.name in ("yuv444p16le", "gbrap16le")


def test_unavailable_format_errors(monkeypatch):
    import nodes.c2c_video.save_formats as sf
    monkeypatch.setattr(sf, "_codec_writable", lambda name: name != "libx265")
    with pytest.raises(SaveVideoError, match="not available"):
        resolve_format("MP4 H.265 10-bit")


def test_prepare_audio_trim():
    audio = tone_audio(2.0)
    pcm, sr = prepare_audio(audio, n_frames=24, fps=24.0)
    want = int(round(24 / 24.0 * sr))
    assert pcm.shape[1] == want


def test_node_execute_ui(fp_dirs, monkeypatch):
    images = gradient_batch(4, 64, 48)
    if not _codec_ok("libx264"):
        pytest.skip("libx264 unavailable")
    out = SaveVideoC2C().execute(
        images=images,
        format="MP4 H.264",
        fps=24.0,
        filename_prefix="test/node",
        quality=80,
    )
    assert "ui" in out and "c2c_save_video" in out["ui"]
    assert out["ui"]["c2c_save_video"][0]["frames"] == 4
    assert out["result"][0]
    assert torch.equal(out["result"][1], images)


@pytest.mark.skipif(not ocio_available(), reason="PyOpenColorIO unavailable")
def test_ocio_roundtrip():
    rgb = np.linspace(0, 1, 64, dtype=np.float32).reshape(1, 8, 8, 1)
    rgb = np.concatenate([rgb, rgb * 0.5, rgb * 0.25], axis=-1)
    mid = transform_chunk(rgb, "sRGB - Display", "ACEScg")
    back = transform_chunk(mid, "ACEScg", "sRGB - Display")
    assert float(np.max(np.abs(back - rgb))) <= 1e-4


def test_interrupt_deletes_partial(fp_dirs, monkeypatch):
    if not _codec_ok("libx264"):
        pytest.skip("libx264 unavailable")
    images = gradient_batch(48, 64, 48)
    spec = resolve_format("MP4 H.264")
    path_info = resolve_output_path(
        spec=spec, naming="ComfyUI counter", filename_prefix="test/interrupt",
        subfolder="", save_output=True, width=64, height=48,
    )
    calls = {"n": 0}

    def interrupt():
        calls["n"] += 1
        if calls["n"] >= 2:
            raise InterruptedError("stopped")

    import nodes.c2c_video.save_encode as se
    monkeypatch.setattr(se, "_progress_and_interrupt", lambda: (None, interrupt))
    monkeypatch.setattr(se, "frames_that_fit", lambda *a, **k: 8)   # 6 chunks, so the check runs between them
    with pytest.raises(InterruptedError):
        encode_video(
            images, spec, path_info=path_info, fps=24.0, quality=80,
            colorspace_in="sRGB - Display", colorspace_out="same as input",
            audio=None, pad_note=None,
        )
    assert not os.path.isfile(path_info[0])


# ── every format: encode, decode back, measure (L7.56; budgets from docs/evidence/L7.56/format_roundtrip.json) ──

def _decode_rgb(path, want_alpha):
    """Decode the way the C2C reader does: accurate-rounding swscale + measured white level. swscale's default path
    converts yuv420p10le to RGB wrongly in this FFmpeg build (F2.10), which would fail every 10-bit 4:2:0 check."""
    from nodes.c2c_video.reader import _stamp, _to_rgb, _white_level
    out = []
    with av.open(path) as c:
        for f in c.decode(c.streams.video[0]):
            deep = any(s in f.format.name for s in ("10", "12", "16"))
            fmt = ("rgba64le" if deep else "rgba") if want_alpha else ("rgb48le" if deep else "rgb24")
            yuv = f.format.name.startswith("yuv")
            _stamp(f, "bt709" if yuv else None, "tv" if yuv else "pc")
            arr = _to_rgb(f, fmt).to_ndarray().astype(np.float64)
            out.append(arr / _white_level(f.format.name, "bt709" if yuv else None, "tv" if yuv else "pc", fmt))
    audio_s = None
    with av.open(path) as c:
        if c.streams.audio:
            a = c.streams.audio[0]
            audio_s = sum(fr.samples for fr in c.decode(a)) / a.rate
    return np.stack(out), audio_s


_BUDGET = {   # label: (min PSNR dB on the gradient, alpha expected, frame size)
    "MP4 H.264": (40.0, False, (128, 96)),
    "MP4 H.265 10-bit": (45.0, False, (128, 96)),
    "MOV ProRes 422": (58.0, False, (128, 96)),
    "MOV ProRes 422 HQ": (58.0, False, (128, 96)),
    "MOV ProRes 4444": (60.0, True, (128, 96)),
    "MOV ProRes 4444 XQ": (60.0, True, (128, 96)),
    "MOV DNxHR HQ": (45.0, False, (256, 144)),
    "MOV DNxHR HQX": (58.0, False, (256, 144)),
    "MOV DNxHR 444": (58.0, False, (256, 144)),
    "MKV FFV1 (lossless)": (95.0, True, (128, 96)),
}


@pytest.mark.parametrize("label", list(_BUDGET))
def test_format_roundtrip_quality_alpha_audio(label, fp_dirs):
    spec = resolve_format(label)
    min_psnr, alpha, (w, h) = _BUDGET[label]
    images = gradient_batch(24, w, h, alpha=alpha)
    info = resolve_output_path(spec=spec, naming="ComfyUI counter", filename_prefix="rt/" + label.split()[1],
                               subfolder="", save_output=True, width=w, height=h)
    r = encode_video(images, spec, path_info=info, fps=24.0, quality=80, colorspace_in="sRGB - Display",
                     colorspace_out="same as input", audio=tone_audio(1.0), pad_note=None)
    dec, audio_s = _decode_rgb(r["paths"][0], alpha)
    assert dec.shape[0] == 24 and dec.shape[1:3] == (h, w)
    assert dec.shape[-1] == (4 if alpha else 3)
    ref = images.numpy().astype(np.float64)[..., :dec.shape[-1]]
    assert _psnr(np.clip(ref, 0, 1), dec) >= min_psnr
    assert audio_s is not None and abs(audio_s - 1.0) <= 1.0 / 24.0


def test_webm_vp9_keeps_alpha_and_audio(fp_dirs):
    spec = resolve_format("WebM VP9")
    images = gradient_batch(12, 128, 96, alpha=True)
    info = resolve_output_path(spec=spec, naming="ComfyUI counter", filename_prefix="rt/vp9", subfolder="",
                               save_output=True, width=128, height=96)
    path = encode_video(images, spec, path_info=info, fps=24.0, quality=80, colorspace_in="sRGB - Display",
                        colorspace_out="same as input", audio=tone_audio(0.5), pad_note=None)["paths"][0]
    with av.open(path) as c:   # VP9 alpha is side data: only libvpx decodes it
        vs = c.streams.video[0]
        dec = av.CodecContext.create("libvpx-vp9", "r")
        frames = []
        for pkt in c.demux(vs):
            if pkt.size:
                frames += dec.decode(pkt)
        frames += dec.decode(None)
    assert len(frames) == 12 and frames[0].format.name == "yuva420p"
    for k in (0, 11):          # the fixture's alpha ramps over time: 0.2 on the first frame, 1.0 on the last
        p = frames[k].planes[3]
        a = np.frombuffer(bytes(p), np.uint8).reshape(p.height, p.line_size)[:, :p.width]
        want = 255 * (0.2 + 0.8 * k / 11)
        assert abs(float(a.mean()) - want) <= 4
    _dec, audio_s = _decode_rgb(path, False)
    assert abs(audio_s - 0.5) <= 1.0 / 24.0


@pytest.mark.parametrize("label,exact", [("PNG sequence 16-bit", 2 / 65535), ("EXR sequence (half)", 5e-4),
                                         ("EXR sequence (float)", 1e-6)])
def test_sequences_exact_to_format_precision(label, exact, fp_dirs):
    oiio = pytest.importorskip("OpenImageIO")
    spec = resolve_format(label)
    images = gradient_batch(6, 64, 48, alpha=True)
    info = resolve_output_path(spec=spec, naming="ComfyUI counter", filename_prefix="rt/seq", subfolder="",
                               save_output=True, width=64, height=48)
    r = encode_video(images, spec, path_info=info, fps=24.0, quality=80, colorspace_in="sRGB - Display",
                     colorspace_out="same as input", audio=tone_audio(0.25), pad_note=None)
    frames = sorted(p for p in r["paths"] if p.endswith((".png", ".exr")))
    assert len(frames) == 6 and any(p.endswith(".wav") for p in r["paths"])
    ref = images.numpy().astype(np.float64)
    if "EXR" in label:
        ref[..., :3] *= ref[..., 3:4]           # EXR is written premultiplied (Nuke's convention)
    for i, pth in enumerate(frames):
        cfg = oiio.ImageSpec()
        cfg.attribute("oiio:UnassociatedAlpha", 1)
        inp = oiio.ImageInput.open(pth, cfg)
        got = np.asarray(inp.read_image(format="float"), dtype=np.float64)
        inp.close()
        assert np.abs(got - ref[i]).max() <= exact


@pytest.mark.parametrize("label", ["GIF", "WebP (animated)"])
def test_gif_and_webp_frames(label, fp_dirs):
    from PIL import Image
    spec = resolve_format(label)
    alpha = label.startswith("WebP")
    images = gradient_batch(10, 64, 48, alpha=alpha)
    info = resolve_output_path(spec=spec, naming="ComfyUI counter", filename_prefix="rt/anim", subfolder="",
                               save_output=True, width=64, height=48)
    path = encode_video(images, spec, path_info=info, fps=12.0, quality=80, colorspace_in="sRGB - Display",
                        colorspace_out="same as input", audio=None, pad_note=None)["paths"][0]
    im = Image.open(path)
    assert getattr(im, "n_frames", 1) == 10 and im.size == (64, 48)
    if alpha:                  # alpha 0.2 on the first frame (the fixture ramps alpha over time)
        a = np.asarray(im.convert("RGBA"))[..., 3]
        assert abs(float(a.mean()) - 0.2 * 255) <= 4


def test_dnxhr_too_small_gets_a_plain_error(fp_dirs):
    spec = resolve_format("MOV DNxHR HQ")
    info = resolve_output_path(spec=spec, naming="ComfyUI counter", filename_prefix="rt/dnx_small", subfolder="",
                               save_output=True, width=128, height=96)
    with pytest.raises(SaveVideoError, match="at least 256x120"):
        encode_video(gradient_batch(4, 128, 96), spec, path_info=info, fps=24.0, quality=80,
                     colorspace_in="sRGB - Display", colorspace_out="same as input", audio=None, pad_note=None)


def test_ui_result_points_at_the_saved_file(fp_dirs):
    # L7.56 live: "subfolder" carried the file's whole relative path, so /view looked for <file>/<file>
    out, _temp, _user = fp_dirs
    spec = resolve_format("MP4 H.264")
    info = resolve_output_path(spec=spec, naming="ComfyUI counter", filename_prefix="live/shot", subfolder="",
                               save_output=True, width=64, height=48)
    r = encode_video(gradient_batch(4, 64, 48), spec, path_info=info, fps=24.0, quality=80,
                     colorspace_in="sRGB - Display", colorspace_out="same as input", audio=None, pad_note=None)
    assert os.path.isfile(os.path.join(str(out), r["subfolder"], r["filename"]))
    assert not r["subfolder"].endswith(r["filename"])
