"""L2.53 (owner A14): Save Video (C2C) gets the 4K pack's encode features - the ffmpeg pipe (same pixels as the PyAV
path), NVENC / CPU HEVC formats, exact NTSC rates, odd sizes padded (4:2:2 too), BT.709 tags, the workflow in the file
(readable by ComfyUI's front end), and image latents decoded in batches."""
from __future__ import annotations

import json
import os
import struct
import sys
import types
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
import torch

PACK = Path(__file__).resolve().parents[1]
if str(PACK) not in sys.path:
    sys.path.insert(0, str(PACK))

from tests.test_save_video import _decode_rgb, fp_dirs, gradient_batch, tone_audio, _patch_folder_paths  # noqa: E402,F401
from nodes.c2c_video import save_encode as SE  # noqa: E402
from nodes.c2c_video import save_ffmpeg as SF  # noqa: E402
from nodes.c2c_video.save_formats import FORMATS, SaveVideoError, available_formats, resolve_format  # noqa: E402

av = pytest.importorskip("av")
needs_ffmpeg = pytest.mark.skipif(SF.ffmpeg_for(FORMATS["MP4 H.264"]) is None, reason="ffmpeg is not installed")

WF = {"last_node_id": 3, "nodes": [{"id": 1, "type": "SaveVideoC2C", "widgets_values": ["a=b;c#d\\e\nf", "é"]}]}
PROMPT = {"1": {"class_type": "SaveVideoC2C", "inputs": {"fps": 24}}}


def _encode(label, images, *, fps=24.0, audio=None, metadata=None, prefix="t/x", cs_out="same as input"):
    spec = resolve_format(label)
    _, h, w, _ = images.shape
    info = SE.resolve_output_path(spec=spec, naming="ComfyUI counter", filename_prefix=prefix, subfolder="",
                                  save_output=True, width=w, height=h)
    return SE.encode_video(images, spec, path_info=info, fps=fps, quality=80, colorspace_in="sRGB - Display",
                           colorspace_out=cs_out, audio=audio, pad_note=None, metadata=metadata)["paths"][0]


# ── the front end's MP4 / MOV reader, re-done in Python (comfyui-frontend 1.53.10 parseIsobmffMetadata) ───────────
def _boxes(buf, start, end):
    i = start
    while i + 8 <= end:
        size, typ = struct.unpack(">I4s", buf[i:i + 8])
        hdr = 8
        if size == 1:
            size, hdr = struct.unpack(">Q", buf[i + 8:i + 16])[0], 16
        elif size == 0:
            size = end - i
        if size < hdr:
            return
        yield typ.decode("latin1"), i + hdr, i + size
        i += size


def _find(buf, start, end, typ):
    return next(((s, e) for t, s, e in _boxes(buf, start, end) if t == typ), None)


def frontend_metadata(path):
    buf = open(path, "rb").read()
    u = _find(buf, 0, len(buf), "udta")
    if not u:
        mv = _find(buf, 0, len(buf), "moov")
        u = mv and _find(buf, mv[0], mv[1], "udta")
    m = u and _find(buf, u[0], u[1], "meta")
    if not m:
        return {}
    k, il = _find(buf, m[0] + 4, m[1], "keys"), _find(buf, m[0] + 4, m[1], "ilst")
    if not k or not il:
        return {}
    keys, i = {}, k[0] + 8
    for n in range(1, struct.unpack(">I", buf[k[0] + 4:k[0] + 8])[0] + 1):
        size = struct.unpack(">I", buf[i:i + 4])[0]
        keys[n], i = buf[i + 8:i + size].decode("utf-8"), i + size
    out = {}
    for t, s, e in _boxes(buf, il[0], il[1]):
        name = keys.get(struct.unpack(">I", t.encode("latin1"))[0])
        d = _find(buf, s, e, "data")
        if name and d and name.lower() in ("workflow", "prompt"):
            out[name.lower()] = json.loads(buf[d[0] + 8:d[1]].decode("utf-8"))
    return out


# ── rates, quantization, options ───────────────────────────────────────────────────────────────────────────────
def test_exact_rate():
    f = SF.exact_rate
    assert f(23.976) == f(23.98) == Fraction(24000, 1001)
    assert f(29.97) == Fraction(30000, 1001) and f(59.94) == Fraction(60000, 1001)
    assert f(24) == 24 and f(25) == 25 and f(30) == 30 and f(12.5) == Fraction(25, 2)


def test_quantize_fast_equals_quantize_rgb():
    rng = np.random.default_rng(0)
    x = rng.random((3, 17, 19, 4), dtype=np.float32) * 1.2 - 0.1
    x[0, 0, :8, 0] = np.array([k + 0.5 for k in range(8)], np.float32) / 255     # exact halves: round half to even
    for bits in (8, 16):
        assert np.array_equal(SE.quantize_fast(x, bits), SE.quantize_rgb(x, bits))


def test_encoder_options_and_cli_mapping():
    nv = FORMATS["MP4 H.265 10-bit (NVENC)"]
    opts = SE.encoder_options(nv, 80)
    assert opts == {"profile": "main10", "rc": "vbr", "cq": "15", "b": "0", "preset": "p5"}
    assert SF._cli_options(opts) == ["-profile:v", "main10", "-rc", "vbr", "-cq", "15", "-b:v", "0", "-preset", "p5"]
    assert SE.encoder_options(FORMATS["MP4 H.264"], 80) == {"crf": "15", "preset": "medium"}


def test_ffmetadata_escaping():
    text = SF.ffmetadata_text({"workflow": WF, "k": "a=b"})
    assert text.startswith(";FFMETADATA1\nworkflow=") and "k=a\\=b\n" in text
    assert "\n" not in text.split("\n")[1].replace("\\\n", "")         # newlines inside values are escaped


def test_new_formats_are_appended_and_nvenc_needs_a_gpu(monkeypatch):
    labels = list(FORMATS)
    assert labels.index("MP4 H.265") > labels.index("EXR sequence (float)")
    assert labels[-5:] == ["MP4 H.265", "MP4 H.264 (NVENC)", "MP4 H.265 (NVENC)", "MP4 H.265 10-bit (NVENC)",
                           "MP4 AV1 (NVENC)"]
    import nodes.c2c_video.save_formats as sf
    monkeypatch.setattr(sf, "_NVIDIA", False)
    assert "MP4 H.264 (NVENC)  [unavailable]" in available_formats()
    with pytest.raises(SaveVideoError, match="NVIDIA GPU"):
        resolve_format("MP4 H.264 (NVENC)")


def test_ffmpeg_path_choice(monkeypatch):
    assert SF.ffmpeg_for(FORMATS["GIF"]) is None and SF.ffmpeg_for(FORMATS["EXR sequence (half)"]) is None
    monkeypatch.setenv("C2C_SAVE_FFMPEG", "0")
    assert SF.ffmpeg_for(FORMATS["MP4 H.264"]) is None


# ── same pixels on both paths ────────────────────────────────────────────────────────────────────────────────────
@needs_ffmpeg
@pytest.mark.parametrize("label,alpha", [("MKV FFV1 (lossless)", False), ("MKV FFV1 (lossless)", True),
                                         ("MOV ProRes 4444", True)])
def test_ffmpeg_path_gives_the_pyav_paths_pixels(label, alpha, fp_dirs, monkeypatch):
    rng = np.random.default_rng(1)
    images = torch.from_numpy(rng.random((5, 48, 64, 4 if alpha else 3), dtype=np.float32))
    if alpha:
        images[..., 3] = images[..., 3].clamp(0.05, 1.0)
    fast = _decode_rgb(_encode(label, images, prefix="pix/fast"), alpha)[0]
    monkeypatch.setenv("C2C_SAVE_FFMPEG", "0")
    slow = _decode_rgb(_encode(label, images, prefix="pix/pyav"), alpha)[0]
    if label.startswith("MKV FFV1"):
        assert np.array_equal(fast, slow)                        # lossless: identical frames, bit for bit
    else:
        assert np.abs(fast - slow).max() <= 2 / 1023             # ProRes: same input, encoder within a 10-bit step


# ── NTSC, tags, odd sizes, metadata ──────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("pyav", [False, True])
def test_ntsc_rate_tags_and_odd_size(pyav, fp_dirs, monkeypatch):
    if pyav:
        monkeypatch.setenv("C2C_SAVE_FFMPEG", "0")
    from nodes.save_video import SaveVideoC2C

    ui = SaveVideoC2C().execute(format="MP4 H.264", fps=23.976, filename_prefix=f"ntsc/{pyav}", quality=80,
                                images=gradient_batch(6, 65, 47))["ui"]["c2c_save_video"][0]
    out = fp_dirs[0]
    path = os.path.join(str(out), ui["subfolder"], ui["filename"])
    with av.open(path) as c:
        vs = c.streams.video[0]
        assert Fraction(vs.average_rate) == Fraction(24000, 1001)
        assert (vs.codec_context.width, vs.codec_context.height) == (66, 48)
        cc = vs.codec_context
        assert (int(cc.color_primaries), int(cc.color_trc), int(cc.colorspace), int(cc.color_range)) == (1, 13, 1, 1)
        assert sum(1 for _ in c.decode(vs)) == 6
    assert any("66" in w for w in ui["warnings"])


@pytest.mark.parametrize("pyav", [False, True])
def test_prores_422_odd_width_is_padded(pyav, fp_dirs, monkeypatch):
    if pyav:
        monkeypatch.setenv("C2C_SAVE_FFMPEG", "0")
    from nodes.save_video import SaveVideoC2C

    ui = SaveVideoC2C().execute(format="MOV ProRes 422 HQ", fps=24, filename_prefix=f"pr/{pyav}", quality=80,
                                images=gradient_batch(3, 65, 47))["ui"]["c2c_save_video"][0]
    path = os.path.join(str(fp_dirs[0]), ui["subfolder"], ui["filename"])
    with av.open(path) as c:
        cc = c.streams.video[0].codec_context
        assert (cc.width, cc.height) == (66, 47)                  # 4:2:2: only the width must be even


@pytest.mark.parametrize("pyav", [False, True])
@pytest.mark.parametrize("label", ["MP4 H.264", "MOV ProRes 422", "MKV FFV1 (lossless)"])
def test_workflow_round_trip(label, pyav, fp_dirs, monkeypatch):
    if pyav:
        monkeypatch.setenv("C2C_SAVE_FFMPEG", "0")
    from nodes.save_video import SaveVideoC2C

    node = SaveVideoC2C()
    ui = node.execute(format=label, fps=24, filename_prefix=f"meta/{pyav}", quality=80, images=gradient_batch(3),
                      audio=tone_audio(0.2), prompt=PROMPT, extra_pnginfo={"workflow": WF})["ui"]["c2c_save_video"][0]
    path = os.path.join(str(fp_dirs[0]), ui["subfolder"], ui["filename"])
    with av.open(path) as c:     # what ComfyUI's metadata readers see (Matroska stores tag names upper case)
        meta = {k.lower(): v for k, v in c.metadata.items()}
        assert json.loads(meta["workflow"]) == WF and json.loads(meta["prompt"]) == PROMPT
    if label != "MKV FFV1 (lossless)":                            # MP4 / MOV: the front end's own drop parser
        assert frontend_metadata(path) == {"workflow": WF, "prompt": PROMPT}
    off = node.execute(format=label, fps=24, filename_prefix=f"meta/off{pyav}", quality=80, images=gradient_batch(3),
                       prompt=PROMPT, extra_pnginfo={"workflow": WF}, save_metadata=False)["ui"]["c2c_save_video"][0]
    with av.open(os.path.join(str(fp_dirs[0]), off["subfolder"], off["filename"])) as c:
        assert "workflow" not in {k.lower() for k in c.metadata}


def test_hevc_8bit_cpu_with_hvc1_tag(fp_dirs):
    path = _encode("MP4 H.265", gradient_batch(4, 64, 48), prefix="hevc/x")
    with av.open(path) as c:
        vs = c.streams.video[0]
        assert vs.codec_context.name == "hevc" and vs.codec_tag == "hvc1"
        assert vs.codec_context.pix_fmt == "yuv420p" and sum(1 for _ in c.decode(vs)) == 4


def test_save_metadata_input_appended(fp_dirs):
    from nodes.save_video import SaveVideoC2C

    t = SaveVideoC2C.INPUT_TYPES()
    opt = list(t["optional"])
    assert opt[-1] == "save_metadata" and opt[-2] == "tile_size"
    assert t["optional"]["save_metadata"][1]["default"] is True
    assert t["hidden"] == {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"}
    assert list(t["required"]) == ["format", "fps", "filename_prefix", "quality"]


# ── image latents decoded in batches ────────────────────────────────────────────────────────────────────────────
def test_image_latent_decoded_in_batches_into_the_file(fp_dirs, monkeypatch):
    mm = types.ModuleType("comfy.model_management")
    mm.throw_exception_if_processing_interrupted = lambda: None
    monkeypatch.setitem(sys.modules, "comfy.model_management", mm)
    from nodes.save_video import SaveVideoC2C

    calls = []

    class VAE:
        def spacial_compression_decode(self):
            return 8

        def decode(self, z):                     # a flat grey frame of the value the latent carries in channel 0
            calls.append(int(z.shape[0]))
            k = z[:, 0, 0, 0].view(-1, 1, 1, 1)
            return k.expand(z.shape[0], z.shape[2] * 8, z.shape[3] * 8, 3).clone()

    n = 40
    z = torch.zeros(n, 4, 90, 160)               # 1280x720 frames: the 4K pack's rule gives 36 per batch
    z[:, 0] = torch.arange(n, dtype=torch.float32).view(n, 1, 1) / n
    out = SaveVideoC2C().execute(format="MKV FFV1 (lossless)", fps=24, filename_prefix="lat/img", quality=80,
                                 latent={"samples": z}, vae=VAE())
    ui = out["ui"]["c2c_save_video"][0]
    assert calls == [36, 4] and ui["frames"] == n
    assert ui["stream"]["method"].startswith("image latent")
    from nodes.c2c_video.reader import _stamp, _to_rgb
    got = []
    with av.open(os.path.join(str(fp_dirs[0]), ui["subfolder"], ui["filename"])) as c:
        for f in c.decode(c.streams.video[0]):
            assert (f.width, f.height) == (1280, 720)
            _stamp(f, "bt709", "tv")
            got.append(float(_to_rgb(f, "rgb48le").to_ndarray()[100, 100, 0]) / 65535)
    assert np.allclose(got, np.arange(n) / n, atol=1 / 2000)
