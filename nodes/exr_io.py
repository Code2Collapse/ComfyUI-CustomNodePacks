"""
EXR I/O nodes (MEC):
  - LoadEXRMEC: Load an EXR file as IMAGE [B,H,W,3] in scene-linear.
  - SaveEXRMEC: Save IMAGE as EXR (16-bit half by default).

Backend priority:
  1. ``OpenImageIO`` — the production path. The only backend that surfaces a
     VFX EXR honestly: named AOV channels, compression codec, metadata, and no
     clipping of HDR values. This is the library Nuke and Katana use for IO.
  2. ``OpenEXR`` + ``Imath`` — RGB only.
  3. ``imageio`` — RGB only; may fall back to a 16-bit TIFF written to a
     ``_fallback.tif`` path (NOT the .exr path) with a warning logged.

Every downgrade is logged with its reason, and a save that was asked for AOVs,
metadata or a specific compression will RAISE rather than quietly write a
lesser file through a backend that cannot honour the request.

Headless and read-only safe; never imports unavailable libs at module
import time. All paths use forward slashes in the info JSON.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any

import numpy as np
import torch

logger = logging.getLogger("MEC.EXRIO")


def _try_oiio_load(path: str) -> tuple[np.ndarray, dict]:
    """OpenImageIO read — the production path, tried first.

    Why OIIO ahead of OpenEXR/imageio: it is the library Nuke and Katana use
    for IO, and it is the only one of the three that surfaces a VFX EXR
    honestly. Measured on an 11-channel test file, OIIO returned every named
    channel (R,G,B,A, N.x/y/z, depth.Z, diffuse.R/G/B) plus the DWAA
    compression setting and the metadata; the OpenCV path that the common
    reference packs use could not even OPEN the file in this environment
    (OpenEXR codec disabled unless OPENCV_IO_ENABLE_OPENEXR is set before cv2
    imports), and when enabled it collapses everything to RGB(A).

    RGB is returned for the IMAGE output, but the full channel list and the
    file metadata go out in `info` so a caller can see what else is in there
    rather than silently losing it.
    """
    import OpenImageIO as oiio  # type: ignore[import-not-found]
    inp = oiio.ImageInput.open(path)
    if inp is None:
        raise RuntimeError(f"OpenImageIO could not open {path!r}: {oiio.geterror()}")
    try:
        spec = inp.spec()
        names = list(spec.channelnames)
        # Read as float32 regardless of the file's half/float storage — no
        # clipping, so HDR values above 1.0 survive. Clamping on load is what
        # destroys speculars and emissives before you ever see them.
        pixels = inp.read_image(format="float32")
        if pixels is None:
            raise RuntimeError(f"OpenImageIO read failed for {path!r}: {inp.geterror()}")
        arr = np.asarray(pixels, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr[..., None]
        arr = arr.reshape(spec.height, spec.width, spec.nchannels)
        # Prefer explicitly named R/G/B; fall back to the first three planes
        # for files that use bare channel names.
        idx = [names.index(c) for c in ("R", "G", "B") if c in names]
        rgb = arr[..., idx] if len(idx) == 3 else arr[..., :3]
        if rgb.shape[-1] == 1:
            rgb = np.repeat(rgb, 3, axis=-1)
        meta = {}
        for p in spec.extra_attribs:
            try:
                meta[str(p.name)] = str(p.value)
            except Exception:  # noqa: BLE001 — a odd attrib must not fail the read
                pass
        info = {
            "backend": "OpenImageIO",
            "width": int(spec.width), "height": int(spec.height),
            "channels": names, "n_channels": int(spec.nchannels),
            "compression": str(spec.getattribute("compression") or ""),
            "metadata": meta,
        }
        return rgb, info
    finally:
        inp.close()


def _try_openexr_load(path: str) -> tuple[np.ndarray, dict]:
    import OpenEXR  # type: ignore[import-not-found]
    import Imath  # type: ignore[import-not-found]
    f = OpenEXR.InputFile(path)
    try:
        h = f.header()
        dw = h["dataWindow"]
        w = dw.max.x - dw.min.x + 1
        height = dw.max.y - dw.min.y + 1
        pt = Imath.PixelType(Imath.PixelType.FLOAT)
        chans = []
        for c in ("R", "G", "B"):
            if c not in h["channels"]:
                raise ValueError(f"EXR {path!r} missing channel {c}")
            buf = f.channel(c, pt)
            arr = np.frombuffer(buf, dtype=np.float32).reshape(height, w)
            chans.append(arr)
        rgb = np.stack(chans, axis=-1)
    finally:
        f.close()
    info = {"backend": "OpenEXR", "width": w, "height": height}
    return rgb, info


def _try_imageio_load(path: str) -> tuple[np.ndarray, dict]:
    import imageio.v3 as iio  # type: ignore[import-not-found]
    arr = iio.imread(path)  # may pick freeimage if installed
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    arr = arr[..., :3].astype(np.float32)
    return arr, {"backend": "imageio"}


class LoadEXRMEC:
    """Load an EXR file as scene-linear IMAGE [1,H,W,3] float32."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"file_path": ("STRING", {"default": ""})},
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "info_json")
    FUNCTION = "load"
    CATEGORY = "MaskEditControl/IO"
    DESCRIPTION = "Load EXR as scene-linear IMAGE. Tries OpenEXR → imageio."

    def load(self, file_path: str):
        if not file_path or not os.path.isfile(file_path):
            raise FileNotFoundError(f"EXR not found: {file_path!r}")
        info: dict[str, Any]
        # OIIO first (see _try_oiio_load for why), then OpenEXR, then imageio.
        # Each fallback is LOGGED with the reason — a silent downgrade here
        # means you find out your AOVs were dropped at delivery, not at load.
        _errs = []
        for _fn, _name in ((_try_oiio_load, "OpenImageIO"),
                           (_try_openexr_load, "OpenEXR"),
                           (_try_imageio_load, "imageio")):
            try:
                rgb, info = _fn(file_path)
                if _errs:
                    info["fell_back_from"] = _errs
                    logger.warning("[MEC] EXR load used %s after: %s", _name, "; ".join(_errs))
                break
            except Exception as exc:  # noqa: BLE001
                _errs.append(f"{_name}: {type(exc).__name__}: {exc}")
        else:
            raise RuntimeError(
                "Could not read " + repr(file_path) + " with any backend.\n  "
                + "\n  ".join(_errs)
                + "\nInstall OpenImageIO into the ComfyUI python for full "
                  "multi-channel EXR support: pip install OpenImageIO"
            )
        info["file"] = os.path.basename(file_path)
        t = torch.from_numpy(np.ascontiguousarray(rgb)).unsqueeze(0)
        return (t, json.dumps(info, indent=2))


#: EXR compressions worth exposing, in the order a compositor thinks about them.
#: dwaa/dwab are lossy but are what most facilities ship for non-data passes —
#: 5-10x smaller than zip at visually lossless settings. piz is the lossless
#: choice for grainy plates; zips is lossless and fastest to read a scanline at
#: a time; none is for maximum-compatibility handoff.
EXR_COMPRESSIONS = ("zips", "zip", "piz", "dwaa", "dwab", "rle", "none")


def _try_oiio_save(path: str, rgb: np.ndarray, half: bool,
                   compression: str = "zips",
                   extra_channels: "dict[str, np.ndarray] | None" = None,
                   metadata: "dict[str, str] | None" = None) -> dict:
    """OpenImageIO write — the production path.

    Three things this does that no reference pack does:

    * writes NAMED EXTRA CHANNELS (AOVs) alongside RGB, so a pass can round-trip
      through ComfyUI instead of being flattened to RGB on the way out;
    * exposes the compression codec, because shipping a grainy plate as dwaa or
      a data pass as anything lossy are both real mistakes;
    * round-trips metadata, so camera/lens/timecode/comment survive.

    half vs float is a real decision, not a default: half (16-bit) is the norm
    for colour passes and halves the file, but data passes — depth, position,
    normals, motion vectors — MUST be full float or they quantise visibly.
    """
    import OpenImageIO as oiio  # type: ignore[import-not-found]
    h, w, _ = rgb.shape
    names = ["R", "G", "B"]
    planes = [rgb[..., 0], rgb[..., 1], rgb[..., 2]]
    for cname, arr in (extra_channels or {}).items():
        a = np.asarray(arr, dtype=np.float32)
        if a.ndim == 3 and a.shape[-1] == 1:
            a = a[..., 0]
        if a.shape[:2] != (h, w):
            raise ValueError(
                f"extra channel {cname!r} is {a.shape[:2]}, expected {(h, w)} to "
                f"match RGB — EXR channels must share one data window"
            )
        names.append(str(cname))
        planes.append(a)

    stacked = np.ascontiguousarray(np.stack(planes, axis=-1).astype(np.float32))
    fmt = "half" if half else "float"
    spec = oiio.ImageSpec(int(w), int(h), len(names), fmt)
    spec.channelnames = tuple(names)
    comp = str(compression or "zips").lower()
    if comp not in EXR_COMPRESSIONS:
        raise ValueError(
            f"unknown EXR compression {compression!r}; expected one of "
            f"{', '.join(EXR_COMPRESSIONS)}"
        )
    spec.attribute("compression", comp)
    for k, v in (metadata or {}).items():
        spec.attribute(str(k), str(v))

    out = oiio.ImageOutput.create(path)
    if out is None:
        raise RuntimeError(f"OpenImageIO cannot write {path!r}: {oiio.geterror()}")
    if not out.open(path, spec):
        raise RuntimeError(f"OpenImageIO open failed for {path!r}: {out.geterror()}")
    try:
        if not out.write_image(stacked):
            raise RuntimeError(f"OpenImageIO write failed for {path!r}: {out.geterror()}")
    finally:
        out.close()
    return {
        "backend": "OpenImageIO", "width": int(w), "height": int(h),
        "channels": names, "n_channels": len(names),
        "bit_depth": fmt, "compression": comp,
        "metadata": dict(metadata or {}),
    }


def _try_openexr_save(path: str, rgb: np.ndarray, half: bool) -> dict:
    import OpenEXR  # type: ignore[import-not-found]
    import Imath  # type: ignore[import-not-found]
    h, w, _ = rgb.shape
    pt = Imath.PixelType(Imath.PixelType.HALF if half else Imath.PixelType.FLOAT)
    header = OpenEXR.Header(w, h)
    header["channels"] = {c: Imath.Channel(pt) for c in ("R", "G", "B")}
    out = OpenEXR.OutputFile(path, header)
    try:
        dtype = np.float16 if half else np.float32
        bufs = {c: rgb[..., i].astype(dtype).tobytes() for i, c in enumerate(("R", "G", "B"))}
        out.writePixels(bufs)
    finally:
        out.close()
    return {"backend": "OpenEXR", "half": half}


def _try_imageio_save(path: str, rgb: np.ndarray) -> dict:
    import imageio.v3 as iio  # type: ignore[import-not-found]
    try:
        iio.imwrite(path, rgb.astype(np.float32))
        return {"backend": "imageio"}
    except Exception as exc:  # noqa: BLE001
        # Last-ditch: write a 16-bit TIFF at the same stem and warn.
        alt = os.path.splitext(path)[0] + "_fallback.tif"
        scaled = (np.clip(rgb, 0.0, 65.535) * 1000.0).astype(np.uint16)
        iio.imwrite(alt, scaled)
        logger.warning(
            "[MEC] EXR write failed (%s); wrote 16-bit TIFF fallback to %s", exc, alt,
        )
        return {"backend": "tiff_fallback", "fallback_path": alt}


class SaveEXRMEC:
    """Save an IMAGE batch to EXR (one file per frame; index suffix appended)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "file_path": ("STRING", {
                    "default": "",
                    "tooltip": "Absolute output path. Batches add _0001, _0002 suffixes.",
                }),
            },
            "optional": {
                "half_float": ("BOOLEAN", {"default": True, "tooltip": "16-bit half vs 32-bit float. half halves the file and is the norm for COLOUR passes. Turn it OFF for DATA passes - depth, position, normals, motion vectors - which quantise visibly at half precision."}),
                "compression": (list(EXR_COMPRESSIONS), {"default": "zips", "tooltip": "EXR codec.\n\nzips = lossless, scanline, fastest to read a single line - a safe default.\nzip = lossless, 16-scanline blocks, slightly smaller than zips.\npiz = lossless and the best choice for GRAINY plates; wavelet-based, handles noise where zip bloats.\ndwaa / dwab = LOSSY but what most facilities ship for non-data passes: 5-10x smaller than zip at visually lossless settings. Never use on a data pass.\nrle = lossless, only good for large flat areas.\nnone = uncompressed, for maximum-compatibility handoff."}),
                "metadata_json": ("STRING", {"default": "", "multiline": True, "tooltip": "Optional JSON object of EXR attributes to embed, e.g. {\"comment\": \"shot_0010\", \"camera\": \"ARRI\"}. Round-trips through LoadEXRMEC."}),
                "aov_alpha": ("MASK", {"tooltip": "Optional MASK written as a named ALPHA channel alongside RGB, instead of being discarded."}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("info_json",)
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "MaskEditControl/IO"
    DESCRIPTION = "Save IMAGE batch as EXR(s)."

    def save(self, image: torch.Tensor, file_path: str, half_float: bool = True,
             compression: str = "zips", metadata_json: str = "",
             aov_alpha=None):
        if not file_path:
            raise ValueError("file_path is required.")
        os.makedirs(os.path.dirname(os.path.abspath(file_path)) or ".", exist_ok=True)
        B = int(image.shape[0])
        results = []
        meta = {}
        if metadata_json and metadata_json.strip():
            try:
                _m = json.loads(metadata_json)
                if not isinstance(_m, dict):
                    raise ValueError("metadata_json must be a JSON object")
                meta = {str(k): str(v) for k, v in _m.items()}
            except Exception as exc:
                raise ValueError(
                    f"metadata_json is not valid JSON: {exc}. Expected an object like "
                    f'{{"comment": "shot_0010"}}'
                ) from exc
        stem, ext = os.path.splitext(file_path)
        if not ext:
            ext = ".exr"
        for i in range(B):
            out_path = file_path if B == 1 else f"{stem}_{i + 1:04d}{ext}"
            rgb = image[i].cpu().numpy().astype(np.float32)
            extra = {}
            if aov_alpha is not None:
                _a = aov_alpha
                _a = _a[i] if getattr(_a, "ndim", 0) == 3 and _a.shape[0] > i else _a
                extra["A"] = (_a.cpu().numpy() if hasattr(_a, "cpu") else np.asarray(_a)).astype(np.float32)
            # OIIO first, then the legacy backends. Each fallback is LOGGED
            # with its reason and the resulting file is described in `info`,
            # so a downgrade that silently drops channels or compression is
            # visible in the node output rather than only in a log file.
            _errs = []
            info = None
            try:
                info = _try_oiio_save(out_path, rgb, half_float,
                                      compression=compression,
                                      extra_channels=extra or None,
                                      metadata=meta)
            except Exception as exc:  # noqa: BLE001
                _errs.append(f"OpenImageIO: {type(exc).__name__}: {exc}")
                if extra or meta or compression != "zips":
                    # The legacy backends cannot honour any of these, so
                    # falling through would silently write a lesser file.
                    raise RuntimeError(
                        "OpenImageIO failed and the legacy EXR backends cannot "
                        "write named AOV channels, metadata or a chosen "
                        "compression, so falling through would silently write a "
                        "lesser file:\n  " + "\n  ".join(_errs)
                        + "\nFix: pip install OpenImageIO into the ComfyUI python."
                    ) from exc
                for _fn, _nm in ((_try_openexr_save, "OpenEXR"),):
                    try:
                        info = _fn(out_path, rgb, half_float); break
                    except Exception as e2:  # noqa: BLE001
                        _errs.append(f"{_nm}: {type(e2).__name__}: {e2}")
                if info is None:
                    info = _try_imageio_save(out_path, rgb)
                info["fell_back_from"] = _errs
                logger.warning("[MEC] EXR save fell back: %s", "; ".join(_errs))
            info["path"] = out_path.replace("\\", "/")
            results.append(info)
        return (json.dumps({"frames": results}, indent=2),)


NODE_CLASS_MAPPINGS = {"LoadEXRMEC": LoadEXRMEC, "SaveEXRMEC": SaveEXRMEC}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadEXRMEC": "Load EXR (MEC)",
    "SaveEXRMEC": "Save EXR (MEC)",
}
