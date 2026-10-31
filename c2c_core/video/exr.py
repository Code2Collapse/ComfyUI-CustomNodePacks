"""Image headers, EXR parts / layers / channels, and reading one layer (owner A21, Rev 3 Part 3.7; ledger L9.89).

A ComfyUI IMAGE holds three or four channels; an EXR can hold dozens (beauty, AOVs, depth, normals, motion,
cryptomatte), in one part or many. ``inspect_image`` lists them the way Nuke shows them - a channel ``diffuse.R``
belongs to layer ``diffuse``; unprefixed R / G / B / A (and Y) are ``rgba``; Z is ``depth``; in a multipart file a part
whose channels carry no prefix is the layer named after the part - and ``read_layer`` reads exactly the chosen
channels. Nothing is clamped, nothing is colour-converted: data layers (depth, normals, position, motion,
cryptomatte) are values, not pictures.

OpenImageIO when installed (every format it reads); without it, OpenCV covers the 8 / 16-bit formats (no EXR, no DPX).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np

# channel suffix -> slot in RGB(A) order
_SLOT = {"r": 0, "red": 0, "x": 0, "u": 0, "g": 1, "green": 1, "y": 1, "v": 1, "b": 2, "blue": 2, "z": 2, "w": 3}
_ALPHA = {"a", "alpha"}
_MAIN = {"r", "g", "b", "a", "y"}               # unprefixed channels of the main picture


@dataclass(frozen=True)
class Layer:
    key: str                                   # what exr_layer names: rgba, diffuse, depth, N, normals ...
    part: int
    channels: tuple[str, ...]                  # the picture channels in R, G, B order (1 to 3 of them)
    alpha: str = ""                            # its alpha channel ("" = none)
    kind: str = "colour"                       # colour | alpha | depth | normal | position | motion | cryptomatte | data
    formats: tuple[str, ...] = ()

    @property
    def is_data(self) -> bool:
        return self.kind not in ("colour", "alpha")

    def to_dict(self) -> dict:
        return {"key": self.key, "part": self.part, "channels": list(self.channels), "alpha": self.alpha or None,
                "kind": self.kind, "data": self.is_data, "formats": list(self.formats)}


@dataclass(frozen=True)
class ImageInfo:
    path: str
    width: int
    height: int
    format: str                                # uint8 | uint16 | half | float ...
    parts: tuple[dict, ...]
    layers: tuple[Layer, ...]
    header: dict = field(default_factory=dict)
    fps: Fraction | None = None
    is_exr: bool = False
    deep: bool = False

    def layer(self, key: str) -> Layer | None:
        for lay in self.layers:
            if lay.key == key:
                return lay
        low = key.lower()
        for lay in self.layers:
            if lay.key.lower() == low:
                return lay
        return None

    def to_dict(self) -> dict:
        return {"file": os.path.basename(self.path), "width": self.width, "height": self.height,
                "format": self.format, "exr": self.is_exr, "deep": self.deep,
                "fps": f"{self.fps.numerator}/{self.fps.denominator}" if self.fps else None,
                "parts": list(self.parts), "layers": [lay.to_dict() for lay in self.layers]}


def have_oiio() -> bool:
    try:
        import OpenImageIO  # noqa: F401  # type: ignore[import-not-found]
        return True
    except ImportError:
        return False


def _split(name: str) -> tuple[str, str]:
    if "." in name:
        layer, chan = name.rsplit(".", 1)
        return layer, chan
    return "", name


def _kind(key: str, suffixes: list[str]) -> str:
    k = key.lower()
    sfx = {s.lower() for s in suffixes}
    if k.startswith("crypto"):
        return "cryptomatte"
    if k in ("depth", "z", "zdepth", "z_depth", "deep") or (sfx == {"z"} and k in ("", "depth")):
        return "depth"
    if k in ("n", "normal", "normals") or "normal" in k:
        return "normal"
    if k in ("p", "position", "world", "worldposition", "pworld", "pref") or "position" in k:
        return "position"
    if k in ("mv", "motion", "motionvector", "motionvectors", "velocity", "vector", "vectors", "forward",
             "backward", "flow") or "motion" in k:
        return "motion"
    if k in ("alpha", "matte", "mask") or sfx <= _ALPHA:
        return "alpha"
    if {"r", "g", "b"} <= sfx or {"red", "green", "blue"} <= sfx or k == "rgba":
        return "colour"
    return "data"


def _order(chans: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """(full name, suffix) pairs in R, G, B(, A) order; unknown suffixes keep the file's order after the known ones."""
    known = sorted((c for c in chans if c[1].lower() in _SLOT), key=lambda c: _SLOT[c[1].lower()])
    rest = [c for c in chans if c[1].lower() not in _SLOT]
    return known + rest


def _group(part: int, part_name: str, names: list[str], formats: list[str], multipart: bool) -> list[Layer]:
    groups: dict[str, list[tuple[str, str, str]]] = {}
    order: list[str] = []
    for name, fmt in zip(names, formats):
        prefix, chan = _split(name)
        if not prefix:
            if chan.lower() in _MAIN:
                prefix = "rgba"
            elif chan.upper() == "Z":
                prefix = "depth"
            else:
                prefix = chan
            if multipart and part_name and prefix in ("rgba", "depth"):
                prefix = part_name if prefix == "rgba" else f"{part_name}.depth"
        if prefix not in groups:
            groups[prefix] = []
            order.append(prefix)
        groups[prefix].append((name, chan, fmt))
    layers: list[Layer] = []
    main_alpha = next((n for n, c, _f in groups.get("rgba", []) if c.lower() in _ALPHA), "")
    for key in order:
        members = groups[key]
        alpha = next((n for n, c, _f in members if c.lower() in _ALPHA), "")
        pic = [(n, c) for n, c, _f in members if n != alpha]
        if not pic and alpha:                       # an alpha-only layer is a picture of its alpha
            pic, alpha = [(alpha, "A")], ""
        pic = _order(pic)[:3]
        kind = _kind(key, [c for _n, c in pic] + ([alpha.rsplit(".", 1)[-1]] if alpha else []))
        if not alpha and kind == "colour" and key != "rgba":
            alpha = main_alpha                      # an RGB AOV is matted by the beauty's alpha, as in Nuke
        fmts = tuple(sorted({f for _n, _c, f in members}))
        layers.append(Layer(key=key, part=part, channels=tuple(n for n, _c in pic), alpha=alpha, kind=kind,
                            formats=fmts))
    return layers


def _json_value(v):
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, bytes):
        return f"<{len(v)} bytes>"
    if isinstance(v, np.ndarray):
        v = v.tolist()
    if isinstance(v, (tuple, list)):
        return [_json_value(x) for x in v][:64]
    return str(v)[:500]


def _header(spec) -> dict:
    out: dict = {}
    for attr in list(getattr(spec, "extra_attribs", []))[:300]:
        try:
            out[str(attr.name)] = _json_value(attr.value)
        except Exception:  # noqa: BLE001 - an attribute OIIO cannot convert is skipped, never fatal
            continue
    return out


def _fps_from(header: dict) -> Fraction | None:
    for key in ("FramesPerSecond", "dpx:FrameRate", "oiio:Movie:FramesPerSecond"):
        v = header.get(key)
        try:
            if isinstance(v, (list, tuple)) and len(v) == 2 and int(v[1]):
                f = Fraction(int(v[0]), int(v[1]))
            elif v is not None:
                f = Fraction(float(v)).limit_denominator(1001000)
            else:
                continue
        except (TypeError, ValueError, ZeroDivisionError):
            continue
        if 0 < f <= 1000:
            return f
    return None


def inspect_image(path: str) -> ImageInfo:
    """Size, parts, layers and header of one image file (one header read per part; no pixels)."""
    if not have_oiio():
        return _inspect_cv2(path)
    import OpenImageIO as oiio  # type: ignore[import-not-found]

    inp = oiio.ImageInput.open(path)
    if inp is None:
        raise RuntimeError(f"cannot open {os.path.basename(path)}: {oiio.geterror()}")
    try:
        is_exr = inp.format_name() == "openexr"
        parts: list[dict] = []
        layers: list[Layer] = []
        specs = []
        i = 0
        while inp.seek_subimage(i, 0):
            specs.append(inp.spec())
            i += 1
        if not specs:
            specs = [inp.spec()]
        multipart = len(specs) > 1
        for p, spec in enumerate(specs):
            names = list(spec.channelnames)
            fmts = [str(f) for f in spec.channelformats] if len(spec.channelformats) == len(names) \
                else [str(spec.format)] * len(names)
            pname = str(spec.getattribute("name") or "") if multipart else ""
            parts.append({"index": p, "name": pname or None, "width": int(spec.width), "height": int(spec.height),
                          "format": str(spec.format), "channels": names, "deep": bool(spec.deep)})
            layers.extend(_group(p, pname or (f"part{p}" if multipart else ""), names, fmts, multipart))
        seen: dict[str, int] = {}
        unique: list[Layer] = []
        for lay in layers:                          # a key used by two parts gets "@part"
            n = seen.get(lay.key, 0)
            seen[lay.key] = n + 1
            unique.append(lay if n == 0 else Layer(**{**lay.__dict__, "key": f"{lay.key}@{lay.part}"}))
        first = specs[0]
        header = _header(first)
        return ImageInfo(path=path, width=int(first.width), height=int(first.height), format=str(first.format),
                         parts=tuple(parts), layers=tuple(unique), header=header, fps=_fps_from(header),
                         is_exr=is_exr, deep=any(bool(s.deep) for s in specs))
    finally:
        inp.close()


def _inspect_cv2(path: str) -> ImageInfo:
    import cv2

    arr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise RuntimeError(f"cannot read {os.path.basename(path)} (install OpenImageIO for EXR / DPX)")
    h, w = arr.shape[:2]
    n = 1 if arr.ndim == 2 else arr.shape[2]
    names = {1: ["Y"], 2: ["Y", "A"], 3: ["R", "G", "B"], 4: ["R", "G", "B", "A"]}.get(n, ["R", "G", "B"])
    fmt = {np.uint8: "uint8", np.uint16: "uint16", np.float32: "float"}.get(arr.dtype.type, str(arr.dtype))
    layers = tuple(_group(0, "", names, [fmt] * len(names), False))
    return ImageInfo(path=path, width=int(w), height=int(h), format=fmt,
                     parts=({"index": 0, "name": None, "width": int(w), "height": int(h), "format": fmt,
                             "channels": names, "deep": False},),
                     layers=layers, header={}, fps=None, is_exr=False, deep=False)


def pick_layer(info: ImageInfo, wanted: str = "auto") -> Layer:
    """``auto``: the main picture (rgba of the first part), else the first colour layer, else the first layer."""
    w = str(wanted or "auto").strip()
    if w.lower() in ("", "auto"):
        for lay in info.layers:
            if lay.key == "rgba" and lay.part == 0:
                return lay
        for lay in info.layers:
            if lay.kind == "colour":
                return lay
        if info.layers:
            return info.layers[0]
        raise ValueError(f"{os.path.basename(info.path)} has no channels")
    lay = info.layer(w)
    if lay is None:
        keys = ", ".join(x.key for x in info.layers)
        raise ValueError(f"layer {w!r} is not in {os.path.basename(info.path)}; its layers: {keys}")
    return lay


def read_layer(path: str, part: int, channels: tuple[str, ...], alpha: str, fmt: str, want_alpha: bool) -> np.ndarray:
    """[H, W, 3 or 4] of the named channels: 1 channel -> grey RGB, 2 -> (X, Y, 0), 3 -> RGB; plus the alpha channel
    when ``want_alpha`` (an opaque one if the layer has none). Values exactly as stored (float stays float)."""
    import OpenImageIO as oiio  # type: ignore[import-not-found]

    read_fmt = {"uint8": "uint8", "uint16": "uint16", "float16": "half", "float32": "float"}[fmt]
    np_out = {"uint8": np.uint8, "uint16": np.uint16, "float16": np.float16, "float32": np.float32}[fmt]
    inp = oiio.ImageInput.open(path)
    if inp is None:
        raise RuntimeError(f"cannot open image {path!r}: {oiio.geterror()}")
    try:
        if not inp.seek_subimage(int(part), 0):
            raise ValueError(f"{os.path.basename(path)} has no part {part} (the sequence's first frame had it)")
        spec = inp.spec()
        names = list(spec.channelnames)
        wanted = list(channels) + ([alpha] if (want_alpha and alpha) else [])
        missing = [c for c in wanted if c not in names]
        if missing:
            raise ValueError(f"{os.path.basename(path)} has no channel {', '.join(missing)} "
                             f"(its channels: {', '.join(names[:40])})")
        idx = [names.index(c) for c in wanted]
        lo, hi = min(idx), max(idx) + 1
        arr = inp.read_image(int(part), 0, lo, hi, read_fmt)
        if arr is None:
            raise RuntimeError(f"cannot read image {path!r}: {inp.geterror()}")
        arr = np.asarray(arr).reshape(spec.height, spec.width, hi - lo)
    finally:
        inp.close()
    pic = [arr[..., i - lo] for i in idx[:len(channels)]]
    if len(pic) == 1:
        pic = pic * 3
    elif len(pic) == 2:
        pic = pic + [np.zeros_like(pic[0])]
    planes = pic[:3]
    if want_alpha:
        if alpha:
            planes.append(arr[..., idx[-1] - lo])
        else:
            one = {np.uint8: 255, np.uint16: 65535}.get(np_out, 1.0)
            planes.append(np.full(planes[0].shape, one, dtype=planes[0].dtype))
    return np.ascontiguousarray(np.stack(planes, axis=-1)).astype(np_out, copy=False)
