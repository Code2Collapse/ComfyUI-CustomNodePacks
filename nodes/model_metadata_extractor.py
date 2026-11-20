"""
ModelMetadataExtractorMEC – Read model file metadata without loading weights.

Supports:
  - ``.safetensors``: spec-compliant header parse — first 8 bytes are
    a little-endian uint64 header length, followed by a UTF-8 JSON
    blob describing tensors and a ``__metadata__`` block. Constant-time;
    no torch import needed.
  - ``.ckpt`` / ``.pt`` / ``.pth``: these are pickle archives. We do
    NOT execute them. We use ``zipfile`` (modern PyTorch save format
    is a zip) to list members and extract sizes; if it's an old
    legacy pickle we fall back to ``pickletools.dis`` against the
    raw bytes to enumerate top-level opcodes safely (no globals
    are imported, no objects constructed).

Outputs JSON metadata + the model's training type (if recognized) +
a SHA256 (head + tail) fingerprint suitable for cache keys.

Security: never imports unknown modules; never unpickles untrusted
data. The ``pickletools.dis`` text output is bounded to first 1 MB.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import pickletools
import struct
import zipfile

from ._is_changed_util import hash_args_and_kwargs

logger = logging.getLogger("MEC.ModelMetadataExtractor")


_FINGERPRINT_BYTES = 1024 * 1024  # 1 MB head, 1 MB tail


def _quick_fingerprint(path: str) -> str:
    """SHA256 of (size || head || tail) — fast & collision-resistant for caches."""
    size = os.path.getsize(path)
    h = hashlib.sha256()
    h.update(struct.pack("<Q", size))
    with open(path, "rb") as fh:
        h.update(fh.read(_FINGERPRINT_BYTES))
        if size > _FINGERPRINT_BYTES * 2:
            fh.seek(-_FINGERPRINT_BYTES, os.SEEK_END)
            h.update(fh.read(_FINGERPRINT_BYTES))
    return h.hexdigest()


def _read_safetensors_header(path: str) -> dict:
    with open(path, "rb") as fh:
        hdr_len_b = fh.read(8)
        if len(hdr_len_b) < 8:
            raise ValueError("Truncated safetensors header.")
        hdr_len = struct.unpack("<Q", hdr_len_b)[0]
        if hdr_len <= 0 or hdr_len > 100 * 1024 * 1024:  # 100 MB sanity cap
            raise ValueError(f"Implausible safetensors header length: {hdr_len}")
        hdr = fh.read(hdr_len)
    parsed = json.loads(hdr.decode("utf-8"))
    custom = parsed.pop("__metadata__", None)
    tensors = []
    total_params = 0
    for name, spec in parsed.items():
        shape = spec.get("shape", [])
        dtype = spec.get("dtype", "?")
        n = 1
        for d in shape:
            n *= int(d)
        total_params += n
        tensors.append({
            "name": name,
            "dtype": dtype,
            "shape": shape,
            "params": n,
        })
    return {
        "format": "safetensors",
        "tensor_count": len(tensors),
        "total_params": total_params,
        "metadata": custom or {},
        "tensors": tensors[:64],  # cap output size
        "tensor_truncated": len(tensors) > 64,
    }


def _read_torch_zip(path: str) -> dict:
    """Modern .pt/.pth/.ckpt are zip archives. List members, no unpickle."""
    info: dict = {"format": "torch_zip", "members": []}
    with zipfile.ZipFile(path, "r") as zf:
        for zi in zf.infolist():
            info["members"].append({
                "name": zi.filename,
                "size": zi.file_size,
                "compressed": zi.compress_size,
            })
        # The 'data.pkl' member is the structure descriptor — we can
        # safely run pickletools.dis on it (no execution).
        for cand in ("data.pkl", "archive/data.pkl"):
            try:
                with zf.open(cand) as fh:
                    data = fh.read(1_000_000)  # cap 1 MB
                # Disassemble to a bounded text trace
                import io as _io
                buf = _io.StringIO()
                # MANUAL bug-fix (Apr 2026): narrowed from broad Exception to
                # the actual error types pickletools.dis raises (ValueError,
                # IndexError, OSError, struct.error, EOFError). Truly
                # unexpected exceptions now propagate.
                try:
                    pickletools.dis(data, buf)
                    info["pickle_disassembly_excerpt"] = buf.getvalue()[:8000]
                except (ValueError, IndexError, OSError, EOFError, struct.error) as exc:
                    info["pickle_disassembly_excerpt"] = f"<dis failed: {exc}>"
                break
            except KeyError:
                continue
    info["member_count"] = len(info["members"])
    info["total_uncompressed"] = sum(m["size"] for m in info["members"])
    return info


def _read_legacy_pickle(path: str) -> dict:
    """Old-style raw-pickle .pt — disassemble safely, no unpickling."""
    with open(path, "rb") as fh:
        data = fh.read(1_000_000)
    import io as _io
    buf = _io.StringIO()
    try:
        pickletools.dis(data, buf)
    except (ValueError, IndexError, OSError, EOFError, struct.error) as exc:
        # MANUAL bug-fix (Apr 2026): narrowed from broad Exception.
        return {"format": "legacy_pickle_unreadable", "error": str(exc)}
    return {
        "format": "legacy_pickle",
        "pickle_disassembly_excerpt": buf.getvalue()[:8000],
    }


def _detect_model_kind(meta: dict) -> str:
    """Heuristic identification of common Stable Diffusion model families."""
    tensors = meta.get("tensors", [])
    names = {t["name"] for t in tensors}
    if any(n.startswith("first_stage_model.") for n in names) and any(
        n.startswith("model.diffusion_model.") for n in names
    ):
        return "stable_diffusion_checkpoint"
    if any("encoder.down" in n and "first_stage_model" not in n for n in names):
        return "vae"
    if any(n.startswith("lora_") or ".lora_up" in n or ".lora_down" in n for n in names):
        return "lora"
    if any(n.startswith("text_model.") or n.startswith("transformer.") for n in names):
        return "text_encoder"
    if meta.get("format") == "safetensors" and tensors:
        return "safetensors_other"
    return "unknown"


# Phase 3b v2 - Feature A: license + lineage detection.
# Civitai / kohya / sd-scripts / a1111 store training metadata under
# well-known prefixes inside the safetensors __metadata__ block.
_LINEAGE_KEYS = (
    "modelspec.license",
    "modelspec.architecture",
    "modelspec.title",
    "modelspec.description",
    "modelspec.author",
    "modelspec.usage_hint",
    "modelspec.tags",
    "modelspec.implementation",
    "modelspec.prediction_type",
    "modelspec.resolution",
    "modelspec.date",
    "modelspec.hash_sha256",
    "ss_base_model_version",
    "ss_sd_model_name",
    "ss_sd_model_hash",
    "ss_new_sd_model_hash",
    "ss_network_module",
    "ss_network_dim",
    "ss_network_alpha",
    "ss_dataset_dirs",
    "ss_tag_frequency",
    "ss_steps",
    "ss_epoch",
    "ss_learning_rate",
    "ss_unet_lr",
    "ss_text_encoder_lr",
    "ss_optimizer",
    "ss_lr_scheduler",
    "ss_seed",
    "ss_clip_skip",
    "ss_mixed_precision",
    "ss_full_fp16",
    "ss_v2",
    "ss_resolution",
    "ss_training_started_at",
    "ss_training_finished_at",
)


def _extract_lineage(meta: dict) -> dict:
    """Pull license/training metadata out of safetensors __metadata__ block."""
    custom = meta.get("metadata") or {}
    lineage: dict = {
        "license": None,
        "author": None,
        "base_model": None,
        "architecture": None,
        "prediction_type": None,
        "resolution": None,
        "trained_steps": None,
        "trained_epochs": None,
        "network_module": None,
        "network_dim": None,
        "network_alpha": None,
        "clip_skip": None,
        "v2_model": None,
        "raw": {},
    }
    if not isinstance(custom, dict):
        return lineage
    for key in _LINEAGE_KEYS:
        if key in custom:
            lineage["raw"][key] = custom[key]
    # Friendly aliases
    lineage["license"] = custom.get("modelspec.license")
    lineage["author"] = custom.get("modelspec.author")
    lineage["architecture"] = custom.get("modelspec.architecture")
    lineage["prediction_type"] = custom.get("modelspec.prediction_type")
    lineage["resolution"] = (
        custom.get("modelspec.resolution") or custom.get("ss_resolution")
    )
    lineage["base_model"] = (
        custom.get("ss_base_model_version")
        or custom.get("ss_sd_model_name")
        or custom.get("modelspec.architecture")
    )
    lineage["trained_steps"] = custom.get("ss_steps")
    lineage["trained_epochs"] = custom.get("ss_epoch")
    lineage["network_module"] = custom.get("ss_network_module")
    lineage["network_dim"] = custom.get("ss_network_dim")
    lineage["network_alpha"] = custom.get("ss_network_alpha")
    lineage["clip_skip"] = custom.get("ss_clip_skip")
    lineage["v2_model"] = custom.get("ss_v2")
    return lineage


# ── L9.174 (A25): checkpoint comparison + provenance / integrity report ────────────────────────────────────────────
# Structure only: headers, zip directories and pickle OPCODES are read; nothing is unpickled or loaded.

_DTYPE_BYTES = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1,
                "U16": 2, "U32": 4, "U64": 8, "BOOL": 1, "F8_E4M3": 1, "F8_E5M2": 1, "F8_E8M0": 1}
_HASH_CHUNK = 8 * 1024 * 1024
_LIST_CAP = 200
# GLOBAL references a torch checkpoint legitimately makes; anything else is reported (never imported).
_SAFE_GLOBAL_PREFIXES = ("torch.", "collections.OrderedDict", "numpy.core.multiarray._reconstruct",
                         "numpy._core.multiarray._reconstruct", "numpy.dtype", "numpy.ndarray", "_codecs.encode",
                         "builtins.set", "builtins.frozenset", "builtins.slice", "__builtin__.set")


def _read_safetensors_full(path: str):
    """-> (header_len, {name: spec}, metadata). Raises ValueError on a malformed header."""
    with open(path, "rb") as fh:
        b = fh.read(8)
        if len(b) < 8:
            raise ValueError("Truncated safetensors header.")
        hdr_len = struct.unpack("<Q", b)[0]
        if hdr_len <= 0 or hdr_len > 100 * 1024 * 1024:
            raise ValueError(f"Implausible safetensors header length: {hdr_len}")
        hdr = fh.read(hdr_len)
    if len(hdr) < hdr_len:
        raise ValueError("Truncated safetensors header.")
    parsed = json.loads(hdr.decode("utf-8"))
    meta = parsed.pop("__metadata__", None) or {}
    return hdr_len, parsed, meta


def _structure(path: str) -> dict:
    """Comparable structure of one model file: {'format', 'tensors': {name: (shape, dtype)} or 'members'}."""
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(f"Model file not found: {path!r}")
    ext = os.path.splitext(path)[1].lower()
    if ext == ".safetensors":
        _hl, parsed, _m = _read_safetensors_full(path)
        return {"format": "safetensors",
                "tensors": {k: (list(v.get("shape", [])), v.get("dtype", "?")) for k, v in parsed.items()}}
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path, "r") as zf:
            return {"format": "torch_zip", "members": {zi.filename.split("/", 1)[-1]: zi.file_size
                                                       for zi in zf.infolist()}}
    return {"format": "legacy_pickle"}


def _capped(items):
    items = list(items)
    return {"count": len(items), "items": items[:_LIST_CAP]}


def compare_models(path_a: str, path_b: str) -> dict:
    """Tensor names / shapes / dtypes (safetensors) or archive members (torch zip) of two files, side by side."""
    a, b = _structure(path_a), _structure(path_b)
    out = {"a": os.path.basename(path_a), "b": os.path.basename(path_b), "format_a": a["format"],
           "format_b": b["format"]}
    if a["format"] != b["format"]:
        out.update(same_structure=False, note="different file formats: only the formats can be compared")
        return out
    if a["format"] == "safetensors":
        ta, tb = a["tensors"], b["tensors"]

        def params(t):
            n = 0
            for shape, _d in t.values():
                p = 1
                for d in shape:
                    p *= int(d)
                n += p
            return n
        common = sorted(set(ta) & set(tb))
        shape_mm = [{"name": k, "a": ta[k][0], "b": tb[k][0]} for k in common if ta[k][0] != tb[k][0]]
        dtype_mm = [{"name": k, "a": ta[k][1], "b": tb[k][1]} for k in common if ta[k][1] != tb[k][1]]
        only_a, only_b = sorted(set(ta) - set(tb)), sorted(set(tb) - set(ta))
        pa, pb = params(ta), params(tb)
        out.update(tensor_count_a=len(ta), tensor_count_b=len(tb), total_params_a=pa, total_params_b=pb,
                   param_delta=pb - pa, common=len(common), only_in_a=_capped(only_a), only_in_b=_capped(only_b),
                   shape_mismatches=_capped(shape_mm), dtype_mismatches=_capped(dtype_mm),
                   same_structure=not (only_a or only_b or shape_mm or dtype_mm))
        return out
    if a["format"] == "torch_zip":
        ma, mb = a["members"], b["members"]
        common = sorted(set(ma) & set(mb))
        size_mm = [{"name": k, "a": ma[k], "b": mb[k]} for k in common if ma[k] != mb[k]]
        only_a, only_b = sorted(set(ma) - set(mb)), sorted(set(mb) - set(ma))
        out.update(member_count_a=len(ma), member_count_b=len(mb), only_in_a=_capped(only_a),
                   only_in_b=_capped(only_b), size_mismatches=_capped(size_mm),
                   same_structure=not (only_a or only_b or size_mm),
                   note="torch zip: archive members compared (tensor shapes live in the pickle, which is not run)")
        return out
    out.update(same_structure=None, note="legacy pickle: no structure can be read without unpickling")
    return out


def _magic_format(path: str) -> str:
    with open(path, "rb") as fh:
        head = fh.read(16)
    if head[:4] == b"PK\x03\x04":
        return "zip"
    if head[:4] == b"GGUF":
        return "gguf"
    if len(head) >= 9:
        n = struct.unpack("<Q", head[:8])[0]
        if 0 < n < 100 * 1024 * 1024 and head[8:9] == b"{":
            return "safetensors"
    if head[:1] == b"\x80":
        return "pickle"
    return "unknown"


_EXT_FORMAT = {".safetensors": ("safetensors",), ".sft": ("safetensors",), ".gguf": ("gguf",),
               ".pt": ("zip", "pickle"), ".pth": ("zip", "pickle"), ".ckpt": ("zip", "pickle"),
               ".bin": ("zip", "pickle")}


def _full_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(_HASH_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def _pickle_globals(data: bytes):
    """GLOBAL / STACK_GLOBAL references in a pickle stream, from the opcodes only (never executed)."""
    refs, strings, error = [], [], None
    try:
        for op, arg, _pos in pickletools.genops(data):
            if op.name in ("SHORT_BINUNICODE", "BINUNICODE", "UNICODE", "BINUNICODE8", "SHORT_BINSTRING",
                           "BINSTRING", "STRING"):
                strings.append(str(arg))
            elif op.name == "GLOBAL":
                refs.append(str(arg).replace(" ", ".").replace("\n", "."))
            elif op.name == "STACK_GLOBAL" and len(strings) >= 2:
                refs.append(f"{strings[-2]}.{strings[-1]}")
            if op.name == "STOP":
                break
    except Exception as exc:  # noqa: BLE001 - truncated / garbage stream: report, never crash
        error = f"{type(exc).__name__}: {str(exc)[:200]}"
    return sorted(set(refs)), error


def _safetensors_checks(path: str, size: int, warnings: list) -> dict:
    hdr_len, parsed, _meta = _read_safetensors_full(path)
    data_len = size - 8 - hdr_len
    spans = []
    for name, spec in parsed.items():
        off = spec.get("data_offsets")
        if not (isinstance(off, list) and len(off) == 2 and all(isinstance(x, int) for x in off)):
            warnings.append(f"tensor {name!r}: missing or malformed data_offsets")
            continue
        b, e = off
        if b < 0 or e < b or e > data_len:
            warnings.append(f"tensor {name!r}: data_offsets {off} outside the data area (0..{data_len})")
            continue
        n = 1
        for d in spec.get("shape", []):
            n *= int(d)
        width = _DTYPE_BYTES.get(str(spec.get("dtype")))
        if width is not None and n * width != e - b:
            warnings.append(f"tensor {name!r}: {e - b} bytes for shape {spec.get('shape')} x {spec.get('dtype')} "
                            f"(expected {n * width})")
        spans.append((b, e, name))
    spans.sort()
    gap, end = 0, 0
    for b, e, name in spans:
        if b < end:
            warnings.append(f"tensor {name!r} overlaps the previous tensor's bytes")
        elif b > end:
            gap += b - end
        end = max(end, e)
    if gap:
        warnings.append(f"{gap} unused byte(s) between tensors")
    if data_len > end:
        warnings.append(f"{data_len - end} trailing byte(s) after the last tensor")
    return {"header_bytes": hdr_len, "tensors": len(parsed), "data_bytes": data_len}


def _integrity_report(path: str, full_hash: bool = False) -> dict:
    """Format by magic bytes vs extension, optional full SHA-256, structure checks, suspicious pickle globals."""
    size = os.path.getsize(path)
    ext = os.path.splitext(path)[1].lower()
    warnings: list = []
    suspicious: list = []
    rep: dict = {"file": os.path.basename(path), "bytes": size, "extension": ext}
    try:
        magic = _magic_format(path)
    except OSError as exc:
        magic = "unreadable"
        warnings.append(f"cannot read the file: {exc}")
    rep["format_by_magic"] = magic
    expected = _EXT_FORMAT.get(ext)
    if expected and magic not in expected:
        warnings.append(f"extension {ext} but the content looks like {magic}")
    if full_hash:
        rep["sha256_full"] = _full_sha256(path)
    try:
        if magic == "safetensors":
            rep["safetensors"] = _safetensors_checks(path, size, warnings)
        elif magic in ("zip", "pickle"):
            data = b""
            if magic == "zip":
                with zipfile.ZipFile(path, "r") as zf:
                    names = [n for n in zf.namelist() if n == "data.pkl" or n.endswith("/data.pkl")]
                    if not names:
                        warnings.append("zip archive without data.pkl: not a torch checkpoint")
                    else:
                        with zf.open(names[0]) as fh:
                            data = fh.read(1_000_000)
            else:
                with open(path, "rb") as fh:
                    data = fh.read(1_000_000)
            if data:
                refs, err = _pickle_globals(data)
                rep["pickle_globals"] = refs
                if err:
                    warnings.append(f"pickle stream could not be fully parsed ({err})")
                suspicious = [r for r in refs if not r.startswith(_SAFE_GLOBAL_PREFIXES)]
    except (ValueError, zipfile.BadZipFile, json.JSONDecodeError, OSError, struct.error) as exc:
        warnings.append(f"structure could not be read: {type(exc).__name__}: {exc}")
    rep["suspicious_globals"] = suspicious
    rep["warnings"] = warnings
    rep["verdict"] = "suspicious" if suspicious else ("warnings" if warnings else "ok")
    rep["note"] = ("structure-only audit: nothing was unpickled or loaded. 'suspicious' = the pickle names "
                   "callables outside the torch / numpy / collections set (code that would run on torch.load).")
    return rep


integrity_report = _integrity_report


class ModelMetadataExtractorMEC:
    """Read .safetensors / .pt / .pth / .ckpt metadata without loading weights."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file_path": ("STRING", {
                    "default": "",
                    "tooltip": "Absolute path to a model file (.safetensors / .pt / .pth / .ckpt).",
                }),
            },
            "optional": {
                "compute_fingerprint": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "Compute SHA256 over (size, first 1 MB, last 1 MB). "
                        "Suitable cache key; far faster than full-file hashing."
                    ),
                }),
                # L9.174 (A25), appended: defaults = the behaviour of every saved workflow
                "compare_path": ("STRING", {
                    "default": "",
                    "tooltip": "Checkpoint comparison: absolute path of a second model file. Compares tensor names, "
                               "shapes and dtypes (safetensors) or archive members (torch zip) without loading "
                               "weights. Result in comparison_json. Empty = off.",
                }),
                "integrity_report": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Provenance and integrity audit: format by magic bytes vs extension, safetensors "
                               "offset / size checks, and callables a .ckpt / .pt pickle would run on load (read "
                               "from the opcodes, never executed). Result in audit_json.",
                }),
                "full_hash": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Also hash the WHOLE file (SHA-256, comparable with the hashes model sites publish). "
                               "Reads every byte: seconds per GB. Result in audit_json.",
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "INT", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = (
        "metadata_json",
        "model_kind",
        "total_params",
        "fingerprint",
        "lineage_json",
        "comparison_json",
        "audit_json",
    )
    OUTPUT_TOOLTIPS = (
        "Full metadata as JSON (header fields, training tags, format).",
        "Detected model kind, one of: stable_diffusion_checkpoint, vae, lora, text_encoder, safetensors_other (any other safetensors file, e.g. a diffusion model or ControlNet), unknown.",
        "Total parameter count summed across all tensors.",
        "SHA256 fingerprint over (size, first 1 MB, last 1 MB) when enabled.",
        "License + lineage information as JSON when present in metadata.",
        "compare_path set: tensors only in A / only in B, shape and dtype mismatches, parameter delta, as JSON. "
        "Empty when off.",
        "integrity_report or full_hash on: format check, full SHA-256, structure warnings, suspicious pickle "
        "callables and a verdict (ok / warnings / suspicious), as JSON. Empty when off.",
    )
    FUNCTION = "extract"
    CATEGORY = "C2C/Diagnostics"
    DESCRIPTION = (
        "Inspect model file metadata WITHOUT unpickling or loading weights. "
        "Safe to run on untrusted .ckpt files. Reports tensor count, params, "
        "training metadata, and a quick fingerprint."
    )

    @classmethod
    def IS_CHANGED(cls, file_path, compute_fingerprint=True, **kwargs):
        fp = file_path or ""
        if file_path and os.path.isfile(file_path):
            try:
                st = os.stat(file_path)
                fp = f"{file_path}:{st.st_mtime_ns}:{st.st_size}"
            except OSError:
                fp = file_path
        cp = kwargs.get("compare_path") or ""
        if cp and os.path.isfile(cp):
            try:
                st = os.stat(cp)
                kwargs = {**kwargs, "compare_path": f"{cp}:{st.st_mtime_ns}:{st.st_size}"}
            except OSError:
                pass
        return hash_args_and_kwargs(fp, compute_fingerprint, **kwargs)

    def extract(self, file_path: str, compute_fingerprint: bool = True, compare_path: str = "",
                integrity_report: bool = False, full_hash: bool = False):
        if not file_path or not os.path.isfile(file_path):
            raise FileNotFoundError(f"Model file not found: {file_path!r}")

        ext = os.path.splitext(file_path)[1].lower()
        size = os.path.getsize(file_path)
        meta: dict
        try:
            if ext == ".safetensors":
                meta = _read_safetensors_header(file_path)
            else:
                # Try modern torch-zip first; fall back to legacy pickle.
                if zipfile.is_zipfile(file_path):
                    meta = _read_torch_zip(file_path)
                else:
                    meta = _read_legacy_pickle(file_path)
        except (OSError, ValueError, json.JSONDecodeError, zipfile.BadZipFile,
                struct.error, EOFError) as exc:
            # MANUAL bug-fix (Apr 2026): narrowed from broad Exception so genuine
            # bugs surface; only known-safe parse/IO failures are caught.
            logger.warning("[MEC] Metadata extraction failed: %s", exc)
            meta = {"format": "error", "error": str(exc)}

        meta["file"] = os.path.basename(file_path)
        meta["bytes"] = size
        kind = _detect_model_kind(meta)
        meta["detected_kind"] = kind
        # Phase 3b v2 - Feature A: license + lineage detection.
        lineage = _extract_lineage(meta)
        meta["lineage"] = lineage
        total_params = int(meta.get("total_params", 0))
        fp = _quick_fingerprint(file_path) if compute_fingerprint else ""
        meta["fingerprint_sha256"] = fp

        comparison = ""
        if (compare_path or "").strip():
            comparison = json.dumps(compare_models(file_path, compare_path.strip()), indent=2, default=str)
        audit = ""
        if integrity_report or full_hash:
            audit = json.dumps(_integrity_report(file_path, bool(full_hash)), indent=2, default=str)

        return (
            json.dumps(meta, indent=2, default=str),
            kind,
            total_params,
            fp,
            json.dumps(lineage, indent=2, default=str),
            comparison,
            audit,
        )


NODE_CLASS_MAPPINGS = {"ModelMetadataExtractorMEC": ModelMetadataExtractorMEC}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ModelMetadataExtractorMEC": "Model Metadata Extractor"
}
