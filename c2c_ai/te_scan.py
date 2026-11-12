"""Which text encoders on disk can the AI assistant chat with? (L6.02 S1, design docs/research/L6.02_tier_local_design.md)

Reads only the safetensors HEADER (tensor names + shapes; no tensor data, no torch) and asks ComfyUI core's own
`comfy.sd.detect_te_model` which family it is - the same detection CLIPLoader uses, so files are recognised by their
weights, not their names. Then classifies whether core's text generation (`CLIP.generate`, the TextGenerate node)
gives correct output for that family:

- core must have a generate path for the family (classes with BaseGenerate / their own generate);
- core's output layer must be right: it builds a separate `lm_head` only for some configs and otherwise multiplies by
  the input embeddings, which is only correct when the original model ties them (verified against the official HF
  configs, docs/evidence/L6.02/tie_embeddings.json, 2026-11-12);
- a family that needs `lm_head` must actually have `lm_head.weight` in the file (text-encoder releases often drop it).

Anything not checked is reported as "not verified" instead of guessed.
"""
from __future__ import annotations

import json
import logging
import os
import struct
import threading
from types import SimpleNamespace
from typing import Optional

log = logging.getLogger("c2c_ai.te_scan")

MAX_HEADER = 100 << 20
# family -> (can_chat, needs_lm_head_key, reason)
_CHAT_OK = {
    "QWEN3_06B": "Qwen3 0.6B - ties its embeddings, so ComfyUI's output layer is right",
    "QWEN3_4B": "Qwen3 4B - ties its embeddings, so ComfyUI's output layer is right",
    "QWEN3VL_4B": "Qwen3-VL 4B - ties its embeddings; can also look at images",
    "QWEN3_8B": "Qwen3 8B - ComfyUI loads its own output layer (lm_head)",
    "QWEN3VL_8B": "Qwen3-VL 8B - ComfyUI loads its own output layer (lm_head); can also look at images",
}
_NEEDS_LM_HEAD = {"QWEN3_8B", "QWEN3VL_8B"}
_WRONG_HEAD = {
    "QWEN25_7B": "Qwen2.5-VL 7B: ComfyUI generates through the input embeddings, but this model does not tie them, "
                 "so its answers would be nonsense",
    "QWEN3VL_32B": "Qwen3-VL 32B as shipped for MiniMax H3 is cut to 50 layers and ComfyUI has no output layer for it",
}
_NO_GENERATE = {
    "QWEN25_3B": "Qwen2.5-VL 3B: ComfyUI cannot generate text with this encoder",
}
_ENCODER_ONLY = {"CLIP_L", "CLIP_H", "CLIP_G", "T5_XXL", "T5_XL", "T5_BASE", "T5_XXL_OLD", "BYT5_SMALL_GLYPH",
                 "JINA_CLIP_2", "T5_GEMMA"}

_CACHE: dict = {}
_LOCK = threading.Lock()


def read_header(path: str) -> dict:
    """{tensor name: shape tuple} from a .safetensors header (8-byte little-endian length + JSON). Raises ValueError."""
    with open(path, "rb") as fh:
        raw = fh.read(8)
        if len(raw) != 8:
            raise ValueError("not a safetensors file (too short)")
        (n,) = struct.unpack("<Q", raw)
        if n <= 0 or n > MAX_HEADER:
            raise ValueError("not a safetensors file (bad header length)")
        meta = json.loads(fh.read(n).decode("utf-8"))
    return {k: tuple(v.get("shape") or ()) for k, v in meta.items() if k != "__metadata__" and isinstance(v, dict)}


class _Shapes(dict):
    """detect_te_model only asks `name in sd` and `sd[name].shape` - shapes are enough."""

    def __getitem__(self, k):
        return SimpleNamespace(shape=dict.__getitem__(self, k))


def detect_family(shapes: dict) -> Optional[str]:
    """Core's detect_te_model on names + shapes; None when core cannot be imported or does not know the file."""
    try:
        import comfy.sd as csd  # type: ignore
    except Exception as exc:  # noqa: BLE001
        log.info("te_scan: ComfyUI core not importable (%s)", type(exc).__name__)
        return None
    try:
        te = csd.detect_te_model(_Shapes(shapes))
    except Exception as exc:  # noqa: BLE001 - an unusual file must not break the scan
        log.info("te_scan: detection failed: %s", exc)
        return None
    return te.name if te is not None else None


def classify(path: str) -> dict:
    """{file, family, params_b, vision, has_lm_head, can_chat, why} for one file. Never raises."""
    out = {"file": path, "family": None, "params_b": None, "vision": False, "has_lm_head": False,
           "can_chat": False, "why": ""}
    if not path.lower().endswith(".safetensors"):
        out["why"] = "only .safetensors text encoders are checked here (GGUF files use the llama.cpp tier)"
        return out
    try:
        shapes = read_header(path)
    except Exception as exc:  # noqa: BLE001
        out["why"] = f"could not read the file header: {exc}"
        return out
    total = 0
    for s in shapes.values():
        n = 1
        for d in s:
            n *= int(d)
        total += n
    out["params_b"] = round(total / 1e9, 2)
    out["vision"] = any(k.startswith(("model.visual.", "visual.", "vision_model.", "vision_tower."))
                        for k in shapes)
    out["has_lm_head"] = any(k.endswith("lm_head.weight") for k in shapes)
    fam = detect_family(shapes)
    out["family"] = fam
    if fam is None:
        out["why"] = "not a text encoder ComfyUI recognises"
    elif fam in _ENCODER_ONLY:
        out["why"] = "an encoder only (CLIP / T5 family): it turns text into conditioning, it cannot write answers"
    elif fam in _WRONG_HEAD:
        out["why"] = _WRONG_HEAD[fam]
    elif fam in _NO_GENERATE:
        out["why"] = _NO_GENERATE[fam]
    elif fam in _CHAT_OK:
        if fam in _NEEDS_LM_HEAD and not out["has_lm_head"]:
            out["why"] = (f"{_CHAT_OK[fam].split(' - ')[0]}: this copy has no lm_head.weight (text-encoder releases "
                          "often drop it), so ComfyUI would answer with a random output layer")
        else:
            out["can_chat"] = True
            out["why"] = _CHAT_OK[fam]
    else:
        out["why"] = f"{fam}: chatting with this family through ComfyUI is not verified yet"
    return out


def classify_cached(path: str) -> dict:
    try:
        st = os.stat(path)
        key = (os.path.normcase(os.path.abspath(path)), st.st_size, st.st_mtime_ns)
    except OSError:
        return classify(path)
    with _LOCK:
        hit = _CACHE.get(key)
    if hit is None:
        hit = classify(path)
        with _LOCK:
            _CACHE[key] = hit
    return dict(hit)
