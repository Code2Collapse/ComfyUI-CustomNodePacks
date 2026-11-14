"""Hugging Face model folders under models/LLM: which ones can chat (ledger L6.02 S4).

A folder counts when it holds ``config.json`` + ``*.safetensors`` weights + a tokenizer (``tokenizer.json`` or
``tokenizer_config.json``). Only ``config.json`` is read - never the weights - so the listing is cheap and safe in a
request. ``can_chat`` = the architecture is a causal language model (``*ForCausalLM``) or a Qwen-VL style
``*ForConditionalGeneration`` with a text model, and transformers is installed. Nothing is downloaded, ever: the
backend loads with ``local_files_only=True``.

Folders searched (no recursion beyond one level): ComfyUI's ``llm`` / ``LLM`` / ``language_models`` folder keys and
``models/LLM``.
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Dict, List, Optional

FOLDER_KEYS = ("llm", "LLM", "language_models")
_VISION_HINTS = ("vision_config", "visual")


def roots() -> List[str]:
    out: List[str] = []
    try:
        import folder_paths  # type: ignore
        for key in FOLDER_KEYS:
            try:
                out += list(folder_paths.get_folder_paths(key))
            except Exception:  # noqa: BLE001 - key not registered
                pass
        out.append(os.path.join(folder_paths.models_dir, "LLM"))
    except Exception:  # noqa: BLE001 - outside ComfyUI
        pass
    seen, uniq = set(), []
    for r in out:
        k = os.path.normcase(os.path.abspath(r))
        if k not in seen and os.path.isdir(r):
            seen.add(k)
            uniq.append(r)
    return uniq


def _has_weights(d: str, names: List[str]) -> bool:
    return any(n.endswith(".safetensors") for n in names)


def _params_from_config(cfg: dict) -> Optional[float]:
    """A rough parameter count (billions) from the config - display only."""
    c = cfg.get("text_config") or cfg
    try:
        h, l, v = int(c["hidden_size"]), int(c["num_hidden_layers"]), int(c["vocab_size"])
        inter = int(c.get("intermediate_size") or 4 * h)
        per_layer = 4 * h * h + 3 * h * inter
        return round((l * per_layer + v * h * (1 if c.get("tie_word_embeddings", True) else 2)) / 1e9, 2)
    except Exception:  # noqa: BLE001
        return None


def classify(folder: str) -> dict:
    names = sorted(os.listdir(folder))
    out = {"folder": folder, "name": os.path.basename(folder.rstrip("\\/")), "can_chat": False, "why": "",
           "architecture": None, "model_type": None, "params_b": None, "vision": False,
           "size_gb": round(sum(os.path.getsize(os.path.join(folder, n)) for n in names
                                if n.endswith(".safetensors")) / 2 ** 30, 2)}
    if "config.json" not in names:
        out["why"] = "no config.json"
        return out
    if not _has_weights(folder, names):
        out["why"] = "no .safetensors weights"
        return out
    if not ({"tokenizer.json", "tokenizer_config.json"} & set(names)):
        out["why"] = "no tokenizer files"
        return out
    try:
        with open(os.path.join(folder, "config.json"), encoding="utf-8") as fh:
            cfg = json.load(fh)
    except Exception as exc:  # noqa: BLE001
        out["why"] = f"config.json unreadable: {exc}"
        return out
    arch = (cfg.get("architectures") or [None])[0]
    out.update(architecture=arch, model_type=cfg.get("model_type"), params_b=_params_from_config(cfg),
               vision=any(k in cfg for k in _VISION_HINTS))
    if not arch:
        out["why"] = "config.json names no architecture"
    elif arch.endswith("ForCausalLM") or (arch.endswith("ForConditionalGeneration") and "text_config" in cfg):
        try:
            import importlib.util
            if importlib.util.find_spec("transformers") is None:
                out["why"] = "transformers is not installed"
                return out
        except Exception:  # noqa: BLE001
            pass
        out["can_chat"] = True
        out["why"] = f"{arch}: a language model transformers can run from this folder"
    else:
        out["why"] = f"{arch} is not a chat / causal language model"
    return out


@lru_cache(maxsize=256)
def _classify_cached(folder: str, stamp: float) -> dict:
    return classify(folder)


def classify_cached(folder: str) -> dict:
    try:
        stamp = os.path.getmtime(os.path.join(folder, "config.json"))
    except OSError:
        stamp = 0.0
    return dict(_classify_cached(folder, stamp))


def list_folders() -> List[Dict]:
    """Every candidate folder one level below each root (and a root that is itself a model folder)."""
    out = []
    for root in roots():
        cands = [root] + [os.path.join(root, n) for n in sorted(os.listdir(root))
                          if os.path.isdir(os.path.join(root, n))]
        for d in cands:
            if os.path.isfile(os.path.join(d, "config.json")):
                out.append(classify_cached(d))
    return out
