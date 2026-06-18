"""Tier-2 backend for ErrorAssistantMEC: local SLM via llama-cpp-python.

We deliberately keep this CPU-only and small. The default model is a 0.5–1B
instruct GGUF quant that fits in <2 GB RAM and produces a 5–10s explanation
when Tier 3 is unavailable.

Public API
----------
get_or_load(model_id, n_threads=0) -> Backend | None
    Returns a singleton Backend or None if llama-cpp-python is not installed.
    Never downloads silently — the user must have placed a GGUF in
    `<comfyui>/models/llm/` (or `<pack>/user/models/`).

Backend.generate(prompt, max_tokens=512) -> str
    Synchronous, blocks the calling thread.
"""
from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from typing import Optional

log = logging.getLogger("MEC.local_llm")

_LOCK = threading.Lock()
_BACKEND = None  # type: Optional["Backend"]


def _candidate_dirs() -> list:
    here = os.path.dirname(os.path.abspath(__file__))
    pack_root = os.path.dirname(here)
    out = [os.path.join(pack_root, "user", "models")]
    try:
        import folder_paths  # type: ignore
        # Honour every folder ComfyUI considers an LLM/text-encoder store.
        # `text_encoders` is added because many users (Qwen/Wan workflows)
        # park GGUF text-encoder/LLM files there and expect them to be
        # picked up by the AI Spine.
        for k in ("llm", "LLM", "language_models", "text_encoders", "clip"):
            try:
                out.extend(folder_paths.get_folder_paths(k))
            except Exception:
                pass
        out.append(os.path.join(folder_paths.models_dir, "llm"))
    except Exception:
        pass
    # De-duplicate while preserving order.
    seen = set(); uniq = []
    for p in out:
        if not p:
            continue
        ap = os.path.abspath(p)
        if ap in seen:
            continue
        seen.add(ap)
        if os.path.isdir(ap):
            uniq.append(ap)
    return uniq


def _resolve_model_path(model_id: str) -> Optional[str]:
    """Find a GGUF for `model_id`. `model_id` is a filename stub or full path."""
    if not model_id:
        return None
    if os.path.isabs(model_id) and os.path.exists(model_id):
        return model_id
    target = model_id.lower()
    for d in _candidate_dirs():
        try:
            for fn in os.listdir(d):
                if fn.lower().endswith(".gguf") and target in fn.lower():
                    return os.path.join(d, fn)
        except Exception:
            continue
    return None


@dataclass
class Backend:
    llm: object  # Llama instance
    model_path: str
    n_ctx: int = 4096

    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        try:
            res = self.llm.create_chat_completion(  # type: ignore[attr-defined]
                messages=[
                    {"role": "system",
                     "content": "You explain Python tracebacks for ComfyUI users. "
                                "Be concise. Always answer with two sections "
                                "labelled exactly CAUSE: and FIXES: (bulleted)."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=int(max_tokens),
                temperature=0.2,
                top_p=0.9,
            )
            return (res["choices"][0]["message"]["content"] or "").strip()
        except Exception as e:
            log.warning("[local_llm] chat_completion failed: %s — falling back to completion()", e)
            try:
                res = self.llm(  # type: ignore[misc]
                    prompt,
                    max_tokens=int(max_tokens),
                    temperature=0.2,
                    top_p=0.9,
                    stop=["</s>", "<|im_end|>"],
                )
                return (res["choices"][0]["text"] or "").strip()
            except Exception as e2:
                log.warning("[local_llm] completion failed: %s", e2)
                return ""


def get_or_load(model_id: Optional[str] = None,
                n_threads: int = 0) -> Optional[Backend]:
    """Lazily build a singleton llama.cpp backend. Returns None if unavailable."""
    global _BACKEND
    with _LOCK:
        if _BACKEND is not None:
            return _BACKEND
        try:
            from llama_cpp import Llama  # type: ignore
        except Exception as e:
            log.info("[local_llm] llama-cpp-python not installed: %s", e)
            return None
        path = _resolve_model_path(model_id or "qwen2.5-0.5b-instruct-q4_k_m")
        if not path:
            log.info("[local_llm] no GGUF matching %r in %s",
                     model_id, _candidate_dirs())
            return None
        try:
            llm = Llama(
                model_path=path,
                n_ctx=4096,
                n_threads=int(n_threads) if n_threads else None,
                n_gpu_layers=0,  # strict CPU to leave VRAM for image models
                verbose=False,
            )
            _BACKEND = Backend(llm=llm, model_path=path)
            log.info("[local_llm] loaded %s", path)
            return _BACKEND
        except Exception as e:
            log.warning("[local_llm] failed to load %s: %s", path, e)
            return None


def unload() -> None:
    """Free the singleton (best-effort)."""
    global _BACKEND
    with _LOCK:
        _BACKEND = None
