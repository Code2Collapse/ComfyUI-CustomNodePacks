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
from typing import Any, Dict, Optional

log = logging.getLogger("MEC.local_llm")

_LOCK = threading.Lock()
_BACKEND = None  # type: Optional["Backend"]

# W5a (2026-07): the default local model is now Qwen3.5-4B Q4_K_M (~2.5 GB RAM,
# CPU inference). Primary = the Claude-Opus-reasoning distilled community build
# (exactly the "small qwen + opus-trained dataset" ask); fallback = the unsloth
# mainline. Both Apache-2.0. Download is EXPLICIT (route/button) — never silent.
DEFAULT_MODEL_STUB = "qwen3.5-4b"
_DL_SOURCES = [
    ("https://huggingface.co/Jackrong/Qwen3.5-4B-Claude-4.6-Opus-Reasoning-Distilled-v2-GGUF"
     "/resolve/main/Qwen3.5-4B.Q4_K_M.gguf",
     "Qwen3.5-4B-OpusDistill.Q4_K_M.gguf"),
    ("https://huggingface.co/unsloth/Qwen3.5-4B-GGUF"
     "/resolve/main/Qwen3.5-4B-Q4_K_M.gguf",
     "Qwen3.5-4B.Q4_K_M.gguf"),
]

# Download state shared with the /c2c/ai/local_model/* routes.
_DL_STATE: Dict[str, Any] = {
    "downloading": False, "progress": 0.0, "bytes_done": 0,
    "bytes_total": 0, "error": None, "path": None,
}
_DL_LOCK = threading.Lock()


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


def _resolve_any_model(model_id: Optional[str]) -> Optional[str]:
    """Resolution order: explicit id → the Qwen3.5-4B default → any qwen GGUF
    → the first GGUF found anywhere. So Tier-2 works with whatever the user
    already has instead of demanding one exact filename."""
    for stub in (model_id, DEFAULT_MODEL_STUB, "qwen"):
        if stub:
            p = _resolve_model_path(stub)
            if p:
                return p
    for d in _candidate_dirs():
        try:
            ggufs = sorted(fn for fn in os.listdir(d) if fn.lower().endswith(".gguf"))
            if ggufs:
                return os.path.join(d, ggufs[0])
        except Exception:
            continue
    return None


def _llm_dir() -> str:
    """The canonical download target: <comfyui>/models/llm (created on demand)."""
    try:
        import folder_paths  # type: ignore
        d = os.path.join(folder_paths.models_dir, "llm")
    except Exception:
        here = os.path.dirname(os.path.abspath(__file__))
        d = os.path.join(os.path.dirname(here), "user", "models")
    os.makedirs(d, exist_ok=True)
    return d


def get_status() -> Dict[str, Any]:
    """Snapshot for the /c2c/ai/local_model/status route + settings panel."""
    try:
        import llama_cpp  # type: ignore  # noqa: F401
        has_runtime = True
    except Exception:
        has_runtime = False
    path = _resolve_any_model(None)
    with _DL_LOCK:
        dl = dict(_DL_STATE)
    return {
        "llama_cpp": has_runtime,
        "model_path": path,
        "model_present": bool(path),
        "loaded": _BACKEND is not None,
        "default_model": DEFAULT_MODEL_STUB,
        "download": dl,
    }


def _download_worker() -> None:
    import urllib.request
    headers = {"User-Agent": "Mozilla/5.0 (ComfyUI C2C local-llm downloader)"}
    tok = os.environ.get("HF_TOKEN", "").strip()
    if tok:
        headers["Authorization"] = f"Bearer {tok}"
    last_err = None
    for url, fname in _DL_SOURCES:
        dest = os.path.join(_llm_dir(), fname)
        part = dest + ".part"
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=60) as r:
                total = int(r.headers.get("Content-Length") or 0)
                with _DL_LOCK:
                    _DL_STATE.update(bytes_total=total, bytes_done=0, progress=0.0)
                done = 0
                with open(part, "wb") as fh:
                    while True:
                        chunk = r.read(1 << 20)   # 1 MiB
                        if not chunk:
                            break
                        fh.write(chunk)
                        done += len(chunk)
                        with _DL_LOCK:
                            _DL_STATE.update(
                                bytes_done=done,
                                progress=(done / total) if total else 0.0,
                            )
            os.replace(part, dest)
            with _DL_LOCK:
                _DL_STATE.update(downloading=False, progress=1.0,
                                 error=None, path=dest)
            log.info("[local_llm] downloaded %s (%.1f GB)", dest, done / 1e9)
            return
        except Exception as e:  # noqa: BLE001
            last_err = e
            log.warning("[local_llm] download from %s failed: %s", url, e)
            try:
                if os.path.exists(part):
                    os.remove(part)
            except Exception:
                pass
    with _DL_LOCK:
        _DL_STATE.update(downloading=False,
                         error=f"all sources failed: {last_err}")


def start_download() -> Dict[str, Any]:
    """Explicitly begin the default-model download in a background thread.
    Idempotent: returns the current state if already downloading/present."""
    existing = _resolve_model_path(DEFAULT_MODEL_STUB)
    if existing:
        return {"status": "already_present", "path": existing}
    with _DL_LOCK:
        if _DL_STATE["downloading"]:
            return {"status": "in_progress", **_DL_STATE}
        _DL_STATE.update(downloading=True, progress=0.0, bytes_done=0,
                         bytes_total=0, error=None, path=None)
    threading.Thread(target=_download_worker, name="c2c-llm-download",
                     daemon=True).start()
    return {"status": "started"}


def _strip_reasoning(text: str) -> str:
    """Reasoning-distilled models (the Qwen3.5-4B Opus distill) 'think out
    loud' before answering. Keep only the final answer: cut everything before
    the first CAUSE: label when present, and drop <think>…</think> blocks."""
    if not text:
        return text
    import re as _re
    text = _re.sub(r"<think>.*?</think>", "", text, flags=_re.S).strip()
    # The thinking often QUOTES the required labels ("labelled exactly CAUSE:
    # and FIXES:"), so the FIRST occurrence can be inside the reasoning. The
    # final answer is the LAST "CAUSE:" that still has a "FIXES:" after it.
    idx = text.rfind("CAUSE:")
    if idx > 0 and "FIXES:" in text[idx:]:
        return text[idx:].strip()
    idx2 = text.find("CAUSE:")
    if idx2 > 0 and "FIXES:" in text[idx2:]:
        return text[idx2:].strip()
    return text


@dataclass
class Backend:
    llm: object  # Llama instance
    model_path: str
    n_ctx: int = 4096

    def generate(self, prompt: str, max_tokens: int = 512,
                 system: Optional[str] = None) -> str:
        """`system=None` keeps the historical CAUSE:/FIXES: error-explainer
        persona. Callers with their OWN output contract (workflow builder,
        diagnose — both want strict JSON) MUST pass a system prompt, or this
        baked-in format instruction fights theirs and the reasoning distill
        burns its whole token budget arguing with itself (observed live)."""
        sys_msg = system if system is not None else (
            "You explain Python tracebacks for ComfyUI users. "
            "Be concise. Always answer with two sections "
            "labelled exactly CAUSE: and FIXES: (bulleted).")
        try:
            res = self.llm.create_chat_completion(  # type: ignore[attr-defined]
                messages=[
                    {"role": "system", "content": sys_msg},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=int(max_tokens),
                temperature=0.2,
                top_p=0.9,
            )
            return _strip_reasoning((res["choices"][0]["message"]["content"] or "").strip())
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
        path = _resolve_any_model(model_id)
        if not path:
            log.info("[local_llm] no GGUF found (looked for %r, %r, any) in %s — "
                     "use the 'Download local AI model' button in C2C AI settings.",
                     model_id, DEFAULT_MODEL_STUB, _candidate_dirs())
            return None
        try:
            llm = Llama(
                model_path=path,
                n_ctx=8192,   # roomier tracebacks; Qwen3.5 trains at 256k, 8k is still light on RAM
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
