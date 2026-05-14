"""
node_explain.py — "What does this node do?" backend for MEC.

HTTP routes
-----------
GET  /mec/node_explain/status
     Returns current backend config, GGUF path, download state, API providers.

GET  /mec/node_explain/{class_name}?backend=auto&nocache=0
     Explain a node.  backend: auto | api | gguf | off
     Response: { success, data: { class_name, headline, purpose, inputs, outputs,
                                  when_to_use, tier, cached } }

POST /mec/node_explain/download
     Body (JSON): { "quant": "Q4_K_M" }   (default Q4_K_M)
     Starts a background GGUF download.
     Response: { success, data: { job_id } }

GET  /mec/node_explain/download/{job_id}
     Returns download progress.
     Response: { success, data: { status, bytes_done, total, pct, path, error } }

LLM routing (backend=auto):
    1. Cloud API  — first provider whose key is in secrets_store
    2. GGUF       — llama-cpp-python + Qwen3.5-2B-Q4_K_M.gguf on disk
    3. Deterministic — parse node INPUT_TYPES / RETURN_TYPES directly (always works)
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
import urllib.request
import urllib.error
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, Optional

log = logging.getLogger("MEC.node_explain")

# ── HuggingFace GGUF constants ─────────────────────────────────────────────
_HF_REPO = "unsloth/Qwen3.5-2B-GGUF"
_QUANT_FILES: Dict[str, str] = {
    "Q4_K_M": "Qwen3.5-2B-Q4_K_M.gguf",   # 1.28 GB — recommended
    "Q5_K_M": "Qwen3.5-2B-Q5_K_M.gguf",   # 1.44 GB
    "Q8_0":   "Qwen3.5-2B-Q8_0.gguf",      # 2.01 GB
}
_DEFAULT_QUANT = "Q4_K_M"

# ── LLM system / user prompts ──────────────────────────────────────────────
_EXPLAIN_SYSTEM = (
    "You are a ComfyUI node documentation expert. "
    "Given node metadata, explain it clearly to a user who may be new to ComfyUI. "
    "Respond ONLY with a valid JSON object — no markdown fences, no commentary.\n"
    "Required schema:\n"
    '{"headline":"<10 words max>","purpose":"<2-3 sentences>","inputs":[{"name":"...","what_for":"..."}],'
    '"outputs":[{"name":"...","what_for":"..."}],"when_to_use":"<1 sentence>"}'
)

# ── In-memory LRU cache for explanations ──────────────────────────────────
_CACHE: "OrderedDict[str, dict]" = OrderedDict()
_CACHE_LOCK = threading.Lock()
_CACHE_MAX = 300

# ── GGUF singleton backend ─────────────────────────────────────────────────
@dataclass
class _GGUFBackend:
    llm: Any  # llama_cpp.Llama instance
    model_path: str

_GGUF_BACKEND: Optional[_GGUFBackend] = None
_GGUF_LOCK = threading.Lock()

# ── Download job tracking ──────────────────────────────────────────────────
_DOWNLOAD_JOBS: Dict[str, Dict[str, Any]] = {}
_DOWNLOAD_LOCK = threading.Lock()


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _envelope_ok(data: Any) -> Dict[str, Any]:
    return {"success": True, "data": data}


def _envelope_err(key: str, msg: str) -> Dict[str, Any]:
    return {"success": False, "error": key, "message": msg}


def _gguf_dest_dir() -> str:
    """Return the directory where we place the GGUF file."""
    try:
        import folder_paths  # type: ignore
        for key in ("llm", "LLM", "language_models"):
            try:
                dirs = folder_paths.get_folder_paths(key)
                if dirs:
                    os.makedirs(dirs[0], exist_ok=True)
                    return dirs[0]
            except Exception:
                continue
        # Fallback: models/llm/ next to ComfyUI's models dir
        mdir = os.path.join(folder_paths.models_dir, "llm")
        os.makedirs(mdir, exist_ok=True)
        return mdir
    except Exception:
        pass
    here = os.path.dirname(os.path.abspath(__file__))
    pack_root = os.path.dirname(here)
    p = os.path.join(pack_root, "user", "models")
    os.makedirs(p, exist_ok=True)
    return p


def _gguf_path(quant: str = _DEFAULT_QUANT) -> Optional[str]:
    """Return the absolute path to the GGUF if it exists on disk."""
    filename = _QUANT_FILES.get(quant, _QUANT_FILES[_DEFAULT_QUANT])
    dest = os.path.join(_gguf_dest_dir(), filename)
    if os.path.isfile(dest):
        return dest
    # Also scan all llm dirs (user may have placed it elsewhere)
    try:
        import folder_paths  # type: ignore
        for key in ("llm", "LLM", "language_models"):
            try:
                for d in folder_paths.get_folder_paths(key):
                    candidate = os.path.join(d, filename)
                    if os.path.isfile(candidate):
                        return candidate
            except Exception:
                continue
    except Exception:
        pass
    return None


def _get_all_node_classes() -> dict:
    """Return ComfyUI's global NODE_CLASS_MAPPINGS (lazy, safe)."""
    try:
        import nodes as _cn  # type: ignore   # ComfyUI's top-level nodes module
        return _cn.NODE_CLASS_MAPPINGS
    except Exception:
        pass
    try:
        import importlib
        m = importlib.import_module("nodes")
        return m.NODE_CLASS_MAPPINGS
    except Exception:
        return {}


def _build_node_context(class_name: str) -> Optional[dict]:
    """Extract structured metadata from a node class."""
    mappings = _get_all_node_classes()
    cls = mappings.get(class_name)
    if cls is None:
        return None

    ctx: dict = {
        "class_name": class_name,
        "category": getattr(cls, "CATEGORY", "unknown"),
        "description": (getattr(cls, "DESCRIPTION", "") or "").strip(),
        "return_types": list(getattr(cls, "RETURN_TYPES", ())),
        "return_names": list(getattr(cls, "RETURN_NAMES", ())),
        "inputs_required": {},
        "inputs_optional": {},
    }

    try:
        input_types = cls.INPUT_TYPES()
        for group in ("required", "optional"):
            raw = input_types.get(group) or {}
            bucket: dict = {}
            for name, spec in raw.items():
                if not isinstance(spec, (list, tuple)) or len(spec) < 1:
                    continue
                type_info = spec[0]
                kwargs = spec[1] if len(spec) > 1 else {}
                bucket[name] = {
                    "type": type_info if isinstance(type_info, str) else "COMBO",
                    "tooltip": str(kwargs.get("tooltip") or kwargs.get("label") or ""),
                    "default": kwargs.get("default"),
                }
            ctx[f"inputs_{group}"] = bucket
    except Exception as exc:
        log.debug("[node_explain] INPUT_TYPES error for %s: %s", class_name, exc)

    return ctx


def _build_prompt(ctx: dict) -> str:
    """Compose the user message sent to the LLM."""
    lines = [
        f"ComfyUI Node: {ctx['class_name']}",
        f"Category: {ctx['category']}",
    ]
    if ctx.get("description"):
        lines.append(f"Description: {ctx['description']}")

    required = ctx.get("inputs_required", {})
    optional = ctx.get("inputs_optional", {})
    all_inputs = {**required, **optional}
    if all_inputs:
        lines.append("Inputs:")
        for name, info in list(all_inputs.items())[:12]:  # cap at 12 to keep prompt short
            tip = info.get("tooltip", "")
            suffix = f" — {tip}" if tip else ""
            req_flag = "(required)" if name in required else "(optional)"
            lines.append(f"  {name} [{info['type']}] {req_flag}{suffix}")

    ret_types = ctx.get("return_types", [])
    ret_names = ctx.get("return_names", [])
    if ret_types:
        lines.append("Outputs:")
        for i, t in enumerate(ret_types):
            name = ret_names[i] if i < len(ret_names) else t.lower()
            lines.append(f"  {name} [{t}]")

    lines.append("\nExplain this node.")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# JSON parsing (strips Qwen3 <think> blocks, extracts first valid JSON)
# ═══════════════════════════════════════════════════════════════════════════
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_REQUIRED_KEYS = {"headline", "purpose", "inputs", "outputs"}


def _parse_llm_json(raw: str) -> Optional[dict]:
    """Strip thinking tokens, find first JSON object, validate keys."""
    text = _THINK_RE.sub("", raw).strip()
    # Strip markdown fences if present
    text = re.sub(r"^```[a-z]*\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE)
    # Find JSON object boundaries
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    end = -1
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end == -1:
        return None
    try:
        obj = json.loads(text[start:end])
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    if not _REQUIRED_KEYS.issubset(obj.keys()):
        return None
    return obj


# ═══════════════════════════════════════════════════════════════════════════
# LLM backends
# ═══════════════════════════════════════════════════════════════════════════

def _try_cloud(ctx: dict) -> Optional[dict]:
    """Try cloud LLM (OpenAI / Anthropic / Gemini / OpenRouter / Groq / DeepSeek)."""
    try:
        from . import secrets_store, cloud_llm  # type: ignore
    except Exception:
        return None

    for provider in secrets_store.PROVIDERS:
        if not secrets_store.has_key_for(provider):
            continue
        model_map = {
            "openai": "gpt-4o-mini",
            "anthropic": "claude-haiku-20240307",
            "gemini": "gemini-1.5-flash",
            "openrouter": "meta-llama/llama-3.1-8b-instruct:free",
            "groq": "llama-3.1-8b-instant",
            "deepseek": "deepseek-chat",
        }
        model = model_map.get(provider, "")
        prompt = _build_prompt(ctx)
        try:
            raw = cloud_llm.generate(
                provider, model, prompt,
                max_tokens=500,
                system=_EXPLAIN_SYSTEM,
            )
        except Exception as e:
            log.debug("[node_explain] cloud %s failed: %s", provider, e)
            continue
        if not raw:
            continue
        parsed = _parse_llm_json(raw)
        if parsed:
            parsed["tier"] = f"cloud/{provider}"
            parsed["cached"] = False
            return parsed

    return None


def _try_gguf(ctx: dict, quant: str = _DEFAULT_QUANT) -> Optional[dict]:
    """Try the local Qwen3.5-2B GGUF via llama-cpp-python."""
    global _GGUF_BACKEND

    path = _gguf_path(quant)
    if not path:
        log.debug("[node_explain] GGUF not found on disk (quant=%s)", quant)
        return None

    # Load or reuse singleton
    with _GGUF_LOCK:
        if _GGUF_BACKEND is None or _GGUF_BACKEND.model_path != path:
            try:
                from llama_cpp import Llama  # type: ignore
            except ImportError:
                log.debug("[node_explain] llama-cpp-python not installed")
                return None
            try:
                log.info("[node_explain] Loading GGUF: %s", path)
                llm = Llama(
                    model_path=path,
                    n_ctx=2048,
                    n_threads=max(1, (os.cpu_count() or 4) - 1),
                    n_gpu_layers=0,   # CPU-only; keeps it off the VRAM budget
                    verbose=False,
                )
                _GGUF_BACKEND = _GGUFBackend(llm=llm, model_path=path)
                log.info("[node_explain] GGUF loaded OK")
            except Exception as e:
                log.warning("[node_explain] GGUF load failed: %s", e)
                return None
        backend = _GGUF_BACKEND

    prompt = _build_prompt(ctx)
    try:
        res = backend.llm.create_chat_completion(
            messages=[
                {"role": "system", "content": _EXPLAIN_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            max_tokens=500,
            temperature=0.2,
            top_p=0.9,
        )
        raw = (res["choices"][0]["message"]["content"] or "").strip()
    except Exception as e:
        log.warning("[node_explain] GGUF inference failed: %s", e)
        return None

    parsed = _parse_llm_json(raw)
    if parsed:
        parsed["tier"] = "gguf/qwen3.5-2b"
        parsed["cached"] = False
        return parsed
    log.debug("[node_explain] GGUF output was not valid JSON:\n%s", raw[:400])
    return None


def _deterministic_explain(ctx: dict) -> dict:
    """Always-works fallback: build explanation from node metadata."""
    class_name = ctx["class_name"]
    description = ctx.get("description", "")
    category = ctx.get("category", "unknown")

    required = ctx.get("inputs_required", {})
    optional = ctx.get("inputs_optional", {})

    inputs_list = []
    for name, info in {**required, **optional}.items():
        tip = info.get("tooltip", "")
        what = tip or f"A {info['type']} value"
        inputs_list.append({"name": name, "what_for": what})

    ret_types = ctx.get("return_types", [])
    ret_names = ctx.get("return_names", [])
    outputs_list = []
    for i, t in enumerate(ret_types):
        name = ret_names[i] if i < len(ret_names) else t.lower()
        outputs_list.append({"name": name, "what_for": f"Output {t} tensor"})

    headline = (description[:72] + "…") if len(description) > 72 else description
    if not headline:
        headline = f"{class_name} — {category} node"

    purpose = description if description else (
        f"{class_name} is a {category} node in ComfyUI. "
        "Connect it to compatible inputs and outputs to use it in your workflow."
    )

    return {
        "headline": headline,
        "purpose": purpose,
        "inputs": inputs_list,
        "outputs": outputs_list,
        "when_to_use": f"Use in a {category} pipeline as part of your ComfyUI workflow.",
        "tier": "deterministic",
        "cached": False,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Main explain entry point
# ═══════════════════════════════════════════════════════════════════════════

def explain_node(class_name: str, backend: str = "auto",
                 quant: str = _DEFAULT_QUANT) -> dict:
    """
    Return an explanation dict for the given node class.
    backend: "auto" | "api" | "gguf" | "off"
    """
    # Cache check
    cache_key = f"{class_name}::{backend}::{quant}"
    with _CACHE_LOCK:
        if cache_key in _CACHE:
            result = dict(_CACHE[cache_key])
            result["cached"] = True
            _CACHE.move_to_end(cache_key)
            return result

    ctx = _build_node_context(class_name)
    if ctx is None:
        return _envelope_err("not_found", f"Node '{class_name}' not in NODE_CLASS_MAPPINGS")

    result: Optional[dict] = None

    if backend in ("auto", "api"):
        result = _try_cloud(ctx)

    if result is None and backend in ("auto", "gguf"):
        result = _try_gguf(ctx, quant=quant)

    if result is None:
        result = _deterministic_explain(ctx)

    result["class_name"] = class_name

    # Store in LRU cache
    with _CACHE_LOCK:
        _CACHE[cache_key] = result
        if len(_CACHE) > _CACHE_MAX:
            _CACHE.popitem(last=False)  # evict oldest

    return result


# ═══════════════════════════════════════════════════════════════════════════
# GGUF download
# ═══════════════════════════════════════════════════════════════════════════

def _download_thread(job_id: str, quant: str) -> None:
    """Background thread: download GGUF from HuggingFace."""
    filename = _QUANT_FILES.get(quant, _QUANT_FILES[_DEFAULT_QUANT])
    url = f"https://huggingface.co/{_HF_REPO}/resolve/main/{filename}"
    dest_dir = _gguf_dest_dir()
    dest = os.path.join(dest_dir, filename)
    part = dest + ".part"

    def _set(**kw: Any) -> None:
        with _DOWNLOAD_LOCK:
            _DOWNLOAD_JOBS[job_id].update(kw)

    _set(status="downloading", filename=filename, dest=dest, url=url,
         bytes_done=0, total=0, error=None, path=None)

    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "MEC-NodeExplain/1.0 (ComfyUI custom node)"},
        )
        with urllib.request.urlopen(req, timeout=600) as resp:
            total = int(resp.headers.get("Content-Length") or 0)
            _set(total=total)
            os.makedirs(dest_dir, exist_ok=True)
            with open(part, "wb") as f:
                done = 0
                while True:
                    chunk = resp.read(131072)  # 128 KB chunks
                    if not chunk:
                        break
                    f.write(chunk)
                    done += len(chunk)
                    _set(bytes_done=done)

        os.replace(part, dest)
        _set(status="done", path=dest, bytes_done=os.path.getsize(dest))
        log.info("[node_explain] GGUF download complete: %s", dest)

    except Exception as exc:
        _set(status="error", error=str(exc))
        log.warning("[node_explain] GGUF download failed: %s", exc)
        if os.path.isfile(part):
            try:
                os.remove(part)
            except OSError:
                pass


def start_download(quant: str = _DEFAULT_QUANT) -> str:
    """Start a GGUF download and return job_id. Idempotent if already downloading."""
    # If already downloaded, return a synthetic done job
    path = _gguf_path(quant)
    if path:
        job_id = f"already_done_{quant}"
        with _DOWNLOAD_LOCK:
            _DOWNLOAD_JOBS[job_id] = {
                "status": "done",
                "filename": _QUANT_FILES.get(quant, ""),
                "dest": path,
                "bytes_done": os.path.getsize(path),
                "total": os.path.getsize(path),
                "error": None,
                "path": path,
            }
        return job_id

    job_id = str(uuid.uuid4())
    with _DOWNLOAD_LOCK:
        _DOWNLOAD_JOBS[job_id] = {
            "status": "queued",
            "quant": quant,
            "started_ts": time.time(),
            "bytes_done": 0,
            "total": 0,
            "error": None,
            "path": None,
        }

    t = threading.Thread(target=_download_thread, args=(job_id, quant), daemon=True)
    t.start()
    return job_id


# ═══════════════════════════════════════════════════════════════════════════
# Route registration
# ═══════════════════════════════════════════════════════════════════════════

def register_routes(server: Any) -> None:
    try:
        from aiohttp import web
    except Exception as e:
        log.warning("[node_explain] aiohttp unavailable: %s", e)
        return

    routes = server.routes

    # ------------------------------------------------------------------
    @routes.get("/mec/node_explain/status")
    async def _status(_req: web.Request) -> web.Response:  # noqa: ARG001
        try:
            from . import secrets_store  # type: ignore
            api_providers = [p for p in secrets_store.PROVIDERS
                             if secrets_store.has_key_for(p)]
        except Exception:
            api_providers = []

        quant_status: Dict[str, Any] = {}
        for q, fname in _QUANT_FILES.items():
            p = _gguf_path(q)
            quant_status[q] = {
                "available": p is not None,
                "path": p,
                "size_mb": round(os.path.getsize(p) / 1e6, 1) if p else None,
            }

        with _CACHE_LOCK:
            cached_nodes = list(_CACHE.keys())

        return web.json_response(_envelope_ok({
            "api_providers": api_providers,
            "gguf_quants": quant_status,
            "dest_dir": _gguf_dest_dir(),
            "cache_size": len(cached_nodes),
            "cache_max": _CACHE_MAX,
        }))

    # ------------------------------------------------------------------
    @routes.get("/mec/node_explain/{class_name}")
    async def _explain(req: web.Request) -> web.Response:
        class_name = req.match_info["class_name"]
        backend = req.query.get("backend", "auto")
        quant = req.query.get("quant", _DEFAULT_QUANT)
        nocache = req.query.get("nocache", "0") == "1"

        if backend not in ("auto", "api", "gguf", "off"):
            backend = "auto"
        if quant not in _QUANT_FILES:
            quant = _DEFAULT_QUANT

        if nocache:
            cache_key = f"{class_name}::{backend}::{quant}"
            with _CACHE_LOCK:
                _CACHE.pop(cache_key, None)

        if backend == "off":
            ctx = _build_node_context(class_name)
            if ctx is None:
                return web.json_response(
                    _envelope_err("not_found", f"Node '{class_name}' not found"),
                    status=404,
                )
            result = _deterministic_explain(ctx)
            result["class_name"] = class_name
        else:
            result = explain_node(class_name, backend=backend, quant=quant)

        if "error" in result and not result.get("success", True):
            return web.json_response(result, status=404)

        return web.json_response(_envelope_ok(result))

    # ------------------------------------------------------------------
    @routes.post("/mec/node_explain/download")
    async def _trigger_download(req: web.Request) -> web.Response:
        try:
            body = await req.json()
            quant = str(body.get("quant", _DEFAULT_QUANT))
        except Exception:
            quant = _DEFAULT_QUANT

        if quant not in _QUANT_FILES:
            return web.json_response(
                _envelope_err("invalid_quant",
                              f"quant must be one of {list(_QUANT_FILES.keys())}"),
                status=400,
            )

        job_id = start_download(quant)
        return web.json_response(_envelope_ok({"job_id": job_id, "quant": quant}))

    # ------------------------------------------------------------------
    @routes.get("/mec/node_explain/download/{job_id}")
    async def _download_progress(req: web.Request) -> web.Response:
        job_id = req.match_info["job_id"]
        with _DOWNLOAD_LOCK:
            job = _DOWNLOAD_JOBS.get(job_id)
        if job is None:
            return web.json_response(
                _envelope_err("not_found", f"Job '{job_id}' not found"),
                status=404,
            )
        total = job.get("total", 0) or 0
        done = job.get("bytes_done", 0) or 0
        pct = round(done / total * 100, 1) if total > 0 else 0
        return web.json_response(_envelope_ok({
            "job_id": job_id,
            "status": job.get("status", "unknown"),
            "filename": job.get("filename", ""),
            "bytes_done": done,
            "total": total,
            "pct": pct,
            "path": job.get("path"),
            "error": job.get("error"),
        }))

    log.info("[node_explain] Routes registered: /mec/node_explain/*")
