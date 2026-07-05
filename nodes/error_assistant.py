"""
ErrorAssistant — 3-tier ComfyUI exception explainer
====================================================

Tier 1: deterministic regex/pattern matcher  (always on, 0 VRAM, 0 ms)
Tier 2: local SLM via llama-cpp-python        (CPU, ~400 MB-2 GB, 0 VRAM)
Tier 3: cloud LLM (OpenAI / Anthropic / Gemini / OpenRouter)

Routing policy is selected by the user in ComfyUI Settings:

    "auto"               -> Tier 3 if cloud key set, else Tier 2 if model
                           cached, else Tier 1.
    "deterministic_only" -> Tier 1 only.
    "local_only"         -> Tier 1 then Tier 2.
    "cloud_only"         -> Tier 1 then Tier 3.

Public surface
--------------
explain(exc, *, node_class=None, inputs_summary=None, mode="auto") -> dict
    Returns {tier, headline, cause, fixes:[...], extra}.
    Always returns something — never raises.

Tier 1 is a curated table of ~40 patterns covering the errors users
actually report. Patterns are ordered by specificity; first match wins.
"""
from __future__ import annotations

import logging
import os
import re
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

log = logging.getLogger("MEC.error_assistant")

# =====================================================================
# Tier 1 — deterministic patterns (Doctor-style: JSON pattern packs)
# =====================================================================
@dataclass
class Pattern:
    """A single deterministic error fingerprint."""
    name: str
    exc_types: tuple[str, ...]            # e.g. ("RuntimeError",) — empty = any
    message_re: re.Pattern[str]           # case-insensitive by default
    cause: str
    fixes: List[str] = field(default_factory=list)
    # Doctor-style metadata
    category: str = "uncategorized"
    priority: int = 100
    confidence: float = 0.8
    source: str = "builtin"               # provenance: which pack file


def _r(pat: str) -> re.Pattern[str]:
    return re.compile(pat, re.IGNORECASE | re.DOTALL)


# ---------------------------------------------------------------------
# JSON pattern-pack loader (hot-reload on file mtime change)
# ---------------------------------------------------------------------
def _patterns_root() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    pack_root = os.path.dirname(here)
    return os.path.join(pack_root, "patterns")


_PATTERNS_LOCK = threading.Lock()
_PATTERNS_CACHE: List[Pattern] = []
_PATTERNS_MTIME: Dict[str, float] = {}   # path -> last-seen mtime


def _scan_pattern_files() -> List[str]:
    root = _patterns_root()
    out: List[str] = []
    if not os.path.isdir(root):
        return out
    for dirpath, _dirs, files in os.walk(root):
        for f in files:
            if f.endswith(".json"):
                out.append(os.path.join(dirpath, f))
    return sorted(out)


def _parse_pattern_dict(d: Dict[str, Any], source: str) -> Optional[Pattern]:
    try:
        pid = str(d["id"])
        regex = d["regex"]
        cause = d.get("cause", "")
        exc_types = tuple(d.get("exc_types") or ())
        return Pattern(
            name=pid,
            exc_types=exc_types,
            message_re=_r(regex),
            cause=cause,
            fixes=list(d.get("fixes") or []),
            category=str(d.get("category", "uncategorized")),
            priority=int(d.get("priority", 100)),
            confidence=float(d.get("confidence", 0.8)),
            source=source,
        )
    except Exception as e:
        log.warning("[error_assistant] bad pattern in %s: %s", source, e)
        return None


def _load_patterns_from_json() -> List[Pattern]:
    import json
    out: List[Pattern] = []
    for path in _scan_pattern_files():
        rel = os.path.relpath(path, _patterns_root()).replace("\\", "/")
        # Skip i18n overlays — they use a different schema (patterns is a dict
        # keyed by EN i18n_key/id, not a list). Loaded separately by
        # error_translator._apply_locale_overlay().
        if rel.startswith("i18n/") or "/i18n/" in rel:
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Defensive: only iterate when patterns is a list (EN pack schema).
            entries = data.get("patterns", [])
            if not isinstance(entries, list):
                continue
            for entry in entries:
                p = _parse_pattern_dict(entry, source=rel)
                if p is not None:
                    out.append(p)
        except Exception as e:
            log.warning("[error_assistant] failed to load pattern pack %s: %s", path, e)
    # Lower priority value = evaluated first (more specific). Stable sort.
    out.sort(key=lambda p: (p.priority, p.name))
    return out


def _get_patterns() -> List[Pattern]:
    """Return current patterns; hot-reload on any pack file mtime change."""
    global _PATTERNS_CACHE, _PATTERNS_MTIME
    with _PATTERNS_LOCK:
        files = _scan_pattern_files()
        cur_mtimes: Dict[str, float] = {}
        for p in files:
            try:
                cur_mtimes[p] = os.path.getmtime(p)
            except OSError:
                cur_mtimes[p] = 0.0
        if cur_mtimes != _PATTERNS_MTIME or not _PATTERNS_CACHE:
            loaded = _load_patterns_from_json()
            if loaded:
                _PATTERNS_CACHE = loaded
                _PATTERNS_MTIME = cur_mtimes
                log.info("[error_assistant] loaded %d patterns from %d pack(s)",
                         len(loaded), len(files))
            elif not _PATTERNS_CACHE:
                # JSON missing/corrupt on first run: fall back to bootstrap list.
                _PATTERNS_CACHE = list(_BOOTSTRAP_PATTERNS)
                _PATTERNS_MTIME = cur_mtimes
                log.warning("[error_assistant] no JSON packs found; using %d "
                            "bootstrap patterns", len(_PATTERNS_CACHE))
        return _PATTERNS_CACHE


# Order matters — first match wins. Most-specific patterns first.
# This list is ONLY used as a bootstrap fallback when patterns/builtin/core.json
# is missing or unreadable. The canonical source of truth is the JSON pack.
_BOOTSTRAP_PATTERNS: List[Pattern] = [
    # ── VRAM / CUDA ────────────────────────────────────────────────
    Pattern(
        "cuda_oom",
        ("RuntimeError", "torch.cuda.OutOfMemoryError"),
        _r(r"out of memory|CUDA out of memory|alloc.*failed"),
        "Your GPU ran out of VRAM while running this node.",
        [
            "Lower batch size / image resolution.",
            "Restart ComfyUI with `--lowvram` or `--novram`.",
            "Set the node's `use_half=True` / dtype `fp16` if available.",
            "If you have an SDXL+CLIP+VAE all loaded, split the loaders so each component is loaded sequentially.",
            "Run `torch.cuda.empty_cache()` between nodes (or insert an Insight Status node).",
        ],
    ),
    Pattern(
        "cuda_device_assert",
        ("RuntimeError",),
        _r(r"device-side assert triggered"),
        "A CUDA kernel hit an assertion — usually an out-of-range index "
        "(prompt token id ≥ vocab size, or a bad pixel index).",
        [
            "Re-run with environment `CUDA_LAUNCH_BLOCKING=1` so the traceback points to the real call site.",
            "Check the prompt for unusual characters; some custom CLIP variants use a smaller vocab.",
            "Verify mask/latent indices fit within their tensor shapes.",
        ],
    ),
    Pattern(
        "cudnn_status",
        ("RuntimeError",),
        _r(r"CUDNN_STATUS|cuDNN error"),
        "cuDNN failed — most often a driver / torch / CUDA version mismatch.",
        [
            "Confirm your torch build matches the installed CUDA: `python -c \"import torch; print(torch.version.cuda)\"`.",
            "Update the NVIDIA driver to a version compatible with this CUDA toolkit.",
            "Try `torch.backends.cudnn.benchmark = False`.",
        ],
    ),

    # ── Tensor shape / dtype ───────────────────────────────────────
    Pattern(
        "matmul_shape",
        ("RuntimeError",),
        _r(r"mat1 and mat2 shapes cannot be multiplied"),
        "Two tensors with incompatible inner dimensions were multiplied — "
        "almost always a model architecture mismatch (e.g. SDXL CLIP wired into "
        "an SD1.5 model, or a Flux VAE wired into an SDXL pipeline).",
        [
            "Check that every loader (CHECKPOINT, CLIP, VAE) is from the *same* model family.",
            "If you used a custom CLIP/text-encoder, confirm its hidden_size matches the diffusion model.",
            "If using LoRA, confirm the LoRA was trained for this base model.",
        ],
    ),
    Pattern(
        "size_mismatch",
        ("RuntimeError",),
        _r(r"size of tensor.*must match|sizes? of tensors? must match|expected.*got.*size"),
        "Tensor sizes don't line up between two operations.",
        [
            "Make sure upstream image/mask/latent has the resolution the consumer expects.",
            "If you resized, did you also resize the mask?",
            "ControlNet / IP-Adapter inputs must match the model's expected resolution.",
        ],
    ),
    Pattern(
        "scalar_type_half",
        ("RuntimeError",),
        _r(r"expected scalar type Half but found Float|expected scalar type Float but found Half"),
        "A tensor's dtype (fp16 vs fp32) doesn't match the model — usually after "
        "wiring a half-precision model to a node that produces fp32 (or vice versa).",
        [
            "Cast at the boundary: `tensor = tensor.half()` or `.float()`.",
            "If the loader has a `precision` widget, set it to match downstream nodes.",
            "Restart ComfyUI with `--force-fp16` (or `--force-fp32`) for consistency.",
        ],
    ),
    Pattern(
        "dtype_bfloat16",
        ("RuntimeError",),
        _r(r"BFloat16|bf16"),
        "bf16 dtype mismatch. Some older GPUs (Turing and below) don't support bf16.",
        [
            "Switch the model to fp16: change `precision` widget or pass `--force-fp16`.",
            "Upgrade torch: `pip install -U torch` (bf16 path improved in 2.4+).",
        ],
    ),

    # ── Files / models ─────────────────────────────────────────────
    Pattern(
        "safetensors_corrupt",
        ("Exception",),
        _r(r"PytorchStreamReader|safetensors_rust\.SafetensorError|HeaderTooLarge|MetadataIncomplete"),
        "The model file is corrupt or truncated — almost always an interrupted download.",
        [
            "Re-download the model. Compare file size against the source (HuggingFace / CivitAI lists exact bytes).",
            "If you used `git-lfs`, run `git lfs pull` to fetch the actual weights.",
            "Verify the SHA256 if the source publishes one.",
        ],
    ),
    Pattern(
        "state_dict_load",
        ("RuntimeError", "KeyError"),
        _r(r"Error\(s\) in loading state_dict|Missing key\(s\) in state_dict|Unexpected key"),
        "The model file structure does not match the loader's expected architecture "
        "— e.g. loading a Flux .safetensors into the SDXL Checkpoint Loader.",
        [
            "Use the loader matching this model family (Flux loader for Flux, SDXL for SDXL, etc.).",
            "If the model has 'fp8' / 'gguf' / 'nf4' in its name, use the matching loader.",
            "Some merged models save as VAE-baked-in — try a 'Checkpoint Loader (with config)' if you have a yaml.",
        ],
    ),
    Pattern(
        "missing_model_file",
        ("FileNotFoundError",),
        _r(r"No such file or directory|cannot find file|does not exist"),
        "A model / weight file is missing from the expected folder.",
        [
            "Check `ComfyUI/models/` for the file. Subfolders matter (`checkpoints/`, `vae/`, `loras/`, `clip/`, `controlnet/`).",
            "If you use `extra_model_paths.yaml`, confirm the path resolves on this OS.",
            "Manager → 'Install models' may have stalled mid-download — verify file size.",
        ],
    ),
    Pattern(
        "module_not_found",
        ("ModuleNotFoundError", "ImportError"),
        _r(r"No module named ['\"]?(?P<mod>[\w.]+)"),
        "A Python package required by this node is not installed in the ComfyUI venv.",
        [
            "From the ComfyUI portable folder run: `python_embeded\\python.exe -m pip install <module>`.",
            "If on Linux/Mac venv: `<venv>/bin/pip install <module>`.",
            "Some custom nodes ship a `requirements.txt` — `pip install -r requirements.txt`.",
            "`<module>` was extracted from the error message above.",
        ],
    ),
    Pattern(
        "dll_load_windows",
        ("OSError", "ImportError"),
        _r(r"DLL load failed|specified module could not be found.*\.dll"),
        "A native DLL (CUDA, MSVC runtime, ffmpeg, opencv) couldn't be loaded.",
        [
            "Reinstall the package that owns the DLL: `pip install --force-reinstall <package>`.",
            "Install the latest Microsoft Visual C++ Redistributable.",
            "If it's a CUDA DLL, your torch CUDA version may not match the driver.",
        ],
    ),

    # ── Wiring / dataflow ──────────────────────────────────────────
    Pattern(
        "none_attribute",
        ("AttributeError",),
        _r(r"'NoneType' object has no attribute"),
        "An upstream node returned `None` — most often a loader silently failed "
        "(missing model, wrong path) or a node has an unconnected required input.",
        [
            "Look at the node *upstream* of this one — not this node. Errors propagate.",
            "Check every loader's filename widget actually selected a file.",
            "If a loader has a 'use_cache' option, try toggling it off.",
        ],
    ),
    Pattern(
        "none_iterable",
        ("TypeError",),
        _r(r"NoneType.*not iterable|object is not iterable"),
        "Code tried to loop over `None` — usually `stitch_data` from a crop node "
        "that didn't run, or a SAM mask list that came back empty.",
        [
            "If the message mentions `stitch_data`: the InpaintCropPro node above must run successfully first.",
            "If SAM-related: the SAM loader / generator may have produced 0 masks — try a different prompt or lower threshold.",
            "Add an Insight Status node to confirm upstream actually executed.",
        ],
    ),
    Pattern(
        "type_image_mask",
        ("TypeError",),
        _r(r"got: \[.*\].*expected.*\[.*\]|argument.*to.*has invalid type"),
        "Wrong tensor type wired into an input — most often IMAGE wired into a MASK socket "
        "or vice versa.",
        [
            "IMAGE tensors are `(B,H,W,3)` float; MASK tensors are `(B,H,W)` float.",
            "Use a Convert IMAGE→MASK or MASK→IMAGE node at the boundary.",
        ],
    ),
    Pattern(
        "key_stitch_data",
        ("KeyError",),
        _r(r"['\"]stitch_data['\"]"),
        "An InpaintStitch / InpaintPasteBack node ran without its matching InpaintCropPro upstream.",
        [
            "Wire the `stitch_data` output of InpaintCropProMEC into this node.",
            "If you bypassed the crop node intentionally, use InpaintPasteBackMEC instead — it doesn't need stitch_data.",
        ],
    ),

    # ── Image / IO ─────────────────────────────────────────────────
    Pattern(
        "pil_truncated",
        ("OSError", "IOError"),
        _r(r"image file is truncated|cannot identify image file"),
        "An input image is corrupt or has an unsupported format.",
        [
            "Re-save the image as PNG or JPEG.",
            "Run `ImageOps.exif_transpose` upstream if it's a phone photo with EXIF rotation.",
        ],
    ),
    Pattern(
        "ffmpeg_missing",
        ("FileNotFoundError",),
        _r(r"ffmpeg.*not found|imageio.*ffmpeg"),
        "The video pipeline needs ffmpeg, which isn't on PATH.",
        [
            "Install: `pip install imageio-ffmpeg` (downloads a static binary).",
            "Or install ffmpeg system-wide and ensure `ffmpeg` is on PATH.",
        ],
    ),

    # ── ComfyUI-specific ───────────────────────────────────────────
    Pattern(
        "comfy_validate_inputs",
        ("Exception",),
        _r(r"Required input is missing|VALIDATE_INPUTS"),
        "ComfyUI rejected the graph because a required input slot is unconnected.",
        [
            "Look for the red dashed input on the failing node — that's the missing wire.",
            "If the input is optional in the docs but marked required in code, update the pack.",
        ],
    ),
    Pattern(
        "samplers_no_sigma",
        ("Exception",),
        _r(r"sigma.*not.*found|scheduler.*not"),
        "The sampler couldn't build a sigma schedule — usually wrong scheduler/sampler combo.",
        [
            "Try `karras` or `normal` scheduler with `dpmpp_2m_sde` sampler — a safe default.",
            "Some samplers (LCM, TCD) need their matching scheduler.",
        ],
    ),

    # ── Range / value ──────────────────────────────────────────────
    Pattern(
        "value_error_generic",
        ("ValueError",),
        _r(r"."),
        "A widget value is outside its allowed range or has the wrong format.",
        [
            "Re-read the widget min/max in the node's tooltip.",
            "If it's a string field, check for trailing whitespace or wrong separator.",
        ],
    ),
]


def match_pattern(exc_type: str, msg: str) -> Optional[Pattern]:
    for p in _get_patterns():
        if p.exc_types and exc_type not in p.exc_types and "Exception" not in p.exc_types:
            continue
        if p.message_re.search(msg):
            return p
    return None


# =====================================================================
# Settings store (read-only here; written by ComfyUI Settings UI)
# =====================================================================
_DEFAULTS = {
    "mode": "cloud_only",                     # auto|deterministic_only|local_only|cloud_only
    # New (v2): three independent toggles. `mode` is derived from them on save
    # for backward compatibility with all existing runtime code paths. On load,
    # if these are absent from the file we derive them from `mode`.
    "tier1_enabled": True,
    "tier2_enabled": False,
    "tier3_enabled": True,
    "cloud_provider": "openai",               # openai|anthropic|gemini|openrouter|groq|deepseek
    "cloud_model": "gpt-4o-mini",
    # API keys are stored encrypted via secrets_store.py — never in this dict
    # W5a: default local model is now Qwen3.5-4B Q4_K_M (Opus-reasoning distill,
    # Apache-2.0, ~2.5 GB CPU). local_llm's resolver also accepts any GGUF the
    # user already has; the C2C AI settings panel offers a one-click download.
    "local_model": "qwen3.5-4b",
    "local_threads": 0,                        # 0 = llama.cpp default
    # Tier 2 backend: "llamacpp" (local GGUF) or "ollama" (HTTP daemon).
    "tier2_backend": "llamacpp",
    "ollama_url": "http://localhost:11434",
    "ollama_model": "qwen3:4b",
    "stream": True,
    "max_tokens": 512,
    "include_traceback": True,                 # send last few frames to LLM
}


# Mode <-> tier-flags mapping (canonical):
#   T1 only           -> deterministic_only
#   T1+T2             -> local_only
#   T1+T3             -> cloud_only
#   T1+T2+T3 / T2+T3  -> auto
#   anything else     -> deterministic_only (Tier 1 always silently runs first)
def _flags_to_mode(t1: bool, t2: bool, t3: bool) -> str:
    if t2 and t3:
        return "auto"
    if t3:
        return "cloud_only"
    if t2:
        return "local_only"
    return "deterministic_only"


def _mode_to_flags(mode: str) -> Dict[str, bool]:
    return {
        "auto":               {"tier1_enabled": True,  "tier2_enabled": True,  "tier3_enabled": True},
        "deterministic_only": {"tier1_enabled": True,  "tier2_enabled": False, "tier3_enabled": False},
        "local_only":         {"tier1_enabled": True,  "tier2_enabled": True,  "tier3_enabled": False},
        "cloud_only":         {"tier1_enabled": True,  "tier2_enabled": False, "tier3_enabled": True},
    }.get(mode, {"tier1_enabled": True, "tier2_enabled": False, "tier3_enabled": True})


def _settings_path() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    pack_root = os.path.dirname(here)
    return os.path.join(pack_root, "user", "error_assistant.json")


def load_settings() -> Dict[str, Any]:
    import json
    p = _settings_path()
    if not os.path.exists(p):
        return dict(_DEFAULTS)
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        out = dict(_DEFAULTS)
        out.update({k: v for k, v in data.items() if k in _DEFAULTS})
        # Back-fill tier flags from `mode` if file pre-dates them.
        if not any(k in data for k in ("tier1_enabled", "tier2_enabled", "tier3_enabled")):
            out.update(_mode_to_flags(out.get("mode", "auto")))
        return out
    except Exception as e:
        log.warning("[error_assistant] settings unreadable: %s", e)
        return dict(_DEFAULTS)


def save_settings(new: Dict[str, Any]) -> None:
    import json
    p = _settings_path()
    os.makedirs(os.path.dirname(p), exist_ok=True)
    cur = load_settings()
    cur.update({k: v for k, v in new.items() if k in _DEFAULTS})
    # If the caller supplied tier flags, they win and override `mode`.
    if any(k in new for k in ("tier1_enabled", "tier2_enabled", "tier3_enabled")):
        cur["mode"] = _flags_to_mode(
            bool(cur.get("tier1_enabled", True)),
            bool(cur.get("tier2_enabled", False)),
            bool(cur.get("tier3_enabled", True)),
        )
    else:
        # Otherwise keep flags consistent with `mode`.
        cur.update(_mode_to_flags(cur.get("mode", "auto")))
    with open(p, "w", encoding="utf-8") as f:
        json.dump(cur, f, indent=2)


# =====================================================================
# Tier 2 / Tier 3 (lazy backends)
# =====================================================================
_TIER_LOCK = threading.Lock()
_local_backend = None  # llama.cpp instance
_cloud_backends: Dict[str, Any] = {}


def _build_prompt(exc_type: str, msg: str, node_class: Optional[str],
                  inputs_summary: Optional[str], traceback_tail: Optional[str]) -> str:
    """Compact, token-frugal prompt for both local and cloud LLMs."""
    parts = [
        "You are an expert ComfyUI debugger. A node just failed. Explain the cause "
        "in 2-3 sentences, then list 3 concrete fixes as bullet points. Be specific. "
        "Do not pad. No disclaimers.\n",
        f"Node: {node_class or 'unknown'}",
        f"Error: {exc_type}: {msg.strip()[:600]}",
    ]
    if inputs_summary:
        parts.append(f"Inputs: {inputs_summary[:400]}")
    if traceback_tail:
        parts.append(f"Traceback (tail):\n{traceback_tail.strip()[:1200]}")
    parts.append(
        "\nReply with two sections:\n"
        "CAUSE: <one paragraph>\n"
        "FIXES:\n- <fix 1>\n- <fix 2>\n- <fix 3>"
    )
    return "\n".join(parts)


def _record_tier_failure(key: str, exc: BaseException, *, hint: str | None = None) -> None:
    """Best-effort: surface tier-2/tier-3 failures in the Doctor panel.

    Falls back to a debug log if _c2c_registry is unavailable (e.g. during
    partial install). Never raises.
    """
    try:
        from . import _c2c_registry  # type: ignore
        _c2c_registry.record_failure(key, exc, hint=hint, group="error_assistant")
    except Exception:
        log.debug("[error_assistant] failure not recorded (registry missing): %s/%s", key, exc)


def _explain_local(prompt: str, settings: Dict[str, Any]) -> Optional[str]:
    """Run the user-selected local model. Backend = "llamacpp" or "ollama".
    Returns text or None if backend unavailable."""
    backend = (settings.get("tier2_backend") or "llamacpp").strip().lower()
    if backend == "ollama":
        try:
            from . import ollama_llm  # type: ignore
        except Exception:
            try:
                import importlib
                ollama_llm = importlib.import_module(
                    "nodes.ollama_llm", package=__package__)
            except Exception as e:
                log.info("[error_assistant] ollama_llm unavailable: %s", e)
                return None
        try:
            return ollama_llm.generate(
                model=settings.get("ollama_model") or "qwen3:4b",
                prompt=prompt,
                url=settings.get("ollama_url"),
                max_tokens=settings.get("max_tokens", 512),
            )
        except Exception as e:
            _record_tier_failure("tier2.ollama", e,
                                 hint="Is the ollama server running on the configured URL?")
            log.warning("[error_assistant] ollama gen failed: %s", e)
            return None
    # Default: llama-cpp-python with a local GGUF file.
    global _local_backend
    try:
        from . import local_llm  # type: ignore
    except Exception:
        # Lazy lookup of sibling module
        local_llm = None
        try:
            import importlib
            local_llm = importlib.import_module("nodes.local_llm",
                                                package=__package__)
        except Exception:
            try:
                from . import local_llm  # type: ignore
            except Exception:
                pass
    if local_llm is None:
        try:
            from . import local_llm  # type: ignore
        except Exception as e:
            log.info("[error_assistant] local_llm unavailable: %s", e)
            return None
    with _TIER_LOCK:
        if _local_backend is None:
            _local_backend = local_llm.get_or_load(
                model_id=settings.get("local_model"),
                n_threads=settings.get("local_threads") or 0,
            )
        if _local_backend is None:
            return None
    try:
        return _local_backend.generate(prompt,
                                       max_tokens=settings.get("max_tokens", 512))
    except Exception as e:
        _record_tier_failure("tier2.llamacpp", e,
                             hint="Check the GGUF path / llama-cpp-python install.")
        log.warning("[error_assistant] local model gen failed: %s", e)
        return None


def _explain_cloud(prompt: str, settings: Dict[str, Any]) -> Optional[str]:
    """Call the user-selected cloud LLM. API key fetched from encrypted store.
    Returns text or None if no key / backend missing."""
    try:
        from . import cloud_llm  # type: ignore
    except Exception:
        try:
            import importlib
            cloud_llm = importlib.import_module("nodes.cloud_llm",
                                                package=__package__)
        except Exception as e:
            log.info("[error_assistant] cloud_llm unavailable: %s", e)
            return None
    try:
        return cloud_llm.generate(
            provider=settings.get("cloud_provider", "openai"),
            model=settings.get("cloud_model", "gpt-4o-mini"),
            prompt=prompt,
            max_tokens=settings.get("max_tokens", 512),
        )
    except Exception as e:
        provider = settings.get("cloud_provider", "openai")
        _record_tier_failure(f"tier3.{provider}", e,
                             hint="Check the API key in the Settings panel and network connectivity.")
        log.warning("[error_assistant] cloud gen failed: %s", e)
        return None


# =====================================================================
# Public entry point
# =====================================================================
def explain(exc: BaseException,
            *,
            node_class: Optional[str] = None,
            inputs_summary: Optional[str] = None,
            traceback_tail: Optional[str] = None,
            mode: Optional[str] = None) -> Dict[str, Any]:
    """Return a structured explanation. Never raises."""
    exc_type = type(exc).__name__
    msg = str(exc) or repr(exc)
    settings = load_settings()
    if mode is None:
        mode = settings.get("mode", "auto")

    # --- Tier 1 always runs first (cheap context for LLM) ---
    p = match_pattern(exc_type, msg)
    tier1 = None
    if p is not None:
        tier1 = {
            "tier": 1,
            "headline": f"{exc_type}: {msg.strip()[:140]}",
            "cause": p.cause,
            "fixes": list(p.fixes),
            "pattern_id": p.name,
            "category": p.category,
            "confidence": p.confidence,
            "provenance": {
                "pack": "ComfyUI-CustomNodePacks",
                "source": p.source,
                "priority": p.priority,
            },
        }

    # --- Runtime tensor introspection (Tier 1.5) ---------------------
    # Pulls live tensor shapes / dtypes / device from the exception's
    # traceback frames. Always cheap; never raises. Augments tier1 when
    # available, becomes the primary answer when no rule pack pattern
    # fires.
    introspection_envelope = None
    try:
        from . import error_introspector  # type: ignore
        report = error_introspector.introspect_exception(
            exc,
            node_class=node_class,
        )
        introspection_envelope = error_introspector.format_report(report)
    except Exception as _ie:
        log.debug("[error_assistant] introspector skipped: %s", _ie)

    if tier1 is not None and introspection_envelope is not None:
        # Attach the runtime facts to the rule-pack hit so the UI can show
        # both the pattern explanation and concrete tensor shapes.
        tier1["introspection"] = introspection_envelope.get("introspection")
        if introspection_envelope.get("model_family"):
            tier1["model_family"] = introspection_envelope["model_family"]

    if mode == "deterministic_only":
        if tier1 is not None:
            return tier1
        if introspection_envelope is not None and (
                introspection_envelope.get("pattern_id") != "introspector_facts_only"
                or (introspection_envelope.get("introspection") or {}).get("frames")):
            return introspection_envelope
        return _fallback(exc_type, msg)

    # --- Build LLM prompt only if we'll actually call one ---
    prompt = _build_prompt(exc_type, msg, node_class, inputs_summary, traceback_tail)

    # --- Tier 3 (cloud) preferred if mode allows ---
    if mode in ("auto", "cloud_only"):
        try:
            from . import secrets_store  # type: ignore
            has_key = secrets_store.has_key_for(settings.get("cloud_provider", "openai"))
        except Exception:
            has_key = False
        if has_key:
            text = _explain_cloud(prompt, settings)
            if text:
                return _format_llm_result(text, tier=3, tier1=tier1, exc_type=exc_type, msg=msg,
                                          provider=settings.get("cloud_provider"),
                                          model=settings.get("cloud_model"))

    # --- Tier 2 (local) ---
    if mode in ("auto", "local_only"):
        text = _explain_local(prompt, settings)
        if text:
            backend = (settings.get("tier2_backend") or "llamacpp").lower()
            if backend == "ollama":
                t2_provider = "ollama"
                t2_model = settings.get("ollama_model") or "qwen3:4b"
            else:
                t2_provider = "local"
                t2_model = settings.get("local_model")
            return _format_llm_result(text, tier=2, tier1=tier1, exc_type=exc_type, msg=msg,
                                      provider=t2_provider,
                                      model=t2_model)

    # --- Fall back to Tier 1, then introspector, then generic fallback ---
    if tier1 is not None:
        return tier1
    if introspection_envelope is not None and (
            introspection_envelope.get("pattern_id") != "introspector_facts_only"
            or (introspection_envelope.get("introspection") or {}).get("frames")):
        return introspection_envelope
    return _fallback(exc_type, msg)


def _fallback(exc_type: str, msg: str) -> Dict[str, Any]:
    return {
        "tier": 1,
        "headline": f"{exc_type}: {msg.strip()[:140]}",
        "cause": "No specific pattern matched. Read the full traceback for details.",
        "fixes": [
            "Click 'Show Traceback' below to see the original Python error.",
            "Search the full error message in the ComfyUI Discord / GitHub issues.",
            "Enable Tier 2 (local model) or Tier 3 (cloud) in Settings → MEC Error Assistant for richer explanations.",
        ],
        "pattern_id": "no_match",
        "category": "uncategorized",
        "confidence": 0.0,
        "provenance": {"pack": "ComfyUI-CustomNodePacks", "source": "fallback"},
    }


def reload_patterns() -> int:
    """Force a re-scan of patterns/ pack files. Returns number of patterns loaded."""
    global _PATTERNS_CACHE, _PATTERNS_MTIME
    with _PATTERNS_LOCK:
        _PATTERNS_CACHE = []
        _PATTERNS_MTIME = {}
    return len(_get_patterns())


def _format_llm_result(text: str, *, tier: int, tier1: Optional[dict],
                       exc_type: str, msg: str, provider: str, model: str) -> Dict[str, Any]:
    """Parse loose CAUSE/FIXES sections out of the LLM's reply."""
    cause = text.strip()
    fixes: List[str] = []
    m = re.search(r"CAUSE:?\s*(.+?)(?:FIXES:?|$)", text, re.IGNORECASE | re.DOTALL)
    if m:
        cause = m.group(1).strip()
    fm = re.search(r"FIXES:?\s*(.+)", text, re.IGNORECASE | re.DOTALL)
    if fm:
        for line in fm.group(1).splitlines():
            line = line.strip().lstrip("-*•").strip()
            if line:
                fixes.append(line)
    if not fixes and tier1:
        fixes = list(tier1["fixes"])
    return {
        "tier": tier,
        "headline": f"{exc_type}: {msg.strip()[:140]}",
        "cause": cause[:1500],
        "fixes": fixes[:6],
        "provider": provider,
        "model": model,
        "tier1_match": (tier1 or {}).get("pattern_id"),
        "category": (tier1 or {}).get("category", "uncategorized"),
        "confidence": (tier1 or {}).get("confidence", 0.5),
        "provenance": {
            "pack": "ComfyUI-CustomNodePacks",
            "source": (tier1 or {}).get("provenance", {}).get("source", "llm"),
            "llm_provider": provider,
            "llm_model": model,
        },
    }
