"""Bootstrap — load saved AI configuration and register backends on startup.

Config file: ``~/.c2c/ai_config.json``. Created by the first-run wizard;
hand-editable.

Schema::

    {
      "version": 1,
      "backends": [
        {"kind": "anthropic", "id": "cloud.anthropic", "model": "claude-3-5-sonnet-latest", "enabled": true},
        {"kind": "openai",    "id": "cloud.openai",    "model": "gpt-4o-mini", "enabled": true},
        {"kind": "qwen",      "id": "cloud.qwen",      "model": "qwen-max-latest", "enabled": false},
        {"kind": "openrouter","id": "cloud.openrouter","model": "anthropic/claude-3.5-sonnet", "enabled": false},
        {"kind": "local",     "id": "local.ollama",    "base_url": "http://127.0.0.1:11434",
                              "display_name": "Ollama", "model": "qwen3.5:9b",
                              "max_context": 32768, "enabled": true}
      ],
      "first_run_completed": true
    }
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from .backends import (
    AnthropicBackend,
    OpenAICompatBackend,
    GeminiBackend,
    CohereBackend,
    AzureOpenAIBackend,
    LlamaCppBackend,
)
from .router import get_router

log = logging.getLogger("c2c_ai.bootstrap")


def _config_dir() -> Path:
    base = os.environ.get("C2C_HOME") or str(Path.home() / ".c2c")
    p = Path(base)
    p.mkdir(parents=True, exist_ok=True)
    return p


AI_CONFIG = _config_dir() / "ai_config.json"


def load_config() -> dict:
    if not AI_CONFIG.is_file():
        return {"version": 1, "backends": [], "first_run_completed": False}
    try:
        with open(AI_CONFIG, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as exc:
        log.error("ai_config.json corrupt: %s — starting fresh", exc)
        return {"version": 1, "backends": [], "first_run_completed": False}


def save_config(cfg: dict) -> None:
    cfg.setdefault("version", 1)
    with open(AI_CONFIG, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2)


def first_run_completed() -> bool:
    return bool(load_config().get("first_run_completed"))


def build_backend(entry: dict):
    kind = entry.get("kind")
    if kind == "anthropic":
        b = AnthropicBackend.build(
            backend_id=entry.get("id", "cloud.anthropic"),
            model=entry.get("model", "claude-3-5-sonnet-latest"),
            display_name=entry.get("display_name", "Claude (Anthropic)"),
            cost_per_1k_input=float(entry.get("cost_per_1k_input", 0.003)),
            cost_per_1k_output=float(entry.get("cost_per_1k_output", 0.015)),
            max_context=int(entry.get("max_context", 200_000)),
        )
    elif kind == "openai":
        b = OpenAICompatBackend.openai(
            model=entry.get("model", "gpt-4o-mini"),
            cost_per_1k_input=float(entry.get("cost_per_1k_input", 0.00015)),
            cost_per_1k_output=float(entry.get("cost_per_1k_output", 0.0006)),
        )
    elif kind == "qwen":
        b = OpenAICompatBackend.qwen(
            model=entry.get("model", "qwen-max-latest"),
            cost_per_1k_input=float(entry.get("cost_per_1k_input", 0.0008)),
            cost_per_1k_output=float(entry.get("cost_per_1k_output", 0.002)),
        )
    elif kind == "openrouter":
        b = OpenAICompatBackend.openrouter(model=entry.get("model", "anthropic/claude-3.5-sonnet"))
    elif kind == "xai":
        b = OpenAICompatBackend.xai(
            model=entry.get("model", "grok-3-mini-fast"),
            cost_per_1k_input=float(entry.get("cost_per_1k_input", 0.0003)),
            cost_per_1k_output=float(entry.get("cost_per_1k_output", 0.0005)),
        )
    elif kind == "groq":
        b = OpenAICompatBackend.groq(
            model=entry.get("model", "llama-3.3-70b-versatile"),
            cost_per_1k_input=float(entry.get("cost_per_1k_input", 0.00059)),
            cost_per_1k_output=float(entry.get("cost_per_1k_output", 0.00079)),
        )
    elif kind == "mistral":
        b = OpenAICompatBackend.mistral(
            model=entry.get("model", "mistral-small-latest"),
            cost_per_1k_input=float(entry.get("cost_per_1k_input", 0.0002)),
            cost_per_1k_output=float(entry.get("cost_per_1k_output", 0.0006)),
        )
    elif kind == "deepseek":
        b = OpenAICompatBackend.deepseek(
            model=entry.get("model", "deepseek-chat"),
            cost_per_1k_input=float(entry.get("cost_per_1k_input", 0.00014)),
            cost_per_1k_output=float(entry.get("cost_per_1k_output", 0.00028)),
        )
    elif kind == "together":
        b = OpenAICompatBackend.together(
            model=entry.get("model", "meta-llama/Llama-3.3-70B-Instruct-Turbo"),
            cost_per_1k_input=float(entry.get("cost_per_1k_input", 0.00088)),
            cost_per_1k_output=float(entry.get("cost_per_1k_output", 0.00088)),
        )
    elif kind == "fireworks":
        b = OpenAICompatBackend.fireworks(
            model=entry.get("model", "accounts/fireworks/models/llama-v3p3-70b-instruct"),
            cost_per_1k_input=float(entry.get("cost_per_1k_input", 0.0009)),
            cost_per_1k_output=float(entry.get("cost_per_1k_output", 0.0009)),
        )
    elif kind == "perplexity":
        b = OpenAICompatBackend.perplexity(
            model=entry.get("model", "sonar"),
            cost_per_1k_input=float(entry.get("cost_per_1k_input", 0.001)),
            cost_per_1k_output=float(entry.get("cost_per_1k_output", 0.001)),
        )
    elif kind == "local":
        b = OpenAICompatBackend.local(
            backend_id=entry.get("id", "local.custom"),
            display_name=entry.get("display_name", "Local server"),
            base_url=entry["base_url"],
            model=entry.get("model", "auto"),
            max_context=int(entry.get("max_context", 32_768)),
        )
    elif kind == "gemini":
        b = GeminiBackend.build(
            backend_id=entry.get("id", "cloud.gemini"),
            model=entry.get("model", "gemini-1.5-flash-latest"),
            display_name=entry.get("display_name", "Gemini (Google)"),
            cost_per_1k_input=float(entry.get("cost_per_1k_input", 0.000075)),
            cost_per_1k_output=float(entry.get("cost_per_1k_output", 0.0003)),
            max_context=int(entry.get("max_context", 1_000_000)),
        )
    elif kind == "cohere":
        b = CohereBackend.build(
            backend_id=entry.get("id", "cloud.cohere"),
            model=entry.get("model", "command-r-08-2024"),
            display_name=entry.get("display_name", "Command R (Cohere)"),
            cost_per_1k_input=float(entry.get("cost_per_1k_input", 0.00015)),
            cost_per_1k_output=float(entry.get("cost_per_1k_output", 0.0006)),
            max_context=int(entry.get("max_context", 128_000)),
        )
    elif kind == "azure_openai":
        ep = entry.get("endpoint") or ""
        dep = entry.get("deployment") or ""
        if not ep or not dep:
            raise ValueError("azure_openai backend requires 'endpoint' and "
                             "'deployment' fields")
        b = AzureOpenAIBackend.build(
            backend_id=entry.get("id", "cloud.azure_openai"),
            endpoint=ep,
            deployment=dep,
            api_version=entry.get("api_version", "2024-02-15-preview"),
            model=entry.get("model", "gpt-4o-mini"),
            display_name=entry.get("display_name", "Azure OpenAI"),
            cost_per_1k_input=float(entry.get("cost_per_1k_input", 0.00015)),
            cost_per_1k_output=float(entry.get("cost_per_1k_output", 0.0006)),
            max_context=int(entry.get("max_context", 128_000)),
        )
    elif kind == "text_encoder_local":
        # B1: local GGUF via llama-cpp-python, loaded from ComfyUI's
        # text_encoders/ folder via folder_paths (per user mandate).
        b = LlamaCppBackend.build(
            backend_id=entry.get("id", "local.llamacpp"),
            model_file=entry.get("model_file", ""),
            display_name=entry.get("display_name", "Local GGUF (llama.cpp)"),
            n_ctx=int(entry.get("n_ctx", 8192)),
            n_gpu_layers=int(entry.get("n_gpu_layers", -1)),
            chat_format=entry.get("chat_format") or None,
        )
    else:
        raise ValueError(f"unknown backend kind: {kind!r}")
    b.info.enabled = bool(entry.get("enabled", True))
    return b


def _probe_ollama(base_url: str = "http://127.0.0.1:11434", timeout: float = 1.5) -> list[str]:
    """Return the list of installed Ollama model tags, or [] if Ollama is not
    reachable. Used by zero-config auto-discovery so a user with Ollama
    running gets working AI features without editing ai_config.json."""
    try:
        import urllib.request
        req = urllib.request.Request(
            f"{base_url.rstrip('/')}/api/tags",
            headers={"Accept": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read().decode("utf-8", errors="replace"))
        models = data.get("models") or []
        return [m.get("name") for m in models if isinstance(m, dict) and m.get("name")]
    except Exception:
        return []


def _auto_discover_backends() -> list[dict]:
    """Build seed entries from environment variables + local probes.

    Per user mandate 2026-05-19 (no stubs, no zero-backend logs):
      * If OPENAI_API_KEY / ANTHROPIC_API_KEY / GEMINI_API_KEY / COHERE_API_KEY
        is set, register the corresponding cloud backend with a sensible
        default model.
      * If Ollama answers at 127.0.0.1:11434, register its first model as
        ``local.ollama``.
    Returns the list of entries actually added (already filtered for
    success).
    """
    out: list[dict] = []

    # Cloud backends gated by env keys
    if os.environ.get("ANTHROPIC_API_KEY"):
        out.append({"kind": "anthropic", "id": "cloud.anthropic",
                    "model": "claude-3-5-sonnet-latest", "enabled": True,
                    "_auto": True})
    if os.environ.get("OPENAI_API_KEY"):
        out.append({"kind": "openai", "id": "cloud.openai",
                    "model": "gpt-4o-mini", "enabled": True, "_auto": True})
    if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
        out.append({"kind": "gemini", "id": "cloud.gemini",
                    "model": "gemini-1.5-flash-latest", "enabled": True,
                    "_auto": True})
    if os.environ.get("COHERE_API_KEY"):
        out.append({"kind": "cohere", "id": "cloud.cohere",
                    "model": "command-r-08-2024", "enabled": True, "_auto": True})
    if os.environ.get("OPENROUTER_API_KEY"):
        out.append({"kind": "openrouter", "id": "cloud.openrouter",
                    "model": "anthropic/claude-3.5-sonnet", "enabled": True,
                    "_auto": True})
    if os.environ.get("DASHSCOPE_API_KEY"):
        out.append({"kind": "qwen", "id": "cloud.qwen",
                    "model": "qwen-max-latest", "enabled": True, "_auto": True})
    if os.environ.get("XAI_API_KEY") or os.environ.get("GROK_API_KEY"):
        out.append({"kind": "xai", "id": "cloud.xai",
                    "model": "grok-3-mini-fast", "enabled": True, "_auto": True})
    if os.environ.get("GROQ_API_KEY"):
        out.append({"kind": "groq", "id": "cloud.groq",
                    "model": "llama-3.3-70b-versatile", "enabled": True, "_auto": True})
    if os.environ.get("MISTRAL_API_KEY"):
        out.append({"kind": "mistral", "id": "cloud.mistral",
                    "model": "mistral-small-latest", "enabled": True, "_auto": True})
    if os.environ.get("DEEPSEEK_API_KEY"):
        out.append({"kind": "deepseek", "id": "cloud.deepseek",
                    "model": "deepseek-chat", "enabled": True, "_auto": True})
    if os.environ.get("TOGETHER_API_KEY") or os.environ.get("TOGETHERAI_API_KEY"):
        out.append({"kind": "together", "id": "cloud.together",
                    "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
                    "enabled": True, "_auto": True})
    if os.environ.get("FIREWORKS_API_KEY"):
        out.append({"kind": "fireworks", "id": "cloud.fireworks",
                    "model": "accounts/fireworks/models/llama-v3p3-70b-instruct",
                    "enabled": True, "_auto": True})
    if os.environ.get("PERPLEXITY_API_KEY") or os.environ.get("PPLX_API_KEY"):
        out.append({"kind": "perplexity", "id": "cloud.perplexity",
                    "model": "sonar", "enabled": True, "_auto": True})

    # B1: Local GGUF in ComfyUI/models/text_encoders/ — picked up via
    # folder_paths so extra_model_paths.yaml entries also count.
    try:
        import folder_paths  # type: ignore
        files = folder_paths.get_filename_list("text_encoders") or []
        ggufs = [f for f in files if isinstance(f, str) and f.lower().endswith(".gguf")]
        if ggufs:
            # Prefer a chat-instruction-tuned model if naming hints exist.
            preferred = next(
                (f for f in ggufs
                 if any(t in f.lower() for t in ("instruct", "chat", "qwen", "llama", "mistral", "phi"))),
                ggufs[0],
            )
            out.append({"kind": "text_encoder_local",
                        "id": "local.llamacpp",
                        "display_name": "Local GGUF (llama.cpp)",
                        "model_file": preferred,
                        "n_ctx": 8192,
                        "n_gpu_layers": -1,
                        "enabled": True,
                        "_auto": True})
    except Exception:
        # ComfyUI not importable yet (e.g. unit tests) — skip silently.
        pass

    # Local Ollama probe
    ollama_models = _probe_ollama()
    if ollama_models:
        # Prefer a chat-capable Qwen/Llama if installed; otherwise first.
        preferred = next(
            (m for m in ollama_models
             if any(tag in m.lower() for tag in ("qwen", "llama", "mistral", "phi"))),
            ollama_models[0],
        )
        out.append({"kind": "local", "id": "local.ollama",
                    "display_name": "Ollama (local)",
                    "base_url": "http://127.0.0.1:11434",
                    "model": preferred,
                    "max_context": 32768,
                    "enabled": True,
                    "_auto": True})

    return out


def bootstrap() -> None:
    """Register every backend from ``ai_config.json`` into the router and
    start the periodic health probe. Idempotent — safe to call from
    ComfyUI's plugin init.

    If the config file has zero usable backends, try environment-based
    auto-discovery (Ollama + cloud API keys) so the user gets a working
    AI spine on first run without manual config editing. Discovered
    entries are persisted to ``ai_config.json`` so the user can see/edit
    them.
    """
    cfg = load_config()
    entries = list(cfg.get("backends") or [])

    # Self-heal: an older build of the JS first-run wizard persisted
    # ``first_run_completed: true`` even when the user clicked "Skip" with
    # zero backends. That permanently suppressed the wizard and left the AI
    # spine silently empty. Reset the flag so the UI shows the setup prompt
    # again.
    if not entries and cfg.get("first_run_completed"):
        cfg["first_run_completed"] = False
        try:
            save_config(cfg)
            log.info("c2c_ai: cleared stale first_run_completed=true "
                     "(no backends were ever configured)")
        except Exception as exc:
            log.warning("c2c_ai: failed to clear stale first_run flag: %s", exc)

    if not entries:
        discovered = _auto_discover_backends()
        if discovered:
            entries = discovered
            cfg["backends"] = discovered
            try:
                save_config(cfg)
                log.info("c2c_ai: auto-discovered %d backend(s) — saved to %s",
                         len(discovered), AI_CONFIG)
            except Exception as exc:
                log.warning("c2c_ai: failed to persist auto-discovered config: %s", exc)

    router = get_router()
    # Track D.1 — always register the deterministic rule-pack backend FIRST
    # so error-explanation features have a guaranteed fallback even when zero
    # LLM backends are configured. Cost = $0, runs offline.
    try:
        from .backends.rulepack import RulePackBackend
        router.register(RulePackBackend())
    except Exception as exc:
        log.warning("c2c_ai: failed to register RulePackBackend: %s", exc)
    # Track D.4 — optionally register the borrowed-encoder backend. Off by
    # default; enable via ai_config.json {"borrowed_encoder_enabled": true}
    # or the Settings toggle (which writes the same key). When ON, the
    # router will route LOCAL-tier requests through whatever generative
    # text encoder ComfyUI happens to have loaded for image/video gen.
    if bool(cfg.get("borrowed_encoder_enabled", False)):
        try:
            from .backends.borrowed import BorrowedEncoderBackend
            router.register(BorrowedEncoderBackend())
            log.info("c2c_ai: BorrowedEncoderBackend enabled (experimental)")
        except Exception as exc:
            log.warning("c2c_ai: failed to register BorrowedEncoderBackend: %s", exc)
    registered = 0
    for entry in entries:
        try:
            backend = build_backend(entry)
            router.register(backend)
            registered += 1
        except Exception as exc:
            log.error("failed to register %s: %s", entry, exc)
    router.start_periodic_probe(interval_s=60.0)
    if registered:
        log.info("c2c_ai bootstrap done: %d backends registered", registered)
    else:
        log.info("c2c_ai bootstrap done: 0 backends "
                 "(set OPENAI_API_KEY / ANTHROPIC_API_KEY / GEMINI_API_KEY / "
                 "XAI_API_KEY / GROQ_API_KEY / MISTRAL_API_KEY / DEEPSEEK_API_KEY / "
                 "TOGETHER_API_KEY / FIREWORKS_API_KEY / PERPLEXITY_API_KEY or "
                 "run Ollama at 127.0.0.1:11434, then restart ComfyUI)")
