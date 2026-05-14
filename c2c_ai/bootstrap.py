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

from .backends import AnthropicBackend, OpenAICompatBackend
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
    elif kind == "local":
        b = OpenAICompatBackend.local(
            backend_id=entry.get("id", "local.custom"),
            display_name=entry.get("display_name", "Local server"),
            base_url=entry["base_url"],
            model=entry.get("model", "auto"),
            max_context=int(entry.get("max_context", 32_768)),
        )
    else:
        raise ValueError(f"unknown backend kind: {kind!r}")
    b.info.enabled = bool(entry.get("enabled", True))
    return b


def bootstrap() -> None:
    """Register every backend from ``ai_config.json`` into the router and
    start the periodic health probe. Idempotent — safe to call from
    ComfyUI's plugin init."""
    cfg = load_config()
    router = get_router()
    for entry in cfg.get("backends", []):
        try:
            backend = build_backend(entry)
            router.register(backend)
        except Exception as exc:
            log.error("failed to register %s: %s", entry, exc)
    router.start_periodic_probe(interval_s=60.0)
    log.info("c2c_ai bootstrap done: %d backends", len(router.all_backends()))
