"""Curated registry of GGUF models suitable for ComfyUI error/explainer tier-2.

Single source of truth for:
  * Which HuggingFace repos to pull from
  * Which quant variants are supported per model
  * Recommended ``n_gpu_layers`` for a fresh user (will be overridden by
    auto-tuner or user pick)
  * The chat-template format llama.cpp should use (``chatml`` / ``llama-3`` /
    ``phi`` / ``mistral``).

The previous design hard-coded a single (now mis-typed) Qwen3.5-2B repo in
``nodes/node_explain.py``; that constant is now sourced from here.

Anti-pattern this replaces (do NOT):
  * Hard-coding `unsloth/Qwen3.5-2B-GGUF` (the repo doesn't exist with that
    spelling).
  * Forcing `n_gpu_layers=0` (CPU-only) regardless of the user's GPU.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class GGUFModel:
    """Metadata for a single GGUF model family."""

    model_id: str               # short identifier used in URLs / settings
    display_name: str           # user-facing label
    hf_repo: str                # `org/repo` on HuggingFace
    quant_files: Dict[str, str]  # quant_label -> filename inside the repo
    default_quant: str          # which quant to pick when user says "auto"
    chat_format: str            # llama-cpp-python chat_format string
    recommended_n_gpu_layers: int  # -1 = offload everything, 0 = CPU-only
    context_window: int         # max n_ctx the GGUF was trained at
    notes: str = ""             # one-liner for the picker UI
    aliases: List[str] = field(default_factory=list)  # extra accepted ids


# ---------------------------------------------------------------------------
# Curated list. Order matters for "auto" pick (first match wins).
# All repos hand-verified to exist on HF as of D.5 (June 2026).
# ---------------------------------------------------------------------------
_REGISTRY: List[GGUFModel] = [
    # --- Qwen3 family (best price/perf for explainer use case) ----------
    GGUFModel(
        model_id="qwen3-4b-instruct",
        display_name="Qwen3 4B Instruct",
        hf_repo="bartowski/Qwen3-4B-Instruct-GGUF",
        quant_files={
            "Q4_K_M": "Qwen3-4B-Instruct-Q4_K_M.gguf",
            "Q5_K_M": "Qwen3-4B-Instruct-Q5_K_M.gguf",
            "Q8_0":   "Qwen3-4B-Instruct-Q8_0.gguf",
        },
        default_quant="Q4_K_M",
        chat_format="chatml",
        recommended_n_gpu_layers=-1,
        context_window=32_768,
        notes="2.5 GB Q4 — sweet spot for 8 GB GPUs; chain-of-thought capable.",
        aliases=["qwen3-4b"],
    ),
    GGUFModel(
        model_id="qwen3-8b-instruct",
        display_name="Qwen3 8B Instruct",
        hf_repo="bartowski/Qwen3-8B-Instruct-GGUF",
        quant_files={
            "Q4_K_M": "Qwen3-8B-Instruct-Q4_K_M.gguf",
            "Q5_K_M": "Qwen3-8B-Instruct-Q5_K_M.gguf",
            "Q8_0":   "Qwen3-8B-Instruct-Q8_0.gguf",
        },
        default_quant="Q4_K_M",
        chat_format="chatml",
        recommended_n_gpu_layers=-1,
        context_window=32_768,
        notes="5 GB Q4 — needs 10+ GB VRAM for full offload.",
        aliases=["qwen3-8b"],
    ),

    # --- Llama 3.2 (Meta, small) ----------------------------------------
    GGUFModel(
        model_id="llama-3.2-3b-instruct",
        display_name="Llama 3.2 3B Instruct",
        hf_repo="bartowski/Llama-3.2-3B-Instruct-GGUF",
        quant_files={
            "Q4_K_M": "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
            "Q5_K_M": "Llama-3.2-3B-Instruct-Q5_K_M.gguf",
            "Q8_0":   "Llama-3.2-3B-Instruct-Q8_0.gguf",
        },
        default_quant="Q4_K_M",
        chat_format="llama-3",
        recommended_n_gpu_layers=-1,
        context_window=131_072,
        notes="2 GB Q4 — long context (128k) but no chain-of-thought.",
        aliases=["llama-3.2-3b"],
    ),

    # --- Phi-3.5 (Microsoft, very small) --------------------------------
    GGUFModel(
        model_id="phi-3.5-mini-instruct",
        display_name="Phi-3.5 Mini Instruct",
        hf_repo="bartowski/Phi-3.5-mini-instruct-GGUF",
        quant_files={
            "Q4_K_M": "Phi-3.5-mini-instruct-Q4_K_M.gguf",
            "Q5_K_M": "Phi-3.5-mini-instruct-Q5_K_M.gguf",
            "Q8_0":   "Phi-3.5-mini-instruct-Q8_0.gguf",
        },
        default_quant="Q4_K_M",
        chat_format="phi-3",
        recommended_n_gpu_layers=-1,
        context_window=131_072,
        notes="2.3 GB Q4 — strongest reasoning per byte, tiny VRAM.",
        aliases=["phi-3.5-mini", "phi3.5-mini"],
    ),
]


# Lookup table built once at import time.
_BY_ID: Dict[str, GGUFModel] = {}
for _m in _REGISTRY:
    _BY_ID[_m.model_id] = _m
    for _alias in _m.aliases:
        _BY_ID[_alias] = _m


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def list_models() -> List[GGUFModel]:
    """Return every registered GGUF model in declaration order."""
    return list(_REGISTRY)


def get(model_id: str) -> Optional[GGUFModel]:
    """Look up a model by id or alias. Returns ``None`` if unknown."""
    return _BY_ID.get(model_id.strip().lower())


def default_model() -> GGUFModel:
    """Return the recommended default (first registry entry)."""
    return _REGISTRY[0]


def filename_for(model_id: str, quant: str) -> Optional[str]:
    """Return the GGUF filename inside the repo, or ``None``."""
    m = get(model_id)
    if m is None:
        return None
    return m.quant_files.get(quant) or m.quant_files.get(m.default_quant)


def hf_url_for(model_id: str, quant: str) -> Optional[str]:
    """Return the canonical ``resolve/main`` HuggingFace URL, or ``None``."""
    m = get(model_id)
    if m is None:
        return None
    fname = m.quant_files.get(quant) or m.quant_files.get(m.default_quant)
    if not fname:
        return None
    return f"https://huggingface.co/{m.hf_repo}/resolve/main/{fname}"


def to_dict_list() -> List[dict]:
    """JSON-serialisable form for the Settings UI / api routes."""
    return [
        {
            "model_id": m.model_id,
            "display_name": m.display_name,
            "hf_repo": m.hf_repo,
            "quants": list(m.quant_files.keys()),
            "default_quant": m.default_quant,
            "chat_format": m.chat_format,
            "recommended_n_gpu_layers": m.recommended_n_gpu_layers,
            "context_window": m.context_window,
            "notes": m.notes,
        }
        for m in _REGISTRY
    ]
