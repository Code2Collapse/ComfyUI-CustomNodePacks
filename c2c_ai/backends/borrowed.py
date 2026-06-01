"""BorrowedEncoderBackend — Tier 2 (Track D.4).

Re-uses a generative text encoder that is **already loaded into VRAM** by
ComfyUI for image / video generation — so we get a local LLM answer for
"free" without holding any additional bytes of GPU memory.

When this works
---------------
The current ComfyUI checkpoint must have loaded a **decoder-style** text
encoder. The common case is:

  * Qwen-Image / Qwen-Image-Edit  →  Qwen2.5-VL family (decoder LM, ChatML).
  * Some HiDream / HunyuanVideo configurations that use Llama-3 as their
    text encoder.

T5 / UMT5 / CLIP are encoder-only and cannot ``generate()`` — we detect this
and stay out of the way (router falls through to the next tier).

Safety
------
* **Default OFF.** Users opt in via the settings toggle "Use my loaded text
  encoders for AI explanations (experimental)" — checked by the bootstrap
  before registering the backend.
* **Never holds the model lock** — we wrap each generate in a try/finally
  that releases the patcher's GPU lease immediately.
* **Never raises**: any failure (no loaded models, encoder isn't generative,
  tokenizer missing, OOM mid-generate) returns an empty ``AskResponse`` so
  the router can escalate to the next tier without surfacing the error.
* **Bounded output**: hard cap of 256 tokens per ask(); short, concrete
  answers only.
"""
from __future__ import annotations

import logging
import re
import threading
import time
from typing import Any, Generator, List, Optional, Tuple

from ..types import (
    AskRequest,
    AskResponse,
    BackendInfo,
    Capability,
    HealthState,
    Tier,
)
from .base import Backend, now_ms

log = logging.getLogger("c2c_ai.borrowed")


# Encoder-only families (cannot run .generate() — keep us silent on these).
_ENCODER_ONLY_HINTS = (
    "t5",            # T5/UMT5 (XXL/v1.1/etc.) — encoder-only for ComfyUI use.
    "umt5",
    "clip",          # CLIP-L / CLIP-G / OpenCLIP — encoder-only.
    "siglip",
    "byt5",
)

# Generative families we'll trust ``.generate()`` on. Probe by class name
# (case-insensitive substring) — works for HF transformers + the wrappers
# Comfy uses in qwen_image / hunyuan_video / hidream loaders.
_GENERATIVE_HINTS = (
    "qwen",
    "llama",
    "phi",
    "mistral",
    "gemma",
    "deepseek",
    "yi",
    "internlm",
)


# ─────────────────────────────────────────────────────────────────────────
# Loaded-model discovery
# ─────────────────────────────────────────────────────────────────────────
def _iter_loaded_text_encoders() -> List[Tuple[Any, Any, str]]:
    """Return ``[(loaded_model_wrapper, inner_module, class_name_lower), …]``.

    Looks at ``comfy.model_management.current_loaded_models``. Each entry is
    a ``LoadedModel`` whose ``.model`` is a ``ModelPatcher`` whose
    ``.model`` is the actual ``nn.Module``. We filter to those whose inner
    module class name suggests a *generative* text encoder.
    """
    out: List[Tuple[Any, Any, str]] = []
    try:
        import comfy.model_management as mm  # type: ignore[import-not-found]
    except Exception as exc:
        log.debug("borrowed: comfy.model_management import failed: %s", exc)
        return out
    try:
        loaded = list(getattr(mm, "current_loaded_models", []) or [])
    except Exception as exc:
        log.debug("borrowed: current_loaded_models read failed: %s", exc)
        return out

    for lm in loaded:
        try:
            patcher = getattr(lm, "model", None)
            inner = getattr(patcher, "model", None) if patcher is not None else None
            if inner is None:
                continue
            cls_name = type(inner).__name__.lower()
            mod_name = (getattr(type(inner), "__module__", "") or "").lower()
            sig = f"{cls_name}|{mod_name}"
            if any(h in sig for h in _ENCODER_ONLY_HINTS):
                continue
            if any(h in sig for h in _GENERATIVE_HINTS):
                # Confirm a generate method exists before claiming it.
                if hasattr(inner, "generate") and callable(getattr(inner, "generate")):
                    out.append((lm, inner, cls_name))
        except Exception:
            continue
    return out


def _find_tokenizer(inner: Any) -> Optional[Any]:
    """Hunt for an HF-style tokenizer attached to this model or its config."""
    for attr in ("tokenizer", "_tokenizer", "tok"):
        t = getattr(inner, attr, None)
        if t is not None and hasattr(t, "encode"):
            return t
    # Try the patcher / parent chain
    parent = getattr(inner, "parent", None)
    if parent is not None:
        for attr in ("tokenizer", "_tokenizer"):
            t = getattr(parent, attr, None)
            if t is not None and hasattr(t, "encode"):
                return t
    return None


# ─────────────────────────────────────────────────────────────────────────
# Prompt assembly — kept compact so even a 2B model gives useful output.
# ─────────────────────────────────────────────────────────────────────────
_SYS_DEFAULT = (
    "You are an expert ComfyUI debugger. Reply in at most 4 sentences. "
    "First sentence: the cause. Then up to 3 bullet fixes, each ≤120 chars. "
    "Plain English. No code fences. No disclaimers."
)


def _build_prompt(messages: List[Any]) -> Tuple[str, str]:
    """Return ``(system, user)`` strings from a Message list."""
    sys_parts: List[str] = []
    user_parts: List[str] = []
    for m in messages:
        role = getattr(m, "role", "user")
        content = getattr(m, "content", "")
        if role == "system":
            sys_parts.append(content)
        else:
            user_parts.append(content)
    system = "\n".join(sys_parts) or _SYS_DEFAULT
    user = "\n".join(user_parts).strip()
    return system, user


def _apply_chat_template(tokenizer: Any, system: str, user: str) -> str:
    """Use the tokenizer's chat template if available; else ChatML fallback."""
    try:
        if hasattr(tokenizer, "apply_chat_template"):
            return tokenizer.apply_chat_template(
                [{"role": "system", "content": system},
                 {"role": "user",   "content": user}],
                tokenize=False,
                add_generation_prompt=True,
            )
    except Exception as exc:
        log.debug("borrowed: apply_chat_template failed: %s", exc)
    # ChatML fallback (Qwen / many others speak this natively).
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


# ─────────────────────────────────────────────────────────────────────────
# Backend
# ─────────────────────────────────────────────────────────────────────────
class BorrowedEncoderBackend(Backend):
    """Generate via an already-loaded ComfyUI text encoder."""

    BACKEND_ID = "local.borrowed"
    MAX_OUTPUT_TOKENS = 256

    def __init__(self) -> None:
        super().__init__(
            BackendInfo(
                id=self.BACKEND_ID,
                tier=Tier.LOCAL,
                display_name="Borrowed text encoder (zero extra VRAM)",
                model="auto-detected",
                capabilities={Capability.CHAT},
                max_context=4_096,
                cost_per_1k_input=0.0,
                cost_per_1k_output=0.0,
                enabled=True,
            )
        )
        # Lock so two simultaneous explain calls don't fight over the GPU.
        # We hold this around the generate() call only — never across the
        # whole ask() so the cost meter / redactor stay non-blocking.
        self._gen_lock = threading.Lock()
        self.health = HealthState(ok=True, last_probe_at=time.time())

    # -------------------------------------------------------------- ask
    def ask(self, req: AskRequest) -> AskResponse:
        t0 = now_ms()
        empty = AskResponse(
            text="", backend_id=self.info.id, model=self.info.model,
            input_tokens=0, output_tokens=0, cost_usd=0.0,
            latency_ms=now_ms() - t0,
        )
        candidates = _iter_loaded_text_encoders()
        if not candidates:
            return empty
        _lm, inner, cls_name = candidates[0]
        tokenizer = _find_tokenizer(inner)
        if tokenizer is None:
            log.debug("borrowed: no tokenizer found on %s", cls_name)
            return empty

        # Reflect detected model in BackendInfo so callers see what we used.
        self.info.model = cls_name

        system, user = _build_prompt(req.messages)
        if not user:
            return empty
        prompt = _apply_chat_template(tokenizer, system, user)

        try:
            import torch  # type: ignore
        except Exception:
            return empty

        max_new = min(int(req.max_tokens or 256), self.MAX_OUTPUT_TOKENS)

        with self._gen_lock:
            try:
                # Find the device the model actually lives on.
                device = next(inner.parameters()).device
            except Exception:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            try:
                tok_out = tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max(256, self.info.max_context - max_new - 8),
                )
            except Exception as exc:
                log.debug("borrowed: tokenize failed: %s", exc)
                return empty
            # HF BatchEncoding supports .to(); plain dicts don't — move each
            # tensor explicitly so we work with either shape.
            try:
                if hasattr(tok_out, "to"):
                    input_ids = tok_out.to(device)
                elif isinstance(tok_out, dict):
                    input_ids = {k: (v.to(device) if hasattr(v, "to") else v)
                                 for k, v in tok_out.items()}
                else:
                    input_ids = tok_out
            except Exception as exc:
                log.debug("borrowed: .to(device) failed: %s", exc)
                return empty

            try:
                ids_tensor = input_ids["input_ids"]
                in_len = int(ids_tensor.shape[-1])
            except Exception:
                in_len = 0

            try:
                with torch.inference_mode():
                    out_ids = inner.generate(
                        **input_ids,
                        max_new_tokens=max_new,
                        do_sample=req.temperature > 0,
                        temperature=max(0.01, min(2.0, float(req.temperature))),
                        pad_token_id=getattr(tokenizer, "pad_token_id",
                                             getattr(tokenizer, "eos_token_id", None)),
                    )
            except Exception as exc:
                log.info("borrowed: generate failed (%s) — router will escalate", exc)
                self.health = HealthState(
                    ok=False, last_error=f"{type(exc).__name__}: {exc}",
                    last_rtt_ms=now_ms() - t0, last_probe_at=time.time(),
                )
                return empty

            try:
                gen_ids = out_ids[0, in_len:]
                text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            except Exception as exc:
                log.debug("borrowed: decode failed: %s", exc)
                return empty

        # Strip Qwen3 <think> blocks if present.
        text = _strip_think(text)

        # Record success in health so the picker shows green.
        self.health = HealthState(
            ok=True, last_rtt_ms=now_ms() - t0,
            last_probe_at=time.time(),
        )

        return AskResponse(
            text=text,
            backend_id=self.info.id,
            model=cls_name,
            input_tokens=in_len,
            output_tokens=int(out_ids.shape[-1]) - in_len if out_ids is not None else 0,
            cost_usd=0.0,
            latency_ms=now_ms() - t0,
        )

    # ------------------------------------------------------------ probe
    def probe(self, timeout: float = 5.0) -> HealthState:
        """Healthy = at least one generative encoder is currently loaded."""
        t0 = now_ms()
        candidates = _iter_loaded_text_encoders()
        if candidates:
            self.info.model = candidates[0][2]
            self.health = HealthState(
                ok=True, last_rtt_ms=now_ms() - t0,
                last_probe_at=time.time(),
            )
        else:
            self.health = HealthState(
                ok=False, last_rtt_ms=now_ms() - t0,
                last_error="no generative text-encoder currently loaded",
                last_probe_at=time.time(),
            )
        return self.health

    def list_models(self, timeout: float = 5.0) -> list[str]:
        cands = _iter_loaded_text_encoders()
        if not cands:
            return [self.info.model]
        # de-dup while preserving order
        seen: set[str] = set()
        out: list[str] = []
        for _, _, name in cands:
            if name in seen:
                continue
            seen.add(name)
            out.append(name)
        return out


# ─────────────────────────────────────────────────────────────────────────
# Qwen3-style <think>...</think> stripper
# Duplicates node_explain._THINK_RE for now — Track D.5 will centralise it
# in c2c_ai/utils/qwen3_filter.py and have router._dispatch apply it to all
# backend responses.
# ─────────────────────────────────────────────────────────────────────────
_THINK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)


def _strip_think(text: str) -> str:
    return _THINK_RE.sub("", text or "").strip()
