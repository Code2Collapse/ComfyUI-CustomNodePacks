"""BorrowedEncoderBackend — Tier LOCAL: chat with a text encoder ComfyUI has ALREADY loaded (ledger L6.02 S3).

ComfyUI keeps the text encoder of the current workflow loaded (``comfy.model_management.current_loaded_models``).
When that encoder is one ComfyUI can chat with (a Qwen 3 / Qwen 3-VL encoder - the header scan's ``can_chat``,
c2c_ai/te_scan.py), the answer costs no extra memory: the loaded weights generate it through ComfyUI's own
``CLIP.tokenize -> generate -> decode`` (shared with CoreTextEncoderBackend: ``chat_with_clip``).

How the loaded encoder is found (no model code, no extra load):

* a loaded ModelPatcher whose reload recipe (``cached_patcher_init``) is core's ``load_clip_model_patcher`` names
  the file it came from; the header scan decides whether ComfyUI can chat with that file;
* the CLIP object that owns the patcher holds the tokenizer; it is found among the patcher's referrers (the CLIP
  object stays alive while the workflow's outputs are cached). If it is gone, nothing is borrowed.

The earlier version called ``inner.generate(**input_ids)`` with a Hugging Face tokenizer it looked for on the module:
ComfyUI's text-encoder modules have neither, so it could not produce text (L6.02 design, section 3).

Safety (unchanged contract): default OFF (``ai_config.json {"borrowed_encoder_enabled": true}``); never raises - any
failure returns an empty answer so the router escalates to the next tier; one generate at a time; output capped at
256 tokens; text only.
"""
from __future__ import annotations

import gc
import logging
import os
import threading
import time
from typing import Any, List, Optional, Tuple

from ..types import AskRequest, AskResponse, BackendInfo, Capability, HealthState, Tier
from .base import Backend, check_cancel, now_ms, refuse_images

log = logging.getLogger("c2c_ai.borrowed")

_LOADER_NAME = "load_clip_model_patcher"          # comfy/sd.py (core 0.39): the reload recipe of a loaded CLIP


def _loaded_patchers() -> List[Any]:
    try:
        import comfy.model_management as mm  # type: ignore[import-not-found]
        return [getattr(lm, "model", None) for lm in list(getattr(mm, "current_loaded_models", []) or [])]
    except Exception as exc:  # noqa: BLE001 - outside ComfyUI
        log.debug("borrowed: no loaded models (%s)", exc)
        return []


def _source_file(patcher: Any) -> Optional[str]:
    """The single file a loaded text encoder came from, from core's own reload recipe."""
    init = getattr(patcher, "cached_patcher_init", None)
    if not init or getattr(init[0], "__name__", "") != _LOADER_NAME:
        return None
    paths = list((init[1] or [None])[0] or [])
    if len(paths) != 1 or not os.path.isfile(str(paths[0])):
        return None                                   # multi-file encoders (dual CLIP etc.) cannot chat
    return str(paths[0])


def _owner_clip(patcher: Any) -> Optional[Any]:
    """The CLIP object whose .patcher is this patcher (it carries the tokenizer)."""
    def is_clip(o):
        return (getattr(o, "patcher", None) is patcher and callable(getattr(o, "tokenize", None))
                and callable(getattr(o, "generate", None)) and callable(getattr(o, "decode", None)))
    for ref in gc.get_referrers(patcher):
        if is_clip(ref):
            return ref
        if isinstance(ref, dict) and ref.get("patcher") is patcher:          # the instance __dict__
            for owner in gc.get_referrers(ref):
                if getattr(owner, "__dict__", None) is ref and is_clip(owner):
                    return owner
    return None


def find_loaded_chat_encoders() -> List[Tuple[Any, str, dict]]:
    """[(clip, file, header classification)] for every loaded encoder ComfyUI can chat with."""
    from .. import te_scan
    out = []
    for patcher in _loaded_patchers():
        if patcher is None:
            continue
        path = _source_file(patcher)
        if not path:
            continue
        try:
            c = te_scan.classify_cached(path)
        except Exception as exc:  # noqa: BLE001
            log.debug("borrowed: header scan failed for %s: %s", path, exc)
            continue
        if not c.get("can_chat"):
            continue
        clip = _owner_clip(patcher)
        if clip is not None:
            out.append((clip, path, c))
    return out


class BorrowedEncoderBackend(Backend):
    """Generate with a text encoder that ComfyUI has already loaded."""

    BACKEND_ID = "local.borrowed"
    MAX_OUTPUT_TOKENS = 256

    def __init__(self) -> None:
        super().__init__(BackendInfo(id=self.BACKEND_ID, tier=Tier.LOCAL,
                                     display_name="Borrowed text encoder (zero extra VRAM)", model="auto-detected",
                                     capabilities={Capability.CHAT}, max_context=4_096, cost_per_1k_input=0.0,
                                     cost_per_1k_output=0.0, enabled=True))
        self._gen_lock = threading.Lock()
        self.health = HealthState(ok=True, last_probe_at=time.time())

    def ask(self, req: AskRequest) -> AskResponse:
        from .core_text_encoder import chat_with_clip
        check_cancel(req)
        refuse_images(req, "The borrowed text encoder")
        t0 = now_ms()
        empty = AskResponse(text="", backend_id=self.info.id, model=self.info.model, input_tokens=0,
                            output_tokens=0, cost_usd=0.0, latency_ms=0)
        try:
            found = find_loaded_chat_encoders()
        except Exception as exc:  # noqa: BLE001 - never raises
            log.debug("borrowed: discovery failed: %s", exc)
            found = []
        if not found:
            self.health = HealthState(ok=False, last_error="no loaded text encoder ComfyUI can chat with",
                                      last_rtt_ms=now_ms() - t0, last_probe_at=time.time())
            return empty
        clip, path, _c = found[0]
        self.info.model = os.path.basename(path)
        req.max_tokens = min(int(req.max_tokens or self.MAX_OUTPUT_TOKENS), self.MAX_OUTPUT_TOKENS)
        with self._gen_lock:
            try:
                out = chat_with_clip(clip, req, self.info.id, self.info.model, vision=False)
            except Exception as exc:  # noqa: BLE001 - the router escalates
                if type(exc).__name__ == "Cancelled":
                    raise
                log.info("borrowed: generate failed (%s) - router will escalate", exc)
                self.health = HealthState(ok=False, last_error=f"{type(exc).__name__}: {exc}",
                                          last_rtt_ms=now_ms() - t0, last_probe_at=time.time())
                return empty
        self.health = HealthState(ok=True, last_rtt_ms=now_ms() - t0, last_probe_at=time.time())
        return out

    def probe(self, timeout: float = 5.0) -> HealthState:
        """Healthy = a loaded encoder ComfyUI can chat with is present (no generation)."""
        t0 = now_ms()
        try:
            found = find_loaded_chat_encoders()
        except Exception as exc:  # noqa: BLE001
            found, err = [], str(exc)
        else:
            err = "no loaded text encoder ComfyUI can chat with"
        if found:
            self.info.model = os.path.basename(found[0][1])
            self.health = HealthState(ok=True, last_rtt_ms=now_ms() - t0, last_probe_at=time.time())
        else:
            self.health = HealthState(ok=False, last_error=err, last_rtt_ms=now_ms() - t0, last_probe_at=time.time())
        return self.health

    def list_models(self, timeout: float = 5.0) -> list[str]:
        names = []
        for _clip, path, _c in find_loaded_chat_encoders():
            n = os.path.basename(path)
            if n not in names:
                names.append(n)
        return names or [self.info.model]
