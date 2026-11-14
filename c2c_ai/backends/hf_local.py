"""Tier LOCAL: chat with a Hugging Face model folder under models/LLM (ledger L6.02 S4).

Config entry::

    {"kind": "hf_local", "id": "local.hf", "model_dir": "Qwen3-4B-Instruct", "device": "cpu", "keep_loaded_min": 5}

- ``model_dir``: a folder name under ComfyUI's LLM folders (c2c_ai/hf_scan.py) or an absolute path. Only folders the
  scan marks ``can_chat`` are accepted.
- Loaded on first use with ``transformers`` (``local_files_only=True``: nothing is ever downloaded), on ``device``
  (default ``cpu``, so the GPU stays free for image / video models), and released after ``keep_loaded_min`` minutes
  without a request.
- The conversation goes through the tokenizer's own chat template; ``<think>`` reasoning is split off; ``cancel`` /
  timeout are checked before the (uninterruptible) generate.
"""
from __future__ import annotations

import logging
import os
import threading
from typing import Optional

from .base import Backend, check_cancel, now_ms, refuse_images
from ..types import AskRequest, AskResponse, BackendInfo, Capability, HealthState, Tier

log = logging.getLogger("c2c_ai.hf_local")


def resolve(model_dir: str) -> str:
    if os.path.isabs(model_dir) and os.path.isdir(model_dir):
        return model_dir
    from .. import hf_scan
    for root in hf_scan.roots():
        p = os.path.join(root, model_dir)
        if os.path.isdir(p):
            return p
    raise RuntimeError(f"model folder {model_dir!r} is not in ComfyUI's LLM folders (models/LLM)")


class HFLocalBackend(Backend):
    def __init__(self, info: BackendInfo, *, path: str, device: str = "cpu", keep_loaded_min: float = 5.0):
        super().__init__(info)
        self._path, self._device = path, str(device or "cpu")
        self._keep_s = max(0.0, float(keep_loaded_min)) * 60.0
        self._model = self._tok = None
        self._timer: Optional[threading.Timer] = None
        self._lock = threading.Lock()
        self.loads = 0

    @classmethod
    def build(cls, backend_id: str = "local.hf", model_dir: str = "", display_name: str = "Local model folder",
              device: str = "cpu", keep_loaded_min: float = 5.0) -> "HFLocalBackend":
        if not model_dir:
            raise ValueError("hf_local needs model_dir (a folder under models/LLM)")
        from .. import hf_scan
        path = resolve(model_dir)
        c = hf_scan.classify_cached(path)
        if not c["can_chat"]:
            raise ValueError(f"{os.path.basename(path)}: {c['why']}")
        name = os.path.basename(path.rstrip("\\/"))
        info = BackendInfo(id=backend_id, tier=Tier.LOCAL, display_name=f"{display_name} - {name}", model=name,
                           capabilities={Capability.CHAT}, max_context=8192)
        return cls(info, path=path, device=device, keep_loaded_min=keep_loaded_min)

    # ---- lifetime ------------------------------------------------------------
    def _ensure(self):
        with self._lock:
            if self._model is None:
                from transformers import AutoModelForCausalLM, AutoTokenizer
                self._tok = AutoTokenizer.from_pretrained(self._path, local_files_only=True)
                self._model = AutoModelForCausalLM.from_pretrained(self._path, local_files_only=True,
                                                                   torch_dtype="auto").to(self._device).eval()
                self.loads += 1
                log.info("c2c_ai.hf_local: loaded %s on %s", self._path, self._device)
            self._arm()
            return self._model, self._tok

    def _arm(self):
        if self._timer is not None:
            self._timer.cancel()
        if self._keep_s > 0:
            self._timer = threading.Timer(self._keep_s, self.unload)
            self._timer.daemon = True
            self._timer.start()

    def unload(self) -> None:
        with self._lock:
            self._model = self._tok = None
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None
        try:
            import comfy.model_management as mm
            mm.soft_empty_cache()
        except Exception:  # noqa: BLE001
            pass

    @property
    def loaded(self) -> bool:
        return self._model is not None

    # ---- API -----------------------------------------------------------------
    def ask(self, req: AskRequest) -> AskResponse:
        import torch
        from .core_text_encoder import split_thinking
        check_cancel(req)
        refuse_images(req, "The local model folder")
        model, tok = self._ensure()
        msgs = [{"role": m.role, "content": m.content} for m in req.messages]
        t0 = now_ms()
        if getattr(tok, "chat_template", None):
            ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
        else:                                           # a base model without a template: plain text turns
            text = "\n\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in msgs) + "\n\nAssistant:"
            ids = tok(text, return_tensors="pt").input_ids
        if not isinstance(ids, torch.Tensor):          # newer transformers may return a BatchEncoding
            ids = ids["input_ids"]
        ids = ids.to(self._device)
        check_cancel(req)                               # the last point before the uninterruptible generate
        temp = float(req.temperature or 0.0)
        kw = dict(max_new_tokens=int(req.max_tokens), do_sample=temp > 0,
                  pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
        if temp > 0:
            kw.update(temperature=temp, top_p=0.95)
        with torch.no_grad():
            out = model.generate(ids, attention_mask=torch.ones_like(ids), **kw)
        new = out[0, ids.shape[-1]:]
        text = split_thinking(tok.decode(new, skip_special_tokens=True))
        self._arm()
        return AskResponse(text=text, backend_id=self.info.id, model=self.info.model,
                           input_tokens=int(ids.shape[-1]), output_tokens=int(new.shape[-1]), cost_usd=0.0,
                           latency_ms=now_ms() - t0)

    def probe(self, timeout: float = 5.0) -> HealthState:
        """Cheap: the folder is still there and still a chat model (never loads the weights)."""
        from .. import hf_scan
        try:
            c = hf_scan.classify_cached(self._path)
            self.health.ok, self.health.last_error = bool(c["can_chat"]), None if c["can_chat"] else c["why"]
        except Exception as exc:  # noqa: BLE001
            self.health.ok, self.health.last_error = False, str(exc)[:200]
        return self.health
