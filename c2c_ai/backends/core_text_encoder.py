"""Tier LOCAL: chat with a text encoder already on disk, through ComfyUI's OWN text generation (ledger L6.02 S2).

Design: docs/research/L6.02_tier_local_design.md (workspace). The same calls core's TextGenerate node makes
(comfy_extras/nodes_textgen.py, core 0.39): ``clip.tokenize(prompt, image=, system_prompt=, min_length=1)`` ->
``clip.generate(tokens, ...)`` -> ``clip.decode(ids)``; reasoning in ``<think>`` is split off the same way.

Only files the header scan (c2c_ai/te_scan.py) marks ``can_chat`` are accepted: families whose output layer core
handles correctly (e.g. the Qwen2.5-VL-7B "Qwen-Image" encoder is refused: core generates through the wrong layer).

Config entry::

    {"kind": "core_text_encoder", "id": "local.core_te", "model_file": "qwen_3_4b.safetensors",
     "keep_loaded_min": 5, "enabled": true}

- The model is loaded on first use with ``comfy.sd.load_clip`` (ComfyUI's model management offloads it when an image
  job needs the memory) and dropped after ``keep_loaded_min`` minutes without a request.
- One conversation becomes ONE prompt for the model's chat template (the templates take a single user turn); system
  messages become the template's system prompt.
- ``timeout_s`` / ``cancel`` are checked before generating; core's generate itself cannot be interrupted.
- Images (data: URLs in Message.images) are passed only when the file has vision weights.
"""
from __future__ import annotations

import base64
import io
import logging
import os
import threading

from .base import Backend, check_cancel, now_ms
from ..types import AskRequest, AskResponse, BackendInfo, Capability, HealthState, Tier

log = logging.getLogger("c2c_ai.core_text_encoder")


def resolve(model_file: str) -> str:
    if os.path.isabs(model_file) and os.path.isfile(model_file):
        return model_file
    import folder_paths  # type: ignore
    for key in ("text_encoders", "clip"):
        try:
            p = folder_paths.get_full_path(key, model_file)
        except Exception:  # noqa: BLE001 - folder key not registered
            p = None
        if p and os.path.isfile(p):
            return p
    raise RuntimeError(f"text encoder {model_file!r} is not in ComfyUI's text_encoders / clip folders")


def _image_tensor(data_url: str):
    """data:image/...;base64,... -> IMAGE tensor [1, H, W, 3] float32 0..1 (ComfyUI's layout)."""
    import numpy as np
    import torch
    from PIL import Image
    head, _, b64 = data_url.partition(",")
    if not head.startswith("data:image/") or ";base64" not in head:
        raise ValueError("images must be data: URLs (base64)")
    img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    return torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0)[None]


def build_prompt(req: AskRequest) -> tuple[str, str]:
    """(system prompt, single user prompt) from the conversation."""
    system = "\n\n".join(m.content for m in req.messages if m.role == "system" and m.content)
    turns = [m for m in req.messages if m.role != "system"]
    if len(turns) == 1:
        return system, turns[0].content
    lines = []
    for m in turns:
        lines.append(f"{'User' if m.role == 'user' else 'Assistant'}: {m.content}")
    return system, "\n\n".join(lines) + "\n\nAnswer the last user message."


def _count_tokens(tokens) -> int:
    """Rough prompt length: core's tokenize returns {name: [[(token, weight), ...], ...]}; layouts differ per model."""
    total = 0
    try:
        for batches in (tokens.values() if isinstance(tokens, dict) else ()):
            for seq in (batches if isinstance(batches, list) else [batches]):
                total += len(seq)
    except Exception:  # noqa: BLE001 - an unexpected layout only loses the count
        return 0
    return total


def split_thinking(text: str) -> str:
    reasoning, sep, answer = text.partition("</think>")
    if sep and reasoning.lstrip().startswith("<think>"):
        return answer.strip()
    return text.strip()


def chat_with_clip(clip, req: AskRequest, backend_id: str, model: str, vision: bool = False) -> AskResponse:
    """One answer through a ComfyUI CLIP object's own chat surface: tokenize -> generate -> decode (core's
    TextGenerate path). Shared by CoreTextEncoderBackend (its own copy) and the borrowed backend (a loaded one)."""
    check_cancel(req)
    images = [im for m in req.messages for im in (getattr(m, "images", None) or [])]
    if images and not vision:
        raise ValueError(f"{model} has no vision weights: send text only, or choose a vision encoder")
    system, prompt = build_prompt(req)
    image = _image_tensor(images[0]) if images else None
    t0 = now_ms()
    kw = {"min_length": 1, "system_prompt": system}
    if image is not None:
        kw["image"] = image
    tokens = clip.tokenize(prompt, **kw)
    check_cancel(req)                                       # the last point before the uninterruptible generate
    temp = float(req.temperature or 0.0)
    ids = clip.generate(tokens, do_sample=temp > 0, max_length=int(req.max_tokens), temperature=max(temp, 1e-5),
                        top_k=50, top_p=0.95, min_p=0.0, repetition_penalty=1.0, seed=None)
    text = split_thinking(clip.decode(ids))
    try:
        n_out = len(ids)
    except Exception:  # noqa: BLE001
        n_out = 0
    return AskResponse(text=text, backend_id=backend_id, model=model, input_tokens=int(_count_tokens(tokens)),
                       output_tokens=int(n_out), cost_usd=0.0, latency_ms=now_ms() - t0)


class CoreTextEncoderBackend(Backend):
    def __init__(self, info: BackendInfo, *, model_file: str, path: str, vision: bool, keep_loaded_min: float = 5.0,
                 loader=None):
        super().__init__(info)
        self._model_file, self._path, self._vision = model_file, path, bool(vision)
        self._keep_s = max(0.0, float(keep_loaded_min)) * 60.0
        self._clip = None
        self._timer: threading.Timer | None = None
        self._lock = threading.Lock()
        self._loader = loader                                   # tests: a stand-in for comfy.sd.load_clip
        self.loads = 0

    @classmethod
    def build(cls, backend_id: str = "local.core_te", model_file: str = "", display_name: str = "ComfyUI text encoder",
              keep_loaded_min: float = 5.0, loader=None) -> "CoreTextEncoderBackend":
        if not model_file:
            raise ValueError("core_text_encoder needs model_file (a .safetensors text encoder)")
        from .. import te_scan
        path = resolve(model_file)
        c = te_scan.classify_cached(path)
        if not c["can_chat"]:
            raise ValueError(f"{os.path.basename(path)}: {c['why']}")
        caps = {Capability.CHAT} | ({Capability.VISION} if c["vision"] else set())
        info = BackendInfo(id=backend_id, tier=Tier.LOCAL, display_name=f"{display_name} - {os.path.basename(path)}",
                           model=os.path.basename(path), capabilities=caps, max_context=8192)
        return cls(info, model_file=model_file, path=path, vision=c["vision"], keep_loaded_min=keep_loaded_min,
                   loader=loader)

    # ---- model lifetime ----------------------------------------------------
    def _ensure(self):
        with self._lock:
            if self._clip is None:
                if self._loader is not None:
                    self._clip = self._loader([self._path])
                else:
                    import comfy.sd
                    import folder_paths  # type: ignore
                    self._clip = comfy.sd.load_clip(ckpt_paths=[self._path],
                                                    embedding_directory=folder_paths.get_folder_paths("embeddings"))
                self.loads += 1
                log.info("c2c_ai.core_te: loaded %s", self._path)
            self._arm_unload()
            return self._clip

    def _arm_unload(self):
        if self._timer is not None:
            self._timer.cancel()
        if self._keep_s > 0:
            self._timer = threading.Timer(self._keep_s, self.unload)
            self._timer.daemon = True
            self._timer.start()

    def unload(self) -> None:
        with self._lock:
            if self._clip is None:
                return
            self._clip = None
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None
        try:
            import comfy.model_management as mm
            mm.soft_empty_cache()
        except Exception:  # noqa: BLE001 - outside ComfyUI (tests)
            pass
        log.info("c2c_ai.core_te: released %s", self._path)

    @property
    def loaded(self) -> bool:
        return self._clip is not None

    # ---- API -----------------------------------------------------------------
    def ask(self, req: AskRequest) -> AskResponse:
        check_cancel(req)
        images = [im for m in req.messages for im in (getattr(m, "images", None) or [])]
        if images and not self._vision:
            raise ValueError(f"{self.info.model} has no vision weights: send text only, or choose a vision encoder")
        clip = self._ensure()
        out = chat_with_clip(clip, req, self.info.id, self.info.model, vision=self._vision)
        self._arm_unload()
        return out

    def probe(self, timeout: float = 5.0) -> HealthState:
        from .. import te_scan
        try:
            ok = os.path.isfile(self._path) and te_scan.classify_cached(self._path)["can_chat"]
            self.health.ok, self.health.last_error = bool(ok), None if ok else "the file is gone or cannot chat"
        except Exception as exc:  # noqa: BLE001
            self.health.ok, self.health.last_error = False, str(exc)
        return self.health
