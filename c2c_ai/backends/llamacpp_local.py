"""Local llama.cpp backend (B1 — text_encoder_local).

Wraps ``llama_cpp.Llama`` and loads a GGUF model from ComfyUI's
``models/text_encoders/`` folder via ``folder_paths.get_full_path``. This
satisfies the user-mandated rule "all model paths must go through
folder_paths" (see user memory `comfyui_model_paths.md`).

Why text_encoders/?
    ComfyUI ships with a registered ``text_encoders`` model folder (the
    same one used for T5/CLIP single-file checkpoints). Re-using it lets
    users drop a GGUF (e.g. ``qwen2.5-3b-instruct-q4_k_m.gguf``) into the
    same tree they already manage with ComfyUI Manager / extra_model_paths.yaml,
    and the auto-discovery picks it up. No new folder key needed.

Config entry::

    {
      "kind": "text_encoder_local",
      "id": "local.llamacpp",
      "display_name": "Local GGUF (llama.cpp)",
      "model_file": "qwen2.5-3b-instruct-q4_k_m.gguf",
      "n_ctx": 8192,
      "n_gpu_layers": -1,     // -1 = offload all to GPU if CUDA build available
      "chat_format": "chatml", // optional override
      "enabled": true
    }
"""

from __future__ import annotations

import logging
import time
from typing import Generator

from .base import Backend, now_ms
from ..types import (
    AskRequest,
    AskResponse,
    BackendInfo,
    Capability,
    HealthState,
    Tier,
)

log = logging.getLogger("c2c_ai.llamacpp_local")


def _resolve_model_path(model_file: str) -> str:
    """Resolve ``model_file`` against ComfyUI's ``text_encoders`` folder.

    Accepts either a bare filename (looked up via folder_paths) or an
    absolute path (used as-is). Raises RuntimeError if neither resolves.
    """
    import os
    if os.path.isabs(model_file) and os.path.isfile(model_file):
        return model_file
    try:
        import folder_paths  # type: ignore
    except Exception as exc:  # pragma: no cover - ComfyUI always provides this
        raise RuntimeError(
            f"folder_paths import failed (are we running outside ComfyUI?): {exc}"
        )
    p = folder_paths.get_full_path("text_encoders", model_file)
    if not p or not os.path.isfile(p):
        raise RuntimeError(
            f"GGUF model {model_file!r} not found in any registered "
            f"text_encoders folder. Drop the file into "
            f"ComfyUI/models/text_encoders/ or add the folder to "
            f"extra_model_paths.yaml."
        )
    return p


class LlamaCppBackend(Backend):
    """llama-cpp-python wrapper. Loads the GGUF lazily on the first probe()
    or ask() so importing the module is cheap even if llama_cpp isn't
    installed yet."""

    def __init__(self, info: BackendInfo, *,
                 model_file: str,
                 n_ctx: int = 8192,
                 n_gpu_layers: int = -1,
                 chat_format: str | None = None,
                 verbose: bool = False) -> None:
        super().__init__(info)
        self._model_file = model_file
        self._n_ctx = int(n_ctx)
        self._n_gpu_layers = int(n_gpu_layers)
        self._chat_format = chat_format
        self._verbose = bool(verbose)
        self._llm = None
        self._resolved_path: str | None = None

    @classmethod
    def build(cls,
              backend_id: str = "local.llamacpp",
              model_file: str = "",
              display_name: str = "Local GGUF (llama.cpp)",
              n_ctx: int = 8192,
              n_gpu_layers: int = -1,
              chat_format: str | None = None) -> "LlamaCppBackend":
        if not model_file:
            raise ValueError("LlamaCppBackend requires model_file (a GGUF filename "
                             "under models/text_encoders/ or an absolute path)")
        # Display the resolved short name for the UI.
        import os
        short = os.path.basename(model_file)
        info = BackendInfo(
            id=backend_id,
            tier=Tier.LOCAL,
            display_name=f"{display_name} — {short}",
            model=short,
            capabilities={Capability.CHAT, Capability.STREAMING},
            max_context=int(n_ctx),
            cost_per_1k_input=0.0,
            cost_per_1k_output=0.0,
        )
        return cls(info,
                   model_file=model_file,
                   n_ctx=n_ctx,
                   n_gpu_layers=n_gpu_layers,
                   chat_format=chat_format)

    # ---- internal --------------------------------------------------------

    def _ensure_loaded(self):
        if self._llm is not None:
            return self._llm
        try:
            from llama_cpp import Llama  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "llama-cpp-python is not installed. Install with "
                "`pip install llama-cpp-python` (CPU) or follow the CUDA "
                f"build instructions for your GPU. Original error: {exc}"
            )
        self._resolved_path = _resolve_model_path(self._model_file)
        kwargs = dict(
            model_path=self._resolved_path,
            n_ctx=self._n_ctx,
            n_gpu_layers=self._n_gpu_layers,
            verbose=self._verbose,
        )
        if self._chat_format:
            kwargs["chat_format"] = self._chat_format
        log.info("c2c_ai.llamacpp: loading %s (n_ctx=%d, n_gpu_layers=%d)",
                 self._resolved_path, self._n_ctx, self._n_gpu_layers)
        self._llm = Llama(**kwargs)
        return self._llm

    @staticmethod
    def _to_messages(req: AskRequest) -> list[dict]:
        return [{"role": m.role, "content": m.content} for m in req.messages]

    # ---- API -------------------------------------------------------------

    def ask(self, req: AskRequest) -> AskResponse:
        llm = self._ensure_loaded()
        t0 = now_ms()
        result = llm.create_chat_completion(
            messages=self._to_messages(req),
            temperature=req.temperature,
            max_tokens=req.max_tokens,
            stream=False,
        )
        latency = now_ms() - t0
        try:
            text = result["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError):
            text = ""
        usage = result.get("usage") or {}
        in_tok = int(usage.get("prompt_tokens", 0))
        out_tok = int(usage.get("completion_tokens", 0))
        return AskResponse(
            text=text,
            backend_id=self.info.id,
            model=self.info.model,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cost_usd=0.0,
            latency_ms=latency,
        )

    def stream(self, req: AskRequest) -> Generator[str, None, AskResponse]:
        llm = self._ensure_loaded()
        t0 = now_ms()
        text_buf: list[str] = []
        in_tok = 0
        out_tok = 0
        it = llm.create_chat_completion(
            messages=self._to_messages(req),
            temperature=req.temperature,
            max_tokens=req.max_tokens,
            stream=True,
        )
        for chunk in it:
            try:
                delta = chunk["choices"][0].get("delta") or {}
                piece = delta.get("content") or ""
            except (KeyError, IndexError, TypeError):
                piece = ""
            if piece:
                text_buf.append(piece)
                yield piece
            usage = chunk.get("usage")
            if usage:
                in_tok = int(usage.get("prompt_tokens", in_tok))
                out_tok = int(usage.get("completion_tokens", out_tok))
        latency = now_ms() - t0
        return AskResponse(
            text="".join(text_buf),
            backend_id=self.info.id,
            model=self.info.model,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cost_usd=0.0,
            latency_ms=latency,
        )

    def probe(self, timeout: float = 5.0) -> HealthState:
        t0 = now_ms()
        try:
            self._ensure_loaded()
            # Cheap sanity round-trip: tokenize a tiny string.
            self._llm.tokenize(b"hi")
            self.health.ok = True
            self.health.last_error = None
        except Exception as exc:
            self.health.ok = False
            self.health.last_error = str(exc)[:200]
        self.health.last_rtt_ms = now_ms() - t0
        self.health.last_probe_at = time.time()
        return self.health

    # ---------------------------------------------------------- list_models
    def list_models(self, timeout: float = 5.0) -> list[str]:
        """Enumerate ``*.gguf`` filenames in every registered
        ``text_encoders`` folder (``folder_paths.get_filename_list``).

        Returns filenames, not absolute paths — the picker writes the chosen
        name back to ``backend.model_file`` which is later resolved with
        ``folder_paths.get_full_path`` at load time.

        Always prepends the currently configured ``model_file`` so the
        active selection is never dropped, even on scan failure.
        """
        out: list[str] = []
        if self._model_file:
            out.append(self._model_file)
        try:
            import folder_paths  # type: ignore
            names = folder_paths.get_filename_list("text_encoders") or []
        except Exception:
            return out
        for n in names:
            if isinstance(n, str) and n.lower().endswith(".gguf") and n not in out:
                out.append(n)
        return out
