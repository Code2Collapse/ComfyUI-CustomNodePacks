# -*- coding: utf-8 -*-
"""Tiny shared helper: make ComfyUI Stop button interrupt long loops.

Call ``check()`` at the top of every per-frame / per-batch loop body.
It raises ``InterruptProcessingException`` (caught by ComfyUI's
executor) when the user clicks Stop, freeing the loop immediately.

Falls back to a no-op when imported outside a running ComfyUI server.
"""
try:
    from comfy.model_management import (
        throw_exception_if_processing_interrupted as _throw,
    )

    def check() -> None:
        _throw()
except Exception:  # noqa: BLE001
    def check() -> None:  # type: ignore[no-redef]
        return None
