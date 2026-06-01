"""Backend adapters — one per provider.

Each adapter exposes:
    info       -> BackendInfo (static)
    health     -> HealthState (cached, refreshed by router)
    ask()      -> AskResponse (non-streaming)
    stream()   -> generator yielding text chunks (optional)
    probe()    -> refresh HealthState (router calls this periodically)

If a backend doesn't support streaming it should leave Capability.STREAMING
out of its capability set; the router degrades gracefully.
"""

from __future__ import annotations

from .base import Backend
from .anthropic import AnthropicBackend
from .openai_compat import OpenAICompatBackend
from .gemini import GeminiBackend
from .cohere import CohereBackend
from .azure_openai import AzureOpenAIBackend
from .llamacpp_local import LlamaCppBackend
from .rulepack import RulePackBackend
from .borrowed import BorrowedEncoderBackend

__all__ = [
    "Backend",
    "AnthropicBackend",
    "OpenAICompatBackend",
    "GeminiBackend",
    "CohereBackend",
    "AzureOpenAIBackend",
    "LlamaCppBackend",
    "RulePackBackend",
    "BorrowedEncoderBackend",
]
