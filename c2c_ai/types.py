"""Shared types for the C2C AI spine.

Kept in one tiny module so backends/router/policy/cost_meter all import the
same Pydantic-free dataclasses without circular imports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Iterable


class Tier(str, Enum):
    CLOUD = "cloud"
    LOCAL = "local"
    DETERMINISTIC = "deterministic"  # Track D.1: rule-pack / introspector / non-LLM


class Policy(str, Enum):
    AUTO = "auto"                 # cheapest healthy backend that fits the capability
    PREFER_LOCAL = "prefer_local"
    PREFER_CLOUD = "prefer_cloud"
    CLOUD_ONLY = "cloud_only"
    LOCAL_ONLY = "local_only"
    DETERMINISTIC_ONLY = "deterministic_only"  # Track D.1: rule-pack only; never LLM


class Capability(str, Enum):
    CHAT = "chat"
    STREAMING = "streaming"
    JSON_MODE = "json_mode"
    TOOL_USE = "tool_use"
    VISION = "vision"


class Sensitivity(str, Enum):
    """How aggressively the redactor scrubs payloads before cloud calls."""

    PUBLIC = "public"         # no scrubbing (e.g. asking about a public model name)
    NORMAL = "normal"         # strip user home paths, drive letters, emails
    SENSITIVE = "sensitive"   # strip everything + REFUSE cloud unless user-confirmed


@dataclass
class Message:
    role: str  # "system" | "user" | "assistant"
    content: str


@dataclass
class BackendInfo:
    """Static identity of a backend (registered at startup)."""

    id: str                                # e.g. "cloud.anthropic", "local.ollama"
    tier: Tier
    display_name: str
    model: str
    capabilities: set[Capability] = field(default_factory=set)
    max_context: int = 8_192
    cost_per_1k_input: float = 0.0         # USD; 0 for local
    cost_per_1k_output: float = 0.0
    enabled: bool = True

    def supports(self, cap: Capability) -> bool:
        return cap in self.capabilities


@dataclass
class HealthState:
    """Live health snapshot — refreshed by the router every 60 s."""

    ok: bool = True
    last_rtt_ms: float = 0.0
    last_error: str | None = None
    last_probe_at: float = 0.0


@dataclass
class AskRequest:
    """Single non-streaming request through the spine."""

    feature: str                                       # "node_explainer" etc.
    messages: list[Message]
    sensitivity: Sensitivity = Sensitivity.NORMAL
    required: set[Capability] = field(default_factory=lambda: {Capability.CHAT})
    max_tokens: int = 1024
    temperature: float = 0.4
    policy_override: Policy | None = None
    json_schema: dict[str, Any] | None = None          # backends that support JSON mode honor this


@dataclass
class AskResponse:
    text: str
    backend_id: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: float
    redacted: bool = False


# ---------- streaming ----------------------------------------------------------

StreamChunk = str
StreamCallback = Callable[[StreamChunk], None]
