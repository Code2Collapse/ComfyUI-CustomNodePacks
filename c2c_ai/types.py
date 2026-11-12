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
    TOOL_USE = "tool_use"    # what the MODEL can do; the C2C spine itself sends text only (see messages_from_json)
    VISION = "vision"        # idem: no image ever travels through Message (A22 audit CNP row 16, L9.39)


class Sensitivity(str, Enum):
    """How aggressively the redactor scrubs payloads before cloud calls."""

    PUBLIC = "public"         # no scrubbing (e.g. asking about a public model name)
    NORMAL = "normal"         # strip user home paths, drive letters, emails
    SENSITIVE = "sensitive"   # strip everything + REFUSE cloud unless user-confirmed


@dataclass
class Message:
    role: str  # "system" | "user" | "assistant"
    content: str


class TextOnlyError(ValueError):
    """A request asked the spine for something it cannot carry (images, tool calls); the message says what."""

    code = "TEXT_ONLY"


_ROLES = ("system", "user", "assistant")


def messages_from_json(raw, *, require_vision: bool = False) -> list:
    """Messages from a route body, refused LOUDLY when they need more than text.

    The backends advertise the model's vision / tool-use capabilities, but Message carries text only: an image part
    used to be passed on (and dropped or str()-ed by the backend) while the caller believed the model saw it.
    """
    if require_vision:
        raise TextOnlyError("require_vision: the C2C AI spine sends text only - images are not sent to any backend. "
                            "Describe the image in text, or use a vision node that calls the model directly.")
    out = []
    for i, m in enumerate(raw or []):
        if not isinstance(m, dict) or m.get("role") not in _ROLES:
            raise TextOnlyError(f"message {i}: needs a role of {', '.join(_ROLES)}")
        content = m.get("content")
        if not isinstance(content, str):
            kind = type(content).__name__
            raise TextOnlyError(f"message {i}: content must be text (got {kind}); image / tool parts are not sent "
                                "by the C2C AI spine")
        out.append(Message(role=m["role"], content=content))
    return out


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
    # L6.01 S5: None = serves every feature; a set = only these features (e.g. the Diagnostics-sidebar backends)
    features: frozenset | None = None

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
    # L6.01 S1: one deadline per request (None = the backend's own default) and a cancel flag. A set() event stops a
    # stream at the next chunk and is checked before a request is sent; the caller then gets `Cancelled`.
    timeout_s: float | None = None
    cancel: Any = None                                 # threading.Event (or anything with is_set())


class Cancelled(RuntimeError):
    """The caller cancelled the request (AskRequest.cancel was set). The router never falls back after this."""


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
