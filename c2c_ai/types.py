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


@dataclass(frozen=True)
class ImagePart:
    """One image sent with a message (L6.01 S6). Bytes never appear in logs or previews."""
    mime: str        # image/png | image/jpeg | image/webp | image/gif
    data: bytes

    def b64(self) -> str:
        import base64
        return base64.b64encode(self.data).decode("ascii")

    def data_url(self) -> str:
        return f"data:{self.mime};base64,{self.b64()}"


@dataclass
class Message:
    role: str  # "system" | "user" | "assistant"
    content: str
    images: tuple = ()   # ImagePart, ...; only VISION backends send them, every other backend refuses in words


class TextOnlyError(ValueError):
    """A request asked the spine for something it cannot carry (images, tool calls); the message says what."""

    code = "TEXT_ONLY"


_ROLES = ("system", "user", "assistant")


IMAGE_MIMES = ("image/png", "image/jpeg", "image/webp", "image/gif")
MAX_IMAGES, MAX_IMAGE_BYTES = 4, 8 << 20


def _image_from_part(part, where: str) -> ImagePart:
    import base64
    import binascii
    url = ((part.get("image_url") or {}).get("url") if isinstance(part.get("image_url"), dict)
           else part.get("image_url")) or ""
    if not isinstance(url, str) or not url.startswith("data:"):
        raise TextOnlyError(f"{where}: images must be inline data: URLs (base64); the server does not fetch links")
    head, _, b64 = url[5:].partition(",")
    mime = head.split(";")[0].strip().lower()
    if mime not in IMAGE_MIMES or ";base64" not in head:
        raise TextOnlyError(f"{where}: image type must be one of {', '.join(IMAGE_MIMES)} as base64")
    try:
        data = base64.b64decode(b64, validate=True)
    except (binascii.Error, ValueError):
        raise TextOnlyError(f"{where}: the image is not valid base64") from None
    if not data or len(data) > MAX_IMAGE_BYTES:
        raise TextOnlyError(f"{where}: an image must be 1 byte to {MAX_IMAGE_BYTES >> 20} MB")
    return ImagePart(mime=mime, data=data)


def messages_from_json(raw, *, require_vision: bool = False) -> list:
    """Messages from a route body. Text, or OpenAI-style content parts: {"type":"text"} and {"type":"image_url"}
    with an inline data: URL (L6.01 S6). Anything else is refused LOUDLY with the reason - never dropped (L9.39: an
    image part used to be passed on and lost while the caller believed the model saw it). `require_vision` is kept
    for callers; images make the router pick a VISION backend either way."""
    out = []
    n_images = 0
    for i, m in enumerate(raw or []):
        if not isinstance(m, dict) or m.get("role") not in _ROLES:
            raise TextOnlyError(f"message {i}: needs a role of {', '.join(_ROLES)}")
        content = m.get("content")
        if isinstance(content, str):
            out.append(Message(role=m["role"], content=content))
            continue
        if not isinstance(content, list):
            raise TextOnlyError(f"message {i}: content must be text or a list of text / image parts "
                                f"(got {type(content).__name__})")
        texts, images = [], []
        for j, part in enumerate(content):
            kind = part.get("type") if isinstance(part, dict) else None
            if kind == "text" and isinstance(part.get("text"), str):
                texts.append(part["text"])
            elif kind == "image_url":
                images.append(_image_from_part(part, f"message {i} part {j}"))
            else:
                raise TextOnlyError(f"message {i} part {j}: only text and image_url parts are sent by the C2C AI "
                                    f"spine (got {kind or type(part).__name__}); tool calls are not supported")
        if images and m["role"] != "user":
            raise TextOnlyError(f"message {i}: images can only be sent in user messages")
        n_images += len(images)
        if n_images > MAX_IMAGES:
            raise TextOnlyError(f"at most {MAX_IMAGES} images per request")
        out.append(Message(role=m["role"], content="\n".join(texts), images=tuple(images)))
    return out


def has_images(messages) -> bool:
    return any(getattr(m, "images", ()) for m in messages)


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
