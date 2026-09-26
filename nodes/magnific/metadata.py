"""Generation metadata for the `metadata` socket every creation node exposes.

ComfyUI re-encodes what a node returns (IMAGE tensors, its own VIDEO/AUDIO
containers), so nothing Magnific stamps on the delivered file survives into the
graph. The facts travel as a value instead: one JSON string per node, built from
`creations_get`, which the Magnific Metadata node unpacks into plain sockets.

The fields are the `creations_get` allow-list (McpCreationResource) flattened
to snake_case — never the raw creation metadata, which carries internal URLs
and other users' asset references. Fetching is best effort: the generation is
already paid for by the time this runs, so a failed lookup degrades to
`{creation_identifier}` rather than failing the node.

Stdlib-only, and free of torch, so it is unit-testable without ComfyUI.
"""

import json
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List

from . import api

# (output key, creations_get key, cast). Order is the order the JSON is written
# in, so the model sits right after the identity — it is what users open the
# JSON for.
FIELDS = (
    ("creation_identifier", "identifier", str),
    ("tool", "tool", str),
    ("model", "modelName", str),
    ("mode", "mode", str),
    ("api", "api", str),
    ("prompt", "prompt", str),
    ("negative_prompt", "negativePrompt", str),
    ("seed", "seed", int),
    ("width", "width", int),
    ("height", "height", int),
    ("aspect_ratio", "aspectRatio", str),
    ("resolution", "resolution", str),
    ("duration", "duration", float),
    ("fps", "fps", float),
    ("created_at", "createdAt", str),
    ("url", "webUrl", str),
)

_UNESCAPES = {'\\"': '"', "\\\\": "\\", "\\n": "\n", "\\r": "\r", "\\t": "\t"}


def _unquote(raw: str) -> Any:
    raw = raw.strip()
    if len(raw) >= 2 and raw[0] == '"' and raw[-1] == '"':
        return re.sub(r'\\["\\nrt]', lambda m: _UNESCAPES[m.group(0)], raw[1:-1])
    if raw == "null":
        return None
    if raw in ("true", "false"):
        return raw == "true"
    return raw


def parse_toon_scalars(text: str) -> Dict[str, Any]:
    """Scalar fields of a creations_get TOON document: the top-level ones plus
    the ones one level down under `metadata:`, flattened onto the same dict
    (the top level wins on a clash). Anything nested deeper — mediaCollection,
    tag tables — is skipped; a scalar is only read at the depth it belongs to,
    so a `prompt:` row inside a nested list never leaks in."""
    fields: Dict[str, Any] = {}
    nested: Dict[str, Any] = {}
    in_metadata = False
    for line in text.split("\n"):
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        if indent == 0:
            in_metadata = False
            match = re.match(r"^([A-Za-z_][\w.]*):(?:\s(.*))?$", line)
            if not match:
                continue
            key, value = match.group(1), match.group(2)
            if key == "metadata" and value is None:
                in_metadata = True
            elif value is not None:
                fields[key] = _unquote(value)
        elif in_metadata and indent == 2:
            match = re.match(r"^\s{2}([A-Za-z_][\w.]*):\s(.+)$", line)
            if match:
                nested[match.group(1)] = _unquote(match.group(2))
    return {**nested, **fields}


def _cast(value: Any, cast) -> Any:
    if value is None or value == "":
        return None
    try:
        if cast is int:
            return int(float(value))
        if cast is float:
            number = float(value)
            return int(number) if number.is_integer() else number
        return str(value)
    except (TypeError, ValueError):
        return None


def normalize(identifier: str, fields: Dict[str, Any]) -> Dict[str, Any]:
    """The JSON entry for one creation. `model` falls back to `mode`, then
    `api`: video and legacy image tools store the model under those keys and
    the whole point of the socket is that `model` is never empty when the
    server knows it."""
    entry: Dict[str, Any] = {}
    for out_key, source_key, cast in FIELDS:
        value = _cast(fields.get(source_key), cast)
        if value is not None:
            entry[out_key] = value
    entry["creation_identifier"] = identifier
    if "model" not in entry:
        fallback = entry.get("mode") or entry.get("api")
        if fallback:
            entry["model"] = fallback
    return entry


def fetch_one(identifier: str) -> Dict[str, Any]:
    try:
        text = api.client().text("creations_get", {"creationIdentifier": identifier})
        return normalize(identifier, parse_toon_scalars(text))
    except Exception as error:  # the media is already in the graph; never fail the node here
        print(f"[Magnific] Could not read metadata for {identifier}: {error}")  # console-ok
        return {"creation_identifier": identifier}


def fetch(identifiers: List[str]) -> List[Dict[str, Any]]:
    """One entry per identifier, in the request order."""
    if not identifiers:
        return []
    if len(identifiers) == 1:
        return [fetch_one(identifiers[0])]
    with ThreadPoolExecutor(max_workers=min(6, len(identifiers))) as pool:
        return list(pool.map(fetch_one, identifiers))


def to_json(entries: List[Dict[str, Any]]) -> str:
    """A single creation is an object; a batch is a list in batch order, so the
    entry at index N describes the IMAGE at batch index N."""
    payload: Any = entries[0] if len(entries) == 1 else entries
    return json.dumps(payload, ensure_ascii=False, indent=2)


def for_identifiers(identifiers: List[str]) -> str:
    return to_json(fetch(identifiers))


def unpack(text: str, index: int = 0) -> Dict[str, Any]:
    """The entry at `index` of a `metadata` socket value. An object counts as a
    one-entry batch; an unparseable or empty value yields {} so every unpacked
    socket reads as its empty default instead of raising mid-graph."""
    try:
        parsed = json.loads(text) if text and text.strip() else None
    except ValueError:
        return {}
    entries = parsed if isinstance(parsed, list) else [parsed] if isinstance(parsed, dict) else []
    if not entries:
        return {}
    entry = entries[max(0, min(index, len(entries) - 1))]
    return entry if isinstance(entry, dict) else {}


def unpack_text(entry: Dict[str, Any], key: str) -> str:
    value = entry.get(key)
    return "" if value is None else str(value)


def unpack_int(entry: Dict[str, Any], key: str, default: int = 0) -> int:
    value = _cast(entry.get(key), int)
    return default if value is None else value


def unpack_json(entry: Dict[str, Any]) -> str:
    return json.dumps(entry, ensure_ascii=False, indent=2) if entry else ""
