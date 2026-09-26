"""Magnific MCP tool orchestration: uploads, generation, upscale, stock and the
folder catalog. Mirrors editor-plugins/src/api/{magnific,services/*}.ts so both
integrations speak the exact same tool contracts.
"""

import re
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

from . import config, tls_trust
from .mcp import McpError, client

TERMINAL_STATUSES = {"completed", "succeeded", "failed", "error", "canceled", "cancelled"}

# Store directly in the project, not in one of its folders (mirrored in
# web/magnific.js and the Save To / Folder Picker combos).
PROJECT_ROOT = "(project root)"

# What stock_download can rasterise to a bitmap the graph can consume.
STOCK_IMAGE_TYPES = {"photo", "vector", "illustration", "psd", "design"}


def _is_failed(status: str) -> bool:
    return status in ("failed", "error") or status.startswith("cancel")


# ---------------------------------------------------------------------------
# TOON parsers — folders_list / images_models_list emit lean text blocks
# (`- reference:` + indented fields), not structuredContent. Ports of
# parseFolders (services/folders.ts) and parseImageModels (services/generation.ts).


def parse_folders(text: str) -> List[dict]:
    items: List[dict] = []
    current: Optional[dict] = None
    for line in text.split("\n"):
        ref = re.match(r"^\s*-\s*reference:\s*(\S+)", line)
        if ref:
            current = {"reference": ref.group(1), "name": ""}
            items.append(current)
            continue
        if current is None:
            continue
        name = re.match(r"^\s+name:\s*(.+?)\s*$", line)
        # The item's own name is the first after its `- reference:` line; the
        # nested `parent` block's name comes later and is ignored.
        if name and not current["name"]:
            current["name"] = re.sub(r'^"(.*)"$', r"\1", name.group(1))
    return [{**item, "name": item["name"] or item["reference"]} for item in items]


def parse_image_models(text: str) -> List[dict]:
    """images_models_list TOON: the per-model `resolutions[N]:` CSV line drives
    the image node's dependent resolution dropdown (models that take no
    resolution simply have no such line); `referenceTypes[N]:` gates which
    Library asset types the model takes (a model without the line is text-only)."""
    models: List[dict] = []
    current: Optional[dict] = None
    for line in text.split("\n"):
        slug = re.match(r"^\s*-\s*slug:\s*(\S+)", line)
        if slug:
            current = {"slug": slug.group(1), "name": "", "resolutions": [], "referenceTypes": []}
            models.append(current)
            continue
        if current is None:
            continue
        name = re.match(r"^\s+name:\s*(.+?)\s*$", line)
        if name and not current["name"]:
            current["name"] = re.sub(r'^"(.*)"$', r"\1", name.group(1))
            continue
        for key in ("resolutions", "referenceTypes"):
            match = re.match(rf"^\s+{key}\[\d+\]:\s*(.+?)\s*$", line)
            if match:
                current[key] = [
                    value for value in (raw.strip().strip('"') for raw in match.group(1).split(",")) if value
                ]
                break
    return [{**model, "name": model["name"] or model["slug"]} for model in models]


# ---------------------------------------------------------------------------
# Library catalog (characters, styles, elements, locations) for the Library
# Reference node. library_list emits TOON; the `references` array comes back
# tabular (`references[N]{id,name,…}:` + CSV rows) when every entry has the same
# fields and as a block list (`- id: 12` + indented fields) otherwise — the
# server drops empty fields per entry, so both forms are seen in the wild.

# The generation tools' own vocabulary: `product` is what the web UI calls an
# Element, `locations` is plural. Mirrors LibraryTypeVocabulary::REFERENCE_TYPES.
LIBRARY_REF_TYPES = ("character", "style", "product", "locations")
# video_generate has no `locations` reference.
VIDEO_LIBRARY_TYPES = ("character", "product", "style")
LIBRARY_TYPE_LABELS = {"character": "Character", "style": "Style", "product": "Element", "locations": "Location"}
# The Library node's `type` combo: label → reference type (None = every type).
# Also the group order of the `asset` combo, so a filter and the sort agree.
LIBRARY_TYPE_FILTERS = {
    "All": None,
    "Characters": "character",
    "Styles": "style",
    "Elements": "product",
    "Locations": "locations",
}
# The Library node's `catalog` combo: label → plugin scope key. `mine` is the
# user's own catalog (library_list `scope: all`); `public` is Magnific's
# catalog, which the server never mixes into `all`.
LIBRARY_CATALOGS = {"My Library": "mine", "Magnific": "public"}
LIBRARY_SCOPES = {"mine": "all", "public": "public"}
DEFAULT_LIBRARY_CATALOG = "My Library"
# images_generate takes at most this many references, Library and image alike
# (GenerateImageTool::MAX_REFERENCES).
MAX_GENERATION_REFERENCES = 12
LIBRARY_PAGE_SIZE = 50  # server cap (LibraryCatalogBuilder::MAX_PER_PAGE)
_LIBRARY_MAX_PAGES = 8  # bounds a pathological account instead of paging forever


def library_type_label(ref_type: str) -> str:
    return LIBRARY_TYPE_LABELS.get(ref_type, ref_type)


def library_source_label(source: str, owned: bool) -> str:
    if owned or source == "mine":
        return "Mine"
    return {
        "favorite": "Favorite",
        "shared_project": "Project",
        "shared_team": "Team",
        "shared_group": "Team",
        "shared_individual": "Shared",
        "public": "Magnific",
    }.get(source, "")


def library_scope_for_catalog(catalog: str) -> str:
    return LIBRARY_CATALOGS.get(catalog, "mine")


def library_type_for_filter(type_filter: str) -> Optional[str]:
    return LIBRARY_TYPE_FILTERS.get(type_filter)


def filter_library_choices(choices: List[dict], type_filter: str) -> List[dict]:
    ref_type = library_type_for_filter(type_filter)
    if ref_type is None:
        return list(choices)
    return [choice for choice in choices if choice["entry"]["type"] == ref_type]


def sort_library_choices(choices: List[dict]) -> List[dict]:
    """Grouped by type in the `type` combo's order, then by name, so a long
    catalog reads as blocks instead of one alphabetical soup."""
    order = {ref_type: index for index, ref_type in enumerate(LIBRARY_TYPE_FILTERS.values()) if ref_type}
    return sorted(choices, key=lambda choice: (order.get(choice["entry"]["type"], len(order)), choice["entry"]["name"].casefold(), choice["label"]))


def _toon_unquote(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == '"' and value[-1] == '"':
        return value[1:-1].replace('\\"', '"')
    return value


def _toon_split_csv(line: str) -> List[str]:
    """TOON tabular row: comma-separated, quoted only when needed, \\" escapes."""
    values: List[str] = []
    current: List[str] = []
    in_quotes = False
    index = 0
    while index < len(line):
        char = line[index]
        if in_quotes:
            if char == "\\" and index + 1 < len(line) and line[index + 1] == '"':
                current.append('"')
                index += 1
            elif char == '"':
                in_quotes = False
            else:
                current.append(char)
        elif char == '"' and not current:
            in_quotes = True
        elif char == ",":
            values.append("".join(current))
            current = []
        else:
            current.append(char)
        index += 1
    values.append("".join(current))
    return values


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _toon_block_rows(body: List[str]) -> List[dict]:
    """`- key: value` entries with their fields at exactly one indentation level;
    anything deeper (a nested `projects[N]` table, say) belongs to a sub-object
    and is ignored, so a nested `name:` can't overwrite the entry's own."""
    rows: List[dict] = []
    current: Optional[dict] = None
    field_indent = -1
    for line in body:
        indent = _indent(line)
        dash = re.match(r"^\s*-\s*(.*)$", line)
        if dash and (current is None or indent < field_indent):
            current = {}
            rows.append(current)
            field_indent = indent + 2
            remainder = dash.group(1)
            if not remainder:
                continue
            line, indent = " " * field_indent + remainder, field_indent
        if current is None or indent != field_indent:
            continue
        field = re.match(r"^\s*([A-Za-z_][A-Za-z0-9_]*)(?:\[\d+\])?(?:\{[^}]*\})?:\s*(.*)$", line)
        if field and field.group(2):
            current[field.group(1)] = _toon_unquote(field.group(2))
    return rows


def parse_library(text: str) -> List[dict]:
    """library_list TOON → [{id, name, type, description, owned, source, isLora,
    status}] keeping only the four generation reference types (the same catalog
    also lists color palettes, agents and contexts). `status` is present only
    while an asset is NOT ready — the picker must skip those."""
    lines = text.split("\n")
    header = None
    start = 0
    for index, line in enumerate(lines):
        header = re.match(r"^(\s*)references\[\d+\](\{[^}]*\})?:\s*$", line)
        if header:
            start = index + 1
            break
    if header is None:
        return []
    base = len(header.group(1))
    body: List[str] = []
    for line in lines[start:]:
        if not line.strip():
            continue
        if _indent(line) <= base:
            break
        body.append(line)
    if header.group(2):
        columns = [column.strip() for column in header.group(2)[1:-1].split(",")]
        rows = [
            dict(zip(columns, values))
            for values in (_toon_split_csv(line.strip()) for line in body)
            if len(values) == len(columns)
        ]
    else:
        rows = _toon_block_rows(body)

    entries: List[dict] = []
    for row in rows:
        raw_id = str(row.get("id", "")).strip()
        name = str(row.get("name", "")).strip()
        ref_type = str(row.get("type", "")).strip()
        if not raw_id.isdigit() or not name or ref_type not in LIBRARY_REF_TYPES:
            continue
        entries.append(
            {
                "id": int(raw_id),
                "name": name,
                "type": ref_type,
                "description": str(row.get("description", "")).strip() or None,
                "owned": str(row.get("owned", "")).strip() == "true",
                "source": str(row.get("source", "")).strip() or "mine",
                "isLora": str(row.get("isLora", "")).strip() == "true",
                "status": str(row.get("status", "")).strip() or None,
            }
        )
    return entries


def library_label(entry: dict) -> str:
    """Combo text for one asset: the `@name` token users type in the web app,
    then type and origin so two same-named assets can be told apart."""
    source = library_source_label(entry["source"], entry["owned"])
    details = " · ".join(part for part in (library_type_label(entry["type"]), source) if part)
    return f"@{entry['name']} ({details})"


# Ports the web app's mention grammar (mentionTokenBoundary.ts): `[A-Za-z0-9_-]`
# names, never mid-word (`email@name` is not a mention), an optional `#img-…`
# suffix, and a colon only continues the name when it opens a variant suffix.
_MENTION_RE = re.compile(r"(^|[^A-Za-z0-9_])@([A-Za-z0-9_-]+)(?:#img-[\w,-]+)?(?![A-Za-z0-9_-])(?!\s*:[A-Za-z0-9_-])")


def mention_names(prompt: str) -> List[str]:
    return [match.group(2) for match in _MENTION_RE.finditer(prompt)]


def with_library_mentions(prompt: str, entries: List[dict]) -> str:
    """The server binds a Library reference to the prompt through its `@name`
    token (rewriting it into the asset's description), so a reference the
    prompt never mentions would ride along unused — append the missing tokens."""
    present = set(mention_names(prompt))
    missing: List[str] = []
    for entry in entries:
        if entry["name"] not in present and entry["name"] not in missing:
            missing.append(entry["name"])
    if not missing:
        return prompt
    return prompt.rstrip() + "".join(f" @{name}" for name in missing)


def library_image_references(entries: List[dict]) -> List[dict]:
    """images_generate shape: the Library numeric id, as a string, in `identifier`
    (the server tells it from a creation sqid by shape)."""
    return [{"type": entry["type"], "identifier": str(entry["id"])} for entry in entries]


def library_video_references(entries: List[dict]) -> List[dict]:
    """video_generate clip shape: the id travels in `custom_model_id`;
    `modifier_key`/`name` are the `@name` mention key the server binds."""
    return [
        {
            "type": entry["type"],
            "custom_model_id": str(entry["id"]),
            "modifier_key": entry["name"],
            "name": entry["name"],
        }
        for entry in entries
    ]


# ---------------------------------------------------------------------------
# Upload pipeline: local bytes → creation identifier.


def upload_media(data: bytes, mime_type: str = "image/png") -> str:
    # Checked here rather than only at creations_finalize_upload: the server
    # rejects the oversized image after the presign and the whole PUT has gone
    # over the wire, which on a 40MB frame is a minute of upload wasted.
    if mime_type.startswith("image/") and len(data) > config.MAX_IMAGE_UPLOAD_BYTES:
        raise McpError(
            f"Image is {len(data) / (1024 * 1024):.1f}MB — Magnific accepts at most "
            f"{config.MAX_IMAGE_UPLOAD_BYTES // (1024 * 1024)}MB per upload"
        )
    mcp = client()
    request = mcp.structured("creations_request_upload", {"mimeType": mime_type})
    uploads = request.get("uploads")
    target = uploads[0] if isinstance(uploads, list) and uploads else request
    put_url = target.get("proxyUploadUrl") or target.get("directUploadUrl") or target.get("url")
    path = target.get("path")
    if not put_url or not path:
        raise McpError("Upload request returned no presigned URL")

    last_failure = "no upload target reachable"
    for attempt in range(3):
        if attempt:
            time.sleep(0.4 * attempt)
        try:
            put = urllib.request.Request(
                put_url, data=data, headers={"Content-Type": mime_type}, method="PUT"
            )
            with urllib.request.urlopen(put, timeout=config.HTTP_TIMEOUT_SECONDS * 4, context=tls_trust.get_ssl_context()):
                break
        except urllib.error.HTTPError as error:
            last_failure = f"HTTP {error.code}"
            if error.code < 500:
                raise McpError(f"Upload failed ({last_failure})") from error
        except OSError as error:
            last_failure = f"network error ({error})"
    else:
        raise McpError(f"Upload failed ({last_failure})")

    finalize = client().structured("creations_finalize_upload", {"path": path})
    identifier = finalize.get("identifier")
    if not identifier:
        raise McpError("Upload finalize returned no creation identifier")
    return identifier


# ---------------------------------------------------------------------------
# creations_wait long-poll until every identifier is terminal.


def wait_for_creations(identifiers: List[str]) -> List[dict]:
    """Returns wait entries ({identifier, status, results:{url,...}} …) once all
    are terminal; raises on any failed creation."""
    if len(identifiers) > 8:  # creations_wait takes at most 8 identifiers
        chunks = [identifiers[i : i + 8] for i in range(0, len(identifiers), 8)]
        with ThreadPoolExecutor(max_workers=len(chunks)) as pool:
            results = list(pool.map(wait_for_creations, chunks))
        return [entry for chunk in results for entry in chunk]

    for _ in range(config.WAIT_MAX_POLLS):
        response = client().structured(
            "creations_wait",
            {"identifiers": identifiers, "timeoutSeconds": config.WAIT_TIMEOUT_SECONDS},
        )
        entries = response.get("results")
        if response.get("allTerminal") and isinstance(entries, list):
            # allTerminal is not trusted blindly: every requested identifier
            # must be present, and the result keeps the request order (the
            # server's order is unspecified).
            by_id = {entry.get("identifier"): entry for entry in entries if isinstance(entry, dict)}
            missing = [identifier for identifier in identifiers if identifier not in by_id]
            if missing:
                raise McpError(f"creations_wait returned no status for {len(missing)} of {len(identifiers)} creations")
            ordered = [by_id[identifier] for identifier in identifiers]
            failed = [entry for entry in ordered if _is_failed(str(entry.get("status", "")))]
            if failed:
                reason = failed[0].get("failureReason") or failed[0].get("status")
                raise McpError(f"Generation failed: {reason}")
            return ordered
    raise McpError("Timed out waiting for the generation to finish")


def result_urls(entries: List[dict]) -> List[str]:
    urls = []
    for entry in entries:
        results = entry.get("results")
        candidates = results if isinstance(results, list) else [results] if isinstance(results, dict) else []
        url = entry.get("url") or next(
            (item.get("url") or item.get("originalUrl") for item in candidates if isinstance(item, dict)),
            None,
        )
        if url:
            urls.append(url)
    return urls


# ---------------------------------------------------------------------------
# Generation / upscale / stock wrappers. Param maps mirror the panel services
# byte-for-byte (services/generation.ts, services/edit.ts, services/stock.ts).


def extract_creation_refs(structured: dict) -> List[dict]:
    # Refs without an identifier are unusable downstream (creations_wait
    # requires one) — dropping them here turns a malformed payload into the
    # callers' clear "returned no creations" error instead of a KeyError.
    if isinstance(structured.get("creations"), list):
        return [ref for ref in structured["creations"] if isinstance(ref, dict) and ref.get("identifier")]
    creation = structured.get("creation")
    if isinstance(creation, dict) and creation.get("identifier"):
        return [creation]
    return []


def generate_args(
    prompt: str,
    mode: str = "auto",
    count: int = 1,
    aspect_ratio: Optional[str] = None,
    references: Optional[List[dict]] = None,
    folder_reference: Optional[str] = None,
    resolution: Optional[str] = None,
    seed: Optional[int] = None,
) -> dict:
    args = {"prompt": prompt, "mode": mode, "count": count}
    # The server picks a random seed when none is sent, so the node's widget
    # has to travel for a saved workflow to reproduce its result.
    if seed is not None:
        args["seed"] = int(seed)
    # 'auto' is not a valid aspectRatio value — omitting the field asks the
    # model to match the reference's ratio.
    if aspect_ratio and aspect_ratio != "auto":
        args["aspectRatio"] = aspect_ratio
    # Same for resolution: the pixel tier ('1k'/'2k'/'4k') is per model and
    # omitting the field is the only way to ask for the model's own default.
    if resolution and resolution != "auto":
        args["resolution"] = resolution
    if references:
        args["references"] = references
    if folder_reference:
        args["folderReference"] = folder_reference
    return args


def generate_image(
    prompt: str,
    mode: str = "auto",
    count: int = 1,
    aspect_ratio: Optional[str] = None,
    references: Optional[List[dict]] = None,
    folder_reference: Optional[str] = None,
    resolution: Optional[str] = None,
    seed: Optional[int] = None,
) -> List[dict]:
    args = generate_args(prompt, mode, count, aspect_ratio, references, folder_reference, resolution, seed)
    refs = extract_creation_refs(client().structured("images_generate", args))
    if not refs:
        raise McpError("images_generate returned no creations")
    return refs


# The upscale combos show the AI Suite display names (Illusio/Sharpy/… and the
# Magnific v1/v2 mode family) instead of API slugs; execution and the /cost
# endpoint map them back here. ComfyUI validates a queued combo value against
# the CURRENT choice list before any node code runs, so the *_CHOICES lists
# keep the legacy raw slugs as valid entries (web/magnific.js hides them from
# the dropdown and rewrites loaded legacy values to their label) — the
# label→slug mapping alone cannot save a workflow saved before the labels.
UPSCALE_ENGINE_BY_LABEL = {
    "Automatic": "automatic",
    "Illusio": "magnific_illusio",
    "Sharpy": "magnific_sharpy",
    "Sparkle": "magnific_sparkle",
}
UPSCALE_ENGINE_CHOICES = list(UPSCALE_ENGINE_BY_LABEL) + list(UPSCALE_ENGINE_BY_LABEL.values())

# 'Automatic' omits the field so the server picks, same as the panel's default.
UPSCALE_MODE_BY_LABEL = {
    "Automatic": None,
    "Magnific v1 (high HDR)": "ultra",
    "Magnific v2 (sublime)": "ultra-sublime",
    "Magnific v2 (photo)": "ultra-photo",
    "Magnific v2 (photo denoiser)": "ultra-denoiser",
}
UPSCALE_MODE_CHOICES = list(UPSCALE_MODE_BY_LABEL) + ["default"] + [
    slug for slug in UPSCALE_MODE_BY_LABEL.values() if slug
]


def upscale_engine_slug(value: Optional[str]) -> Optional[str]:
    return UPSCALE_ENGINE_BY_LABEL.get(value, value)


def upscale_mode_slug(value: Optional[str]) -> Optional[str]:
    if value in UPSCALE_MODE_BY_LABEL:
        return UPSCALE_MODE_BY_LABEL[value]
    return None if value in (None, "", "default") else value


def upscale_args(creation_identifier: Optional[str], folder_reference: Optional[str], options: dict) -> dict:
    args = {"scale": options.get("scale", "2x")}
    if creation_identifier:
        args["creationIdentifier"] = creation_identifier
    for key in ("precision", "presets", "engine", "mode", "prompt"):
        if options.get(key):
            args[key] = options[key]
    for key in ("creativity", "resemblance", "hdr", "fractality", "sharpness", "grain", "ultraDetail"):
        if options.get(key) is not None:
            args[key] = options[key]
    if folder_reference:
        args["folderReference"] = folder_reference
    return args


def upscale_image(creation_identifier: str, folder_reference: Optional[str], options: dict) -> dict:
    args = upscale_args(creation_identifier, folder_reference, options)
    refs = extract_creation_refs(client().structured("images_upscale", args))
    if not refs:
        raise McpError("images_upscale returned no creation")
    return refs[0]


def simulate_cost(tool: str, arguments: dict) -> dict:
    """Credit-cost preview ({credits, isUnlimited, certainty, range?})."""
    return client().structured("simulate_cost", {"tool": tool, "arguments": arguments})


def _single_creation(structured: dict, tool: str) -> dict:
    refs = extract_creation_refs(structured)
    if not refs:
        raise McpError(f"{tool} returned no creation")
    return refs[0]


# Keyframes/references take the creation identifier in the `url` field, and
# `type` is the MEDIA KIND of that identifier — a video reference announced as an
# image is routed to the image validators and rejected.
def _clip_media(kind: str, identifier: str) -> dict:
    return {"type": kind, "url": identifier}


# video_generate is clip-based; the node drives exactly one clip.
def generate_video_args(
    prompt: str,
    slug: Optional[str],
    duration: int,
    aspect_ratio: Optional[str] = None,
    resolution: Optional[str] = None,
    sound_effects: bool = False,
    start_identifier: Optional[str] = None,
    end_identifier: Optional[str] = None,
    references: Optional[List[dict]] = None,
    folder_reference: Optional[str] = None,
    library_references: Optional[List[dict]] = None,
    seed: Optional[int] = None,
) -> dict:
    """`references` items are {"type": "image"|"video", "identifier": …} — the
    same {type, identifier} vocabulary the images_generate wrapper takes.
    `library_references` are already in clip shape (library_video_references)
    and share the clip's `references` array."""
    clip: dict = {"prompt": prompt, "duration": duration}
    if slug and slug != "auto":
        clip["slug"] = slug
    # Clip-level: the video clients read `$clip['seed']` (Seedance appends it
    # as `--seed N`). Models without seed support ignore it server-side.
    if seed is not None:
        clip["seed"] = int(seed)
    if aspect_ratio and aspect_ratio != "auto":
        clip["aspectRatio"] = aspect_ratio
    if resolution:
        clip["resolution"] = resolution
    if sound_effects:
        clip["withSoundEffects"] = True
    keyframes = {}
    # Start/end frames are images by contract, not by assumption: the request
    # validator pins keyframes.start/end.type to `image` (a source video travels
    # as a video reference instead).
    if start_identifier:
        keyframes["start"] = _clip_media("image", start_identifier)
    if end_identifier:
        keyframes["end"] = _clip_media("image", end_identifier)
    if keyframes:
        clip["keyframes"] = keyframes
    clip_references = [_clip_media(ref["type"], ref["identifier"]) for ref in references or []]
    clip_references.extend(library_references or [])
    if clip_references:
        clip["references"] = clip_references
    args: dict = {"video": {"clips": [clip]}}
    if folder_reference:
        args["folderReference"] = folder_reference
    return args


def generate_video(args: dict) -> List[dict]:
    structured = client().structured("video_generate", args)
    refs = extract_creation_refs(structured)
    if not refs:
        raise McpError("video_generate returned no creations")
    return refs


# Mirrors the panel's VideoEditService: Topaz upscales by factor and needs its
# own neutral param set; Magnific/Precision target an output width.
_TOPAZ_DEFAULTS = {
    "targetFps": 0,
    "focus": 0.5,
    "sharpen": 0.5,
    "enhancementModel": "proteus",
    "frameInterpolation": "chronos",
}


def upscale_video_args(creation_identifier: Optional[str], mode: str, options: dict) -> dict:
    if mode == "topaz":
        mode_params: dict = {"upscaleFactor": options.get("upscaleFactor", 2), **_TOPAZ_DEFAULTS}
    else:
        mode_params = {"targetResolution": options.get("targetResolution", 1920)}
        for key in ("fpsBoost", "smartGrain", "sharpen", "creativity", "flavor", "premiumQuality", "turbo", "strength"):
            if options.get(key) is not None:
                mode_params[key] = options[key]
    args = {"mode": mode, **mode_params}
    if creation_identifier:
        args["creationIdentifier"] = creation_identifier
    if options.get("preview"):
        args["preview"] = True
    if options.get("folderReference"):
        args["folderReference"] = options["folderReference"]
    return args


def upscale_video(args: dict) -> dict:
    return _single_creation(client().structured("video_upscale", args), "video_upscale")


MUSIC_MODELS = [
    "google-lyria",
    "google-lyria-3",
    "google-lyria-3-pro",
    "elevenlabs-music-generation",
    "elevenlabs-music-generation-v2",
]

# Known fixed-duration models (30s server-side); everything else — including
# models discovered dynamically from the schema enum — gets durationSeconds.
MUSIC_FIXED_DURATION = {"google-lyria", "google-lyria-3"}

# AI Suite display names for the music combo (mirrors the audio panel's
# catalog); slugs the map doesn't know — e.g. a model added to the schema enum
# later — show and pass through as-is. The node's choice list also keeps the
# raw slugs as valid (hidden) entries for pre-label saved workflows.
MUSIC_MODEL_LABELS = {
    "google-lyria": "Google Lyria",
    "google-lyria-3": "Lyria 3 Short",
    "google-lyria-3-pro": "Lyria 3 Long",
    "elevenlabs-music-generation": "ElevenLabs Music",
    "elevenlabs-music-generation-v2": "ElevenLabs Music v2",
}
_MUSIC_SLUG_BY_LABEL = {label: slug for slug, label in MUSIC_MODEL_LABELS.items()}
# A duplicated label would silently drop a slug from the reverse map and route
# the combo pick to the wrong model.
assert len(_MUSIC_SLUG_BY_LABEL) == len(MUSIC_MODEL_LABELS), "duplicate music model label"


def music_model_label(slug: str) -> str:
    return MUSIC_MODEL_LABELS.get(slug, slug)


def music_model_slug(value: str) -> str:
    return _MUSIC_SLUG_BY_LABEL.get(value, value)


def music_args(
    prompt: str,
    model: str,
    duration_seconds: int,
    instrumental: bool,
    folder_reference: Optional[str] = None,
    force_duration: bool = False,
) -> dict:
    args: dict = {"prompt": prompt, "model": model, "instrumental": instrumental}
    # The cost simulator requires durationSeconds for every model (force_duration).
    if force_duration or model not in MUSIC_FIXED_DURATION:
        args["durationSeconds"] = duration_seconds
    if folder_reference:
        args["folderReference"] = folder_reference
    return args


def generate_music(args: dict) -> dict:
    return _single_creation(client().structured("audio_music_generate", args), "audio_music_generate")


def tts_args(text: str, voice_id: int, speed: float, folder_reference: Optional[str] = None) -> dict:
    args: dict = {"text": text, "voiceId": voice_id}
    if abs(speed - 1.0) > 1e-6:
        args["speed"] = speed
    if folder_reference:
        args["folderReference"] = folder_reference
    return args


def generate_tts(args: dict) -> dict:
    return _single_creation(client().structured("audio_tts", args), "audio_tts")


def skin_enhance(creation_identifier: str, folder_reference: Optional[str], options: dict) -> dict:
    args: dict = {"creationIdentifier": creation_identifier}
    for key in ("version", "optimizedFor"):
        if options.get(key):
            args[key] = options[key]
    for key in ("sharpen", "smartGrain", "skinDetail"):
        if options.get(key) is not None:
            args[key] = options[key]
    if folder_reference:
        args["folderReference"] = folder_reference
    return _single_creation(client().structured("images_skin_enhancer", args), "images_skin_enhancer")


def remove_background(creation_identifier: str, folder_reference: Optional[str]) -> dict:
    args: dict = {"creationIdentifier": creation_identifier}
    if folder_reference:
        args["folderReference"] = folder_reference
    return _single_creation(client().structured("images_remove_background", args), "images_remove_background")


def retouch(
    creation_identifier: str,
    mask_identifier: str,
    mode: str,
    prompt: Optional[str],
    folder_reference: Optional[str],
) -> dict:
    args: dict = {
        "creationIdentifier": creation_identifier,
        "maskCreationIdentifier": mask_identifier,
        "mode": mode,
    }
    # Erase ignores the prompt server-side — omit it so the call states intent.
    if mode == "replace" and prompt:
        args["prompt"] = prompt
    if folder_reference:
        args["folderReference"] = folder_reference
    return _single_creation(client().structured("images_retouch", args), "images_retouch")


def stock_search(query: str, filters: dict, page: int = 1, per_page: int = 20) -> dict:
    args = {"query": query, "page": page, "per_page": per_page}
    for key in ("content_type", "license", "ai_generated", "orientation"):
        if filters.get(key):
            args[key] = filters[key]
    return client().structured("stock_search", args)


def stock_download(item_id: int, item_type: Optional[str], folder_reference: Optional[str]) -> dict:
    # Illustrations rasterise through the vector path (stock_download has no
    # `illustration` type); unknown types download as photo renders.
    normalized = "vector" if item_type == "illustration" else item_type
    kind = normalized if normalized in ("vector", "psd", "video", "icon") else "photo"
    args = {"id": item_id, "type": kind}
    if folder_reference:
        # Stock tools take snake_case folder_reference (UUID), unlike the
        # camelCase folderReference of the images_* tools.
        args["folder_reference"] = folder_reference
    return client().structured("stock_download", args)


# ---------------------------------------------------------------------------
# Folder catalog for the Save To node. Flattens projects and their nested
# folders into "Project / Folder / …" display paths, children listed right
# under their parent. Cached briefly so INPUT_TYPES (called on every
# /object_info refresh) stays cheap.

_folders_cache: dict = {"at": 0.0, "choices": []}
_folders_lock = threading.Lock()
# Folders change mid-session (create folder → press R), so they expire fast;
# the model/voice catalogs change rarely and keep the long TTL.
_FOLDERS_TTL_SECONDS = 30
_CATALOG_TTL_SECONDS = 120
# folders_list emits NO child counts, so every folder must be probed for
# children. Level 1 (folders inside each project — the two-level UX) gets its
# own generous cap so a large library never leaves later projects folder-less;
# deeper nesting shares a small budget. Anything beyond the caps simply lists
# without its nested folders.
_FOLDER_CHILD_PROBES = 32
_FOLDER_DEEP_PROBES = 8
_MAX_FOLDER_DEPTH = 3


def _uniquify_labels(choices: List[dict]) -> List[dict]:
    # Projects from different workspaces can share a name; the combo maps the
    # picked label back to a reference, so labels must be unique.
    seen: dict = {}
    for choice in choices:
        count = seen.get(choice["label"], 0)
        seen[choice["label"]] = count + 1
        if count:
            choice["label"] = f"{choice['label']} ({count + 1})"
    return choices


def _fetch_folder_choices() -> List[dict]:
    projects = parse_folders(client().text("folders_list", {"onlyProjects": True}))
    roots = [
        {"label": project["name"], "reference": project["reference"], "children": []}
        for project in projects
    ]

    deep_budget = _FOLDER_DEEP_PROBES
    frontier = roots
    for depth in range(_MAX_FOLDER_DEPTH - 1):
        if not frontier:
            break
        if depth == 0:
            batch = frontier[:_FOLDER_CHILD_PROBES]
        else:
            if deep_budget <= 0:
                break
            batch = frontier[:deep_budget]
            deep_budget -= len(batch)
        frontier = []
        with ThreadPoolExecutor(max_workers=4) as pool:
            children_per_node = list(
                pool.map(
                    lambda node: parse_folders(
                        client().text("folders_list", {"parentReference": node["reference"]})
                    ),
                    batch,
                )
            )
        for node, children in zip(batch, children_per_node):
            node["children"] = [
                {"label": f"{node['label']} / {child['name']}", "reference": child["reference"], "children": []}
                for child in children
            ]
            frontier.extend(node["children"])

    choices: List[dict] = []

    def flatten(node: dict) -> None:
        choices.append({"label": node["label"], "reference": node["reference"]})
        for child in node["children"]:
            flatten(child)

    for root in roots:
        flatten(root)
    return _uniquify_labels(choices)


def folder_choices(force: bool = False) -> List[dict]:
    with _folders_lock:
        # Freshness by timestamp, not content: an empty catalog is a valid,
        # cacheable answer (truthiness would re-crawl on every refresh).
        fresh = _folders_cache["at"] and time.time() - _folders_cache["at"] < _FOLDERS_TTL_SECONDS
        if not force and fresh:
            return _folders_cache["choices"]
        choices = _fetch_folder_choices()
        _folders_cache.update(at=time.time(), choices=choices)
        return choices


def clear_catalog_caches() -> None:
    """Drop every per-session catalog cache — folders, models, voices belong
    to an account, so sign-out/sign-in must not leak them across sessions.
    Each reset takes the cache's own lock, so a fetch in flight during the
    switch finishes its write first and is then wiped, never the reverse."""
    with _folders_lock:
        _folders_cache.update(at=0.0, choices=[])
    with _image_models_lock:
        _image_models_cache.update(at=0.0, models=[])
    with _video_models_lock:
        _video_models_cache.update(at=0.0, models=[])
    with _music_models_lock:
        _music_models_cache.update(at=0.0, models=[])
    with _voices_lock:
        _voices_cache.update(at=0.0, choices=[])
    with _library_lock:
        for cache in _library_caches.values():
            cache.update(at=0.0, choices=[])


def folder_reference_for(label: str) -> Optional[str]:
    # Always resolve through folder_choices() so the lookup honors the TTL —
    # reading the raw cache would serve stale references after expiry.
    for choice in folder_choices():
        if choice["label"] == label:
            return choice["reference"]
    return None


# Model/voice catalogs for the node combos, same caching rationale.

_image_models_cache: dict = {"at": 0.0, "models": []}
_image_models_lock = threading.Lock()


def image_models() -> List[dict]:
    with _image_models_lock:
        if _image_models_cache["at"] and time.time() - _image_models_cache["at"] < _CATALOG_TTL_SECONDS:
            return _image_models_cache["models"]
        models = parse_image_models(client().text("images_models_list"))
        _image_models_cache.update(at=time.time(), models=models)
        return models


def model_choices() -> List[str]:
    return [model["slug"] for model in image_models()]


def parse_video_models(text: str) -> List[dict]:
    """video_models_list TOON: per model, the `durations[N]:`, `resolutions[N]:`
    and `aspectRatios[N]:` CSV lines drive the node's dependent dropdowns. The
    nested `references[N]:` block (list form `- type: image` or tabular
    `references[N]{type,limit}:` + CSV rows) lists the reference types the
    model takes, Library ones included — that gates the Library Reference input."""
    models: List[dict] = []
    current: Optional[dict] = None
    # Inside a model's `references` block: its header indent, and for the
    # tabular form the index of the `type` column (-1 = list form).
    refs_indent: Optional[int] = None
    refs_type_column = -1
    for line in text.split("\n"):
        if not line.strip():
            continue
        slug = re.match(r"^\s*-\s*slug:\s*(\S+)", line)
        if slug:
            current = {"slug": slug.group(1), "durations": [], "resolutions": [], "aspectRatios": [], "referenceTypes": []}
            models.append(current)
            refs_indent = None
            continue
        if current is None:
            continue
        if refs_indent is not None:
            if _indent(line) > refs_indent:
                if refs_type_column >= 0:
                    values = _toon_split_csv(line.strip())
                    ref_type = values[refs_type_column] if refs_type_column < len(values) else ""
                else:
                    match = re.match(r"^\s*-\s*type:\s*(\S+)", line)
                    ref_type = _toon_unquote(match.group(1)) if match else ""
                if ref_type and ref_type not in current["referenceTypes"]:
                    current["referenceTypes"].append(ref_type)
                continue
            refs_indent = None  # dedent closes the block; re-read the line as a model field
        refs_header = re.match(r"^(\s+)references\[\d+\](\{[^}]*\})?:\s*$", line)
        if refs_header:
            refs_indent = len(refs_header.group(1))
            columns = [c.strip() for c in refs_header.group(2)[1:-1].split(",")] if refs_header.group(2) else []
            refs_type_column = columns.index("type") if "type" in columns else -1
            continue
        for key, pattern in (
            ("durations", r"^\s+durations\[\d+\]:\s*(.+?)\s*$"),
            ("resolutions", r"^\s+resolutions\[\d+\]:\s*(.+?)\s*$"),
            ("aspectRatios", r"^\s+aspectRatios\[\d+\]:\s*(.+?)\s*$"),
        ):
            match = re.match(pattern, line)
            if match:
                values = [v.strip().strip('"') for v in match.group(1).split(",")]
                if key == "durations":
                    current[key] = [int(v) for v in values if v.isdigit() and int(v) > 0]
                else:
                    current[key] = [v for v in values if v]
                break
    return models


_video_models_cache: dict = {"at": 0.0, "models": []}
_video_models_lock = threading.Lock()


def video_models() -> List[dict]:
    with _video_models_lock:
        if _video_models_cache["at"] and time.time() - _video_models_cache["at"] < _CATALOG_TTL_SECONDS:
            return _video_models_cache["models"]
        models = parse_video_models(client().text("video_models_list"))
        _video_models_cache.update(at=time.time(), models=models)
        return models


def video_model_choices() -> List[str]:
    return [model["slug"] for model in video_models()]


_music_models_cache: dict = {"at": 0.0, "models": []}
_music_models_lock = threading.Lock()


def music_model_choices() -> List[str]:
    """There is no music-models catalog tool; the enum lives in the
    audio_music_generate input schema published by tools/list."""
    with _music_models_lock:
        if _music_models_cache["at"] and time.time() - _music_models_cache["at"] < _CATALOG_TTL_SECONDS:
            return _music_models_cache["models"]
        tools = {tool.get("name"): tool for tool in client().list_tools()}
        enum = (
            tools.get("audio_music_generate", {})
            .get("inputSchema", {})
            .get("properties", {})
            .get("model", {})
            .get("enum")
        )
        models = [str(value) for value in enum] if isinstance(enum, list) and enum else list(MUSIC_MODELS)
        _music_models_cache.update(at=time.time(), models=models)
        return models


def parse_voices(text: str) -> List[dict]:
    """audio_voices_list TOON: one numeric `id:` line per voice (inline or after
    a bare `-`), then name/gender/languages fields."""
    voices: List[dict] = []
    current: Optional[dict] = None
    for line in text.split("\n"):
        id_match = re.search(r"(?:^|\s)id:\s*(\d+)\s*$", line)
        if id_match:
            current = {"id": int(id_match.group(1)), "name": id_match.group(1), "gender": "", "language": ""}
            voices.append(current)
            continue
        if current is None:
            continue
        name = re.match(r"^\s+name:\s*(.+?)\s*$", line)
        if name:
            # Provider names sometimes carry padding inside the quotes.
            current["name"] = re.sub(r'^"(.*)"$', r"\1", name.group(1)).strip()
            continue
        gender = re.match(r'^\s+gender:\s*"?(\w+)"?\s*$', line)
        if gender:
            current["gender"] = gender.group(1)
            continue
        language = re.match(r"^\s+languages(?:\[\d+\])?:\s*(.+?)\s*$", line)
        if language and not current["language"]:
            current["language"] = re.sub(r'^"(.*)"$', r"\1", language.group(1).split(",")[0].strip())
    return voices


_voices_cache: dict = {"at": 0.0, "choices": []}
_voices_lock = threading.Lock()


def voice_choices() -> List[dict]:
    """[{label, id}] — label is 'Name (gender, language)', uniquified."""
    with _voices_lock:
        if _voices_cache["at"] and time.time() - _voices_cache["at"] < _CATALOG_TTL_SECONDS:
            return _voices_cache["choices"]
        choices = []
        for voice in parse_voices(client().text("audio_voices_list")):
            details = ", ".join(part for part in (voice["gender"], voice["language"]) if part)
            choices.append({"label": f"{voice['name']} ({details})" if details else voice["name"], "id": voice["id"]})
        choices = _uniquify_labels(choices)
        _voices_cache.update(at=time.time(), choices=choices)
        return choices


def voice_id_for(label: str) -> Optional[int]:
    for choice in voice_choices():
        if choice["label"] == label:
            return choice["id"]
    return None


# Library assets change mid-session (train a character → press R), so the
# catalog expires as fast as folders do. One cache per scope: the personal and
# the public catalog are separate server calls, and a failed public fetch must
# leave the personal one untouched.
_library_caches: dict = {scope: {"at": 0.0, "choices": []} for scope in LIBRARY_SCOPES}
_library_lock = threading.Lock()


def _fetch_library_entries(scope: str = "mine") -> List[dict]:
    entries: List[dict] = []
    page = 1
    while True:
        text = client().text("library_list", {"page": page, "perPage": LIBRARY_PAGE_SIZE, "scope": LIBRARY_SCOPES[scope]})
        entries.extend(parse_library(text))
        last_page = re.search(r"^\s+lastPage:\s*(\d+)", text, re.M)
        if page >= min(int(last_page.group(1)) if last_page else page, _LIBRARY_MAX_PAGES):
            break
        page += 1
    # A shared asset can arrive through two buckets (favorite + team); keep one.
    seen: set = set()
    return [entry for entry in entries if not (entry["id"] in seen or seen.add(entry["id"]))]


def library_choices(scope: str = "mine", force: bool = False) -> List[dict]:
    """[{label, entry}] for every READY asset of one catalog — an asset still
    processing (or failed) can't guide a generation, so it never reaches the
    combo. `scope` is a LIBRARY_SCOPES key (`mine` | `public`)."""
    if scope not in LIBRARY_SCOPES:
        scope = "mine"
    with _library_lock:
        cache = _library_caches[scope]
        fresh = cache["at"] and time.time() - cache["at"] < _FOLDERS_TTL_SECONDS
        if not force and fresh:
            return cache["choices"]
        choices = sort_library_choices(
            _uniquify_labels(
                [{"label": library_label(entry), "entry": entry} for entry in _fetch_library_entries(scope) if not entry["status"]]
            )
        )
        cache.update(at=time.time(), choices=choices)
        return choices


def library_entry_for(label: str, scope: str = "mine", force: bool = False) -> Optional[dict]:
    for choice in library_choices(scope, force=force):
        if choice["label"] == label:
            return choice["entry"]
    return None


def parse_creations(text: str) -> List[dict]:
    """creations_search TOON: lean items (identifier + prompt/tool)."""
    items: List[dict] = []
    current: Optional[dict] = None
    for line in text.split("\n"):
        identifier = re.match(r"^\s*-\s*identifier:\s*(\S+)", line)
        if identifier:
            current = {"identifier": identifier.group(1), "prompt": "", "tool": ""}
            items.append(current)
            continue
        if current is None:
            continue
        for key in ("prompt", "tool"):
            match = re.match(rf"^\s+{key}:\s*(.+?)\s*$", line)
            if match:
                current[key] = re.sub(r'^"(.*)"$', r"\1", match.group(1))
                break
    return items


def creation_media(identifier: str) -> dict:
    """creations_get TOON → {url, previewUrl, thumbnailUrl, status} (top-level fields)."""
    text = client().text("creations_get", {"creationIdentifier": identifier})
    media = {"identifier": identifier}
    for key in ("url", "previewUrl", "thumbnailUrl", "status"):
        match = re.search(rf"^{key}:\s*(.+?)\s*$", text, re.M)
        if match:
            media[key] = re.sub(r'^"(.*)"$', r"\1", match.group(1))
    return media


def folder_creations(reference: str, is_root: bool, file_type: str = "image", limit: int = 24) -> List[dict]:
    """Creations of one media type in a project/folder, with thumbnails.
    creations_search is lean (no URLs), so thumbnails come from one
    creations_get per item — bounded by `limit` and fetched in parallel.
    Audio creations may carry no thumbnail; the gallery shows a placeholder."""
    text = client().text(
        "creations_search",
        {"from": "project-root" if is_root else "folder", "reference": reference, "fileType": file_type},
    )
    items = parse_creations(text)[:limit]
    if not items:
        return []
    with ThreadPoolExecutor(max_workers=6) as pool:
        media = list(pool.map(lambda item: creation_media(item["identifier"]), items))
    return [
        {**item, "thumbnailUrl": m.get("thumbnailUrl") or m.get("previewUrl") or ""}
        for item, m in zip(items, media)
    ]


def download_bytes(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=config.HTTP_TIMEOUT_SECONDS * 4, context=tls_trust.get_ssl_context()) as response:
        return response.read()
