"""Lexica.art search API client.

Public search API — no auth required:
  GET https://lexica.art/api/v1/search?q=<query>
  → { "images": [ { id, src, srcSmall, prompt, gallery, width, height, nsfw, ... } ] }

We normalize each image to the unified prompt-record schema (see _UNIFIED_SCHEMA
at bottom of this module for the canonical shape).
"""
from __future__ import annotations

import logging
from typing import Any
from urllib.parse import quote_plus

log = logging.getLogger("c2c_ai.prompts.lexica")

LEXICA_URL = "https://lexica.art/api/v1/search"
USER_AGENT = "ComfyUI-C2C-PromptLibrary/1.0"
_TIMEOUT_SECS = 15


async def search(query: str, *, limit: int = 50) -> list[dict[str, Any]]:
    """Search Lexica and return normalized prompt records.

    Errors propagate to the caller (api_routes maps them to 502).
    """
    try:
        import aiohttp
    except ImportError as exc:  # pragma: no cover — aiohttp ships with ComfyUI
        raise RuntimeError("aiohttp not available") from exc

    q = (query or "").strip() or "portrait"
    url = f"{LEXICA_URL}?q={quote_plus(q)}"
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    timeout = aiohttp.ClientTimeout(total=_TIMEOUT_SECS)

    async with aiohttp.ClientSession(headers=headers, timeout=timeout) as session:
        async with session.get(url) as resp:
            resp.raise_for_status()
            data = await resp.json(content_type=None)

    images = data.get("images") or []
    out: list[dict[str, Any]] = []
    for img in images[: max(1, int(limit))]:
        prompt = (img.get("prompt") or "").strip()
        if not prompt:
            continue
        img_id = str(img.get("id") or "")
        title = prompt if len(prompt) <= 90 else (prompt[:87] + "...")
        out.append(
            {
                "id": f"lexica:{img_id}",
                "source": "lexica",
                "source_label": "Lexica",
                "title": title,
                "positive": prompt,
                "negative": "",  # lexica does not expose negatives
                "model_hint": "any",
                "thumbnail_url": img.get("srcSmall") or img.get("src") or "",
                "preview_url": img.get("src") or "",
                "tags": [],
                "categories": [],
                "original_url": f"https://lexica.art/prompt/{img_id}" if img_id else "",
                "width": int(img.get("width") or 0),
                "height": int(img.get("height") or 0),
                "nsfw": bool(img.get("nsfw", False)),
            }
        )
    return out


# Canonical shape — keep in sync across every source adapter.
_UNIFIED_SCHEMA = {
    "id": "str — '<source>:<id>'",
    "source": "str — short id (lexica|civitai|...)",
    "source_label": "str — display name",
    "title": "str — single-line preview (<=90 chars)",
    "positive": "str — positive prompt",
    "negative": "str — negative prompt or ''",
    "model_hint": "str — sdxl|sd1.5|flux|sd3|wan|any",
    "thumbnail_url": "str — small preview image URL",
    "preview_url": "str — full preview image URL",
    "tags": "list[str]",
    "categories": "list[str]",
    "original_url": "str — link to source page",
    "width": "int",
    "height": "int",
    "nsfw": "bool",
}
