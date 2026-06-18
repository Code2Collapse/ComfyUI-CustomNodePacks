"""HTTP routes exposed under ``/c2c/prompts/*`` for the prompt gallery UI.

    GET  /c2c/prompts/sources            → list configured sources
    GET  /c2c/prompts/categories         → static taxonomy tree (for Builder tab)
    GET  /c2c/prompts/search             → search one source
                                            ?q=<text> &source=<id> &limit=<n>
                                            &nsfw=<0|1>
    POST /c2c/prompts/cache/clear        → drop in-memory cache

Returns are wrapped ``{success: bool, data: ..., message?: str}`` for
consistency with the other ``/c2c/*`` route packs.
"""
from __future__ import annotations

import logging

log = logging.getLogger("c2c_ai.prompts")


# Static taxonomy mirrors promptdexter's top-level structure so the Builder
# tab (Phase 2 second pass) and the Gallery's category chips share one source
# of truth.  Keep additions human-curated.
_TAXONOMY = {
    "realistic": [
        "People", "Selfie", "Portrait", "Editorial",
        "Product Photography", "Fashion", "Food",
        "Wedding", "Travel", "Fitness & Sports",
        "Animals", "Vehicles", "Architecture",
        "Interiors", "Nature",
    ],
    "style": [
        "Anime", "Digital Art", "Traditional Art", "Illustration",
        "3D", "Sci-Fi", "Cyberpunk", "Surreal",
        "Cinematic", "B&W", "Vintage",
    ],
}


def _source_registry() -> dict:
    """Return the dict of currently-enabled source modules.

    Imports are lazy so a broken source can't take down the rest.
    """
    sources = {}
    try:
        from . import lexica
        sources["lexica"] = lexica
    except Exception as exc:  # pragma: no cover
        log.warning("[prompts] lexica disabled: %s", exc)
    try:
        from . import promptdexter
        sources["promptdexter"] = promptdexter
    except Exception as exc:  # pragma: no cover
        log.warning("[prompts] promptdexter disabled: %s", exc)
    return sources


def _source_catalog() -> list[dict]:
    """Public catalog (UI list, includes disabled placeholders)."""
    enabled = set(_source_registry().keys())
    return [
        {"id": "lexica",       "label": "Lexica.art",      "api": True,  "enabled": "lexica" in enabled},
        {"id": "civitai",      "label": "Civitai",         "api": True,  "enabled": "civitai" in enabled},
        {"id": "openart",      "label": "OpenArt",         "api": True,  "enabled": "openart" in enabled},
        {"id": "promptdexter", "label": "PromptDexter",    "api": False, "enabled": "promptdexter" in enabled},
        {"id": "prompthero",   "label": "PromptHero",      "api": False, "enabled": False},
        {"id": "playgroundai", "label": "Playground AI",   "api": False, "enabled": False},
        {"id": "magespace",    "label": "Mage.space",      "api": False, "enabled": False},
    ]


def register_routes(server) -> None:
    try:
        from aiohttp import web
    except Exception as exc:
        log.error("aiohttp unavailable: %s", exc)
        return

    from . import cache as _cache

    routes = server.routes

    def _ok(data, **extra):
        return web.json_response({"success": True, "data": data, **extra})

    def _err(key: str, msg: str, code: int = 400):
        return web.json_response(
            {"success": False, "error": key, "message": msg}, status=code
        )

    @routes.get("/c2c/prompts/sources")
    async def _sources(_req):
        return _ok(_source_catalog())

    @routes.get("/c2c/prompts/categories")
    async def _categories(_req):
        return _ok(_TAXONOMY)

    @routes.get("/c2c/prompts/cache_stats")
    async def _cache_stats(_req):
        return _ok(_cache.stats())

    @routes.post("/c2c/prompts/cache/clear")
    async def _cache_clear(_req):
        _cache.clear()
        return _ok({"cleared": True})

    @routes.get("/c2c/prompts/search")
    async def _search(req):
        q = req.query.get("q", "").strip()
        source = req.query.get("source", "lexica").strip()
        try:
            limit = max(1, min(100, int(req.query.get("limit", "50"))))
        except Exception:
            limit = 50
        allow_nsfw = req.query.get("nsfw", "0").lower() in ("1", "true", "yes", "on")

        sources = _source_registry()
        if source not in sources:
            return _err(
                "unknown_source",
                f"Source '{source}' is not enabled. Available: {sorted(sources)}",
                code=400,
            )

        cache_key = f"{source}::{q.lower()}::{limit}::{int(allow_nsfw)}"
        cached = _cache.get(cache_key)
        if cached is not None:
            return _ok(cached, cached=True, source=source, query=q)

        try:
            results = await sources[source].search(q, limit=limit)
        except Exception as exc:
            log.exception("[prompts] %s search failed for %r", source, q)
            return _err("fetch_failed", f"{source}: {exc}", code=502)

        if not allow_nsfw:
            results = [r for r in results if not r.get("nsfw")]

        _cache.put(cache_key, results)
        return _ok(results, cached=False, source=source, query=q)

    log.info("[c2c.prompts] routes registered (/c2c/prompts/*)")
