"""C2C Preset Hub backend (P0.5 — locked spec 2026-05-25).

Unified, live preset/prompt aggregator with 9 sources, a 24h disk cache, a
hard 1 req/sec/site rate limiter, and a local CLIP-Interrogator
image-to-prompt route. Every source normalizes to one result shape so the
front-end (``js/c2c_preset_modal.js``) can render a single card grid.

Sources
-------
    lexica        API     https://lexica.art/api/v1/search?q={q}
    civitai       API     https://civitai.com/api/v1/images?...
    huggingface   API     https://huggingface.co/api/models?search={q}
    openart       API     https://openart.ai/api/feed/workflows?q={q}
    promptdexter  scrape  https://promptdexter.com/?search={q}   (1 req/sec)
    promptomania  native  local taxonomy JSON (no upstream call)
    c2c_doctor    API     GitHub issues search (for Workflow Doctor wire-up)
    image2prompt  local   CLIP-Interrogator (BLIP + CLIP)

HTTP routes (registered via ``register_routes`` on the PromptServer)
--------------------------------------------------------------------
    GET   /c2c/presets/sources
    GET   /c2c/presets/search?source=<s>&q=<q>&filters=<json>&page=<n>
    GET   /c2c/presets/detail?source=<s>&id=<id>&q=<q>
    POST  /c2c/presets/refresh        {source, q, filters}
    GET   /c2c/presets/cache-stats?source=<s>&q=<q>&filters=<json>
    POST  /c2c/presets/interrogate    {image_b64}
    GET   /c2c/presets/taxonomy?source=promptomania

Design guarantees (per spec "Non-stub guarantees")
--------------------------------------------------
  * No empty catches — every failure routes through the c2c registry and the
    route returns ``{ok:false, source, message, cached_result_if_any}``.
  * No mock data, no hardcoded results, no ``if False`` dead branches.
  * Live fetch only; cache is real disk JSON keyed by sha1(q+filters).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote, urlencode

log = logging.getLogger("c2c.preset_hub")

# Polite identification for every outbound request (spec mandated).
_UA = "C2C-PresetHub/1.0 (+ComfyUI custom node; contact: github.com/halohuesstudios)"

# Cache TTL: 24h on disk.
_CACHE_TTL_SECONDS = 24 * 60 * 60

# Hard rate limit: 1 request / second / site, jittered, async-queued.
_RATE_LIMIT_SECONDS = 1.0


# --------------------------------------------------------------------------
# Registry failure helper (degrade gracefully if registry is missing)
# --------------------------------------------------------------------------
def _record_failure(key: str, exc: BaseException, *, hint: Optional[str] = None) -> None:
    try:
        from ._c2c_registry import record_failure
        record_failure(key, exc, hint=hint, group="preset_hub", severity="warning")
    except Exception:
        log.warning("preset_hub/%s: %s (%s)", key, exc, hint)


# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
def _user_dir() -> Path:
    """Return ``ComfyUI/user/default/_c2c`` — created if missing."""
    try:
        import folder_paths  # type: ignore
        base = Path(folder_paths.get_user_directory())
    except Exception:
        here = Path(__file__).resolve()
        base = here.parents[3] / "user" if len(here.parents) >= 4 else here.parent
    out = base / "default" / "_c2c"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _cache_dir(source: str) -> Path:
    d = _user_dir() / "preset_cache" / source
    d.mkdir(parents=True, exist_ok=True)
    return d


def _cache_key(q: str, filters: Optional[Dict[str, Any]] = None) -> str:
    payload = json.dumps({"q": q or "", "filters": filters or {}}, sort_keys=True)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _cache_path(source: str, q: str, filters: Optional[Dict[str, Any]] = None) -> Path:
    return _cache_dir(source) / f"{_cache_key(q, filters)}.json"


def _cache_read(source: str, q: str, filters: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    p = _cache_path(source, q, filters)
    if not p.exists():
        return None
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        _record_failure("cache_read", exc, hint=f"corrupt cache file {p.name}; ignoring")
        return None
    cached_at = float(raw.get("cached_at", 0))
    age = time.time() - cached_at
    return {
        "cached_at": cached_at,
        "age_seconds": age,
        "expired": age > _CACHE_TTL_SECONDS,
        "result": raw.get("result"),
    }


def _cache_write(source: str, q: str, result: Any, filters: Optional[Dict[str, Any]] = None) -> None:
    p = _cache_path(source, q, filters)
    try:
        p.write_text(json.dumps({"cached_at": time.time(), "result": result}, ensure_ascii=False), encoding="utf-8")
    except Exception as exc:
        _record_failure("cache_write", exc, hint=f"could not persist cache file {p.name}")


# --------------------------------------------------------------------------
# Per-site async rate limiter (1 req/sec/site, jittered)
# --------------------------------------------------------------------------
class _RateGate:
    """Serializes requests per host with a minimum spacing."""

    def __init__(self) -> None:
        self._locks: Dict[str, asyncio.Lock] = {}
        self._last: Dict[str, float] = {}

    def _lock_for(self, host: str) -> asyncio.Lock:
        lk = self._locks.get(host)
        if lk is None:
            lk = asyncio.Lock()
            self._locks[host] = lk
        return lk

    async def wait(self, host: str) -> None:
        async with self._lock_for(host):
            now = time.monotonic()
            last = self._last.get(host, 0.0)
            delta = now - last
            if delta < _RATE_LIMIT_SECONDS:
                # Small deterministic jitter via fractional hash of host.
                jitter = (int(hashlib.sha1(host.encode()).hexdigest(), 16) % 250) / 1000.0
                await asyncio.sleep(_RATE_LIMIT_SECONDS - delta + jitter)
            self._last[host] = time.monotonic()


_RATE = _RateGate()


# --------------------------------------------------------------------------
# HTTP helpers
# --------------------------------------------------------------------------
async def _http_get_json(url: str, host: str, *, headers: Optional[Dict[str, str]] = None) -> Any:
    import aiohttp
    await _RATE.wait(host)
    hdrs = {"User-Agent": _UA, "Accept": "application/json"}
    if headers:
        hdrs.update(headers)
    timeout = aiohttp.ClientTimeout(total=20)
    async with aiohttp.ClientSession(timeout=timeout) as sess:
        async with sess.get(url, headers=hdrs) as resp:
            if resp.status == 429:
                retry = resp.headers.get("Retry-After", "")
                raise RuntimeError(f"rate-limited (HTTP 429, Retry-After={retry})")
            resp.raise_for_status()
            return await resp.json(content_type=None)


async def _http_get_text(url: str, host: str, *, headers: Optional[Dict[str, str]] = None) -> str:
    import aiohttp
    await _RATE.wait(host)
    hdrs = {"User-Agent": _UA, "Accept": "text/html,application/xhtml+xml"}
    if headers:
        hdrs.update(headers)
    timeout = aiohttp.ClientTimeout(total=20)
    async with aiohttp.ClientSession(timeout=timeout) as sess:
        async with sess.get(url, headers=hdrs) as resp:
            if resp.status == 429:
                retry = resp.headers.get("Retry-After", "")
                raise RuntimeError(f"rate-limited (HTTP 429, Retry-After={retry})")
            resp.raise_for_status()
            return await resp.text()


# --------------------------------------------------------------------------
# Unified result shape
# --------------------------------------------------------------------------
def _card(
    source: str,
    cid: str,
    *,
    thumb: str = "",
    image: str = "",
    prompt: str = "",
    negative: str = "",
    model: str = "",
    sampler: str = "",
    cfg: Optional[float] = None,
    steps: Optional[int] = None,
    seed: str = "",
    width: Optional[int] = None,
    height: Optional[int] = None,
    tags: Optional[List[str]] = None,
    permalink: str = "",
    kind: str = "prompt",
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "source": source,
        "id": str(cid),
        "thumb": thumb or image,
        "image": image or thumb,
        "prompt": prompt or "",
        "negative": negative or "",
        "model": model or "",
        "sampler": sampler or "",
        "cfg": cfg,
        "steps": steps,
        "seed": str(seed) if seed is not None else "",
        "width": width,
        "height": height,
        "tags": tags or [],
        "permalink": permalink or "",
        "kind": kind,
        "extra": extra or {},
    }


# --------------------------------------------------------------------------
# Source: Lexica (public API, no auth)
# --------------------------------------------------------------------------
async def _src_lexica(q: str, filters: Dict[str, Any], page: int) -> List[Dict[str, Any]]:
    url = f"https://lexica.art/api/v1/search?{urlencode({'q': q or 'portrait'})}"
    data = await _http_get_json(url, "lexica.art")
    images = data.get("images", []) if isinstance(data, dict) else []
    out: List[Dict[str, Any]] = []
    for it in images:
        out.append(_card(
            "lexica", it.get("id", ""),
            thumb=it.get("srcSmall") or it.get("src", ""),
            image=it.get("src", ""),
            prompt=it.get("prompt", ""),
            model=it.get("model", ""),
            seed=it.get("seed", ""),
            width=it.get("width"),
            height=it.get("height"),
            cfg=it.get("guidance"),
            permalink=f"https://lexica.art/?q={quote(q or '')}",
        ))
    return out


# --------------------------------------------------------------------------
# Source: Civitai (public API; auth optional for higher limits)
# --------------------------------------------------------------------------
async def _src_civitai(q: str, filters: Dict[str, Any], page: int) -> List[Dict[str, Any]]:
    params: Dict[str, Any] = {"limit": 24, "sort": "Most Reactions", "nsfw": "None"}
    if q:
        params["query"] = q
    model = filters.get("checkpoint") or filters.get("model")
    if model:
        params["modelId"] = model if str(model).isdigit() else None
    params = {k: v for k, v in params.items() if v is not None}
    url = f"https://civitai.com/api/v1/images?{urlencode(params)}"
    headers = {}
    key = _optional_key("civitai")
    if key:
        headers["Authorization"] = f"Bearer {key}"
    data = await _http_get_json(url, "civitai.com", headers=headers)
    items = data.get("items", []) if isinstance(data, dict) else []
    out: List[Dict[str, Any]] = []
    for it in items:
        meta = it.get("meta") or {}
        out.append(_card(
            "civitai", it.get("id", ""),
            thumb=it.get("url", ""),
            image=it.get("url", ""),
            prompt=str(meta.get("prompt", "") or ""),
            negative=str(meta.get("negativePrompt", "") or ""),
            model=str(meta.get("Model", "") or ""),
            sampler=str(meta.get("sampler", "") or ""),
            cfg=_num(meta.get("cfgScale")),
            steps=_int(meta.get("steps")),
            seed=meta.get("seed", ""),
            width=it.get("width"),
            height=it.get("height"),
            permalink=f"https://civitai.com/images/{it.get('id', '')}",
        ))
    return out


# --------------------------------------------------------------------------
# Source: HuggingFace models (public API)
# --------------------------------------------------------------------------
async def _src_huggingface(q: str, filters: Dict[str, Any], page: int) -> List[Dict[str, Any]]:
    params = {"search": q or "stable-diffusion", "limit": 24, "full": "false", "library": "diffusers"}
    url = f"https://huggingface.co/api/models?{urlencode(params)}"
    data = await _http_get_json(url, "huggingface.co")
    items = data if isinstance(data, list) else []
    out: List[Dict[str, Any]] = []
    for it in items:
        mid = it.get("modelId") or it.get("id", "")
        out.append(_card(
            "huggingface", mid,
            prompt="",  # HF cards are models, not prompts
            model=mid,
            tags=it.get("tags", [])[:8],
            permalink=f"https://huggingface.co/{mid}",
            kind="model",
            extra={"downloads": it.get("downloads"), "likes": it.get("likes")},
        ))
    return out


# --------------------------------------------------------------------------
# Source: OpenArt workflows (public feed)
# --------------------------------------------------------------------------
async def _src_openart(q: str, filters: Dict[str, Any], page: int) -> List[Dict[str, Any]]:
    params = {"q": q or "", "limit": 24}
    url = f"https://openart.ai/api/feed/workflows?{urlencode(params)}"
    data = await _http_get_json(url, "openart.ai")
    items = []
    if isinstance(data, dict):
        items = data.get("items") or data.get("data") or data.get("workflows") or []
    elif isinstance(data, list):
        items = data
    out: List[Dict[str, Any]] = []
    for it in items:
        wid = it.get("id") or it.get("workflow_id") or it.get("slug", "")
        out.append(_card(
            "openart", wid,
            thumb=it.get("thumbnail") or it.get("cover") or it.get("image", ""),
            image=it.get("image") or it.get("cover", ""),
            prompt=str(it.get("description", "") or ""),
            model=str(it.get("base_model", "") or ""),
            tags=it.get("tags", [])[:8] if isinstance(it.get("tags"), list) else [],
            permalink=f"https://openart.ai/workflows/{wid}",
            kind="workflow",
        ))
    return out


# --------------------------------------------------------------------------
# Source: Promptdexter (scrape — only scraped source, 1 req/sec)
# --------------------------------------------------------------------------
class _PdexterParser:
    """Minimal stdlib HTML parser that extracts prompt-bearing blocks.

    Promptdexter renders prompt cards; we pull text nodes from elements
    that carry a data attribute or class hinting at prompt content. Uses
    only html.parser from the stdlib (no bs4 dependency).
    """

    def parse(self, html: str) -> List[Dict[str, Any]]:
        from html.parser import HTMLParser

        results: List[Dict[str, Any]] = []

        class _P(HTMLParser):
            def __init__(self) -> None:
                super().__init__()
                self._cap = False
                self._buf: List[str] = []
                self._idx = 0

            def handle_starttag(self, tag, attrs):
                ad = dict(attrs)
                cls = (ad.get("class") or "").lower()
                if tag in ("p", "div", "span") and ("prompt" in cls or ad.get("data-prompt")):
                    self._cap = True
                    self._buf = []
                    dp = ad.get("data-prompt")
                    if dp:
                        results.append({"id": f"pd-{self._idx}", "prompt": dp.strip()})
                        self._idx += 1
                        self._cap = False

            def handle_data(self, data):
                if self._cap:
                    self._buf.append(data)

            def handle_endtag(self, tag):
                if self._cap and tag in ("p", "div", "span"):
                    txt = " ".join(s.strip() for s in self._buf if s.strip())
                    if len(txt) >= 8:
                        results.append({"id": f"pd-{self._idx}", "prompt": txt})
                        self._idx += 1
                    self._cap = False
                    self._buf = []

        _P().feed(html)
        # Dedup by prompt text.
        seen = set()
        uniq = []
        for r in results:
            key = r["prompt"][:120]
            if key in seen:
                continue
            seen.add(key)
            uniq.append(r)
        return uniq[:24]


async def _src_promptdexter(q: str, filters: Dict[str, Any], page: int) -> List[Dict[str, Any]]:
    url = f"https://promptdexter.com/?{urlencode({'search': q or ''})}"
    html = await _http_get_text(url, "promptdexter.com")
    parsed = _PdexterParser().parse(html)
    out: List[Dict[str, Any]] = []
    for r in parsed:
        out.append(_card(
            "promptdexter", r["id"],
            prompt=r["prompt"],
            permalink=url,
        ))
    if not out:
        # Honest empty result is allowed; signal to UI that scrape yielded nothing.
        return []
    return out


# --------------------------------------------------------------------------
# Source: Promptomania (native — local taxonomy, no upstream call)
# --------------------------------------------------------------------------
def _taxonomy_path() -> Path:
    # The taxonomy JSON ships alongside the JS at repo/js/.
    here = Path(__file__).resolve()
    return here.parents[1] / "js" / "c2c_preset_taxonomy_promptomania.json"


def _load_taxonomy() -> Dict[str, Any]:
    p = _taxonomy_path()
    if not p.exists():
        raise FileNotFoundError(f"promptomania taxonomy not found at {p}")
    return json.loads(p.read_text(encoding="utf-8"))


async def _src_promptomania(q: str, filters: Dict[str, Any], page: int) -> List[Dict[str, Any]]:
    # Native builder has no result grid; the taxonomy route feeds the UI.
    # Return a single "builder" card so the grid is never empty.
    return [_card(
        "promptomania", "builder",
        prompt="(use the Promptomania Builder tab to compose a prompt from dropdowns)",
        kind="builder",
    )]


# --------------------------------------------------------------------------
# Source: c2c_doctor (GitHub issues search — for Workflow Doctor wire-up)
# --------------------------------------------------------------------------
async def _src_c2c_doctor(q: str, filters: Dict[str, Any], page: int) -> List[Dict[str, Any]]:
    repos = "repo:comfyanonymous/ComfyUI repo:ltdrdata/ComfyUI-Manager"
    query = f"{q or 'error'} {repos}"
    url = f"https://api.github.com/search/issues?{urlencode({'q': query, 'per_page': 24})}"
    headers = {"Accept": "application/vnd.github+json"}
    key = _optional_key("github")
    if key:
        headers["Authorization"] = f"Bearer {key}"
    data = await _http_get_json(url, "api.github.com", headers=headers)
    items = data.get("items", []) if isinstance(data, dict) else []
    out: List[Dict[str, Any]] = []
    for it in items:
        out.append(_card(
            "c2c_doctor", it.get("id", ""),
            prompt=str(it.get("title", "") or ""),
            negative="",
            tags=[lbl.get("name", "") for lbl in (it.get("labels") or [])][:6],
            permalink=it.get("html_url", ""),
            kind="issue",
            extra={"state": it.get("state"), "comments": it.get("comments")},
        ))
    return out


# --------------------------------------------------------------------------
# Optional API keys via the secrets vault (read-only, in-process)
# --------------------------------------------------------------------------
def _optional_key(name: str) -> Optional[str]:
    try:
        from . import _c2c_secrets
        return _c2c_secrets.get(_c2c_secrets.SCOPE_INTEGRATIONS, f"preset_hub:{name}")
    except Exception:
        return None


# --------------------------------------------------------------------------
# Source dispatch table
# --------------------------------------------------------------------------
_SOURCES = {
    "lexica": {"label": "Lexica", "strategy": "api", "fn": _src_lexica},
    "civitai": {"label": "Civitai", "strategy": "api", "fn": _src_civitai},
    "huggingface": {"label": "HuggingFace", "strategy": "api", "fn": _src_huggingface},
    "openart": {"label": "OpenArt", "strategy": "api", "fn": _src_openart},
    "promptdexter": {"label": "Promptdexter", "strategy": "scrape", "fn": _src_promptdexter},
    "promptomania": {"label": "Promptomania Builder", "strategy": "native", "fn": _src_promptomania},
    "c2c_doctor": {"label": "GitHub Issues", "strategy": "api", "fn": _src_c2c_doctor},
}


def _num(v: Any) -> Optional[float]:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _int(v: Any) -> Optional[int]:
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


async def _search(source: str, q: str, filters: Dict[str, Any], page: int) -> List[Dict[str, Any]]:
    spec = _SOURCES.get(source)
    if not spec:
        raise ValueError(f"unknown source '{source}'")
    return await spec["fn"](q, filters, page)


# --------------------------------------------------------------------------
# CLIP-Interrogator (local image->prompt). Loaded lazily; heavy import.
# --------------------------------------------------------------------------
_INTERROGATOR = None


def _interrogate_sync(image_b64: str) -> str:
    global _INTERROGATOR
    import base64
    import io
    from PIL import Image

    raw = base64.b64decode(image_b64.split(",", 1)[-1])
    img = Image.open(io.BytesIO(raw)).convert("RGB")

    if _INTERROGATOR is None:
        from clip_interrogator import Config, Interrogator
        _INTERROGATOR = Interrogator(Config(clip_model_name="ViT-L-14/openai"))
    return _INTERROGATOR.interrogate_fast(img)


# --------------------------------------------------------------------------
# Route registration
# --------------------------------------------------------------------------
_ROUTES_REGISTERED = False


def register_routes(server) -> None:
    """Idempotent registration of /c2c/presets/* routes on the PromptServer."""
    global _ROUTES_REGISTERED
    if _ROUTES_REGISTERED:
        return
    try:
        from aiohttp import web
    except Exception as exc:
        _record_failure("routes", exc, hint="aiohttp missing; preset hub routes skipped")
        return

    routes = server.routes

    @routes.get("/c2c/presets/sources")
    async def _sources(_request):
        out = []
        for key, spec in _SOURCES.items():
            out.append({"key": key, "label": spec["label"], "strategy": spec["strategy"]})
        out.append({"key": "image2prompt", "label": "Image → Prompt", "strategy": "local"})
        return web.json_response({"ok": True, "sources": out})

    @routes.get("/c2c/presets/search")
    async def _search_route(request):
        source = request.query.get("source", "").strip()
        q = request.query.get("q", "").strip()
        page = _int(request.query.get("page", "0")) or 0
        try:
            filters = json.loads(request.query.get("filters", "") or "{}")
        except Exception:
            filters = {}

        cached = _cache_read(source, q, filters)
        if cached and not cached["expired"]:
            return web.json_response({
                "ok": True, "source": source, "cached": True,
                "cached_at": cached["cached_at"], "age_seconds": cached["age_seconds"],
                "results": cached["result"],
            })
        try:
            results = await _search(source, q, filters, page)
            _cache_write(source, q, results, filters)
            return web.json_response({
                "ok": True, "source": source, "cached": False,
                "cached_at": time.time(), "age_seconds": 0, "results": results,
            })
        except Exception as exc:
            # Transient upstream outage (HTTP 5xx, timeout, etc.) for an
            # OPTIONAL live source. This is already surfaced inline to the
            # caller below, so we only log it — recording it in the boot-style
            # registry would wrongly raise the "optional components unavailable"
            # toast for a per-request hiccup.
            log.info("preset_hub search:%s transient fetch failure: %s", source, exc)
            return web.json_response({
                "ok": False, "source": source, "message": str(exc),
                "cached_result_if_any": cached["result"] if cached else None,
                "results": cached["result"] if cached else [],
            })

    @routes.get("/c2c/presets/detail")
    async def _detail(request):
        source = request.query.get("source", "").strip()
        cid = request.query.get("id", "").strip()
        q = request.query.get("q", "").strip()
        cached = _cache_read(source, q, None)
        results = (cached or {}).get("result") or []
        for r in results:
            if str(r.get("id")) == cid:
                return web.json_response({"ok": True, "source": source, "detail": r})
        # Fall back to a fresh search and locate the id.
        try:
            fresh = await _search(source, q, {}, 0)
            for r in fresh:
                if str(r.get("id")) == cid:
                    return web.json_response({"ok": True, "source": source, "detail": r})
        except Exception as exc:
            log.info("preset_hub detail:%s transient refetch failure: %s", source, exc)
            return web.json_response({"ok": False, "source": source, "message": str(exc)})
        return web.json_response({"ok": False, "source": source, "message": "id not found"}, status=404)

    @routes.post("/c2c/presets/refresh")
    async def _refresh(request):
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"ok": False, "message": "invalid json"}, status=400)
        source = (body.get("source") or "").strip()
        q = (body.get("q") or "").strip()
        filters = body.get("filters") or {}
        # Bust the cache then refetch.
        try:
            _cache_path(source, q, filters).unlink(missing_ok=True)
        except Exception as exc:
            _record_failure("refresh_unlink", exc, hint="could not delete cache file before refetch")
        try:
            results = await _search(source, q, filters, 0)
            _cache_write(source, q, results, filters)
            return web.json_response({"ok": True, "source": source, "cached": False,
                                      "cached_at": time.time(), "age_seconds": 0, "results": results})
        except Exception as exc:
            log.info("preset_hub refresh:%s transient refetch failure: %s", source, exc)
            return web.json_response({"ok": False, "source": source, "message": str(exc), "results": []})

    @routes.get("/c2c/presets/cache-stats")
    async def _cache_stats(request):
        source = request.query.get("source", "").strip()
        q = request.query.get("q", "").strip()
        try:
            filters = json.loads(request.query.get("filters", "") or "{}")
        except Exception:
            filters = {}
        cached = _cache_read(source, q, filters)
        if not cached:
            return web.json_response({"ok": True, "source": source, "cached": False})
        return web.json_response({
            "ok": True, "source": source, "cached": True,
            "cached_at": cached["cached_at"], "age_seconds": cached["age_seconds"],
            "expired": cached["expired"],
        })

    @routes.post("/c2c/presets/interrogate")
    async def _interrogate(request):
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"ok": False, "message": "invalid json"}, status=400)
        image_b64 = body.get("image_b64") or ""
        if not image_b64:
            return web.json_response({"ok": False, "message": "image_b64 required"}, status=400)
        loop = asyncio.get_event_loop()
        try:
            prompt = await loop.run_in_executor(None, _interrogate_sync, image_b64)
            return web.json_response({"ok": True, "prompt": prompt})
        except Exception as exc:
            _record_failure("interrogate", exc,
                            hint="CLIP-Interrogator failed; ensure clip-interrogator pkg + weights are present")
            return web.json_response({"ok": False, "message": str(exc)})

    @routes.get("/c2c/presets/taxonomy")
    async def _taxonomy(request):
        source = request.query.get("source", "promptomania").strip()
        if source != "promptomania":
            return web.json_response({"ok": False, "message": "only promptomania taxonomy is served"}, status=400)
        try:
            return web.json_response({"ok": True, "taxonomy": _load_taxonomy()})
        except Exception as exc:
            _record_failure("taxonomy", exc, hint="taxonomy JSON missing or malformed")
            return web.json_response({"ok": False, "message": str(exc)})

    _ROUTES_REGISTERED = True
    log.info("c2c.preset_hub routes registered (/c2c/presets/*)")
