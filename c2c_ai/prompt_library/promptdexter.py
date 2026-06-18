"""PromptDexter.com scraper.

PromptDexter has no public JSON API.  The site is a Next.js app whose
homepage and category pages render gallery cards directly as HTML
``<article>`` blocks containing the slug + thumbnail + title, and whose
detail pages embed the full prompt body inside the RSC streaming payload
(``self.__next_f.push([1, "..."])``).  We parse both with stdlib regex
to keep this module dependency-free beyond aiohttp (which ships with
ComfyUI).

Public entry point::

    async def search(query: str, *, limit: int = 50) -> list[dict]

Results match the unified prompt-record schema defined in
``lexica.py``'s ``_UNIFIED_SCHEMA`` block.
"""
from __future__ import annotations

import asyncio
import logging
import re
from typing import Any
from urllib.parse import quote_plus

log = logging.getLogger("c2c_ai.prompts.promptdexter")

BASE_URL = "https://promptdexter.com"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 "
    "ComfyUI-C2C-PromptLibrary/1.0"
)
_TIMEOUT_SECS = 15
_DETAIL_CONCURRENCY = 6

# Known category slugs as exposed by the site nav at the time of writing.
# Used to map free-text queries onto a /prompts/<category> browse URL.
_CATEGORY_SLUGS = {
    "anime", "digital-art", "editorial", "illustration",
    "people", "product-photography", "sci-fi", "selfie",
    "traditional-art", "portrait", "fashion", "food",
    "wedding", "travel", "fitness", "animals", "vehicles",
    "architecture", "interiors", "nature", "3d", "cyberpunk",
    "surreal", "cinematic",
}

# Card regex — matches a gallery <article> with an anchor that has the
# title attribute and href="/prompt/<slug>", and an <img> with src/alt.
_CARD_RE = re.compile(
    r'<article[^>]*>\s*<a[^>]*title="(?P<title>[^"]+)"[^>]*'
    r'href="/prompt/(?P<slug>[a-z0-9\-]+)"[^>]*>\s*'
    r'<img[^>]*src="(?P<thumb>[^"]+)"',
    re.IGNORECASE,
)

# RSC payload extractor.
_RSC_RE = re.compile(
    r'self\.__next_f\.push\(\[1,"((?:[^"\\]|\\.)*)"\]\)'
)

# JSON-escape patterns we want to undo on the RSC stream.
_UNESCAPE = [
    (r'\\"', '"'),
    (r'\\n', '\n'),
    (r'\\r', ''),
    (r'\\t', '\t'),
    (r'\\u003c', '<'),
    (r'\\u003e', '>'),
    (r'\\u0026', '&'),
    (r'\\u002f', '/'),
    (r"\\u0027", "'"),
]


def _unescape(s: str) -> str:
    for pat, rep in _UNESCAPE:
        s = re.sub(pat, rep, s)
    # collapse leftover backslash-backslash AFTER the targeted unescapes
    # so we don't double-eat escapes above.
    return s.replace("\\\\", "\\")


def _slugify(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return s


def _humanize_slug(slug: str) -> str:
    return slug.replace("-", " ").strip().capitalize()


async def _fetch(session, url: str) -> str | None:
    try:
        async with session.get(url, allow_redirects=True) as resp:
            if resp.status >= 400:
                return None
            return await resp.text()
    except Exception as exc:  # pragma: no cover — network errors
        log.debug("[promptdexter] fetch %s failed: %s", url, exc)
        return None


def _parse_cards(html: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for m in _CARD_RE.finditer(html):
        slug = m.group("slug")
        if slug in seen:
            continue
        seen.add(slug)
        thumb = m.group("thumb")
        if thumb and thumb.startswith("/"):
            thumb = BASE_URL + thumb
        out.append(
            {
                "slug": slug,
                "title": m.group("title").strip(),
                "thumbnail": thumb,
            }
        )
    return out


def _extract_detail_text(html: str) -> tuple[str, str]:
    """Return (full_prompt_text, meta_description) from a /prompt/<slug> page."""
    # 1) Quick meta description (always present, may be truncated).
    desc = ""
    m = re.search(r'<meta name="description"\s+content="([^"]+)"', html)
    if m:
        desc = m.group(1)

    # 2) Try to recover the full prompt text from the RSC stream which
    #    embeds the canonical prompt body under a "text":"..." key.
    full = ""
    parts: list[str] = []
    for m in _RSC_RE.finditer(html):
        parts.append(m.group(1))
    if parts:
        blob = _unescape("".join(parts))
        # We look for the *longest* "text":"..." string after the prompt
        # heading — the site's detail pages have multiple "text" keys but
        # only the prompt body is long-form prose.
        candidates = re.findall(r'"text":"((?:[^"\\]|\\.){80,5000})"', blob)
        if candidates:
            # pick longest after another unescape pass
            best = max((_unescape(c) for c in candidates), key=len)
            full = best
    if not full:
        full = desc
    return full.strip(), desc.strip()


async def _fetch_detail(session, sem, slug: str) -> tuple[str, str, str]:
    async with sem:
        html = await _fetch(session, f"{BASE_URL}/prompt/{slug}")
    if not html:
        return ("", "", f"{BASE_URL}/prompt/{slug}")
    full, desc = _extract_detail_text(html)
    return (full, desc, f"{BASE_URL}/prompt/{slug}")


async def _browse_pages(session, query: str) -> list[dict[str, Any]]:
    """Decide which URL(s) to scrape based on the query."""
    cards: list[dict[str, Any]] = []
    seen_slugs: set[str] = set()

    def _add(new):
        for c in new:
            if c["slug"] in seen_slugs:
                continue
            seen_slugs.add(c["slug"])
            cards.append(c)

    q = (query or "").strip().lower()
    pages_to_try: list[str] = []

    if q:
        slug = _slugify(q)
        if slug in _CATEGORY_SLUGS:
            pages_to_try.append(f"{BASE_URL}/prompts/{slug}")
        # Always also try the search endpoint variant — site uses a
        # client-side filter, but the homepage HTML still contains every
        # listed slug so we filter post-hoc.
        pages_to_try.append(f"{BASE_URL}/?q={quote_plus(q)}")
        pages_to_try.append(BASE_URL + "/")
    else:
        pages_to_try.append(BASE_URL + "/")

    for url in pages_to_try:
        html = await _fetch(session, url)
        if not html:
            continue
        _add(_parse_cards(html))
        if len(cards) >= 60:
            break

    if q:
        # post-hoc title/slug filter
        tokens = [t for t in re.split(r"\s+", q) if t]
        if tokens:
            def _match(card):
                hay = (card["title"] + " " + card["slug"]).lower()
                return all(t in hay for t in tokens)
            filtered = [c for c in cards if _match(c)]
            if filtered:
                return filtered
            # if filtering yields nothing, fall back to unfiltered cards
    return cards


async def search(query: str, *, limit: int = 50) -> list[dict[str, Any]]:
    """Search PromptDexter and return normalized prompt records.

    Strategy:
      1. Fetch one or more browse pages depending on the query.
      2. Parse the ``<article>`` cards to get slug/title/thumbnail.
      3. Fan out to per-slug detail pages (bounded concurrency) to
         recover the full prompt body from the embedded RSC stream.
      4. Map to the unified record schema.
    """
    try:
        import aiohttp
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("aiohttp not available") from exc

    limit = max(1, min(100, int(limit)))
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml",
        "Accept-Language": "en-US,en;q=0.9",
    }
    timeout = aiohttp.ClientTimeout(total=_TIMEOUT_SECS)

    async with aiohttp.ClientSession(headers=headers, timeout=timeout) as session:
        cards = await _browse_pages(session, query)
        if not cards:
            return []

        cards = cards[:limit]
        sem = asyncio.Semaphore(_DETAIL_CONCURRENCY)
        details = await asyncio.gather(
            *(_fetch_detail(session, sem, c["slug"]) for c in cards),
            return_exceptions=False,
        )

    out: list[dict[str, Any]] = []
    for card, (full, desc, original_url) in zip(cards, details):
        prompt = full or desc
        if not prompt:
            # If we couldn't even get the meta description, still
            # surface the card (title is useful) — keep prompt as title
            # so a downstream AI can expand it.
            prompt = card["title"]
        title = card["title"]
        if len(title) > 90:
            title = title[:87] + "..."
        out.append(
            {
                "id": f"promptdexter:{card['slug']}",
                "source": "promptdexter",
                "source_label": "PromptDexter",
                "title": title,
                "positive": prompt,
                "negative": "",
                "model_hint": "any",
                "thumbnail_url": card["thumbnail"] or "",
                "preview_url": card["thumbnail"] or "",
                "tags": [],
                "categories": [],
                "original_url": original_url,
                "width": 0,
                "height": 0,
                "nsfw": False,
            }
        )
    return out
