"""
Model Browser backend routes — P2.1
Routes:
  GET  /c2c/models/search?source=civitai|huggingface&q=...&type=...&page=N
  POST /c2c/models/download   {"source","modelId","fileId","destDir"}
  GET  /c2c/models/dest_dirs
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import urllib.parse
from pathlib import Path
from typing import Any

log = logging.getLogger("C2C.ModelBrowser")

_CIVITAI_BASE = "https://civitai.com/api/v1"
_HF_API_BASE  = "https://huggingface.co/api"

# Local-source type → folder_paths key fragments. Lets the type filter
# narrow LOCAL browsing (e.g. "Checkpoint" -> only the checkpoints dir).
_TYPE_TO_FOLDER = {
    "checkpoint":         ["checkpoints"],
    "lora":               ["loras"],
    "vae":                ["vae"],
    "controlnet":         ["controlnet"],
    "textualinversion":   ["embeddings"],
    "upscaler":           ["upscale_models"],
}

# 5-minute in-memory cache keyed by (source, q, type, page)
_search_cache: dict[tuple, tuple[float, Any]] = {}
_CACHE_TTL = 300.0

# ── helpers ──────────────────────────────────────────────────────────────────

def _now() -> float:
    import time
    return time.monotonic()


def _cache_get(key: tuple) -> Any | None:
    hit = _search_cache.get(key)
    if hit and _now() - hit[0] < _CACHE_TTL:
        return hit[1]
    return None


def _cache_set(key: tuple, val: Any) -> None:
    _search_cache[key] = (_now(), val)


# Civitai's API sits behind Cloudflare, which 403-blocks the default
# `Python/aiohttp` User-Agent with a human-check page, and 401s restricted
# content without a token. Always send a browser-like UA, attach the user's
# CIVITAI_API_KEY when present, and translate the common failures into plain
# English so the panel shows something actionable instead of "HTTP 403".
_BROWSER_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
               "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")


def _friendly_http_error(status: int, url: str) -> str:
    host = urllib.parse.urlparse(url).netloc
    if "civitai" in host:
        if status in (401, 403):
            return ("Civitai blocked the request (HTTP %d — Cloudflare/auth). "
                    "Create an API key at civitai.com → Account Settings → API Keys, "
                    "then set the CIVITAI_API_KEY environment variable and restart ComfyUI."
                    % status)
        if status == 429:
            return "Civitai rate limit hit (HTTP 429). Wait a minute and try again."
    if "huggingface" in host and status in (401, 403):
        return ("HuggingFace denied the request (HTTP %d). For gated models set the "
                "HF_TOKEN environment variable and restart ComfyUI." % status)
    return f"HTTP {status} from {host}"


async def _http_get(url: str, headers: dict | None = None) -> Any:
    import aiohttp
    hdrs = {"User-Agent": _BROWSER_UA, "Accept": "application/json"}
    host = urllib.parse.urlparse(url).netloc
    if "civitai" in host:
        key = os.environ.get("CIVITAI_API_KEY", "").strip()
        if key:
            hdrs["Authorization"] = f"Bearer {key}"
    elif "huggingface" in host:
        tok = os.environ.get("HF_TOKEN", "").strip()
        if tok:
            hdrs["Authorization"] = f"Bearer {tok}"
    if headers:
        hdrs.update(headers)
    async with aiohttp.ClientSession() as sess:
        async with sess.get(url, headers=hdrs, timeout=aiohttp.ClientTimeout(total=15)) as r:
            if r.status != 200:
                raise RuntimeError(_friendly_http_error(r.status, url))
            try:
                return await r.json(content_type=None)
            except Exception:
                # Cloudflare human-check pages return HTML with a 200 on some paths
                raise RuntimeError(_friendly_http_error(403, url))


# ── Civitai search ────────────────────────────────────────────────────────────

# Civitai rejects `page` combined with `query` (HTTP 400: "Cannot use page param
# with query search. Use cursor-based pagination.") and silently IGNORES `page`
# on browse (page 2 returns page-1 items). So ALL Civitai paging is cursor-based:
# page 1 sends no cursor, and each response's metadata.nextCursor is remembered so
# the panel's Next button reaches page N sequentially.
_civitai_cursors: dict[tuple, str] = {}   # (q, type, page) -> cursor that REACHES that page


async def _civitai_search(q: str, model_type: str, page: int) -> dict:
    params: dict[str, Any] = {
        "limit": 20,
        "sort": "Most Downloaded",
    }
    if q:
        params["query"] = q
    if page > 1:
        cur = _civitai_cursors.get((q, model_type, page))
        if cur:
            params["cursor"] = cur
        # Cold jump to page N without a recorded cursor falls back to page 1
        # results — the UI only pages sequentially, so this stays consistent.
    if model_type:
        params["types"] = model_type
    qs = urllib.parse.urlencode(params, doseq=True)
    url = f"{_CIVITAI_BASE}/models?{qs}"
    data = await _http_get(url)
    items = data.get("items", [])
    meta = data.get("metadata", {}) or {}
    if meta.get("nextCursor"):
        _civitai_cursors[(q, model_type, page + 1)] = str(meta["nextCursor"])
    return {"items": items, "total": meta.get("totalItems", len(items))}


# ── HuggingFace search ────────────────────────────────────────────────────────

_HF_TYPE_TAGS = {
    "Checkpoint": "stable-diffusion",
    "LORA": "lora",
    "VAE": "vae",
    "ControlNet": "controlnet",
    "TextualInversion": "textual-inversion",
    "Upscaler": "upscaler",
}

async def _hf_search(q: str, model_type: str, page: int) -> dict:
    params: dict[str, Any] = {
        "limit": 20,
        "offset": (page - 1) * 20,
        "sort": "downloads",
        "direction": -1,
        "full": "true",
    }
    tag = _HF_TYPE_TAGS.get(model_type)
    if tag:
        params["filter"] = tag
    if q:
        params["search"] = q
    qs = urllib.parse.urlencode(params)
    url = f"{_HF_API_BASE}/models?{qs}"
    data = await _http_get(url)
    items = []
    for m in (data if isinstance(data, list) else []):
        items.append({
            "id"           : m.get("modelId") or m.get("id"),
            "name"         : m.get("modelId") or m.get("id"),
            "type"         : model_type or "Model",
            "image"        : None,
            "description"  : m.get("description", ""),
            "stats"        : {"downloadCount": m.get("downloads", 0)},
            "siblings"     : m.get("siblings", []),
            "modelVersions": [],
            "_source"      : "huggingface",
        })
    return {"items": items, "total": len(items)}


# ── Local (on-disk) search ───────────────────────────────────────────────────
# The OFFLINE source: lists models already in ComfyUI's model folders via
# folder_paths.get_filename_list. No network, no API key, no Cloudflare —
# so it ALWAYS works. This is the default source, which is why opening the
# browser no longer shows a "loading error" when Civitai/HF are unreachable.
async def _local_search(q: str, model_type: str, page: int) -> dict:
    try:
        import folder_paths
    except Exception:
        return {"items": [], "total": 0}
    q_lower = q.lower() if q else ""
    items: list[dict] = []
    for key, _paths in folder_paths.folder_names_and_paths.items():
        # Type filter: narrow folders by the type→folder map (loose match).
        if model_type:
            frags = _TYPE_TO_FOLDER.get(model_type.lower(), [model_type.lower()])
            if not any(f in key.lower() for f in frags):
                continue
        try:
            files = folder_paths.get_filename_list(key)
        except Exception:
            files = []
        for fn in files:
            if q_lower and q_lower not in fn.lower():
                continue
            items.append({
                "id"          : fn,
                "name"        : fn,
                "type"        : key,
                "image"       : None,
                "description" : f"{key}/{fn}",
                "stats"       : {"downloadCount": 0},
                "siblings"    : [],
                "modelVersions": [],
                "_source"     : "local",
            })
    per_page = 20
    total = len(items)
    start = (page - 1) * per_page
    return {"items": items[start:start + per_page], "total": total}


# ── ComfyUI dest dirs ─────────────────────────────────────────────────────────

_NOT_MODEL_KEYS = {"custom_nodes"}      # registered in folder_paths but not a place for model files


def _get_comfyui_model_dirs() -> list[str]:
    """The download destinations: every folder_paths model folder, relative to the ComfyUI root when inside it,
    else absolute - exactly what resolve_dest_dir accepts (the old parents[1] arithmetic turned an extra_model_paths
    folder into a wrong relative path)."""
    try:
        import folder_paths
        base = _comfy_base().resolve()
        dirs = []
        for key, paths in folder_paths.folder_names_and_paths.items():
            if key in _NOT_MODEL_KEYS:
                continue
            for p in paths[0]:
                ap = Path(p).resolve()
                try:
                    dirs.append(ap.relative_to(base).as_posix())
                except ValueError:
                    dirs.append(ap.as_posix())
        return sorted(set(dirs)) if dirs else _default_dirs()
    except Exception:
        return _default_dirs()


def _default_dirs() -> list[str]:
    return [
        "models/checkpoints",
        "models/loras",
        "models/vae",
        "models/controlnet",
        "models/embeddings",
        "models/upscale_models",
        "models/ipadapter",
    ]


# ── Download ─────────────────────────────────────────────────────────────────

_HF_REPO_RE = re.compile(r"^[A-Za-z0-9][\w.-]*/[\w.-]+$")
_SAFE_NAME_RE = re.compile(r"^[\w][\w .()+-]*\.[A-Za-z0-9]{1,12}$")


class DownloadRefused(ValueError):
    """A download request that would write outside the model folders, or with an id that is not an id."""


def _model_roots() -> list[Path]:
    """Every folder ComfyUI stores models in (folder_paths), plus models_dir: a download may only land inside one."""
    roots: list[Path] = []
    try:
        import folder_paths
        roots.append(Path(folder_paths.models_dir).resolve())
        for paths in folder_paths.folder_names_and_paths.values():
            for p in paths[0]:
                roots.append(Path(p).resolve())
    except Exception:
        pass
    return roots


def _comfy_base() -> Path:
    try:
        import folder_paths
        return Path(folder_paths.models_dir).parent
    except Exception:
        return Path(__file__).parents[3]  # ComfyUI root


def resolve_dest_dir(dest_dir: str, roots: list[Path] | None = None, base: Path | None = None) -> Path:
    """The absolute folder for a download, refused unless it is inside one of ComfyUI's model folders (L5.01: the
    request body used to be joined to the ComfyUI root unchecked - '../..' or an absolute path wrote anywhere)."""
    roots = _model_roots() if roots is None else roots
    base = _comfy_base() if base is None else base
    raw = str(dest_dir or "").strip()
    if not raw or "\x00" in raw:
        raise DownloadRefused("destDir is empty")
    cand = Path(raw)
    abs_dest = (cand if cand.is_absolute() else base / cand).resolve()
    for r in roots:
        if abs_dest == r or r in abs_dest.parents:
            return abs_dest
    raise DownloadRefused(f"destDir '{raw}' is not inside a ComfyUI model folder; pick one from /c2c/models/dest_dirs")


def download_plan(source: str, model_id: str, file_id: str) -> tuple[str, dict, str]:
    """(url, headers, file name) for a model file. Secrets go in headers only - never in the URL, the response or
    the log (the Civitai token used to ride in ?token= and was returned to the browser)."""
    hdrs = {"User-Agent": _BROWSER_UA}
    if source == "civitai":
        if not str(file_id).isdigit():
            raise DownloadRefused("Civitai fileId must be the numeric file id")
        if not re.fullmatch(r"[\w-]{1,64}", str(model_id)):
            raise DownloadRefused("Civitai modelId must be an id")
        key = os.environ.get("CIVITAI_API_KEY", "").strip()
        if key:
            hdrs["Authorization"] = f"Bearer {key}"
        return f"https://civitai.com/api/download/models/{file_id}", hdrs, f"{model_id}_{file_id}.safetensors"
    if source == "huggingface":
        if not _HF_REPO_RE.fullmatch(str(model_id)):
            raise DownloadRefused("Hugging Face modelId must be 'owner/repo'")
        parts = str(file_id).split("/")
        if not file_id or any(x in ("", ".", "..") for x in parts):
            raise DownloadRefused("Hugging Face fileId must be a file path inside the repo")
        name = parts[-1]
        if not _SAFE_NAME_RE.fullmatch(name):
            raise DownloadRefused(f"file name '{name}' is not a plain model file name")
        tok = os.environ.get("HF_TOKEN", "").strip()
        if tok:
            hdrs["Authorization"] = f"Bearer {tok}"
        url = f"https://huggingface.co/{model_id}/resolve/main/" + "/".join(urllib.parse.quote(x) for x in parts)
        return url, hdrs, name
    raise DownloadRefused(f"Unknown source: {source}")


async def _fetch_to(url: str, hdrs: dict, dest_file: Path) -> bool:
    """Stream to '<file>.part' and rename only when complete; a failed or partial download leaves nothing."""
    import aiohttp

    part = dest_file.with_name(dest_file.name + ".part")
    try:
        async with aiohttp.ClientSession() as sess:
            async with sess.get(url, headers=hdrs, timeout=aiohttp.ClientTimeout(total=3600)) as r:
                if r.status != 200:
                    log.error("Download failed %s: %s", url, _friendly_http_error(r.status, url))
                    return False
                with open(part, "wb") as fh:
                    async for chunk in r.content.iter_chunked(65536):
                        fh.write(chunk)
        os.replace(part, dest_file)
        log.info("Model downloaded: %s", dest_file)
        return True
    except Exception as exc:
        log.error("Download error for %s: %s", url, exc)
        return False
    finally:
        try:
            if part.exists():
                part.unlink()
        except OSError:
            pass


async def _download_model(source: str, model_id: str, file_id: str, dest_dir: str) -> dict:
    """Queue a background download of a model file (Civitai: numeric file id; Hugging Face: path in the repo).
    Raises DownloadRefused for a destination outside the model folders or a malformed id."""
    abs_dest = resolve_dest_dir(dest_dir)
    url, hdrs, fname = download_plan(source, model_id, file_id)
    abs_dest.mkdir(parents=True, exist_ok=True)
    dest_file = abs_dest / fname
    asyncio.create_task(_fetch_to(url, hdrs, dest_file))
    return {"status": "queued", "dest": str(dest_file)}


# ── Route registration ────────────────────────────────────────────────────────

def register_routes(server) -> None:
    from aiohttp import web

    @server.routes.get("/c2c/models/search")
    async def model_search(request: web.Request) -> web.Response:
        source = request.rel_url.query.get("source", "local").lower()
        q      = request.rel_url.query.get("q", "").strip()
        mtype  = request.rel_url.query.get("type", "").strip()
        try:
            page = int(request.rel_url.query.get("page", "1"))
        except ValueError:
            page = 1

        cache_key = (source, q, mtype, page)
        cached = _cache_get(cache_key)
        if cached is not None:
            return web.json_response(cached, headers={"X-C2C-Cache": "hit"})

        try:
            if source == "local":
                result = await _local_search(q, mtype, page)
            elif source == "civitai":
                result = await _civitai_search(q, mtype, page)
            elif source == "huggingface":
                result = await _hf_search(q, mtype, page)
            else:
                return web.json_response({"error": f"Unknown source: {source}"}, status=400)
            _cache_set(cache_key, result)
            return web.json_response(result)
        except Exception as exc:
            log.error("Model search error: %s", exc, exc_info=True)
            return web.json_response({"error": str(exc)}, status=502)

    @server.routes.post("/c2c/models/download")
    async def model_download(request: web.Request) -> web.Response:
        try:
            body    = await request.json()
            source  = body.get("source", "civitai")
            model_id = str(body.get("modelId", ""))
            file_id  = str(body.get("fileId", ""))
            dest_dir = body.get("destDir", "models/checkpoints")
            if not model_id or not file_id:
                return web.json_response({"error": "modelId and fileId required"}, status=400)
            result = await _download_model(source, model_id, file_id, dest_dir)
            return web.json_response(result)
        except DownloadRefused as exc:
            return web.json_response({"error": str(exc)}, status=400)
        except Exception as exc:
            log.error("Model download error: %s", exc, exc_info=True)
            return web.json_response({"error": str(exc)}, status=500)

    @server.routes.get("/c2c/models/dest_dirs")
    async def model_dest_dirs(_request: web.Request) -> web.Response:
        return web.json_response({"dirs": _get_comfyui_model_dirs()})

    log.info("Model browser routes registered (/c2c/models/*).")
