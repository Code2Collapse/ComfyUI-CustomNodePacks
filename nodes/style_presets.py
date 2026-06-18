"""
style_presets.py — v2.0 P2: prompt-style preset library.

A preset is a reusable bundle of prompt fragments + sampler defaults that the
user can apply to a CLIPTextEncode (positive/negative) and KSampler pair.

Storage:  <comfyui>/user/c2c_style_presets/<id>.json
Schema:
    {
        "id":        "preset_xxx",
        "name":      "Anime Cinematic",
        "category":  "anime" | "photoreal" | "stylized" | "lighting" | "custom",
        "model_hint": "sdxl" | "sd1.5" | "flux" | "any",
        "positive":  "masterpiece, best quality, ...",
        "negative":  "lowres, blurry, ...",
        "sampler":   {"sampler_name": "dpmpp_2m", "scheduler": "karras",
                      "steps": 30, "cfg": 6.5, "denoise": 1.0},
        "lora_hints": ["lcm-lora-sdxl", "..."],
        "tags":      ["anime", "cinematic"],
        "builtin":   true | false,
        "created":   1714672881.123
    }

Routes
------
GET    /c2c/styles                → list (no full payload — for index)
GET    /c2c/styles/{id}           → full preset
POST   /c2c/styles                → create/update body=full preset (sans id) or {id, ...}
DELETE /c2c/styles/{id}           → delete (built-ins can be hidden, not deleted)
GET    /c2c/styles/seed           → re-seed built-ins (idempotent)

On first import the module seeds 12 built-in presets covering the main
model families.
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
import uuid
from typing import Any, Dict, List, Optional

log = logging.getLogger("C2C.style_presets")

_DIR_LOCK = threading.Lock()
_NAME_RE = re.compile(r"^[\w \-\(\)\.]{1,80}$")
_ID_RE = re.compile(r"^[a-zA-Z0-9_\-]{1,64}$")


# ----------------------------------------------------------- built-ins
BUILTIN_PRESETS: List[Dict[str, Any]] = [
    {
        "id": "builtin_sdxl_photoreal", "name": "Photoreal Portrait (SDXL)",
        "category": "photoreal", "model_hint": "sdxl",
        "positive": "RAW photo, ultra detailed, 85mm portrait, soft natural light, "
                    "skin pores, subsurface scattering, professional color grading, "
                    "high dynamic range",
        "negative": "cartoon, anime, illustration, lowres, blurry, "
                    "oversaturated, plastic skin, deformed hands, extra fingers",
        "sampler": {"sampler_name": "dpmpp_2m", "scheduler": "karras",
                    "steps": 30, "cfg": 6.0, "denoise": 1.0},
        "lora_hints": [], "tags": ["photoreal", "portrait", "sdxl"],
    },
    {
        "id": "builtin_sdxl_anime", "name": "Anime Cinematic (SDXL)",
        "category": "anime", "model_hint": "sdxl",
        "positive": "masterpiece, best quality, ultra detailed, anime key visual, "
                    "cinematic lighting, vibrant colors, dynamic composition, "
                    "studio Ghibli inspired, painterly background",
        "negative": "lowres, bad anatomy, bad hands, jpeg artifacts, watermark, "
                    "signature, blurry, 3d, photoreal",
        "sampler": {"sampler_name": "euler_ancestral", "scheduler": "normal",
                    "steps": 28, "cfg": 7.0, "denoise": 1.0},
        "lora_hints": [], "tags": ["anime", "cinematic", "sdxl"],
    },
    {
        "id": "builtin_sdxl_cyberpunk", "name": "Cyberpunk Neon (SDXL)",
        "category": "stylized", "model_hint": "sdxl",
        "positive": "cyberpunk cityscape, neon signs, wet pavement reflections, "
                    "rain, volumetric fog, cinematic, blade runner aesthetic, "
                    "ultra wide angle, HDR",
        "negative": "lowres, blurry, daylight, rural, washed out, low contrast",
        "sampler": {"sampler_name": "dpmpp_sde", "scheduler": "karras",
                    "steps": 28, "cfg": 7.5, "denoise": 1.0},
        "lora_hints": [], "tags": ["cyberpunk", "neon", "sdxl"],
    },
    {
        "id": "builtin_sdxl_watercolor", "name": "Watercolor Illustration (SDXL)",
        "category": "stylized", "model_hint": "sdxl",
        "positive": "delicate watercolor painting, soft pastel palette, paper texture, "
                    "loose brush strokes, ink outlines, whitespace composition",
        "negative": "photoreal, 3d, oil painting, harsh lines, sharp digital edges",
        "sampler": {"sampler_name": "dpmpp_2m", "scheduler": "karras",
                    "steps": 28, "cfg": 6.5, "denoise": 1.0},
        "lora_hints": [], "tags": ["watercolor", "illustration", "sdxl"],
    },
    {
        "id": "builtin_sdxl_film_noir", "name": "Film Noir B&W (SDXL)",
        "category": "lighting", "model_hint": "sdxl",
        "positive": "black and white film noir, harsh side lighting, venetian blind shadows, "
                    "cigarette smoke, 1940s, 35mm grain, chiaroscuro, detective story",
        "negative": "color, oversaturated, cartoon, anime, modern phone, lowres",
        "sampler": {"sampler_name": "dpmpp_2m", "scheduler": "karras",
                    "steps": 30, "cfg": 6.5, "denoise": 1.0},
        "lora_hints": [], "tags": ["noir", "black and white", "sdxl"],
    },
    {
        "id": "builtin_sdxl_low_poly", "name": "Low-Poly 3D Render (SDXL)",
        "category": "stylized", "model_hint": "sdxl",
        "positive": "low poly 3d render, flat shaded geometric forms, isometric view, "
                    "minimalist pastel palette, soft ambient occlusion, octane render",
        "negative": "photoreal, high detail textures, organic surfaces, blurry",
        "sampler": {"sampler_name": "dpmpp_2m", "scheduler": "karras",
                    "steps": 26, "cfg": 6.0, "denoise": 1.0},
        "lora_hints": [], "tags": ["low-poly", "3d", "sdxl"],
    },
    {
        "id": "builtin_flux_photoreal", "name": "Photoreal (FLUX)",
        "category": "photoreal", "model_hint": "flux",
        "positive": "A hyperrealistic photograph captured on a Hasselblad H6D-100c "
                    "with an 80mm f/2.8 lens, soft window light from camera right, "
                    "shallow depth of field, beautifully detailed subject in focus, "
                    "cinematic color grade reminiscent of a Roger Deakins still.",
        "negative": "",  # FLUX largely ignores negative
        "sampler": {"sampler_name": "euler", "scheduler": "simple",
                    "steps": 20, "cfg": 1.0, "denoise": 1.0},
        "lora_hints": [], "tags": ["photoreal", "flux"],
    },
    {
        "id": "builtin_flux_cinematic_scene", "name": "Cinematic Scene (FLUX)",
        "category": "lighting", "model_hint": "flux",
        "positive": "A wide cinematic still from a 2010s art-house film, golden hour light "
                    "spilling through tall windows, dust particles drifting in the beam, "
                    "muted teal-and-amber color palette, anamorphic lens flare, "
                    "shallow depth of field, 35mm film grain, composed on the rule of thirds.",
        "negative": "",
        "sampler": {"sampler_name": "euler", "scheduler": "simple",
                    "steps": 22, "cfg": 1.0, "denoise": 1.0},
        "lora_hints": [], "tags": ["cinematic", "flux"],
    },
    {
        "id": "builtin_sd15_dreamlike", "name": "Dreamlike Soft (SD1.5)",
        "category": "stylized", "model_hint": "sd1.5",
        "positive": "(masterpiece:1.2), (best quality:1.2), dreamlike, soft pastel colors, "
                    "diffuse bloom lighting, ethereal atmosphere, painterly",
        "negative": "(worst quality:1.4), (low quality:1.4), lowres, blurry, jpeg artifacts, "
                    "watermark, signature, deformed",
        "sampler": {"sampler_name": "dpmpp_2m", "scheduler": "karras",
                    "steps": 28, "cfg": 7.0, "denoise": 1.0},
        "lora_hints": [], "tags": ["dreamlike", "sd1.5"],
    },
    {
        "id": "builtin_any_hd_quality", "name": "Quality booster (any)",
        "category": "stylized", "model_hint": "any",
        "positive": "ultra detailed, sharp focus, intricate textures, high dynamic range, "
                    "professional color grading",
        "negative": "lowres, blurry, jpeg artifacts, watermark, signature, deformed",
        "sampler": {"sampler_name": "dpmpp_2m", "scheduler": "karras",
                    "steps": 30, "cfg": 6.5, "denoise": 1.0},
        "lora_hints": [], "tags": ["quality", "booster"],
    },
    {
        "id": "builtin_any_clean_neg", "name": "Clean negative (any)",
        "category": "custom", "model_hint": "any",
        "positive": "",
        "negative": "lowres, blurry, jpeg artifacts, watermark, signature, text, "
                    "extra limbs, deformed hands, deformed eyes, mutated, lowres, "
                    "cropped, worst quality, low quality",
        "sampler": {},
        "lora_hints": [], "tags": ["negative-only"],
    },
    {
        "id": "builtin_sdxl_oil_painting", "name": "Oil Painting (SDXL)",
        "category": "stylized", "model_hint": "sdxl",
        "positive": "oil painting on canvas, thick impasto strokes, classical chiaroscuro, "
                    "Rembrandt lighting, baroque composition, museum quality",
        "negative": "photoreal, 3d, digital art, flat colors, lowres, blurry",
        "sampler": {"sampler_name": "dpmpp_2m", "scheduler": "karras",
                    "steps": 30, "cfg": 6.5, "denoise": 1.0},
        "lora_hints": [], "tags": ["oil-painting", "classical", "sdxl"],
    },
]


# ----------------------------------------------------------- fs helpers
def _presets_dir() -> str:
    try:
        import folder_paths  # type: ignore
        base = getattr(folder_paths, "get_user_directory", None)
        root = base() if callable(base) else os.path.join(folder_paths.base_path, "user")
    except Exception:
        root = os.path.join(os.getcwd(), "user")
    d = os.path.join(root, "c2c_style_presets")
    with _DIR_LOCK:
        os.makedirs(d, exist_ok=True)
    return d


def _path_for(pid: str) -> str:
    if not _ID_RE.match(pid):
        raise ValueError("invalid preset id")
    return os.path.join(_presets_dir(), f"{pid}.json")


def _read(pid: str) -> Optional[Dict[str, Any]]:
    p = _path_for(pid)
    if not os.path.isfile(p):
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        log.warning("[style_presets] read failed %s: %s", p, e)
        return None


def _write(data: Dict[str, Any]) -> str:
    pid = data["id"]
    p = _path_for(pid)
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, p)
    return p


def _seed_builtins() -> int:
    """Write any missing built-in presets to disk. Returns count written."""
    count = 0
    for bi in BUILTIN_PRESETS:
        p = _path_for(bi["id"])
        if os.path.isfile(p):
            continue
        data = dict(bi)
        data["builtin"] = True
        data["created"] = time.time()
        try:
            _write(data)
            count += 1
        except Exception as e:
            log.warning("[style_presets] seed %s failed: %s", bi["id"], e)
    if count:
        log.info("[style_presets] seeded %d built-in presets", count)
    return count


def _list_all() -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    d = _presets_dir()
    try:
        for fn in sorted(os.listdir(d)):
            if not fn.endswith(".json"):
                continue
            try:
                with open(os.path.join(d, fn), "r", encoding="utf-8") as f:
                    js = json.load(f)
                out.append({
                    "id": js.get("id"),
                    "name": js.get("name"),
                    "category": js.get("category", "custom"),
                    "model_hint": js.get("model_hint", "any"),
                    "tags": js.get("tags", []),
                    "builtin": bool(js.get("builtin", False)),
                    "created": js.get("created", 0),
                })
            except Exception as e:
                log.warning("[style_presets] list skip %s: %s", fn, e)
    except FileNotFoundError:
        pass
    return out


# ----------------------------------------------------------- validation
_ALLOWED_CATEGORIES = {"photoreal", "anime", "stylized", "lighting", "custom"}
_ALLOWED_MODEL_HINTS = {"sdxl", "sd1.5", "sd3", "flux", "wan", "any"}


def _validate_payload(body: Dict[str, Any]) -> Dict[str, Any]:
    name = body.get("name")
    if not isinstance(name, str) or not _NAME_RE.match(name):
        raise ValueError("invalid name (1-80 chars, word/space/-/() only)")
    cat = body.get("category", "custom")
    if cat not in _ALLOWED_CATEGORIES:
        raise ValueError(f"invalid category: {cat}")
    mh = body.get("model_hint", "any")
    if mh not in _ALLOWED_MODEL_HINTS:
        raise ValueError(f"invalid model_hint: {mh}")
    pos = body.get("positive", "")
    neg = body.get("negative", "")
    if not isinstance(pos, str) or not isinstance(neg, str):
        raise ValueError("positive/negative must be strings")
    if len(pos) > 4000 or len(neg) > 4000:
        raise ValueError("positive/negative max 4000 chars")
    sampler = body.get("sampler") or {}
    if not isinstance(sampler, dict):
        raise ValueError("sampler must be an object")
    tags = body.get("tags") or []
    if not (isinstance(tags, list) and all(isinstance(t, str) for t in tags)):
        raise ValueError("tags must be a list of strings")
    return {
        "name": name, "category": cat, "model_hint": mh,
        "positive": pos, "negative": neg,
        "sampler": sampler, "lora_hints": body.get("lora_hints") or [],
        "tags": tags,
    }


# ----------------------------------------------------------- routes
def register_routes(server: Any) -> None:
    try:
        from aiohttp import web
    except Exception as e:
        log.warning("[style_presets] aiohttp unavailable: %s", e)
        return

    routes = server.routes
    _seed_builtins()  # ensure built-ins on disk

    @routes.get("/c2c/styles")
    async def _list(_req: "web.Request") -> "web.Response":
        return web.json_response({"success": True, "data": _list_all()})

    @routes.get("/c2c/styles/seed")
    async def _seed(_req: "web.Request") -> "web.Response":
        n = _seed_builtins()
        return web.json_response({"success": True, "data": {"written": n}})

    @routes.get("/c2c/styles/{pid}")
    async def _get(req: "web.Request") -> "web.Response":
        pid = req.match_info.get("pid", "")
        try:
            data = _read(pid)
        except ValueError as e:
            return web.json_response({"success": False, "error": str(e)}, status=400)
        if data is None:
            return web.json_response({"success": False, "error": "not_found"}, status=404)
        return web.json_response({"success": True, "data": data})

    @routes.post("/c2c/styles")
    async def _save(req: "web.Request") -> "web.Response":
        try:
            body = await req.json()
        except Exception:
            return web.json_response({"success": False, "error": "invalid_json"}, status=400)
        if not isinstance(body, dict):
            return web.json_response({"success": False, "error": "invalid_body"}, status=400)
        try:
            clean = _validate_payload(body)
        except ValueError as e:
            return web.json_response({"success": False, "error": str(e)}, status=400)

        pid = body.get("id")
        is_update = isinstance(pid, str) and _ID_RE.match(pid) and os.path.isfile(_path_for(pid))
        if not is_update:
            pid = f"preset_{uuid.uuid4().hex[:10]}"

        # Built-ins can be overridden by saving with same id; preserve created ts.
        existing = _read(pid) if is_update else None
        record = {
            "id": pid,
            **clean,
            "builtin": bool(existing.get("builtin")) if existing else False,
            "created": existing.get("created", time.time()) if existing else time.time(),
            "updated": time.time(),
        }
        try:
            _write(record)
        except Exception as e:
            log.exception("[style_presets] write failed")
            return web.json_response({"success": False, "error": "write_failed",
                                       "message": str(e)}, status=500)
        return web.json_response({"success": True, "data": record})

    @routes.delete("/c2c/styles/{pid}")
    async def _delete(req: "web.Request") -> "web.Response":
        pid = req.match_info.get("pid", "")
        try:
            p = _path_for(pid)
        except ValueError as e:
            return web.json_response({"success": False, "error": str(e)}, status=400)
        if not os.path.isfile(p):
            return web.json_response({"success": False, "error": "not_found"}, status=404)
        # Refuse to delete built-ins (they would be re-seeded anyway).
        try:
            with open(p, "r", encoding="utf-8") as f:
                rec = json.load(f)
            if rec.get("builtin"):
                return web.json_response(
                    {"success": False, "error": "builtin_immutable",
                     "message": "Built-in presets cannot be deleted; override by saving with the same id."},
                    status=400)
        except Exception:
            pass
        try:
            os.remove(p)
        except Exception as e:
            return web.json_response({"success": False, "error": "delete_failed",
                                       "message": str(e)}, status=500)
        return web.json_response({"success": True, "data": {"id": pid}})

    log.info("[style_presets] Routes registered: GET/POST /c2c/styles, GET/DELETE /c2c/styles/{id}, GET /c2c/styles/seed")
