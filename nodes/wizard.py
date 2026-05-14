"""
wizard.py — Phase 15: Workflow Wizard / Guided Mode (backend).

Filesystem-backed wizard template store. Each wizard is one JSON file under
`<user>/mec_wizards/<id>.json` shaped:

    {
        "id":          "txt2img_basic",
        "title":       "Text-to-Image (SD 1.5)",
        "description": "5-step guided setup for a basic txt2img pipeline.",
        "version":     1,
        "steps": [
            {
                "title":     "1. Load a model",
                "hint":      "Pick a checkpoint that defines the style.",
                "node_type": "CheckpointLoaderSimple",
                "widget":    "ckpt_name"
            },
            ...
        ]
    }

Ships with several built-in templates that get materialised into the user
folder on first load (only if they are missing — never overwritten).

Routes
------
GET    /mec/wizard/templates           → list (no full steps; for index)
GET    /mec/wizard/templates/{id}      → full template
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
from typing import Any, Dict, List, Optional

log = logging.getLogger("MEC.wizard")

_DIR_LOCK = threading.Lock()

_BUILTINS: List[Dict[str, Any]] = [
    {
        "id": "txt2img_basic",
        "title": "Text-to-Image (SD 1.5 / SDXL)",
        "description": "5-step guided setup for a basic text-to-image pipeline.",
        "version": 1,
        "steps": [
            {"title": "1. Load a model",
             "hint":  "Pick a checkpoint — bigger isn't always better.",
             "node_type": "CheckpointLoaderSimple",
             "widget":    "ckpt_name"},
            {"title": "2. Positive prompt",
             "hint":  "Describe what you want — keep it under 77 tokens.",
             "node_type": "CLIPTextEncode",
             "widget":    "text"},
            {"title": "3. Image size",
             "hint":  "Stick to 512×512 for SD 1.5 or 1024×1024 for SDXL.",
             "node_type": "EmptyLatentImage",
             "widget":    "width"},
            {"title": "4. Sampler",
             "hint":  "euler / dpmpp_2m are safe defaults. 20–30 steps.",
             "node_type": "KSampler",
             "widget":    "sampler_name"},
            {"title": "5. Decode & save",
             "hint":  "VAEDecode → SaveImage. Hit Queue!",
             "node_type": "SaveImage",
             "widget":    None},
        ],
    },
    {
        "id": "img2img_inpaint",
        "title": "Inpainting",
        "description": "Mask an area and regenerate just that region.",
        "version": 1,
        "steps": [
            {"title": "1. Load source image",
             "hint":  "LoadImage — the image you want to modify.",
             "node_type": "LoadImage",
             "widget":    "image"},
            {"title": "2. Mask the area",
             "hint":  "Right-click LoadImage → Open in MaskEditor.",
             "node_type": "LoadImage",
             "widget":    None},
            {"title": "3. Prompt the replacement",
             "hint":  "Describe ONLY what should appear inside the mask.",
             "node_type": "CLIPTextEncode",
             "widget":    "text"},
            {"title": "4. Inpaint sampler",
             "hint":  "Use VAEEncodeForInpaint then KSampler at 0.6–0.85 denoise.",
             "node_type": "KSampler",
             "widget":    "denoise"},
            {"title": "5. Save result",
             "hint":  "VAEDecode → SaveImage.",
             "node_type": "SaveImage",
             "widget":    None},
        ],
    },
    {
        "id": "lora_stack",
        "title": "LoRA Stack",
        "description": "Layer multiple LoRAs onto a base checkpoint.",
        "version": 1,
        "steps": [
            {"title": "1. Base checkpoint",
             "hint":  "Pick the base model the LoRAs were trained against.",
             "node_type": "CheckpointLoaderSimple",
             "widget":    "ckpt_name"},
            {"title": "2. First LoRA",
             "hint":  "LoraLoader — set strength_model around 0.7 to start.",
             "node_type": "LoraLoader",
             "widget":    "lora_name"},
            {"title": "3. Strength tuning",
             "hint":  "Use the 🎚 scrubber (right-click LoraLoader) to fine-tune.",
             "node_type": "LoraLoader",
             "widget":    "strength_model"},
            {"title": "4. Prompt with trigger words",
             "hint":  "Many LoRAs require specific trigger phrases.",
             "node_type": "CLIPTextEncode",
             "widget":    "text"},
            {"title": "5. Generate",
             "hint":  "KSampler → VAEDecode → SaveImage. Done!",
             "node_type": "KSampler",
             "widget":    "seed"},
        ],
    },
]


def _wizards_dir() -> str:
    try:
        import folder_paths  # type: ignore
        base = getattr(folder_paths, "get_user_directory", None)
        if callable(base):
            root = base()
        else:
            root = os.path.join(folder_paths.base_path, "user")
    except Exception:
        root = os.path.join(os.getcwd(), "user")
    d = os.path.join(root, "mec_wizards")
    with _DIR_LOCK:
        os.makedirs(d, exist_ok=True)
    return d


def _seed_builtins() -> None:
    d = _wizards_dir()
    for bi in _BUILTINS:
        path = os.path.join(d, f"{bi['id']}.json")
        if os.path.isfile(path):
            continue
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(bi, f, ensure_ascii=False, indent=2)
            log.info("[wizard] Seeded built-in: %s", bi["id"])
        except Exception as e:
            log.warning("[wizard] Failed to seed %s: %s", bi["id"], e)


def _path_for(wid: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_\-]", "", wid)
    if not safe:
        raise ValueError("invalid wizard id")
    return os.path.join(_wizards_dir(), f"{safe}.json")


def _read(wid: str) -> Optional[Dict[str, Any]]:
    p = _path_for(wid)
    if not os.path.isfile(p):
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        log.warning("[wizard] read %s failed: %s", p, e)
        return None


def _list_all() -> List[Dict[str, Any]]:
    d = _wizards_dir()
    out: List[Dict[str, Any]] = []
    try:
        for fn in sorted(os.listdir(d)):
            if not fn.endswith(".json"):
                continue
            try:
                with open(os.path.join(d, fn), "r", encoding="utf-8") as f:
                    obj = json.load(f)
                out.append({
                    "id":          obj.get("id"),
                    "title":       obj.get("title"),
                    "description": obj.get("description") or "",
                    "step_count":  len(obj.get("steps") or []),
                    "version":     obj.get("version", 1),
                })
            except Exception as e:
                log.debug("[wizard] skip %s: %s", fn, e)
    except FileNotFoundError:
        pass
    return out


def register_routes(server: Any) -> None:
    try:
        from aiohttp import web
    except Exception as e:
        log.warning("[wizard] aiohttp unavailable: %s", e)
        return

    _seed_builtins()
    routes = server.routes

    @routes.get("/mec/wizard/templates")
    async def _list(_req: web.Request) -> web.Response:
        try:
            data = _list_all()
        except Exception as e:
            log.exception("[wizard] list failed")
            return web.json_response(
                {"success": False, "error": "list_failed", "message": str(e)},
                status=500)
        return web.json_response({"success": True, "data": {"templates": data}})

    @routes.get(r"/mec/wizard/templates/{wid:[A-Za-z0-9_\-]+}")
    async def _get(req: web.Request) -> web.Response:
        wid = req.match_info.get("wid", "")
        obj = _read(wid)
        if obj is None:
            return web.json_response(
                {"success": False, "error": "not_found"}, status=404)
        return web.json_response({"success": True, "data": obj})

    log.info("[wizard] Routes registered: /mec/wizard/templates")
