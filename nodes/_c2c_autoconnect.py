"""C2C Auto-connect predictor.

Provides slot-level "what node connects next?" predictions to the JS frontend.

Sources of evidence, in order of precedence:
    1. ``user/default/c2c_autoconnect_history.json``         — live + scanned, persisted
    2. Workflow corpus scan ``user/default/workflows/*.json`` — auto-scanned on boot
    3. Bundled curated defaults                              — handpicked common pairings
    4. (Pass 2) LLM fallback                                 — opt-in only

HTTP routes (idempotent registration via ``register_routes(server)``):
    GET  /c2c/autoconnect/suggest?cls=<C>&dir=output&slot=<name>&type=<T>&limit=5
    POST /c2c/autoconnect/record           {edges: [[src_cls, src_slot, dst_cls, dst_slot], ...]}
    GET  /c2c/autoconnect/type_glossary
    GET  /c2c/autoconnect/compatible?type=<T>&dir=input          (or output)
    GET  /c2c/autoconnect/stats

History JSON schema (version 1)::

    {
      "version": 1,
      "edges": {"SrcClass:src_slot->DstClass:dst_slot": <count>},
      "slot_recent": {"Class:dir:slot_name": [{"cls": "...", "slot": "...", "count": N, "ts": ...}]},
      "scanned_files": ["abs/path/wf.json", ...],
      "updated": <unix_ts>
    }

The frontend file is ``js/c2c_autoconnect.js``.
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from collections import defaultdict
from typing import Any, Dict, List, Tuple

log = logging.getLogger("c2c.autoconnect")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
def _user_dir() -> str:
    try:
        import folder_paths  # type: ignore
        gud = getattr(folder_paths, "get_user_directory", None)
        if callable(gud):
            return gud()
        bp = getattr(folder_paths, "base_path", None) or ""
        if bp:
            return os.path.join(bp, "user")
    except Exception:
        pass
    # last-resort guess based on this file's location: …/custom_nodes/<pack>/nodes/_c2c_autoconnect.py
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(here, "..", "..", "..", "user"))


def _history_path() -> str:
    return os.path.join(_user_dir(), "default", "c2c_autoconnect_history.json")


def _workflows_dir() -> str:
    return os.path.join(_user_dir(), "default", "workflows")


# ---------------------------------------------------------------------------
# Bundled curated defaults (high-confidence common pairings)
# ---------------------------------------------------------------------------
# format: "SrcClass:src_slot_name" -> [("DstClass", "dst_slot_name", base_weight), ...]
CURATED: Dict[str, List[Tuple[str, str, int]]] = {
    # CheckpointLoaderSimple outputs
    "CheckpointLoaderSimple:MODEL":    [("KSampler", "model", 50), ("KSamplerAdvanced", "model", 30), ("ModelSamplingDiscrete", "model", 10)],
    "CheckpointLoaderSimple:CLIP":     [("CLIPTextEncode", "clip", 60)],
    "CheckpointLoaderSimple:VAE":      [("VAEDecode", "vae", 50), ("VAEEncode", "vae", 30)],
    # UNet / Diffusion model loaders
    "UNETLoader:MODEL":                [("KSampler", "model", 40), ("KSamplerAdvanced", "model", 25)],
    "DiffusionModelLoader:MODEL":      [("KSampler", "model", 40)],
    # CLIP loaders
    "CLIPLoader:CLIP":                 [("CLIPTextEncode", "clip", 60)],
    "DualCLIPLoader:CLIP":             [("CLIPTextEncode", "clip", 60)],
    # VAE loader
    "VAELoader:VAE":                   [("VAEDecode", "vae", 50), ("VAEEncode", "vae", 30)],
    # CLIPTextEncode → KSampler conditioning
    "CLIPTextEncode:CONDITIONING":     [("KSampler", "positive", 40), ("KSampler", "negative", 40), ("KSamplerAdvanced", "positive", 25)],
    # EmptyLatentImage → KSampler.latent_image
    "EmptyLatentImage:LATENT":         [("KSampler", "latent_image", 50)],
    "EmptySD3LatentImage:LATENT":      [("KSampler", "latent_image", 40)],
    # KSampler / KSamplerAdvanced → VAEDecode → PreviewImage / SaveImage
    "KSampler:LATENT":                 [("VAEDecode", "samples", 60), ("LatentUpscale", "samples", 10)],
    "KSamplerAdvanced:LATENT":         [("VAEDecode", "samples", 60)],
    "VAEDecode:IMAGE":                 [("PreviewImage", "images", 50), ("SaveImage", "images", 40), ("ImageScale", "image", 5)],
    "VAEEncode:LATENT":                [("KSampler", "latent_image", 30)],
    # ControlNet
    "ControlNetLoader:CONTROL_NET":    [("ControlNetApplyAdvanced", "control_net", 30)],
    "LoraLoader:MODEL":                [("KSampler", "model", 35)],
    "LoraLoader:CLIP":                 [("CLIPTextEncode", "clip", 35)],
    # Image processing
    "LoadImage:IMAGE":                 [("VAEEncode", "pixels", 25), ("PreviewImage", "images", 20), ("ImageScale", "image", 15)],
    "LoadImage:MASK":                  [("VAEEncodeForInpaint", "mask", 15), ("InvertMask", "mask", 10)],
}

# ---------------------------------------------------------------------------
# Type glossary (shipped to frontend)
# ---------------------------------------------------------------------------
TYPE_GLOSSARY: Dict[str, str] = {
    "MODEL":        "A diffusion / UNet model object. Carries the network weights used for denoising; output of checkpoint or UNet loaders, input of KSampler-family nodes.",
    "CLIP":         "A text-encoder (CLIP / T5) object. Feed it into CLIPTextEncode to turn a prompt into conditioning.",
    "VAE":          "A Variational Auto-Encoder. Decodes latents to pixels (VAEDecode) or encodes pixels to latents (VAEEncode).",
    "CONDITIONING": "Encoded prompt embeddings. Connect into KSampler 'positive'/'negative' to steer generation.",
    "LATENT":       "Compressed image tensor in latent space (B,4,H/8,W/8 for SD; 16ch for SDXL/Flux). VAEDecode turns it into pixels.",
    "IMAGE":        "RGB tensor, shape (B,H,W,3), float32 in [0,1]. Standard pixel image format in ComfyUI.",
    "MASK":         "Grayscale mask, shape (B,H,W), float32 in [0,1]. 1 = inside / paint, 0 = outside / keep.",
    "CONTROL_NET":  "A ControlNet model. Apply with ControlNetApply / ControlNetApplyAdvanced to bias generation toward a reference image.",
    "LORA":         "Low-Rank Adaptation weights. Use LoraLoader to combine with a base MODEL+CLIP.",
    "STYLE_MODEL":  "Style transfer model (e.g. T2I-Adapter style). Applied via StyleModelApply.",
    "CLIP_VISION":  "A vision encoder (CLIP-ViT) used by IP-Adapter / Revision / Style models.",
    "SAMPLER":      "Sampler configuration object (used by KSamplerCustom / advanced sampler chains).",
    "SIGMAS":       "Noise schedule tensor consumed by custom sampler nodes.",
    "GUIDER":       "A guidance configuration object for the new modular sampler API.",
    "NOISE":        "Random-noise generator object for sampler chains.",
    "INT":          "Integer scalar (widget). Typical use: seed, steps, dimensions.",
    "FLOAT":        "Floating-point scalar (widget). Typical use: cfg, denoise, strength.",
    "STRING":       "Text string. Typically a prompt, file path, or class name.",
    "BOOLEAN":      "True / False toggle (widget).",
    "COMBO":        "Drop-down chooser with a fixed list of named options.",
    "AUDIO":        "Audio tensor / waveform dict ({waveform, sample_rate}).",
    "VIDEO":        "Video tensor or container; conventions vary by node-pack.",
    "BBOX":         "Bounding-box rectangle(s).",
    "FACE":         "Face-detection / face-region object.",
    "*":            "Wildcard — accepts any type (no constraint).",
}

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
_LOCK = threading.RLock()
_STATE: Dict[str, Any] = {
    "version": 1,
    "edges": {},          # type: Dict[str, int]
    "slot_recent": {},    # type: Dict[str, List[dict]]
    "scanned_files": [],  # type: List[str]
    "updated": 0,
}
_LOADED = False


def _load_history() -> None:
    global _LOADED
    with _LOCK:
        if _LOADED:
            return
        path = _history_path()
        if os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    data = json.load(fh) or {}
                if isinstance(data, dict) and int(data.get("version", 0)) >= 1:
                    _STATE["edges"]         = dict(data.get("edges") or {})
                    _STATE["slot_recent"]   = dict(data.get("slot_recent") or {})
                    _STATE["scanned_files"] = list(data.get("scanned_files") or [])
                    _STATE["updated"]       = int(data.get("updated") or 0)
            except Exception as exc:
                log.warning("history load failed: %s", exc)
        _LOADED = True


def _save_history() -> None:
    with _LOCK:
        path = _history_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        _STATE["updated"] = int(time.time())
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump({
                "version": 1,
                "edges":          _STATE["edges"],
                "slot_recent":    _STATE["slot_recent"],
                "scanned_files":  _STATE["scanned_files"],
                "updated":        _STATE["updated"],
            }, fh, ensure_ascii=False, indent=2)
        os.replace(tmp, path)


def _ekey(s_cls: str, s_slot: str, d_cls: str, d_slot: str) -> str:
    return f"{s_cls}:{s_slot}->{d_cls}:{d_slot}"


def _rkey(cls: str, direction: str, slot: str) -> str:
    return f"{cls}:{direction}:{slot}"


def _bump_edge(s_cls: str, s_slot: str, d_cls: str, d_slot: str, n: int = 1) -> None:
    if not s_cls or not d_cls:
        return
    k = _ekey(s_cls, s_slot or "", d_cls, d_slot or "")
    _STATE["edges"][k] = int(_STATE["edges"].get(k, 0)) + n
    # also record under the source-output slot's recent list
    rk = _rkey(s_cls, "output", s_slot or "")
    recent = _STATE["slot_recent"].setdefault(rk, [])
    for entry in recent:
        if entry.get("cls") == d_cls and entry.get("slot") == d_slot:
            entry["count"] = int(entry.get("count", 0)) + n
            entry["ts"] = int(time.time())
            break
    else:
        recent.insert(0, {"cls": d_cls, "slot": d_slot, "count": n, "ts": int(time.time())})
    # cap to 16 entries, sorted by count desc
    recent.sort(key=lambda e: (-int(e.get("count", 0)), -int(e.get("ts", 0))))
    if len(recent) > 16:
        del recent[16:]


# ---------------------------------------------------------------------------
# Workflow scanning
# ---------------------------------------------------------------------------
def _iter_workflow_edges(doc: Any) -> List[Tuple[str, str, str, str]]:
    """Return list of (src_cls, src_slot_name, dst_cls, dst_slot_name) edges.

    Supports the ComfyUI front-end JSON format: top-level ``nodes`` array of
    objects each with ``id``, ``type``, ``inputs``, ``outputs``; and top-level
    ``links`` array of ``[link_id, src_node_id, src_slot_idx, dst_node_id,
    dst_slot_idx, type_str]`` tuples.
    """
    if not isinstance(doc, dict):
        return []
    nodes = doc.get("nodes") or []
    links = doc.get("links") or []
    if not isinstance(nodes, list) or not isinstance(links, list):
        return []
    by_id: Dict[int, dict] = {}
    for n in nodes:
        if isinstance(n, dict) and "id" in n:
            try:
                by_id[int(n["id"])] = n
            except Exception:
                continue
    out: List[Tuple[str, str, str, str]] = []
    for link in links:
        if not isinstance(link, (list, tuple)) or len(link) < 5:
            continue
        try:
            src_id, src_idx, dst_id, dst_idx = int(link[1]), int(link[2]), int(link[3]), int(link[4])
        except Exception:
            continue
        sn = by_id.get(src_id); dn = by_id.get(dst_id)
        if not sn or not dn:
            continue
        s_cls = str(sn.get("type") or "")
        d_cls = str(dn.get("type") or "")
        if not s_cls or not d_cls:
            continue
        s_outs = sn.get("outputs") or []
        d_ins  = dn.get("inputs") or []
        s_slot = ""
        d_slot = ""
        if 0 <= src_idx < len(s_outs) and isinstance(s_outs[src_idx], dict):
            s_slot = str(s_outs[src_idx].get("name") or s_outs[src_idx].get("type") or "")
        if 0 <= dst_idx < len(d_ins) and isinstance(d_ins[dst_idx], dict):
            d_slot = str(d_ins[dst_idx].get("name") or d_ins[dst_idx].get("type") or "")
        out.append((s_cls, s_slot, d_cls, d_slot))
    return out


def _scan_workflows() -> int:
    """Scan workflows directory and merge edges into state. Returns count of new files scanned."""
    wf_dir = _workflows_dir()
    if not os.path.isdir(wf_dir):
        return 0
    seen = set(_STATE.get("scanned_files") or [])
    new_count = 0
    new_edges = 0
    for root, _dirs, files in os.walk(wf_dir):
        for fn in files:
            if not fn.lower().endswith(".json"):
                continue
            path = os.path.join(root, fn)
            if path in seen:
                continue
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    doc = json.load(fh)
            except Exception:
                continue
            edges = _iter_workflow_edges(doc)
            for (sc, ss, dc, ds) in edges:
                _bump_edge(sc, ss, dc, ds, 1)
                new_edges += 1
            seen.add(path)
            new_count += 1
    _STATE["scanned_files"] = sorted(seen)
    if new_count:
        log.info("autoconnect: scanned %d new workflow file(s), %d edges added", new_count, new_edges)
        _save_history()
    return new_count


# ---------------------------------------------------------------------------
# Suggestion engine
# ---------------------------------------------------------------------------
_EDGE_RE = re.compile(r"^(?P<sc>[^:]+):(?P<ss>[^:]*)->(?P<dc>[^:]+):(?P<ds>[^:]*)$")


def _suggest_from_output(cls: str, slot: str, slot_type: str, limit: int) -> List[Dict[str, Any]]:
    """Suggestions where `(cls, slot)` is an OUTPUT and we want destination inputs."""
    bag: Dict[Tuple[str, str], Dict[str, Any]] = {}

    # 1. learned edges (highest priority — both class+slot match)
    prefix = f"{cls}:{slot}->"
    for k, count in _STATE.get("edges", {}).items():
        if not k.startswith(prefix):
            continue
        m = _EDGE_RE.match(k)
        if not m:
            continue
        dc, ds = m.group("dc"), m.group("ds")
        entry = bag.setdefault((dc, ds), {"cls": dc, "slot": ds, "score": 0.0, "sources": []})
        entry["score"] += float(count) * 3.0  # learned weight
        if "learned" not in entry["sources"]:
            entry["sources"].append("learned")

    # 2. learned edges by TYPE only (looser fallback)
    if slot_type and slot_type != "*":
        for k, count in _STATE.get("edges", {}).items():
            m = _EDGE_RE.match(k)
            if not m:
                continue
            if m.group("sc") != cls or m.group("ss") != slot:
                # also accept edges with same source CLASS+type even if slot name varies
                # (rare; skip for now)
                continue
            entry_key = (m.group("dc"), m.group("ds"))
            if entry_key in bag:
                continue
            entry = bag.setdefault(entry_key, {"cls": entry_key[0], "slot": entry_key[1], "score": 0.0, "sources": []})
            entry["score"] += float(count) * 1.0

    # 3. curated defaults
    curated = CURATED.get(f"{cls}:{slot}") or []
    for (dc, ds, w) in curated:
        entry = bag.setdefault((dc, ds), {"cls": dc, "slot": ds, "score": 0.0, "sources": []})
        entry["score"] += float(w)
        if "curated" not in entry["sources"]:
            entry["sources"].append("curated")

    ranked = sorted(bag.values(), key=lambda e: -e["score"])
    return ranked[:max(1, int(limit))]


def _suggest_from_input(cls: str, slot: str, slot_type: str, limit: int) -> List[Dict[str, Any]]:
    """Suggestions where `(cls, slot)` is an INPUT and we want source outputs."""
    bag: Dict[Tuple[str, str], Dict[str, Any]] = {}
    suffix = f"->{cls}:{slot}"
    for k, count in _STATE.get("edges", {}).items():
        if not k.endswith(suffix):
            continue
        m = _EDGE_RE.match(k)
        if not m:
            continue
        sc, ss = m.group("sc"), m.group("ss")
        entry = bag.setdefault((sc, ss), {"cls": sc, "slot": ss, "score": 0.0, "sources": []})
        entry["score"] += float(count) * 3.0
        if "learned" not in entry["sources"]:
            entry["sources"].append("learned")
    # curated reverse lookup
    for src_key, lst in CURATED.items():
        sc, _, ss_split = src_key.partition(":")
        ss = ss_split
        for (dc, ds, w) in lst:
            if dc == cls and ds == slot:
                entry = bag.setdefault((sc, ss), {"cls": sc, "slot": ss, "score": 0.0, "sources": []})
                entry["score"] += float(w)
                if "curated" not in entry["sources"]:
                    entry["sources"].append("curated")
    ranked = sorted(bag.values(), key=lambda e: -e["score"])
    return ranked[:max(1, int(limit))]


def suggest(cls: str, direction: str, slot: str, slot_type: str = "*", limit: int = 5) -> Dict[str, Any]:
    _load_history()
    direction = (direction or "output").lower()
    if direction == "output":
        ranked = _suggest_from_output(cls or "", slot or "", slot_type or "*", limit)
    else:
        ranked = _suggest_from_input(cls or "", slot or "", slot_type or "*", limit)
    # attach normalized confidence (max-normalized)
    if ranked:
        top = ranked[0]["score"] or 1.0
        for r in ranked:
            r["confidence"] = round(min(1.0, r["score"] / top), 3)
    return {"suggestions": ranked, "cls": cls, "dir": direction, "slot": slot, "type": slot_type}


def record_edges(edges: List[List[str]]) -> int:
    """Record one or more (s_cls, s_slot, d_cls, d_slot) live edges."""
    _load_history()
    n = 0
    with _LOCK:
        for e in edges:
            if not isinstance(e, (list, tuple)) or len(e) < 4:
                continue
            sc, ss, dc, ds = str(e[0]), str(e[1]), str(e[2]), str(e[3])
            if not sc or not dc:
                continue
            _bump_edge(sc, ss, dc, ds, 1)
            n += 1
        if n:
            _save_history()
    return n


# ---------------------------------------------------------------------------
# /object_info compatibility cache (computed on demand, then cached)
# ---------------------------------------------------------------------------
_COMPAT_CACHE: Dict[str, List[str]] = {}
_COMPAT_LOCK = threading.Lock()


def _build_compat_index() -> None:
    """Walk NODE_CLASS_MAPPINGS once to build {type -> [classes that accept it as input], ...}."""
    if _COMPAT_CACHE:
        return
    with _COMPAT_LOCK:
        if _COMPAT_CACHE:
            return
        try:
            import nodes as _comfy_nodes  # type: ignore
            ncm = getattr(_comfy_nodes, "NODE_CLASS_MAPPINGS", {}) or {}
        except Exception:
            ncm = {}
        in_idx: Dict[str, set] = defaultdict(set)
        out_idx: Dict[str, set] = defaultdict(set)
        for cls_name, cls_obj in ncm.items():
            try:
                it_fn = getattr(cls_obj, "INPUT_TYPES", None)
                it = it_fn() if callable(it_fn) else (it_fn or {})
                for section in ("required", "optional"):
                    section_d = (it or {}).get(section) or {}
                    for slot_name, spec in section_d.items():
                        if isinstance(spec, (list, tuple)) and spec:
                            tp = spec[0]
                            if isinstance(tp, (list, tuple)):
                                tp = "COMBO"
                            in_idx[str(tp)].add(cls_name)
                rt = getattr(cls_obj, "RETURN_TYPES", ()) or ()
                for tp in rt:
                    out_idx[str(tp)].add(cls_name)
            except Exception:
                continue
        for tp, s in in_idx.items():
            _COMPAT_CACHE[f"in:{tp}"] = sorted(s)
        for tp, s in out_idx.items():
            _COMPAT_CACHE[f"out:{tp}"] = sorted(s)


def compatible(slot_type: str, direction: str) -> List[str]:
    _build_compat_index()
    key = ("in" if direction.lower() == "input" else "out") + ":" + (slot_type or "")
    return list(_COMPAT_CACHE.get(key, []))


# ---------------------------------------------------------------------------
# HTTP routes
# ---------------------------------------------------------------------------
_ROUTES_REGISTERED = False


def register_routes(server) -> None:
    global _ROUTES_REGISTERED
    if _ROUTES_REGISTERED:
        return
    try:
        from aiohttp import web
    except Exception as exc:
        log.warning("aiohttp not importable, skipping autoconnect routes: %s", exc)
        return

    _load_history()
    try:
        _scan_workflows()
    except Exception as exc:
        log.warning("workflow scan failed: %s", exc)

    routes = server.routes

    @routes.get("/c2c/autoconnect/suggest")
    async def _suggest(request):
        q = request.query
        cls = (q.get("cls") or "").strip()
        direction = (q.get("dir") or "output").strip().lower()
        slot = (q.get("slot") or "").strip()
        slot_type = (q.get("type") or "*").strip()
        try:
            limit = int(q.get("limit") or "5")
        except Exception:
            limit = 5
        try:
            return web.json_response(suggest(cls, direction, slot, slot_type, limit))
        except Exception as exc:
            return web.json_response({"error": str(exc)}, status=400)

    @routes.post("/c2c/autoconnect/record")
    async def _record(request):
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "invalid json"}, status=400)
        edges = body.get("edges") if isinstance(body, dict) else None
        if not isinstance(edges, list):
            return web.json_response({"error": "edges must be a list"}, status=400)
        n = record_edges(edges)
        return web.json_response({"recorded": n})

    @routes.get("/c2c/autoconnect/type_glossary")
    async def _glossary(_request):
        return web.json_response({"glossary": TYPE_GLOSSARY})

    @routes.get("/c2c/autoconnect/compatible")
    async def _compat(request):
        slot_type = (request.query.get("type") or "").strip()
        direction = (request.query.get("dir") or "input").strip().lower()
        return web.json_response({"type": slot_type, "dir": direction, "classes": compatible(slot_type, direction)})

    @routes.get("/c2c/autoconnect/stats")
    async def _stats(_request):
        _load_history()
        return web.json_response({
            "version": 1,
            "edge_count": len(_STATE.get("edges") or {}),
            "scanned_files": len(_STATE.get("scanned_files") or []),
            "curated_keys": len(CURATED),
            "glossary_entries": len(TYPE_GLOSSARY),
            "updated": _STATE.get("updated", 0),
            "history_path": _history_path(),
            "workflows_dir": _workflows_dir(),
        })

    _ROUTES_REGISTERED = True
    log.info("c2c.autoconnect routes registered (/c2c/autoconnect/*) edges=%d scanned=%d",
             len(_STATE.get("edges") or {}), len(_STATE.get("scanned_files") or []))
