"""ai_workflow_builder.py — natural-language → runnable ComfyUI graph.

W5/Section-7 AI-Spine capability: POST /c2c/ai/build_workflow with a plain
English request ("build me a wan2.1 image-to-video workflow…") and get back a
loadable LiteGraph JSON. Architecture adapted from artokun/comfyui-mcp's
"compact tool mode" (never hand a small local model the full 2000-node
schema):

  stage 1  keyword-score the LIVE NODE_CLASS_MAPPINGS registry down to ~40
           candidate nodes (our MEC/C2C/NukeMax/WanDirector namespaces are
           boosted so in-house nodes win over stock equivalents),
  stage 2  prompt the LLM with compact per-node schemas (widgets + IO only)
           and ask for a small JSON plan,
  stage 3  mechanically validate the plan against the registry (unknown node
           types, unknown widgets, broken/type-mismatched links) and retry
           ONCE with the validation errors fed back,
  stage 4  convert the plan to LiteGraph JSON (widget order derived from
           INPUT_TYPES, control_after_generate inserted after seed widgets,
           topological left→right layout).

The LLM call reuses the error-assistant's tier system (local GGUF / Ollama /
cloud, user-configured in C2C AI settings). With no backend available the
route still works for the classic text-to-image request via a deterministic
template, and always reports which path produced the graph.

Author: Code2Collapse. Apache-2.0.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger("MEC.ai_workflow_builder")

# Always considered, whatever the request: the spine of most graphs.
_CORE_NODES = [
    "CheckpointLoaderSimple", "CLIPTextEncode", "KSampler", "VAEDecode",
    "VAEEncode", "EmptyLatentImage", "SaveImage", "LoadImage", "PreviewImage",
]
# Namespace boost: in-house packs preferred over stock when relevant.
_HOUSE_PREFIXES = ("WanDirector", "WNE_", "NukeMax_", "C2C", "MEC")

_WIDGET_TYPES = {"INT", "FLOAT", "STRING", "BOOLEAN"}


def _registry() -> Dict[str, Any]:
    import sys
    import nodes as comfy_nodes  # ComfyUI's live registry
    reg = getattr(comfy_nodes, "NODE_CLASS_MAPPINGS", None)
    if reg is None:
        # A sibling "nodes" package can shadow ComfyUI's nodes.py when the
        # import order is unusual (tests, tools). Find the real one.
        for mod in sys.modules.values():
            cand = getattr(mod, "NODE_CLASS_MAPPINGS", None)
            if isinstance(cand, dict) and "KSampler" in cand:
                return cand
        raise RuntimeError("ComfyUI node registry not found")
    return reg


def _tokens(text: str) -> List[str]:
    return [t for t in re.split(r"[^a-z0-9]+", text.lower()) if len(t) > 2]


def _node_score(name: str, cls: Any, toks: List[str]) -> float:
    hay = (name + " " + str(getattr(cls, "CATEGORY", "")) + " "
           + str(getattr(cls, "DESCRIPTION", ""))).lower()
    score = sum(2.0 if t in name.lower() else 1.0 for t in toks if t in hay)
    if score and name.startswith(_HOUSE_PREFIXES):
        score += 1.5
    return score


def pick_candidates(request: str, k: int = 40) -> List[str]:
    reg = _registry()
    toks = _tokens(request)
    scored = []
    for name, cls in reg.items():
        s = _node_score(name, cls, toks)
        if s > 0:
            scored.append((s, name))
    scored.sort(reverse=True)
    out = [n for _, n in scored[:k]]
    for c in _CORE_NODES:
        if c in reg and c not in out:
            out.append(c)
    return out


def _widget_entries(node_type: str) -> List[Tuple[str, Any, Any]]:
    """Ordered (name, type_spec, default) for entries that render as widgets."""
    reg = _registry()
    cls = reg[node_type]
    it = cls.INPUT_TYPES()
    entries: List[Tuple[str, Any, Any]] = []
    for section in ("required", "optional"):
        for name, spec in (it.get(section) or {}).items():
            t = spec[0] if isinstance(spec, (tuple, list)) and spec else spec
            opts = spec[1] if isinstance(spec, (tuple, list)) and len(spec) > 1 and isinstance(spec[1], dict) else {}
            if isinstance(t, list):                       # combo
                entries.append((name, t, opts.get("default", t[0] if t else "")))
            elif isinstance(t, str) and t in _WIDGET_TYPES and not opts.get("forceInput"):
                dflt = opts.get("default", 0 if t == "INT" else 0.0 if t == "FLOAT" else "" if t == "STRING" else False)
                entries.append((name, t, dflt))
    return entries


def _conn_inputs(node_type: str) -> List[Tuple[str, str]]:
    """Ordered (name, type) for connection inputs (non-widget or forceInput)."""
    reg = _registry()
    it = reg[node_type].INPUT_TYPES()
    out: List[Tuple[str, str]] = []
    for section in ("required", "optional"):
        for name, spec in (it.get(section) or {}).items():
            t = spec[0] if isinstance(spec, (tuple, list)) and spec else spec
            opts = spec[1] if isinstance(spec, (tuple, list)) and len(spec) > 1 and isinstance(spec[1], dict) else {}
            if isinstance(t, str) and (t not in _WIDGET_TYPES or opts.get("forceInput")):
                out.append((name, t))
    return out


def _outputs(node_type: str) -> List[Tuple[str, str]]:
    reg = _registry()
    cls = reg[node_type]
    types = list(getattr(cls, "RETURN_TYPES", ()) or ())
    names = list(getattr(cls, "RETURN_NAMES", ()) or ())
    return [(names[i] if i < len(names) else str(t), str(t)) for i, t in enumerate(types)]


def compact_schema(node_type: str) -> str:
    """One node as a few prompt lines — the 'describe_tool' half of compact mode."""
    widgets = _widget_entries(node_type)
    conns = _conn_inputs(node_type)
    outs = _outputs(node_type)
    wtxt = ", ".join(
        f"{n}={json.dumps(d)}" + (f" (one of {t[:8]})" if isinstance(t, list) else f":{t}")
        for n, t, d in widgets) or "none"
    ctxt = ", ".join(f"{n}:{t}" for n, t in conns) or "none"
    otxt = ", ".join(f"[{i}] {n}:{t}" for i, (n, t) in enumerate(outs)) or "none"
    return f"- {node_type}\n    inputs: {ctxt}\n    widgets: {wtxt}\n    outputs: {otxt}"


# ── Plan validation (artokun-style: catch it before it runs) ─────────


def validate_plan(plan: Dict[str, Any]) -> List[str]:
    errs: List[str] = []
    reg = _registry()
    nodes = plan.get("nodes")
    links = plan.get("links") or []
    if not isinstance(nodes, list) or not nodes:
        return ["plan.nodes must be a non-empty list"]
    ids = {}
    for nd in nodes:
        nid, ntype = nd.get("id"), nd.get("type")
        if ntype not in reg:
            errs.append(f"unknown node type: {ntype!r}")
            continue
        if nid in ids:
            errs.append(f"duplicate node id {nid}")
        ids[nid] = ntype
        wnames = {w[0] for w in _widget_entries(ntype)}
        for wname in (nd.get("widgets") or {}):
            if wname not in wnames:
                errs.append(f"node {nid} ({ntype}): unknown widget {wname!r}")
    for ln in links:
        try:
            frm, out_idx, to, in_name = ln[0], int(ln[1]), ln[2], str(ln[3])
        except Exception:
            errs.append(f"malformed link {ln!r} (want [from_id, out_index, to_id, input_name])")
            continue
        if frm not in ids or to not in ids:
            errs.append(f"link {ln!r}: unknown node id")
            continue
        outs = _outputs(ids[frm])
        if out_idx < 0 or out_idx >= len(outs):
            errs.append(f"link {ln!r}: {ids[frm]} has no output index {out_idx}")
            continue
        cins = dict(_conn_inputs(ids[to]))
        if in_name not in cins:
            errs.append(f"link {ln!r}: {ids[to]} has no connection input {in_name!r}")
            continue
        ot, itp = outs[out_idx][1], cins[in_name]
        if ot != itp and "*" not in (ot, itp):
            errs.append(f"link {ln!r}: type mismatch {ot} → {itp}")
    return errs


# ── Plan → LiteGraph JSON ────────────────────────────────────────────


def plan_to_graph(plan: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a validated plan into JSON app.loadGraphData() accepts."""
    reg = _registry()
    nodes_in = plan["nodes"]
    links_in = plan.get("links") or []

    # Topological depth for a simple left→right layout.
    depth = {nd["id"]: 0 for nd in nodes_in}
    for _ in range(len(nodes_in)):
        changed = False
        for ln in links_in:
            d = depth[ln[0]] + 1
            if d > depth[ln[2]]:
                depth[ln[2]] = d
                changed = True
        if not changed:
            break
    lanes: Dict[int, int] = {}

    id_map = {nd["id"]: i + 1 for i, nd in enumerate(nodes_in)}   # numeric ids
    out_nodes = []
    for nd in nodes_in:
        ntype = nd["type"]
        gid = id_map[nd["id"]]
        d = depth[nd["id"]]
        lane = lanes.get(d, 0)
        lanes[d] = lane + 1

        widgets_values: List[Any] = []
        for wname, wtype, dflt in _widget_entries(ntype):
            val = (nd.get("widgets") or {}).get(wname, dflt)
            widgets_values.append(val)
            # The frontend appends a control_after_generate widget after seed
            # inputs; its value must be present in widgets_values or every
            # later widget shifts by one on load.
            if wname in ("seed", "noise_seed") and str(wtype) == "INT":
                widgets_values.append("fixed")

        inputs = []
        for name, t in _conn_inputs(ntype):
            inputs.append({"name": name, "type": t, "link": None})
        outputs = []
        for i, (name, t) in enumerate(_outputs(ntype)):
            outputs.append({"name": name, "type": t, "links": [], "slot_index": i})

        out_nodes.append({
            "id": gid, "type": ntype,
            "pos": [80 + d * 420, 80 + lane * 320],
            "size": [315, 120], "flags": {}, "order": gid - 1, "mode": 0,
            "inputs": inputs, "outputs": outputs,
            "properties": {"Node name for S&R": ntype},
            "widgets_values": widgets_values,
        })

    out_links = []
    by_gid = {n["id"]: n for n in out_nodes}
    for li, ln in enumerate(links_in, start=1):
        frm, out_idx, to, in_name = id_map[ln[0]], int(ln[1]), id_map[ln[2]], str(ln[3])
        tgt = by_gid[to]
        in_idx = next(i for i, inp in enumerate(tgt["inputs"]) if inp["name"] == in_name)
        ltype = by_gid[frm]["outputs"][out_idx]["type"]
        out_links.append([li, frm, out_idx, to, in_idx, ltype])
        by_gid[frm]["outputs"][out_idx]["links"].append(li)
        tgt["inputs"][in_idx]["link"] = li

    return {
        "last_node_id": len(out_nodes), "last_link_id": len(out_links),
        "nodes": out_nodes, "links": out_links,
        "groups": [], "config": {}, "extra": {}, "version": 0.4,
    }


# ── LLM plumbing (reuses the error-assistant tier system) ────────────


_SYSTEM = """You design ComfyUI node graphs. Reply with ONE JSON object only:
{"nodes":[{"id":"a","type":"NodeTypeName","widgets":{"widget_name":value}},...],
 "links":[["a",0,"b","input_name"],...]}
Rules: use ONLY node types from the catalog below; connect every input listed
under "inputs" for a node you use; widgets you omit keep defaults; a link is
[from_id, output_index, to_id, input_name] — input_name must appear in the
TARGET node's "inputs" list, and the source output type must equal the input
type. Data flows source→consumer: e.g. EmptyLatentImage output [0] LATENT
feeds KSampler's latent_image input as ["lat",0,"ks","latent_image"].
Mini example (3 nodes):
{"nodes":[{"id":"ld","type":"LoadImage","widgets":{}},
 {"id":"enc","type":"VAEEncode","widgets":{}}],
 "links":[["ld",0,"enc","pixels"]]}
No prose, no markdown fences."""


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Last valid JSON object in the text (reasoning models think out loud)."""
    best = None
    for m in re.finditer(r"\{", text or ""):
        s = text[m.start():]
        try:
            dec = json.JSONDecoder()
            obj, _ = dec.raw_decode(s)
            if isinstance(obj, dict) and "nodes" in obj:
                best = obj
        except Exception:
            continue
    return best


def _complete(prompt: str, system: Optional[str] = None) -> Tuple[Optional[str], str]:
    """Try tier2 (local/ollama) then tier3 (cloud). Returns (text, provider).

    `system` replaces the local backend's default CAUSE:/FIXES: persona —
    required whenever the caller wants a different output contract (JSON),
    or the two format instructions fight and the model wastes its budget.
    """
    try:
        from . import error_assistant as ea
        from . import local_llm
    except Exception:
        import importlib
        ea = importlib.import_module("nodes.error_assistant", package=__package__)
        local_llm = importlib.import_module("nodes.local_llm", package=__package__)
    settings = ea.load_settings()
    settings = dict(settings)
    settings["max_tokens"] = 1600
    if system is not None:
        settings["system_prompt"] = system
    if settings.get("tier2_enabled"):
        txt = ea._explain_local(prompt, settings)
        if txt:
            return local_llm._strip_reasoning(txt), "local"
    if settings.get("tier3_enabled"):
        txt = ea._explain_cloud(prompt, settings)
        if txt:
            return local_llm._strip_reasoning(txt), "cloud"
    return None, "none"


# Deterministic fallback so the feature works with zero AI backends.
_TXT2IMG_PLAN = {
    "nodes": [
        {"id": "ckpt", "type": "CheckpointLoaderSimple", "widgets": {}},
        {"id": "pos", "type": "CLIPTextEncode", "widgets": {"text": "a beautiful scene"}},
        {"id": "neg", "type": "CLIPTextEncode", "widgets": {"text": "blurry, low quality"}},
        {"id": "lat", "type": "EmptyLatentImage", "widgets": {"width": 1024, "height": 1024}},
        {"id": "ks", "type": "KSampler", "widgets": {"steps": 20, "cfg": 7.0}},
        {"id": "dec", "type": "VAEDecode", "widgets": {}},
        {"id": "save", "type": "SaveImage", "widgets": {}},
    ],
    "links": [
        ["ckpt", 0, "ks", "model"], ["ckpt", 1, "pos", "clip"], ["ckpt", 1, "neg", "clip"],
        ["pos", 0, "ks", "positive"], ["neg", 0, "ks", "negative"], ["lat", 0, "ks", "latent_image"],
        ["ks", 0, "dec", "samples"], ["ckpt", 2, "dec", "vae"], ["dec", 0, "save", "images"],
    ],
}


def build_workflow(request: str, plan_override: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    attempts: List[Dict[str, Any]] = []
    provider = "override"
    plan = plan_override
    if plan is None:
        cands = pick_candidates(request)
        catalog = "\n".join(compact_schema(n) for n in cands)
        base_prompt = f"Request: {request}\n\nNode catalog:\n{catalog}\n\nJSON:"
        prompt = base_prompt
        for attempt in range(2):
            text, provider = _complete(prompt, system=_SYSTEM)
            if not text:
                break
            cand = _extract_json(text)
            errs = validate_plan(cand) if cand else ["no JSON object with a 'nodes' list in reply"]
            attempts.append({"provider": provider, "errors": errs[:6]})
            if cand and not errs:
                plan = cand
                break
            prompt = (base_prompt + "\nYour previous answer had these problems, fix them:\n- "
                      + "\n- ".join(errs[:6]) + "\nJSON:")
        if plan is None and re.search(r"t2i|text.?to.?image|txt2img|generate.*image|image from",
                                      request, re.I):
            plan, provider = dict(_TXT2IMG_PLAN), "deterministic-template"
    if plan is None:
        return {"ok": False, "error": "no AI backend produced a valid plan "
                "(enable Tier-2/Tier-3 in C2C AI settings)", "attempts": attempts}
    errs = validate_plan(plan)
    if errs:
        return {"ok": False, "error": "plan failed validation", "details": errs[:10],
                "attempts": attempts}
    graph = plan_to_graph(plan)
    return {"ok": True, "graph": graph, "plan": plan, "provider": provider,
            "attempts": attempts, "node_count": len(graph["nodes"])}


def register_routes(server: Any) -> None:
    from aiohttp import web

    @server.routes.post("/c2c/ai/build_workflow")
    async def _build(request):  # noqa: ANN001
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "invalid JSON body"}, status=400)
        req_text = str(body.get("request") or "").strip()
        override = body.get("plan_override")
        if not req_text and not override:
            return web.json_response({"ok": False, "error": "missing 'request'"}, status=400)
        import asyncio
        loop = asyncio.get_event_loop()
        try:
            result = await loop.run_in_executor(None, build_workflow, req_text, override)
            return web.json_response(result, status=200 if result.get("ok") else 422)
        except Exception as exc:  # noqa: BLE001
            log.warning("[ai_workflow_builder] failed: %s", exc)
            return web.json_response({"ok": False, "error": str(exc)[:300]}, status=500)

    log.info("[MEC] ai_workflow_builder route registered: POST /c2c/ai/build_workflow")
