"""
workflow_doctor.py — v2.0 P2: static analysis of a ComfyUI workflow graph.

Runs a battery of deterministic checks against a serialized graph (the same
shape `app.graph.serialize()` produces in the frontend) and returns a list of
findings ranked by severity. The JS layer can then offer a one-click "Ask AI
to dig deeper" handoff to `/c2c/ai/stream` with feature="workflow_doctor".

Wire-format
-----------
POST /c2c/doctor/analyze
Body:
    {
        "workflow": { "nodes": [...], "links": [...] }   # LiteGraph serialize
    }
Reply:
    {
        "success": true,
        "data": {
            "summary": {"errors": int, "warnings": int, "infos": int},
            "findings": [
                {
                    "id": "missing_input",
                    "severity": "error" | "warning" | "info",
                    "node_id": 12,
                    "node_type": "KSampler",
                    "title": "Required input 'model' is not connected",
                    "detail": "...",
                    "fix_hint": "..."
                },
                ...
            ],
            "stats": {"nodes": 17, "links": 14, "checkpoints": 1, ...}
        }
    }

GET /c2c/doctor/rules
    Reply: list of {id, title, severity_default, description}.

The checks here are entirely offline / no-AI. They are the same kind of
checks a senior ComfyUI user does by eye: input wiring, sampler sanity,
model-family mismatches, CFG/steps/denoise extremes, orphans, cycles.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional, Tuple

log = logging.getLogger("MEC.workflow_doctor")

# -------------------------------------------------------------- rule catalog
RULES: List[Dict[str, str]] = [
    {"id": "missing_input",        "severity": "error",
     "title": "Required input is not connected",
     "description": "A node has a non-optional input slot with no incoming link."},
    {"id": "empty_text",           "severity": "warning",
     "title": "CLIPTextEncode has empty text",
     "description": "An empty prompt usually produces poor results — even an empty negative should be intentional."},
    {"id": "disabled_node",        "severity": "info",
     "title": "Node is muted/bypassed",
     "description": "LiteGraph mode == 4 means the node is muted; downstream nodes may receive null."},
    {"id": "duplicate_seed",       "severity": "warning",
     "title": "Multiple KSamplers share the same seed",
     "description": "Two or more samplers using the same fixed seed will produce identical noise — usually unintended."},
    {"id": "cfg_extreme",          "severity": "warning",
     "title": "KSampler cfg is outside the usable range",
     "description": "cfg < 1 or > 20 typically yields washed-out or burned outputs."},
    {"id": "steps_extreme",        "severity": "warning",
     "title": "KSampler steps is outside the usable range",
     "description": "steps < 4 (under-sampled) or > 150 (wasted compute) for typical samplers."},
    {"id": "denoise_zero",         "severity": "warning",
     "title": "KSampler denoise = 0",
     "description": "Zero denoise means the sampler passes the latent through unchanged."},
    {"id": "lora_stack_deep",      "severity": "info",
     "title": "Deep LoRA stack",
     "description": "More than 5 LoRA loaders chained — quality may degrade and load time grows."},
    {"id": "orphan_output",        "severity": "info",
     "title": "Output node has no upstream sampler",
     "description": "A SaveImage / PreviewImage is connected but no sampler feeds into it — output will be empty or pass-through."},
    {"id": "unconnected_terminal", "severity": "info",
     "title": "Sampler/Decoder has no consumer",
     "description": "A KSampler / VAEDecode produces output that nothing uses — wasted compute."},
    {"id": "checkpoint_mismatch",  "severity": "warning",
     "title": "Multiple checkpoint families in one graph",
     "description": "Mixing SDXL and SD1.5 components (CLIP / VAE / model) generally fails or produces noise."},
    {"id": "cycle",                "severity": "error",
     "title": "Graph contains a cycle",
     "description": "LiteGraph graphs must be acyclic; execution will fail or hang."},
    {"id": "empty_latent_size",    "severity": "warning",
     "title": "EmptyLatentImage dimensions are unusual",
     "description": "Width/height not a multiple of 8, or below 64, or above 4096."},
    {"id": "no_sampler",           "severity": "info",
     "title": "No sampler in graph",
     "description": "Workflow contains no KSampler/SamplerCustom — this may be intentional (pure I/O) but unusual."},
    # ---- v2 P8: mask-hygiene rules
    {"id": "temporal_gaussian_long_batch", "severity": "warning",
     "title": "Gaussian temporal smoothing on long batch",
     "description": "MaskTemporalMEC with temporal_mode=gaussian on >12-frame batch causes mask drag/smearing on fast motion."},
    {"id": "sam3_no_neg_boxes", "severity": "info",
     "title": "SAM 3.1 used without negative bbox prompts",
     "description": "SAM 3.x benefits from negative bounding boxes to exclude false positives — consider supplying input_boxes_labels."},
    {"id": "video_inpaint_no_roto", "severity": "warning",
     "title": "Video inpaint without roto_quality",
     "description": "InpaintCropProMEC on multi-frame input should set roto_quality=True for crisp edges and laplacian-pyramid blending."},
    {"id": "vitmatte_manual_trimap", "severity": "info",
     "title": "ViTMatte with manual trimap mode",
     "description": "ViTMatte typically produces best results with trimap_mode=auto; manual trimaps may under/over-segment hair and fine edges."},
    {"id": "multiframe_integrity_disabled", "severity": "error",
     "title": "Mask integrity check disabled on multi-frame workflow",
     "description": "MaskRefineMEC on a multi-frame batch with enable_integrity_check=False loses silent-failure detection. Enable it or add MaskTemporalMEC."},
]

_RULES_BY_ID = {r["id"]: r for r in RULES}


def _finding(rule_id: str, *, node_id: Optional[int] = None,
             node_type: Optional[str] = None, detail: str = "",
             fix_hint: str = "", severity_override: Optional[str] = None,
             title_override: Optional[str] = None) -> Dict[str, Any]:
    rule = _RULES_BY_ID[rule_id]
    return {
        "id": rule_id,
        "severity": severity_override or rule["severity"],
        "title": title_override or rule["title"],
        "node_id": node_id,
        "node_type": node_type,
        "detail": detail,
        "fix_hint": fix_hint,
    }


# ------------------------------------------------------------ graph helpers
def _normalize_nodes(wf: Dict[str, Any]) -> List[Dict[str, Any]]:
    nodes = wf.get("nodes")
    if isinstance(nodes, list):
        return [n for n in nodes if isinstance(n, dict)]
    return []


def _normalize_links(wf: Dict[str, Any]) -> List[Tuple[int, int, int, int, int, str]]:
    """LiteGraph links are arrays: [link_id, src_node, src_slot, dst_node, dst_slot, type]."""
    out: List[Tuple[int, int, int, int, int, str]] = []
    for ln in wf.get("links", []) or []:
        if isinstance(ln, list) and len(ln) >= 5:
            try:
                out.append((int(ln[0]), int(ln[1]), int(ln[2]),
                            int(ln[3]), int(ln[4]),
                            str(ln[5]) if len(ln) > 5 else ""))
            except (TypeError, ValueError):
                continue
    return out


_WIDGET_LAYOUT_CACHE: Dict[str, List[str]] = {}


def _widget_layout_for(node_type: str) -> List[str]:
    """Return the positional widget-name layout for a node class, as it would
    appear in LiteGraph's `widgets_values` array.

    A widget is created for every primitive input (INT/FLOAT/STRING/BOOLEAN/COMBO).
    Slot-type inputs (IMAGE, MASK, LATENT, MODEL, CLIP, VAE, CONDITIONING, ...)
    do NOT consume a widgets_values slot. Some widgets (seed/INT with control,
    text/STRING multiline) emit an extra positional value — we handle the most
    common: seed → followed by `control_after_generate` string.
    """
    if node_type in _WIDGET_LAYOUT_CACHE:
        return _WIDGET_LAYOUT_CACHE[node_type]
    layout: List[str] = []
    try:
        from nodes import NODE_CLASS_MAPPINGS  # type: ignore
        cls = NODE_CLASS_MAPPINGS.get(node_type)
        if cls is not None:
            it = cls.INPUT_TYPES() if hasattr(cls, "INPUT_TYPES") else {}
            for section in ("required", "optional"):
                for name, spec in (it.get(section) or {}).items():
                    if not isinstance(spec, (list, tuple)) or not spec:
                        continue
                    t = spec[0]
                    # COMBO widget: first element is a list of choices
                    if isinstance(t, (list, tuple)):
                        layout.append(name)
                        continue
                    if t in ("INT", "FLOAT", "STRING", "BOOLEAN"):
                        layout.append(name)
                        # seed widgets emit a hidden control_after_generate slot
                        if t == "INT" and name in ("seed", "noise_seed"):
                            layout.append("control_after_generate")
    except Exception:  # noqa: BLE001
        pass
    _WIDGET_LAYOUT_CACHE[node_type] = layout
    return layout


def _widget_value(node: Dict[str, Any], name_hints: Iterable[str]) -> Any:
    """Look up a widget value by name. Robust against the three shapes we
    actually see in the wild:

      1. LiteGraph serialize(): `widgets_values` is a positional list; widget
         names are NOT included. We recover them from NODE_CLASS_MAPPINGS.
      2. API JSON (graphToPrompt): `inputs` is a dict of {name: value-or-link}.
      3. Legacy/test shape: node carries `widgets` metadata with names.
    """
    hints = list(name_hints)
    # (1) legacy `widgets`+`widgets_values` paired arrays
    widgets = node.get("widgets") or []
    values = node.get("widgets_values") or []
    if widgets and values and len(widgets) == len(values):
        for w, v in zip(widgets, values):
            if (w or {}).get("name", "") in hints:
                return v
    # (2) API-JSON `inputs` dict
    inputs = node.get("inputs")
    if isinstance(inputs, dict):
        for hint in hints:
            if hint in inputs:
                val = inputs[hint]
                # API-JSON puts links as [node_id, slot] lists — those aren't widget values
                if not isinstance(val, list):
                    return val
    # (3) positional via INPUT_TYPES layout
    if values:
        ntype = node.get("type") or ""
        layout = _widget_layout_for(ntype)
        for hint in hints:
            if hint in layout:
                idx = layout.index(hint)
                if idx < len(values):
                    return values[idx]
    return None


def _has_incoming_link(links: List[Tuple[int, int, int, int, int, str]],
                       dst_node: int, dst_slot: int) -> bool:
    for _, _, _, dn, ds, _ in links:
        if dn == dst_node and ds == dst_slot:
            return True
    return False


def _has_outgoing_link(links: List[Tuple[int, int, int, int, int, str]],
                       src_node: int) -> bool:
    for _, sn, _, _, _, _ in links:
        if sn == src_node:
            return True
    return False


def _detect_cycle(nodes: List[Dict[str, Any]],
                  links: List[Tuple[int, int, int, int, int, str]]) -> bool:
    adj: Dict[int, List[int]] = {}
    for n in nodes:
        adj.setdefault(int(n.get("id", -1)), [])
    for _, sn, _, dn, _, _ in links:
        adj.setdefault(sn, []).append(dn)
    WHITE, GRAY, BLACK = 0, 1, 2
    color: Dict[int, int] = {k: WHITE for k in adj}

    def dfs(u: int) -> bool:
        color[u] = GRAY
        for v in adj.get(u, ()):
            c = color.get(v, WHITE)
            if c == GRAY:
                return True
            if c == WHITE and dfs(v):
                return True
        color[u] = BLACK
        return False

    for k in list(adj.keys()):
        if color[k] == WHITE and dfs(k):
            return True
    return False


# --------------------------------------------------------- analyzer entry
_SAMPLER_TYPES = ("KSampler", "KSamplerAdvanced", "SamplerCustom",
                  "SamplerCustomAdvanced")
_OUTPUT_TYPES = ("SaveImage", "PreviewImage", "VHS_VideoCombine",
                 "SaveAnimatedWEBP", "SaveAnimatedPNG")
_CHECKPOINT_TYPES = ("CheckpointLoaderSimple", "CheckpointLoader",
                     "UNETLoader")


def analyze(workflow: Dict[str, Any]) -> Dict[str, Any]:
    """Run all static checks. Returns the response payload (see module docstring)."""
    nodes = _normalize_nodes(workflow)
    links = _normalize_links(workflow)
    findings: List[Dict[str, Any]] = []

    # ---------- stats
    stats: Dict[str, Any] = {
        "nodes": len(nodes), "links": len(links),
        "checkpoints": 0, "samplers": 0, "loras": 0,
        "controlnets": 0, "outputs": 0,
    }

    # ---------- cycle
    if _detect_cycle(nodes, links):
        findings.append(_finding(
            "cycle",
            detail="The graph contains at least one cycle; ComfyUI cannot execute cyclic graphs.",
            fix_hint="Find the loop by toggling links and break it. Use Auto-Checkpoint to revert if needed."))

    # ---------- per-node checks
    sampler_seeds: Dict[int, List[int]] = {}
    checkpoint_families: List[str] = []
    has_sampler = False

    for n in nodes:
        nid = int(n.get("id", -1))
        ntype = str(n.get("type", "") or n.get("class_type", ""))
        if not ntype:
            continue

        # disabled
        if n.get("mode") == 4:
            findings.append(_finding(
                "disabled_node", node_id=nid, node_type=ntype,
                detail=f"Node {ntype} (id {nid}) is muted (mode=4).",
                fix_hint="Right-click the node and unmute if you want it to run."))

        # missing required inputs
        for slot_idx, inp in enumerate(n.get("inputs") or []):
            if not isinstance(inp, dict):
                continue
            if inp.get("link") is not None:
                continue
            # Heuristic: an input is "required" if it has no `widget` field
            # (widget-only inputs default to widget value, no link needed).
            if inp.get("widget") is not None:
                continue
            iname = inp.get("name", f"slot{slot_idx}")
            findings.append(_finding(
                "missing_input", node_id=nid, node_type=ntype,
                detail=f"{ntype}.{iname} has no incoming link.",
                fix_hint=f"Connect a {inp.get('type', '?')} output into '{iname}'.",
                title_override=f"Required input '{iname}' is not connected"))

        # CLIPTextEncode empty
        if ntype.startswith("CLIPTextEncode"):
            text = _widget_value(n, ("text",))
            if isinstance(text, list) and text:
                text = text[0]
            if not (isinstance(text, str) and text.strip()):
                findings.append(_finding(
                    "empty_text", node_id=nid, node_type=ntype,
                    detail="The prompt text widget is empty.",
                    fix_hint="Either enter a prompt or mute the node."))

        # KSampler sanity
        if ntype in _SAMPLER_TYPES:
            stats["samplers"] += 1
            has_sampler = True
            cfg = _widget_value(n, ("cfg",))
            steps = _widget_value(n, ("steps",))
            denoise = _widget_value(n, ("denoise",))
            seed = _widget_value(n, ("seed", "noise_seed"))
            if isinstance(cfg, (int, float)) and (cfg < 1 or cfg > 20):
                findings.append(_finding(
                    "cfg_extreme", node_id=nid, node_type=ntype,
                    detail=f"cfg = {cfg}",
                    fix_hint="Try cfg between 4 and 9 for most samplers."))
            if isinstance(steps, int) and (steps < 4 or steps > 150):
                findings.append(_finding(
                    "steps_extreme", node_id=nid, node_type=ntype,
                    detail=f"steps = {steps}",
                    fix_hint="Try 20–40 steps for most samplers."))
            if isinstance(denoise, (int, float)) and denoise == 0:
                findings.append(_finding(
                    "denoise_zero", node_id=nid, node_type=ntype,
                    detail="denoise = 0 will pass the latent through unchanged.",
                    fix_hint="Set denoise > 0 (1.0 for txt2img, ~0.6 for img2img refine)."))
            if isinstance(seed, int) and seed >= 0:
                sampler_seeds.setdefault(seed, []).append(nid)

            if not _has_outgoing_link(links, nid):
                findings.append(_finding(
                    "unconnected_terminal", node_id=nid, node_type=ntype,
                    detail=f"{ntype} (id {nid}) output is not connected.",
                    fix_hint="Pipe its LATENT into VAEDecode → SaveImage / PreviewImage."))

        # LoRA / ControlNet / Checkpoint stats
        if ntype.startswith("LoraLoader"):
            stats["loras"] += 1
        if ntype.startswith("ControlNet") or "ControlNet" in ntype:
            stats["controlnets"] += 1
        if ntype in _CHECKPOINT_TYPES:
            stats["checkpoints"] += 1
            ckpt = _widget_value(n, ("ckpt_name", "unet_name"))
            if isinstance(ckpt, str):
                fam = _guess_family(ckpt)
                if fam:
                    checkpoint_families.append(fam)

        # EmptyLatentImage size
        if ntype == "EmptyLatentImage":
            w = _widget_value(n, ("width",))
            h = _widget_value(n, ("height",))
            bad = []
            for axis, val in (("width", w), ("height", h)):
                if isinstance(val, int):
                    if val < 64 or val > 4096 or val % 8 != 0:
                        bad.append(f"{axis}={val}")
            if bad:
                findings.append(_finding(
                    "empty_latent_size", node_id=nid, node_type=ntype,
                    detail="Unusual latent size: " + ", ".join(bad),
                    fix_hint="Use multiples of 8 between 64 and 4096."))

        if ntype in _OUTPUT_TYPES:
            stats["outputs"] += 1

        # ---------- v2 P8: mask-hygiene rules ----------
        # Detect whether this is a multi-frame / video workflow once.
        # We approximate "video / multi-frame" by checking known video loaders
        # OR EmptyLatentImage batch_size > 1.
        # (Computed lazily below — see _is_multiframe_graph.)

        # Rule 1: gaussian temporal smoothing on long batches
        if ntype == "MaskTemporalMEC":
            tmode = _widget_value(n, ("temporal_mode",))
            if tmode == "gaussian" and _approx_batch_size(nodes) > 12:
                findings.append(_finding(
                    "temporal_gaussian_long_batch", node_id=nid, node_type=ntype,
                    detail=f"MaskTemporalMEC temporal_mode=gaussian with approx batch>{12}.",
                    fix_hint="Switch temporal_mode to 'raft_flow' for motion-preserving stabilization."))

        # Rule 2: SAM 3.1 without negative bbox prompts
        if ntype in ("UnifiedSegmentation",) or "SAM3" in ntype:
            seg_mode = _widget_value(n, ("model_name", "model", "seg_model", "segmenter"))
            uses_sam3 = isinstance(seg_mode, str) and ("sam3" in seg_mode.lower() or "sam 3" in seg_mode.lower())
            if uses_sam3 or "SAM3" in ntype:
                neg_slot_names = ("neg_bbox_json", "input_boxes_labels", "negative_bbox")
                has_neg = False
                for inp in n.get("inputs") or []:
                    if isinstance(inp, dict) and inp.get("name") in neg_slot_names and inp.get("link") is not None:
                        has_neg = True
                        break
                wv = _widget_value(n, neg_slot_names)
                if isinstance(wv, str) and wv.strip():
                    has_neg = True
                if not has_neg:
                    findings.append(_finding(
                        "sam3_no_neg_boxes", node_id=nid, node_type=ntype,
                        detail="SAM 3.x detected without negative bbox prompts (neg_bbox_json).",
                        fix_hint="Set neg_bbox_json to a [x1,y1,x2,y2] around regions that should be excluded."))

        # Rule 3: video inpaint without roto_quality
        if ntype == "InpaintCropProMEC":
            roto = _widget_value(n, ("roto_quality",))
            if _approx_batch_size(nodes) > 1 and roto is not True:
                findings.append(_finding(
                    "video_inpaint_no_roto", node_id=nid, node_type=ntype,
                    detail="InpaintCropProMEC running on a multi-frame batch with roto_quality=False.",
                    fix_hint="Enable roto_quality=True for crisp 1-px erosion + laplacian-pyramid blending."))

        # Rule 4: ViTMatte-capable node with non-auto trimap, OR MaskMattingMEC
        # using a vitmatte backend without a subject_preset tuned for it.
        if "ViTMatte" in ntype or "VitMatte" in ntype:
            tri = _widget_value(n, ("trimap_mode", "trimap"))
            if isinstance(tri, str) and tri.lower() not in ("auto", ""):
                findings.append(_finding(
                    "vitmatte_manual_trimap", node_id=nid, node_type=ntype,
                    detail=f"trimap_mode='{tri}' (not 'auto') for a ViTMatte-capable node.",
                    fix_hint="Set trimap_mode='auto' unless you have a verified hand-painted trimap."))
        if ntype == "MaskMattingMEC":
            matter = _widget_value(n, ("matter",))
            subj = _widget_value(n, ("subject_preset", "preset"))
            if (isinstance(matter, str) and "vitmatte" in matter.lower() and
                    isinstance(subj, str) and subj.lower() in ("general", "none", "")):
                findings.append(_finding(
                    "vitmatte_manual_trimap", node_id=nid, node_type=ntype,
                    detail=f"matter='{matter}' with subject_preset='{subj}'.",
                    title_override="ViTMatte backend without a tuned subject preset",
                    fix_hint="Pick subject_preset=hair / fur / hard_edge so the auto-trimap matches the subject."))

        # Rule 5: integrity disabled in multi-frame
        if ntype == "MaskRefineMEC":
            integrity = _widget_value(n, ("enable_integrity_check",))
            if _approx_batch_size(nodes) > 1 and integrity is False:
                findings.append(_finding(
                    "multiframe_integrity_disabled", node_id=nid, node_type=ntype,
                    detail="MaskRefineMEC on multi-frame batch with enable_integrity_check=False.",
                    fix_hint="Enable enable_integrity_check=True or add a MaskTemporalMEC node."))

    # ---------- duplicate seeds
    for seed, sampler_ids in sampler_seeds.items():
        if len(sampler_ids) > 1:
            findings.append(_finding(
                "duplicate_seed",
                detail=f"Seed {seed} is shared by samplers: {sorted(sampler_ids)}",
                fix_hint="Randomize one of them or use control_after_generate=randomize."))

    # ---------- LoRA stack depth
    if stats["loras"] > 5:
        findings.append(_finding(
            "lora_stack_deep",
            detail=f"{stats['loras']} LoRA loaders chained.",
            fix_hint="Consider merging LoRAs offline or trimming to the strongest 2–3."))

    # ---------- checkpoint family mismatch
    fams = set(f for f in checkpoint_families if f != "unknown")
    if len(fams) > 1:
        findings.append(_finding(
            "checkpoint_mismatch",
            detail=f"Multiple model families detected: {sorted(fams)}",
            fix_hint="Make sure each sampler uses a CLIP / VAE matching its checkpoint family."))

    # ---------- orphan outputs
    for n in nodes:
        ntype = str(n.get("type", "") or n.get("class_type", ""))
        if ntype not in _OUTPUT_TYPES:
            continue
        nid = int(n.get("id", -1))
        # Walk upstream BFS, look for any sampler.
        if not _upstream_has(nodes, links, nid, _SAMPLER_TYPES):
            findings.append(_finding(
                "orphan_output", node_id=nid, node_type=ntype,
                detail=f"{ntype} (id {nid}) has no upstream sampler.",
                fix_hint="Connect a KSampler → VAEDecode → this output node."))

    # ---------- no sampler at all
    if not has_sampler and nodes:
        findings.append(_finding(
            "no_sampler",
            detail="No sampler nodes in the graph.",
            fix_hint="Add a KSampler if you intended to generate an image."))

    # ---------- summary
    summary = {"errors": 0, "warnings": 0, "infos": 0}
    for f in findings:
        s = f["severity"]
        if s == "error":
            summary["errors"] += 1
        elif s == "warning":
            summary["warnings"] += 1
        else:
            summary["infos"] += 1

    return {"summary": summary, "findings": findings, "stats": stats}


# ------------------------------------------------------------ helpers
def _approx_batch_size(nodes: List[Dict[str, Any]]) -> int:
    """Heuristic batch-size for the graph. Looks at EmptyLatentImage.batch_size,
    VHS_LoadVideo.frame_load_cap, and any node with a 'batch_size' widget."""
    max_b = 1
    video_loader_types = ("VHS_LoadVideo", "VHS_LoadVideoPath", "LoadVideo")
    for n in nodes:
        ntype = str(n.get("type", "") or n.get("class_type", ""))
        bs = _widget_value(n, ("batch_size", "frame_load_cap"))
        if isinstance(bs, int) and bs > max_b:
            max_b = bs
        if ntype in video_loader_types:
            # If no widget value found, assume >1.
            if max_b == 1:
                max_b = 16
    return max_b


def _guess_family(ckpt_name: str) -> str:
    n = ckpt_name.lower()
    if "xl" in n or "sdxl" in n or "pony" in n or "illustrious" in n:
        return "sdxl"
    if "sd3" in n or "stable-diffusion-3" in n:
        return "sd3"
    if "flux" in n:
        return "flux"
    if "sd15" in n or "sd_15" in n or "v1-5" in n or "anyloraCheckpoint" in n:
        return "sd1.5"
    if "cascade" in n:
        return "cascade"
    if "wan" in n:
        return "wan"
    return "unknown"


def _upstream_has(nodes: List[Dict[str, Any]],
                  links: List[Tuple[int, int, int, int, int, str]],
                  start: int, type_set: Tuple[str, ...]) -> bool:
    by_id = {int(n.get("id", -1)): str(n.get("type", "") or n.get("class_type", ""))
             for n in nodes}
    # Build reverse adj: dst -> [src,...]
    rev: Dict[int, List[int]] = {}
    for _, sn, _, dn, _, _ in links:
        rev.setdefault(dn, []).append(sn)
    seen = {start}
    stack = [start]
    while stack:
        u = stack.pop()
        for v in rev.get(u, ()):
            if v in seen:
                continue
            if by_id.get(v, "") in type_set:
                return True
            seen.add(v)
            stack.append(v)
    return False


# ------------------------------------------------------------ aiohttp routes
def register_routes(server: Any) -> None:
    try:
        from aiohttp import web
    except Exception as e:
        log.warning("[workflow_doctor] aiohttp unavailable: %s", e)
        return

    routes = server.routes

    @routes.post("/c2c/doctor/analyze")
    async def _analyze(req: "web.Request") -> "web.Response":
        try:
            body = await req.json()
        except Exception:
            body = {}
        wf = body.get("workflow") if isinstance(body, dict) else None
        if not isinstance(wf, dict):
            return web.json_response(
                {"success": False, "error": "missing_workflow"}, status=400)
        try:
            data = analyze(wf)
        except Exception as e:
            log.exception("[workflow_doctor] analyze failed")
            return web.json_response(
                {"success": False, "error": "analyze_failed", "message": str(e)},
                status=500)
        return web.json_response({"success": True, "data": data})

    @routes.get("/c2c/doctor/rules")
    async def _rules(_req: "web.Request") -> "web.Response":
        return web.json_response({"success": True, "data": RULES})

    log.info("[workflow_doctor] Routes registered: POST /c2c/doctor/analyze, GET /c2c/doctor/rules")
