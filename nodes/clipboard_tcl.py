# FILE: nodes/clipboard_tcl.py
# FEATURE: F5 — Nuke-style TCL copy/paste for ComfyUI graphs
# INTEGRATES WITH: web/extensions/nukenodemax/clipboard_tcl.js
"""
Nuke `.nk`-style TCL serialiser / parser for a subgraph.

Output format (NO JSON, NO UUIDs, single-line-diff friendly):

    set cut_paste_input [stack 0]
    push $cut_paste_input
    LoadImage {
     image_path foo.png
     xpos 100
     ypos 50
    }
    set N1_LoadImage [stack 0]
    push $N1_LoadImage
    Blur {
     inputs 1
     size 12
     xpos 200
     ypos 50
    }
    set N2_Blur [stack 0]
    end_group

For multi-input nodes we push each source alias in *reverse slot order*
before the node block; the node consumes them via `inputs N`.

Server endpoints (registered when ComfyUI is running):
    POST /nukenodemax/copy_tcl    {nodes:[...], links:[[from_id, from_slot, to_id, to_slot], ...]}
    POST /nukenodemax/paste_tcl   raw text/plain TCL
"""
from __future__ import annotations

import json
import logging
import re
from typing import Dict, List, Tuple

log = logging.getLogger("MEC.clipboard_tcl")


# =====================================================================
# Serialiser
# =====================================================================
def _alias(node_id: int, name: str) -> str:
    safe = re.sub(r"\W+", "_", name) or "node"
    return f"N{int(node_id)}_{safe}"


def _emit_value(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return str(v)
    if v is None:
        return "{}"
    s = str(v)
    if not s:
        return "{}"
    if any(c.isspace() for c in s) or "{" in s or "}" in s:
        return "{" + s.replace("{", r"\{").replace("}", r"\}") + "}"
    return s


def serialize(nodes: List[Dict], links: List[Tuple[int, int, int, int]]) -> str:
    """
    nodes: [{id, class_type, name, widgets:{...}, xpos, ypos, selected}, ...]
    links: [(from_id, from_slot, to_id, to_slot), ...]
    """
    by_id = {n["id"]: n for n in nodes}
    # Inputs of each node sorted by slot.
    inputs_by_id: Dict[int, List[Tuple[int, int]]] = {n["id"]: [] for n in nodes}
    for f_id, f_slot, t_id, t_slot in links:
        if f_id in by_id and t_id in by_id:
            inputs_by_id.setdefault(t_id, []).append((t_slot, f_id))
    for k in inputs_by_id:
        inputs_by_id[k] = [(slot, src) for slot, src in sorted(inputs_by_id[k])]

    # Topological order by greedy: node whose all input sources have aliases.
    emitted: List[int] = []
    pending = list(by_id.keys())
    aliases: Dict[int, str] = {}
    out = ["set cut_paste_input [stack 0]"]
    # Stack is initially [cut_paste_input]; push it once for the first natural flow.
    out.append("push $cut_paste_input")
    natural_top: int = -1  # id of the node at top of stack (last `set N..`)

    while pending:
        progressed = False
        for nid in list(pending):
            srcs = inputs_by_id.get(nid, [])
            if any(src not in aliases for _slot, src in srcs):
                continue
            n = by_id[nid]
            # Determine pushes needed in reverse slot order.
            slot_to_src = {slot: src for slot, src in srcs}
            num_in = (max(slot_to_src) + 1) if slot_to_src else 0
            if num_in:
                # Slot 0 must be on TOP of stack just before the block.
                # Push from highest slot down to slot 0 -> stack top = slot 0.
                desired_top = slot_to_src.get(0)
                if num_in == 1 and desired_top == natural_top:
                    pass  # natural flow; no push needed
                else:
                    for s in range(num_in - 1, -1, -1):
                        src = slot_to_src.get(s)
                        if src is None:
                            out.append("push 0")
                        else:
                            out.append(f"push ${aliases[src]}")
            # Emit block.
            out.append(f"{n['class_type']} {{")
            if num_in:
                out.append(f" inputs {num_in}")
            for k, v in (n.get("widgets") or {}).items():
                out.append(f" {k} {_emit_value(v)}")
            if "name" in n and n["name"]:
                out.append(f" name {_emit_value(n['name'])}")
            out.append(f" xpos {int(n.get('xpos', 0))}")
            out.append(f" ypos {int(n.get('ypos', 0))}")
            if n.get("selected"):
                out.append(" selected true")
            out.append("}")
            alias = _alias(nid, n.get("name") or n["class_type"])
            aliases[nid] = alias
            out.append(f"set {alias} [stack 0]")
            natural_top = nid
            emitted.append(nid)
            pending.remove(nid)
            progressed = True
        if not progressed:
            # Cycle / disconnected — emit remainder with no inputs declared.
            for nid in pending:
                n = by_id[nid]
                out.append(f"{n['class_type']} {{")
                for k, v in (n.get("widgets") or {}).items():
                    out.append(f" {k} {_emit_value(v)}")
                out.append(f" xpos {int(n.get('xpos', 0))}")
                out.append(f" ypos {int(n.get('ypos', 0))}")
                out.append("}")
                aliases[nid] = _alias(nid, n.get("name") or n["class_type"])
                out.append(f"set {aliases[nid]} [stack 0]")
            pending = []
    out.append("end_group")
    return "\n".join(out) + "\n"


# =====================================================================
# Parser
# =====================================================================
_TOKEN_RE = re.compile(r"\{|\}|\$\w+|\[[^\]]*\]|\S+")


def _tokenize(text: str) -> List[str]:
    out = []
    for line in text.splitlines():
        # strip line comments starting with `#`
        line = re.sub(r"#.*$", "", line)
        out.extend(_TOKEN_RE.findall(line))
    return out


def _parse_braced(tokens: List[str], i: int) -> Tuple[str, int]:
    """Read a brace-quoted value starting at tokens[i] == '{'. Returns (value, next_i)."""
    assert tokens[i] == "{"
    depth = 1
    parts: List[str] = []
    i += 1
    while i < len(tokens) and depth > 0:
        if tokens[i] == "{":
            depth += 1
            parts.append("{")
        elif tokens[i] == "}":
            depth -= 1
            if depth == 0:
                break
            parts.append("}")
        else:
            parts.append(tokens[i])
        i += 1
    return " ".join(parts), i + 1


def parse(text: str) -> Dict:
    """Returns {'nodes': [...], 'links': [...], 'aliases': {...}}."""
    tokens = _tokenize(text)
    stack: List[str] = []          # alias names
    aliases: Dict[str, int] = {}   # alias -> assigned id
    nodes: List[Dict] = []
    links: List[Tuple[int, int, int, int]] = []
    next_id = 1
    i = 0
    n = len(tokens)
    while i < n:
        tok = tokens[i]
        if tok in ("end_group",):
            i += 1
            continue
        if tok == "set" and i + 3 < n and tokens[i + 2] == "[" or \
           (tok == "set" and i + 1 < n and tokens[i + 1].startswith(("N", "cut_paste_"))):
            # Forms: `set NAME [stack 0]`  → alias top of stack to NAME.
            name = tokens[i + 1]
            # Scan to closing `]` (it was tokenised whole if simple, otherwise we consume).
            j = i + 2
            while j < n and tokens[j] != "[stack 0]" and not tokens[j].startswith("[stack"):
                j += 1
            i = j + 1
            if stack:
                top = stack[-1]
                aliases[name] = aliases.get(top, -1)
            continue
        if tok == "push" and i + 1 < n:
            ref = tokens[i + 1]
            if ref.startswith("$"):
                ref = ref[1:]
            stack.append(ref)
            i += 2
            continue
        # Otherwise expect a node block: ClassName { ... }
        if i + 1 < n and tokens[i + 1] == "{":
            class_type = tok
            block, i = _parse_braced(tokens, i + 1)
            # Re-tokenise block contents pair-wise.
            inner = _tokenize(block)
            widgets: Dict[str, str] = {}
            num_inputs = 0
            name = None
            xpos = ypos = 0
            selected = False
            j = 0
            while j < len(inner):
                key = inner[j]
                j += 1
                if j >= len(inner):
                    break
                if inner[j] == "{":
                    val, j = _parse_braced(inner, j)
                else:
                    val = inner[j]
                    j += 1
                if key == "inputs":
                    try:
                        num_inputs = int(val)
                    except ValueError:
                        num_inputs = 0
                elif key == "name":
                    name = val
                elif key == "xpos":
                    xpos = int(float(val))
                elif key == "ypos":
                    ypos = int(float(val))
                elif key == "selected":
                    selected = val.lower() in ("true", "1", "yes")
                else:
                    widgets[key] = val
            nid = next_id
            next_id += 1
            # Pop `num_inputs` items from stack, slot 0 = top.
            popped: List[Tuple[int, str]] = []
            for slot in range(num_inputs):
                if not stack:
                    break
                src_alias = stack.pop()
                src_id = aliases.get(src_alias)
                if src_id and src_id != -1:
                    popped.append((slot, src_alias))
                    links.append((src_id, 0, nid, slot))
            nodes.append({
                "id": nid, "class_type": class_type, "name": name,
                "widgets": widgets, "xpos": xpos, "ypos": ypos,
                "selected": selected,
            })
            stack.append(f"_NODE_{nid}")
            aliases[f"_NODE_{nid}"] = nid
            continue
        i += 1
    return {"nodes": nodes, "links": links, "aliases": aliases}


# =====================================================================
# Comfy nodes (in/out the graph)
# =====================================================================
class TclSerializeMEC:
    DESCRIPTION = "Serialize a subgraph JSON description into Nuke-style TCL."
    CATEGORY = "MaskEditControl/Clipboard"
    FUNCTION = "to_tcl"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("tcl",)

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"subgraph_json": ("STRING", {"multiline": True,
                                                          "default": "{\"nodes\":[],\"links\":[]}"})}}

    def to_tcl(self, subgraph_json: str):
        d = json.loads(subgraph_json)
        return (serialize(d.get("nodes", []), d.get("links", [])),)


class TclParseMEC:
    DESCRIPTION = "Parse Nuke-style TCL into a subgraph JSON description."
    CATEGORY = "MaskEditControl/Clipboard"
    FUNCTION = "to_json"
    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("subgraph_json", "node_count")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"tcl": ("STRING", {"multiline": True, "default": ""})}}

    def to_json(self, tcl: str):
        d = parse(tcl)
        return (json.dumps(d, indent=1), len(d["nodes"]))


NODE_CLASS_MAPPINGS = {
    "TclSerializeMEC": TclSerializeMEC,
    "TclParseMEC": TclParseMEC,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "TclSerializeMEC": "TCL Serialize (MEC)",
    "TclParseMEC": "TCL Parse (MEC)",
}


# =====================================================================
# Server routes
# =====================================================================
def register_routes(server) -> None:
    """Register /nukenodemax/copy_tcl and /paste_tcl on the ComfyUI server."""
    try:
        from aiohttp import web
    except Exception:
        log.warning("[clipboard_tcl] aiohttp unavailable, skipping route registration")
        return

    routes = server.routes

    @routes.post("/nukenodemax/copy_tcl")
    async def _copy_tcl(req):
        try:
            body = await req.json()
        except Exception as e:
            return web.json_response({"ok": False, "error": str(e)}, status=400)
        tcl = serialize(body.get("nodes", []), body.get("links", []))
        return web.Response(text=tcl, content_type="text/plain")

    @routes.post("/nukenodemax/paste_tcl")
    async def _paste_tcl(req):
        text = await req.text()
        try:
            d = parse(text)
        except Exception as e:
            return web.json_response({"ok": False, "error": str(e)}, status=400)
        return web.json_response({"ok": True, **d})

    log.info("[clipboard_tcl] routes registered")
