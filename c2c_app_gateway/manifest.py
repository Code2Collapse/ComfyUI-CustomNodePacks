"""Public app manifest (schema 1), private release bindings, and run-value validation.

The manifest is the ONLY thing a runner learns about an app: logical input / output ids, types, constraints, defaults,
limits. The private release (ComfyUI API prompt + bindings from logical ids to node inputs + declared output nodes)
stays on the gateway. Every message raised here for a runner names only logical ids.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
import secrets

SCHEMA = 1
ID_RE = re.compile(r"^[a-z][a-z0-9_]{0,47}$")
INPUT_TYPES = ("image", "text", "number", "int", "choice", "seed", "boolean")
OUTPUT_TYPES = ("image", "video", "audio", "file")
OUTPUT_KINDS = {"images": ("image", "file"), "gifs": ("video", "image", "file"), "videos": ("video", "file"),
                "audio": ("audio", "file"), "files": ("file",)}
UPLOAD_MIME = {"image/png": ("png", lambda b: b.startswith(b"\x89PNG\r\n\x1a\n")),
               "image/jpeg": ("jpg", lambda b: b.startswith(b"\xff\xd8\xff")),
               "image/webp": ("webp", lambda b: len(b) > 12 and b[:4] == b"RIFF" and b[8:12] == b"WEBP")}
MAX_SEED = 2 ** 53 - 1
LIMIT_DEFAULTS = {"max_runtime_s": 600, "max_concurrent_per_user": 1, "runs_per_hour": 30}
LIMIT_CAPS = {"max_runtime_s": 6 * 3600, "max_concurrent_per_user": 16, "runs_per_hour": 10000}


class ManifestError(ValueError):
    """A manifest / release the publisher must fix (message is for the owner / publisher)."""


class InputError(ValueError):
    """A run value a runner must fix (message names only logical ids)."""


def _num(v, what):
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ManifestError(f"{what} must be a number.")
    return v


def canonical_hash(manifest: dict) -> str:
    body = {k: v for k, v in manifest.items() if k not in ("schema_hash", "release")}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def validate_manifest(m: dict, app_id: str) -> dict:
    """Return a normalised copy of a publisher's manifest (unknown keys dropped) with its schema hash."""
    if not isinstance(m, dict):
        raise ManifestError("The manifest must be a JSON object.")
    out = {"schema": SCHEMA, "app_id": app_id, "name": str(m.get("name") or app_id)[:120],
           "description": str(m.get("description") or "")[:4000], "inputs": [], "outputs": []}
    seen = set()
    inputs = m.get("inputs")
    if not isinstance(inputs, list) or len(inputs) > 64:
        raise ManifestError("inputs must be a list of at most 64 entries.")
    for i, x in enumerate(inputs):
        if not isinstance(x, dict) or not ID_RE.match(str(x.get("id", ""))):
            raise ManifestError(f"Input {i + 1}: id must be lower-case letters, digits and '_' (start with a letter).")
        iid, t = x["id"], x.get("type")
        if iid in seen:
            raise ManifestError(f"Input '{iid}' is listed twice.")
        seen.add(iid)
        if t not in INPUT_TYPES:
            raise ManifestError(f"Input '{iid}': type must be one of {', '.join(INPUT_TYPES)}.")
        e = {"id": iid, "type": t, "label": str(x.get("label") or iid)[:80], "required": bool(x.get("required", False))}
        if x.get("help"):
            e["help"] = str(x["help"])[:400]
        if t == "image":
            mb = _num(x.get("max_mb", 20), f"Input '{iid}' max_mb")
            if not 0 < mb <= 200:
                raise ManifestError(f"Input '{iid}': max_mb must be between 0 and 200.")
            acc = x.get("accept") or list(UPLOAD_MIME)
            if not isinstance(acc, list) or not acc or any(a not in UPLOAD_MIME for a in acc):
                raise ManifestError(f"Input '{iid}': accept may list only {', '.join(UPLOAD_MIME)}.")
            e.update(max_mb=mb, accept=sorted(set(acc)))
        elif t == "text":
            ml = x.get("max_len", 2000)
            if isinstance(ml, bool) or not isinstance(ml, int) or not 1 <= ml <= 100000:
                raise ManifestError(f"Input '{iid}': max_len must be an integer 1..100000.")
            e["max_len"] = ml
            if "default" in x:
                if not isinstance(x["default"], str) or len(x["default"]) > ml:
                    raise ManifestError(f"Input '{iid}': default must be text within max_len.")
                e["default"] = x["default"]
        elif t in ("number", "int"):
            lo, hi = _num(x.get("min"), f"Input '{iid}' min"), _num(x.get("max"), f"Input '{iid}' max")
            if lo > hi:
                raise ManifestError(f"Input '{iid}': min is above max.")
            e.update(min=lo, max=hi)
            if t == "int" and (lo != int(lo) or hi != int(hi)):
                raise ManifestError(f"Input '{iid}': int bounds must be whole numbers.")
            if "step" in x:
                e["step"] = _num(x["step"], f"Input '{iid}' step")
            if "default" in x:
                d = _num(x["default"], f"Input '{iid}' default")
                if not lo <= d <= hi or (t == "int" and d != int(d)):
                    raise ManifestError(f"Input '{iid}': default is outside min..max.")
                e["default"] = int(d) if t == "int" else d
        elif t == "choice":
            opts = x.get("options")
            if (not isinstance(opts, list) or not opts or len(opts) > 256
                    or any(not isinstance(o, str) or not o or len(o) > 200 for o in opts) or len(set(opts)) != len(opts)):
                raise ManifestError(f"Input '{iid}': options must be 1..256 distinct non-empty strings.")
            e["options"] = list(opts)
            if "default" in x:
                if x["default"] not in opts:
                    raise ManifestError(f"Input '{iid}': default must be one of the options.")
                e["default"] = x["default"]
        elif t == "seed":
            if "default" in x:
                d = x["default"]
                if isinstance(d, bool) or not isinstance(d, int) or not 0 <= d <= MAX_SEED:
                    raise ManifestError(f"Input '{iid}': default seed must be an integer 0..2^53-1.")
                e["default"] = d
        elif t == "boolean":
            if "default" in x:
                if not isinstance(x["default"], bool):
                    raise ManifestError(f"Input '{iid}': default must be true or false.")
                e["default"] = x["default"]
        out["inputs"].append(e)
    outputs = m.get("outputs")
    if not isinstance(outputs, list) or not outputs or len(outputs) > 32:
        raise ManifestError("outputs must list 1..32 public outputs.")
    for i, x in enumerate(outputs):
        if not isinstance(x, dict) or not ID_RE.match(str(x.get("id", ""))):
            raise ManifestError(f"Output {i + 1}: id must be lower-case letters, digits and '_'.")
        if x["id"] in seen:
            raise ManifestError(f"'{x['id']}' is used twice.")
        seen.add(x["id"])
        if x.get("type") not in OUTPUT_TYPES:
            raise ManifestError(f"Output '{x['id']}': type must be one of {', '.join(OUTPUT_TYPES)}.")
        out["outputs"].append({"id": x["id"], "type": x["type"], "label": str(x.get("label") or x["id"])[:80],
                               "preview": bool(x.get("preview", x["type"] == "image"))})
    lim = dict(LIMIT_DEFAULTS)
    for k, v in (m.get("limits") or {}).items():
        if k not in LIMIT_DEFAULTS:
            raise ManifestError(f"Unknown limit '{k}'.")
        if isinstance(v, bool) or not isinstance(v, int) or not 1 <= v <= LIMIT_CAPS[k]:
            raise ManifestError(f"Limit '{k}' must be an integer 1..{LIMIT_CAPS[k]}.")
        lim[k] = v
    out["limits"] = lim
    out["cancel"] = m.get("cancel", "allowed")
    if out["cancel"] not in ("allowed", "never"):
        raise ManifestError("cancel must be 'allowed' or 'never'.")
    out["schema_hash"] = canonical_hash(out)
    return out


def validate_private(private: dict, manifest: dict) -> dict:
    """Check a private release against its manifest: every public input bound to real node inputs, every output to
    a node in the prompt, nothing bound that the manifest does not declare."""
    if not isinstance(private, dict):
        raise ManifestError("The private release must be a JSON object.")
    prompt = private.get("prompt")
    if not isinstance(prompt, dict) or not prompt:
        raise ManifestError("The private release needs the ComfyUI API prompt (graphToPrompt output).")
    for nid, node in prompt.items():
        if not isinstance(node, dict) or not isinstance(node.get("class_type"), str) \
                or not isinstance(node.get("inputs"), dict):
            raise ManifestError(f"Prompt node {nid} is not an API-format node.")
    bindings, outs = private.get("bindings") or {}, private.get("outputs") or {}
    in_ids = {x["id"]: x for x in manifest["inputs"]}
    out_ids = {x["id"]: x for x in manifest["outputs"]}
    if set(bindings) != set(in_ids):
        miss, extra = sorted(set(in_ids) - set(bindings)), sorted(set(bindings) - set(in_ids))
        raise ManifestError(f"Bindings must cover exactly the public inputs (missing {miss}, not in manifest {extra}).")
    norm_b = {}
    for iid, targets in bindings.items():
        if not isinstance(targets, list) or not targets:
            raise ManifestError(f"Input '{iid}' must bind to at least one node input.")
        norm_b[iid] = []
        for t in targets:
            nid, key = str((t or {}).get("node", "")), (t or {}).get("input")
            if nid not in prompt or not isinstance(key, str) or key not in prompt[nid]["inputs"]:
                raise ManifestError(f"Input '{iid}' binds to node {nid} input {key!r}, which the prompt does not have.")
            if isinstance(prompt[nid]["inputs"][key], list):
                raise ManifestError(f"Input '{iid}' binds to node {nid} input {key!r}, which is a link, not a value.")
            norm_b[iid].append({"node": nid, "input": key})
    if set(outs) != set(out_ids):
        raise ManifestError("Output bindings must cover exactly the public outputs.")
    norm_o = {}
    for oid, o in outs.items():
        nid, kind = str((o or {}).get("node", "")), (o or {}).get("kind", "images")
        if nid not in prompt:
            raise ManifestError(f"Output '{oid}' reads node {nid}, which the prompt does not have.")
        if kind not in OUTPUT_KINDS or out_ids[oid]["type"] not in OUTPUT_KINDS[kind]:
            raise ManifestError(f"Output '{oid}': kind {kind!r} does not fit type {out_ids[oid]['type']}.")
        norm_o[oid] = {"node": nid, "kind": kind}
    return {"prompt": copy.deepcopy(prompt), "bindings": norm_b, "outputs": norm_o,
            "worker": str(private.get("worker") or "default")}


def resolve_values(manifest: dict, values: dict, files: dict) -> dict:
    """Validate a runner's values (+ uploaded file bytes keyed by input id) against the manifest.
    Returns {input_id: value} with uploads as {"upload": bytes, "ext": str}."""
    if not isinstance(values, dict):
        raise InputError("values must be a JSON object.")
    spec = {x["id"]: x for x in manifest["inputs"]}
    unknown = sorted((set(values) | set(files)) - set(spec))
    if unknown:
        raise InputError(f"Unknown input(s): {', '.join(unknown)}.")
    out = {}
    for iid, s in spec.items():
        t = s["type"]
        if t == "image":
            if iid in values:
                raise InputError(f"'{iid}' must be uploaded as a file.")
            data = files.get(iid)
            if data is None:
                if s["required"]:
                    raise InputError(f"'{iid}' is required.")
                continue
            if len(data) > s["max_mb"] * 1024 * 1024:
                raise InputError(f"'{iid}' is larger than {s['max_mb']} MB.")
            ext = next((UPLOAD_MIME[m][0] for m in s["accept"] if UPLOAD_MIME[m][1](data)), None)
            if not ext:
                raise InputError(f"'{iid}' must be one of: {', '.join(s['accept'])}.")
            out[iid] = {"upload": data, "ext": ext}
            continue
        if iid not in values:
            if "default" in s:
                out[iid] = s["default"]
            elif t == "seed":
                out[iid] = secrets.randbelow(MAX_SEED)
            elif s["required"]:
                raise InputError(f"'{iid}' is required.")
            continue                                                       # optional, no default: keep the private value
        v = values[iid]
        if t == "text":
            if not isinstance(v, str) or len(v) > s["max_len"]:
                raise InputError(f"'{iid}' must be text of at most {s['max_len']} characters.")
            if "\x00" in v:
                raise InputError(f"'{iid}' contains a null character.")
        elif t in ("number", "int"):
            if isinstance(v, bool) or not isinstance(v, (int, float)) or v != v or not s["min"] <= v <= s["max"]:
                raise InputError(f"'{iid}' must be a number from {s['min']} to {s['max']}.")
            if t == "int":
                if v != int(v):
                    raise InputError(f"'{iid}' must be a whole number.")
                v = int(v)
        elif t == "choice":
            if v not in s["options"]:
                raise InputError(f"'{iid}' must be one of the listed options.")
        elif t == "seed":
            if isinstance(v, bool) or not isinstance(v, int) or not 0 <= v <= MAX_SEED:
                raise InputError(f"'{iid}' must be an integer seed from 0 to {MAX_SEED}.")
        elif t == "boolean":
            if not isinstance(v, bool):
                raise InputError(f"'{iid}' must be true or false.")
        out[iid] = v
    return out


def build_prompt(private: dict, resolved: dict, uploaded: dict) -> dict:
    """Deep-copy the private prompt and write each bound value. ``uploaded`` maps input id -> worker file name."""
    prompt = copy.deepcopy(private["prompt"])
    for iid, targets in private["bindings"].items():
        if iid in uploaded:
            val = uploaded[iid]
        elif iid in resolved:
            val = resolved[iid]
        else:
            continue                                                       # optional upload left out: keep default
        for t in targets:
            prompt[t["node"]]["inputs"][t["input"]] = val
    return prompt
