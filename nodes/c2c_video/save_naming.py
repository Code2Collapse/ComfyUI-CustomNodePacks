"""Save Video (C2C) naming (L9.40, docs/save-video-v2.md): naming = Folder Version Incrementer (or default) fills the
version path by itself.

1. A wired / typed `subfolder` still wins (the old way: the incrementer's subfolder output wired in).
2. Else the first Folder Version Incrementer node in the prompt - the hidden PROMPT holds every node, also unconnected
   ones - gives its settings, and its own `increment()` runs with them: the same disk scan, the same per-run lease, so
   the node and Save Video never disagree about the version.
3. Else the incrementer runs with its defaults (source name detected from the loaders in the prompt).

The incrementer creates no folder by itself (reserve_version off), so a version is used only when a file is written: a
preview run or a failed write leaves the next version free.
"""

from __future__ import annotations

import inspect
import os
from typing import Any

NAMING_CHOICES = ["default", "ComfyUI counter", "Folder Version Incrementer"]
FOLDER = "Folder Version Incrementer"
_ALIASES = {"Folder Version": FOLDER, "folder version": FOLDER, "ComfyUI counter": "ComfyUI counter"}
DEFAULT_SETTING = FOLDER
INCREMENTER = "FolderIncrementer"


def canonical(naming: str) -> str:
    n = str(naming or "").strip()
    return _ALIASES.get(n, n)


def _incrementer_module():
    try:
        from ... import folder_incrementer as fi            # custom_nodes.<pack>.nodes.c2c_video -> <pack>
    except (ImportError, ValueError):
        import folder_incrementer as fi                     # tests: the pack root is on sys.path
    return fi


def _node_order(node_id: str):
    parts = []
    for p in str(node_id).split(":"):
        parts.append((0, int(p)) if p.isdigit() else (1, p))
    return parts


def find_incrementer(prompt: Any) -> tuple[str, dict] | None:
    """(node id, its literal inputs) of the first Folder Version Incrementer in an API prompt (subgraph ids such as
    "12:3" included), or None."""
    if not isinstance(prompt, dict):
        return None
    hits = [(nid, node) for nid, node in prompt.items()
            if isinstance(node, dict) and node.get("class_type") == INCREMENTER]
    if not hits:
        return None
    nid, node = min(hits, key=lambda kv: _node_order(kv[0]))
    inputs = node.get("inputs") or {}
    literal = {k: v for k, v in inputs.items() if not (isinstance(v, list) and len(v) == 2 and isinstance(v[1], int))}
    return str(nid), literal


def _accepted(fi) -> set[str]:
    sig = inspect.signature(fi.FolderIncrementer.increment)
    return {p for p in sig.parameters if p not in ("self", "prompt", "unique_id")}


def resolve_version(prompt: Any, own_id: Any = None, *, reserve: bool = False) -> dict:
    """What the incrementer gives for this run: subfolder_path, filename_prefix, version_string, version_path, the
    settings' source, and the incrementer node id (None = built-in defaults)."""
    fi = _incrementer_module()
    found = find_incrementer(prompt)
    settings: dict = {}
    origin = "built-in incrementer (no Folder Version Incrementer in the graph)"
    inc_id = None
    if found:
        inc_id, literal = found
        ok = _accepted(fi)
        settings = {k: v for k, v in literal.items() if k in ok}
        origin = f"Folder Version Incrementer node {inc_id}"
    settings["reserve_version"] = bool(reserve)
    out = fi.FolderIncrementer().increment(**settings, prompt=prompt, unique_id=inc_id or own_id)
    (version_string, version_num, folder_name, subfolder_path, filename_prefix, output_filename, source_stem,
     _cur, _nxt, _nxt_stem, version_path, info_json) = out
    base = str(settings.get("base_path") or "").strip()
    return {"version_string": version_string, "version": int(version_num), "folder": folder_name,
            "subfolder_path": str(subfolder_path).replace("\\", "/"),
            "filename_prefix": str(filename_prefix).replace("\\", "/"),
            "version_path": str(version_path).replace("\\", "/"), "base_path": base, "source": source_stem,
            "origin": origin, "node": inc_id}


def outside_output(base_path: str, output_dir: str) -> bool:
    if not base_path:
        return False
    try:
        return os.path.commonpath([os.path.abspath(base_path), os.path.abspath(output_dir)]) != \
            os.path.abspath(output_dir)
    except ValueError:                                      # different drives on Windows
        return True


def predict_path(*, prompt: Any, unique_id: Any, naming: str, ext: str, sequence: bool, filename_prefix: str,
                 subfolder: str, save_output: bool) -> dict:
    """Where the next write would go - a dry run: no folder is created and no version is taken (outside a run the
    incrementer's lease does not apply, so this is the disk's next free version)."""
    from .save_encode import _import_folder_paths, effective_naming
    fp = _import_folder_paths()
    eff = effective_naming(naming, "")
    out_root = fp.get_output_directory()
    if not save_output:
        return {"naming": eff, "path": "(temp folder - preview only, no version taken)", "version": None}
    sub = str(subfolder or "").strip().replace("\\", "/")
    if eff == FOLDER and not sub:
        v = resolve_version(prompt, unique_id)
        stem = os.path.basename(v["filename_prefix"]) or "video"
        if outside_output(v["base_path"], out_root):
            root, rel = v["base_path"], v["subfolder_path"]
        else:
            root, rel = out_root, os.path.relpath(v["version_path"], out_root).replace("\\", "/")
        name = stem if sequence else f"{stem}.{ext}"
        return {"naming": eff, "path": os.path.join(root, rel, name).replace("\\", "/"),
                "version": v["version_string"], "origin": v["origin"]}
    if eff == FOLDER:
        stem = os.path.basename((filename_prefix or "C2C/video").replace("\\", "/")) or "video"
        name = stem if sequence else f"{stem}.{ext}"
        return {"naming": eff, "path": os.path.join(out_root, sub, name).replace("\\", "/"), "version": None,
                "origin": "the wired subfolder"}
    prefix = (filename_prefix or "C2C/video").strip().replace("\\", "/")
    full_dir, base, counter, _sub, _p = fp.get_save_image_path(prefix, out_root, 0, 0)
    stem = f"{base}_{int(counter):05}"
    name = stem if sequence else f"{stem}.{ext}"
    return {"naming": eff, "path": os.path.join(full_dir, name).replace("\\", "/"), "version": None,
            "origin": "ComfyUI counter"}
