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
import re
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


# ---- the name from the Load Video (L9.55, owner 2026-10-25) --------------------------------------------------------
# "Source name + suffix": Load Video's name_prefix + the source's stem + its name_suffix (no suffix unless the user adds
# one); Save Video's own filename_prefix wins when it is not FROM_LOAD. With the version folder the file carries the
# folder's version: the stem's own version token is replaced, else _<version> is added. Never the source path.
FROM_LOAD = "auto (from Load)"
_SEQ_EXT = {".exr", ".dpx", ".png", ".tif", ".tiff", ".jpg", ".jpeg", ".hdr", ".cin", ".tga", ".bmp", ".webp"}
_SEQ_TOKEN = re.compile(r"(?:[._-]?(?:#+|%0?\d*d|\[\d+\s*-\s*\d+\])|[._-]\d{2,})$")
_VERSION_TOKEN = re.compile(r"(?i)(?<![a-z0-9])v\d{1,6}(?![a-z0-9])")


def source_stem(video: Any) -> str:
    """The source's name without folders, extension or sequence frame token: clip.mov -> clip, D:/p/shot010.####.exr /
    shot010.%04d.exr / shot010.[1001-1100].exr / shot010.1001.exr -> shot010, a sequence folder -> its name."""
    s = re.sub(r"\s*\[(input|output|temp)\]\s*$", "", str(video or "").strip()).replace("\\", "/").rstrip("/")
    name = s.rsplit("/", 1)[-1]
    stem, ext = os.path.splitext(name)
    if ext.lower() not in _SEQ_EXT and ext and not re.search(r"#|%0?\d*d|\[\d", name):
        return stem.strip(" ._-") or "video"                  # a movie: its stem as it is (shot_010.mov stays)
    if not ext and "." not in name and not re.search(r"#|%0?\d*d|\[\d", name):
        return name.strip(" ._-") or "video"                  # a sequence folder
    return _SEQ_TOKEN.sub("", stem if ext else name).strip(" ._-") or "video"


def with_version(base: str, version_string: str | None) -> str:
    """The file name carrying the folder's version (owner: "filename has same version number as folder")."""
    if not version_string:
        return base
    found = list(_VERSION_TOKEN.finditer(base))
    if found:
        last = found[-1]
        return base[:last.start()] + str(version_string) + base[last.end():]
    return f"{base}_{version_string}"


def followed_load(prompt: Any, unique_id: Any, follow: Any = "auto") -> str | None:
    """The Load Video this Save takes its name from: the one its video_info wire comes from, else the one it follows."""
    from . import handles_io as HIO
    if not isinstance(prompt, dict) or unique_id is None:
        return None
    mine = (prompt.get(str(unique_id)) or {}).get("inputs") or {}
    vi = mine.get("video_info")
    if HIO._is_link(vi) and str((prompt.get(str(vi[0])) or {}).get("class_type")) in HIO.LOADER_CLASSES:
        return str(vi[0])
    which = ("images", "IMAGE") if HIO._is_link(mine.get("images")) else ("latent", "LATENT")
    return HIO.resolve_follow(prompt, unique_id, *which, follow=follow)["load"]


def name_from_load(prompt: Any, unique_id: Any, follow: Any = "auto") -> tuple[str | None, str]:
    """(prefix + stem + suffix of the followed Load Video, note) - read from the Load's own inputs in the prompt, so it
    is known before that Load runs. (None, why) without one."""
    src = followed_load(prompt, unique_id, follow)
    if src is None:
        return None, "no Load Video to take the name from"
    ins = (prompt.get(src) or {}).get("inputs") or {}
    # L9.89: Load Video (C2C) in server path mode names its source in 'path' (the merged Load Video Path)
    stem = source_stem(ins.get("path") if str(ins.get("location") or "") == "server path" else ins.get("video"))
    return _clean(ins.get("name_prefix")) + stem + _clean(ins.get("name_suffix")), f"named after Load Video #{src}"


def _clean(v: Any) -> str:
    """A prefix / suffix as file-name text: no folders, no characters Windows refuses."""
    return re.sub(r'[\\/:*?"<>|]+', "_", v) if isinstance(v, str) else ""


def outside_output(base_path: str, output_dir: str) -> bool:
    if not base_path:
        return False
    try:
        return os.path.commonpath([os.path.abspath(base_path), os.path.abspath(output_dir)]) != \
            os.path.abspath(output_dir)
    except ValueError:                                      # different drives on Windows
        return True


def predict_path(*, prompt: Any, unique_id: Any, naming: str, ext: str, sequence: bool, filename_prefix: str,
                 subfolder: str, save_output: bool, follow: Any = "auto") -> dict:
    """Where the next write would go - a dry run: no folder is created and no version is taken (outside a run the
    incrementer's lease does not apply, so this is the disk's next free version)."""
    from .save_encode import _import_folder_paths, effective_naming
    fp = _import_folder_paths()
    eff = effective_naming(naming, "")
    out_root = fp.get_output_directory()
    if not save_output:
        return {"naming": eff, "path": "(temp folder - preview only, no version taken)", "version": None}
    sub = str(subfolder or "").strip().replace("\\", "/")
    base = None
    if str(filename_prefix or "").strip() == FROM_LOAD:           # L9.55: the Load Video's name
        base, _why = name_from_load(prompt, unique_id, follow)
        filename_prefix = f"C2C/{base}" if base else "C2C/video"
    if eff == FOLDER and not sub:
        v = resolve_version(prompt, unique_id)
        stem = (with_version(base, v["version_string"]) if base else os.path.basename(v["filename_prefix"])) or "video"
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
