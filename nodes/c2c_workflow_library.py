"""c2c_workflow_library.py - backend for the C2C Workflow Library panel.

Modelled on gregowahoo/comfyui-workflow-finder (MIT): scans one or more
folders of saved ComfyUI workflow .json files, builds a lightweight
"fingerprint" of each (node types, custom titles, text snippets, node count,
created/modified timestamps), and detects which custom-node packages each
workflow requires.

Semantic *scoring* lives on the JS side (js/c2c_node_taxonomy.js) so the
node-capability map has a single source of truth. This backend only does the
filesystem work the browser cannot:

  * GET  /c2c/library/locations          -> configured + auto-detected dirs
  * POST /c2c/library/locations          -> persist the dir list
  * POST /c2c/library/scan               -> fingerprint every .json in the
                                            enabled dirs + class->package map
  * GET  /c2c/library/load?path=...       -> raw workflow JSON for graph
                                            preview (path-validated against
                                            scanned roots; no traversal)

No stubs. Every value is computed from disk. Errors return HTTP 200 with
success=false so the JS renders them instead of choking on fetch failures.

License: Apache-2.0
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


# ---------------------------------------------------------------------------
# Paths / config
# ---------------------------------------------------------------------------
def _comfy_root() -> Optional[Path]:
    """Return the ComfyUI root directory (dir containing main.py)."""
    try:
        import folder_paths  # type: ignore
        bp = getattr(folder_paths, "base_path", None)
        if bp:
            return Path(bp)
    except Exception:
        pass
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        if (parent / "main.py").is_file() and (parent / "comfy").is_dir():
            return parent
    return None


def _user_dir() -> Path:
    """ComfyUI user directory, where we persist the location list."""
    try:
        import folder_paths  # type: ignore
        gud = getattr(folder_paths, "get_user_directory", None)
        if callable(gud):
            return Path(gud())
    except Exception:
        pass
    root = _comfy_root()
    if root:
        return root / "user"
    return Path(__file__).resolve().parent.parent / "user"


def _config_path() -> Path:
    d = _user_dir() / "c2c"
    try:
        d.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    return d / "workflow_library.json"


def _default_dirs() -> List[str]:
    """Auto-detect the standard ComfyUI workflows folder(s)."""
    out: List[str] = []
    root = _comfy_root()
    if root:
        cand = root / "user" / "default" / "workflows"
        if cand.is_dir():
            out.append(str(cand))
    # Any extra workflow roots ComfyUI knows about.
    try:
        import folder_paths  # type: ignore
        gud = getattr(folder_paths, "get_user_directory", None)
        if callable(gud):
            wf = Path(gud()) / "default" / "workflows"
            if wf.is_dir() and str(wf) not in out:
                out.append(str(wf))
    except Exception:
        pass
    return out


def _load_locations() -> List[Dict[str, Any]]:
    """Return [{path, enabled}, ...] - persisted, seeded from defaults."""
    cfg = _config_path()
    try:
        with open(cfg, "r", encoding="utf-8") as f:
            data = json.load(f)
        entries = data.get("directories")
        if isinstance(entries, list) and entries:
            return [
                {"path": str(e.get("path", "")),
                 "enabled": bool(e.get("enabled", True))}
                for e in entries if e.get("path")
            ]
    except FileNotFoundError:
        pass
    except Exception:
        pass
    return [{"path": d, "enabled": True} for d in _default_dirs()]


def _save_locations(entries: List[Dict[str, Any]]) -> None:
    try:
        with open(_config_path(), "w", encoding="utf-8") as f:
            json.dump({"directories": entries}, f, indent=2)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Workflow fingerprinting
# ---------------------------------------------------------------------------
_MAX_JSON_BYTES = 16 * 1024 * 1024  # skip absurdly large files


def extract_workflow_fingerprint(json_path: str) -> Optional[Dict[str, Any]]:
    """Parse a workflow .json into a searchable fingerprint dict.

    Handles both UI-export format (top-level "nodes" array) and API/prompt
    format ({id: {class_type, inputs, _meta}}).
    """
    try:
        if os.path.getsize(json_path) > _MAX_JSON_BYTES:
            return None
        with open(json_path, "r", encoding="utf-8", errors="ignore") as f:
            data = json.load(f)
    except Exception:
        return None

    nodes: List[str] = []
    titles: List[str] = []
    snippets: List[str] = []

    def harvest(ntype: Any, ntitle: Any, wvals: Any) -> None:
        if ntype:
            nodes.append(str(ntype))
        if ntitle and ntitle != ntype:
            titles.append(str(ntitle))
        if isinstance(wvals, dict):
            wvals = list(wvals.values())
        if isinstance(wvals, (list, tuple)):
            for v in wvals:
                if isinstance(v, str) and 15 < len(v) < 600:
                    snippets.append(v[:200])

    if isinstance(data, dict) and isinstance(data.get("nodes"), list):
        for n in data["nodes"]:
            if isinstance(n, dict):
                harvest(n.get("type", ""), n.get("title", ""),
                        n.get("widgets_values", []))
    elif isinstance(data, dict):
        for _id, n in data.items():
            if isinstance(n, dict):
                ct = n.get("class_type", "")
                tl = (n.get("_meta", {}) or {}).get("title", "")
                vals = [v for v in (n.get("inputs", {}) or {}).values()
                        if isinstance(v, str)]
                harvest(ct, tl, vals)

    if not nodes:
        return None

    try:
        st = os.stat(json_path)
        created = st.st_ctime
        modified = st.st_mtime
    except Exception:
        created = modified = 0.0

    # Preserve order while de-duplicating.
    uniq_nodes = list(dict.fromkeys(nodes))
    uniq_titles = list(dict.fromkeys(titles))

    return {
        "path": json_path,
        "filename": Path(json_path).name,
        "nodes": uniq_nodes,
        "titles": uniq_titles,
        "text_snippets": snippets[:6],
        "node_count": len(nodes),
        "created": created,
        "modified": modified,
        "required_packages": [],
    }


# ---------------------------------------------------------------------------
# Custom-node package detection
# ---------------------------------------------------------------------------
def get_install_root(workflow_dir: str) -> Optional[str]:
    """Derive a ComfyUI install root from a workflow folder path.

    Standard layout: [root]/user/default/workflows.
    """
    p = Path(workflow_dir)
    for parent in list(p.parents)[:4]:
        try:
            if (parent / "custom_nodes").is_dir() or (parent / "main.py").is_file():
                return str(parent)
        except Exception:
            pass
    return None


_NCM_RE = re.compile(
    r'NODE_CLASS_MAPPINGS\s*(?:=\s*\{|\.update\s*\(\s*\{)([^}]*)\}',
    re.DOTALL,
)
_KEY_RE = re.compile(r'["\']([A-Za-z][A-Za-z0-9_\- ]*)["\']')


def scan_custom_nodes(install_root: str) -> Dict[str, str]:
    """Return {node_class_name: package_folder} for one ComfyUI install.

    Reads NODE_CLASS_MAPPINGS keys from each package's Python files (init
    first, then up to a handful more).
    """
    cn_dir = os.path.join(install_root, "custom_nodes")
    if not os.path.isdir(cn_dir):
        return {}

    result: Dict[str, str] = {}
    try:
        packages = [p for p in os.listdir(cn_dir)
                    if os.path.isdir(os.path.join(cn_dir, p))
                    and not p.startswith(".")]
    except Exception:
        return {}

    for pkg in packages:
        pkg_path = os.path.join(cn_dir, pkg)
        py_files: List[str] = []
        init = os.path.join(pkg_path, "__init__.py")
        if os.path.isfile(init):
            py_files.append(init)
        try:
            for fn in os.listdir(pkg_path):
                if fn.endswith(".py") and fn != "__init__.py":
                    py_files.append(os.path.join(pkg_path, fn))
                if len(py_files) >= 8:
                    break
        except Exception:
            pass

        found_any = False
        for py_file in py_files:
            try:
                with open(py_file, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                for m in _NCM_RE.finditer(content):
                    for key in _KEY_RE.findall(m.group(1)):
                        if key not in result:
                            result[key] = pkg
                            found_any = True
            except Exception:
                continue
            if found_any:
                break
    return result


def build_class_pkg_map(dir_entries: List[Dict[str, Any]]) -> Dict[str, str]:
    """Unified {class_name: package} across the install roots of all dirs."""
    result: Dict[str, str] = {}
    roots: Set[str] = set()
    for e in dir_entries:
        if not e.get("enabled"):
            continue
        root = get_install_root(e["path"])
        if root and root not in roots:
            roots.add(root)
            result.update(scan_custom_nodes(root))
    return result


# ---------------------------------------------------------------------------
# Scan driver
# ---------------------------------------------------------------------------
# Roots permitted for /load (set on every scan) - prevents path traversal.
_ALLOWED_ROOTS: Set[str] = set()


def _is_allowed(path: str) -> bool:
    try:
        real = os.path.realpath(path)
    except Exception:
        return False
    for root in _ALLOWED_ROOTS:
        try:
            if os.path.commonpath([real, root]) == root:
                return True
        except Exception:
            continue
    return False


def scan_library(dir_entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Fingerprint every .json under each enabled dir; tag required packages."""
    global _ALLOWED_ROOTS
    enabled = [e for e in dir_entries if e.get("enabled") and os.path.isdir(e["path"])]
    _ALLOWED_ROOTS = {os.path.realpath(e["path"]) for e in enabled}

    class_to_pkg = build_class_pkg_map(enabled)

    fingerprints: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    total_files = 0

    for e in enabled:
        for dp, _dirs, files in os.walk(e["path"]):
            for fn in files:
                if not fn.lower().endswith(".json"):
                    continue
                total_files += 1
                full = os.path.join(dp, fn)
                rp = os.path.realpath(full)
                if rp in seen:
                    continue
                fp = extract_workflow_fingerprint(full)
                if not fp:
                    continue
                seen.add(rp)
                pkgs = sorted({class_to_pkg[nt] for nt in fp["nodes"]
                               if nt in class_to_pkg})
                fp["required_packages"] = pkgs
                fp["source_dir"] = e["path"]
                fingerprints.append(fp)

    all_pkgs = sorted({p for fp in fingerprints for p in fp["required_packages"]})
    return {
        "success": True,
        "workflows": fingerprints,
        "workflow_count": len(fingerprints),
        "file_count": total_files,
        "packages": all_pkgs,
        "scanned_at": time.time(),
    }


# ---------------------------------------------------------------------------
# Aiohttp route registration
# ---------------------------------------------------------------------------
_ROUTES_REGISTERED = False


def register_routes(server) -> None:
    global _ROUTES_REGISTERED
    if _ROUTES_REGISTERED or server is None:
        return
    try:
        from aiohttp import web
    except Exception:
        return
    import asyncio
    import functools

    routes = server.routes if hasattr(server, "routes") else server.app.router

    async def _run_blocking(fn, *a, **kw):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, functools.partial(fn, *a, **kw))

    @routes.get("/c2c/library/locations")
    async def _locations(_req):
        entries = _load_locations()
        for e in entries:
            e["exists"] = os.path.isdir(e["path"])
        return web.json_response({"success": True, "directories": entries})

    @routes.post("/c2c/library/locations")
    async def _set_locations(req):
        try:
            body = await req.json()
        except Exception:
            return web.json_response(
                {"success": False, "error": "bad_json"}, status=200)
        raw = body.get("directories", [])
        entries = [
            {"path": os.path.normpath(str(e.get("path", ""))),
             "enabled": bool(e.get("enabled", True))}
            for e in raw if e.get("path")
        ]
        # De-duplicate by path, keep first.
        seen: Set[str] = set()
        dedup = []
        for e in entries:
            if e["path"] not in seen:
                seen.add(e["path"])
                dedup.append(e)
        _save_locations(dedup)
        for e in dedup:
            e["exists"] = os.path.isdir(e["path"])
        return web.json_response({"success": True, "directories": dedup})

    @routes.post("/c2c/library/scan")
    async def _scan(req):
        try:
            body = await req.json()
        except Exception:
            body = {}
        entries = body.get("directories")
        if not isinstance(entries, list) or not entries:
            entries = _load_locations()
        else:
            entries = [
                {"path": str(e.get("path", "")),
                 "enabled": bool(e.get("enabled", True))}
                for e in entries if e.get("path")
            ]
        try:
            result = await _run_blocking(scan_library, entries)
            return web.json_response(result)
        except Exception as exc:
            return web.json_response(
                {"success": False, "error": "exception",
                 "detail": str(exc)[:500]}, status=200)

    @routes.get("/c2c/library/load")
    async def _load(req):
        path = req.query.get("path") or ""
        if not path or not _is_allowed(path):
            return web.json_response(
                {"success": False, "error": "path_not_allowed"}, status=200)
        try:
            if os.path.getsize(path) > _MAX_JSON_BYTES:
                return web.json_response(
                    {"success": False, "error": "file_too_large"}, status=200)
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                data = json.load(f)
            return web.json_response({"success": True, "workflow": data})
        except Exception as exc:
            return web.json_response(
                {"success": False, "error": "exception",
                 "detail": str(exc)[:500]}, status=200)

    _ROUTES_REGISTERED = True
