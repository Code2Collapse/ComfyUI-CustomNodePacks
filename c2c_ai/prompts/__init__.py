"""c2c_ai.prompts — versioned Jinja2 system prompts for AI features.

All system prompts used by C2C AI features live as ``.j2`` template files
under ``c2c_ai/prompts/templates/``. This module:

  • Loads templates by canonical name (no inline prompt strings anywhere
    in the codebase — JS or Python).
  • Renders them with optional Jinja2 (full feature set) or a tiny
    stdlib ``{{ var }}`` interpolator when Jinja2 isn't installed.
  • Tracks per-template version + golden SHA-256 (computed against the
    rendered output with default variables) in ``MANIFEST.json``. The
    accompanying regression test ``tests/test_prompt_golden.py`` asserts
    that the live render still matches the frozen golden — any
    accidental prompt drift fails CI.

Public surface (stable, used by api_routes.py and JS via REST):

    list_templates() -> list[dict]      # name, version, sha256, vars
    has(name) -> bool
    load(name) -> str                   # raw template text
    render(name, **vars) -> str         # interpolated text
    golden_sha256(name) -> str          # hash of default render
    verify_goldens() -> list[dict]      # per-template ok/got/want
    freeze_goldens() -> dict            # rewrites MANIFEST.json (dev tool)
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
from pathlib import Path
from typing import Any

__all__ = [
    "list_templates",
    "has",
    "load",
    "render",
    "golden_sha256",
    "verify_goldens",
    "freeze_goldens",
    "TEMPLATES_DIR",
    "MANIFEST_PATH",
]

_HERE = Path(__file__).resolve().parent
TEMPLATES_DIR = _HERE / "templates"
MANIFEST_PATH = _HERE / "MANIFEST.json"

_lock = threading.Lock()
_manifest_cache: dict[str, Any] | None = None
_template_cache: dict[str, str] = {}

# -------- Jinja2 detection --------------------------------------------------

try:                                            # pragma: no cover - env-dep
    from jinja2 import Environment, FileSystemLoader, StrictUndefined, select_autoescape
    _HAS_JINJA2 = True
    _env = Environment(
        loader=FileSystemLoader(str(TEMPLATES_DIR)),
        undefined=StrictUndefined,
        autoescape=select_autoescape(disabled_extensions=("j2",), default=False),
        keep_trailing_newline=True,
    )
except Exception:                               # pragma: no cover - env-dep
    _HAS_JINJA2 = False
    _env = None


# -------- manifest ----------------------------------------------------------

def _load_manifest() -> dict[str, Any]:
    global _manifest_cache
    with _lock:
        if _manifest_cache is not None:
            return _manifest_cache
        if not MANIFEST_PATH.is_file():
            _manifest_cache = {"version": 1, "templates": {}}
        else:
            try:
                _manifest_cache = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
            except Exception as exc:
                raise RuntimeError(f"prompt MANIFEST.json unreadable: {exc}") from exc
        return _manifest_cache


def _save_manifest(manifest: dict[str, Any]) -> None:
    global _manifest_cache
    with _lock:
        MANIFEST_PATH.write_text(
            json.dumps(manifest, indent=2, sort_keys=False) + "\n",
            encoding="utf-8",
        )
        _manifest_cache = manifest


def _manifest_for(name: str) -> dict[str, Any]:
    m = _load_manifest()
    entry = m.get("templates", {}).get(name)
    if entry is None:
        # Unknown template — return minimal default
        return {"version": "0.0.0", "vars": [], "default_vars": {}, "golden_sha256": ""}
    return entry


# -------- discovery + loading ----------------------------------------------

_VALID_NAME = re.compile(r"^[a-zA-Z0-9_][a-zA-Z0-9_.-]{0,127}$")

def _resolve_path(name: str) -> Path:
    if not _VALID_NAME.match(name):
        raise ValueError(f"invalid template name: {name!r}")
    # Allow either bare name (looked up as <name>.j2) or full filename.
    candidates = [TEMPLATES_DIR / f"{name}.j2", TEMPLATES_DIR / name]
    for p in candidates:
        try:
            p_res = p.resolve()
        except Exception:
            continue
        # Defence-in-depth: prevent path traversal even though regex blocks it.
        if TEMPLATES_DIR.resolve() not in p_res.parents and p_res.parent != TEMPLATES_DIR.resolve():
            raise ValueError(f"template path escapes templates dir: {name!r}")
        if p_res.is_file():
            return p_res
    raise FileNotFoundError(f"prompt template not found: {name!r}")


def has(name: str) -> bool:
    try:
        _resolve_path(name)
        return True
    except (FileNotFoundError, ValueError):
        return False


def load(name: str) -> str:
    """Return the raw template text (no rendering)."""
    cached = _template_cache.get(name)
    if cached is not None:
        return cached
    p = _resolve_path(name)
    txt = p.read_text(encoding="utf-8")
    _template_cache[name] = txt
    return txt


# -------- stdlib mini-renderer (fallback when Jinja2 absent) ---------------

_VAR_RE = re.compile(r"\{\{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\}\}")

def _stdlib_render(text: str, variables: dict[str, Any]) -> str:
    missing: list[str] = []
    def _sub(m: re.Match[str]) -> str:
        key = m.group(1)
        if key not in variables:
            missing.append(key)
            return ""
        v = variables[key]
        return "" if v is None else str(v)
    rendered = _VAR_RE.sub(_sub, text)
    if missing:
        raise KeyError(f"prompt template missing variables: {sorted(set(missing))}")
    return rendered


# -------- render ------------------------------------------------------------

def render(name: str, **variables: Any) -> str:
    """Render a template with the given variables.

    Strict: passing a var not declared in the manifest, or omitting a
    declared var (without a default), raises.
    """
    entry = _manifest_for(name)
    declared = set(entry.get("vars") or [])
    defaults = dict(entry.get("default_vars") or {})

    extras = set(variables) - declared
    if declared and extras:
        raise ValueError(
            f"render({name!r}): unknown variables {sorted(extras)} "
            f"(declared in MANIFEST: {sorted(declared)})"
        )

    effective: dict[str, Any] = {}
    for k in declared:
        if k in variables:
            effective[k] = variables[k]
        elif k in defaults:
            effective[k] = defaults[k]
        else:
            raise KeyError(
                f"render({name!r}): required variable {k!r} not supplied "
                f"and no default in MANIFEST"
            )

    text = load(name)
    if _HAS_JINJA2:
        tpl = _env.from_string(text)
        return tpl.render(**effective)
    return _stdlib_render(text, effective)


# -------- golden hashes -----------------------------------------------------

def _sha256(s: str) -> str:
    return "sha256:" + hashlib.sha256(s.encode("utf-8")).hexdigest()


def golden_sha256(name: str) -> str:
    """Hash of the template rendered with its declared default variables."""
    entry = _manifest_for(name)
    defaults = dict(entry.get("default_vars") or {})
    declared = set(entry.get("vars") or [])
    # Render with defaults; any declared-without-default fails loudly.
    for k in declared:
        if k not in defaults:
            raise KeyError(
                f"golden_sha256({name!r}): declared var {k!r} has no default"
            )
    out = render(name, **defaults) if declared else render(name)
    return _sha256(out)


def list_templates() -> list[dict[str, Any]]:
    """Enumerate every .j2 file on disk + its manifest entry."""
    if not TEMPLATES_DIR.is_dir():
        return []
    out: list[dict[str, Any]] = []
    for p in sorted(TEMPLATES_DIR.glob("*.j2")):
        name = p.stem
        entry = _manifest_for(name)
        out.append({
            "name": name,
            "version": entry.get("version", "0.0.0"),
            "vars": list(entry.get("vars") or []),
            "default_vars": dict(entry.get("default_vars") or {}),
            "golden_sha256": entry.get("golden_sha256", ""),
            "file": p.name,
            "bytes": p.stat().st_size,
        })
    return out


def verify_goldens() -> list[dict[str, Any]]:
    """Recompute every golden hash and compare to MANIFEST. Returns a list
    of ``{name, ok, want, got}`` records — caller asserts ``all(r['ok'])``."""
    results: list[dict[str, Any]] = []
    for tpl in list_templates():
        name = tpl["name"]
        want = tpl["golden_sha256"]
        try:
            got = golden_sha256(name)
            ok = (want == got) if want else False
        except Exception as exc:
            got = f"ERR: {exc}"
            ok = False
        results.append({"name": name, "ok": ok, "want": want, "got": got})
    return results


def freeze_goldens() -> dict[str, Any]:
    """DEV TOOL: recompute all golden hashes and rewrite MANIFEST.json.

    Run this only after deliberately editing a prompt — it makes the
    new render the new golden. Returns the manifest that was written.
    """
    manifest = _load_manifest()
    manifest.setdefault("version", 1)
    manifest.setdefault("templates", {})
    for tpl in list_templates():
        name = tpl["name"]
        entry = manifest["templates"].setdefault(name, {})
        entry.setdefault("version", "1.0.0")
        entry.setdefault("vars", [])
        entry.setdefault("default_vars", {})
        try:
            entry["golden_sha256"] = golden_sha256(name)
        except Exception as exc:
            entry["golden_sha256"] = f"ERR: {exc}"
    _save_manifest(manifest)
    # Bust caches so subsequent reads reload.
    global _manifest_cache, _template_cache
    _manifest_cache = manifest
    _template_cache.clear()
    return manifest


# Reset cache when imported (so tests / freeze cycles see fresh state).
_manifest_cache = None
