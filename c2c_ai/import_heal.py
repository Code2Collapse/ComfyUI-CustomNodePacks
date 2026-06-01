"""Phase E.1 — Failed-import harvester.

Discovers custom-node packs that failed to import (typically due to missing
Python dependencies), classifies the missing requirements via the existing
``dependency_checker.check_conflicts`` triage, and emits a structured
``FailedPack`` per pack so the UI layer (E.3) can render heal buttons.

Discovery strategy
------------------
1. **Preferred** — read ``nodes.FAILED_IMPORT_MODULES`` if ComfyUI exposes it.
2. **Fallback** — re-attempt import of every immediate child of every
   ``custom_nodes`` folder that is NOT already present in ``sys.modules``
   as ``custom_nodes.<name>``. Capture the resulting ``ImportError`` /
   ``ModuleNotFoundError`` traceback and missing module names.

This module is import-only — it does NOT install anything (E.2 owns that)
and it does NOT register routes (E.3 wires the route to E.2).

Safety
------
* Never imports or evaluates pack code beyond ``importlib.import_module``.
* Defensive try/except around every external call. A single broken pack
  must never crash the harvester.
* No I/O outside the workspace except ``importlib.metadata`` queries.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import os
import re
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger("c2c_ai.import_heal")

# ---------------------------------------------------------------------------
# Public dataclass
# ---------------------------------------------------------------------------

# Recommendation buckets, ordered safest → most-destructive.
ACTION_AUTO_SAFE = "auto_safe"          # 0 breaking, 0 risky — install silently
ACTION_NEEDS_REVIEW = "needs_review"    # has risky entries — needs user "OK"
ACTION_BLOCKED = "blocked"              # any breaking — never auto-install
ACTION_UNKNOWN = "unknown"              # no requirements.txt / cannot triage


@dataclass
class FailedPack:
    """A custom-node pack that failed to import.

    Attributes
    ----------
    name : str
        Pack folder name (e.g. ``"ComfyUI-OmniVoice"``).
    pack_path : str
        Absolute path to the pack folder.
    req_path : Optional[str]
        Absolute path to ``requirements.txt`` if one exists.
    missing_modules : List[str]
        Python import names parsed from ``ModuleNotFoundError`` messages
        in the traceback (e.g. ``["gradio", "librosa"]``).
    error_summary : str
        First line of the captured exception, truncated to 240 chars.
    traceback_tail : str
        Last ~12 lines of the traceback, useful for the Doctor UI.
    report : Optional[dict]
        ``ConflictReport.to_dict()`` from ``dependency_checker``; ``None``
        if no requirements file or triage failed.
    recommended_action : str
        One of ``ACTION_AUTO_SAFE`` / ``ACTION_NEEDS_REVIEW`` /
        ``ACTION_BLOCKED`` / ``ACTION_UNKNOWN``.
    safe_to_install : List[str]
        Missing module names that fall in ``report.safe`` (or all missing
        modules when ``recommended_action == ACTION_UNKNOWN``).
    """
    name: str
    pack_path: str
    req_path: Optional[str]
    missing_modules: List[str] = field(default_factory=list)
    error_summary: str = ""
    traceback_tail: str = ""
    report: Optional[dict] = None
    recommended_action: str = ACTION_UNKNOWN
    safe_to_install: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "pack_path": self.pack_path,
            "req_path": self.req_path,
            "missing_modules": list(self.missing_modules),
            "error_summary": self.error_summary,
            "traceback_tail": self.traceback_tail,
            "report": self.report,
            "recommended_action": self.recommended_action,
            "safe_to_install": list(self.safe_to_install),
        }


# ---------------------------------------------------------------------------
# Pack discovery
# ---------------------------------------------------------------------------

_MNF_RE = re.compile(r"No module named ['\"]([A-Za-z0-9_.\-]+)['\"]")


def _custom_nodes_dirs() -> List[Path]:
    """Return every ``custom_nodes`` root configured in ComfyUI.

    Falls back to a single dir relative to this file when ``folder_paths``
    is unavailable (e.g. when this module is imported in a unit test).
    """
    try:
        import folder_paths  # type: ignore
        roots = folder_paths.get_folder_paths("custom_nodes")
        return [Path(r) for r in roots if r and os.path.isdir(r)]
    except Exception:
        # Walk up from .../ComfyUI/custom_nodes/ComfyUI-CustomNodePacks/c2c_ai/
        here = Path(__file__).resolve()
        for parent in here.parents:
            if parent.name == "custom_nodes":
                return [parent]
        return []


def _find_requirements(pack_dir: Path) -> Optional[Path]:
    """Locate ``requirements.txt`` (case-insensitive) in the pack root."""
    if not pack_dir.is_dir():
        return None
    for entry in pack_dir.iterdir():
        if entry.is_file() and entry.name.lower() == "requirements.txt":
            return entry
    return None


def _parse_missing_modules(tb_text: str) -> List[str]:
    """Extract every ``No module named 'X'`` import name from a traceback."""
    seen: List[str] = []
    for m in _MNF_RE.finditer(tb_text or ""):
        name = m.group(1).split(".")[0]
        if name and name not in seen:
            seen.append(name)
    return seen


def _try_module_path(pack_dir: Path) -> Optional[Tuple[str, str]]:
    """Re-attempt import of one pack. Returns ``(error_summary, tb_tail)``
    on failure, or ``None`` if import succeeded / was skipped."""
    name = pack_dir.name
    if name.startswith(".") or name.endswith(".disabled") or name == "__pycache__":
        return None

    # Check if it's a package (dir with __init__.py) or a single-file node module.
    init_py = pack_dir / "__init__.py"
    if not init_py.is_file() and not (pack_dir.suffix == ".py" and pack_dir.is_file()):
        return None

    mod_key = f"custom_nodes.{name}"
    if mod_key in sys.modules and sys.modules[mod_key] is not None:
        # Already imported successfully on this interpreter — not failed.
        return None

    try:
        # Locate the spec WITHOUT actually executing it via importlib.import_module.
        # We can't use importlib.import_module because ComfyUI's custom_nodes
        # parent package may not be on sys.path; loading via spec_from_file_location
        # mirrors what ComfyUI itself does and lets us capture the real exception.
        spec = importlib.util.spec_from_file_location(
            mod_key, str(init_py), submodule_search_locations=[str(pack_dir)]
        )
        if spec is None or spec.loader is None:
            return ("could not create import spec", "")
        module = importlib.util.module_from_spec(spec)
        # Don't pollute sys.modules permanently — restore on failure.
        prev = sys.modules.get(mod_key)
        sys.modules[mod_key] = module
        try:
            spec.loader.exec_module(module)  # type: ignore[union-attr]
        except BaseException:
            if prev is None:
                sys.modules.pop(mod_key, None)
            else:
                sys.modules[mod_key] = prev
            raise
        return None
    except SystemExit:
        # Some packs call sys.exit on missing GPU; treat as failure.
        return ("SystemExit during import", "")
    except BaseException as exc:
        tb_text = traceback.format_exc(limit=24)
        tail = "\n".join((tb_text or "").splitlines()[-12:])
        summary = f"{type(exc).__name__}: {exc}".strip()[:240]
        return (summary, tail)


def _comfy_failed_imports() -> Dict[str, Tuple[str, str]]:
    """Try ``nodes.FAILED_IMPORT_MODULES`` (or known equivalents).

    Returns ``{pack_name: (error_summary, traceback_tail)}``. Empty when
    ComfyUI does not expose the attribute (current behavior on portable
    v0.22.0 — see plan.md E.1 fallback note).
    """
    try:
        import nodes as _comfy_nodes  # type: ignore
    except Exception:
        return {}
    attr = getattr(_comfy_nodes, "FAILED_IMPORT_MODULES", None)
    if not attr:
        return {}
    out: Dict[str, Tuple[str, str]] = {}
    try:
        # Tolerate either {name: exc} or [(name, exc), ...]
        items = attr.items() if hasattr(attr, "items") else attr
        for name, exc in items:
            tb_text = ""
            if isinstance(exc, BaseException):
                tb_text = "".join(
                    traceback.format_exception(type(exc), exc, exc.__traceback__)
                )
                summary = f"{type(exc).__name__}: {exc}".strip()[:240]
            else:
                tb_text = str(exc)
                summary = tb_text.splitlines()[0][:240] if tb_text else "unknown"
            tail = "\n".join(tb_text.splitlines()[-12:])
            out[str(name)] = (summary, tail)
    except Exception as exc:  # pragma: no cover — defensive
        log.warning("[import_heal] failed to read FAILED_IMPORT_MODULES: %s", exc)
    return out


# ---------------------------------------------------------------------------
# Triage
# ---------------------------------------------------------------------------

def _triage(pack: FailedPack) -> None:
    """Fill in ``report``, ``recommended_action``, ``safe_to_install`` from
    the missing modules and the pack's requirements file (if any)."""
    if not pack.req_path:
        # No requirements file → we still know the missing modules from the
        # traceback. Without a spec to triage against we cannot guarantee
        # safety, so mark "needs_review" instead of "auto_safe".
        pack.recommended_action = (
            ACTION_NEEDS_REVIEW if pack.missing_modules else ACTION_UNKNOWN
        )
        pack.safe_to_install = list(pack.missing_modules)
        return

    try:
        # Lazy import — dependency_checker lives at pack root, not under c2c_ai/.
        # The package is registered to sys.path by ComfyUI when loading
        # ComfyUI-CustomNodePacks; for unit-test contexts we fall back.
        try:
            from ..dependency_checker import check_conflicts  # type: ignore
        except Exception:
            # Last-resort: try absolute path.
            pack_root = Path(__file__).resolve().parent.parent
            sys.path.insert(0, str(pack_root))
            try:
                from dependency_checker import check_conflicts  # type: ignore
            finally:
                # Don't permanently mutate sys.path.
                try:
                    sys.path.remove(str(pack_root))
                except ValueError:
                    pass
        report = check_conflicts(pack.req_path)
    except Exception as exc:
        log.warning("[import_heal] triage failed for %s: %s", pack.name, exc)
        pack.recommended_action = ACTION_NEEDS_REVIEW
        pack.safe_to_install = list(pack.missing_modules)
        return

    pack.report = report.to_dict()

    # Cross-reference: only install entries the pack ACTUALLY lacks
    # (per ModuleNotFoundError parsing). If we have no missing-module list
    # (FAILED_IMPORT_MODULES path with opaque error), trust the report.
    missing_lc = {m.lower() for m in pack.missing_modules}

    # Known dist-name → import-name aliases. Used as a fallback when the
    # distribution is not installed (so importlib.metadata can't tell us
    # its top-level modules). Keep this short — only well-known mismatches.
    _DIST_TO_IMPORT_ALIASES: Dict[str, Tuple[str, ...]] = {
        "py-cpuinfo": ("cpuinfo",),
        "opencv-python": ("cv2",),
        "opencv-python-headless": ("cv2",),
        "opencv-contrib-python": ("cv2",),
        "pillow": ("pil",),
        "scikit-image": ("skimage",),
        "scikit-learn": ("sklearn",),
        "pyyaml": ("yaml",),
        "protobuf": ("google",),
        "python-dateutil": ("dateutil",),
        "beautifulsoup4": ("bs4",),
        "msgpack-python": ("msgpack",),
        "pycryptodome": ("crypto",),
        "ffmpeg-python": ("ffmpeg",),
        "huggingface-hub": ("huggingface_hub",),
        "sentence-transformers": ("sentence_transformers",),
        "pyturbojpeg": ("turbojpeg",),
        "gitpython": ("git",),
    }

    def _dist_provides_module(pkg: str, module: str) -> bool:
        """Best-effort check: does dist ``pkg`` provide top-level ``module``?

        Strategy: try ``importlib.metadata.packages_distributions()`` (true
        runtime mapping for installed dists), then fall back to a small
        alias table for uninstalled common-mismatch cases, then to
        normalised string equality.
        """
        pkg_lc = pkg.lower().replace("_", "-")
        mod_lc = module.lower()
        # 1. Static normalised match (covers 80%+ of cases).
        if pkg_lc == mod_lc or pkg_lc.replace("-", "") == mod_lc.replace("-", "").replace("_", ""):
            return True
        # 2. importlib.metadata reverse-lookup (only works if installed).
        try:
            import importlib.metadata as _md
            pd = getattr(_md, "packages_distributions", None)
            if pd is not None:
                rev = pd()  # {top_level_import: [dist_names]}
                for tl, dists in rev.items():
                    if tl.lower() != mod_lc:
                        continue
                    if any(d.lower().replace("_", "-") == pkg_lc for d in dists):
                        return True
        except Exception:
            pass
        # 3. Alias table fallback for uninstalled common mismatches.
        for alias in _DIST_TO_IMPORT_ALIASES.get(pkg_lc, ()):
            if alias.lower() == mod_lc:
                return True
        # 4. Heuristic: dist with leading "py-" prefix often provides bare module.
        if pkg_lc.startswith("py-") and pkg_lc[3:] == mod_lc:
            return True
        return False

    def _matches_missing(pkg: str) -> bool:
        if not missing_lc:
            return True
        if any(_dist_provides_module(pkg, m) for m in missing_lc):
            return True
        return False

    # Final defensive filter — never let a VCS-URL artefact reach the
    # installer even if dependency_checker missed one.
    def _is_real_dist(name: str) -> bool:
        bad = {"git", "hg", "svn", "bzr", "http", "https", "file", "ftp", ""}
        return name.lower() not in bad

    safe_entries = [e for e in report.safe
                    if _is_real_dist(e.package) and _matches_missing(e.package)]
    risky_entries = [e for e in report.risky
                     if _is_real_dist(e.package) and _matches_missing(e.package)]
    breaking_entries = [e for e in report.breaking
                        if _is_real_dist(e.package) and _matches_missing(e.package)]

    pack.safe_to_install = [e.package for e in safe_entries]

    if breaking_entries:
        pack.recommended_action = ACTION_BLOCKED
    elif risky_entries:
        pack.recommended_action = ACTION_NEEDS_REVIEW
    elif pack.safe_to_install:
        pack.recommended_action = ACTION_AUTO_SAFE
    else:
        # All requirements are satisfied — the import failure is something
        # other than a missing dep (logic bug, missing model, etc.).
        pack.recommended_action = ACTION_UNKNOWN


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def collect_failed_imports(*, rescan: bool = False) -> List[FailedPack]:
    """Discover and triage every custom-node pack that failed to import.

    Parameters
    ----------
    rescan : bool
        If True, force a fresh re-attempt of every pack even when ComfyUI's
        ``FAILED_IMPORT_MODULES`` claims none failed. Useful for the manual
        "Refresh" button in the Doctor UI.
    """
    failed_raw: Dict[str, Tuple[str, str]] = {}
    if not rescan:
        failed_raw.update(_comfy_failed_imports())

    pack_paths: Dict[str, Path] = {}
    for root in _custom_nodes_dirs():
        for entry in root.iterdir():
            if not entry.is_dir():
                continue
            if entry.name.startswith(".") or entry.name.endswith(".disabled"):
                continue
            if entry.name == "__pycache__":
                continue
            pack_paths.setdefault(entry.name, entry)

    # Fallback discovery: re-attempt import for every pack not already in
    # sys.modules. This catches the (current portable v0.22.0) case where
    # FAILED_IMPORT_MODULES is absent.
    if not failed_raw or rescan:
        for name, pack_dir in pack_paths.items():
            result = _try_module_path(pack_dir)
            if result is not None:
                failed_raw[name] = result

    out: List[FailedPack] = []
    for name, (summary, tail) in failed_raw.items():
        pack_dir = pack_paths.get(name)
        if pack_dir is None:
            # Couldn't locate the pack directory — skip (defensive).
            continue
        req_path = _find_requirements(pack_dir)
        pack = FailedPack(
            name=name,
            pack_path=str(pack_dir),
            req_path=str(req_path) if req_path else None,
            missing_modules=_parse_missing_modules(tail or summary),
            error_summary=summary,
            traceback_tail=tail,
        )
        _triage(pack)
        out.append(pack)

    out.sort(key=lambda p: (
        # Order: auto_safe first (cheapest wins), then needs_review, blocked, unknown.
        {ACTION_AUTO_SAFE: 0, ACTION_NEEDS_REVIEW: 1,
         ACTION_BLOCKED: 2, ACTION_UNKNOWN: 3}.get(p.recommended_action, 4),
        p.name.lower(),
    ))
    return out


__all__ = [
    "FailedPack",
    "ACTION_AUTO_SAFE",
    "ACTION_NEEDS_REVIEW",
    "ACTION_BLOCKED",
    "ACTION_UNKNOWN",
    "collect_failed_imports",
]
