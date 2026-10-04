"""Static hygiene checks on C2C front-end sources across our packs."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

WORKSPACE = Path(__file__).resolve().parents[2]

PACK_JS_DIRS = {
    "ComfyUI-CustomNodePacks": WORKSPACE / "ComfyUI-CustomNodePacks" / "js",
    "ComfyUI-NukeMaxNodes": WORKSPACE / "ComfyUI-NukeMaxNodes" / "web",
    "ComfyUI-WanNodeExperiments": WORKSPACE / "ComfyUI-WanNodeExperiments" / "web",
    "ComfyUI-MiniMaxSuite": WORKSPACE / "ComfyUI-MiniMaxSuite" / "web",
    "ComfyUI-WanAnimatePreprocessV2": WORKSPACE / "ComfyUI-WanAnimatePreprocessV2" / "js",
    "ComfyUI-GLM_Image": WORKSPACE / "ComfyUI-GLM_Image" / "web",
    "ComfyUI-WanAnimalPreprocessor": WORKSPACE / "ComfyUI-WanAnimalPreprocessor" / "web",
}

KEYFRAMES_RE = re.compile(r"@keyframes\s+([a-zA-Z0-9_-]+)")
# The damage this guards against (found 2026-09-30 in 17 files): newlines
# stripped from a block of // comments, so several comment lines - and the
# import statement after them - became ONE comment line:
#   // ...at// module-eval time. ...See _c2c_lite.js.import { LITE } from "./_c2c_lite.js";
# Signatures: a // comment line containing another "//" glued to text, and a
# // comment line ending in an import statement GLUED to the text before it
# ("...See _c2c_lite.js.import {"). A usage example in a doc comment
# ("//   import { x } from './x.js';") has whitespace there and is fine.
JOINED_COMMENT_RE = re.compile(r"^\s*//.*[A-Za-z0-9.,;)]//\s")
IMPORT_IN_COMMENT_RE = re.compile(
    r"^\s*//.*[^\s/]import\s*(?:\{[^}]*\}\s*from\s*|\*\s+as\s+\w+\s+from\s*|\w+\s+from\s*)?[\"'][^\"']+\.js[\"']\s*;?\s*$")
VENDORED_SKIP = re.compile(r"three[^/\\]*\.js$", re.I)
SETINTERVAL_RE = re.compile(r"\bsetInterval\s*\(")
ALLOW_INTERVAL_RE = re.compile(r"c2c-allow-interval:")
NODES2_PATHS = [
    WORKSPACE / "ComfyUI-CustomNodePacks" / "js" / "c2c_ui" / "nodes2.js",
    WORKSPACE / "ComfyUI-NukeMaxNodes" / "web" / "c2c_ui" / "nodes2.js",
    WORKSPACE / "ComfyUI-WanNodeExperiments" / "web" / "c2c_ui" / "nodes2.js",
    WORKSPACE / "ComfyUI-MiniMaxSuite" / "web" / "c2c_ui" / "nodes2.js",
    WORKSPACE / "ComfyUI-WanAnimatePreprocessV2" / "js" / "c2c_ui" / "nodes2.js",
    WORKSPACE / "ComfyUI-GLM_Image" / "web" / "c2c_ui" / "nodes2.js",
    WORKSPACE / "ComfyUI-WanAnimalPreprocessor" / "web" / "c2c_ui" / "nodes2.js",
]
SERIALIZE_IN_INTERVAL_RE = re.compile(
    r"setInterval\s*\([^)]*serialize\s*\(",
    re.MULTILINE | re.DOTALL,
)
LEGACY_MENU_PATCH_RE = re.compile(
    r"safePatch\s*\([^,]+,\s*[\"']get(?:Canvas|Node)MenuOptions[\"']",
)


def _js_files() -> list[Path]:
    out: list[Path] = []
    for root in PACK_JS_DIRS.values():
        if not root.parent.exists():
            continue
        for p in root.rglob("*.js"):
            if any(part in p.parts for part in ("node_modules", "third_party", "playwright")):
                continue
            if VENDORED_SKIP.search(p.name):
                continue
            out.append(p)
    return out


def _cnp_js_files() -> list[Path]:
    root = PACK_JS_DIRS["ComfyUI-CustomNodePacks"]
    if not root.parent.exists():
        return []
    out: list[Path] = []
    for p in root.rglob("*.js"):
        if any(part in p.parts for part in ("node_modules", "third_party", "playwright")):
            continue
        if VENDORED_SKIP.search(p.name):
            continue
        out.append(p)
    return out


@pytest.mark.parametrize("path", _js_files(), ids=lambda p: str(p.relative_to(WORKSPACE)))
def test_keyframes_use_c2c_or_mec_prefix(path: Path):
    text = path.read_text(encoding="utf-8")
    bad = [m.group(1) for m in KEYFRAMES_RE.finditer(text)
           if not re.match(r"^(c2c|mec)-", m.group(1))]
    assert not bad, f"{path}: @keyframes must start with c2c- or mec-: {bad}"


@pytest.mark.parametrize("path", _js_files(), ids=lambda p: str(p.relative_to(WORKSPACE)))
def test_no_comment_swallowed_code(path: Path):
    text = path.read_text(encoding="utf-8")
    for i, line in enumerate(text.splitlines(), 1):
        probe = line.replace("://", ":__")      # URLs are not joined comments
        if IMPORT_IN_COMMENT_RE.search(probe):
            pytest.fail(f"{path}:{i}: an import statement is inside a // comment: {line.strip()[:120]!r}")
        if JOINED_COMMENT_RE.search(probe):
            pytest.fail(f"{path}:{i}: comment lines were joined (newlines lost): {line.strip()[:120]!r}")


def test_brand_copies_identical():
    import importlib.util
    spec = importlib.util.spec_from_file_location("_c2c_test_brand", Path(__file__).with_name("test_brand.py"))
    tb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tb)
    CANONICAL, COPIES, _copy = tb.CANONICAL, tb.COPIES, tb._copy

    canonical = CANONICAL.read_text(encoding="utf-8")
    for pack in sorted(COPIES):
        path = _copy(pack)
        if not path.parent.parent.exists():
            pytest.skip(f"{pack} not in workspace")
        assert path.read_text(encoding="utf-8") == canonical, f"{pack} brand drift"


def test_nodes2_copies_identical():
    canonical = NODES2_PATHS[0]
    if not canonical.exists():
        pytest.skip("canonical nodes2.js missing")
    text = canonical.read_text(encoding="utf-8")
    for path in NODES2_PATHS[1:]:
        if not path.parent.parent.exists():
            pytest.skip(f"{path.parent.parent.name} not in workspace")
        assert path.read_text(encoding="utf-8") == text, f"nodes2 drift: {path.relative_to(WORKSPACE)}"


@pytest.mark.parametrize("path", _cnp_js_files(), ids=lambda p: str(p.relative_to(WORKSPACE)))
def test_setinterval_has_allow_comment_or_runtime(path: Path):
    """Recurring polls use runtime.every(); boot waits annotate c2c-allow-interval."""
    rel = str(path.relative_to(WORKSPACE))
    if rel.endswith("_c2c_runtime.js"):
        return
    text = path.read_text(encoding="utf-8")
    if "getRuntime().every(" in text and "setInterval" not in text:
        return
    for i, line in enumerate(text.splitlines(), 1):
        if SETINTERVAL_RE.search(line) and not ALLOW_INTERVAL_RE.search(line):
            pytest.fail(
                f"{path}:{i}: setInterval without c2c-allow-interval annotation — "
                f"use getRuntime().every() or annotate boot wait: {line.strip()[:100]!r}"
            )


@pytest.mark.parametrize("path", _cnp_js_files(), ids=lambda p: str(p.relative_to(WORKSPACE)))
def test_no_serialize_inside_setinterval(path: Path):
    text = path.read_text(encoding="utf-8")
    if SERIALIZE_IN_INTERVAL_RE.search(text):
        pytest.fail(f"{path}: graph.serialize() inside setInterval — use onGraphChange")


@pytest.mark.parametrize("path", _js_files(), ids=lambda p: str(p.relative_to(WORKSPACE)))
def test_menu_patches_use_compat_layer(path: Path):
    """Direct menu safePatch outside _c2c_compat.js is banned (B-2)."""
    if path.name == "_c2c_compat.js":
        return
    text = path.read_text(encoding="utf-8")
    if LEGACY_MENU_PATCH_RE.search(text):
        pytest.fail(f"{path}: menu patch must go through _c2c_compat.js legacy*Menu helpers")


def test_wan_report_copies_identical():
    a = WORKSPACE / "ComfyUI-WanAnimatePreprocessV2" / "js" / "_c2c_report.js"
    b = WORKSPACE / "ComfyUI-WanNodeExperiments" / "web" / "js" / "_c2c_report.js"
    if not a.exists() or not b.exists():
        pytest.skip("Wan report copies not present")
    assert a.read_text(encoding="utf-8") == b.read_text(encoding="utf-8")
