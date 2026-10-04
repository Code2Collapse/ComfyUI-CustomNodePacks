"""_c2c_compat.js — static contract checks."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

WORKSPACE = Path(__file__).resolve().parents[2]
COMPAT = Path(__file__).resolve().parents[1] / "js" / "_c2c_compat.js"
NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="node not on PATH")

_EXPORTS = (
    "hasMenuHooks",
    "legacyCanvasMenu",
    "legacyNodeMenu",
    "graphReadable",
    "onGraphRead",
)


def test_compat_exports_present():
    text = COMPAT.read_text(encoding="utf-8")
    for name in _EXPORTS:
        assert re.search(rf"export function {name}\b", text), f"missing export: {name}"


def test_compat_graph_readable_checks_configuring():
    text = COMPAT.read_text(encoding="utf-8")
    assert "_configuring" in text
    assert "graphReadable" in text


def test_compat_legacy_menu_skips_when_hooks_exist():
    text = COMPAT.read_text(encoding="utf-8")
    assert "hasMenuHooks()" in text
    assert "safePatch" in text


@pytest.mark.parametrize("path", [COMPAT], ids=["_c2c_compat.js"])
def test_compat_passes_node_check(path: Path):
    r = subprocess.run(
        [NODE, "--check", str(path)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert r.returncode == 0, r.stderr or r.stdout
