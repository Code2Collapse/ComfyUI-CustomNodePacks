"""README and NODE_REFERENCE node counts must match live registration."""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
HELPER = REPO / "tests" / "_load_summary.py"
MARKER_BEGIN = "<!-- C2C:NODE-COUNTS:BEGIN -->"
MARKER_END = "<!-- C2C:NODE-COUNTS:END -->"


def _helper():
    spec = importlib.util.spec_from_file_location("_load_summary", HELPER)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_render_counts_block():
    return _helper().render_counts_block


def test_write_docs_is_idempotent_and_keeps_both_markers(tmp_path):
    """The first version dropped the END marker, so a second --write-docs failed."""
    mod = _helper()
    doc = tmp_path / "doc.md"
    doc.write_text(f"before\n{MARKER_BEGIN}\n{MARKER_END}\nafter\n", encoding="utf-8")
    block = mod.render_counts_block({"total": 2, "families": [["A", 2]], "failed": []})
    mod._replace_marker_block(doc, block)
    first = doc.read_text(encoding="utf-8")
    mod._replace_marker_block(doc, block)
    assert doc.read_text(encoding="utf-8") == first
    assert first.count(MARKER_BEGIN) == 1 and first.count(MARKER_END) == 1
    assert first.startswith("before\n") and first.endswith("after\n")


def _extract_marker_block(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    begin = text.find(MARKER_BEGIN)
    end = text.find(MARKER_END)
    assert begin >= 0 and end >= 0 and end > begin, f"Marker pair missing in {path}"
    inner = text[begin + len(MARKER_BEGIN) : end]
    return inner.strip("\n")


def _run_load_summary() -> dict:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    proc = subprocess.run(
        [sys.executable, str(HELPER)],
        cwd=str(REPO),
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=600,
        env=env,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"_load_summary.py failed (exit {proc.returncode}):\n"
            f"{proc.stdout}\n{proc.stderr}"
        )
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith("C2C_LOAD_SUMMARY_JSON="):
            return json.loads(line.split("=", 1)[1])
    raise RuntimeError("_load_summary.py did not emit C2C_LOAD_SUMMARY_JSON= line")


def test_node_counts_match_docs():
    summary = _run_load_summary()
    if summary.get("failed"):
        details = "; ".join(
            f"{rec['key']} ({rec['error']})" for rec in summary["failed"]
        )
        pytest.fail(
            "Registry reported import failures; this environment cannot verify "
            f"the documented counts: {details}"
        )

    # --write-docs stores block.rstrip() between the markers; compare like with like
    expected = _load_render_counts_block()(summary).strip("\n")
    readme_block = _extract_marker_block(REPO / "README.md")
    ref_block = _extract_marker_block(REPO / "NODE_REFERENCE.md")
    fix = "run `python tests/_load_summary.py --write-docs`"

    assert readme_block == expected, f"README.md node-count block drift; {fix}"
    assert ref_block == expected, f"NODE_REFERENCE.md node-count block drift; {fix}"
