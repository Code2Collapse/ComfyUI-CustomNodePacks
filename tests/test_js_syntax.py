"""Every shipped .js must PARSE as an ES module.

This exists because js/universal_reroute.js shipped with a static import
spliced into the middle of an `import().then().catch()` chain, which orphaned
the `.then` and made the whole file a SyntaxError ("Unexpected token '.'").
The browser silently dropped that extension - the Universal Reroute node had no
frontend at all - and nothing in the Python test suite could see it, because
Python never parses our JavaScript.

Node is used as the parser when present. It is the same engine the browser
uses, so it cannot disagree with production about what is valid syntax.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

PACK = Path(__file__).resolve().parents[1]
JS_DIRS = [PACK / "js"]

NODE = shutil.which("node")


def _js_files():
    out = []
    for d in JS_DIRS:
        if not d.is_dir():
            continue
        for f in sorted(d.rglob("*.js")):
            if "node_modules" in f.parts or "third_party" in f.parts:
                continue
            out.append(f)
    return out


@pytest.mark.skipif(NODE is None, reason="node is not installed; cannot parse JS")
def test_every_shipped_js_parses_as_an_es_module():
    # INVARIANT: a file that does not parse is not "degraded", it is ABSENT -
    # the browser drops the whole module, so the node loses its entire UI.
    files = _js_files()
    assert files, "no JS found - the glob is wrong, not the pack"

    failures = []
    with tempfile.TemporaryDirectory() as td:
        probe = Path(td) / "probe.mjs"      # .mjs so node parses it as a module
        for f in files:
            probe.write_bytes(f.read_bytes())
            r = subprocess.run([NODE, "--check", str(probe)],
                               capture_output=True, text=True, timeout=60)
            if r.returncode != 0:
                first = next((ln for ln in r.stderr.splitlines()
                              if "Error" in ln or "^" not in ln and ln.strip()), "")
                failures.append(f"{f.relative_to(PACK)}: {first.strip()[:160]}")

    NL = chr(10) + "  "
    assert not failures, "JavaScript that will not load in the browser:" + NL + NL.join(failures)


@pytest.mark.skipif(NODE is None, reason="node is not installed; cannot parse JS")
def test_the_universal_reroute_regression_specifically():
    # INVARIANT: pins the exact shape of the original defect - a static import
    # sitting between a dynamic import() and the .then() that consumes it.
    src = (PACK / "js" / "universal_reroute.js").read_text(encoding="utf-8")
    lines = [ln.strip() for ln in src.splitlines()]
    for i, ln in enumerate(lines[:-1]):
        if ln.startswith("import ") and " from " in ln:
            nxt = lines[i + 1]
            assert not nxt.startswith((".then", ".catch")), (
                f"line {i + 2} starts with {nxt[:20]!r} directly after a static import - "
                "an expression has been split by an inserted import statement"
            )
