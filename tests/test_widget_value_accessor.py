"""L2.43: no front-end script may REPLACE a widget's `value` accessor with its own copy.

Frontend 1.53 defines `get value() { return this._state.value }` - the widget-value store entry Nodes 2.0 draws.
parameter_memory.js replaced it with a closure (`get() { return _v }`), so after a saved workflow loaded the node
held the saved values but Nodes 2.0 showed the defaults (docs/evidence/L2.43, measured with every C2C pack: only CNP,
only that file). A redefinition must call the original getter / setter it found.
"""
import re
from pathlib import Path

JS = Path(__file__).resolve().parents[1] / "js"
DEFINE = re.compile(r"Object\.defineProperty\(\s*([A-Za-z_$][\w$.]*)\s*,\s*[\"']value[\"']\s*,\s*\{")


def test_value_redefinitions_wrap_the_original_accessor():
    offenders = []
    for f in sorted(JS.rglob("*.js")):
        src = f.read_text(encoding="utf-8", errors="replace")
        for m in DEFINE.finditer(src):
            block = src[m.end(): m.end() + 900]
            before = src[max(0, m.start() - 1500): m.start()]
            wraps = "getOwnPropertyDescriptor" in before and re.search(r"\borig(Get|Set)\b|\bdesc\.(get|set)\b", before + block)
            if not wraps:
                offenders.append(f"{f.relative_to(JS.parent)}:{src.count(chr(10), 0, m.start()) + 1}")
    assert not offenders, "widget value accessor replaced instead of wrapped: " + ", ".join(offenders)


def test_parameter_memory_calls_the_core_accessor():
    src = (JS / "parameter_memory.js").read_text(encoding="utf-8")
    assert "origGet.call(self)" in src and "origSet.call(this, v)" in src
    assert "get() { return _v; }" not in src
