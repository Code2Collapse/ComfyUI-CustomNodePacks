"""c2c_auto_connector.js (L2.39, owner A9 2026-10-09: "Nearest node, exact types" + "Every matching input").

The live proof is docs/evidence/L2.38/scripts/auto_connector_probe.py (8 cases, both renderers). These pin the rules.
"""
from __future__ import annotations

from pathlib import Path

JS = (Path(__file__).resolve().parents[1] / "js" / "c2c_auto_connector.js").read_text(encoding="utf-8")


def _fn(name: str) -> str:
    a = JS.index(f"function {name}(")
    return JS[a:JS.index("\nfunction ", a + 10) if "\nfunction " in JS[a + 10:] else len(JS)]


def test_exact_types_only_never_wildcards_or_widget_inputs():
    body = _fn("wireable")
    assert "!inp.widget" in body and 'inp.type !== "*"' in body
    assert "o.type === inp.type" in _fn("plan"), "exact type strings, no comma-list or '*' matching"


def test_first_input_per_type_and_exactly_one_output():
    body = _fn("plan")
    assert "seen.has(inp.type)" in body
    assert "outs.length === 1" in body


def test_target_is_the_nearest_upstream_node_not_the_last_clicked():
    assert "_lastNode" not in JS and "onNodeSelected" not in JS
    assert "if (right > left + 20) continue;" in _fn("candidates")


def test_ties_and_core_wired_nodes_are_left_alone_and_wires_are_one_undo_step():
    body = _fn("autoWire")
    assert "TIE_RATIO" in body and "alreadyWired(newNode)" in body and "asOneUndoStep(" in body
    assert "(node.outputs || []).some((o) => o.links?.length)" in _fn("alreadyWired"), \
        "a node created from a wire dragged out of an INPUT is wired on its output side by core"


def test_batch_adds_share_one_counter_by_reference():
    assert "const tick = _tick;" in JS and "if (tick.count > 1) return;" in JS
