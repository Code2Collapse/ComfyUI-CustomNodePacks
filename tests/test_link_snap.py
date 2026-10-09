"""c2c_link_snap.js (L2.39, owner A9): a wire released anywhere on a node connects to the matching slot.

The live proof is docs/evidence/L2.38/scripts/snap_all_c2c_nodes.py (every C2C node type with an IMAGE input) and
link_snap_exact_probe.py (drops that already fit are unchanged). These pin the design facts that make it work.
"""
from __future__ import annotations

import re
from pathlib import Path

JS = (Path(__file__).resolve().parents[1] / "js" / "c2c_link_snap.js").read_text(encoding="utf-8")


def test_setting_is_declarative_and_on_by_default():
    m = re.search(r'id: SETTING_ID,.*?defaultValue: (\w+)', JS, re.S)
    assert 'const SETTING_ID = "c2c.linkSnap.matchingSlot"' in JS
    assert m and m.group(1) == "true"
    assert "settings.addSetting(" not in JS


def test_drop_falls_back_only_when_the_slot_under_the_pointer_does_not_fit():
    a = JS.index("function patchDrop")
    body = JS[a:JS.index("\nfunction ", a + 10)]
    assert 'safePatch(proto, "dropOnNode"' in body, "dropOnNode is looked up at call time: patch the prototype"
    assert "if (slot && !fits(this, node, slot)) return this.connectToNode(node, event);" in body
    assert "return orig.call(this, node, event);" in body


def test_preview_is_a_listener_not_a_processMouseMove_patch():
    # Core binds processMouseMove to the canvas element at creation; a prototype patch never runs.
    assert '"processMouseMove"' not in JS
    assert 'addEventListener("pointermove"' in JS
    assert "canvas._highlight_pos = node.getInputSlotPos(found.slot);" in JS
