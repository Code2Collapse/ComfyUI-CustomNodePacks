"""L2.35: slot double-click works in Nodes 2.0, where slots are DOM elements the canvas listeners never see.

Live proof: docs/evidence/L2.35/scripts/slot_dblclick_probe.py (both renderers; --before replays the pre-fix files:
Get/Set and Suggest did nothing in Nodes 2.0).
"""
from __future__ import annotations

from pathlib import Path

JS = Path(__file__).resolve().parents[1] / "js"


def test_shared_module_resolves_dom_slots_by_slot_key():
    src = (JS / "_c2c_slot_dblclick.js").read_text(encoding="utf-8")
    assert "export function onDomSlotDoubleClick" in src
    assert 'closest(".lg-slot")' in src and "data-slot-key" in src
    assert 'document.addEventListener("dblclick", _onDomDblClick, true)' in src
    assert "globalThis.LiteGraph?.vueNodesMode" in src, "classic keeps its canvas listeners; no double handling"


def test_both_features_register_and_respect_the_one_setting():
    getset = (JS / "c2c_slot_getset.js").read_text(encoding="utf-8")
    suggest = (JS / "c2c_autoconnect.js").read_text(encoding="utf-8")
    assert "onDomSlotDoubleClick(" in getset and "slotDoubleClickMode() !== DBL_GETSET" in getset
    assert "onDomSlotDoubleClick(" in suggest and "slotDoubleClickMode() !== DBL_SUGGEST" in suggest
