"""L2.27 (owner A9: "Load Video (C2C) and OmniScale Load Video" - no preview).

Core's upload preview plays the ORIGINAL file in the browser, and Chrome cannot decode ProRes, DNxHR, HEVC 10-bit or
FFV1: those clips showed no preview at all. Both loaders now use the server-transcoded preview (/c2c/video/preview)
and hide core's. Live proof: docs/evidence/L2.27/scripts/loader_switch_probe.py (both renderers).
"""
from __future__ import annotations

import re
from pathlib import Path

JS = (Path(__file__).resolve().parents[1] / "js" / "c2c_video_loader.js").read_text(encoding="utf-8")


def _cfg(name: str) -> str:
    m = re.search(rf"\n    {name}: \{{(.*?)\n    \}},", JS, re.S)
    assert m, f"no LOADERS entry for {name}"
    return m.group(1)


def test_both_upload_loaders_use_the_transcoded_preview_and_hide_cores():
    for name in ("LoadVideoC2C", "OmniScaleLoadVideo"):
        body = _cfg(name)
        assert "hasPreview: true" in body and "hideCorePreview: true" in body, name


def test_core_preview_is_hidden_in_both_renderers():
    a = JS.index("function hideCorePreview")
    body = JS[a:JS.index("\nfunction ", a + 10)]
    assert 'x.name === "video-preview"' in body and "setHidden(w, true)" in body          # classic: a widget
    assert "[data-c2c-no-core-preview] .video-preview" in body                          # Nodes 2.0: store-drawn
