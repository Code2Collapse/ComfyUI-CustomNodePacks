"""C2C settings must be declared on registerExtension({ settings: [...] }), not imperatively in setup().

ComfyUI 1.52 registers declarative settings at registerExtension time and fires onChange immediately.
Lite mode strips setup() for heavy extensions but keeps settings — imperative addSetting in setup()
makes those toggles vanish.
"""
from __future__ import annotations

import re
from pathlib import Path

JS = Path(__file__).resolve().parents[1] / "js"
PAT = re.compile(r"settings\.addSetting\s*\(")
# _c2c_settings_grouping.js patches s.addSetting (no "settings.addSetting(" substring).
ALLOWLIST: set[str] = set()


def _offenders() -> list[str]:
    bad = []
    for f in sorted(JS.rglob("*.js")):
        rel = f.relative_to(JS.parent).as_posix()
        if rel in ALLOWLIST:
            continue
        src = f.read_text(encoding="utf-8", errors="replace")
        for m in PAT.finditer(src):
            line = src[:m.start()].count("\n") + 1
            bad.append(f"{rel}:{line}")
    return bad


def test_no_imperative_settings_add_setting():
    bad = _offenders()
    assert not bad, "imperative settings.addSetting( — use registerExtension({ settings: [...] }):\n" + "\n".join(bad)
