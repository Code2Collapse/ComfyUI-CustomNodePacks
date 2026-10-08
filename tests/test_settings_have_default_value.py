"""Every C2C setting declares `defaultValue`, never `default`.

ComfyUI reads `defaultValue`. A setting declared with `default:` opens EMPTY in the settings panel, the
"Reset C2C settings" button cannot restore it, and its code only works through its own getSettingValue fallback.
Thirteen settings had it (A9 "settings are buggy", 2026-10-08); this keeps it from coming back.
"""
from __future__ import annotations

import re
from pathlib import Path

JS = Path(__file__).resolve().parents[1] / "js"
# a settings-like object: `{ id: ..., ... }`, allowing one level of nested braces (attrs: { min, max })
OBJ = re.compile(r"\{\s*id:\s*(?:[^{}]|\{[^{}]*\})*\}")


def _offenders() -> list[str]:
    bad = []
    for f in sorted(JS.rglob("*.js")):
        src = f.read_text(encoding="utf-8", errors="replace")
        for m in OBJ.finditer(src):
            body = m.group(0)
            if not (re.search(r"\bname:\s*", body) and re.search(r"\btype:\s*", body)):
                continue
            if re.search(r"(^|[\s,{])default:\s*", body) and not re.search(r"\bdefaultValue:\s*", body):
                line = src[:m.start()].count("\n") + 1
                bad.append(f"{f.relative_to(JS.parent).as_posix()}:{line}")
    return bad


def test_no_setting_uses_default_instead_of_default_value():
    bad = _offenders()
    assert not bad, "settings declared with `default:` (ComfyUI reads `defaultValue:`):\n" + "\n".join(bad)
