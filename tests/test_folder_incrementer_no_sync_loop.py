"""FolderIncrementer's UI must not re-sync itself forever.

Measured 2026-09-27 with the user's H3 workflow at idle: folder_incrementer.js
took ~700 ms of main thread per second (~150 setTimeout laps/s) and the canvas
fell to 8 redraws/s. syncSourceFilename() wrote source_filename through
forceWidgetRefresh(), which fired the widget's callback, which scheduled
syncSourceFilename() again - with the same value, forever. After the fix:
2.4 ms/s and 59 redraws/s on the same workflow.

The function lives inside the node's closure, so it is exercised here by
lifting its source into node with the pieces it touches faked.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "js" / "folder_incrementer.js"
NODE = shutil.which("node")


def _function(src: str, name: str) -> str:
    start = src.index(f"function {name}(")
    depth, i = 0, src.index("{", start)
    while True:
        if src[i] == "{":
            depth += 1
        elif src[i] == "}":
            depth -= 1
            if depth == 0:
                return src[start:i + 1]
        i += 1


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_a_sync_that_writes_the_same_value_does_not_reschedule_itself():
    src = SRC.read_text(encoding="utf-8")
    body = _function(src, "forceWidgetRefresh")
    harness = """
const app = { canvas: {} };
let scheduled = 0, callbacks = 0;
const node = { onWidgetChanged() {} };
const w = { value: "", callback() { callbacks++; if (!node._fiWriting) scheduled++; } };
%FN%
forceWidgetRefresh(w, "plate_v001");       // a real change: callback runs, as our own write
forceWidgetRefresh(w, "plate_v001");       // the re-sync writes the same stem
forceWidgetRefresh(w, "plate_v001");
const writingAfter = !!node._fiWriting;
w.callback("user typed");                  // a real user edit still schedules a sync
process.stdout.write(JSON.stringify({ callbacks, scheduled, writingAfter, value: w.value }));
""".replace("%FN%", body)
    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / "fi.mjs"
        f.write_text(harness, encoding="utf-8")
        p = subprocess.run([NODE, str(f)], capture_output=True, text=True, timeout=60)
    assert p.returncode == 0, p.stderr[:800]
    out = json.loads(p.stdout)
    assert out["callbacks"] == 2, "an unchanged value must not fire the callback again"
    assert out["scheduled"] == 1, "only the user's own edit may schedule a re-sync"
    assert out["writingAfter"] is False


def test_every_resync_callback_respects_our_own_writes():
    src = SRC.read_text(encoding="utf-8")
    hook_at = src.index("sfWidgetHook.callback = function")
    hook = src[hook_at:src.index("};", hook_at)]
    # source_filename's own hook returns early on our writes ...
    assert "if (node._fiWriting) return;" in hook
    assert hook.index("if (node._fiWriting) return;") < hook.index("setTimeout(syncSourceFilename, 0);")
    # ... and every other widget hook checks the flag on the scheduling line
    rest = src[:hook_at] + src[hook_at + len(hook):]
    for m in re.finditer(r"^(.*)setTimeout\(syncSourceFilename, 0\);", rest, flags=re.M):
        assert "_fiWriting" in m.group(1), f"unguarded re-sync: {m.group(0).strip()}"
