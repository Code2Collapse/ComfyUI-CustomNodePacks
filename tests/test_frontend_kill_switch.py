"""L2.49: C2C_NO_FRONTEND=1 stops ComfyUI serving this pack's JS (the nodes still load). The switch lets the owner tell
in one restart whether a browser problem comes from this pack."""
from __future__ import annotations

import os
from pathlib import Path

INIT = Path(__file__).resolve().parents[1] / "__init__.py"
BLOCK = ('WEB_DIRECTORY = "./js"\n'
         'if __import__("os").environ.get("C2C_NO_FRONTEND", "").strip() == "1":\n'
         '    WEB_DIRECTORY = None\n')


def _web_directory(env_value):
    """Evaluate the WEB_DIRECTORY lines of __init__.py with the given environment value."""
    src = INIT.read_text(encoding="utf-8").replace("\r\n", "\n")
    assert BLOCK in src, "the WEB_DIRECTORY kill switch is missing from __init__.py"
    old = os.environ.get("C2C_NO_FRONTEND")
    try:
        if env_value is None:
            os.environ.pop("C2C_NO_FRONTEND", None)
        else:
            os.environ["C2C_NO_FRONTEND"] = env_value
        scope: dict = {}
        exec(BLOCK, scope)
        return scope["WEB_DIRECTORY"]
    finally:
        if old is None:
            os.environ.pop("C2C_NO_FRONTEND", None)
        else:
            os.environ["C2C_NO_FRONTEND"] = old


def test_front_end_served_by_default():
    assert _web_directory(None) == "./js" and _web_directory("0") == "./js"


def test_kill_switch_serves_no_front_end():
    assert _web_directory("1") is None
