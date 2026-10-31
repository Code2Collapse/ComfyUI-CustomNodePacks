"""Moved to c2c_core/video/budget.py (L9.73, the vendored shared core). This path stays so every existing import,
monkeypatch and saved reference keeps working: it IS that module (sys.modules alias), not a copy."""
import sys

try:
    from ...c2c_core.video import budget as _m          # custom_nodes.<pack>.nodes.c2c_video -> <pack>.c2c_core
except (ImportError, ValueError):
    from c2c_core.video import budget as _m             # tests: the pack root is on sys.path

sys.modules[__name__] = _m
