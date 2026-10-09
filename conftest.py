"""Root conftest – prevent pytest from crashing on the ComfyUI package __init__.py.

The root __init__.py uses relative imports (from .folder_incrementer ...)
which fail outside of ComfyUI's runtime.  We monkeypatch pytest's Package
class so its setup() silently swallows that ImportError.
"""
import sys
import types

# ── 1. Stub ComfyUI-specific modules ──────────────────────────────────
for mod_name in ("folder_paths", "comfy", "comfy.utils"):
    if mod_name not in sys.modules:
        stub = types.ModuleType(mod_name)
        stub.__path__ = []
        sys.modules[mod_name] = stub

# ── 2. Monkeypatch Package.setup to tolerate __init__.py import errors ─
from _pytest.python import Package

_original_setup = Package.setup


def _safe_setup(self):
    try:
        _original_setup(self)
    except Exception:
        pass  # Swallow root __init__.py import errors


Package.setup = _safe_setup




# ── 3. No network downloads from unit tests (L2.37) ─────────────────────
# The optical-flow test reached ProPainter's RAFT weight downloader (21 MB) mid-suite and only fell back to LK when
# the download failed. Unit tests must not touch the network: C2C_NO_DOWNLOADS=1 makes the downloader refuse at
# once, so nodes take their documented offline fallback; a weight already on disk still loads. Set as an env var
# (not a patch) so nothing is imported early. C2C_TESTS_ALLOW_DOWNLOADS=1 opts a deliberate real-weights run back in.
import os as _os

if _os.environ.get("C2C_TESTS_ALLOW_DOWNLOADS") != "1":
    _os.environ.setdefault("C2C_NO_DOWNLOADS", "1")
