"""Register the renderfarm package under the plain name 'renderfarm' so tests
can import it outside ComfyUI (the pack folder name contains hyphens, which
blocks normal dotted imports). Relative imports inside the package then
resolve as renderfarm.<submodule>."""

import importlib.util
import os
import sys

_PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if "renderfarm" not in sys.modules:
    spec = importlib.util.spec_from_file_location(
        "renderfarm", os.path.join(_PKG_DIR, "__init__.py"),
        submodule_search_locations=[_PKG_DIR])
    mod = importlib.util.module_from_spec(spec)
    sys.modules["renderfarm"] = mod
    spec.loader.exec_module(mod)
