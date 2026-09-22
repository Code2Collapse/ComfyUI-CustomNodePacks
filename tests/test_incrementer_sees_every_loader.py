"""The incrementer must recognise every reader in this workspace.

THE REPORTED BUG: with several Load Video / Load Image / OCIO read / Nuke read
nodes in one graph the incrementer gets confused, and a VAE Decode wired into
it still yields the REF IMAGE name rather than the video or EXR name.

The cause was not the filename handling — `file_path` and `exr_path` were
already in FILENAME_WIDGETS, and a numbered EXR sequence already classified as
a moving-image source. The cause was INPUT_LOADER_TYPES, which listed only the
core and VHS loaders. That list does two jobs and both failed:

  * it STOPS the upstream walk, so an unrecognised reader was walked straight
    past and the search escaped into another island to find a ref image;
  * it drives the many-loaders disambiguator, so an EXR read could never beat
    a recognised LoadImage.

The right filename, taken from the wrong node.

WHY THIS TEST EXISTS AND NOT JUST A FIX: the list is a hardcoded copy of
knowledge that lives in the node registry, so it goes stale every time a
reader is added — which is exactly how it got stale. This reads the LIVE packs
and fails when a reader is not covered, naming it.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

PACK = Path(__file__).resolve().parents[1]
WORKSPACE = PACK.parent
if str(PACK) not in sys.path:
    sys.path.insert(0, str(PACK))

JS = PACK / "js" / "folder_incrementer.js"

# Imported HERE, at module scope, before anything puts ComfyUI's root on
# sys.path: once it is there, `import folder_incrementer` resolves ComfyUI's
# own nodes.py instead and dies on a missing submodule. The live-reader check
# below therefore runs in a SUBPROCESS rather than polluting this one.
from folder_incrementer import _resolve_stem_and_ext  # noqa: E402


def _js_list(name: str) -> list[str]:
    """The string entries of a `const NAME = [ ... ];` array in the JS."""
    src = JS.read_text(encoding="utf-8")
    m = re.search(r"const\s+" + re.escape(name) + r"\s*=\s*\[(.*?)\]\s*;", src, re.S)
    assert m, f"{name} not found in folder_incrementer.js"
    body = re.sub(r"//[^\n]*", "", m.group(1))          # strip comments
    return re.findall(r'"([^"]+)"', body)


LOADERS = _js_list("INPUT_LOADER_TYPES")
WIDGETS = _js_list("FILENAME_WIDGETS")


def _covers(node_id: str) -> bool:
    """The JS matches with `cls.includes(t)`, so a prefix entry counts."""
    return any(t in node_id for t in LOADERS)


# ── the regression, stated directly ─────────────────────────────────────────

@pytest.mark.parametrize("node_id", [
    "LoadEXRMEC",
    "NukeMax_EXRSequenceLoad",
    "NukeMax_VideoSequenceLoad",
    "NukeMax_EXRChannelRouter",
    "NukeMax_ReadMultiPass",
    "MiniMaxH3_DCCBridge",
])
def test_the_vfx_readers_are_recognised_as_loaders(node_id):
    """These are the nodes the bug was about. Unrecognised, the upstream walk
    goes straight past them and finds a ref image in another island."""
    assert _covers(node_id), (
        f"{node_id} is not in INPUT_LOADER_TYPES, so the walk will not stop "
        "there and the incrementer will report some other node's filename")


@pytest.mark.parametrize("widget", [
    "file_path", "exr_path", "image_path", "sequence_path",
    "uploaded_file", "exr_sequence", "path", "source",
])
def test_the_path_widgets_the_readers_use_are_known(widget):
    assert widget in WIDGETS, f"{widget} is not in FILENAME_WIDGETS"


def test_the_core_loaders_are_still_covered():
    """The fix must not have displaced what already worked."""
    for node_id in ("LoadImage", "LoadVideo", "VHS_LoadVideo", "LoadAudio"):
        assert _covers(node_id), node_id


# ── the part that keeps it from going stale ─────────────────────────────────

def _live_readers() -> list[tuple[str, str]]:
    """(node id, pack) for every node in the workspace that READS a file.

    A reader is a node that takes no IMAGE input, has a string widget naming
    a path, and outputs an IMAGE or MASK - a source of picture the
    incrementer should be able to name.

    Run in a SUBPROCESS: loading six packs needs ComfyUI's root on sys.path,
    and putting it there in this process breaks `import folder_incrementer`
    for every other test in the file.
    """
    import json
    import subprocess

    code = r"""
import sys, json, types, importlib.util
from pathlib import Path
sys.path.insert(0, r"D:\PROJECT\ComfyUI_windows_portable\ComfyUI")
try:
    import server
    if getattr(server.PromptServer, "instance", None) is None:
        class _R:
            def get(self,*a,**k): return lambda fn: fn
            def post(self,*a,**k): return lambda fn: fn
            def delete(self,*a,**k): return lambda fn: fn
        server.PromptServer.instance = types.SimpleNamespace(routes=_R())
except Exception:
    pass
WS = Path(sys.argv[1])
out = []
for pack in ("ComfyUI-NukeMaxNodes", "ComfyUI-CustomNodePacks",
             "ComfyUI-MiniMaxSuite", "ComfyUI-WanNodeExperiments"):
    root = WS / pack
    if not (root / "__init__.py").is_file():
        continue
    spec = importlib.util.spec_from_file_location(
        "rd_" + pack.replace("-", "_"), root / "__init__.py",
        submodule_search_locations=[str(root)])
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:
        continue
    for nid, cls in (getattr(mod, "NODE_CLASS_MAPPINGS", {}) or {}).items():
        try:
            sp = cls.INPUT_TYPES()
        except Exception:
            continue
        fields = {}
        fields.update(sp.get("required") or {})
        fields.update(sp.get("optional") or {})
        tys = {n: (v[0] if isinstance(v, (list, tuple)) and v else v)
               for n, v in fields.items()}
        if any(t == "IMAGE" for t in tys.values()):
            continue
        has_path = any(
            t == "STRING" and any(k in n.lower() for k in
                ("path", "file", "dir", "folder", "sequence", "pattern"))
            for n, t in tys.items())
        if not has_path:
            continue
        rets = tuple(getattr(cls, "RETURN_TYPES", ()) or ())
        if "IMAGE" in rets or "MASK" in rets:
            out.append([nid, pack])
print("READERS_JSON:" + json.dumps(out))
"""
    r = subprocess.run(
        [sys.executable, "-c", code, str(WORKSPACE)],
        capture_output=True, text=True, timeout=600)
    for line in (r.stdout or "").splitlines():
        if line.startswith("READERS_JSON:"):
            return [tuple(x) for x in json.loads(line[len("READERS_JSON:"):])]
    return []


def test_every_live_reader_in_the_workspace_is_covered():
    """The list is a hardcoded copy of what the registry already knows, so it
    goes stale whenever a reader is added - which is exactly how this bug
    happened. Reading the live packs turns silent rot into a failing test.
    """
    readers = _live_readers()
    assert readers, (
        "no readers were found at all, so this check proved nothing - the "
        "subprocess could not load the packs")
    missing = [f"{nid} ({pack})" for nid, pack in readers if not _covers(nid)]
    assert not missing, (
        "These nodes read a file and output a picture, but the incrementer "
        "does not recognise them as loaders - so an upstream walk passes "
        "straight through and reports some other node's filename:\n  "
        + "\n  ".join(sorted(missing))
        + "\nAdd them to INPUT_LOADER_TYPES in js/folder_incrementer.js.")


# ── the Python half, which was never the problem ────────────────────────────

@pytest.mark.parametrize("filename,stem,ext", [
    ("shot_v001.1001.exr", "shot_v001", ".exr"),
    ("plate.0001.png", "plate", ".png"),
    ("BLD_0100_comp_v003.####.exr", "BLD_0100_comp_v003", ".exr"),
    ("shot.1001.dpx", "shot", ".dpx"),
    ("clip.mov", "clip", ".mov"),
])
def test_a_sequence_name_loses_its_frame_number_not_its_version(filename, stem, ext):
    """`shot_v001.1001.exr` must become `shot_v001`, not `shot`: the version
    token is part of the name and the frame number is not, and confusing the
    two files the render under the wrong shot."""
    got_stem, got_ext = _resolve_stem_and_ext(filename, "", "auto")
    assert (got_stem, got_ext) == (stem, ext)
