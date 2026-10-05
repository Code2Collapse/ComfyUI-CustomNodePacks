"""The Magnific port, and the four things that had to change on the way in.

The pack is the vendor's own, copied in almost unaltered - which is the point:
a re-sync from upstream should be a file copy, not a merge. But four edits are
load-bearing, and every one of them is the kind that fails SILENTLY or fails
at a distance:

  a short ../ count in the JS 404s, and one 404 in a pack module takes the
  whole pack's front-end down, not just these nodes;

  a version that reads "0.0.0" is below every possible floor;

  a raise in assert_not_blocked lets a CDN file stop the nodes running;

  an unguarded PromptServer.instance raises AttributeError at import, which
  removes every CNP node from /object_info.

None of those show up as a failing Magnific node. They show up as "the UI is
broken" or "my nodes are gone", days later. So they are pinned here, with the
reason attached, because the next person to re-sync from upstream will
otherwise copy the four problems back in.

No network. Nothing here signs in or calls the service.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

PACK = Path(__file__).resolve().parents[1]
PY = PACK / "nodes" / "magnific"
JS = PACK / "js" / "magnific"

EXPECTED_NODES = {
    "MagnificSaveTo", "MagnificGenerateImage", "MagnificUpscaleImage",
    "MagnificStockSearch", "MagnificGenerateVideo", "MagnificUpscaleVideo",
    "MagnificGenerateMusic", "MagnificVoiceover", "MagnificSkinEnhancer",
    "MagnificRemoveBackground", "MagnificRetouch", "MagnificStockPicker",
    "MagnificCreationPicker", "MagnificLibraryReference", "MagnificMetadata",
}


def source(rel: str) -> str:
    p = PACK / rel
    assert p.exists(), f"{rel} is gone"
    return p.read_text(encoding="utf-8")


# -- the port arrived intact ------------------------------------------------

def test_every_node_the_vendor_ships_is_registered():
    from nodes.magnific import NODE_CLASS_MAPPINGS

    assert set(NODE_CLASS_MAPPINGS) == EXPECTED_NODES


def test_every_node_has_a_display_name():
    from nodes.magnific import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

    missing = set(NODE_CLASS_MAPPINGS) - set(NODE_DISPLAY_NAME_MAPPINGS)
    assert not missing, f"no menu label for {sorted(missing)}"


def test_they_reach_the_packs_own_mappings():
    """Registered in nodes/magnific/ but not wired into __init__.py is the
    failure that looks exactly like success from inside the subpackage."""
    import re

    root = source("__init__.py")
    assert "_MAGNIFIC_MAPPINGS" in root
    # since L2.01 the merge is the ordered _C2C_FAMILIES table: (label, mappings, display)
    table = root.split("_C2C_FAMILIES = [", 1)[1].split("\n]", 1)[0]
    assert re.search(r'\(\s*"[^"]+",\s*_MAGNIFIC_MAPPINGS,\s*_MAGNIFIC_DISPLAY\s*\)', table)


def test_the_node_ids_are_not_renamed():
    """A workflow saved against the vendor's pack has to keep resolving. The
    cost is that both packs installed at once register the same fifteen IDs -
    documented at the CATEGORY, not worked around by renaming."""
    from nodes.magnific import NODE_CLASS_MAPPINGS

    assert all(k.startswith("Magnific") for k in NODE_CLASS_MAPPINGS)


# -- change 1: the JS import depth ------------------------------------------

def test_the_js_reaches_comfyui_from_one_directory_deeper():
    """js/magnific/x.js serves at /extensions/<pack>/magnific/x.js, so ComfyUI
    is three levels up, not two. Upstream sat at the web root and used two.

    A short count is a 404, and a 404 on a module import is not contained:
    the browser's preloader fails the whole graph of pack modules with it.
    """
    for f in JS.glob("*.js"):
        text = f.read_text(encoding="utf-8")
        for m in re.finditer(r'from\s+"((?:\.\./)+)scripts/', text):
            assert m.group(1) == "../../../", (
                f"{f.name} imports ComfyUI via {m.group(1)!r}; from "
                f"js/magnific/ it must be '../../../'")


def test_the_js_files_are_actually_there():
    """The Python half works without them and the nodes still register, so a
    missing front-end is invisible until someone adds the node."""
    assert (JS / "magnific.js").exists()
    assert (JS / "widget_remap.js").exists()


def test_the_local_import_between_them_is_unchanged():
    """Same directory, so this one must NOT have been re-depthed along with
    the ComfyUI imports."""
    assert 'from "./widget_remap.js"' in (JS / "magnific.js").read_text(encoding="utf-8")


# -- change 2: the version -------------------------------------------------

def test_the_version_is_a_literal_not_a_file_read():
    """Upstream read a sibling pyproject.toml. Inside CNP that path points at
    nodes/pyproject.toml, which does not exist - and the except branch
    returned "0.0.0", which is below any minVersion the service could
    publish. The nodes would have reported themselves retired on day one."""
    from nodes.magnific import config

    assert re.fullmatch(r"\d+\.\d+\.\d+", config.PLUGIN_VERSION), config.PLUGIN_VERSION
    assert config.PLUGIN_VERSION != "0.0.0"
    # Judge the code, not the comment explaining why the code changed - the
    # first version of this searched the raw text for "pyproject" and failed
    # on its own explanation.
    tree = ast.parse(source("nodes/magnific/config.py"))
    assigned = [n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                and any(getattr(t, "id", "") == "PLUGIN_VERSION" for t in n.targets)]
    assert len(assigned) == 1, "PLUGIN_VERSION is set more than once"
    assert isinstance(assigned[0].value, ast.Constant),         "PLUGIN_VERSION is computed again - a failing read returns 0.0.0"


def test_the_version_matches_what_credits_records():
    """CREDITS.md is where the re-sync starts, so it has to name the release
    that is actually sitting in the tree."""
    from nodes.magnific import config

    credits = source("CREDITS.md")
    assert "magnific" in credits.lower(), "the port is not in the licence record"
    assert config.PLUGIN_VERSION in credits, (
        f"CREDITS.md does not record {config.PLUGIN_VERSION}")


# -- change 3: the killswitch ----------------------------------------------

def test_a_remote_manifest_cannot_stop_the_nodes_running():
    """Upstream raised VersionBlockedError from assert_not_blocked, called at
    the top of every node. That let a file on a CDN decide whether source in
    this repository executes. It warns now."""
    tree = ast.parse(source("nodes/magnific/update_check.py"))
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "assert_not_blocked")
    raises = [n for n in ast.walk(fn) if isinstance(n, ast.Raise)]
    assert not raises, "assert_not_blocked raises again - the killswitch is back"


def test_it_still_tells_the_user_when_the_version_is_retired():
    """Not raising is not the same as hiding it. Silently ignoring the floor
    would leave the user guessing when calls start failing."""
    tree = ast.parse(source("nodes/magnific/update_check.py"))
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "assert_not_blocked")
    body = ast.unparse(fn)
    assert "warning" in body, "the retirement notice was dropped along with the raise"


def test_it_warns_once_rather_than_once_per_node_per_run():
    """Every one of the fifteen nodes calls this. A per-call warning turns one
    retired version into fifteen lines per queued prompt."""
    src = source("nodes/magnific/update_check.py")
    assert "_warned" in src


def test_the_check_still_fails_open_when_offline():
    """The vendor got this right and the port must not lose it: an
    unreachable CDN must not take the nodes with it."""
    from nodes.magnific import update_check

    update_check._cache.update(at=0.0, state=None)
    original = update_check.config.MANIFEST_URL
    update_check.config.MANIFEST_URL = "http://127.0.0.1:1/nope.json"
    try:
        state = update_check.state(force=True)
    finally:
        update_check.config.MANIFEST_URL = original
        update_check._cache.update(at=0.0, state=None)
    assert state["blocked"] is False
    assert state["update_available"] is False


# -- change 4: importing must never need a server ---------------------------

def test_the_route_mount_survives_a_missing_promptserver():
    """`except ImportError` was the upstream guard. PromptServer imports fine
    in a headless load and then has no .instance, which is AttributeError -
    and an exception at pack-import level does not lose these fifteen nodes,
    it loses all of CNP from /object_info."""
    src = source("nodes/magnific/__init__.py")
    fn = next(n for n in ast.walk(ast.parse(src))
              if isinstance(n, ast.FunctionDef) and n.name == "_register_routes")
    handlers = [h for n in ast.walk(fn) if isinstance(n, ast.Try) for h in n.handlers]
    assert handlers, "nothing is guarded"
    for h in handlers:
        name = getattr(h.type, "id", None)
        assert name == "Exception", (
            f"caught {name} - a narrower guard is what let AttributeError "
            "through upstream")


def test_importing_the_subpackage_does_not_need_aiohttp_or_a_server():
    """This is the whole reason the routes moved into a function: the import
    itself has to be inert."""
    import importlib

    import nodes.magnific as m

    importlib.reload(m)
    assert len(m.NODE_CLASS_MAPPINGS) == 15


def test_the_pack_root_guards_the_import_too():
    """Belt and braces, and the pattern every other CNP subpackage follows -
    a failure is recorded and reported in the UI rather than thrown."""
    root = source("__init__.py")
    block = root[root.index("# Magnific"):root.index("# VAE Clean")]
    assert "except Exception" in block
    assert "record_failure" in block
    assert "_MAGNIFIC_MAPPINGS, _MAGNIFIC_DISPLAY = {}, {}" in block


# -- the service boundary ---------------------------------------------------

def test_credentials_are_not_stored_in_the_repository():
    """Sign-in writes a token blob. It belongs in the user's home, not next to
    the source - a stray auth.json under a tracked directory is a token in
    the next commit."""
    from nodes.magnific import config

    assert ".magnific" in config.AUTH_FILE
    assert str(PACK).lower() not in config.AUTH_FILE.lower()


@pytest.mark.parametrize("node_id", sorted(EXPECTED_NODES))
def test_no_node_calls_out_at_import_or_schema_time(node_id):
    """INPUT_TYPES runs on every /object_info request. A node that reached the
    service there would make opening the menu depend on the network, and a
    signed-out user would see the whole pack stall."""
    from nodes.magnific import NODE_CLASS_MAPPINGS

    cls = NODE_CLASS_MAPPINGS[node_id]
    spec = cls.INPUT_TYPES()
    assert isinstance(spec, dict)
    assert isinstance(cls.RETURN_TYPES, tuple)
