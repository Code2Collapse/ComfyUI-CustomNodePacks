"""The house look on the canvas, and the copy of it in every pack.

Each Code2Collapse pack ships _c2c_brand.js and colours ONLY its own nodes -
the packs cannot import each other (a cross-pack import 404s on a standalone
install and takes the pack's whole front-end down). So the six copies must
stay identical, and the one decision that matters - "is this node mine?" - is
exercised here through the real file, not a Python paraphrase of it.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

WORKSPACE = Path(__file__).resolve().parents[2]
CANONICAL = WORKSPACE / "ComfyUI-CustomNodePacks" / "js" / "_c2c_brand.js"
COPIES = {
    "ComfyUI-CustomNodePacks": "js",
    "ComfyUI-NukeMaxNodes": "web",
    "ComfyUI-WanNodeExperiments": "web",
    "ComfyUI-MiniMaxSuite": "web",
    "ComfyUI-WanAnimatePreprocessV2": "js",
    "ComfyUI-GLM_Image": "web",
    "ComfyUI-WanAnimalPreprocessor": "web",
}


def _copy(pack):
    return WORKSPACE / pack / COPIES[pack] / "_c2c_brand.js"


@pytest.mark.parametrize("pack", sorted(COPIES))
def test_every_pack_ships_the_same_brand(pack):
    path = _copy(pack)
    if not (WORKSPACE / pack).exists():
        pytest.skip(f"{pack} is not in this workspace")
    assert path.exists(), f"{pack} has no _c2c_brand.js - its nodes lose the house look"
    assert path.read_text(encoding="utf-8") == CANONICAL.read_text(encoding="utf-8"), (
        f"{pack}'s brand has drifted from CustomNodePacks'")


@pytest.mark.parametrize("pack", sorted(COPIES))
def test_the_copy_sits_at_the_web_root_its_import_expects(pack):
    """Every copy imports ../../scripts/app.js, which is right only at the root
    of the pack's WEB_DIRECTORY. One level deeper is a 404."""
    if not (WORKSPACE / pack).exists():
        pytest.skip(f"{pack} is not in this workspace")
    init = (WORKSPACE / pack / "__init__.py").read_text(encoding="utf-8")
    m = re.search(r'WEB_DIRECTORY\s*=\s*["\']\.?/?([\w/]+)["\']', init)
    assert m, f"{pack} declares no WEB_DIRECTORY, so the brand never loads"
    assert m.group(1).strip("/") == COPIES[pack]


def test_node_colours_are_literal_hex():
    """Node colours are painted on the canvas, which cannot parse var() - it
    paints black and throws nothing."""
    src = CANONICAL.read_text(encoding="utf-8")
    block = src[src.index("export const BRAND"):src.index("});", src.index("export const BRAND"))]
    for key in ("title", "body", "ink"):
        m = re.search(rf'{key}:\s*"(#[0-9a-fA-F]{{6}})"', block)
        assert m, f"BRAND.{key} is not a literal hex colour"
    assert "var(" not in block


def test_the_users_colour_always_wins():
    """A colour from a saved workflow (restored before onNodeCreated) or from
    the Colors menu must never be overwritten."""
    src = CANONICAL.read_text(encoding="utf-8")
    assert "if (!this.color) this.color = BRAND.title" in src
    assert "if (!this.bgcolor) this.bgcolor = BRAND.body" in src


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not on PATH")
def test_a_pack_recognises_its_own_nodes_and_nobody_elses(tmp_path):
    src = CANONICAL.read_text(encoding="utf-8")
    start = src.index("/** \"/extensions/")
    end = src.index("function enabled()")
    probe = tmp_path / "probe.mjs"
    probe.write_text(src[start:end] + r'''
const cases = [
  // [served url, python_module, expected]
  ["http://h/extensions/ComfyUI-NukeMaxNodes/_c2c_brand.js", "custom_nodes.ComfyUI-NukeMaxNodes", true],
  // the Linux install uses the GitHub folder names
  ["http://h/extensions/ComfyUI-NukeNodePack/_c2c_brand.js", "custom_nodes.ComfyUI-NukeNodePack", true],
  // served under a pyproject project name, module still the folder
  ["http://h/extensions/comfyui-nukemax-nodes/_c2c_brand.js", "custom_nodes.ComfyUI-NukeMaxNodes", true],
  // another pack's node
  ["http://h/extensions/ComfyUI-NukeMaxNodes/_c2c_brand.js", "custom_nodes.ComfyUI-MiniMaxSuite", false],
  // a core node
  ["http://h/extensions/ComfyUI-NukeMaxNodes/_c2c_brand.js", "nodes", false],
  ["http://h/extensions/ComfyUI-NukeMaxNodes/_c2c_brand.js", "comfy_extras.nodes_video", false],
  // no python_module: left alone rather than guessed at
  ["http://h/extensions/ComfyUI-NukeMaxNodes/_c2c_brand.js", undefined, false],
  // a folder name with a space, url-encoded by the browser
  ["http://h/extensions/My%20Pack/_c2c_brand.js", "custom_nodes.My Pack", true],
];
const out = cases.map(([u, m, want]) => {
  const got = belongsTo({ python_module: m }, servedFolder(u));
  return { u, m, want, got };
});
process.stdout.write(JSON.stringify(out));
''', encoding="utf-8")
    p = subprocess.run([shutil.which("node"), str(probe)], capture_output=True, text=True, timeout=60)
    assert p.returncode == 0, p.stderr[:600]
    wrong = [c for c in json.loads(p.stdout) if c["got"] != c["want"]]
    assert not wrong, wrong


def test_the_setting_is_registered_once_and_read_by_every_copy():
    src = CANONICAL.read_text(encoding="utf-8")
    assert "if (!window.__C2C_BRAND_REG__)" in src
    assert 'getSettingValue?.("c2c.brand.nodeColors", true)' in src
