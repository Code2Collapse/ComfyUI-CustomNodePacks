"""The vault's clear-text manifest must never describe its own contents.

`buildInterfaceManifest` in js/c2c_vault.js writes `vault_interface`, which is
a plain STRING widget: it travels in the workflow JSON, in every queued prompt,
and in the PNG metadata. Anything it contains is readable by whoever receives
the file - which is precisely who the vault is protecting the contents from.

So the manifest carries the PUBLIC half of a promoted parameter only
(id, label, type, value, optional socket). The half that says which widget of
which internal node it drives lives inside the ciphertext. See the module
docstring of nodes/vault_promote.py.

The JS builds that object with an explicit whitelist. This test exists because
the natural "tidy-up" of a whitelist is a spread - `{...p}` - and that single
character would silently start publishing `node` and `widget`, with the vault
still working perfectly in every other respect.

The function is lifted OUT OF THE SHIPPED FILE and executed, rather than
copied here: a copy would stay green while the shipped widget drifted.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
VAULT_JS = ROOT / "js" / "c2c_vault.js"
FORBIDDEN = ("node", "widget", "nodeId", "widgetName", "_comboOptions")


def _extract(source: str, name: str) -> str:
    marker = f"function {name}("
    start = source.find(marker)
    assert start != -1, (
        f"{name} is gone from {VAULT_JS.name}. If it was renamed or inlined "
        "this leak check is broken and must be repointed, not deleted."
    )
    brace = source.find("{", start)
    depth = 0
    for i in range(brace, len(source)):
        if source[i] == "{":
            depth += 1
        elif source[i] == "}":
            depth -= 1
            if depth == 0:
                return source[start:i + 1]
    raise AssertionError(f"unbalanced braces extracting {name}")


@pytest.fixture(scope="module")
def node_exe():
    exe = shutil.which("node")
    if not exe:
        pytest.skip("node not on PATH")
    return exe


def _run(node_exe, payload):
    src = _extract(VAULT_JS.read_text(encoding="utf-8"), "buildInterfaceManifest")
    harness = (
        src
        + "\nconst a = JSON.parse(process.argv[1]);"
        + "\nconsole.log(JSON.stringify(buildInterfaceManifest("
          "a.mode, a.nodeCount, a.boundary, a.inTypes, a.outTypes, a.params)));"
    )
    proc = subprocess.run([node_exe, "-e", harness, json.dumps(payload)],
                          check=True, capture_output=True, text=True, timeout=30)
    return json.loads(proc.stdout.strip())


BASE = {
    "mode": "locked",
    "nodeCount": 4,
    "boundary": {"boundary_in": [{"name": "image"}], "boundary_out": [{"name": "out"}]},
    "inTypes": ["IMAGE"],
    "outTypes": ["IMAGE"],
}


def test_the_extracted_function_is_the_shipped_one():
    src = _extract(VAULT_JS.read_text(encoding="utf-8"), "buildInterfaceManifest")
    assert src.count("{") == src.count("}")
    assert "params" in src and "return" in src


def test_a_promoted_parameter_publishes_only_its_public_half(node_exe):
    """The whole point. `node` and `widget` must not survive the build."""
    out = _run(node_exe, dict(BASE, params=[{
        "id": "p0", "name": "seed", "type": "INT", "value": 12345,
        # the secret half, deliberately passed in to prove it is dropped
        "node": "7", "widget": "lora_name", "_comboOptions": ["secret.safetensors"],
    }]))
    assert out["params"] == [{"id": "p0", "name": "seed", "type": "INT", "value": 12345}]


def test_no_forbidden_key_appears_anywhere_in_the_manifest(node_exe):
    """Not just in params - anywhere. A leak in `in`/`out` would be just as bad."""
    out = _run(node_exe, dict(BASE, params=[{
        "id": "p0", "name": "s", "type": "FLOAT", "value": 0.5,
        "node": "3", "widget": "strength",
    }]))
    blob = json.dumps(out)
    for key in FORBIDDEN:
        assert f'"{key}"' not in blob, f"the manifest published {key!r}: {blob}"


def test_the_secret_value_itself_never_appears(node_exe):
    """A label may be anything; the internal widget NAME must not leak."""
    out = _run(node_exe, dict(BASE, params=[{
        "id": "p0", "name": "quality", "type": "STRING", "value": "high",
        "node": "9", "widget": "secret_internal_widget_name",
    }]))
    assert "secret_internal_widget_name" not in json.dumps(out)


def test_a_wired_parameter_keeps_its_socket(node_exe):
    out = _run(node_exe, dict(BASE, params=[{
        "id": "p1", "name": "strength", "type": "FLOAT", "value": 1.0, "socket": 3,
        "node": "3", "widget": "strength",
    }]))
    assert out["params"][0]["socket"] == 3
    assert "node" not in out["params"][0]


def test_no_params_is_an_empty_list_not_a_missing_key(node_exe):
    """The Python side treats a missing key as empty, but an explicit [] keeps
    a hand-inspected manifest honest about having nothing promoted."""
    out = _run(node_exe, dict(BASE, params=[]))
    assert out["params"] == []


def test_the_leak_check_can_actually_fail(node_exe):
    """Negative control: a spread instead of the whitelist must be caught.

    This is the exact change the whitelist exists to prevent, so the test
    proves it would be noticed.
    """
    src = _extract(VAULT_JS.read_text(encoding="utf-8"), "buildInterfaceManifest")
    leaky = src.replace("""      const entry = {
        id: p.id,
        name: p.name,
        type: p.type,
        value: p.value,
      };""", "      const entry = { ...p };")
    assert leaky != src, "the whitelist block changed shape; repoint this control"
    harness = (
        leaky
        + "\nconst a = JSON.parse(process.argv[1]);"
        + "\nconsole.log(JSON.stringify(buildInterfaceManifest("
          "a.mode, a.nodeCount, a.boundary, a.inTypes, a.outTypes, a.params)));"
    )
    payload = dict(BASE, params=[{"id": "p0", "name": "s", "type": "INT",
                                  "value": 1, "node": "3", "widget": "seed"}])
    proc = subprocess.run([node_exe, "-e", harness, json.dumps(payload)],
                          check=True, capture_output=True, text=True, timeout=30)
    assert '"widget"' in proc.stdout, "a spread did NOT leak — this control is broken"
