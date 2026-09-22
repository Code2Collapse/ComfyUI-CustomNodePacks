"""Promoted parameters on the C2C Vault.

The feature: expose a widget from inside the vault on the vault node itself, so
it can be driven from outside while the contents stay sealed - the way subgraph
promotion works, except that here the thing being promoted out of is encrypted.

That last part is the whole difficulty. A promoted parameter has a public half
(label, type, value) and a secret half (which widget of which internal node it
drives). Putting the secret half in the clear-text manifest would name the
contents - `{"node": "7", "widget": "lora_name"}` gives away the LoRA stack -
which is most of what the vault exists to hide. So the halves are stored apart,
and the tests below exist mainly to keep them apart.

CPU-only, no crypto keys needed for the resolution tests.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

PACK = Path(__file__).resolve().parents[1]
if str(PACK) not in sys.path:
    sys.path.insert(0, str(PACK))

for mod in ("folder_paths", "comfy", "comfy.utils", "server"):
    if mod not in sys.modules:
        stub = types.ModuleType(mod)
        stub.__path__ = []
        sys.modules[mod] = stub

from nodes.vault_promote import (  # noqa: E402
    MAX_VAULT_PARAMS,
    VaultPromotionError,
    describe,
    parse_params,
    parse_promoted,
    resolve_overrides,
)


def subgraph(promoted=None):
    return {
        "nodes": [{"id": "3", "class_type": "KSampler", "widgets": {"seed": 1}}],
        "links": [],
        "boundary_in": [],
        "boundary_out": [{"name": "out"}],
        "promoted": promoted if promoted is not None else
        [{"id": "p0", "node": "3", "widget": "seed"}],
    }


def iface(params):
    return {"mode": "locked", "params": params}


# ── the separation that makes it safe ───────────────────────────────────────

def test_the_manifest_never_names_the_internal_node_or_widget():
    """The public half must not describe the contents.

    This is the property the whole design turns on. If a future change starts
    writing `node`/`widget` into the manifest, the vault still works perfectly
    and silently leaks its own contents - so it is asserted directly.
    """
    params = parse_params(iface([
        {"id": "p0", "name": "seed", "type": "INT", "value": 12345},
    ]))
    assert params[0].keys() <= {"id", "name", "type", "value", "socket"}
    assert "node" not in params[0]
    assert "widget" not in params[0]


def test_the_secret_half_lives_in_the_subgraph():
    table = parse_promoted(subgraph())
    assert table == {"p0": ("3", "seed")}


def test_the_label_is_the_owners_choice_not_the_internal_widget_name():
    """`name` is decoration. Calling it "x" must still drive `seed`."""
    ov = resolve_overrides(
        subgraph(),
        iface([{"id": "p0", "name": "x", "type": "INT", "value": 7}]),
        {},
    )
    assert ov == {"3": {"seed": 7}}


# ── resolution ──────────────────────────────────────────────────────────────

def test_a_typed_value_reaches_the_internal_widget():
    ov = resolve_overrides(
        subgraph(),
        iface([{"id": "p0", "name": "seed", "type": "INT", "value": 999}]),
        {},
    )
    assert ov == {"3": {"seed": 999}}


def test_a_wired_parameter_reads_its_socket_instead_of_its_value():
    ov = resolve_overrides(
        subgraph(),
        iface([{"id": "p0", "name": "seed", "type": "INT",
                "value": 111, "socket": 2}]),
        {2: 222},
    )
    assert ov == {"3": {"seed": 222}}, "the stored value should have been ignored"


def test_converting_to_a_socket_keeps_the_same_id():
    """Flipping between typed and wired must not re-point the parameter.

    The id is the only thing that addresses the internal widget, so if the UI
    ever regenerated it on conversion the knob would silently start driving
    something else.
    """
    widget_form = {"id": "p0", "name": "seed", "type": "INT", "value": 5}
    wired_form = dict(widget_form, socket=0)
    a = resolve_overrides(subgraph(), iface([widget_form]), {})
    b = resolve_overrides(subgraph(), iface([wired_form]), {0: 5})
    assert a == b


def test_an_unconnected_wired_parameter_is_named():
    with pytest.raises(VaultPromotionError, match=r"nothing is connected to param_3"):
        resolve_overrides(
            subgraph(),
            iface([{"id": "p0", "name": "seed", "type": "INT", "socket": 3}]),
            {},
        )


def test_a_parameter_the_vault_does_not_know_is_refused():
    """Manifest and ciphertext out of step. Guessing here would drive the
    wrong widget, so it stops instead."""
    with pytest.raises(VaultPromotionError, match="re-lock the vault"):
        resolve_overrides(
            subgraph(),
            iface([{"id": "ghost", "name": "seed", "type": "INT", "value": 1}]),
            {},
        )


def test_duplicate_ids_are_refused():
    with pytest.raises(VaultPromotionError, match="share the id"):
        parse_params(iface([
            {"id": "p0", "name": "a", "type": "INT", "value": 1},
            {"id": "p0", "name": "b", "type": "INT", "value": 2},
        ]))


def test_a_socket_out_of_range_is_named_with_the_range():
    with pytest.raises(VaultPromotionError, match=r"param_0\.\.param_7"):
        parse_params(iface([{"id": "p0", "name": "s", "type": "INT",
                             "socket": MAX_VAULT_PARAMS}]))


def test_a_type_with_no_widget_must_be_wired():
    with pytest.raises(VaultPromotionError, match="connect it to param_N"):
        resolve_overrides(
            subgraph(),
            iface([{"id": "p0", "name": "img", "type": "IMAGE", "value": None}]),
            {},
        )


def test_no_params_is_not_an_error():
    assert resolve_overrides(subgraph(), iface([]), {}) == {}
    assert resolve_overrides(subgraph(), {}, {}) == {}


def test_a_malformed_manifest_is_refused_rather_than_ignored():
    with pytest.raises(VaultPromotionError, match="must be a list"):
        parse_params({"params": "seed=1"})


# ── the report ──────────────────────────────────────────────────────────────

def test_the_report_shows_values_and_wires_differently():
    text = describe(iface([
        {"id": "p0", "name": "seed", "type": "INT", "value": 42},
        {"id": "p1", "name": "strength", "type": "FLOAT", "socket": 1},
    ]))
    assert "seed = 42" in text
    assert "strength <- param_1 (wired)" in text


def test_the_report_says_when_there_are_none():
    assert "No promoted parameters" in describe({})


# ── the node wiring ─────────────────────────────────────────────────────────

def test_the_vault_exposes_param_sockets():
    from nodes.vault_node import C2C_VaultLocked

    optional = C2C_VaultLocked.INPUT_TYPES()["optional"]
    for i in range(MAX_VAULT_PARAMS):
        assert f"param_{i}" in optional
    # and they are distinct from the boundary sockets
    assert "input_0" in optional and "param_0" in optional


def test_a_broken_manifest_gives_a_sentence_not_a_traceback():
    from nodes.vault_node import _parse_interface

    with pytest.raises(RuntimeError, match="not valid JSON"):
        _parse_interface("{not json")
    with pytest.raises(RuntimeError, match="must be an object"):
        _parse_interface("[1,2]")
    assert _parse_interface("") == {}


def test_a_promotion_pointing_outside_the_vault_is_caught_by_the_executor():
    from nodes.vault_exec import VaultExecError, execute_subgraph

    with pytest.raises(VaultExecError, match="not in this vault"):
        execute_subgraph(subgraph(), {}, {"99": {"seed": 1}})


def test_the_manifest_round_trips_through_json():
    """It travels as a STRING widget, so it has to survive serialisation."""
    params = [{"id": "p0", "name": "seed", "type": "INT", "value": 42},
              {"id": "p1", "name": "s", "type": "FLOAT", "socket": 0}]
    again = parse_params(json.loads(json.dumps(iface(params))))
    assert [p["id"] for p in again] == ["p0", "p1"]
    assert again[1]["socket"] == 0
