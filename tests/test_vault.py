"""C2C Vault: crypto round-trip, tamper rejection, and no-plaintext-leak.

CPU-only, no server. Every test carries an INVARIANT line.

The negative tests matter more than the positive one here: an encryption bug
that still round-trips looks completely fine, so "it decrypts" proves almost
nothing on its own.
"""

from __future__ import annotations

import base64
import json
import sys
import types
from pathlib import Path

import pytest

PACK_ROOT = Path(__file__).resolve().parents[1]
if str(PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(PACK_ROOT))

pytest.importorskip("cryptography", reason="C2C Vault needs `cryptography` for AES-GCM")

from nodes.vault_crypto import (  # noqa: E402
    MODE_LOCKED,
    MODE_SEALED,
    VaultError,
    boundary_hash,
    lock_subgraph,
    payload_mode,
    payload_vault_id,
    seal_subgraph,
    unlock_subgraph,
    unseal_for_run,
    unseal_with_password,
)
from nodes.vault_boundary import build_interface_manifest, derive_boundary  # noqa: E402
from nodes.vault_exec import VaultExecError, execute_subgraph  # noqa: E402

PASSWORD = "correct horse battery staple"
VAULT_ID = "vault-abc123"

# Fast KDF for tests only. Production uses 600_000; running that per test would
# add minutes for no extra coverage - the iteration count is asserted separately.
FAST = 10_000


def _subgraph():
    """Two pure nodes: Constant -> Add. No models, no disk, no GPU."""
    return {
        "nodes": [
            {"id": "1", "class_type": "_VaultTestConstant", "widgets": {"value": 2.0}},
            {"id": "2", "class_type": "_VaultTestAdd", "widgets": {"addend": 5.0}},
        ],
        "links": [{"from": "1", "from_slot": 0, "to": "2", "to_slot": 0}],
        "boundary_in": [],
        "boundary_out": [{"name": "result", "from": "2", "from_slot": 0}],
        "secret_note": "SUPERSECRETLAYERNAME",
    }


@pytest.fixture
def registry(monkeypatch):
    """Register two trivial nodes into ComfyUI's registry for the executor."""
    class _Constant:
        FUNCTION = "run"
        RETURN_TYPES = ("FLOAT",)

        @classmethod
        def INPUT_TYPES(cls):
            return {"required": {"value": ("FLOAT", {"default": 0.0})}}

        def run(self, value):
            return (float(value),)

    class _Add:
        FUNCTION = "run"
        RETURN_TYPES = ("FLOAT",)

        @classmethod
        def INPUT_TYPES(cls):
            return {"required": {"a": ("FLOAT", {}), "addend": ("FLOAT", {"default": 0.0})}}

        def run(self, a, addend):
            return (float(a) + float(addend),)

    mod = sys.modules.get("nodes")
    if mod is None or not hasattr(mod, "NODE_CLASS_MAPPINGS"):
        mod = types.ModuleType("nodes")
        mod.NODE_CLASS_MAPPINGS = {}
        monkeypatch.setitem(sys.modules, "nodes", mod)
    mapping = dict(getattr(mod, "NODE_CLASS_MAPPINGS", {}))
    mapping.update({"_VaultTestConstant": _Constant, "_VaultTestAdd": _Add})
    monkeypatch.setattr(mod, "NODE_CLASS_MAPPINGS", mapping, raising=False)
    return mapping


# ── round trip ───────────────────────────────────────────────────────────────

def test_lock_unlock_round_trip_is_exact():
    # INVARIANT: decrypting with the right password returns the original object.
    sg = _subgraph()
    payload = lock_subgraph(sg, PASSWORD, vault_id=VAULT_ID, iterations=FAST)
    assert unlock_subgraph(payload, PASSWORD, vault_id=VAULT_ID) == sg


def test_vault_execution_matches_running_the_graph_directly(registry):
    # INVARIANT: the vault is a pass-through for RESULTS - locking a graph must
    # not change what it computes. Constant(2) -> Add(+5) == 7 either way.
    direct = registry["_VaultTestAdd"]().run(registry["_VaultTestConstant"]().run(2.0)[0], 5.0)[0]

    sg = _subgraph()
    payload = lock_subgraph(sg, PASSWORD, vault_id=VAULT_ID, iterations=FAST)
    through_vault = execute_subgraph(unlock_subgraph(payload, PASSWORD, vault_id=VAULT_ID), {})

    assert through_vault["result"] == direct == 7.0


# ── negative cases: these are the real proof ────────────────────────────────

def test_wrong_password_raises():
    # INVARIANT: a wrong password yields nothing, via the GCM tag.
    payload = lock_subgraph(_subgraph(), PASSWORD, vault_id=VAULT_ID, iterations=FAST)
    with pytest.raises(VaultError):
        unlock_subgraph(payload, "wrong password", vault_id=VAULT_ID)


def test_single_flipped_byte_raises():
    # INVARIANT: authenticated encryption - ANY modification is detected, not
    # just a malformed header. Flips a byte deep in the ciphertext.
    payload = lock_subgraph(_subgraph(), PASSWORD, vault_id=VAULT_ID, iterations=FAST)
    raw = bytearray(base64.b64decode(payload))
    raw[-8] ^= 0x01
    with pytest.raises(VaultError):
        unlock_subgraph(base64.b64encode(bytes(raw)).decode(), PASSWORD, vault_id=VAULT_ID)


def test_payload_moved_to_another_vault_raises():
    # INVARIANT: the vault_id is AAD, so a blob cannot be transplanted into a
    # different vault even with the correct password.
    payload = lock_subgraph(_subgraph(), PASSWORD, vault_id=VAULT_ID, iterations=FAST)
    with pytest.raises(VaultError):
        unlock_subgraph(payload, PASSWORD, vault_id="some-other-vault")


def test_header_tampering_is_detected():
    # INVARIANT: the header is authenticated even though it is stored in clear.
    # Lowering the KDF iteration count would make brute force cheap; it must fail.
    payload = lock_subgraph(_subgraph(), PASSWORD, vault_id=VAULT_ID, iterations=FAST)
    raw = bytearray(base64.b64decode(payload))
    i = raw.index(VAULT_ID.encode()) + len(VAULT_ID)
    raw[i:i + 4] = (999).to_bytes(4, "big")          # iterations -> 999
    with pytest.raises(VaultError):
        unlock_subgraph(base64.b64encode(bytes(raw)).decode(), PASSWORD, vault_id=VAULT_ID)


def test_wrong_password_and_tampering_report_the_same_thing():
    # INVARIANT: no oracle. If the two were distinguishable an attacker could
    # confirm a guessed password by the error text alone.
    payload = lock_subgraph(_subgraph(), PASSWORD, vault_id=VAULT_ID, iterations=FAST)
    raw = bytearray(base64.b64decode(payload))
    raw[-1] ^= 0xFF

    with pytest.raises(VaultError) as bad_pw:
        unlock_subgraph(payload, "nope", vault_id=VAULT_ID)
    with pytest.raises(VaultError) as tampered:
        unlock_subgraph(base64.b64encode(bytes(raw)).decode(), PASSWORD, vault_id=VAULT_ID)
    assert str(bad_pw.value) == str(tampered.value)


# ── leakage ─────────────────────────────────────────────────────────────────

def test_payload_contains_no_plaintext():
    # INVARIANT: the blob that ships with the workflow must not carry the
    # password, node class names, widget values, or internal strings.
    sg = _subgraph()
    payload = lock_subgraph(sg, PASSWORD, vault_id=VAULT_ID, iterations=FAST)
    raw = base64.b64decode(payload)
    for secret in (PASSWORD, "SUPERSECRETLAYERNAME", "_VaultTestConstant", "_VaultTestAdd"):
        assert secret.encode() not in raw, f"{secret!r} leaked into the payload"
        assert secret not in payload, f"{secret!r} leaked into the base64 text"


def test_vault_id_is_readable_without_the_password():
    # INVARIANT: the id is authenticated but NOT secret - the UI must be able to
    # say which vault a blob belongs to before prompting.
    payload = lock_subgraph(_subgraph(), PASSWORD, vault_id=VAULT_ID, iterations=FAST)
    assert payload_vault_id(payload) == VAULT_ID


def test_same_plaintext_twice_gives_different_ciphertext():
    # INVARIANT: fresh salt and nonce per lock. Identical output would leak that
    # two vaults hold the same graph, and would be catastrophic for GCM.
    a = lock_subgraph(_subgraph(), PASSWORD, vault_id=VAULT_ID, iterations=FAST)
    b = lock_subgraph(_subgraph(), PASSWORD, vault_id=VAULT_ID, iterations=FAST)
    assert a != b


# ── serialization + node contract ───────────────────────────────────────────

def test_payload_survives_workflow_save_load():
    # INVARIANT: the blob is a plain STRING widget, so it must round-trip through
    # json.dumps/loads byte-identically - that is how it rides in the workflow.
    payload = lock_subgraph(_subgraph(), PASSWORD, vault_id=VAULT_ID, iterations=FAST)
    workflow = {"nodes": [{"type": "C2C_VaultLocked",
                           "widgets_values": [VAULT_ID, payload]}]}
    restored = json.loads(json.dumps(workflow))["nodes"][0]["widgets_values"][1]
    assert restored == payload
    assert unlock_subgraph(restored, PASSWORD, vault_id=VAULT_ID) == _subgraph()


def test_missing_session_key_says_vault_locked_and_nothing_else():
    # INVARIANT: before unlock the node must reveal NOTHING about the contents -
    # not the node count, not the classes, not what is missing.
    from nodes.vault_node import SESSIONS, C2C_VaultLocked

    payload = lock_subgraph(_subgraph(), PASSWORD, vault_id=VAULT_ID, iterations=FAST)
    SESSIONS.drop(VAULT_ID)
    with pytest.raises(RuntimeError) as exc:
        C2C_VaultLocked().execute(vault_id=VAULT_ID, vault_payload=payload)
    msg = str(exc.value)
    assert "Vault locked" in msg
    for leak in ("_VaultTestConstant", "_VaultTestAdd", "SUPERSECRETLAYERNAME", "result"):
        assert leak not in msg, f"locked-node error leaked {leak!r}"


def test_production_iteration_count_is_not_weakened():
    # INVARIANT: the shipped default stays at the OWASP floor. The tests run at
    # 10k for speed; this makes sure that shortcut never becomes the default.
    from nodes.vault_crypto import KDF_ITERATIONS
    assert KDF_ITERATIONS >= 600_000


def test_cycle_in_subgraph_is_reported_not_hung(registry):
    # INVARIANT: a malformed vault must not spin forever inside the executor.
    # Needs the registry: classes are resolved BEFORE the topological sort, so
    # without it the missing-class error fires first and masks the cycle check.
    sg = _subgraph()
    sg["links"].append({"from": "2", "from_slot": 0, "to": "1", "to_slot": 0})
    with pytest.raises(VaultExecError, match="cycle"):
        execute_subgraph(sg, {})


# ── SEALED mode: runs without a password, opens only with one ───────────────

def test_sealed_runs_without_a_password(registry):
    # INVARIANT: the whole point of sealing - a recipient executes it with no
    # prompt, and gets the same answer as running the graph directly.
    payload = seal_subgraph(_subgraph(), PASSWORD, vault_id=VAULT_ID, iterations=FAST)
    out = execute_subgraph(unseal_for_run(payload, vault_id=VAULT_ID), {})
    assert out["result"] == 7.0


def test_sealed_node_execute_needs_no_session_key(registry):
    # INVARIANT: at the NODE level too - no unlock, no session, it just runs.
    from nodes.vault_node import SESSIONS, C2C_VaultSealed
    payload = seal_subgraph(_subgraph(), PASSWORD, vault_id=VAULT_ID, iterations=FAST)
    SESSIONS.drop(VAULT_ID)
    out = C2C_VaultSealed().execute(vault_id=VAULT_ID, vault_payload=payload)
    assert out[0] == 7.0


def test_sealed_opens_with_the_password():
    # INVARIANT: the author can re-open on ANY machine, because the data key is
    # ALSO wrapped under the password - losing the original graph is recoverable.
    payload = seal_subgraph(_subgraph(), PASSWORD, vault_id=VAULT_ID, iterations=FAST)
    assert unseal_with_password(payload, PASSWORD, vault_id=VAULT_ID) == _subgraph()


def test_sealed_open_refuses_a_wrong_password():
    # INVARIANT: "runs freely" must NOT mean "readable by anyone". Running and
    # opening are separate rights; only running is free.
    payload = seal_subgraph(_subgraph(), PASSWORD, vault_id=VAULT_ID, iterations=FAST)
    with pytest.raises(VaultError):
        unseal_with_password(payload, "wrong", vault_id=VAULT_ID)


def test_sealed_rejects_tamper_and_vault_swap():
    # INVARIANT: same AAD guarantees as locked mode - the header binds vault_id,
    # KDF params and the boundary, so neither editing nor transplanting works.
    payload = seal_subgraph(_subgraph(), PASSWORD, vault_id=VAULT_ID, iterations=FAST)
    with pytest.raises(VaultError):
        unseal_for_run(payload, vault_id="other-vault")
    raw = bytearray(base64.b64decode(payload)); raw[-4] ^= 0x01
    with pytest.raises(VaultError):
        unseal_for_run(base64.b64encode(bytes(raw)).decode(), vault_id=VAULT_ID)


def test_sealed_payload_contains_no_plaintext():
    # INVARIANT: sealing is weaker than locking, but the blob must STILL not
    # carry readable internals - that is the copying it is meant to deter.
    payload = seal_subgraph(_subgraph(), PASSWORD, vault_id=VAULT_ID, iterations=FAST)
    raw = base64.b64decode(payload)
    for secret in (PASSWORD, "SUPERSECRETLAYERNAME", "_VaultTestConstant", "_VaultTestAdd"):
        assert secret.encode() not in raw
        assert secret not in payload


def test_modes_are_distinguishable_without_a_key():
    # INVARIANT: the UI must know which node type a blob belongs in before it can
    # prompt for anything.
    locked = lock_subgraph(_subgraph(), PASSWORD, vault_id=VAULT_ID, iterations=FAST)
    sealed = seal_subgraph(_subgraph(), PASSWORD, vault_id=VAULT_ID, iterations=FAST)
    assert payload_mode(locked) == MODE_LOCKED
    assert payload_mode(sealed) == MODE_SEALED


def test_locked_payload_cannot_be_run_as_sealed():
    # INVARIANT: a LOCKED vault must never be auto-openable by swapping it into a
    # sealed node - that would strip the password requirement entirely.
    locked = lock_subgraph(_subgraph(), PASSWORD, vault_id=VAULT_ID, iterations=FAST)
    with pytest.raises(VaultError, match="not a sealed vault"):
        unseal_for_run(locked, vault_id=VAULT_ID)


# -- boundary wiring -----------------------------------------------------------
#
# These pin the bug that made every UI-locked vault unrunnable: js/c2c_vault.js
# discarded the links crossing the selection edge and posted
# boundary_in: [], boundary_out: [], so the executor saw a subgraph with no
# declared outputs and refused it. The Python side was fine; nothing exercised
# the shape the UI actually sent.

def _open_subgraph():
    """Add(+5) whose FIRST input arrives from outside, as a boundary input."""
    return {
        "nodes": [{"id": "2", "class_type": "_VaultTestAdd", "widgets": {"addend": 5.0}}],
        "links": [],
        "boundary_in": [{"name": "in_0", "to": "2", "to_slot": 0}],
        "boundary_out": [{"name": "out_0", "from": "2", "from_slot": 0}],
    }


def test_boundary_input_is_injected_into_the_named_slot(registry):
    # INVARIANT: a value supplied at the boundary reaches the right slot of the
    # right node. Off-by-one here would silently feed it to 'addend' instead.
    out = execute_subgraph(_open_subgraph(), {"in_0": 10.0})
    assert out["out_0"] == pytest.approx(15.0)


def test_missing_boundary_input_is_named_not_swallowed(registry):
    # INVARIANT: an unsupplied boundary input says WHICH one, rather than
    # surfacing a TypeError about missing keyword arguments.
    with pytest.raises(VaultExecError, match="in_0"):
        execute_subgraph(_open_subgraph(), {})


def test_no_declared_outputs_is_empty_here_and_loud_one_level_up(registry):
    # INVARIANT: the executor is NOT self-guarding - it returns {} for a
    # subgraph with no outputs. That is exactly what the UI used to build, and
    # {} would read as a working vault that produced nothing. The loud rejection
    # lives one level up, in vault_node._run, so pin that it is still there.
    sg = _subgraph()
    sg["boundary_out"] = []
    assert execute_subgraph(sg, {}) == {}
    node_src = (PACK_ROOT / "nodes" / "vault_node.py").read_text(encoding="utf-8")
    assert "the subgraph declares no outputs" in node_src


def test_boundary_hash_covers_inputs_and_outputs_separately():
    # INVARIANT: the boundary is authenticated, so swapping an input for an
    # output of the same name must not collide.
    a = boundary_hash([{"name": "x", "to": "1", "to_slot": 0}], [])
    b = boundary_hash([], [{"name": "x", "from": "1", "from_slot": 0}])
    assert a != b


# -- JS/Python parity ----------------------------------------------------------

def test_js_input_cap_matches_the_declared_sockets():
    # INVARIANT: MAX_VAULT_INPUTS in the JS gates the selection at lock time;
    # input_0..N in the node declares what can actually be wired. If they drift,
    # the UI happily locks a selection the node cannot accept.
    import re
    js = (PACK_ROOT / "js" / "c2c_vault.js").read_text(encoding="utf-8")
    node = (PACK_ROOT / "nodes" / "vault_node.py").read_text(encoding="utf-8")
    m_in = re.search(r"MAX_VAULT_INPUTS\s*=\s*(\d+)", js)
    m_out = re.search(r"MAX_VAULT_OUTPUTS\s*=\s*(\d+)", js)
    assert m_in, "MAX_VAULT_INPUTS not found in c2c_vault.js"
    assert m_out, "MAX_VAULT_OUTPUTS not found in c2c_vault.js"
    # Count the LIVE schema, not source text. The sockets are generated in a
    # loop (vault_node.py `optional[f"input_{i}"]`), so grepping for a literal
    # "input_0" measured 0 and the test failed for a reason that had nothing to
    # do with the contract it exists to protect.
    from nodes.vault_node import C2C_VaultLocked, MAX_VAULT_INPUTS, MAX_VAULT_OUTPUTS
    spec = C2C_VaultLocked.INPUT_TYPES()
    declared_in = len([k for k in (spec.get("optional") or {})
                       if re.fullmatch(r"input_\d+", k)])
    declared_out = len(C2C_VaultLocked.RETURN_NAMES)

    assert int(m_in.group(1)) == declared_in, (
        "js MAX_VAULT_INPUTS=" + m_in.group(1) + " but INPUT_TYPES declares "
        + str(declared_in) + " input sockets"
    )
    assert int(m_out.group(1)) == declared_out, (
        "js MAX_VAULT_OUTPUTS=" + m_out.group(1) + " but RETURN_NAMES declares "
        + str(declared_out) + " outputs"
    )
    assert MAX_VAULT_INPUTS == declared_in
    assert MAX_VAULT_OUTPUTS == declared_out
    # RETURN_TYPES must stay the same length as RETURN_NAMES or ComfyUI mislabels
    # every socket after the first mismatch.
    assert len(C2C_VaultLocked.RETURN_TYPES) == declared_out


def test_js_no_longer_posts_empty_boundaries():
    # INVARIANT: the regression itself. Hardcoded empty boundary lists in the
    # lock request are what made every UI-locked vault unrunnable.
    js = (PACK_ROOT / "js" / "c2c_vault.js").read_text(encoding="utf-8")
    assert "boundary_in: [], boundary_out: []" not in js
    assert "boundary_in, boundary_out" in js


def test_lock_without_password_raises():
    # INVARIANT: locking requires a password — empty string must fail at crypto.
    with pytest.raises(VaultError):
        lock_subgraph(_subgraph(), "", vault_id=VAULT_ID, iterations=FAST)


def test_derive_boundary_three_node_two_in_one_out():
    # INVARIANT: boundary derivation classifies 2 entering wires and 1 leaving.
    sel = {"1", "2", "3"}
    inputs_by_node = {
        "1": [{"slot": 0, "origin_id": "ext_a", "origin_slot": 0}],
        "2": [{"slot": 0, "origin_id": "ext_b", "origin_slot": 0}],
        "3": [{"slot": 0, "origin_id": "1", "origin_slot": 0}],
    }
    outputs_by_node = {
        "1": [{"slot": 0, "targets": []}],
        "2": [{"slot": 0, "targets": []}],
        "3": [{"slot": 0, "targets": [{"target_id": "ext_c", "target_slot": 0}]}],
    }
    links, b_in, b_out = derive_boundary(sel, inputs_by_node, outputs_by_node)
    assert len(b_in) == 2
    assert len(b_out) == 1
    assert any(l["from"] == "1" and l["to"] == "3" for l in links)


def test_workflow_save_load_restores_interface_manifest():
    # INVARIANT: vault_interface round-trips through json and restores socket names.
    iface = build_interface_manifest(
        "locked", 3,
        [{"name": "driver", "to": "1", "to_slot": 0}],
        [{"name": "result", "from": "2", "from_slot": 0}],
        in_types=["IMAGE"], out_types=["IMAGE"],
    )
    payload = lock_subgraph(_subgraph(), PASSWORD, vault_id=VAULT_ID, iterations=FAST)
    workflow = {
        "nodes": [{
            "type": "C2C_VaultLocked",
            "inputs": [{"name": "driver", "type": "IMAGE", "link": None}],
            "outputs": [{"name": "result", "type": "IMAGE", "links": []}],
            "properties": {"vault_slots": {"in": 1, "out": 1}},
            "widgets_values": [VAULT_ID, payload, json.dumps(iface)],
        }],
    }
    restored = json.loads(json.dumps(workflow))["nodes"][0]
    assert json.loads(restored["widgets_values"][2])["in"][0]["name"] == "driver"
    assert restored["inputs"][0]["name"] == "driver"
    assert restored["properties"]["vault_slots"]["out"] == 1


def test_multi_output_boundary_mapping(registry):
    # INVARIANT: _run returns every boundary_out value, not just the first.
    from nodes.vault_node import C2C_VaultSealed

    class _Twin:
        FUNCTION = "run"
        RETURN_TYPES = ("FLOAT", "FLOAT")

        @classmethod
        def INPUT_TYPES(cls):
            return {"required": {"value": ("FLOAT", {"default": 1.0})}}

        def run(self, value):
            return (float(value), float(value) * 2.0)

    mod = sys.modules["nodes"]
    mapping = dict(mod.NODE_CLASS_MAPPINGS)
    mapping["_VaultTestTwin"] = _Twin
    mod.NODE_CLASS_MAPPINGS = mapping

    sg = {
        "nodes": [{"id": "1", "class_type": "_VaultTestTwin", "widgets": {"value": 3.0}}],
        "links": [],
        "boundary_in": [],
        "boundary_out": [
            {"name": "a", "from": "1", "from_slot": 0},
            {"name": "b", "from": "1", "from_slot": 1},
        ],
    }
    payload = seal_subgraph(sg, PASSWORD, vault_id=VAULT_ID, iterations=FAST)
    result = C2C_VaultSealed().execute(vault_id=VAULT_ID, vault_payload=payload, vault_interface="{}")
    assert result[0] == 3.0
    assert result[1] == 6.0
    assert all(v is None for v in result[2:])
