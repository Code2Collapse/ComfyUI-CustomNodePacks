"""C2C Vault v2 on the node (A17): run in every mode, the edit lifecycle (password -> code -> graph -> verified
re-lock), lock-time refusals, hidden inputs. In-memory secrets store, fast KDF, fake nodes (test_vault's registry).
"""

import base64
import json

import pytest

pytest.importorskip("cryptography")

from nodes import vault_2fa, vault_crypto_v2 as v2, vault_node, vault_store  # noqa: E402
from nodes.vault_crypto import VaultError, seal_subgraph  # noqa: E402
from tests.test_vault import _subgraph, registry  # noqa: E402,F401
from tests.test_vault_v2 import FAST, T0, code_at  # noqa: E402

PASSWORD = "correct horse battery staple"
IFACE = {"mode": "locked", "in": [], "out": [{"name": "result", "type": "FLOAT"}], "params": []}


@pytest.fixture(autouse=True)
def fast(monkeypatch):
    monkeypatch.setattr(vault_store, "BACKEND", {})
    monkeypatch.setattr(v2, "ARGON2_V1", dict(FAST))
    vault_2fa._pending.clear()
    vault_node.SESSIONS.drop()
    yield
    vault_node.SESSIONS.drop()


def lock(mode="locked", vault_id="vault-n1", recipients=None, sub=None):
    body = {"vault_id": vault_id, "password": PASSWORD, "mode": mode, "subgraph": sub or _subgraph(),
            "layout": {"nodes": [{"id": 1, "type": "_VaultTestConstant", "pos": [0, 0]}]}, "interface": IFACE,
            "recipients": recipients}
    return vault_node.lock_v2(body)


def run(node_cls, payload, vault_id="vault-n1", iface=IFACE):
    return node_cls().execute(vault_id, payload, json.dumps(iface), unique_id="77")[0]


def test_locked_v2_runs_only_with_a_session(registry):
    payload, kek, mode = lock()
    assert mode == v2.MODE_LOCKED and v2.is_v2(payload)
    with pytest.raises(RuntimeError, match="Vault locked"):
        run(vault_node.C2C_VaultLocked, payload)
    vault_node._open_session("vault-n1", payload, PASSWORD)
    assert run(vault_node.C2C_VaultLocked, payload) == 7.0


def test_a_changed_public_interface_stops_the_run(registry):
    payload, kek, _ = lock()
    vault_node.SESSIONS.put("vault-n1", kek)
    bad = json.loads(json.dumps(IFACE))
    bad["out"][0]["name"] = "renamed"
    with pytest.raises(RuntimeError, match="public interface was changed"):
        run(vault_node.C2C_VaultLocked, payload, iface=bad)


def test_sealed_for_this_install_runs_without_a_password(registry):
    payload, _, mode = lock(mode="sealed")
    assert mode == v2.MODE_SEALED_INSTALLS
    assert run(vault_node.C2C_VaultSealed, payload) == 7.0


def test_sealed_for_another_install_does_not_run_here(registry, monkeypatch):
    friend = "c2cvk1:" + base64.b64encode(b"\x11" * 32).decode()
    payload, _, _ = lock(mode="sealed", recipients=[friend])
    monkeypatch.setattr(vault_store, "BACKEND", {})                    # a different install: a different key
    from nodes import vault_keys
    vault_keys.install_private_key()
    with pytest.raises(RuntimeError, match="not sealed for this install"):
        run(vault_node.C2C_VaultSealed, payload)


def test_portable_seal_runs_anywhere(registry, monkeypatch):
    payload, _, mode = lock(mode="sealed_portable")
    monkeypatch.setattr(vault_store, "BACKEND", {})
    assert mode == v2.MODE_SEALED_PORTABLE and run(vault_node.C2C_VaultSealed, payload) == 7.0


def test_a_legacy_seal_still_runs_and_says_it_is_obfuscation(registry, monkeypatch):
    warned = []
    monkeypatch.setattr(vault_node, "_notify", lambda uid, text: warned.append((uid, text)))
    old = seal_subgraph(_subgraph(), PASSWORD, vault_id="vault-n1", iterations=10_000)
    assert run(vault_node.C2C_VaultSealed, old, iface={}) == 7.0
    assert warned and "legacy seal" in warned[0][1] and warned[0][0] == "77"


def test_a_locked_payload_in_a_sealed_node_is_refused(registry):
    payload, _, _ = lock()
    with pytest.raises(RuntimeError, match="needs a C2C Vault Locked node"):
        run(vault_node.C2C_VaultSealed, payload)


# ── editing ────────────────────────────────────────────────────────────────────────────────────────────────────────

def enrol():
    """Enrolled one step in the past, so the codes the tests then use at the real time are fresh."""
    import time as _t
    info = vault_2fa.begin_enrolment()
    then = _t.time() - 60
    assert vault_2fa.confirm_enrolment(code_at(info["secret_base32"], then), now=then)
    return info["secret_base32"]


def test_editing_needs_the_password_then_a_fresh_code(registry, monkeypatch):
    secret = enrol()
    payload, _, _ = lock()
    with pytest.raises(VaultError):
        vault_node.EDITS.begin("vault-n1", payload, "wrong password")
    edit_id = vault_node.EDITS.begin("vault-n1", payload, PASSWORD)
    import time as _t
    now = _t.time()
    with pytest.raises(PermissionError, match="not right"):
        vault_node.EDITS.verify(edit_id, "000000" if code_at(secret, now) != "000000" else "111111")
    result = vault_node.EDITS.verify(edit_id, code_at(secret, now))
    assert result["exec"]["nodes"][1]["class_type"] == "_VaultTestAdd" and result["layout"]["nodes"][0]["id"] == 1
    with pytest.raises(PermissionError, match="expired"):
        vault_node.EDITS.verify(edit_id, code_at(secret, now))          # the password step is single-use


def test_a_run_session_never_opens_a_vault_for_editing(registry):
    import asyncio

    payload, kek, _ = lock()
    vault_node.SESSIONS.put("vault-n1", kek)

    class Req:
        async def json(self):
            return {"vault_id": "vault-n1", "payload": payload}

    routes = {}

    class Server:
        class routes_cls:
            def post(self, path):
                return lambda fn: routes.setdefault(path, fn)

            def get(self, path):
                return lambda fn: routes.setdefault(path, fn)
        routes = routes_cls()

    monkey = vault_node._ROUTES_REGISTERED
    vault_node._ROUTES_REGISTERED = False
    try:
        vault_node.register_routes(Server())
    finally:
        vault_node._ROUTES_REGISTERED = monkey or vault_node._ROUTES_REGISTERED
    resp = asyncio.run(routes["/c2c_vault/open"](Req()))
    assert resp.status == 410 and b"second factor" in resp.body


def test_commit_re_encrypts_verifies_and_the_editor_can_run_it(registry):
    secret = enrol()
    payload, _, _ = lock()
    import time as _t
    result = vault_node.EDITS.verify(vault_node.EDITS.begin("vault-n1", payload, PASSWORD), code_at(secret, _t.time()))
    edited = json.loads(json.dumps(result["exec"]))
    edited["nodes"][1]["widgets"]["addend"] = 10.0
    new = vault_node.EDITS.commit(result["edit_token"], edited, result["layout"], IFACE, None)
    assert new != payload and v2.parse(new)["body"][0] != v2.parse(payload)["body"][0]
    assert run(vault_node.C2C_VaultLocked, new) == 12.0                  # the editor's session runs it at once
    out, _, _ = v2.open_with_password(new, PASSWORD)
    assert out["exec"]["nodes"][1]["widgets"]["addend"] == 10.0
    with pytest.raises(PermissionError, match="expired"):
        vault_node.EDITS.commit(result["edit_token"], edited, None, IFACE, None)   # one commit per session


def test_a_legacy_seal_is_upgraded_to_v2_on_commit(registry):
    secret = enrol()
    old = seal_subgraph(_subgraph(), PASSWORD, vault_id="vault-n1", iterations=10_000)
    import time as _t
    result = vault_node.EDITS.verify(vault_node.EDITS.begin("vault-n1", old, PASSWORD), code_at(secret, _t.time()))
    assert result["legacy"] is True and result["mode"] == v2.MODE_SEALED_INSTALLS
    new = vault_node.EDITS.commit(result["edit_token"], result["exec"], None, IFACE, None)
    assert v2.is_v2(new) and v2.parse(new)["mode"] == v2.MODE_SEALED_INSTALLS
    assert run(vault_node.C2C_VaultSealed, new) == 7.0


def test_unsupported_inner_nodes_are_refused_at_lock(registry):
    class Lister:
        FUNCTION = "run"
        RETURN_TYPES = ("FLOAT",)
        INPUT_IS_LIST = True

        @classmethod
        def INPUT_TYPES(cls):
            return {"required": {"a": ("FLOAT", {})}}

    class Lazy:
        FUNCTION = "run"
        RETURN_TYPES = ("FLOAT",)

        @classmethod
        def INPUT_TYPES(cls):
            return {"required": {"a": ("FLOAT", {"lazy": True})}, "hidden": {"x": "AUTH_TOKEN_COMFY_ORG"}}

    registry.update({"_Lister": Lister, "_Lazy": Lazy})
    sub = _subgraph()
    sub["nodes"] += [{"id": "3", "class_type": "_Lister", "widgets": {}}, {"id": "4", "class_type": "_Lazy",
                                                                           "widgets": {}}]
    with pytest.raises(VaultError) as e:
        lock(sub=sub)
    msg = str(e.value)
    assert "list inputs" in msg and "lazy input 'a'" in msg and "AUTH_TOKEN_COMFY_ORG" in msg


def test_inner_nodes_get_the_vaults_id_and_no_prompt(registry):
    seen = {}

    class Hidden:
        FUNCTION = "run"
        RETURN_TYPES = ("FLOAT",)

        @classmethod
        def INPUT_TYPES(cls):
            return {"required": {"a": ("FLOAT", {})}, "hidden": {"uid": "UNIQUE_ID", "prompt": "PROMPT",
                                                                 "info": "EXTRA_PNGINFO"}}

        def run(self, a, uid, prompt, info):
            seen.update(uid=uid, prompt=prompt, info=info)
            return (a,)

    registry["_Hidden"] = Hidden
    sub = _subgraph()
    sub["nodes"].append({"id": "3", "class_type": "_Hidden", "widgets": {}})
    sub["links"].append({"from": "2", "from_slot": 0, "to": "3", "to_name": "a"})
    sub["boundary_out"] = [{"name": "result", "from": "3", "from_slot": 0}]
    payload, kek, _ = lock(sub=sub)
    vault_node.SESSIONS.put("vault-n1", kek)
    assert run(vault_node.C2C_VaultLocked, payload) == 7.0
    assert seen == {"uid": "77", "prompt": None, "info": None}


def test_a_password_change_re_locks_under_the_new_password_only(registry):
    secret = enrol()
    payload, _, _ = lock()
    import time as _t
    result = vault_node.EDITS.verify(vault_node.EDITS.begin("vault-n1", payload, PASSWORD), code_at(secret, _t.time()))
    old_salt = v2.parse(payload)["salt"]
    new = vault_node.EDITS.commit(result["edit_token"], result["exec"], result["layout"], IFACE, None,
                                  new_password="a brand new passphrase")
    assert v2.parse(new)["salt"] != old_salt                             # a new KEK, not a re-wrap under the old one
    with pytest.raises(VaultError):
        v2.open_with_password(new, PASSWORD, vault_id="vault-n1")
    out, _, _ = v2.open_with_password(new, "a brand new passphrase", vault_id="vault-n1")
    assert out["exec"] == result["exec"]
    assert run(vault_node.C2C_VaultLocked, new) == 7.0                   # the editor's session holds the new key
    vault_node.SESSIONS.drop()
    vault_node._open_session("vault-n1", new, "a brand new passphrase")
    assert run(vault_node.C2C_VaultLocked, new) == 7.0


def _outer_around(inner_cls, inner_id, inner_payload):
    """An outer graph: the inner vault node -> Add(+1). Unvaulted, the same graph gives (2 + 5) + 1 = 8."""
    return {"nodes": [{"id": "1", "class_type": inner_cls, "widgets": {
                          "vault_id": inner_id, "vault_payload": inner_payload, "vault_interface": json.dumps(IFACE)}},
                      {"id": "2", "class_type": "_VaultTestAdd", "widgets": {"addend": 1.0}}],
            "links": [{"from": "1", "from_slot": 0, "to": "2", "to_slot": 0}],
            "boundary_in": [], "boundary_out": [{"name": "result", "from": "2", "from_slot": 0}]}


def test_a_sealed_vault_runs_inside_a_locked_vault(registry, monkeypatch):
    registry.update({"C2C_VaultSealed": vault_node.C2C_VaultSealed, "C2C_VaultLocked": vault_node.C2C_VaultLocked})
    inner, _, _ = lock(mode="sealed_portable", vault_id="vault-inner")
    outer, kek, _ = lock(vault_id="vault-n1", sub=_outer_around("C2C_VaultSealed", "vault-inner", inner))
    assert inner not in outer and "_VaultTestConstant" not in outer      # the inner vault rides inside the ciphertext
    vault_node.SESSIONS.put("vault-n1", kek)
    assert run(vault_node.C2C_VaultLocked, outer) == 8.0


def test_a_locked_vault_inside_a_vault_needs_its_own_key(registry):
    registry.update({"C2C_VaultLocked": vault_node.C2C_VaultLocked})
    inner, inner_kek, _ = lock(vault_id="vault-inner")
    outer, kek, _ = lock(vault_id="vault-n1", sub=_outer_around("C2C_VaultLocked", "vault-inner", inner))
    vault_node.SESSIONS.drop()
    vault_node.SESSIONS.put("vault-n1", kek)                            # the outer key alone opens only the outer
    with pytest.raises(RuntimeError, match="Vault locked"):
        run(vault_node.C2C_VaultLocked, outer)
    vault_node.SESSIONS.put("vault-inner", inner_kek)
    assert run(vault_node.C2C_VaultLocked, outer) == 8.0


def test_an_async_inner_node_gives_the_same_result_as_unvaulted(registry):
    class AsyncDouble:
        FUNCTION = "run"
        RETURN_TYPES = ("FLOAT",)

        @classmethod
        def INPUT_TYPES(cls):
            return {"required": {"a": ("FLOAT", {})}}

        async def run(self, a):
            import asyncio
            await asyncio.sleep(0)
            return (a * 2.0,)

    registry["_AsyncDouble"] = AsyncDouble
    sub = _subgraph()
    sub["nodes"].append({"id": "3", "class_type": "_AsyncDouble", "widgets": {}})
    sub["links"].append({"from": "2", "from_slot": 0, "to": "3", "to_name": "a"})
    sub["boundary_out"] = [{"name": "result", "from": "3", "from_slot": 0}]
    from nodes.vault_exec import execute_subgraph
    direct = execute_subgraph(json.loads(json.dumps(sub)), {})
    payload, kek, _ = lock(sub=sub)
    vault_node.SESSIONS.put("vault-n1", kek)
    assert run(vault_node.C2C_VaultLocked, payload) == direct["result"] == 14.0
