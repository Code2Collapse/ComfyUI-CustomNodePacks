"""C2C Vault v2 (A17): envelope, install keys, second factor. No network, no OS credential store (in-memory store).

A cheaper Argon2id profile (64 MiB, the floor) keeps the suite fast; the production profile is pinned separately.
"""

import base64
import json
import struct

import pytest

pytest.importorskip("cryptography")

from nodes import vault_2fa, vault_crypto_v2 as v2, vault_keys, vault_store  # noqa: E402
from nodes.vault_crypto import VaultError  # noqa: E402

FAST = {"memory_kib": 65536, "iterations": 2, "lanes": 1}
SENTINEL = "SENTINEL-ZEBRA-7741"


@pytest.fixture(autouse=True)
def memory_store(monkeypatch):
    monkeypatch.setattr(vault_store, "BACKEND", {})
    vault_2fa._pending.clear()
    yield


def plain(extra=""):
    ex = {"nodes": [{"id": "1", "class_type": "CLIPTextEncode", "widgets": {"text": SENTINEL + extra}}],
          "links": [], "boundary_in": [{"name": "clip", "to": "1", "to_name": "clip"}],
          "boundary_out": [{"name": "cond", "from": "1", "from_slot": 0}]}
    return {"exec": ex, "layout": {"nodes": [{"id": 1, "type": "CLIPTextEncode", "title": SENTINEL + "-title"}]}}


IFACE = {"mode": "locked", "in": [{"name": "clip", "type": "CLIP"}], "out": [{"name": "cond", "type": "CONDITIONING"}],
         "params": [{"id": "p0", "name": "steps", "type": "INT", "value": 20}]}


def lock(mode=v2.MODE_LOCKED, password="pässwörd ✓", recipients=None):
    return v2.lock(plain(), password, vault_id="vault-t1", mode=mode, iface=IFACE, recipients=recipients, kdf=FAST)


# ── envelope ───────────────────────────────────────────────────────────────────────────────────────────────────────

def test_round_trip_locked_and_no_plaintext_in_the_payload():
    payload, kek = lock()
    assert SENTINEL not in payload and SENTINEL not in base64.b64decode(payload).decode("latin-1")
    out, kek2, p = v2.open_with_password(payload, "pässwörd ✓", vault_id="vault-t1")
    assert out["exec"]["nodes"][0]["widgets"]["text"] == SENTINEL and kek2 == kek
    assert v2.open_with_kek(payload, kek)["layout"]["nodes"][0]["title"] == SENTINEL + "-title"


def test_unicode_passwords_are_normalised_not_truncated():
    payload, _ = lock(password="é" * 40)                          # e + combining accent, decomposed
    v2.open_with_password(payload, "é" * 40)                     # the precomposed form opens it
    with pytest.raises(VaultError):
        v2.open_with_password(payload, "é" * 39)                 # one character short: not truncated


def test_wrong_password_and_every_tamper_give_the_same_message():
    payload, _ = lock()
    blob = bytearray(base64.b64decode(payload))
    messages = set()
    with pytest.raises(VaultError) as e:
        v2.open_with_password(payload, "wrong")
    messages.add(str(e.value))
    header_len = len(v2.parse(payload)["header"])
    for pos in (header_len + 3, header_len + 40, len(blob) - 5, len(blob) - 30):    # wraps, body, tag
        bad = bytearray(blob)
        bad[pos] ^= 0x01
        with pytest.raises(VaultError) as e:
            v2.open_with_password(base64.b64encode(bytes(bad)).decode(), "pässwörd ✓")
        messages.add(str(e.value))
    assert len(messages) == 1


def test_header_fields_are_authenticated():
    payload, _ = lock()
    p = v2.parse(payload)
    blob = bytearray(base64.b64decode(payload))
    salt_at = p["header"].index(p["salt"])
    for pos in (salt_at, salt_at + 16, salt_at + 48, len(p["header"]) - 1):    # salt, boundary hash, interface hash
        bad = bytearray(blob)
        bad[pos] ^= 0x80
        with pytest.raises(VaultError):
            v2.open_with_password(base64.b64encode(bytes(bad)).decode(), "pässwörd ✓")


def test_kdf_cannot_be_downgraded_or_inflated():
    payload, _ = lock()
    blob = bytearray(base64.b64decode(payload))
    p = v2.parse(payload)
    kdf_at = len(v2.MAGIC2) + 3 + len(p["vault_id"]) + 1
    for mem in (1024, 4 * 1024 * 1024):
        bad = bytearray(blob)
        bad[kdf_at:kdf_at + 4] = struct.pack("!I", mem)
        with pytest.raises(VaultError, match="KDF parameters out of range"):
            v2.parse(base64.b64encode(bytes(bad)).decode())


def test_truncated_oversized_and_malformed_payloads_are_refused_before_decryption():
    payload, _ = lock()
    blob = base64.b64decode(payload)
    for cut in (5, 20, 60, len(blob) - 10):
        with pytest.raises(VaultError):
            v2.open_with_password(base64.b64encode(blob[:cut]).decode(), "pässwörd ✓")
    with pytest.raises(VaultError, match="too large"):
        v2.parse("A" * (v2.MAX_PAYLOAD_BYTES * 2))
    for junk in ("", "not base64 !!", base64.b64encode(b"C2CVLT1xxxx").decode()):
        with pytest.raises(VaultError):
            v2.parse(junk)


def test_every_lock_has_fresh_salt_dek_and_nonces():
    a, _ = lock()
    b, _ = lock()
    pa, pb = v2.parse(a), v2.parse(b)
    assert pa["salt"] != pb["salt"] and pa["pw"][0] != pb["pw"][0] and pa["body"][0] != pb["body"][0]
    assert pa["body"][1] != pb["body"][1]


def test_the_vault_id_binds_the_payload():
    payload, _ = lock()
    with pytest.raises(VaultError, match="does not belong"):
        v2.open_with_password(payload, "pässwörd ✓", vault_id="vault-other")


def test_relock_keeps_the_password_and_changes_all_material():
    payload, kek = lock()
    edited = plain("-edited")
    new = v2.relock(payload, kek, edited, iface=IFACE)
    assert v2.parse(new)["body"][0] != v2.parse(payload)["body"][0]
    out, _, _ = v2.open_with_password(new, "pässwörd ✓")
    assert out["exec"]["nodes"][0]["widgets"]["text"].endswith("-edited")
    with pytest.raises(VaultError):
        v2.relock(payload, b"\x00" * 32, edited, iface=IFACE)       # a foreign key cannot re-lock


def test_the_public_interface_is_authenticated():
    payload, _ = lock()
    p = v2.parse(payload)
    v2.check_interface(p, IFACE)
    renamed = json.loads(json.dumps(IFACE))
    renamed["out"][0]["type"] = "IMAGE"
    with pytest.raises(VaultError, match="public interface was changed"):
        v2.check_interface(p, renamed)
    valued = json.loads(json.dumps(IFACE))
    valued["params"][0]["value"] = 35                                   # a value is not part of it
    valued["params"][0]["socket"] = 2                                   # nor converting it to an input
    v2.check_interface(p, valued)
    relabelled = json.loads(json.dumps(IFACE))
    relabelled["params"][0]["name"] = "denoise"
    with pytest.raises(VaultError, match="public interface was changed"):
        v2.check_interface(p, relabelled)


def test_production_kdf_profile_is_pinned():
    assert v2.ARGON2_V1 == {"memory_kib": 131072, "iterations": 3, "lanes": 4}
    v2.check_kdf(v2.ARGON2_V1)


# ── sealed for installs / portable ─────────────────────────────────────────────────────────────────────────────────

def test_sealed_for_installs_runs_only_where_it_was_sealed_for():
    mine = vault_keys.install_private_key()
    other = vault_keys._raw_public(base64.b64decode(base64.b64encode(b"\x07" * 32)))
    payload, _ = lock(mode=v2.MODE_SEALED_INSTALLS, recipients=[vault_keys.install_public_key()])
    assert v2.open_with_install_key(payload, mine)["exec"]["nodes"][0]["widgets"]["text"] == SENTINEL
    with pytest.raises(VaultError, match="not sealed for this install"):
        v2.open_with_install_key(payload, b"\x07" * 32)
    assert v2.info(payload)["installs"] == [v2.key_id(vault_keys.install_public_key()).hex()]
    assert other != vault_keys.install_public_key()


def test_a_reseal_after_editing_keeps_the_same_installs():
    mine = vault_keys.install_private_key()
    friend_priv = b"\x05" * 32
    friend = vault_keys._raw_public(friend_priv)
    payload, kek = lock(mode=v2.MODE_SEALED_INSTALLS, recipients=[vault_keys.install_public_key(), friend])
    out, kek, _ = v2.open_with_password(payload, "pässwörd ✓")
    new = v2.relock(payload, kek, plain("-x"), iface=IFACE, recipients=v2.recipients_of(out))
    for priv in (mine, friend_priv):
        assert v2.open_with_install_key(new, priv)["exec"]["nodes"][0]["widgets"]["text"].endswith("-x")
    assert len(v2.info(new)["installs"]) == 2


def test_sealed_for_installs_needs_an_install():
    with pytest.raises(VaultError, match="at least one install"):
        lock(mode=v2.MODE_SEALED_INSTALLS, recipients=[])


def test_portable_seal_runs_anywhere_and_still_needs_the_password_to_edit_flow():
    payload, _ = lock(mode=v2.MODE_SEALED_PORTABLE)
    assert v2.open_portable(payload)["exec"]["nodes"][0]["widgets"]["text"] == SENTINEL
    with pytest.raises(VaultError):
        v2.open_with_install_key(payload, vault_keys.install_private_key())
    with pytest.raises(VaultError, match="not a portable seal"):
        v2.open_portable(lock()[0])


def test_install_key_text_round_trip_and_validation():
    pub = vault_keys.install_public_key()
    assert vault_keys.parse_public_text(vault_keys.public_text(pub)) == pub
    for bad in ("", "c2cvk1:###", "c2cvk1:" + base64.b64encode(b"x" * 31).decode(), "ssh-rsa AAA"):
        with pytest.raises(VaultError):
            vault_keys.parse_public_text(bad)
    assert vault_keys.install_private_key() == vault_keys.install_private_key()     # created once


# ── second factor ──────────────────────────────────────────────────────────────────────────────────────────────────

T0 = 1_800_000_000.0


def code_at(secret_b32, t):
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.twofactor.totp import TOTP

    secret = base64.b32decode(secret_b32 + "=" * (-len(secret_b32) % 8))
    return TOTP(secret, 6, hashes.SHA1(), 30, enforce_key_length=False).generate(int(t)).decode()


def enrol():
    info = vault_2fa.begin_enrolment()
    assert vault_2fa.confirm_enrolment(code_at(info["secret_base32"], T0), now=T0)
    return info


def test_enrolment_needs_a_first_code_and_stores_no_plain_recovery_code():
    info = vault_2fa.begin_enrolment()
    assert not vault_2fa.enrolled()
    good = code_at(info["secret_base32"], T0)
    wrong = str((int(good) + 500000) % 1000000).zfill(6)
    assert vault_2fa.confirm_enrolment(wrong, now=T0) is False and not vault_2fa.enrolled()
    assert vault_2fa.confirm_enrolment(good, now=T0)
    assert vault_2fa.enrolled() and info["otpauth_uri"].startswith("otpauth://totp/")
    dump = json.dumps(vault_store.BACKEND)
    assert not any(c in dump for c in info["recovery_codes"])
    with pytest.raises(PermissionError, match="already set up"):
        vault_2fa.begin_enrolment()


def test_a_code_works_once_and_its_neighbours_are_accepted():
    info = enrol()
    s = info["secret_base32"]
    assert vault_2fa.verify(code_at(s, T0 + 60), now=T0 + 60)
    assert not vault_2fa.verify(code_at(s, T0 + 60), now=T0 + 60)           # replay
    assert vault_2fa.verify(code_at(s, T0 + 120), now=T0 + 150)             # one step late is fine
    assert not vault_2fa.verify(code_at(s, T0 + 300), now=T0 + 420)         # four steps late is not


def test_wrong_codes_lock_the_second_factor_and_it_persists():
    info = enrol()
    for _ in range(vault_2fa.MAX_FAILS):
        assert not vault_2fa.verify("123456", now=T0 + 30)
    with pytest.raises(PermissionError, match="Too many wrong codes"):
        vault_2fa.verify(code_at(info["secret_base32"], T0 + 60), now=T0 + 60)
    assert vault_2fa.locked_out(now=T0 + 60) > 0
    assert vault_2fa.verify(code_at(info["secret_base32"], T0 + 400), now=T0 + 400)   # after the lockout


def test_recovery_codes_are_single_use_and_reset_needs_one():
    info = enrol()
    rc = info["recovery_codes"][0]
    assert vault_2fa.verify(rc.lower(), now=T0 + 30)
    assert not vault_2fa.verify(rc, now=T0 + 31)
    assert vault_2fa.recovery_codes_left() == 7
    assert not vault_2fa.reset("999999", now=T0 + 32)
    assert vault_2fa.enrolled()
    assert vault_2fa.reset(info["recovery_codes"][1], now=T0 + 33)
    assert not vault_2fa.enrolled()


def test_no_second_factor_means_no_editing():
    with pytest.raises(PermissionError, match="No second factor"):
        vault_2fa.verify("123456")
