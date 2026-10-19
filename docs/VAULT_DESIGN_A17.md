# C2C Vault v2 - audit, threat model, design (owner directive A17, 2026-10-19)

Written before the code (A7). Owner decisions: `docs/evidence/A17/owner_answers_2026-10-19_vault.md` in the C2C
workspace (D0.25). Source read: `nodes/vault_crypto.py`, `vault_node.py`, `vault_exec.py`, `vault_boundary.py`,
`vault_promote.py`, `_c2c_secrets.py`, `js/c2c_vault.js` (1,970 lines), `docs/vault_howto.md`, and the 84 vault tests.

## 1. What exists (v1)

- **Locked.** AES-256-GCM over the node list. The key is PBKDF2-HMAC-SHA256 (600k iterations) of the password. The
  header (vault id, KDF parameters, interface hash) is authenticated as associated data. Wrong password and
  tampering give one identical error.
- **Run session.** The password goes over a local HTTP route, and the derived key is held in server memory for 8
  hours (`C2C_VAULT_TTL_MINUTES`). Five failures lock that vault id for 5 minutes.
- **Sealed.** A random data key is wrapped twice: under the password, and under `_PACK_SECRET =
  "c2c-vault-sealed-v1"`, a constant in the source.
- **Execution.** An in-process executor runs the inner nodes directly, so ComfyUI's progress and history never
  name them. V3 nodes, async nodes and named sockets are supported.
- **Whole-workflow encryption.** Separate routes, with save, storage and export hooks.

## 2. Defects against A17

| # | Sev | Defect | Evidence |
|---|-----|--------|----------|
| V1 | critical | Sealed = obfuscation: every sealed vault opens with the pack constant, by anyone who has the pack. | `vault_crypto._PACK_SECRET`, `unseal_for_run` |
| V2 | critical | No real editing. "Open" shows a read-only list of class names ("Edits here are NOT saved"); changing a vault means rebuilding it. | `js/c2c_vault.js openVaultOverlay` |
| V3 | high | A Locked vault's run session also opens it for editing (`/c2c_vault/open` returns the plaintext graph to any caller while the 8 h session is open): run and edit authorization are the same thing. | `vault_node._open` |
| V4 | high | No second factor anywhere. | - |
| V5 | medium | KDF is PBKDF2; A17 asks for Argon2id (memory-hard). | `derive_key` |
| V6 | medium | Unsupported inner-node patterns (hidden inputs, list inputs and outputs, lazy inputs, graph expansion) are found at run time, not refused at lock time. | `vault_exec` |
| V7 | medium | The clear-text interface manifest is not authenticated: socket names and types can be edited unnoticed. | `vault_interface` widget |
| V8 | low | The manifest publishes the inner node count. | `buildInterfaceManifest` |
| V9 | low | Throttling is in memory only (a restart resets it). It slows online guessing only; offline guessing is limited by the KDF cost alone. | `_SessionStore` |

## 3. Threat model (what is promised, what is not)

**Protected against:** reading a saved workflow, a template, a shared .json or a PNG's metadata. Searching it for node
names, prompts, model paths or values. Loading it without the pack. Copying or pasting the vault node. Tampering with
the ciphertext, the tag, the nonce, the header or the public interface. Wrong passwords and wrong codes. A browser
tab or extension reading the edit session's graph from storage.

**Not protected against, by construction:** someone who controls the machine that RUNS the graph. To run, the
graph is decrypted in that process. Memory, a patched executor or a debugger all expose it. This holds for Locked
(whoever has the password can run it, therefore decrypt it) and for Sealed (the recipient's install can decrypt
what it runs).

The consequence: **editing needs password + second factor** is an authorization policy enforced by the honest UI and
server. It is not a cryptographic separation. Whoever can run a Locked vault can decrypt it offline with the
password. No offline scheme can do better, and the docs say so.

## 4. Envelope v2 (`C2CVLT2`)

```
header (associated data, authenticated):
  magic "C2CVLT2" | version 2 | mode (0 locked, 1 sealed for installs, 2 sealed portable)
  vault_id (u8 length + utf-8) | kdf id (1 = argon2id) | memory KiB u32 | iterations u32 | lanes u8 | salt (16)
  boundary hash (32) | interface hash (32: socket names / types + promoted ids / labels / types, never values)
body:
  password wrap:   nonce(12) | AES-GCM(KEK = Argon2id(password, salt), DEK, AAD = header)
  install wraps:   count u16 | per recipient: key id (8) | ephemeral X25519 public (32) | nonce (12) |
                   AES-GCM(HKDF-SHA256(X25519(eph, recipient), info = "c2c-vault-v2" + vault_id), DEK, AAD = header)
  portable wrap:   (mode 2 only) nonce | AES-GCM(HKDF(pack constant, salt), DEK, AAD = header)
  ciphertext:      nonce(12) | AES-GCM(DEK, plaintext, AAD = header)
plaintext: {"format": 2, "exec": {nodes, links, boundary_in, boundary_out, promoted}, "layout": <the inner nodes'
            own ComfyUI serialization: positions, sizes, titles, colours, widget values, links>}
```

- **KDF profile `argon2id-v1`:** 128 MiB, 3 passes, 4 lanes. Measured 0.11 s on the 32-core build machine
  (`docs/evidence/A17/vault/argon2_bench.json`; PBKDF2 600k = 0.16 s).
- **Header bounds:** a header below 64 MiB / 2 passes or above 1 GiB / 10 / 16 is refused. A tampered header can
  neither weaken the KDF nor exhaust memory.
- **Fresh material:** a fresh random DEK, salt and nonces on every lock and re-lock. The DEK is used for one
  encryption only, so nonces never repeat under a key.
- **Size limits:** the payload is refused above 64 MiB decoded and when truncated, before any decryption.
  Decryption is all-or-nothing (GCM verifies before returning plaintext), so no partial graph ever exists.
- **Password changes:** decrypt with the old password, re-encrypt with the new one, verify the new payload by
  decrypting it, and only then replace it.

## 5. Sealed custody (owner: "both, plus portable")

- **Sealed for installs (default, mode 1).** Every install has an X25519 key pair. The private key lives in the OS
  credential store (`keyring`, via `_c2c_secrets`, scope `vault`); the public key is a short text
  `c2cvk1:<base64>`, shown in the vault menu.
  - Sealing wraps the DEK for this install plus any recipient public keys the author pastes in.
  - Runs are passwordless only on those installs. No secret ever travels: a recipient sends a public key, not a
    secret.
  - A recipient's install can decrypt what it runs (section 3).
- **Sealed portable (mode 2).** The explicit, labelled choice "Sealed - portable (obfuscation only)" keeps today's
  run-anywhere behaviour.
- **Legacy v1 seals** keep running, with a visible "legacy seal - obfuscation only" warning. Opening one (password +
  second factor) and locking it re-seals it as v2.
- **Migration and portability.** Moving to a new machine means exporting the install key (an explicit, warned
  action) or re-sealing the vault for the new install's public key. A vault sealed only for an install whose key is
  lost can still be opened with the password and re-sealed.

## 6. Second factor (owner: "per person, per install")

- **Enrolment.** TOTP per RFC 6238 (SHA-1, 6 digits, 30 s), using the `cryptography` package's
  `twofactor.totp`.
  - A 160-bit secret is generated and stored in the OS credential store, never in a workflow and never derived
    from a password.
  - The base32 secret and its `otpauth://` URI are shown once, together with 8 single-use recovery codes. Only
    SHA-256 hashes of the codes are stored.
- **Verification.** A window of ±1 step. A code is accepted once only (the last accepted step is stored). Five
  failures cause a 5-minute lockout.
- **Recovery.** A recovery code is accepted in place of a code, once, and lets you re-enrol.
- **Reset.** Requires a valid code or recovery code. If both are lost, deleting the `c2c-secrets:vault` entries in
  the OS credential store resets it. That is a machine-owner action and is documented as such: the second factor
  protects against other users of this ComfyUI and casual access, not against the machine's administrator.
- **Another editor on another machine** enrols their own authenticator there. Nobody's secret is shared.

## 7. Edit lifecycle

1. **Open / Edit:** password dialog -> `POST /c2c_vault/edit/begin`. The server derives the KEK, unwraps the DEK and
   decrypts. The plaintext stays on the server; the reply is only "second factor required" plus an edit id that
   expires in 2 minutes.
2. **Code** -> `POST /c2c_vault/edit/verify`. Only now does the server return the graph (layout + exec), together
   with an edit token. The run session is never enough.
3. **Editing.** The real nodes open in a separate workflow tab, "Vault edit - <name>". Boundary inputs and outputs
   appear as front-end-only placeholder nodes. Ordinary editing and undo work.
   - The tab is never persisted: tab restore, drafts and user-data saves are intercepted.
   - Save, Save As and Export are refused there with "Lock the vault to save".
   - Node copy inside the tab is kept in memory, never in the browser clipboard storage.
4. **Lock / close.** The tab's graph is serialized -> `POST /c2c_vault/edit/commit`. The server encrypts with a
   fresh DEK, salt and nonces, decrypts the result to verify it, and returns it. Only then does the front end
   replace the vault node's payload and interface and close the tab.
   - On any failure the tab stays open and the old payload is untouched.
   - The server drops the edit session's plaintext and keys (best effort; see section 9).
5. **Reload, crash or closed browser.** The tab is gone and nothing plaintext was stored. The outer workflow holds
   the last valid payload. Unsaved edits are lost, and the editor says so before you close it.

The outer workflow can be saved while a vault is being edited: it holds only ciphertext.

## 8. Execution and compatibility

The in-process executor stays, because ComfyUI's executor would publish inner node ids. Two changes:
- **Refused at lock time with a clear message:** list inputs and outputs, lazy inputs, graph expansion, and hidden
  inputs other than UNIQUE_ID / PROMPT / EXTRA_PNGINFO.
- **Supplied at run time:**
  - UNIQUE_ID becomes the vault node's own id, so inner progress shows as the vault's.
  - PROMPT and EXTRA_PNGINFO are None, so a Save Image inside a vault never embeds the plaintext graph in the PNG.

Nested vaults run with their own envelope and key. An outer vault's edit tab shows an inner vault as a vault node;
opening it needs its own password + code.

## 9. Secret handling limits (stated)

- Passwords exist only in the dialog's input element and in one request body. They are never stored, logged or put
  in an exception.
- KEK and DEK live in Python `bytes`, and in `bytearray` where the API allows it, and are overwritten on close.
- Python, JavaScript and GPU runtimes copy and garbage-collect memory, so zeroization is best effort, not a
  guarantee.
- Run sessions keep the DEK, never the password, for `C2C_VAULT_TTL_MINUTES`.
- Edit sessions expire after 30 minutes idle.

## 10. Tests (A17 s6)

- **Cryptographic:** passwords, codes, tamper of every header field / tag / nonce / wrap, truncation, oversize, KDF
  bounds, nonce and salt freshness, password change, migration v1 -> v2, legacy runs.
- **Leaks:** sentinel strings in titles, prompts, widget values, model names, ids and links. Scanned in the saved
  workflow, payload, interface, history, server log, PNG metadata, browser storage, clipboard and autosave.
- **Execution:** single and multiple inputs and outputs, branches, V3 and async nodes, nested vaults, refused
  patterns. Outputs are compared with running the same graph unvaulted.
- **Front end:** the edit lifecycle in both renderers, and inside a subgraph.

Ledger: L9.29 (this design), L9.30 (envelope v2, keys, TOTP), L9.31 (edit lifecycle, leaks, compatibility).
