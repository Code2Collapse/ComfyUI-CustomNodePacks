# C2C Vault as a protected native subgraph (design, A18)

Owner directive A18 (2026-10-25, workspace `docs/evidence/A18/`). Supersedes the execution part of
`VAULT_DESIGN_A17.md`; the A17 envelope (Argon2id + AES-256-GCM, data key wrapped per opener) stays.

## Why the old vault fails

`nodes/vault_exec.py` runs the inner nodes itself (topological order, direct calls). It has no lazy evaluation
(`check_lazy_status`), no list semantics, no caching, no progress, and gives every hidden input `None`; the lock-time
check refuses what it cannot run. The reported refusals: `LazySwitchKJ` (lazy inputs - a real gap) and
`NKDMaskOps` / `NKDMaskOpsLean` (a BUG: they declare hidden inputs in the tuple form `("PROMPT",)`, which ComfyUI
accepts; the check compared strings and refused hidden inputs the vault supports).

## What a vault is now

| State | In the workflow / browser | On the server |
|---|---|---|
| Locked (normal) | one node, class `C2C_SecureSubgraph` (or `..._Output` when the inside has output nodes): public sockets, promoted controls, the encrypted payload. No inner node, link, value or layout anywhere. | Executes by decrypting and returning a NATIVE EXPANSION: ComfyUI's scheduler runs the inner nodes. |
| Open (editing) | a real native subgraph (the frontend's own subgraph node + definition), entered like any subgraph. Exists only in that tab's memory. | An edit session holds the data key; exit = re-encrypt. |

### Run path (native scheduler, no custom executor)

1. The vault node declares its boundary inputs `lazy` and requests none of them (`check_lazy_status` -> `[]`).
2. `execute` reads its own prompt entry (hidden `DYNPROMPT` + `UNIQUE_ID`) to learn which upstream node / slot feeds
   each boundary input, decrypts the payload with the key the security profile allows, and returns
   `{"result": <links to inner outputs>, "expand": <inner API graph>}`. Inner nodes that read a boundary input
   reference the UPSTREAM node directly, so an upstream branch only runs if an inner node really needs it - the same
   lazy behaviour as an open subgraph (LazySwitchKJ's unused branch never runs).
3. ComfyUI adds the inner nodes as ephemeral nodes with the vault as their display node: lazy inputs, hidden inputs,
   lists, batches, async nodes, caching, interrupts and progress are core's own.
4. Hidden inputs of inner nodes: `PROMPT` = core's original prompt (the encrypted vault, not the inside);
   `EXTRA_PNGINFO` = the outer workflow as the frontend serialized it (encrypted vault only - the frontend hooks make
   sure an open vault is serialized locked). Save Image inside a vault therefore embeds nothing secret.

### Leak control on the server

A wrapper on `PromptServer.send_sync` rewrites messages about ephemeral nodes of a vault:
`executing` / `progress` / `executed` / `execution_cached` keep only the vault's id (inner previews still show on the
vault node - owner: allowed); `execution_error` keeps the inner node TYPE and the exception message, drops
`current_inputs`, `current_outputs` and the traceback (owner: "type + message, no values"). History keeps core's
original prompt (encrypted). Residual, documented: the server console log of the machine running the vault, and any
code running in that process (`DYNPROMPT`), can see the inside - a machine that executes a graph can read it.

### Edit path

Double-click / right-click Open -> password + TOTP (this install) -> server checks the editor list -> returns the
inner UI graph -> the frontend builds a native subgraph node in place of the vault node (same id, same links) and
enters it. Leaving the subgraph (breadcrumb, Escape, navigation to the parent) locks it at once: the frontend produces
the API form with ComfyUI's own `graphToPrompt` (so widgets, primitives, reroutes, nested subgraphs and dynamic
prompts convert exactly as for any subgraph), the server re-encrypts under the session key, and the vault node
replaces the subgraph node. While open, every serialization (save, autosave, export, clipboard, queue,
`EXTRA_PNGINFO`) writes the vault in its last locked form.

### Payload (envelope v3, `C2CVLT3`)

Same construction as v2 (random data key; AES-256-GCM; Argon2id password wrap; X25519 per-install wraps; the header
is the AAD of every operation), plus: `profile`, `editors` (install public keys allowed to open for editing, the
creator's by default), `outputs_nodes` flag. Plaintext = `{"ui": <subgraph definition + nested definitions>,
"api": {"nodes": {...}, "inputs": [...], "outputs": [...], "promoted": [...]}, "interface": {...}}`. v1 / v2 payloads
of old Locked / Sealed nodes stay readable; their `api` is derived from the old flattened format.

### Security profiles (per vault, chosen when locking)

| Profile | Run | Edit | Network | Who can read it at run time |
|---|---|---|---|---|
| Sealed - authorized installations (default) | no password, only on installs whose key was added | password + TOTP + editor list | none | those installs' processes |
| Locked | password once per browser tab (held in server memory for that tab's client id) | same | none | the running process |
| Sealed - owner-hosted | inputs sent to the owner's server, outputs returned (explicit consent) | owner only | owner's server | the owner's server only |
| Legacy portable | anywhere | same | none | anyone with a debugger - labelled OBFUSCATION ONLY |

Installations are added by installation code, key file, or registration on the owner's server (owner: all three).

### Old vaults

Workflows with `C2C_VaultLocked` / `C2C_VaultSealed` load as before and are upgraded automatically: the frontend
replaces each with a `C2C_SecureSubgraph` node carrying the SAME payload (v1 / v2 stay valid), keeping links. Old
sealed portable vaults show "Legacy - obfuscation only" and offer an upgrade to the default profile.

### Missing custom nodes

A vault records the git URL + pinned commit of every pack its inside uses (encrypted). On a machine lacking some, the
run stops with "This workflow needs custom nodes that are not installed - Install" (pack names not shown, owner); the
user's click clones the pinned commits and asks for a restart. Asked on every run attempt until installed. Without
CustomNodePacks itself, ComfyUI's own missing-node dialog offers CustomNodePacks (the node carries its `aux_id`).

### Failure modes

| Failure | Behaviour |
|---|---|
| No key for this profile (install not authorized, tab not unlocked) | the run stops before anything executes: "this vault is not authorized on this installation" / "enter the vault password" |
| Tampered / truncated payload, wrong password | GCM authentication fails: one message for both |
| Inner node class missing | install offer (above); never names the class to recipients |
| Inner node raises | error shows the inner node type + message only |
| Editing tab closed / crash while open | the workflow on disk still holds the last locked payload (it was never written open) |
| Two tabs edit the same vault | the second commit is refused (payload hash changed under it) |

### Budgets

Decrypt + expansion per run: one AES-GCM of the payload (< 1 ms per MB) + Argon2id only for password paths (~0.1 s).
No polling. Message filter: once a vault has run in this process, each message is walked once (its size; the
messages are small) and only ids the vault's expansion really made are rewritten.

### Kill switch

`C2C_VAULT_NATIVE=off` - old vault nodes run through the old executor and the new node refuses with that reason.

### As built (measured 2026-10-25, owner stack core 0.39.0-43, frontend 1.53.10, both renderers)

- **Server** (`nodes/secure_subgraph.py`, `nodes/secure_subgraph_routes.py`): LazySwitchKJ + NKDMaskOps (V3, hidden
  PROMPT / EXTRA_PNGINFO) run inside on the core scheduler; the lazy branch never ran. Core reports an inner failure
  under the vault's real id with the vault's class (DECISIONS F2.25), so the filter names the inner TYPE from the node
  that started last. Live messages only move that state; the history replay does not. Custom events (CNP's own
  `nukenodemax.insight`) and the binary preview / text events are rewritten too. Evidence:
  `docs/evidence/A18/vault_native_live.json`.
- **Front end** (`js/c2c_secure_subgraph.js`): `graphToPrompt` flattens an open subgraph, so while a vault is open
  the prompt is rewritten back to the locked vault (its last snapshot) before it is sent (F2.27). The Locked run
  password is asked in `api.queuePrompt`, not in `graphToPrompt`, because other code calls that too. A vault takes
  the replaced node's id BEFORE it joins the graph, and its wires go back by name (`in_<k>` through `input_map`,
  promoted subgraph inputs skipped). Leaving the vault locks it; going deeper into a nested subgraph does not.
  Nested definitions are stripped from saves and removed after the lock. Evidence:
  `docs/evidence/A18/vault_frontend_live.json` (+ `vault_frontend_live2.json`: nested, copy guard, vault inside a
  subgraph, palette).
- **Secrets**: the OS credential store is shared by every ComfyUI install of the OS user; `C2C_SECRETS_BACKEND=file`
  isolates an install (the testbed sets it, F2.26).
- **Old menus**: the old "Lock / Seal selection" entries are gone (they built the custom-executor vault); the
  command palette and "Seal whole workflow" use the native vault.
