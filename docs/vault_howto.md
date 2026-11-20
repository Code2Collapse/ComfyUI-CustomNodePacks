# C2C Vault - quickstart

A vault is a subgraph whose inside is encrypted. It runs with ComfyUI's own scheduler, like any subgraph. Opening it
needs the vault password and a code from your authenticator app.

Design records (for developers): [VAULT_DESIGN_A18.md](VAULT_DESIGN_A18.md) (what ships) and
[VAULT_DESIGN_A17.md](VAULT_DESIGN_A17.md) (the earlier design and threat model).

## Which vault do I want?

| You want | Choose |
|---|---|
| To share a workflow that runs on your machine and on machines you approve, without telling anyone a password | **Seal: authorized installations** (default) |
| Anyone who knows the password to run it | **Lock: password to run** |
| It to run anywhere, and you accept that a knowledgeable person can extract it | **Legacy portable** (obfuscation only) |
| Other people to run it without ever receiving the graph | Not a vault profile yet ("owner-hosted" is listed as coming). For whole workflows, use the C2C App Gateway: [APP_GATEWAY_DEPLOY.md](APP_GATEWAY_DEPLOY.md) |

In every profile, only someone with the password **and** your authenticator can open the inside.

What a vault does not protect: the machine that runs it has to decrypt the inside to run it, so someone who controls
that machine (memory, a debugger, a changed ComfyUI) can read it. A Locked vault can be decrypted by anyone who has
its password.

## 1. Protect some nodes

1. Select the nodes.
2. Right-click an empty spot of the canvas and choose **🔒 Seal as C2C Vault (authorized installations)…** or
   **🔒 Lock as C2C Vault (password to run)…**. To protect an existing subgraph, right-click its node and choose
   **🔒 Protect this subgraph as C2C Vault…**.
3. The dialog explains each profile. Type the vault password twice and press **Lock**.

The selection becomes one vault node. Its sockets are the wires that crossed the selection, and its controls are
the subgraph's promoted widgets. This works on the root graph and inside subgraphs. Nodes with lazy inputs
(LazySwitchKJ), hidden inputs (NKD Mask Ops) and lists run inside like anywhere else. Protecting is one undo step.

The command palette has the same actions: **C2C Vault: Lock selected nodes (password to run)**, **C2C Vault: Seal
selected nodes (runs without password)** and **C2C Vault: Seal whole workflow**.

## 2. Run it

- **Sealed:** it simply runs, on your install and on the installs you authorized.
- **Locked:** the first run in a browser tab asks for the password. After that, the tab runs it without asking. A new
  tab, or a ComfyUI restart, asks again.

What the inside makes still shows: previews, progress, and saved files. Errors name the inner node's type and the
message, never its values.

## 3. Open and edit it

1. Double-click the vault, or right-click it and choose **🔓 Open / edit vault…**.
2. Type the password and the 6-digit code your authenticator app shows. A code works once; if you just used it, wait
   for the next one (the dialog says so).
3. The real nodes open as a normal subgraph. Edit them, including subgraphs nested inside.
4. **Leave the subgraph** (breadcrumb, or navigate back) and it is encrypted again at once.

While it is open, saving, autosave, export and queueing still write the vault locked, and copying the open vault is
refused.

First time on an install: opening asks you to set up the authenticator. Scan the QR code (or type the setup key),
save the 8 recovery codes, and type the first code. If a setup key was ever pasted anywhere else, set the
authenticator up again: right-click a vault → **📱 Two-factor setup…**. It first asks for a current code or a recovery
code, then shows a new QR code. The old key stops working.

## 4. Let another install run a Sealed vault

The other person right-clicks any vault → **🔑 My installation code** and sends you the code (it is a public key, safe
to share), or saves it as a key file with **Save key file**. Paste the code into the Seal dialog's box, one code per
line, or use **Import key file…**. Your own install is always included.

## 5. Change who may run or edit it, or the password

Right-click the vault → **👥 Manage installations & editors…** or **🔐 Change password…**. Both ask for the password
and your authenticator code first, and never show the inside.

- Add an installation by pasting its code or importing its key file, or remove one. An installation can be a runner,
  an editor, or both.
- Your own installation cannot remove itself as an editor, and a Sealed vault always keeps your install as a runner.
- A password change re-encrypts the vault; the old password no longer opens the new copy.
- Removing an installation protects new copies of the workflow only. A copy that installation already has still opens
  there. The dialog says so.
- An old (v1) vault must be opened and locked once before its access can change.

## 6. Missing custom nodes

A vault records which custom node packs its inside uses (encrypted). When a machine lacks one, the run stops with
"This workflow needs custom nodes that are not installed". Right-click the vault → **📦 Install missing custom
nodes…**. The dialog says how many packs are missing and from which host, never their names. Install clones each pack
at the recorded commit (only from github.com, gitlab.com, codeberg.org or bitbucket.org) and asks you to restart
ComfyUI. Installing runs that pack's code at the next start: only accept vaults from people you trust. The environment
variable `C2C_VAULT_INSTALL=off` turns this off.

## 7. Old vaults

Workflows with the old **C2C Vault - Locked / Sealed** nodes open as before and are upgraded automatically to the new
vault, with the same payload, sockets and controls. A promoted parameter that had been turned into a `param_N` socket
keeps its stored value but loses that wire. Reconnect it after the upgrade.

## Vault nodes

You normally never add these from the node menu: the Protect commands create them, and old workflows bring the
deprecated ones. All six are in 🐺 C2C/🧰 Core/Vault. Their parameters (`vault_id`, `vault_payload`, the generated
sockets) are listed in [NODE_REFERENCE](../NODE_REFERENCE.md#c2c_securesubgraph); never edit `vault_id` or
`vault_payload` by hand.

| Node | What it is |
|---|---|
| **C2C Vault (protected subgraph)** (`C2C_SecureSubgraph`) | The vault node made by Lock / Seal / Protect. Its inputs `in_<n>` and outputs `out_<n>` are the wires that crossed the selection. When it runs, it decrypts the inside and hands it to ComfyUI's scheduler. |
| **C2C Vault (protected subgraph, saves)** (`C2C_SecureSubgraphOutput`) | The same, chosen automatically when the inside saves or previews something. It is an output node, so it runs even with nothing wired after it. |
| **C2C Vault — Locked (password to run)** (`C2C_VaultLocked`) | DEPRECATED. The old Locked vault. Kept so saved workflows load; the front end upgrades it to the new vault on load (step 7). |
| **C2C Vault — Sealed (runs without password)** (`C2C_VaultSealed`) | DEPRECATED. The old Sealed vault, kept and upgraded the same way. A legacy (v1) seal is obfuscation only and says so; open it and lock it to upgrade. |
| **Vault input (edit view)** (`C2C_VaultEditInput`) | DEPRECATED placeholder for one vault input inside the old (A17) vault edit view. That view was never saved, so you should not meet one. It never runs: "This is a vault editor placeholder: lock the vault to use it." |
| **Vault output (edit view)** (`C2C_VaultEditOutput`) | The same placeholder for one vault output. |

Common errors on a vault node:
- **"This C2C Vault is empty: lock a selection or a subgraph first."**: the node was added by hand from the menu.
- **"This vault is not authorized on this ComfyUI installation"**: send the author your installation code (step 4).
- **"This vault needs its password to run"**: a Locked vault in a new browser tab; enter the password.
- **"C2C Vault native execution is switched off (C2C_VAULT_NATIVE=off)."**: the environment variable turns the new
  vault off; unset it.
- **"This C2C Vault cannot run: ... Open it and save it again, or ask its author for a fresh copy."**: the decrypted
  inside failed its check.

## Not yet in the new vault

The **Sealed - owner-hosted** profile (recipients send inputs to your server and get outputs back) is shown in the
dialog as coming and cannot be chosen (L9.49). For whole workflows, the C2C App Gateway does this today
([APP_GATEWAY_DEPLOY.md](APP_GATEWAY_DEPLOY.md)). Adding installations through an owner's server is not built either.
