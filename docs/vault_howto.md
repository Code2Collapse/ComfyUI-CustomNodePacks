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

You do not need to make a subgraph first: the vault does that for you, in the same step.

1. Select the nodes - or select a group to protect everything inside it.
2. Do one of these (they all open the same dialog):
   - press **Alt+Shift+L**;
   - click the lock button in the selection toolbox above the selection;
   - right-click one of the selected nodes (or an empty spot of the canvas) and choose
     **🔒 Lock as C2C Vault (password to run)…** or **🔒 Seal as C2C Vault (authorized installations)…**.
   To protect a subgraph you already have, right-click its node and choose **🔒 Protect this subgraph as C2C Vault…**.
3. The dialog shows what becomes the vault (the node names, and how many inputs and outputs the vault will have).
   Give the vault a name (a selected group's title is used by default), choose **who can run it** - Sealed
   (recommended), Locked or Portable - and type the vault password twice (the meter shows its strength;
   **Generate a strong password** makes one for you). For Sealed you can add other installations here too.
4. Press **Lock 3 nodes** (the button counts your selection). The dialog shows each stage while it works.

The selection becomes one vault node, selected and ready. Its sockets are the wires that crossed the selection, and
its controls are the subgraph's promoted widgets. Its body shows the profile, how many nodes are inside and, for a
Locked vault, whether this tab has unlocked it; **Open…** there opens it. This works on the root graph and inside
subgraphs. Nodes with lazy inputs (LazySwitchKJ), hidden inputs (NKD Mask Ops) and lists run inside like anywhere
else. Locking is one undo step: **Ctrl+Z** gives back exactly the nodes you selected.

If something goes wrong, nothing changes: the dialog stays open and says what failed and why, and your nodes are
exactly as they were. A selection that contains a vault is refused (lock the nodes around it separately). A selection
that contains a subgraph is converted with it; if this ComfyUI loses one of that subgraph's wires while grouping, you
are told and should unpack it first (right-click › **Unpack Subgraph**).

The command palette has the same actions: **Lock selection as C2C Vault…** (the dialog lets you choose),
**C2C Vault: Lock selected nodes (password to run)**, **C2C Vault: Seal selected nodes (runs without password)** and
**C2C Vault: Seal whole workflow**.

## 2. Run it

- **Sealed:** it simply runs, on your install and on the installs you authorized.
- **Locked:** the first run in a browser tab asks for the password (the **Run "name"** dialog; a wrong password is
  said there and you can try again). After that, the tab runs it without asking, and the vault's body says
  "Unlocked in this tab". A new tab, or a ComfyUI restart, asks again.

What the inside makes still shows: previews, progress, and saved files. Errors name the inner node's type and the
message, never its values.

## 3. Open and edit it

1. Double-click the vault, press **Open…** in its body, or right-click it and choose **🔓 Open / edit vault…**.
2. Type the password and the 6-digit code your authenticator app shows (a recovery code works too). The bar under the
   code shows when the app's code changes. A code works once; if you just used it, wait for the next one. A wrong
   password or code is said inside the dialog, which stays open with your password kept.
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

## 8. Lock parameter values (keep the nodes, hide their values)

A vault hides whole nodes. When the layout may stay public but some values may not (a prompt, a seed, LoRA
strengths, a model choice), lock the values instead.

1. Right-click a node - or select several nodes or a group and right-click one of them or the canvas - and choose
   **🔒 Lock parameter values…** (the command palette has **Lock parameter values of the selection…**). A selected
   subgraph node locks the nodes inside it.
2. The dialog shows how many values in how many nodes will be locked. Type a password twice (**Generate** makes a
   strong one) and press **Lock**. Every value is encrypted in your browser (PBKDF2-SHA256, 600 000 rounds, then
   AES-256-GCM); the password is never sent anywhere.
3. Each node keeps its place and wires; the locked widgets disappear and a card says "Values locked". Saving,
   autosave, export and copy-paste keep only the encrypted form (undo back past the lock itself returns the plain
   values to this tab, as before you locked them). A seed's "control after generate" is locked
   with it, so a locked seed does not change between runs.

**Run:** the first run in a browser tab asks for the password (**Run with locked values**; a wrong password is said
there). After that the tab runs without asking. The tab that locked the values runs them right away. ComfyUI's
history (including the inputs shown with a run error), the image metadata (`prompt`) and the queue view show
`<locked>` in place of each locked value.

**Change a value:** right-click › **🔓 Remove value lock…** (password needed) puts the values back in the widgets,
visible and saved in plain text again; edit, then lock again. **🔓 Unlock locked values (this tab)…** unlocks without
running.

Not lockable: custom editors whose value is not a plain text, number or switch (the dialog lists them), widgets
driven by a wire, vault nodes (a vault encrypts its inside already). Nodes with locked values cannot be locked as a
vault; remove their lock first. Locked values only ever go into a node of the type they were locked on: if a copy
of the lock ends up on another kind of node, the run stops with "locked values sit on a node of another type" and
names that node.

What it does not hide: your own ComfyUI sees the values while it runs them (memory, and the request that queues the
run). If a locked value is invalid - for example a locked model is not installed - ComfyUI's validation error names
it, and so does the server log (which any page of this ComfyUI can read); an error message written by a node can
quote its input too. A node that shows or saves its input (Show Text, a caption file) outputs the value. Anyone with the workflow can
try passwords offline, so use a long one. Without ComfyUI-CustomNodePacks the workflow opens with neutral
placeholder values in those widgets and would run with them.

## 9. Access keys for Sealed vaults (expiry, offline)

A Sealed vault runs without a password on the installations you sealed it for. With access keys, those installations
also need a key from you, valid until a date you choose. No server is involved: the key is signed by your
installation and checked on theirs.

**Author**

1. Right-click the vault › **👥 Manage installations & editors…**, confirm with the password and your authenticator
   code. Turn on **Require an access key to run** and press **Apply**. Save the workflow and send it.
2. Open the same dialog again and press **Issue an access key…**. Tick the installations, choose the last day the key
   works (it ends at midnight UTC that day), add a note if you like, and press **Issue key**. Copy the key or
   **Save key file** (`.c2caccess`) and send it.
3. Editors never need a key. Only the installation that turned keys on can issue them.

**Recipient:** the first run without a key stops with "This vault needs an access key from its author" and the
**Access key** dialog opens. Paste the key or **Open key file…** and press **Use this key**; the dialog says until
when this installation may run the vault. You can also open it from the vault's right-click menu: **🎫 Enter access
key…**. A newer key with a later date extends access.

Plain refusals you can meet: the key **expired on** a date; it is **for other installations** (send the author your
installation code); it is **for a different vault**; it **was revoked**; or **this computer's clock is set earlier**
than the last time a key was checked here (correct the date and time).

**Revoking:** in the access dialog tick **Revoke every key issued so far** and apply. Keys issued before stop opening
the NEW copy of the workflow; issue fresh keys for the people who keep access. A copy someone already has keeps
working with its old key until that key expires - that is why every key has an expiry date. The check runs inside
ComfyUI on the recipient's machine: someone who changes ComfyUI's code can skip it, as with every vault. It is a
control for honest installations, not copy protection.

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
