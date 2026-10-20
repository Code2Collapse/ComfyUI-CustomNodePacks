# C2C Vault — quickstart

A vault replaces a group of nodes with one node. The nodes inside travel in the workflow as AES-256-GCM ciphertext
(vault format v2, `C2CVLT2`); anyone who opens the workflow sees the vault's inputs and outputs, not your nodes.
The design, threat model and limits are in [VAULT_DESIGN_A17.md](VAULT_DESIGN_A17.md).

## Which vault do I want?

| Kind | Who can run it | Who can read or edit it |
|---|---|---|
| **Locked** | whoever has the password | password + the editor's authenticator code |
| **Sealed for installs** (default seal) | this install, plus the installs whose keys you pasted, with no password | password + authenticator code |
| **Sealed, portable** | any ComfyUI with this pack | same, but see below: obfuscation only |
| **Legacy seal** (made before v2) | any ComfyUI with this pack; the node is marked "Legacy seal" | open it and lock it again to upgrade |

A portable seal and a legacy seal keep the graph out of the workflow file, but anyone with the pack can decrypt them.
Use them only when that is enough.

## 1. Lock or seal a selection

Select the nodes. Right-click a node → **C2C Vault → Lock selected nodes** (or **Seal**); the empty-canvas menu has
the same two as **Lock selection** / **Seal selection**. You can
also add a vault node and press **Lock selected nodes** / **Lock what's connected** on it. This works on the root
graph and inside subgraphs.

1. Type a password twice, or use **Generate strong password** and **Copy** it. There is no password recovery.
2. Review the wires that enter and leave the selection. These become the vault's inputs and outputs; rename them here.
3. Sealing only: choose who may run it without the password. **This install, and the installs whose keys I paste** is
   the default. **Anywhere (portable)** is obfuscation only.

The password is stretched with Argon2id (128 MiB, 3 passes) on the server. It is never stored in the workflow, in
browser storage or in the history.

## 2. Run it

- **Locked:** a Locked vault runs right away for the person who locked it. Anyone else types the password into the
  field on the node. **Lock again** forgets the key; so does restarting ComfyUI.
- **Sealed:** queue as usual. The node says where it runs: "Sealed for 1 install(s) · runs here without a password",
  "Portable seal · runs anywhere · obfuscation only" or "Legacy seal".

## 3. Seal a vault for someone else's install

The recipient opens the canvas menu → **C2C Vault → This install's key…** and sends you the `c2cvk1:…` line. That
line is a public key and is safe to share. Paste it into the seal dialog, one key per line. The vault then runs
without a password on your install and on theirs, and on no other.

## 4. Edit it

Press **Edit…** on a Locked vault, or **Open for editing** on a Sealed one. Editing needs the password and then a
six-digit code from your authenticator app.

- **First time on an install:** the vault shows a setup key and 8 recovery codes. Add the key to an authenticator app
  (Google Authenticator, 1Password, Aegis…), save the recovery codes, then type a code to confirm. The second factor
  belongs to you on this install. It is stored in the OS credential store, not in the workflow.
- The vault's real nodes open in a separate **Vault edit** tab. A bar there says nothing is saved until you lock it.
  In that tab Save, Save As, Export, Duplicate and Queue are refused. Drafts are not written to browser storage, and
  a node you copy stays in memory, so it cannot be pasted into a saved workflow.
- **Lock & close** encrypts the new contents with fresh keys and checks them before it replaces the vault's payload.
  Then save the workflow that holds the vault. **Discard edits** leaves the vault as it was.
- **Change password…** (on the same bar) sets a new password. It applies at **Lock & close**, when the old password
  stops opening the vault. Until then, nothing changes.
- After 5 wrong codes, editing pauses for 5 minutes. An edit session closes after 30 minutes without activity.

**Lost your authenticator?** Canvas menu → **C2C Vault → Second factor (authenticator)…** → type a recovery code to
reset it. You set it up again the next time you edit a vault.

## 5. Sharing and portability

- A workflow with a vault opens on a ComfyUI without this pack as an ordinary missing node. Saving it again keeps the
  vault's payload, inputs and outputs unchanged.
- A Locked vault runs anywhere the pack is installed, given the password.
- A vault sealed for installs runs only on the installs it was sealed for. Elsewhere it stops with a message saying
  so, and the node reads "Sealed for other installs · needs the password here". To add that install, the author can
  seal it again with the new key. Or someone with the password opens it there for editing and presses
  **Lock & close**: every vault sealed for installs also includes the install that locks it.

---

**Honest scope.** A vault has to be decrypted to run. Whoever can run it, whether by password or as a sealed-for
install, can in principle extract it from memory, and so can anyone who can run Python inside that ComfyUI. The
authenticator code protects *editing* through the UI; it is a policy gate on this install, not extra encryption.
What the vault guarantees is that the nodes inside are not readable from the workflow file, the browser's storage or
the history without the right key.
