# C2C Vault - quickstart

A vault is a subgraph whose inside is encrypted. It runs with ComfyUI's own scheduler, like any subgraph. Opening it
needs the vault password and a code from your authenticator app.

## Which vault do I want?

| You want | Choose |
|---|---|
| To share a workflow that runs on your machine and on machines you approve, without telling anyone a password | **Seal: authorized installations** (default) |
| Anyone who knows the password to run it | **Lock: password to run** |
| It to run anywhere, and you accept that a knowledgeable person can extract it | **Legacy portable** (obfuscation only) |

In every profile, only someone with the password **and** your authenticator can open the inside.

## 1. Protect some nodes

1. Select the nodes.
2. Right-click an empty spot of the canvas and choose **🔒 Seal as C2C Vault** (or **🔒 Lock as C2C Vault**). To
   protect an existing subgraph, right-click its node and choose **Protect this subgraph as C2C Vault**.
3. The dialog explains each profile. Type the vault password twice and press **Lock**.

The selection becomes one vault node. Its sockets are the wires that crossed the selection, and its controls are
the subgraph's promoted widgets. This works on the root graph and inside subgraphs. Nodes with lazy inputs
(LazySwitchKJ), hidden inputs (NKD Mask Ops) and lists run inside like anywhere else.

The command palette has the same actions: **C2C Vault: Lock / Seal selected nodes**, and **Seal whole workflow**.

## 2. Run it

- **Sealed:** it simply runs, on your install and on the installs you authorized.
- **Locked:** the first run in a browser tab asks for the password. After that, the tab runs it without asking. A new
  tab, or a ComfyUI restart, asks again.

What the inside makes still shows: previews, progress, and saved files. Errors name the inner node's type and the
message, never its values.

## 3. Open and edit it

1. Double-click the vault, or right-click it and choose **Open / edit vault…**.
2. Type the password and the 6-digit code your authenticator app shows. A code works once; if you just used it, wait
   for the next one (the dialog says so).
3. The real nodes open as a normal subgraph. Edit them, including subgraphs nested inside.
4. **Leave the subgraph** (breadcrumb, or navigate back) and it is encrypted again at once.

While it is open, saving, autosave, export and queueing still write the vault locked, and copying the open vault is
refused.

First time on an install: opening asks you to set up the authenticator. Scan the QR code (or type the setup key),
save the 8 recovery codes, and type the first code. If a setup key was ever pasted anywhere else, set the
authenticator up again: right-click a vault → **Two-factor setup…**. It first asks for a current code or a recovery
code, then shows a new QR code. The old key stops working.

## 4. Let another install run a Sealed vault

The other person right-clicks any vault → **My installation code** and sends you the code (it is a public key, safe
to share). Paste it into the Seal dialog's box, one code per line. Your own install is always included.

## 5. Old vaults

Workflows with the old **C2C Vault - Locked / Sealed** nodes open as before and are upgraded automatically to the new
vault, with the same payload, sockets and controls. A promoted parameter that had been turned into a `param_N` socket
keeps its stored value but loses that wire. Reconnect it after the upgrade.

## Not yet in the new vault

Changing the password and managing who may edit (L9.48), owner-hosted runs (L9.49), and installing missing custom
nodes from inside a vault (L9.50) are planned.
