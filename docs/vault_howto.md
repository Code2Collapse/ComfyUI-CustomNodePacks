# C2C Vault — quickstart

Password-locked subgraphs for ComfyUI. The graph inside travels as AES-GCM ciphertext; recipients see an opaque blob instead of your node wiring.

## 1. Lock a selection

Select the nodes you want to hide. Right-click the canvas → **C2C Vault → Lock selection (password to run)** (or **Seal selection** if the recipient should run without a password).

## 2. Set a password

Step 1 of the modal: enter a password and confirm it. Use **Generate strong password** for a 20-character random secret, then **Copy** — there is no password file and no recovery.

## 3. Review the boundary

Step 2 lists wires **entering** and **leaving** your selection. Rename sockets here — recipients see these names, not the encrypted internals. Click **Lock** or **Seal**.

## 4. Wire the vault node

The original nodes are removed; one vault node replaces them at the selection centre. Connect upstream nodes to the named input sockets and downstream nodes to the outputs.

## 5. Unlock to run (locked mode)

Click **Unlock…** on the vault node (or double-click). The password goes to the server once and is never stored in the workflow. The session lasts until you click **Lock session** or restart ComfyUI.

**Sealed** vaults skip this step — they queue without a password.

## 6. Edit safely

Double-click (or **Open for editing…** on sealed vaults) opens a **floating overlay** — not a native subgraph. Close the editor before saving. Native subgraphs would write decrypted nodes into the `.json` on Ctrl+S.

---

**Honest scope:** stops casual inspection and copying. Anyone who can run Python in the ComfyUI process can dump memory while the graph executes. A lock on a door, not a safe.
