# Node Identity: hide titles, name nodes by what they do

Node Identity is a front-end tool in CustomNodePacks. It adds no node and never changes what a workflow does.

- **Hide or show every node title** with one shortcut, and choose per node.
- **Give nodes short, unique names** worked out from the graph itself, for example `SDXL · Initial Generation`,
  `SDXL · Detail Refinement`, `PROMPT · Positive`, `IMAGE · Final Decode`, `OUTPUT · Final Render`.
- **The admin decides** who may rename, which rules apply, and can lock the names.

## Using it

| What | Where |
|---|---|
| Hide / show all titles | **Alt+Shift+T**, the "Titles" button in the panel, or Settings -> C2C -> Canvas -> Node Identity |
| Title of one node: follow global / always show / always hide | right-click the node -> Node Identity |
| Rename the selection, the selected group or the whole workflow | right-click the canvas -> Node Identity, the selection toolbox tag, or the panel |
| The panel | OmniPill -> **Names** (command palette: "C2C: Open Node Identity") |
| Type your own name for a node | right-click the node -> Node Identity -> Rename this node..., or edit the name in the preview |
| Restore the original titles | panel -> Restore original titles (for the chosen scope), or right-click a node |
| Undo a rename | Ctrl+Z (one step per Apply) or the panel's Undo button |

Every rename shows a **preview first**: the current title, the proposed name (you can edit it), and why. Nothing is
written until **Apply**; **Cancel** writes nothing. Keyboard shortcuts can be changed in Settings -> Keybinding
(search for "C2C:").

While titles are hidden, pointing at a node shows its name and its node type. The title bar, the collapse button,
badges (node source, Muted / Bypassed, pinned, price) and the selection outline stay, and no node changes size or
position.

## How names are made

Rules only - no AI service, no randomness. The same workflow always gets the same names.

The engine looks at each node's type and category, its widgets (model file, image file, size, prompt words), its
connections (what feeds it and where its output goes), and the group it sits in:

- a sampler fed by an empty latent is an *Initial Generation*; one fed by another sampler is a *Detail Refinement*
  (or *Upscale Refinement* when an upscale sits in between); the model family found upstream (H3, WAN, FLUX, LTX,
  SDXL, ...) is the first part of the name;
- a text encoder is *Positive* or *Negative* depending on which sampler input it ends up in (also through a
  subgraph boundary);
- a decode that ends in a save node is the *Final Decode*; save nodes are *Final Render* / *Video Render*;
- nodes of other packs get their category plus their display name (`KJNODES · Resize Image v2`,
  `RGTHREE · Image Comparer`, `H3 · Multishot Sampler (one node)`).

Names are unique in the whole workflow (subgraphs included). When two nodes would get the same name, the engine
first adds a detail (prompt words, size, group) and only if that is still not enough numbers them `01`, `02`, ...
A name that is unique never gets a number. Pasted or duplicated nodes get the next free number.

**Kept as they are:** names you typed yourself (also names given before Node Identity existed), subgraph nodes,
Reroute / Note / Primitive, KJNodes Set/Get (their title is managed by KJNodes), rgthree Bookmark (its title is the
bookmark), types the admin excluded, and any node whose title a file-name pattern such as `%KSampler.seed%` uses
while the node has no "Node name for S&R" property. "Rename" also keeps names it made earlier; **Regenerate**
recomputes those (never your own names). Loading or refreshing a workflow never renames anything.

Template (admin): `{stage} · {function}` by default; tokens `{stage}`, `{function}`, `{detail}`, `{type}`; an empty
part is dropped together with its separator.

## What is saved

- The name is the node's normal title (`title` in the workflow JSON).
- `properties.c2c_identity` keeps `original` (the title before the first change - used by Restore), `alias`,
  `source` (`auto` or `manual`) and the per-node title override (`show` / `hide`).
- Hiding titles is your own preference (a ComfyUI setting), not part of the workflow.
- Type, class, ids, inputs, outputs, links and widget values are never touched.

**The API prompt.** ComfyUI's front end copies every node title into the API prompt as `_meta.title`; core uses it
only in error messages ("Node 'X' not found"). Renaming therefore changes `_meta.title` and nothing else: the
`class_type` and `inputs` the server executes are byte-identical, and the server serves a renamed prompt from its
cache (measured, see the evidence below).

## Admin control

The admin is the **ComfyUI page on this machine** - the same owner check CustomNodePacks uses for the Control Center,
stored secrets and the AI settings (`nodes/_local_ui_guard.py`: the connection must come from this machine, the page
must be opened as `127.0.0.1` / `localhost`, and no other site may send the request). Through an SSH tunnel the page
still counts as this machine. A browser on another computer (ComfyUI started with `--listen`) or behind a reverse
proxy with its own host name is **not** the admin.

The admin's rules live in `node_identity_policy.json` in the C2C home folder (`$C2C_HOME`, default `~/.c2c`). They
are not ComfyUI settings on purpose: ComfyUI lets any client write settings, the rules route lets only the admin.

| Rule | Default | Meaning |
|---|---|---|
| Feature on | on | off: nobody renames, titles are always shown |
| Lock names | off | names are read-only for everyone except the admin |
| Others may rename | off | other clients may rename / restore / edit names |
| Others may hide titles | on | other clients may hide titles for themselves |
| Keep names people typed | on | the engine never replaces a typed name |
| Use prompt words as detail | on | prompt words may tell two prompt nodes apart |
| Excluded types | none | node types that are never renamed |
| Template / separator | `{stage} · {function}` / ` · ` | the shape of a name |
| AI naming hook | off | uses a namer another extension registers (`window.C2CNodeIdentity.registerNamer(fn)`); the rules stay the fallback |

The server answers every write action (`POST /c2c/node_identity/authorize`) and refuses rule changes from anyone but
the admin (`POST /c2c/node_identity/policy` -> 403 `ADMIN_ONLY`). A client that is not allowed sees the panel
read-only and why.

**Limit:** names live in the workflow in each person's browser. Someone can still rename a node with ComfyUI's own
title editor or by editing the workflow file. The lock governs Node Identity; it is not a protection of workflow files.
Blocking ComfyUI's own title editor would need a change to the ComfyUI front end, which this pack does not make.

## Compatibility

Measured on ComfyUI core 0.39.0 + front end 1.53.10 (the owner's stack), classic canvas and Nodes 2.0, at the root and
inside a subgraph, with CustomNodePacks, MiniMax Suite, KJNodes and rgthree-comfy loaded. Evidence:
`docs/evidence/node_identity/` in the workspace (live script, JSON, screenshots).

- Classic canvas: a hidden node draws no title text through LiteGraph's `onDrawTitleText` hook, set on that node
  only (no prototype is changed); the hook is removed when the title is shown again.
- Nodes 2.0: one style sheet hides the title text of the nodes on screen (`visibility`, so the header keeps its
  size); it follows entering and leaving subgraphs.
- Older front ends: not measured.
