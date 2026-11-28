# Briefing for Antigravity (agy): cross-check of the C2C CustomNodePacks audit work

You are asked to review this work, tell us what is wrong, and give instructions. Your instructions will be followed.
UI/UX is the weakest part; the owner wants AusBoss-level UI (reference: `D:\PROJECT\Custom_Nodes\third_party\ComfyUI-AusBoss`).

## Where things are
- Repo: `D:\PROJECT\Custom_Nodes\ComfyUI-CustomNodePacks` (Apache-2.0), branch `audit/ai-vfx-2026-11-27` (local only, not pushed).
- The owner's task: `D:\PROJECT\Custom_Nodes\new_tas.md` (full AI + VFX audit, 11 phases).
- The owner's external audit (102 nodes, line-cited): `D:\PROJECT\Custom_Nodes\docs\c2c_backend_audit.md`;
  production research: `D:\PROJECT\Custom_Nodes\docs\research_doc.md` (target: commercial 2K/EXR VFX on Windows,
  MiniMax H3, 17n+5 frame grid, 24-124 frames).
- Workspace rules: `D:\PROJECT\Custom_Nodes\CLAUDE.md`, `D:\PROJECT\Custom_Nodes\ORDERS.md`, `.claude\rules\`.
- UI quality bar notes: memory files `editor-design-quality-bar.md`, `feedback_compact_conditional_ui.md`,
  `node-chrome-both-renderers.md` in `C:\Users\varig\.claude\projects\d--PROJECT-Custom-Nodes\memory\`.

## Work done so far (commits on the branch)
1. `769865f` / `e9b7803` (on main): new node Mask Combine (C2C) `nodes/mask_combine.py`, core Autogrow inputs.
2. `f57bdec` masking correctness: `nodes/mask_matting/node.py` (neg_bbox exclusion after matting, edge feather
   before core/holdout, holdout wins, 1-frame external_mask), `nodes/mask_propagate_video.py` (flow direction),
   `nodes/smart_crop.py` + `nodes/image_mask_editor/node.py` (soft grow/shrink), `nodes/mask_matting/refine_node.py`
   (speck removal, chroma lock), `nodes/mask_combine.py` (in-place reduction).
3. `9160b35` security: local-page-only guards in `nodes/c2c_workflow_library.py`, `nodes/secure_subgraph_routes.py`,
   `nodes/watch_folder_routes.py`, `nodes/webhook_routes.py`, `nodes/integrity_guard.py`, `c2c_ai/api_routes.py`,
   `nodes/mec_diagnostics_api.py`; tests `tests/test_audit_security.py`.
4. UNCOMMITTED, unreviewed: five agents were verifying every claim of `c2c_backend_audit.md` and fixing confirmed
   ones when they hit a usage limit mid-way. Their partial edits are in the working tree (`git status`), across
   c2c_distributed/, image mask editor JS, layer effects, inpaint suite, VAE decode, frequency/grain, helpers,
   magnific tensors, mask toolkit, ONYX/ViTMatte, core SAM3, temporal, face fixer, ... Review these diffs critically.

## Reports (drafts)
- `docs/audit_2026-11-27/AI_RESEARCH.md` (model/licence research, verified URLs).
- Scratch drafts: `C:\Users\varig\AppData\Local\Temp\claude\7935d62e9704f5c9\9e174cc7-4897-4d85-8cb7-1f6099a2206d\scratchpad\audit\`
  `masking.md`, `bugs.md`, `VERIFY_BRIEF.md`, `verify_*.md` (if written), repro scripts with outputs.

## What we ask of you
1. Cross-check the commits and the uncommitted diffs: wrong fixes, missed bugs, broken compatibility (node ids,
   inputs, defaults, saved workflows must never break).
2. UI/UX review of the node front-ends (`js/`), against AusBoss: what to change, node by node, and how.
3. Tell us what the owner asked for that is still missing.
Give concrete, ordered instructions (file, change, how to verify).
