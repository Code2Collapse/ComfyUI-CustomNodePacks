# SimpleWatermarkRemover: not a C2C node

`SimpleWatermarkRemover` is a node id of **ComfyUI-AusBoss** (MIT), a legacy alias of AusBoss's LaMa Inpaint node
(`node_lama_inpaint.py`; see the workspace document `docs/A25_AUSBOSS_PARITY_PLAN.md`, row SimpleWatermarkRemover).
CustomNodePacks has never registered this id. The A25 plan says it must not, because the same id would collide with
AusBoss when both packs are installed.

- There is nothing to migrate: no C2C row, no C2C class (tested: `test_legacy_compat_ids_have_no_replacement_row`).
- A workflow that uses it needs ComfyUI-AusBoss installed.
- Watermark cleanup in C2C goes through the inpaint nodes (Inpaint Crop Pro / Inpaint Stitch Pro, ProPainterMEC for
  video). This has not been compared with AusBoss's LaMa output: parity NOT RUN, because there are no LaMa weights and
  no AusBoss install here.

**Owner decision (not taken):** L9.176 lists this node among the "deprecated compatibility nodes". If you meant a
C2C node, name it, and it will get a page like the others.
