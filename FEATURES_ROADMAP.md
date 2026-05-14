# MEC ComfyUI Feature Roadmap
> **READ THIS FILE at the start of every session.**  
> Update `STATUS` and `COMPLETED` fields after finishing each item.  
> Last updated: 2026-05-14

---

## How to use this file

Each feature has:
- **STATUS**: `[ ]` = not started · `[~]` = in progress · `[x]` = done
- **Phase**: order to implement
- **Files**: which files to create or edit
- **Notes**: gotchas, design decisions, dependencies

---

## Phase 1 — "What does this node do?" Tooltip  ← CURRENT
**User story:** Hover a node title for 800 ms → floating card explains the node in plain English.  
**LLM backend:** Cloud API → Qwen3.5-2B-Q4_K_M GGUF (auto-download) → deterministic fallback.

| Item | STATUS |
|---|---|
| `nodes/node_explain.py` — backend routes + LLM routing + GGUF download | `[x]` |
| `js/mec_node_explain.js` — canvas hover → popover with CSS | `[x]` |
| `__init__.py` — register routes | `[x]` |
| `nodes/cloud_llm.py` — add optional `system=` param to `generate()` | `[x]` |

### Key design decisions (Phase 1)
- Default GGUF: `unsloth/Qwen3.5-2B-GGUF` → `Qwen3.5-2B-Q4_K_M.gguf` (1.28 GB)
- Download URL: `https://huggingface.co/unsloth/Qwen3.5-2B-GGUF/resolve/main/Qwen3.5-2B-{quant}.gguf`
- Download destination: `models/llm/` (via `folder_paths`) or `user/models/` fallback
- Cache: Python in-memory LRU (300 entries) — NOT persisted to disk (LLM responses can be stale)
- JS cache: `Map<className, data>` per session — cleared on page reload
- Popover: `position:fixed`, z-index 99999, `pointer-events:auto` so user can scroll
- Hide logic: 150 ms delay so user can move mouse to popover without it vanishing
- Qwen3 thinking tokens (`<think>...</think>`) are stripped before JSON parse
- Routes: `GET /mec/node_explain/{class_name}`, `GET /mec/node_explain/status`, `POST /mec/node_explain/download`, `GET /mec/node_explain/download/{job_id}`
- ComfyUI settings: `mec.node_explain.backend` (auto/api/gguf/off), `mec.node_explain.gguf_quant` (Q4_K_M/Q5_K_M/Q8_0)

---

## Phase 2 — Error Toast Plain-English Translator (JS side)
**User story:** Intercept red error toasts → rewrite "CUDA out of memory" → friendly plain-English with suggested fix.  
**Relation to existing code:** `error_assistant.py` already does server-side explanation. This is a JS-side toast interceptor that calls `/mec/explain_error` and replaces the toast text.

| Item | STATUS |
|---|---|
| `js/mec_error_toast.js` — MutationObserver on `.p-toast-message-error` + `/mec/translate_error` | `[x]` |
| `nodes/error_translator.py` — `POST /mec/translate_error` bridges to `error_assistant.explain()` | `[x]` |
| `__init__.py` — register routes | `[x]` |

### Notes
- ComfyUI toast selector: `.p-toast .p-toast-message-text` or via `app.extensionManager.toast`
- Must NOT modify ComfyUI core; use `MutationObserver` on toast container
- Show "original" message on hover over the friendly message (tooltip-in-tooltip)

---

## Phase 3 — Execution Flame Graph
**User story:** After a run completes, show a horizontal bar chart of per-node execution time.  
**Relation to existing code:** `insight.py` already collects per-node timing (`elapsed_ms`). The `mec_diagnostics_sidebar.js` has a statistics tab. Extend it with a new "Flame" tab.

| Item | STATUS |
|---|---|
| `js/mec_flamegraph.js` — floating ⏱ panel with horizontal heat bars + CSV export | `[x]` |
| `nodes/flamegraph.py` — `GET /mec/diagnostics/flamegraph` reads `mec_diagnostics_api._BUFFER` | `[x]` |
| `__init__.py` — register routes | `[x]` |

### Notes
- Data already in `_BUFFER` (each event has `elapsed_ms`, `node_id`, `node_class`)
- Bar width = `elapsed_ms / total_ms * available_width`; colour = heatmap (green→red)
- Sort by elapsed_ms descending so bottlenecks are at top
- Only show last completed prompt run (filter by `prompt_id`)

---

## Phase 4 — Live Tensor Inspector
**User story:** Click any LATENT/IMAGE/MASK wire mid-graph → side panel shows shape, dtype, min/max/mean.  
**Relation to existing code:** `VAELatentInspectorMEC` node already exists (executes in graph). This is a real-time wired-link inspector triggered by clicking a link on the canvas.

| Item | STATUS |
|---|---|
| `js/mec_tensor_inspector.js` — right-click node menu "🔬 Inspect tensor outputs" | `[x]` |
| `nodes/tensor_inspector.py` — wraps `execution.get_output_data`, stats ring buffer, routes | `[x]` |
| `__init__.py` — register routes | `[x]` |

### Notes
- LiteGraph link click: override `LGraphCanvas.prototype.processLinkClick`
- Tensor stats arrive via insight hook's `node_done` event — map output slot to tensor

---

## Phase 5 — Complexity Meter HUD
**User story:** Small floating HUD showing node count, estimated VRAM, and Easy/Medium/Advanced rating.

| Item | STATUS |
|---|---|
| `js/mec_complexity_hud.js` — floating chip with Easy/Medium/Advanced tier | `[x]` |
| VRAM estimate: deferred to Phase 13 (Render Cost Estimator) | `[~]` |

### Notes
- Rating thresholds: ≤8 nodes = Easy, ≤25 = Medium, >25 = Advanced
- Update on `graph-changed` event from LiteGraph
- Position: bottom-left, above status bar, moveable via drag

---

## Phase 6 — Connection Compatibility Hints
**User story:** When dragging a wire from a slot, ghost-highlight only valid target sockets.

| Item | STATUS |
|---|---|
| `js/mec_compatibility_hints.js` — wraps `LGraphCanvas.drawNode` to pulse compatible slots | `[x]` |

### Notes
- Hook `LGraphCanvas.prototype.drawConnections` or `onDrawOverlay`
- Type compatibility: exact match OR registered type conversions
- Highlight color: green border glow on valid sockets; dim invalid ones

---

## Phase 7 — Seed Sweep UI
**User story:** Right-click a seed widget → "Sweep seeds N→N+X" to auto-queue a grid of outputs.

| Item | STATUS |
|---|---|
| `js/mec_seed_sweep.js` — right-click context menu on INT widgets named "seed" | `[x]` |
| Python: no backend needed — JS queues multiple `/prompt` runs | `[x]` |

### Notes
- Detect "seed" widget: `widget.name.toLowerCase().includes("seed")`
- Default sweep: current seed to current+7 (8 images grid)
- Queue via `app.queuePrompt(0, 1)` after updating seed value in node
- Show progress indicator: "Sweep 3/8 running"

---

## Phase 8 — "What's Wired" Mini Map Legend
**User story:** Floating legend showing active model name, sampler, and resolution in plain text.

| Item | STATUS |
|---|---|
| `js/mec_whats_wired.js` — scan graph for CheckpointLoaderSimple, KSampler, VAEDecode | `[x]` |

### Notes
- Scan `app.graph._nodes` for known node types on each `graph-changed`
- Extract widget values by name: `node.widgets.find(w => w.name === "ckpt_name")`
- Show: "Model: v1-5  ·  Sampler: euler  ·  512×512"
- Position: top-right overlay on canvas

---

## Phase 9 — LoRA Weight Scrubber
**User story:** Inline micro-slider on LoRA loader nodes — live re-queues on mouseup.  
**Relation to existing code:** `ParameterHistoryMEC` already tracks parameter changes.

| Item | STATUS |
|---|---|
| `js/mec_lora_scrubber.js` — inject inline slider on `LoraLoader*` node widgets | `[x]` |

### Notes
- Hook `beforeRegisterNodeDef` for nodes whose `type` is `COMBO` and category includes "loaders"
- Range: 0.0 to 2.0, step 0.05
- Debounce mouseup → `app.queuePrompt()`

---

## Phase 10 — Prompt Token Counter
**User story:** Live count of CLIP tokens used in text encoder nodes, with 75/150 boundary warnings.

| Item | STATUS |
|---|---|
| `js/mec_token_counter.js` — hook STRING widgets on CLIPTextEncode nodes | `[x]` |
| `nodes/token_counter.py` — `/mec/token_count` route (exact CLIP count when CLIP loaded; heuristic fallback) | `[x]` |
| Token estimate: simple word-split heuristic (no tokenizer dependency in browser) | `[x]` |

### Notes
- Exact tokenization requires server round-trip; use word-count × 1.3 as estimate for live display
- Exact count: call `/mec/token_count` with the prompt text → server uses CLIP tokenizer if loaded
- Display: small badge on widget bottom "~63 / 77 tokens"

---

## Phase 11 — Node Group Presets ("Macro Nodes")
**User story:** Save a selected subgraph as a named preset; reload with one click.

| Item | STATUS |
|---|---|
| `js/mec_group_presets.js` — right-click selection → "Save as Preset"; sidebar gallery | `[x]` |
| `nodes/group_presets.py` — `/mec/presets` CRUD routes (dedicated module, not diagnostics_api) | `[x]` |
| Storage: `user/mec_presets/*.json` (subgraph JSON blobs + base64 thumb ≤64 KB) | `[x]` |

### Notes
- Use `app.graph.serialize()` filtered to selected nodes + their connections
- Re-import: append to current graph at cursor position
- Show preset thumbnails as base64 screenshots

---

## Phase 12 — Named Wire Labels
**User story:** Allow users to label connections (e.g. "depth pass", "albedo") shown as floating tags on edges.

| Item | STATUS |
|---|---|
| `js/mec_wire_labels.js` — double-click a link → prompt for label; render as canvas text | `[x]` |
| Storage: `graph.extra.mec_wire_labels` map (persisted in workflow JSON) | `[x]` |

### Notes
- `app.graph.links[link_id].custom_label = "depth pass"` (extra data ComfyUI serializes)
- Render in `LGraphCanvas.prototype.drawConnections` hook
- Clear label: double-click again → empty input → delete

---

## Phase 13 — Render Cost Estimator
**User story:** Live VRAM + estimated time indicator as nodes are wired up.

| Item | STATUS |
|---|---|
| `js/mec_cost_estimator.js` — 💰 floating chip → POST workflow API JSON to backend | `[x]` |
| `nodes/cost_estimator.py` — `/mec/cost_estimate` route using `_BUFFER` historical means + fallback profiles | `[x]` |

### Notes
- VRAM estimate: per-node lookup table (from profiled runs stored in `user/timing_history.json`)
- Time estimate: sum of median execution times for each node class
- Reuse insight.py's recorded timing data

---

## Phase 14 — Frame Range Overlay
**User story:** Mini timeline scrubber on IMAGE preview nodes for batch/animation outputs.  
**Relation to existing code:** `VideoFramePlayerMEC` already exists.

| Item | STATUS |
|---|---|
| `js/mec_frame_overlay.js` — scrubber widget added to any node returning multiple IMAGEs via onExecuted hook (no backend) | `[x]` |

---

## Phase 15 — Workflow Wizard / Guided Mode
**User story:** Step-by-step overlay that highlights the next node to configure.

| Item | STATUS |
|---|---|
| `js/mec_workflow_wizard.js` — JSON-driven step definitions; highlight overlays | `[x]` |
| `nodes/wizard.py` — `/mec/wizard/templates[/{id}]` routes + built-in seeder | `[x]` |
| `user/mec_wizards/*.json` — wizard definition files (3 built-ins shipped) | `[x]` |

### Notes
- Wizard definition: `[{ "title": "1. Load a model", "node_type": "CheckpointLoaderSimple", "widget": "ckpt_name", "hint": "..." }]`
- Highlight: draw pulsing border around target node via `onDrawForeground` hook
- Trigger: button in diagnostics sidebar or `/wizard` URL param

---

## Phase 16 — A/B Split Canvas
**User story:** Run two workflow variants side-by-side with a slider diff.  
**Relation to existing code:** `VideoComparerMEC` already implements wipe/diff/scopes.

| Item | STATUS |
|---|---|
| `js/mec_ab_split.js` — capture last-N session outputs and present a wipe-slider A/B compare panel | `[x]` |

### Notes
- LiteGraph supports multiple graph instances; ComfyUI frontend may resist this
- Simpler approach: compare last-N outputs of two different prompt_ids using VideoComparerMEC wipe

---

## Phase 17 — Casual UX Features
| Feature | File | STATUS |
|---|---|---|
| "Surprise Me" button | `js/mec_surprise_me.js` | `[x]` |
| Workflow Mood Board (output gallery) | `js/mec_mood_board.js` | `[x]` |
| Quick Style Presets (aesthetic tag palette) | `js/mec_style_presets.js` | `[x]` |
| Drag-to-Remix (output → img2img auto-wire) | `js/mec_drag_remix.js` | `[x]` |
| Confetti / Vibe FX on completion | `js/mec_completion_fx.js` | `[x]` |

---

## Phase 18 — Cross-Cutting Polish
| Feature | File | STATUS |
|---|---|---|
| Sticky Notes with color coding | `js/mec_sticky_notes.js` | `[x]` |
| Auto-layout button (pure-JS layered Sugiyama, no dagre dep) | `js/mec_autolayout.js` | `[x]` |
| Undo history visual panel | `js/mec_undo_panel.js` | `[x]` |
| Right-click → "Isolate Subgraph" | `js/mec_isolate_subgraph.js` | `[x]` |

---

## Phase 19 — VFX-Specific
| Feature | File | STATUS |
|---|---|---|
| OCIO / Colorspace Badges on image nodes | `js/mec_colorspace_badges.js` | `[x]` |
| Named Wire Labels (duplicate of Phase 12) | — | see Phase 12 |

---

## Relationship map: new ideas ↔ existing nodes

| New Feature | Existing node/file it extends or replaces |
|---|---|
| Tooltip ("What does this node do?") | NEW — integrates with `mec_diagnostics_sidebar.js` settings tab |
| Error Toast Translator | `error_assistant.py` (server already does it), `mec_diagnostics_sidebar.js` (Diagnostics tab shows errors) |
| Execution Flame Graph | `insight.py` (already records timing), `mec_diagnostics_api.py` (statistics endpoint) |
| Live Tensor Inspector | `VAELatentInspectorMEC` node (graph-level); new wire-click inspector is complementary |
| LoRA Weight Scrubber | `ParameterHistoryMEC` (tracks changes); scrubber triggers new changes |
| A/B Split Canvas | `VideoComparerMEC` (already has wipe/diff/scopes) |
| Frame Range Overlay | `VideoFramePlayerMEC` (already exists); extend to generic IMAGE nodes |
| Workflow Wizard | `mec_diagnostics_sidebar.js` (add a "Wizard" tab) |
| Complexity Meter | `insight.py` (VRAM delta data), `mec_progress_hud.js` (HUD overlay pattern) |
| Drag-to-Remix | `mec_advanced_paint.js` / `VideoMaskEditorMEC` (img2img pipelines exist) |

---

## Implementation notes (persisted across sessions)

### Route namespace
All MEC routes use `/mec/` prefix. Avoid collision with ComfyUI-Doctor (`/doctor/`).

### JS extension pattern
```js
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
app.registerExtension({ name: "MEC.FeatureName", setup() {...}, settings: [...] });
```

### LLM system prompt override
- `cloud_llm.generate(provider, model, prompt, max_tokens, system=None)` — `system=None` uses default traceback prompt
- `node_explain.py` passes its own system prompt via the `system=` kwarg

### GGUF model loading (separate from error_assistant)
- `node_explain.py` maintains its own `_EXPLAIN_BACKEND` singleton (separate from `local_llm._BACKEND`)
- Uses `llama_cpp.Llama` directly; n_ctx=2048 (sufficient for node explain)
- Strip `<think>...</think>` blocks from Qwen3.5 output before JSON parse

### Download system
- `node_explain.py` has its own `_DOWNLOAD_JOBS` dict (does NOT reuse mec_diagnostics_api jobs)
- Thread-safe: `threading.Lock` on job dict
- Partial downloads saved as `{filename}.part`, renamed to `{filename}` on success

### Session start checklist
1. Read this file — check which phase is next
2. Read `FEATURES_ROADMAP.md` status column
3. Check `nodes/node_explain.py` and `js/mec_node_explain.js` for current Phase 1 state
4. If Phase 1 is `[x]`, move to Phase 2

---

## Completed sessions log

| Date | Phase | What was done |
|---|---|---|
| 2026-05-14 | Phase 1 | node_explain.py + mec_node_explain.js + cloud_llm.py update + __init__.py wired |
| 2026-05-14 | Phase 2 | error_translator.py + mec_error_toast.js (MutationObserver-based toast rewrite) |
| 2026-05-14 | Phase 3 | flamegraph.py + mec_flamegraph.js (per-node timing panel, CSV export) |
| 2026-05-14 | Phase 4 | tensor_inspector.py (wraps get_output_data) + mec_tensor_inspector.js |
| 2026-05-14 | Phase 5 | mec_complexity_hud.js (Easy/Medium/Advanced floating chip) |
| 2026-05-14 | Phase 6 | mec_compatibility_hints.js (pulse compatible slots while dragging links) |
| 2026-05-14 | Phase 7 | mec_seed_sweep.js (right-click sweep menu, sequential seeds via queuePrompt) |
| 2026-05-14 | Phase 8 | mec_whats_wired.js (model/sampler/size/LoRA legend, scans graph every 3 s) |
| 2026-05-14 | Phase 9 | mec_lora_scrubber.js (inline strength scrubber overlay with Apply+Queue) |
| 2026-05-14 | Phase 10 | token_counter.py (CLIP tokenizer + heuristic fallback) + mec_token_counter.js (live badge) |
| 2026-05-14 | Phase 11 | group_presets.py (CRUD under user/mec_presets/) + mec_group_presets.js (gallery + paste at cursor) |
| 2026-05-14 | Phase 12 | mec_wire_labels.js (dbl-click link to label, persists in graph.extra) |
| 2026-05-14 | Phase 13 | cost_estimator.py (rolling means from _BUFFER) + mec_cost_estimator.js (💰 chip) |
| 2026-05-14 | Phase 14 | mec_frame_overlay.js (scrubber on multi-image previews via onExecuted hook + addDOMWidget) |
| 2026-05-14 | Phase 15 | wizard.py (built-in template seeder, 3 templates) + mec_workflow_wizard.js (🧙 chip, pulsing highlight) |
| 2026-05-14 | Phase 16 | mec_ab_split.js (⚖ chip, session output ring buffer, wipe-slider compare) |
| 2026-05-14 | Phase 17 | mec_surprise_me.js + mec_mood_board.js + mec_style_presets.js + mec_drag_remix.js + mec_completion_fx.js |
| 2026-05-14 | Phase 18 | mec_sticky_notes.js + mec_autolayout.js (pure-JS layered) + mec_undo_panel.js + mec_isolate_subgraph.js |
| 2026-05-14 | Phase 19 | mec_colorspace_badges.js (heuristic OCIO badge on IMAGE-typed nodes) |
| 2026-05-14 | Polish  | FolderIncrementer `suffix` widget (basename suffix, e.g. `_Inpaint`); progress HUD rewrite with rolling-window ETA + rate (it/s) + smoothing + animated header bar + "⏳ NN%" title prefix; tqdm bar_format now includes `n/total` and `rate_fmt` |
