# AI-Spine — build workflows and diagnose errors in plain English

> The AI-Spine is a small, local-first agent layer over ComfyUI. It does two
> things: **builds a runnable graph from a plain-English request**, and
> **explains a failed run in one sentence with a fix you can apply in one
> click**. It runs on your own local model (a ~2.5 GB Qwen GGUF), a local
> Ollama daemon, or a hosted API — whichever you configure in **C2C AI
> settings**. Nothing leaves your machine unless you pick a cloud provider.

---

## 1. Build a workflow from a description

**Open it:** command palette → **"C2C: Build workflow from description (AI)"**,
or the shortcut **Ctrl+Alt+B**. A dialog opens; type what you want and press
**Build**.

```
basic text to image with SDXL at 1024, 30 steps
```

The agent designs a graph using *this workspace's* nodes where an in-house
equivalent exists (MEC / C2C / NukeMax / WanDirector namespaces are preferred
over stock nodes), validates it against the live node registry, and — after
you confirm — loads it onto the canvas.

### How it works (and why it won't hallucinate junk)

The design follows the "compact tool mode" pattern: a small local model is
never handed the full 2000-node schema. Instead:

1. **Candidate selection** — your request is keyword-scored against the live
   `NODE_CLASS_MAPPINGS`, narrowed to ~40 relevant nodes.
2. **Compact schemas** — the model sees only those nodes' widgets and typed
   inputs/outputs, plus a worked wiring example.
3. **Mechanical validation** — every plan is checked against the real
   registry: unknown node types, unknown widgets, and **type-mismatched
   links** are all rejected. If validation fails, the errors are fed back and
   the model gets one retry.
4. **Conversion** — a validated plan becomes LiteGraph JSON with correct
   widget ordering (including the `control_after_generate` slot after seed
   widgets) and a tidy left-to-right layout.

Because the validator runs *before* anything reaches your canvas, a small
model that invents a fake node or wires two incompatible types simply gets
corrected or falls back — it can never drop a broken graph on you. With no AI
backend configured at all, a common **text-to-image** request still produces a
working graph from a built-in template.

The dialog always tells you which path produced the graph (`local`, `cloud`,
`deterministic-template`, …) and, if it failed, the specific validation
errors.

### API

```
POST /c2c/ai/build_workflow
{ "request": "text to image, 1024x1024, 30 steps" }
→ { ok, graph, plan, provider, node_count, attempts }
```

---

## 2. Diagnose a failed run — with a one-click fix

When a workflow errors, a **🩺 diagnosis card** appears in the corner. It reads
the *failing node's live configuration* (its widget values and what's wired
into it), names the parameter most likely at fault in plain English, and tells
you what to change — no raw traceback.

> *"The denoise strength is set to 0.0, meaning the sampler thinks there's
> nothing to denoise, so it produces zero timesteps and fails. Set denoise to
> a value between 0.0 and 1.0 (typically 0.5 to 1.0)."*

When the fix is a single widget change on that node, the card shows an
**Apply fix** button (e.g. `denoise → 1.0`). Click it, confirm, and the widget
is set on the real node — then just run again. A **Show node** button centers
the graph on the culprit.

### How it decides

1. **Instant pattern match** — a curated library covers the common cases
   (CUDA out of memory, unconnected input, missing file, size mismatch,
   missing package) with no model call at all.
2. **Model diagnosis** — for anything else, the failing node's compact schema
   and live values go to your configured model under a strict-JSON contract.
3. **Safety gate** — any suggested `Apply fix` is validated against the real
   node type: the widget must exist, combo values must be legal, numbers are
   coerced. An impossible suggestion is dropped, so the button never sets
   something the node can't accept.
4. **Fallback** — if all else fails you still get a plain-English message, never
   a Python traceback.

### API

```
POST /c2c/ai/diagnose
{ exc_type, message, traceback, node_id, node_type, widgets, upstream }
→ { ok, cause, fix, apply: {widget, value} | null, provider }
```

---

## 3. Choosing a backend (C2C AI settings)

Open **Settings → C2C AI**. Three tiers, independently toggleable:

| Tier | What it is | When to use |
|---|---|---|
| **Tier 1** | Curated pattern library (no model) | Always on — instant answers for common errors |
| **Tier 2** | Local model: a GGUF via llama.cpp, or Ollama | Private, offline, free — the recommended default |
| **Tier 3** | Hosted API (OpenAI / Anthropic / Gemini / others) | Highest quality; needs an API key |

### Local model (Tier 2)

The default local model is **Qwen3.5-4B (Opus-reasoning distill, Q4_K_M,
~2.5 GB, Apache-2.0)**. The C2C AI settings panel has a **one-click download**
button; or drop any GGUF into `ComfyUI/models/LLM/` and the resolver will find
it. Set `tier2_backend` to `ollama` and point `ollama_url` at your daemon to
use Ollama instead.

> **Note on reasoning-distill models.** Qwen "thinks out loud" before its
> answer. The pack strips that automatically and takes only the final JSON, so
> you never see the `<think>…</think>` scratchpad. Each caller (builder,
> diagnosis, error explainer) passes its own output contract to the model, so
> the formats never collide.

---

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| "no AI backend produced a valid plan" | Enable Tier-2 or Tier-3 in C2C AI settings; the local model needs to be downloaded first |
| Build dialog spins for ~a minute | The first request loads the local model into RAM; later requests are fast |
| Diagnosis card never appears | It only shows on an actual `execution_error`; check the browser console for a fetch failure to `/c2c/ai/diagnose` |
| "Apply fix" absent on the card | The model didn't propose a *single-widget* fix, or the proposed change failed validation — the cause/fix text is still shown |
| Local model gives odd wiring | Small models miss sometimes; the validator catches it and retries once, then falls back. A hosted Tier-3 model or a larger local GGUF wires more accurately |

---

## License

Apache-2.0. The workflow-builder and diagnosis architecture is adapted from
the compact-tool-mode pattern of
[artokun/comfyui-mcp](https://github.com/artokun/comfyui-mcp). The default
local model is [Qwen3.5-4B](https://huggingface.co) (Apache-2.0).
