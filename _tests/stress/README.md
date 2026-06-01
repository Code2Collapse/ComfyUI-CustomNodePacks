# C2C AI Spine — Stress Tests

Visual UI + backend HTTP smoke for the C2C AI sidebar, settings panel,
first-run wizard, and `/c2c/ai/*` + `/c2c/doctor/*` routes.

## Files
- `boot_server.bat` — detached ComfyUI launcher (`start /MIN cmd /c ...`).
- `wait_ready.py` — poll `/system_stats` until ready (180s budget).
- `backend_stress.py` — 11 HTTP probes against `/c2c/ai/*` and `/c2c/doctor/*`.
- `ui_stress.py` — Playwright + Chromium visual flow (sidebar → settings panel →
  first-run wizard → doctor.explain). DOM interaction only (per
  `comfyui_qa_stress_test.md` v2 rules — no `/object_info`, no
  `LiteGraph.createNode`).

## Run
```
boot_server.bat
python wait_ready.py
python backend_stress.py
python ui_stress.py
```

## Outputs (absolute paths; not in repo)
- `D:\PROJECT\Custom_Nodes\_AUDIT\stress_test\backend_results.jsonl`
- `D:\PROJECT\Custom_Nodes\_AUDIT\stress_test\ui_results.jsonl`
- `D:\PROJECT\Custom_Nodes\_AUDIT\stress_test\screenshots\`

## Last verified
- 2026-06-01 — backend 11/11 PASS, UI 5/5 PASS (Welcome + Step 1 of 3 wizard
  screens visually confirmed, D.7 callout content verified).
