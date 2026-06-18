"""C2C AI spine — visual UI stress test (Playwright + Chromium).

DOM-only interactions. Screenshots after every action.
No /object_info or /prompt calls from this driver.
No node-instantiation on canvas.

Stages
------
1. boot     — open ComfyUI, wait for canvas + Vue shell mount, screenshot
2. sidebar  — open the C2C AI sidebar tab, screenshot, assert key DOM
3. settings — within the tab, assert backends table renders + status pills
4. wizard   — invoke runFirstRunWizard() through page.evaluate (UI hook
              already-exposed on window for testability) OR fall back to
              triggering it via settings (currently no other path); we
              prefer the registered command if present
5. doctor   — call /c2c/doctor/explain via fetch inside the page context
              (this is the ONE allowed page.evaluate fetch because it
              targets OUR backend, not /object_info; it lets us assert
              JSON shape from within the same origin)

Pass/fail per stage written to _AUDIT/stress_test/ui_results.jsonl.
Screenshots saved under _AUDIT/stress_test/screenshots/ui_stageN_*.png.
Failures dumped under _AUDIT/stress_test/failures/ with DOM HTML snapshot.
"""
from __future__ import annotations
import json, sys, time, traceback
from pathlib import Path
from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

BASE = "http://127.0.0.1:8188"
ROOT = Path(r"D:\PROJECT\Custom_Nodes\_AUDIT\stress_test")
SHOTS = ROOT / "screenshots"
FAIL  = ROOT / "failures"
SHOTS.mkdir(parents=True, exist_ok=True)
FAIL.mkdir(parents=True, exist_ok=True)
LOG = ROOT / "ui_results.jsonl"
LOG.write_text("", encoding="utf-8")

results: list[dict] = []

def _log(name: str, verdict: str, detail: str = "", **extra) -> None:
    rec = {"name": name, "verdict": verdict, "detail": detail[:600],
           "ts": time.time(), **extra}
    results.append(rec)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    marker = {"PASS": "[ OK ]", "FAIL": "[FAIL]", "WARN": "[WARN]"}.get(verdict, "[ ?? ]")
    print(f"{marker} {name}: {detail[:200]}", flush=True)


def _shot(page, name: str) -> str:
    p = SHOTS / f"{name}.png"
    try:
        page.screenshot(path=str(p), full_page=False, caret="initial",
                        animations="disabled", timeout=30000)
    except Exception:
        # Font-load timeout: snapshot anyway with no fonts wait
        page.screenshot(path=str(p), full_page=False, timeout=5000,
                        omit_background=False)
    return str(p)


def _dump(page, name: str, exc: Exception | None = None) -> None:
    p = FAIL / f"{name}.html"
    try:
        p.write_text(page.content(), encoding="utf-8")
    except Exception:
        pass
    if exc is not None:
        (FAIL / f"{name}.txt").write_text(traceback.format_exc(), encoding="utf-8")


# ────────────────────────────────────────────────────────────────────────
# Stage 1 — boot
# ────────────────────────────────────────────────────────────────────────
def stage_boot(page):
    page.goto(BASE, wait_until="domcontentloaded", timeout=45000)
    page.wait_for_selector("canvas", timeout=30000)
    # Wait for Vue shell — any sidebar button OR LiteGraph instance
    page.wait_for_function(
        "() => !!document.querySelector('.side-bar-button') "
        "|| !!document.querySelector('.p-button') "
        "|| !!(window.app && window.app.graph)",
        timeout=30000,
    )
    # Give the sidebar render one more tick
    page.wait_for_timeout(2000)
    _shot(page, "ui_stage1_boot")
    # Assertions
    sb = page.locator(".side-bar-button, .p-button").count()
    assert sb > 0, "no sidebar buttons mounted"
    return f"sidebar_buttons={sb}"


# ────────────────────────────────────────────────────────────────────────
# Stage 2 — open C2C AI sidebar tab
# ────────────────────────────────────────────────────────────────────────
def stage_sidebar(page):
    # Wait until the extension has fully registered (test hook present).
    try:
        page.wait_for_function(
            "() => !!(window.__C2C_AI_SETTINGS__ && window.__C2C_AI_SETTINGS__.runFirstRunWizard)",
            timeout=30000,
        )
    except PWTimeout:
        raise AssertionError("window.__C2C_AI_SETTINGS__ never set "
                              "(c2c_ai_settings.js failed to load/register)")
    # Click the sidebar button by title (works in headless Chromium).
    clicked = page.evaluate("""() => {
        for (const el of document.querySelectorAll('button, [role=button], .p-button, .side-bar-button')) {
            const t = (el.title || el.getAttribute('aria-label') || el.textContent || '');
            if (/C2C AI Backends/i.test(t)) { el.click(); return t.slice(0,80); }
        }
        return null;
    }""")
    assert clicked, "C2C AI Backends sidebar button not found"
    # Wait for OUR panel's signature text: "C2C AI BACKENDS" header
    # appears immediately; then either a backends row, an error box, or
    # the loading… placeholder. We wait for transition AWAY from loading…
    # by polling for the "Backends" h4 (which only appears once data
    # resolves), or for a red error box.
    try:
        # Wait for either a probe button (proves table rendered) or red
        # error banner. .c2c-test class is unique to OUR panel.
        page.wait_for_function(
            r"""() => document.querySelectorAll('.c2c-test, .c2c-en').length > 0
                    || /Could not load status/i.test(document.body.innerText)""",
            timeout=45000,
        )
    except PWTimeout:
        _shot(page, "ui_stage2_sidebar_stuck")
        raise AssertionError("C2C AI panel never produced .c2c-test/.c2c-en or "
                              "error banner (see ui_stage2_sidebar_stuck.png)")
    page.wait_for_timeout(400)
    _shot(page, "ui_stage2_sidebar_open")
    return f"clicked '{clicked}', panel resolved"


# ────────────────────────────────────────────────────────────────────────
# Stage 3 — settings panel renders backend rows + first-run hint
# ────────────────────────────────────────────────────────────────────────
def stage_settings_panel(page):
    # The panel render uses .c2c-en (enable checkbox) and .c2c-test (probe).
    # Even with zero cloud keys, the deterministic RulePack row may not
    # carry a checkbox — but the table itself + the "Backends" h4 must
    # exist now (stage 2 waited for them).
    has_test_btn = page.locator(".c2c-test").count()
    has_en_chk   = page.locator(".c2c-en").count()
    has_backends_h = page.locator("section h4").filter(has_text="Backends").count()
    _shot(page, "ui_stage3_settings_rendered")
    assert has_backends_h > 0 or has_test_btn > 0, \
        f"backends section did not render (h={has_backends_h} btn={has_test_btn})"
    return f"backends_header={has_backends_h} test_buttons={has_test_btn} enable_checkboxes={has_en_chk}"


# ────────────────────────────────────────────────────────────────────────
# Stage 4 — first-run wizard (smoke via direct invocation)
# ────────────────────────────────────────────────────────────────────────
def stage_wizard_smoke(page):
    # Force-open the wizard via the exposed test handle (set by
    # c2c_ai_settings.js: window.__C2C_AI_SETTINGS__).
    available = page.evaluate(
        "() => !!(window.__C2C_AI_SETTINGS__ && window.__C2C_AI_SETTINGS__.runFirstRunWizard)"
    )
    assert available, "window.__C2C_AI_SETTINGS__.runFirstRunWizard not exposed"
    # Fire-and-forget; wizard renders modal asynchronously.
    page.evaluate("() => { window.__C2C_AI_SETTINGS__.runFirstRunWizard(); }")
    # Welcome screen first (D.7 callout)
    try:
        page.wait_for_function(
            "() => /WELCOME TO C2C AI/i.test(document.body.innerText)",
            timeout=10000,
        )
    except PWTimeout:
        _shot(page, "ui_stage4_wizard_no_welcome")
        raise AssertionError("wizard Welcome step never appeared")
    # Assert D.7 polish content
    body = page.evaluate("() => document.body.innerText")
    assert "built-in" in body.lower() and "explainer" in body.lower(), \
        "D.7 'built-in error explainer' callout missing from Welcome"
    assert "82" in body and ("patterns" in body.lower()), \
        "D.7 '82 hand-curated patterns' detail missing"
    _shot(page, "ui_stage4_wizard_welcome")
    # Click Continue
    clicked_continue = page.evaluate("""() => {
        const btn = [...document.querySelectorAll('button')]
            .find(b => /^continue$/i.test((b.textContent || '').trim()));
        if (!btn) return false;
        btn.click(); return true;
    }""")
    assert clicked_continue, "Continue button not found on Welcome"
    # Now Step 1 of 3 — note: /c2c/ai/local/detect takes ~9-12s
    try:
        page.wait_for_function(
            "() => /Step 1 of 3/i.test(document.body.innerText)",
            timeout=30000,
        )
    except PWTimeout:
        _shot(page, "ui_stage4_wizard_no_step1")
        raise AssertionError("wizard Step 1 modal never appeared after Continue")
    _shot(page, "ui_stage4_wizard_step1")
    # Dismiss with Esc x2
    page.keyboard.press("Escape")
    page.wait_for_timeout(300)
    page.keyboard.press("Escape")
    return "welcome+step1 visible; D.7 callout content verified"


# ────────────────────────────────────────────────────────────────────────
# Stage 5 — doctor explain shape via in-page fetch
# ────────────────────────────────────────────────────────────────────────
def stage_doctor_explain(page):
    js = """
    async () => {
        const r = await fetch('/c2c/doctor/explain?newbie=1');
        if (!r.ok) throw new Error('http ' + r.status);
        const d = await r.json();
        return {
            success: d.success,
            newbie:  d.newbie,
            count:   (d.items||[]).length,
            counts:  d.counts,
            sources: [...new Set((d.items||[]).map(i => i.source))],
        };
    }
    """
    d = page.evaluate(js)
    assert d["success"] is True, f"doctor.success={d['success']}"
    assert d["newbie"] is True
    assert isinstance(d["count"], int)
    _shot(page, "ui_stage5_doctor_fetched")
    return f"items={d['count']} counts={d['counts']} sources={d['sources']}"


STAGES = [
    ("stage1_boot",            stage_boot),
    ("stage2_sidebar",         stage_sidebar),
    ("stage3_settings_panel",  stage_settings_panel),
    ("stage4_wizard_smoke",    stage_wizard_smoke),
    ("stage5_doctor_explain",  stage_doctor_explain),
]


def main() -> int:
    print(f"\n{'='*70}\nC2C AI VISUAL UI STRESS TEST\n{'='*70}", flush=True)
    passed = 0
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        ctx = browser.new_context(viewport={"width": 1600, "height": 1000})
        page = ctx.new_page()
        # Forward console errors to a file
        console_log = []
        page.on("console", lambda m: console_log.append(f"{m.type}: {m.text}"))
        page.on("pageerror", lambda e: console_log.append(f"pageerror: {e}"))
        try:
            for name, fn in STAGES:
                try:
                    detail = fn(page) or ""
                    _log(name, "PASS", detail)
                    passed += 1
                except AssertionError as e:
                    _log(name, "FAIL", f"assertion: {e}")
                    _dump(page, name, e)
                except PWTimeout as e:
                    _log(name, "FAIL", f"timeout: {e}")
                    _dump(page, name, e)
                except Exception as e:
                    _log(name, "FAIL", f"{type(e).__name__}: {e}")
                    _dump(page, name, e)
        finally:
            (ROOT / "ui_console.log").write_text("\n".join(console_log), encoding="utf-8")
            ctx.close()
            browser.close()
    fail = len(STAGES) - passed
    print(f"\n{'='*70}\nUI TOTAL: {passed}/{len(STAGES)} passed, {fail} failed", flush=True)
    print(f"Log: {LOG}\nShots: {SHOTS}", flush=True)
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
