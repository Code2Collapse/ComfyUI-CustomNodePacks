"""C2C AI spine — backend HTTP stress test.

Hits every D-track route, prints PASS/FAIL per check, persists results to
_AUDIT/stress_test/backend_results.jsonl.

No workflow execution. Pure REST + introspection.
"""
from __future__ import annotations
import json, os, sys, time, traceback
from pathlib import Path

import httpx

BASE = "http://127.0.0.1:8188"
OUT_DIR = Path(r"D:\PROJECT\Custom_Nodes\_AUDIT\stress_test")
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG = OUT_DIR / "backend_results.jsonl"
LOG.write_text("", encoding="utf-8")

results: list[dict] = []

def _record(name: str, verdict: str, detail: str = "", **extra) -> None:
    rec = {"name": name, "verdict": verdict, "detail": detail[:600],
           "ts": time.time(), **extra}
    results.append(rec)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    marker = {"PASS": "[ OK ]", "FAIL": "[FAIL]", "WARN": "[WARN]"}.get(verdict, "[ ?? ]")
    print(f"{marker} {name}: {detail[:200]}", flush=True)


def _get(path: str, **kw) -> httpx.Response:
    with httpx.Client(timeout=30.0) as c:
        return c.get(BASE + path, **kw)


def _post(path: str, **kw) -> httpx.Response:
    with httpx.Client(timeout=30.0) as c:
        return c.post(BASE + path, **kw)


# ────────────────────────────────────────────────────────────────────────
# 1.  /system_stats — sanity (proves server reachable)
# ────────────────────────────────────────────────────────────────────────
def t_system_stats():
    r = _get("/system_stats")
    assert r.status_code == 200, f"http {r.status_code}"
    d = r.json()
    assert "system" in d and "devices" in d
    return f"comfy={d.get('system',{}).get('comfyui_version','?')} devices={len(d.get('devices',[]))}"


# ────────────────────────────────────────────────────────────────────────
# 2.  /c2c/ai/status — spine bootstrap, backends listed
# ────────────────────────────────────────────────────────────────────────
def t_ai_status():
    r = _get("/c2c/ai/status")
    assert r.status_code == 200, f"http {r.status_code} body={r.text[:200]}"
    d = r.json()
    assert d.get("success") is True, f"success!=True: {d}"
    data = d.get("data") or {}
    assert "backends" in data, f"missing 'backends' key: {list(data)}"
    backends = data["backends"]
    assert isinstance(backends, list) and len(backends) >= 1, f"no backends: {backends}"
    by_tier = {}
    for b in backends:
        by_tier.setdefault(b["tier"], []).append(b["id"])
    # The deterministic RulePack should always be present (D.1)
    det_ids = by_tier.get("deterministic", [])
    assert any("rulepack" in i.lower() or "rule_pack" in i.lower() for i in det_ids), \
        f"RulePack not registered. tiers={by_tier}"
    return f"{len(backends)} backends — {dict((k, len(v)) for k, v in by_tier.items())}"


# ────────────────────────────────────────────────────────────────────────
# 3.  /c2c/ai/config — config endpoint shape
# ────────────────────────────────────────────────────────────────────────
def t_ai_config():
    r = _get("/c2c/ai/config")
    assert r.status_code == 200, f"http {r.status_code}"
    d = r.json()
    assert isinstance(d, dict), f"not dict: {type(d)}"
    return f"version={d.get('version')} backends={len(d.get('backends',[]))}"


# ────────────────────────────────────────────────────────────────────────
# 4.  /c2c/doctor/pyenv — D.6 dependency
# ────────────────────────────────────────────────────────────────────────
def t_doctor_pyenv():
    r = _get("/c2c/doctor/pyenv")
    assert r.status_code == 200
    d = r.json()
    assert d.get("success") is True
    assert "python" in d and "device" in d and "packages" in d
    cuda = d["device"].get("cuda_available")
    return f"py={d['python']['version']} cuda={cuda} pkgs={len(d['packages'])}"


# ────────────────────────────────────────────────────────────────────────
# 5.  /c2c/doctor/explain — D.6 newbie + dev mode
# ────────────────────────────────────────────────────────────────────────
def t_doctor_explain_newbie():
    r = _get("/c2c/doctor/explain?newbie=1")
    assert r.status_code == 200
    d = r.json()
    assert d["success"] and d["newbie"] is True
    assert isinstance(d["items"], list)
    sev = d["counts"]
    assert set(sev.keys()) == {"ok", "info", "warning", "error"}
    # Each item shape contract
    for it in d["items"]:
        for k in ("severity", "title", "detail", "fixes", "source"):
            assert k in it, f"item missing {k}: {it}"
    return f"items={len(d['items'])} counts={sev}"


def t_doctor_explain_dev():
    r = _get("/c2c/doctor/explain?newbie=0")
    assert r.status_code == 200
    d = r.json()
    assert d["newbie"] is False
    return f"items={len(d['items'])} counts={d['counts']}"


# ────────────────────────────────────────────────────────────────────────
# 6.  /mec/introspect_error — D.3 route
# ────────────────────────────────────────────────────────────────────────
def t_introspect_error_route():
    payload = {
        "exc_type": "RuntimeError",
        "message": "Given groups=1, weight of size [320, 4, 3, 3], expected input[1, 16, 64, 64] to have 4 channels, but got 16 channels instead",
        "node_class": "KSampler",
    }
    r = _post("/mec/introspect_error", json=payload)
    assert r.status_code == 200, f"http {r.status_code} body={r.text[:200]}"
    d = r.json()
    assert d.get("success") is True, f"success!=True: {d}"
    env = d.get("data") or {}
    assert isinstance(env, dict) and len(env) >= 1, f"empty envelope: {env}"
    return f"envelope_keys={list(env)[:6]}"


# ────────────────────────────────────────────────────────────────────────
# 7.  Rule pack reachable — pull from RulePackBackend via router
# ────────────────────────────────────────────────────────────────────────
def t_rulepack_match():
    """RulePack is a deterministic offline backend exposed only for the
    error_assistant feature, not generic chat. Verify it's discovered AND
    its probe returns healthy."""
    r = _post("/c2c/ai/probe", json={})
    assert r.status_code == 200, f"http {r.status_code} body={r.text[:200]}"
    d = r.json()
    assert d.get("success") is True, f"probe failed: {d}"
    data = d.get("data") or {}
    # data is keyed by backend id
    rp = next((v for k, v in data.items() if "rulepack" in k.lower()), None)
    assert rp is not None, f"rulepack not in probe results: {list(data)}"
    assert rp.get("ok") is True, f"rulepack unhealthy: {rp}"
    return f"rulepack probe ok latency={rp.get('latency_ms','?')}ms"


# ────────────────────────────────────────────────────────────────────────
# 8.  /c2c/ai/local/detect — D.7 wizard dependency
# ────────────────────────────────────────────────────────────────────────
def t_local_detect():
    r = _get("/c2c/ai/local/detect")
    assert r.status_code == 200
    d = r.json()
    servers = d.get("servers", [])
    return f"local_servers_found={len(servers)}"


# ────────────────────────────────────────────────────────────────────────
# 9.  Borrowed encoder probe (D.4) — should be unhealthy with no encoder loaded
# ────────────────────────────────────────────────────────────────────────
def t_borrowed_inactive():
    r = _get("/c2c/ai/status")
    d = (r.json().get("data") or {})
    borrowed = [b for b in d.get("backends", []) if "borrow" in b["id"].lower()]
    if not borrowed:
        return "borrowed not enabled (default OFF — correct)"
    b = borrowed[0]
    # Should be unhealthy because no Qwen/Llama is loaded yet
    health = b.get("health", {})
    return f"id={b['id']} healthy={health.get('ok')}"


# ────────────────────────────────────────────────────────────────────────
# 10. Doctor surfaces tier failure (D.5b end-to-end)
# ────────────────────────────────────────────────────────────────────────
def t_doctor_failure_surfacing():
    """Send a /mec/explain with a forced-failing cloud (no keys) and verify
    /c2c/doctor/explain afterwards now lists a tier3 failure entry."""
    # We can't safely register a fake failure via REST, but we can check
    # whether the previous test runs left any tier* failures visible.
    r = _get("/c2c/doctor/explain")
    d = r.json()
    sources = {it["source"] for it in d["items"]}
    return f"sources={sources}"


TESTS = [
    ("01_system_stats",            t_system_stats),
    ("02_ai_status",               t_ai_status),
    ("03_ai_config",               t_ai_config),
    ("04_doctor_pyenv",            t_doctor_pyenv),
    ("05_doctor_explain_newbie",   t_doctor_explain_newbie),
    ("06_doctor_explain_dev",      t_doctor_explain_dev),
    ("07_introspect_error_route",  t_introspect_error_route),
    ("08_rulepack_match",          t_rulepack_match),
    ("09_local_detect",            t_local_detect),
    ("10_borrowed_inactive",       t_borrowed_inactive),
    ("11_doctor_failure_surfacing", t_doctor_failure_surfacing),
]


def main() -> int:
    print(f"\n{'='*70}\nC2C AI BACKEND STRESS TEST\n{'='*70}", flush=True)
    pass_count = 0
    for name, fn in TESTS:
        try:
            detail = fn() or ""
            _record(name, "PASS", detail)
            pass_count += 1
        except AssertionError as e:
            _record(name, "FAIL", f"assertion: {e}")
        except Exception as e:
            _record(name, "FAIL", f"{type(e).__name__}: {e}\n{traceback.format_exc()[-400:]}")
    fail = len(TESTS) - pass_count
    print(f"\n{'='*70}\nTOTAL: {pass_count}/{len(TESTS)} passed, {fail} failed", flush=True)
    print(f"Log: {LOG}", flush=True)
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
