# VENDORED - canonical copy: D:\PROJECT\Custom_Nodes\tools\vendored\c2c_ui_smoke.py
# Do not edit a copy inside a pack; edit the canonical file and run tools/sync_vendored.py --write.
"""C2C UI smoke test for ONE pack (ORDERS Phase 0 step 5 / CI; ledger L0.06).

Boots a throw-away CPU ComfyUI with only this pack installed, opens the UI in headless Chromium
with software WebGL (SwiftShader), adds every node the pack registers - in the classic LiteGraph
renderer and in Nodes 2.0 - screenshots each one, and FAILS if a node cannot be created, or the
page throws, or the browser logs an error that comes from this pack's own front-end files.

    python tests/ui/c2c_ui_smoke.py --core <ComfyUI checkout> [--out ui-artifacts] [--port 8199]
            [--renderer classic|nodes2|both] [--python <interpreter>]

Exit 0 = clean, 1 = UI defects found (see <out>/report.json), 2 = harness could not boot.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

PACK_DIR = Path(__file__).resolve().parents[2]
PACK = PACK_DIR.name
WIN = os.name == "nt"
SWIFTSHADER = ["--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"]
APP = "(window.comfyAPI?.app?.app || window.app)"
READY = f"() => {{ try {{ return !!{APP}.canvas?.graph && !!window.LiteGraph?.createNode; }} catch (e) {{ return false; }} }}"
ADD = f"""(type) => {{ const a = {APP}; const g = a.canvas.graph; g.clear();
  a.canvas.ds.scale = 1; a.canvas.ds.offset = [0, 0];
  const n = window.LiteGraph.createNode(type); if (!n) return false;
  n.pos = [360, 240]; g.add(n); window.__c2cNode = n; a.canvas.setDirty(true, true); return true; }}"""
MEASURE = f"""() => {{ const a = {APP}; const n = window.__c2cNode; if (!n) return null;
  const r = a.canvas.canvas.getBoundingClientRect(); const s = a.canvas.ds.scale, o = a.canvas.ds.offset;
  const th = window.LiteGraph.NODE_TITLE_HEIGHT || 30;
  return {{ alive: n.graph === a.canvas.graph, size: [Math.round(n.size[0]), Math.round(n.size[1])],
           widgets: (n.widgets || []).length,
           box: [r.left + (n.pos[0] + o[0]) * s, r.top + (n.pos[1] + o[1] - th) * s, n.size[0] * s, (n.size[1] + th) * s] }}; }}"""


def _get(url: str, timeout: float = 60):
    with urllib.request.urlopen(urllib.request.Request(url, headers={"Connection": "close"}), timeout=timeout) as r:
        return r.status, r.read()


def _post_setting(base: str, key: str, value) -> None:
    req = urllib.request.Request(f"{base}/settings/{key}", data=json.dumps(value).encode(), method="POST",
                                 headers={"Content-Type": "application/json", "Connection": "close"})
    urllib.request.urlopen(req, timeout=30).read()


def _link(link: Path, target: Path) -> None:
    if WIN:
        subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)], check=True, capture_output=True)
    else:
        link.symlink_to(target, target_is_directory=True)


def start_server(core: Path, python: str, port: int, out: Path):
    base = Path(tempfile.mkdtemp(prefix="c2c_ui_smoke_"))
    (base / "custom_nodes").mkdir()
    _link(base / "custom_nodes" / PACK, PACK_DIR)
    settings = base / "user" / "default" / "comfy.settings.json"
    settings.parent.mkdir(parents=True)
    settings.write_text(json.dumps({"Comfy.TutorialCompleted": True}), encoding="utf-8")
    log = out / "server.log"
    cmd = [python, "-s", str(core / "main.py"), "--cpu", "--listen", "127.0.0.1", "--port", str(port),
           "--base-directory", str(base), "--disable-auto-launch", "--preview-method", "none", "--disable-metadata"]
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    proc = subprocess.Popen(cmd, cwd=str(core), stdout=open(log, "w", encoding="utf-8"), stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL, env=env)
    url = f"http://127.0.0.1:{port}"
    t0 = time.time()
    while time.time() - t0 < 900:
        time.sleep(3)
        if proc.poll() is not None:
            raise RuntimeError(f"ComfyUI exited with {proc.returncode}; tail:\n{log.read_text(encoding='utf-8', errors='replace')[-2500:]}")
        if "To see the GUI go to" not in log.read_text(encoding="utf-8", errors="replace"):
            continue
        try:
            if _get(url + "/object_info/KSampler", 120)[0] == 200:
                return proc, url, base
        except Exception:
            pass
    proc.kill()
    raise RuntimeError("ComfyUI never served node definitions within 900 s")


def stop_server(proc, base: Path) -> None:
    if proc.poll() is None:
        if WIN:
            subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True)
        else:
            proc.terminate()
        try:
            proc.wait(30)
        except Exception:
            proc.kill()
    link = base / "custom_nodes" / PACK
    try:
        if WIN:
            os.rmdir(link)          # removes the junction only, never the pack
        else:
            link.unlink()
    except OSError:
        pass
    if os.path.lexists(link):       # never let rmtree walk through a live link into the pack
        print(f"WARNING: could not remove the link {link}; leaving {base} in place")
        return
    shutil.rmtree(base, ignore_errors=True)


def _from_this_pack(msg) -> bool:
    loc = (msg.location or {}).get("url", "") if hasattr(msg, "location") else ""
    return f"/extensions/{PACK}/" in loc or f"/extensions/{PACK}/" in msg.text or PACK in msg.text


def run(args) -> int:
    from playwright.sync_api import sync_playwright
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    try:
        proc, url, base = start_server(Path(args.core).resolve(), args.python, args.port, out)
    except Exception as exc:
        (out / "report.json").write_text(json.dumps({"pack": PACK, "boot_error": str(exc)}, indent=1), encoding="utf-8")
        print(f"BOOT FAILED: {exc}")
        return 2
    report = {"pack": PACK, "url": url, "renderers": {}}
    try:
        info = json.loads(_get(url + "/object_info", 300)[1])
        ids = sorted(k for k, v in info.items()
                     if str(v.get("python_module", "")) in (f"custom_nodes.{PACK}",) or
                     str(v.get("python_module", "")).startswith(f"custom_nodes.{PACK}."))
        report["nodes_registered"] = len(ids)
        boot_log = (out / "server.log").read_text(encoding="utf-8", errors="replace")
        report["import_failed"] = any("IMPORT FAILED" in ln and PACK in ln for ln in boot_log.splitlines())
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True, args=SWIFTSHADER)
            try:
                for vue in {"classic": [False], "nodes2": [True], "both": [False, True]}[args.renderer]:
                    tag = "nodes2" if vue else "classic"
                    _post_setting(url, "Comfy.VueNodes.Enabled", vue)
                    page = browser.new_page(viewport={"width": 1600, "height": 1100})
                    errs = []
                    page.on("pageerror", lambda e: errs.append({"kind": "pageerror", "text": str(e)[:400]}))
                    page.on("console", lambda m: errs.append({"kind": "console.error", "text": m.text[:400]})
                            if m.type == "error" and _from_this_pack(m) else None)
                    page.goto(url, wait_until="domcontentloaded", timeout=300000)
                    for _ in range(300):
                        page.wait_for_timeout(1000)
                        if page.evaluate(READY):
                            break
                    page.evaluate("() => document.getElementById('splash-loader')?.remove()")
                    for _ in range(60):                     # defs ready + startup workflow done
                        page.evaluate(ADD, "KSampler")
                        page.wait_for_timeout(2000)
                        m = page.evaluate(MEASURE)
                        if m and m["alive"] and m["widgets"] > 0 and m["size"] != [140, 60]:
                            break
                    boot_errors = list(errs)
                    rows = []
                    for nid in ids:
                        errs.clear()
                        row = {"id": nid, "created": page.evaluate(ADD, nid)}
                        page.wait_for_timeout(args.settle)
                        m = page.evaluate(MEASURE) if row["created"] else None
                        if m:
                            row.update(size=m["size"], widgets=m["widgets"], alive=m["alive"])
                            x, y, w, h = m["box"]
                            png = out / tag / f"{nid}.png"
                            png.parent.mkdir(parents=True, exist_ok=True)
                            page.screenshot(path=str(png), clip={"x": max(0, x - 12), "y": max(0, y - 12),
                                                                 "width": min(1600, w + 24), "height": min(1100, h + 24)})
                        row["errors"] = list(errs)
                        rows.append(row)
                    page.close()
                    bad = [r for r in rows if not r["created"] or r["errors"] or not r.get("alive", True)]
                    report["renderers"][tag] = {"nodes": len(rows), "boot_errors": boot_errors, "defects": bad, "rows": rows}
                    print(f"{PACK} [{tag}]: {len(rows)} nodes, {len(bad)} with defects, {len(boot_errors)} boot errors")
            finally:
                browser.close()
    finally:
        stop_server(proc, base)
    (out / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    failed = report.get("import_failed") or any(v["defects"] or v["boot_errors"] for v in report["renderers"].values())
    print(f"report: {out / 'report.json'}  {'FAIL' if failed else 'PASS'}")
    return 1 if failed else 0


def main() -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--core", required=True, help="a ComfyUI checkout (the directory holding main.py)")
    ap.add_argument("--out", default="ui-artifacts")
    ap.add_argument("--port", type=int, default=8199)
    ap.add_argument("--renderer", choices=("classic", "nodes2", "both"), default="both")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--settle", type=int, default=1200, help="ms after adding a node before measuring")
    return run(ap.parse_args())


if __name__ == "__main__":
    sys.exit(main())
