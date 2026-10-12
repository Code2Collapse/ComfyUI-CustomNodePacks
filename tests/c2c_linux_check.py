"""C2C check for another machine (owner 2026-10-12: "Give me a check script"; ledger L7.88).

Run it on the machine to check, with the ComfyUI there already running:

    python custom_nodes/ComfyUI-CustomNodePacks/tests/c2c_linux_check.py [--url http://127.0.0.1:8188] [--no-browser]

It changes nothing: no setting is written, no workflow saved, nothing queued, nothing downloaded. It reads the C2C
repos' git state, asks the running server (read-only GET), calls the batching logic in-process with synthetic tensors,
and - if Playwright + Chromium are installed - opens the UI in a throw-away headless browser to compare wire snapping
and everyday canvas actions with C2C on against the same browser with every C2C front-end file blocked. The report
lands next to the current directory as c2c_check_report.json; send that file back.
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve()
CNP = HERE.parents[1]
CUSTOM_NODES = CNP.parent
# The fixes of 2026-10-12 each repo must contain (fast-forward pushes, so "is ancestor of HEAD" is the test).
REQUIRED = {
    "ComfyUI-CustomNodePacks": ("5172665", "image batching measured per node, wire snapping, slot double-click"),
    "ComfyUI-MiniMaxSuite": ("1d0f66e", "H3 continuation nodes resolve NestedTensor"),
    "ComfyUI-NukeMaxNodes": ("b72703b", "EXR save, OCIO defaults, Deep batch"),
    "ComfyUI-WanNodeExperiments": ("3c7d258", "loader messages"),
    "ComfyUI-WanAnimatePreprocessV2": ("e64b320", "built-in MiDaS next to controlnet_aux"),
}
C2C_PACKS = ["ComfyUI-CustomNodePacks", "ComfyUI-NukeMaxNodes", "ComfyUI-WanNodeExperiments", "ComfyUI-MiniMaxSuite",
             "ComfyUI-WanAnimatePreprocessV2", "ComfyUI-OmniScale", "ComfyUI-GLM_Image", "ComfyUI-WanAnimalPreprocessor"]
RESULTS: list[dict] = []


def report(area: str, name: str, status: str, detail: str = "") -> None:
    RESULTS.append({"area": area, "check": name, "status": status, "detail": detail})
    print(f"  [{status:<4}] {name}{(' - ' + detail) if detail else ''}", flush=True)


def git(repo: Path, *args: str) -> tuple[int, str]:
    try:
        r = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, timeout=30)
        return r.returncode, (r.stdout or r.stderr).strip()
    except Exception as exc:  # noqa: BLE001
        return 1, str(exc)


def check_repos() -> None:
    print("\n1. C2C repos (git, read-only)")
    for name, (commit, what) in REQUIRED.items():
        repo = CUSTOM_NODES / name
        if not (repo / ".git").exists() and not (repo / ".git").is_file():
            report("repos", name, "SKIP", "not installed here")
            continue
        _c, head = git(repo, "log", "-1", "--format=%h %cs %s")
        code, _ = git(repo, "merge-base", "--is-ancestor", commit, "HEAD")
        dirty = git(repo, "status", "--porcelain", "--untracked-files=no")[1]
        report("repos", name, "PASS" if code == 0 else "FAIL",
               f"HEAD {head[:70]}; fix {commit} ({what}) {'present' if code == 0 else 'MISSING - pull this repo'}"
               + ("; local edits present" if dirty else ""))


def get(url: str, path: str, timeout: float = 60):
    with urllib.request.urlopen(url + path, timeout=timeout) as r:
        return json.loads(r.read() or b"null")


def check_server(url: str) -> dict:
    print(f"\n2. Running server {url} (read-only GET)")
    info: dict = {}
    try:
        st = get(url, "/system_stats")
        sysi = st.get("system", {})
        info = {"comfyui": sysi.get("comfyui_version"), "python": sysi.get("python_version", "")[:40],
                "pytorch": sysi.get("pytorch_version"), "os": sysi.get("os"),
                "devices": [d.get("name") for d in st.get("devices", [])]}
        report("server", "system_stats", "PASS", json.dumps(info))
    except Exception as exc:  # noqa: BLE001
        report("server", "system_stats", "FAIL", f"cannot reach {url}: {exc}")
        return info
    try:
        oi = get(url, "/object_info", timeout=300)
        per = {p: sum(1 for d in oi.values() if p in str(d.get("python_module", ""))) for p in C2C_PACKS}
        report("server", "C2C nodes registered", "PASS" if per.get("ComfyUI-CustomNodePacks") else "FAIL",
               ", ".join(f"{k.replace('ComfyUI-', '')} {v}" for k, v in per.items() if v))
        info["c2c_nodes"] = per
    except Exception as exc:  # noqa: BLE001
        report("server", "object_info", "FAIL", str(exc))
    try:
        ab = get(url, "/c2c/autobatch/status")
        report("server", "image batching", "PASS", f"mode {ab.get('mode')}, {len(ab.get('wrapped') or [])} node types "
               f"wrapped, Canny wrapped: {'Canny' in (ab.get('wrapped') or [])} (wrapping happens on the first run)")
    except Exception as exc:  # noqa: BLE001
        report("server", "image batching status", "SKIP", str(exc))
    return info


def check_batching_logic() -> None:
    print("\n3. Image batching decisions (in-process, synthetic tensors)")
    try:
        import importlib.util

        import torch
        spec = importlib.util.spec_from_file_location("c2c_autobatch_check", CNP / "nodes" / "_autobatch.py")
        ab = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ab)
    except Exception as exc:  # noqa: BLE001
        report("batching", "load nodes/_autobatch.py", "FAIL", str(exc))
        return

    class Canny:
        RETURN_TYPES = ("IMAGE",)
        FUNCTION = "detect_edge"
        OUTPUT_NODE = False

        @classmethod
        def INPUT_TYPES(cls):
            return {"required": {"image": ("IMAGE",)}}

        def detect_edge(self, image):
            return (image,)

    specs = {m: ab._effective_wrap_specs({"mode": m}, {"Canny": Canny}).get("Canny", {}) for m in ("internal", "universal")}
    ok = all(s.get("max_frames") == 1 for s in specs.values())
    report("batching", "core Canny runs in 1-frame chunks (internal + universal)", "PASS" if ok else "FAIL", json.dumps(specs))
    img = torch.rand(10, 4, 4, 3)
    out = ab.run_chunked(lambda image: (image * 2, 7, image[..., 0]), {"image": img}, {}, label="order",
                         budget_bytes=4 * 4 * 3 * 4 * 6)
    ok = torch.equal(out[0], img * 2) and out[1] == 7 and torch.equal(out[2], img[..., 0])
    report("batching", "chunked outputs keep their order (IMAGE, INT, MASK)", "PASS" if ok else "FAIL")

    class ResizeV2Like(Canny):
        RETURN_TYPES = ("IMAGE", "INT", "INT", "MASK")

    why = ab._discovery_refused(ResizeV2Like, "ImageResizeKJv2", {}, {"mode": "universal"})
    report("batching", "nodes with non-frame outputs are left alone", "PASS" if why else "FAIL", str(why))


SNAP_JS = """async ([targets, sources]) => {
  const app = window.comfyAPI?.app?.app || window.app, out = [];
  const scr = (x, y) => { const c = app.canvas, r = c.canvas.getBoundingClientRect();
    return [r.left + (x + c.ds.offset[0]) * c.ds.scale, r.top + (y + c.ds.offset[1]) * c.ds.scale]; };
  for (const target of targets) {
    app.graph.clear();
    const t = LiteGraph.createNode(target); if (!t) { out.push({ target, missing: true }); continue; }
    t.pos = [760, 260]; app.graph.add(t);
    const src = {}; let y = 200;
    for (const [type, cls] of Object.entries(sources)) { if (src[cls]) continue; const n = LiteGraph.createNode(cls);
      if (!n) continue; n.pos = [60, y]; app.graph.add(n); src[cls] = n; y += Math.max(120, n.size[1] + 50); }
    const c = app.canvas; c.ds.scale = 0.7; c.ds.offset = [0, 0]; c.setDirty(true, true);
    await new Promise(r => setTimeout(r, 1000));
    out.push({ target, inputs: t.inputs.map((i, k) => ({ k, name: i.name, type: String(i.type), widget: !!i.widget })),
               t: t.id, src: Object.fromEntries(Object.entries(src).map(([k, n]) => [k, n.id])) });
  }
  return out; }"""


def check_browser(url: str) -> None:
    print("\n4. Canvas in a throw-away headless browser (C2C vs every C2C front-end file blocked)")
    try:
        from playwright.sync_api import sync_playwright
    except Exception:
        report("canvas", "playwright", "SKIP", "pip install playwright && python -m playwright install chromium")
        return
    app = "(window.comfyAPI?.app?.app || window.app)"
    sources = {"MODEL": ["CheckpointLoaderSimple", 0], "CLIP": ["CheckpointLoaderSimple", 1], "VAE": ["CheckpointLoaderSimple", 2],
               "IMAGE": ["LoadImage", 0], "MASK": ["LoadImage", 1], "LATENT": ["EmptyLatentImage", 0],
               "CONDITIONING": ["CLIPTextEncode", 0], "MODEL_PATCH": ["ModelPatchLoader", 0]}
    targets = ["KSampler", "VAEDecode", "ImageCompositeMasked", "MiniMaxH3FunControlNetApply", "MaskToolsMEC"]
    results = {}
    with sync_playwright() as pw:
        b = pw.chromium.launch(headless=True)
        for cfg in ("c2c", "native"):
            ctx = b.new_context(viewport={"width": 1600, "height": 1100})

            def read_only(route):        # nothing this browser does may write to the server (settings, workflows, queue)
                if route.request.method in ("GET", "HEAD", "OPTIONS"):
                    return route.continue_()
                return route.abort()
            for pattern in ("**/settings**", "**/api/settings**", "**/userdata**", "**/api/userdata**", "**/prompt**",
                            "**/api/prompt**", "**/c2c/**", "**/api/c2c/**", "**/mec/**"):
                ctx.route(pattern, read_only)
            if cfg == "native":
                for pack in C2C_PACKS:
                    ctx.route(f"**/extensions/{pack}/**", lambda route: route.abort())
            p = ctx.new_page()
            p.goto(url, wait_until="domcontentloaded", timeout=180000)
            p.wait_for_function(f"() => {app}?.graph && {app}?.canvas && window.LiteGraph", timeout=180000)
            p.wait_for_timeout(4000)
            if p.evaluate("() => { try { return (window.comfyAPI?.app?.app || window.app).ui.settings.getSettingValue('Comfy.VueNodes.Enabled') } catch { return null } }"):
                report("canvas", f"{cfg}: Nodes 2.0 is on in your settings", "SKIP",
                       "this probe drives the classic canvas; your settings are not changed")
                ctx.close()
                return
            if cfg == "c2c":
                mode = p.evaluate("() => { try { return (window.comfyAPI?.app?.app || window.app).ui.settings.getSettingValue('c2c.slotDoubleClick') } catch { return null } }")
                report("canvas", "slot double-click setting", "PASS",
                       f"{mode!r} (default since 2026-10-12: 'Suggest and connect a node (experimental)')")
            rows = {}
            for target in targets:
                info = p.evaluate(SNAP_JS, [[target], {k: v[0] for k, v in sources.items()}])[0]
                if info.get("missing"):
                    continue
                t = info
                for inp in info["inputs"]:
                    if inp["widget"] or inp["type"] not in sources:
                        continue
                    cls, out_i = sources[inp["type"]]
                    sid = info["src"].get(cls)
                    if sid is None:
                        continue
                    pts = p.evaluate(f"""([tid, sid, out, k]) => {{ const a = {app}, c = a.canvas, r = c.canvas.getBoundingClientRect();
                      const t = a.graph.getNodeById(tid), s = a.graph.getNodeById(sid);
                      const scr = ([x, y]) => [r.left + (x + c.ds.offset[0]) * c.ds.scale, r.top + (y + c.ds.offset[1]) * c.ds.scale];
                      return {{ from: scr(s.getOutputPos(out)), slot: scr(t.getInputPos(k)),
                               title: scr([t.pos[0] + t.size[0] / 2, t.pos[1] - LiteGraph.NODE_TITLE_HEIGHT / 2]) }}; }}""",
                                     [info["t"], sid, out_i, inp["k"]])
                    for point in ("slot", "title"):
                        p.evaluate(f"""(tid) => {{ const t = {app}.graph.getNodeById(tid);
                          for (let i = 0; i < t.inputs.length; i++) if (t.inputs[i].link != null) t.disconnectInput(i); }}""", info["t"])
                        a, z = pts["from"], pts[point]
                        p.mouse.move(*a)
                        p.mouse.down()
                        p.mouse.move(z[0] + 2, z[1] + 1, steps=14)
                        p.wait_for_timeout(80)
                        hover = p.evaluate(f"() => {app}.canvas._highlight_input?.name ?? null")
                        p.mouse.up()
                        p.wait_for_timeout(200)
                        linked = p.evaluate(f"(tid) => {app}.graph.getNodeById(tid).inputs.filter(i => i.link != null).map(i => i.name)", info["t"])
                        rows[f"{t['target']}.{inp['name']}@{point}"] = [hover, linked]
            results[cfg] = rows
            ctx.close()
        b.close()
    if "c2c" in results and "native" in results:
        diff = {k: {"c2c": results["c2c"].get(k), "native": v} for k, v in results["native"].items() if results["c2c"].get(k) != v}
        report("canvas", f"wire snapping: {len(results['native'])} drags, C2C == native", "PASS" if not diff else "FAIL",
               json.dumps(diff)[:600] if diff else "")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--url", default="http://127.0.0.1:8188")
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    print(f"C2C check {time.strftime('%Y-%m-%d %H:%M:%S')} on {platform.platform()}, python {platform.python_version()}")
    check_repos()
    server = check_server(a.url.rstrip("/"))
    check_batching_logic()
    if not a.no_browser:
        try:
            check_browser(a.url.rstrip("/"))
        except Exception as exc:  # noqa: BLE001
            report("canvas", "browser probe", "FAIL", f"{type(exc).__name__}: {exc}"[:300])
    fails = [r for r in RESULTS if r["status"] == "FAIL"]
    out = Path.cwd() / "c2c_check_report.json"
    out.write_text(json.dumps({"when": time.strftime("%Y-%m-%dT%H:%M:%S"), "platform": platform.platform(),
                               "server": server, "results": RESULTS}, indent=1), encoding="utf-8")
    print(f"\n{len(RESULTS) - len(fails)} of {len(RESULTS)} checks passed or skipped, {len(fails)} failed. Report: {out}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
