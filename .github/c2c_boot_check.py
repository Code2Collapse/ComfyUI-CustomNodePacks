# VENDORED - canonical copy: D:\PROJECT\Custom_Nodes\tools\vendored\c2c_boot_check.py
# Do not edit a copy inside a pack; edit the canonical file and run tools/sync_vendored.py --write.
"""C2C boot + registration check for ONE pack (CI; ledger L0.06).

Test suites stay on the developer's machine (owner rule 2026-10-23: "never git push test folders at all. keep them
in local only"), so CI checks what a fresh install of the pack does: it boots a throw-away CPU ComfyUI with only this
pack installed and FAILS if the pack does not import, registers no node, or prints a traceback from its own files
while booting. The UI smoke test (tests/ui/c2c_ui_smoke.py) runs locally.

    python .github/c2c_boot_check.py --core <ComfyUI checkout> [--out ci-artifacts] [--port 8199]
            [--python <interpreter>]

Exit 0 = clean, 1 = the pack is broken (see <out>/boot_report.json), 2 = ComfyUI itself could not boot.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

PACK_DIR = Path(__file__).resolve().parents[1]
PACK = PACK_DIR.name
WIN = os.name == "nt"


def _get(url: str, timeout: float = 60):
    with urllib.request.urlopen(urllib.request.Request(url, headers={"Connection": "close"}), timeout=timeout) as r:
        return r.status, r.read()


def _link(link: Path, target: Path) -> None:
    if WIN:
        subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)], check=True, capture_output=True)
    else:
        link.symlink_to(target, target_is_directory=True)


def start_server(core: Path, python: str, port: int, out: Path):
    base = Path(tempfile.mkdtemp(prefix="c2c_boot_check_"))
    (base / "custom_nodes").mkdir()
    _link(base / "custom_nodes" / PACK, PACK_DIR)
    log = out / "server.log"
    cmd = [python, "-s", str(core / "main.py"), "--cpu", "--listen", "127.0.0.1", "--port", str(port),
           "--base-directory", str(base), "--disable-auto-launch", "--preview-method", "none", "--disable-metadata"]
    home = base / "home"            # packs keep some state in ~ (~/.c2c, ~/.magnific): keep it in the sandbox
    home.mkdir()
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1", "HOME": str(home),
           "USERPROFILE": str(home), "C2C_HOME": str(home / ".c2c"), "C2C_VIDEO_INDEX_DIR": str(home / "video_index")}
    proc = subprocess.Popen(cmd, cwd=str(core), stdout=open(log, "w", encoding="utf-8"), stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL, env=env)
    url = f"http://127.0.0.1:{port}"
    t0 = time.time()
    while time.time() - t0 < 900:
        time.sleep(3)
        if proc.poll() is not None:
            stop_server(proc, base)
            raise RuntimeError(f"ComfyUI exited with {proc.returncode}; tail:\n"
                               f"{log.read_text(encoding='utf-8', errors='replace')[-2500:]}")
        if "To see the GUI go to" not in log.read_text(encoding="utf-8", errors="replace"):
            continue
        try:
            if _get(url + "/object_info/KSampler", 120)[0] == 200:
                return proc, url, base, time.time() - t0
        except Exception:
            pass
    stop_server(proc, base)
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


def pack_tracebacks(log: str) -> list[str]:
    """Traceback blocks whose frames are in this pack's files (one string per block, at most 40 lines each)."""
    blocks, cur = [], None
    for ln in log.splitlines():
        if ln.startswith("Traceback (most recent call last)"):
            cur = [ln]
            continue
        if cur is None:
            continue
        cur.append(ln)
        if not ln.startswith((" ", "\t")):          # the exception line ends the block
            if any(PACK in x for x in cur):
                blocks.append("\n".join(cur[:40]))
            cur = None
    return blocks


def run(args) -> int:
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    report = {"pack": PACK, "python": sys.version.split()[0], "platform": platform.platform()}
    try:
        proc, url, base, boot_s = start_server(Path(args.core).resolve(), args.python, args.port, out)
    except Exception as exc:
        report["boot_error"] = str(exc)
        (out / "boot_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
        print(f"BOOT FAILED: {exc}")
        return 2
    try:
        report["boot_seconds"] = round(boot_s, 1)
        info = json.loads(_get(url + "/object_info", 300)[1])
        try:
            stats = json.loads(_get(url + "/system_stats", 60)[1]).get("system", {})
            report["core_version"] = stats.get("comfyui_version")
            report["frontend"] = stats.get("required_frontend_version")
        except Exception:
            pass
        ids = sorted(k for k, v in info.items()
                     if str(v.get("python_module", "")) == f"custom_nodes.{PACK}"
                     or str(v.get("python_module", "")).startswith(f"custom_nodes.{PACK}."))
        report["nodes_from_pack"] = {"count": len(ids), "ids": ids}
    finally:
        stop_server(proc, base)
    log = (out / "server.log").read_text(encoding="utf-8", errors="replace")
    failed_lines = [ln for ln in log.splitlines() if "IMPORT FAILED" in ln and PACK in ln]
    report["import_failed"] = {"failed": bool(failed_lines), "lines": failed_lines}
    report["tracebacks"] = pack_tracebacks(log)
    problems = []
    if failed_lines:
        problems.append("the pack failed to import")
    if not ids:
        problems.append("the pack registered no node")
    if report["tracebacks"]:
        problems.append(f"{len(report['tracebacks'])} traceback(s) from the pack's files while booting")
    report["problems"] = problems
    (out / "boot_report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"{PACK}: {len(ids)} nodes registered, boot {report['boot_seconds']} s; "
          + ("; ".join(problems) if problems else "clean"))
    for tb in report["tracebacks"][:3]:
        print(re.sub(r"\n{2,}", "\n", tb))
    print(f"report: {out / 'boot_report.json'}  {'FAIL' if problems else 'PASS'}")
    return 1 if problems else 0


def main() -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--core", required=True, help="a ComfyUI checkout (the directory holding main.py)")
    ap.add_argument("--out", default="ci-artifacts")
    ap.add_argument("--port", type=int, default=8199)
    ap.add_argument("--python", default=sys.executable)
    return run(ap.parse_args())


if __name__ == "__main__":
    sys.exit(main())
