"""Shared plumbing for the automation panels: Scheduled Queue, Watch Folder, Webhooks (L9.39).

- own_base_url() / queue_prompt(): queue on THIS server (its real address and port), never a hard-coded 8188.
- parse_api_workflow(): API prompts only. A UI workflow (nodes + links) cannot be queued server-side without the
  front-end's widget mapping, so it is refused with a plain explanation.
- Persistence in user/default/c2c/automation.json (atomic write). Secrets never go here: they live in the C2C
  secrets vault (nodes/_c2c_secrets.py) or, without a backend, in memory only.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import threading
import uuid
from pathlib import Path
from typing import Any

log = logging.getLogger("C2C.Automation")

UI_WORKFLOW_ERROR = ("This is a UI workflow. Export it with Workflow > Export (API), or use 'Use current workflow'.")
STORE_NAME = "automation.json"
SECTIONS = ("schedules", "watchers", "webhooks_in", "webhooks_out")
_LOCK = threading.Lock()
_STORE_OVERRIDE: Path | None = None          # tests point this at a temp dir


# ── own server ───────────────────────────────────────────────────────────────

def _cli_args():
    try:
        from comfy.cli_args import args  # type: ignore
        return args
    except Exception:  # noqa: BLE001 - outside ComfyUI
        return None


def own_base_url(server=None) -> str:
    """http(s)://host:port of the running server, from the address it actually bound."""
    args = _cli_args()
    if server is None:
        try:
            import server as _srv  # type: ignore
            server = _srv.PromptServer.instance
        except Exception:  # noqa: BLE001
            server = None
    addr = getattr(server, "address", None)
    port = getattr(server, "port", None)
    if not port:
        listen = str(getattr(args, "listen", "") or "127.0.0.1")
        addr = listen.split(",")[0].strip()
        port = getattr(args, "port", None) or 8188
    addr = str(addr or "127.0.0.1").strip()
    if addr in ("0.0.0.0", ""):
        addr = "127.0.0.1"
    elif addr == "::":
        addr = "::1"
    host = f"[{addr}]" if ":" in addr and not addr.startswith("[") else addr
    scheme = "https" if getattr(args, "tls_keyfile", None) else "http"
    return f"{scheme}://{host}:{int(port)}"


def _error_text(body: Any) -> str:
    """Plain English from a /prompt 400 body: the error message plus the first node error."""
    if not isinstance(body, dict):
        return str(body)[:300]
    err = body.get("error")
    msg = err.get("message", "") if isinstance(err, dict) else str(err or "")
    details = err.get("details", "") if isinstance(err, dict) else ""
    for nid, ne in (body.get("node_errors") or {}).items():
        for e in (ne or {}).get("errors", [])[:1]:
            cls = (ne or {}).get("class_type", "")
            msg = f"{msg} - node {nid} {cls}: {e.get('message', '')} {e.get('details', '')}".strip()
            break
        break
    if details and details not in msg:
        msg = f"{msg} ({details})"
    return msg.strip() or "the server refused the prompt"


async def queue_prompt(prompt: dict, client_id: str = "c2c-automation", server=None) -> dict:
    """Queue an API prompt on this server through its own /prompt route (validation included).
    Returns {"ok": bool, "prompt_id": str|None, "error": str|None}."""
    import aiohttp
    url = own_base_url(server) + "/prompt"
    try:
        async with aiohttp.ClientSession() as sess:
            async with sess.post(url, json={"prompt": prompt, "client_id": client_id}, ssl=False,
                                 timeout=aiohttp.ClientTimeout(total=30)) as r:
                try:
                    body = await r.json(content_type=None)
                except Exception:  # noqa: BLE001
                    body = {"error": (await r.text())[:300]}
                if r.status == 200 and isinstance(body, dict) and body.get("prompt_id"):
                    return {"ok": True, "prompt_id": body["prompt_id"], "error": None}
                return {"ok": False, "prompt_id": None, "error": f"HTTP {r.status}: {_error_text(body)}"}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "prompt_id": None, "error": f"could not reach {url}: {exc}"}


# ── workflows ────────────────────────────────────────────────────────────────

def _is_api(obj: Any) -> bool:
    return (isinstance(obj, dict) and bool(obj)
            and all(isinstance(v, dict) and "class_type" in v for v in obj.values()))


def parse_api_workflow(obj: Any) -> dict:
    """API prompt from: an API dict, {"prompt": {...}}, or {"output": {...}} (graphToPrompt). UI format refused."""
    if isinstance(obj, (str, bytes)):
        obj = json.loads(obj)
    if not isinstance(obj, dict) or not obj:
        raise ValueError("The workflow is empty.")
    if "nodes" in obj and "links" in obj:
        raise ValueError(UI_WORKFLOW_ERROR)
    for key in ("prompt", "output"):
        inner = obj.get(key)
        if _is_api(inner):
            return inner
    if _is_api(obj):
        return obj
    raise ValueError("Not an API workflow: every entry needs a class_type and inputs.")


SEED_INPUTS = ("seed", "noise_seed")


def reseed(prompt: dict, rng=None) -> dict:
    """New literal seeds for every seed / noise_seed input (links are left alone). ComfyUI caches an identical
    prompt completely, so a scheduled or triggered re-run of the same workflow would otherwise do nothing."""
    import random
    rng = rng or random.SystemRandom()
    out = json.loads(json.dumps(prompt))
    for node in out.values():
        ins = node.get("inputs") or {}
        for k in SEED_INPUTS:
            if isinstance(ins.get(k), int) and not isinstance(ins.get(k), bool):
                ins[k] = rng.randint(0, 2 ** 48)
    return out


def apply_overrides(prompt: dict, overrides: Any) -> tuple[dict, list[str]]:
    """{"<node id>": {"<input>": value}} onto literal inputs of existing nodes; returns (prompt, ignored keys)."""
    out = json.loads(json.dumps(prompt))
    ignored: list[str] = []
    if not isinstance(overrides, dict):
        return out, ignored
    for nid, vals in overrides.items():
        node = out.get(str(nid))
        if node is None or not isinstance(vals, dict):
            ignored.append(str(nid))
            continue
        ins = node.setdefault("inputs", {})
        for k, v in vals.items():
            if isinstance(ins.get(k), list) or isinstance(v, (dict, list)):
                ignored.append(f"{nid}.{k}")          # never rewire links or inject structures
                continue
            ins[k] = v
    return out, ignored


def load_api_workflow(path: str) -> dict:
    p = Path(str(path or "").strip().strip('"'))
    if not p.is_file():
        raise ValueError(f"Workflow file not found: {p}")
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Workflow file is not valid JSON: {exc}") from exc
    return parse_api_workflow(obj)


# ── file injection (Watch Folder) ────────────────────────────────────────────

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".exr"}
VIDEO_EXTS = {".mp4", ".mov", ".webm", ".avi", ".mkv", ".m4v", ".mxf"}
# class_type -> (input name, "copy" into input/c2c_watch or "abs" absolute path), first match wins
IMAGE_LOADERS = {"LoadImage": ("image", "copy"), "LoadImageMask": ("image", "copy")}
VIDEO_LOADERS = {"LoadVideoPathC2C": ("video", "abs"), "VHS_LoadVideoPath": ("video", "abs"),
                 "LoadVideoC2C": ("video", "copy"), "VHS_LoadVideo": ("video", "copy")}
WATCH_SUBDIR = "c2c_watch"


def file_kind(path: str) -> str:
    ext = Path(path).suffix.lower()
    return "image" if ext in IMAGE_EXTS else "video" if ext in VIDEO_EXTS else "other"


def copy_to_input(src: str, input_dir: str | None = None) -> str:
    """Copy into <input>/c2c_watch/ under a unique name; returns the name relative to the input dir."""
    if input_dir is None:
        import folder_paths  # type: ignore
        input_dir = folder_paths.get_input_directory()
    dst_dir = Path(input_dir) / WATCH_SUBDIR
    dst_dir.mkdir(parents=True, exist_ok=True)
    s = Path(src)
    dst = dst_dir / s.name
    if dst.exists():
        dst = dst_dir / f"{s.stem}_{uuid.uuid4().hex[:6]}{s.suffix}"
    shutil.copy2(s, dst)
    return f"{WATCH_SUBDIR}/{dst.name}"


def inject_file(prompt: dict, path: str, input_dir: str | None = None) -> tuple[dict, str]:
    """Put `path` into the first loader that takes its kind. Returns (new prompt, "node id class_type")."""
    kind = file_kind(path)
    table = IMAGE_LOADERS if kind == "image" else VIDEO_LOADERS if kind == "video" else {}
    if not table:
        raise ValueError(f"{Path(path).name}: not an image or video file.")
    out = json.loads(json.dumps(prompt))
    for nid in sorted(out, key=lambda k: (len(str(k)), str(k))):
        node = out[nid]
        spec = table.get(node.get("class_type"))
        if not spec:
            continue
        name, how = spec
        ins = node.setdefault("inputs", {})
        if node.get("class_type") == "LoadVideoC2C" and str(ins.get("location") or "") == "server path":
            name, how = "path", "abs"          # L9.89: the merged Load Video Path mode takes the absolute path
        ins[name] = (str(Path(path).resolve()) if how == "abs" else copy_to_input(path, input_dir))
        return out, f"{nid} {node['class_type']}"
    want = ", ".join(table)
    raise ValueError(f"The workflow has no {kind} loader to put the new file into (one of: {want}).")


# ── persistence ──────────────────────────────────────────────────────────────

def store_path() -> Path:
    if _STORE_OVERRIDE is not None:
        return _STORE_OVERRIDE / STORE_NAME
    import folder_paths  # type: ignore
    return Path(folder_paths.get_user_directory()) / "default" / "c2c" / STORE_NAME


def load_section(name: str) -> list:
    try:
        data = json.loads(store_path().read_text(encoding="utf-8"))
    except FileNotFoundError:
        return []
    except Exception as exc:  # noqa: BLE001
        log.warning("automation store unreadable (%s) - starting empty", exc)
        return []
    val = data.get(name, []) if isinstance(data, dict) else []
    return val if isinstance(val, list) else []


def save_section(name: str, items: list) -> None:
    """Replace one section; secrets are stripped as a last guard."""
    clean = [{k: v for k, v in it.items() if k != "secret"} for it in items if isinstance(it, dict)]
    with _LOCK:
        p = store_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                data = {}
        except Exception:  # noqa: BLE001
            data = {}
        data[name] = clean
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
        os.replace(tmp, p)


# ── secrets ──────────────────────────────────────────────────────────────────

_MEM_SECRETS: dict[str, str] = {}


def put_secret(name: str, value: str) -> str:
    """Store a webhook secret. Returns where: "vault" or "memory" (no backend: lost on restart)."""
    if not value:
        drop_secret(name)
        return "none"
    try:
        from . import _c2c_secrets as S
        S.set_(S.SCOPE_INTEGRATIONS, name, value)
        _MEM_SECRETS.pop(name, None)
        return "vault"
    except Exception as exc:  # noqa: BLE001
        log.info("secret %s kept in memory only: %s", name, exc)
        _MEM_SECRETS[name] = value
        return "memory"


def get_secret(name: str) -> str:
    if name in _MEM_SECRETS:
        return _MEM_SECRETS[name]
    try:
        from . import _c2c_secrets as S
        return S.get(S.SCOPE_INTEGRATIONS, name) or ""
    except Exception:  # noqa: BLE001
        return ""


def drop_secret(name: str) -> None:
    _MEM_SECRETS.pop(name, None)
    try:
        from . import _c2c_secrets as S
        if S.has(S.SCOPE_INTEGRATIONS, name):
            S.delete(S.SCOPE_INTEGRATIONS, name)
    except Exception:  # noqa: BLE001
        pass
