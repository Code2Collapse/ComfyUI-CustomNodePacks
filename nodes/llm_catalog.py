"""Curated GGUF catalog for the Diagnostics Tier 2 (llama.cpp) assistant (L6.05).

The catalog is data (``data/llm_catalog.json``): every repo / file / size / SHA-256 was read from the Hugging Face API
by ``tools/c2c_llm_catalog_verify.py``. This module loads it, maps a catalog id to its real file name (catalog ids
such as ``qwen3-0.6b-instruct-q8_0`` are not substrings of the file ``Qwen3-0.6B-Q8_0.gguf``, so a name search alone
never found them), and recommends the largest general-purpose model that fits the memory free right now.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from typing import Optional

log = logging.getLogger("C2C.LLMCatalog")
CATALOG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "llm_catalog.json")
GIB = 1 << 30
# llama.cpp needs the weights plus the KV cache and scratch buffers: ~20 % on top of the file + 0.5 GiB at the
# assistant's context size. Only a fraction of free RAM is offered: ComfyUI and its models live in the same RAM.
OVERHEAD_FRACTION, OVERHEAD_FIXED, RAM_SHARE = 0.20, GIB // 2, 0.6
_PREFERRED_FAMILIES = ("Qwen 3", "Qwen 2.5 / Coder", "Google Gemma 3", "Meta Llama 3.2")
_LOCK = threading.Lock()
_CACHE: dict = {"mtime": None, "data": None}


def load(path: Optional[str] = None) -> dict:
    """The catalog dict ({schema, default, models: [...]}); re-read when the file changes. Never raises: a missing
    or broken file gives an empty catalog and a log line."""
    p = path or CATALOG_PATH
    try:
        mtime = os.path.getmtime(p)
    except OSError:
        log.warning("[llm_catalog] %s is missing; the model list is empty", p)
        return {"schema": 1, "default": "", "models": []}
    with _LOCK:
        if path is None and _CACHE["mtime"] == mtime and _CACHE["data"] is not None:
            return _CACHE["data"]
        try:
            with open(p, encoding="utf-8") as fh:
                data = json.load(fh)
            if not isinstance(data.get("models"), list):
                raise ValueError("no 'models' list")
        except (OSError, ValueError) as exc:
            log.warning("[llm_catalog] %s is unreadable (%s); the model list is empty", p, exc)
            return {"schema": 1, "default": "", "models": []}
        if path is None:
            _CACHE.update(mtime=mtime, data=data)
        return data


def by_id(model_id: str, catalog: Optional[dict] = None) -> Optional[dict]:
    mid = (model_id or "").strip().lower()
    return next((m for m in (catalog or load())["models"] if m.get("id", "").lower() == mid), None)


def by_repo_file(repo: str, fname: str, catalog: Optional[dict] = None) -> Optional[dict]:
    return next((m for m in (catalog or load())["models"] if m.get("repo") == repo and m.get("file") == fname), None)


def need_bytes(entry: dict) -> int:
    return int(int(entry.get("size") or 0) * (1 + OVERHEAD_FRACTION)) + OVERHEAD_FIXED


_GPU_OFFLOAD: dict = {}


def llama_gpu_offload() -> bool:
    """True only when the installed llama-cpp-python can put layers on the GPU (the CPU wheel cannot, whatever torch
    sees). Checked once per process; False when llama-cpp-python is not installed."""
    if "ok" not in _GPU_OFFLOAD:
        ok = False
        try:
            import importlib.util
            if importlib.util.find_spec("llama_cpp") is not None:
                import llama_cpp  # type: ignore
                fn = getattr(llama_cpp, "llama_supports_gpu_offload", None)
                ok = bool(fn()) if callable(fn) else False
        except Exception:  # noqa: BLE001 - a broken install counts as "no GPU offload"
            ok = False
        _GPU_OFFLOAD["ok"] = ok
    return _GPU_OFFLOAD["ok"]


def memory_free() -> dict:
    """{ram_available, vram_free} in bytes (None when unknown or unusable). Cheap: no model is touched. Free VRAM is
    only reported when llama.cpp can offload to the GPU."""
    out = {"ram_available": None, "vram_free": None, "gpu_offload": llama_gpu_offload()}
    try:
        import psutil  # type: ignore
        out["ram_available"] = int(psutil.virtual_memory().available)
    except Exception:
        pass
    if out["gpu_offload"]:
        try:
            import torch  # type: ignore
            if torch.cuda.is_available():
                out["vram_free"] = int(torch.cuda.mem_get_info()[0])
        except Exception:
            pass
    return out


def recommend(catalog: Optional[dict] = None, mem: Optional[dict] = None) -> dict:
    """{id, fits, budget, reason}: the largest 'chat' model (preferred families first) whose estimated need fits the
    budget = RAM_SHARE x available RAM (or free VRAM when that is larger: llama.cpp can offload all layers)."""
    cat = catalog or load()
    mem = mem if mem is not None else memory_free()
    ram = mem.get("ram_available") or 0
    vram = mem.get("vram_free") or 0
    budget = max(int(ram * RAM_SHARE), int(vram * 0.9))
    chat = [m for m in cat["models"] if m.get("use") == "chat" and m.get("size")]
    if not chat:
        return {"id": cat.get("default") or "", "fits": False, "budget": budget, "reason": "the catalog is empty"}
    if not budget:
        return {"id": cat.get("default") or chat[0]["id"], "fits": False, "budget": 0,
                "reason": "free memory is unknown; showing the default"}
    rank = {f: i for i, f in enumerate(_PREFERRED_FAMILIES)}
    fitting = [m for m in chat if need_bytes(m) <= budget]
    if not fitting:
        small = min(chat, key=lambda m: m["size"])
        return {"id": small["id"], "fits": False, "budget": budget,
                "reason": f"even the smallest model needs ~{need_bytes(small) / GIB:.1f} GiB; "
                          f"{budget / GIB:.1f} GiB is free for it"}
    best = max(fitting, key=lambda m: (m["size"] // (GIB // 2), -rank.get(m.get("group"), 99), m["size"]))
    where = "GPU" if vram * 0.9 >= ram * RAM_SHARE and vram else "RAM"
    return {"id": best["id"], "fits": True, "budget": budget,
            "reason": f"largest general model that fits {budget / GIB:.1f} GiB of free {where} "
                      f"(needs ~{need_bytes(best) / GIB:.1f} GiB)"}


def _label(n: int) -> str:
    return f"~{n / GIB:.1f} GB" if n >= GIB else f"~{n / (1 << 20):.0f} MB"


def public(catalog: Optional[dict] = None, mem: Optional[dict] = None) -> dict:
    """What the sidebar needs: the models with a size label and estimated memory, plus the recommendation."""
    cat = catalog or load()
    models = []
    for m in cat["models"]:
        need = need_bytes(m)
        size = int(m.get("size") or 0)
        models.append({**m, "ram": _label(need), "need_bytes": need, "size_label": _label(size)})
    return {"schema": cat.get("schema", 1), "verified": cat.get("verified", ""), "default": cat.get("default", ""),
            "models": models, "recommended": recommend(cat, mem)}
