# FILE: nodes/propainter_bridge.py
# FEATURE: P0 — ProPainter dependency wrapper / model loader
# INTEGRATES WITH: propainter_temporal_inpaint.py, propainter_flow_refine.py
"""
Single source of truth for everything ProPainter related: import guards,
model download / load, device placement, and a tiny in-process LRU cache so
P1 and P2 don't reload weights between queue runs.

Hard rules implemented here:
  - All imports are guarded. HAS_PROPAINTER tells callers whether to error
    out with a clear `pip install` message.
  - No "cuda" hardcode. Device follows tensors / comfy.model_management.
  - No `pass`, no `...`, no NotImplementedError.
"""
from __future__ import annotations

import logging
import os
import sys
import threading
from dataclasses import dataclass
from typing import Optional, Tuple

import torch

log = logging.getLogger("MEC.propainter")


# ---------- ProPainter, only where the USER put it --------------------
# ProPainter (sczhou/ProPainter) is under the S-Lab License 1.0: non-commercial use only. This pack never ships,
# downloads or installs it (owner 2026-10-12, ledger L7.92) - it used to fetch the GitHub zip into third_party/ on
# the first import, i.e. at every fresh ComfyUI start, without asking. It is used when the user already has a
# checkout (it has no setup.py, and the PyPI name `propainter` is an unrelated ~6 KB package):
#   1. the folder in the environment variable C2C_PROPAINTER_DIR, or
#   2. <this pack>/third_party/ProPainter (a copy the user placed there; third_party/ is not part of the repo).
PROPAINTER_REPO = "https://github.com/sczhou/ProPainter"
PROPAINTER_LICENSE_NOTE = ("ProPainter is licensed under the S-Lab License 1.0 (non-commercial use only); see "
                           f"{PROPAINTER_REPO}/blob/main/LICENSE. Using it is your decision - this pack does not "
                           "download or install it.")


def _propainter_dir_is_valid(d: str) -> bool:
    """Verify the directory looks like a checkout of sczhou/ProPainter."""
    if not d or not os.path.isdir(d):
        return False
    return (
        os.path.isfile(os.path.join(d, "model", "propainter.py"))
        and os.path.isfile(os.path.join(d, "model", "recurrent_flow_completion.py"))
        and os.path.isdir(os.path.join(d, "RAFT"))
    )


def _propainter_candidates() -> list:
    here = os.path.dirname(os.path.abspath(__file__))
    pack_root = os.path.dirname(here)
    out = []
    env = (os.environ.get("C2C_PROPAINTER_DIR") or "").strip()
    if env:
        out.append(os.path.abspath(env))
    out.append(os.path.join(pack_root, "third_party", "ProPainter"))
    return out


def _add_installed_propainter_to_path() -> Optional[str]:
    """Put an existing ProPainter checkout on sys.path. Never downloads, never creates anything."""
    for candidate in _propainter_candidates():
        if _propainter_dir_is_valid(candidate):
            if candidate not in sys.path:
                sys.path.insert(0, candidate)
            return candidate
    return None


_VENDORED_DIR = _add_installed_propainter_to_path()


# ---------- guarded propainter imports --------------------------------
RecurrentFlowCompleteNet = None  # type: ignore
InpaintGenerator = None  # type: ignore
RAFT = None  # type: ignore
InputPadder = None  # type: ignore
HAS_PROPAINTER = False
_IMPORT_ERROR: Optional[BaseException] = None

try:
    if _VENDORED_DIR is None:
        # Without a checkout the bare `model` / `RAFT` names below could resolve to some OTHER package's top-level
        # modules - never import code we did not locate on purpose.
        raise ImportError("no ProPainter checkout found in: " + ", ".join(_propainter_candidates()))
    # Real sczhou/ProPainter — top-level layout (no `propainter.` prefix).
    from model.recurrent_flow_completion import RecurrentFlowCompleteNet as _RFC  # type: ignore
    from model.propainter import InpaintGenerator as _IG  # type: ignore
    from RAFT import RAFT as _RAFT  # type: ignore
    from RAFT.utils.utils import InputPadder as _IP  # type: ignore
    RecurrentFlowCompleteNet = _RFC
    InpaintGenerator = _IG
    RAFT = _RAFT
    InputPadder = _IP
    HAS_PROPAINTER = True
except Exception as _e1:
    # Fallback: tolerate users who installed a custom `propainter` package
    # with the same submodule layout (rare).
    try:
        from propainter.model.recurrent_flow_completion import RecurrentFlowCompleteNet as _RFC  # type: ignore
        from propainter.model.propainter import InpaintGenerator as _IG  # type: ignore
        from propainter.RAFT import RAFT as _RAFT  # type: ignore
        from propainter.RAFT.utils.utils import InputPadder as _IP  # type: ignore
        RecurrentFlowCompleteNet = _RFC
        InpaintGenerator = _IG
        RAFT = _RAFT
        InputPadder = _IP
        HAS_PROPAINTER = True
    except Exception as _e2:
        _IMPORT_ERROR = _e1 if "propainter" in repr(_e1) else _e2


# ---------- comfy folder paths (may not exist in pure-test env) -------
try:
    import folder_paths  # type: ignore
    _MODELS_ROOT = os.path.join(folder_paths.models_dir, "propainter")
except Exception:
    _MODELS_ROOT = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "models", "propainter")
    )

# Nothing is created at import: the folder is the user's to make when they add the weights.

# Where the user gets the weights themselves (the official v0.1.0 release). Not downloaded by this pack.
_DEFAULT_URLS = {
    "raft-things.pth":
        "https://github.com/sczhou/ProPainter/releases/download/v0.1.0/raft-things.pth",
    "recurrent_flow_completion.pth":
        "https://github.com/sczhou/ProPainter/releases/download/v0.1.0/recurrent_flow_completion.pth",
    "ProPainter.pth":
        "https://github.com/sczhou/ProPainter/releases/download/v0.1.0/ProPainter.pth",
}


# ---------- public errors ---------------------------------------------
class ProPainterMissingError(RuntimeError):
    """Raised by P1 / P2 when HAS_PROPAINTER is False."""


def require_propainter() -> None:
    """Call before any ProPainter use. Single chokepoint for the error msg."""
    if HAS_PROPAINTER:
        return
    target = _propainter_candidates()[-1]
    raise ProPainterMissingError(
        "ProPainter is not installed. This node uses it only when you already have it.\n"
        f"{PROPAINTER_LICENSE_NOTE}\n"
        "To use it: get the code yourself and either set the environment variable C2C_PROPAINTER_DIR to that "
        f"folder or put it at\n    {target}\n(e.g. git clone {PROPAINTER_REPO}.git \"{target}\"), then restart "
        "ComfyUI.\n"
        f"Details: {type(_IMPORT_ERROR).__name__}: {_IMPORT_ERROR}"
    )


# ---------- device helper --------------------------------------------
def get_device(preferred: Optional[torch.device] = None) -> torch.device:
    """Return preferred device or comfy's intermediate device, never hardcoded."""
    if preferred is not None:
        return preferred
    try:
        import comfy.model_management as mm  # type: ignore
        return mm.get_torch_device()
    except Exception:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------- weights: the user's own files ----------------------------
def ensure_weight(name: str) -> str:
    """The absolute path of a ProPainter weight the user placed in models/propainter/. Never downloads (the
    weights carry the same non-commercial licence as the code). Raises FileNotFoundError naming where to put it."""
    path = os.path.join(_MODELS_ROOT, name)
    if os.path.isfile(path):
        return path
    url = _DEFAULT_URLS.get(name)
    raise FileNotFoundError(
        f"ProPainter weight '{name}' is not in {_MODELS_ROOT}. This pack does not download it. "
        + (f"The official file is {url}. " if url else "")
        + f"Put it at {path} and run again. {PROPAINTER_LICENSE_NOTE}"
    )


# ---------- model bundle ---------------------------------------------
@dataclass
class ProPainterModels:
    raft: torch.nn.Module
    flow_complete: torch.nn.Module
    inpaint: torch.nn.Module
    device: torch.device
    half: bool


_LOAD_LOCK = threading.Lock()
_CACHED: Optional[ProPainterModels] = None


def _strip_module(state_dict):
    """RAFT / ProPainter checkpoints sometimes carry the DataParallel `module.` prefix."""
    out = {}
    for k, v in state_dict.items():
        out[k[7:] if k.startswith("module.") else k] = v
    return out


def load_models(device: Optional[torch.device] = None,
                half: bool = True) -> ProPainterModels:
    """Load (or return cached) ProPainter model bundle on the given device."""
    require_propainter()
    global _CACHED
    dev = get_device(device)
    with _LOAD_LOCK:
        if (_CACHED is not None
                and _CACHED.device == dev
                and _CACHED.half == half):
            return _CACHED

        # ---- RAFT (flow estimator) -------------------------------
        # ProPainter's RAFT wrapper takes a Namespace; mimic its `iters`-only init.
        import argparse
        raft_args = argparse.Namespace(
            model=ensure_weight("raft-things.pth"),
            small=False, mixed_precision=False, alternate_corr=False,
        )
        raft_net = torch.nn.DataParallel(RAFT(raft_args))
        raft_state = torch.load(raft_args.model, map_location="cpu")
        raft_net.load_state_dict(raft_state, strict=False)
        raft_module = raft_net.module
        raft_module.to(dev).eval()

        # ---- Recurrent Flow Completion ---------------------------
        fc_path = ensure_weight("recurrent_flow_completion.pth")
        fc_net = RecurrentFlowCompleteNet()
        fc_state = torch.load(fc_path, map_location="cpu")
        fc_net.load_state_dict(_strip_module(fc_state), strict=False)
        fc_net.to(dev).eval()
        for p in fc_net.parameters():
            p.requires_grad = False

        # ---- InpaintGenerator -----------------------------------
        ip_path = ensure_weight("ProPainter.pth")
        ip_net = InpaintGenerator()
        ip_state = torch.load(ip_path, map_location="cpu")
        ip_net.load_state_dict(_strip_module(ip_state), strict=False)
        ip_net.to(dev).eval()
        for p in ip_net.parameters():
            p.requires_grad = False

        if half and dev.type == "cuda":
            fc_net = fc_net.half()
            ip_net = ip_net.half()

        _CACHED = ProPainterModels(
            raft=raft_module, flow_complete=fc_net, inpaint=ip_net,
            device=dev, half=half and dev.type == "cuda",
        )
        log.info("[propainter_bridge] models ready on %s (half=%s)",
                 dev, _CACHED.half)
        return _CACHED


def free_models() -> None:
    """Release the cached bundle and free VRAM."""
    global _CACHED
    with _LOAD_LOCK:
        _CACHED = None
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


# ---------- tensor utilities shared by P1 / P2 -----------------------
def to_propainter_video(images_bhwc: torch.Tensor,
                        device: torch.device,
                        half: bool) -> torch.Tensor:
    """Comfy IMAGE (B, H, W, C) float[0,1] -> propainter video (1, B, C, H, W) [-1,1]."""
    x = images_bhwc.to(device)
    if half:
        x = x.half()
    x = x.permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)
    x = x * 2.0 - 1.0
    return x.unsqueeze(0)  # (1, B, C, H, W)


def from_propainter_video(video_1bchw: torch.Tensor) -> torch.Tensor:
    """ProPainter (1, B, C, H, W) [-1,1] -> Comfy IMAGE (B, H, W, C) float[0,1] cpu."""
    x = video_1bchw.squeeze(0).clamp(-1.0, 1.0)
    x = (x + 1.0) * 0.5
    x = x.permute(0, 2, 3, 1).contiguous().float().cpu()
    return x


def to_propainter_mask(masks_bhw: torch.Tensor,
                       device: torch.device,
                       half: bool) -> torch.Tensor:
    """Comfy MASK (B, H, W) float[0,1] -> propainter mask (1, B, 1, H, W) {0,1}."""
    m = masks_bhw.to(device)
    if half:
        m = m.half()
    m = (m > 0.5).to(m.dtype)
    return m.unsqueeze(0).unsqueeze(2)


# ---------- tiny health check (importable, never raises) -------------
def status() -> Tuple[bool, str]:
    if not HAS_PROPAINTER:
        return False, f"propainter not importable: {_IMPORT_ERROR!r}"
    missing = [n for n in _DEFAULT_URLS
               if not os.path.isfile(os.path.join(_MODELS_ROOT, n))]
    if missing:
        return True, f"propainter OK; weights missing: {missing}"
    return True, "propainter OK; weights present"
