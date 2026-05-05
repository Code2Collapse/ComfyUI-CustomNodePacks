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
import urllib.request
from dataclasses import dataclass
from typing import Optional, Tuple

import torch

log = logging.getLogger("MEC.propainter")


# ---------- vendored ProPainter (sczhou/ProPainter has no setup.py) --
# The PyPI package named `propainter` is an unrelated name-squatter (~6 KB).
# We vendor the real upstream repo at `third_party/ProPainter/` and inject
# its path so its top-level modules `RAFT`, `model.propainter`,
# `model.recurrent_flow_completion` resolve.
#
# If the directory is missing (fresh git clone — `third_party/` is gitignored
# so it never ships in the pack), fetch the upstream zip from GitHub on first
# import. No git binary required, no pip install — just urllib + zipfile.
_PROPAINTER_ZIP_URL = "https://github.com/sczhou/ProPainter/archive/refs/heads/main.zip"


def _propainter_dir_is_valid(d: str) -> bool:
    """Verify the directory looks like a checkout of sczhou/ProPainter."""
    if not os.path.isdir(d):
        return False
    return (
        os.path.isfile(os.path.join(d, "model", "propainter.py"))
        and os.path.isfile(os.path.join(d, "model", "recurrent_flow_completion.py"))
        and os.path.isdir(os.path.join(d, "RAFT"))
    )


def _fetch_propainter_source(target_dir: str) -> bool:
    """Download + extract sczhou/ProPainter main.zip into target_dir.
    Returns True on success."""
    import io
    import shutil
    import tempfile
    import zipfile

    parent = os.path.dirname(target_dir)
    os.makedirs(parent, exist_ok=True)
    log.info("[propainter_bridge] fetching ProPainter source from %s", _PROPAINTER_ZIP_URL)
    try:
        with urllib.request.urlopen(_PROPAINTER_ZIP_URL, timeout=120) as resp:
            blob = resp.read()
    except Exception as e:
        log.warning("[propainter_bridge] download failed: %s", e)
        return False
    try:
        with zipfile.ZipFile(io.BytesIO(blob)) as zf:
            with tempfile.TemporaryDirectory(dir=parent) as tmp:
                zf.extractall(tmp)
                # Top-level inside the zip is `ProPainter-main/`.
                roots = [os.path.join(tmp, n) for n in os.listdir(tmp)]
                roots = [r for r in roots if os.path.isdir(r)]
                if not roots:
                    log.warning("[propainter_bridge] zip had no top-level dir")
                    return False
                src = roots[0]
                # Atomic-ish: move into place. If target already exists, replace.
                if os.path.exists(target_dir):
                    shutil.rmtree(target_dir, ignore_errors=True)
                shutil.move(src, target_dir)
        ok = _propainter_dir_is_valid(target_dir)
        if ok:
            log.info("[propainter_bridge] ProPainter source installed at %s", target_dir)
        else:
            log.warning("[propainter_bridge] extracted dir failed validation: %s", target_dir)
        return ok
    except Exception as e:
        log.warning("[propainter_bridge] extract failed: %s", e)
        return False


def _add_vendored_propainter_to_path() -> Optional[str]:
    here = os.path.dirname(os.path.abspath(__file__))
    pack_root = os.path.dirname(here)
    candidate = os.path.join(pack_root, "third_party", "ProPainter")
    if not _propainter_dir_is_valid(candidate):
        if not _fetch_propainter_source(candidate):
            return None
    if candidate not in sys.path:
        sys.path.insert(0, candidate)
    return candidate


_VENDORED_DIR = _add_vendored_propainter_to_path()


# ---------- guarded propainter imports --------------------------------
RecurrentFlowCompleteNet = None  # type: ignore
InpaintGenerator = None  # type: ignore
RAFT = None  # type: ignore
InputPadder = None  # type: ignore
HAS_PROPAINTER = False
_IMPORT_ERROR: Optional[BaseException] = None

try:
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

os.makedirs(_MODELS_ROOT, exist_ok=True)


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
    here = os.path.dirname(os.path.abspath(__file__))
    pack_root = os.path.dirname(here)
    target = os.path.join(pack_root, "third_party", "ProPainter")
    raise ProPainterMissingError(
        "ProPainter source not available. The bridge tries to auto-download it\n"
        f"from {_PROPAINTER_ZIP_URL} into {target}\n"
        "but that step failed (no internet, blocked URL, or extraction error).\n"
        "Manual fix: clone the repo there yourself:\n"
        f"    git clone https://github.com/sczhou/ProPainter.git \"{target}\"\n"
        f"Original import error: {type(_IMPORT_ERROR).__name__}: {_IMPORT_ERROR}"
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


# ---------- weight download ------------------------------------------
def _download(url: str, dst: str) -> None:
    log.info("[propainter_bridge] downloading %s -> %s", url, dst)
    tmp = dst + ".part"
    with urllib.request.urlopen(url, timeout=60) as resp, open(tmp, "wb") as f:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    os.replace(tmp, dst)


def ensure_weight(name: str) -> str:
    """Resolve a weight filename inside the propainter models folder, downloading
    if missing. Returns the absolute path. Raises FileNotFoundError on failure."""
    path = os.path.join(_MODELS_ROOT, name)
    if os.path.isfile(path):
        return path
    url = _DEFAULT_URLS.get(name)
    if not url:
        raise FileNotFoundError(
            f"Unknown ProPainter weight '{name}'. "
            f"Drop the .pth file under {_MODELS_ROOT}."
        )
    try:
        _download(url, path)
    except Exception as e:
        raise FileNotFoundError(
            f"Could not download {name} from {url}: {e}.\n"
            f"Manually place the file at {path}."
        ) from e
    return path


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
