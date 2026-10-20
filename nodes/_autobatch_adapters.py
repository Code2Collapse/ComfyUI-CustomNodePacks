"""Operation adapters for Image Batching's Universal mode (L2.54, owner A14 2026-10-15).

An adapter re-implements ONE known operation (core, third-party or C2C) so that it gives the SAME result faster, and
Universal uses it only where it was proven so on this machine:

* fingerprint - the SHA-256 of the original function's source; a changed original (a core update) is never adapted
  until the adapter is revalidated against it;
* self-test - before first use the adapter runs against the ORIGINAL on small random inputs covering the borders and
  the parameter extremes; any difference disables it for the session, with the reason in the diagnostics;
* measured choice - the first eligible call per (operation, shape bucket, device) times the original and the adapter
  once on a slice of at most four frames and keeps the faster; the choice is persisted
  (user/default/c2c/autobatch_profiles.json) and re-measured when the fingerprint, the adapter version, the device,
  torch's version or the bucket changes, or when a run is more than twice as slow as predicted;
* no side effects - adapters are pure tensor functions that never mutate their inputs, so a failure before a result
  exists falls back to the original call; a genuine error from the original is raised unchanged.

Kill switch: ``C2C_AUTOBATCH_ADAPTERS=0``. Design: docs/evidence/L2.54/design_universal_image_batching.md (work area).
"""
from __future__ import annotations

import hashlib
import inspect
import json
import logging
import math
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

import torch
import torch.nn.functional as F

log = logging.getLogger("C2C.autobatch.adapters")

PROFILE_VERSION = 1
SLICE_FRAMES = 4
REMEASURE_IF_SLOWER = 2.0


def adapters_enabled() -> bool:
    return os.environ.get("C2C_AUTOBATCH_ADAPTERS", "").strip() != "0"


# ── adapters ─────────────────────────────────────────────────────────────────────────────────────────────────────────

def grow_mask_batched(mask: torch.Tensor, expand: int, tapered_corners: bool) -> torch.Tensor:
    """Core Grow Mask in one pass over the batch: per iteration a 3x3 max (min for negative expand) - the cross
    footprint when tapered, the full square otherwise. Core runs scipy.ndimage.grey_dilation / grey_erosion frame by
    frame with mode 'reflect'; for a 3x3 window the reflected neighbour is the edge pixel itself, which the max / min
    already includes, so ignoring out-of-frame pixels selects exactly the same values (bit-identical, tested)."""
    h, w = int(mask.shape[-2]), int(mask.shape[-1])
    x = mask.reshape((-1, 1, h, w))
    n = abs(int(expand))
    if n == 0:
        return x.reshape((-1, h, w)).clone()
    neg = expand < 0
    if neg:
        x = -x
    for _ in range(n):
        if tapered_corners:
            p = F.pad(x, (1, 1, 1, 1), mode="replicate")
            x = torch.maximum(torch.maximum(torch.maximum(p[..., 1:-1, :-2], p[..., 1:-1, 2:]),
                                            torch.maximum(p[..., :-2, 1:-1], p[..., 2:, 1:-1])), x)
        else:
            x = F.max_pool2d(x, 3, stride=1, padding=1)
    if neg:
        x = -x
    return x.reshape((-1, h, w))


def feather_mask_strips(mask: torch.Tensor, left: int, top: int, right: int, bottom: int) -> torch.Tensor:
    """Core Feather Mask with one multiply per border strip instead of one per column / row. Core multiplies each
    pixel of the strip by the Python float (i+1)/n, which torch rounds to the tensor's dtype; here the same ramp is a
    tensor of that dtype, applied in core's order (left, right, top, bottom), so every product is the same IEEE
    operation (bit-identical, tested, overlapping strips included). Measured 1.2-26x faster on CPU
    (docs/evidence/L2.54/feather_adapter_probe.json)."""
    out = mask.reshape((-1, mask.shape[-2], mask.shape[-1])).clone()
    h, w = int(out.shape[-2]), int(out.shape[-1])
    left, right, top, bottom = min(left, w), min(right, w), min(top, h), min(bottom, h)
    kw = {"dtype": out.dtype, "device": out.device}
    if left:
        out[:, :, :left] *= torch.tensor([(x + 1.0) / left for x in range(left)], **kw)
    if right:
        out[:, :, w - right:] *= torch.tensor([(x + 1) / right for x in range(right)][::-1], **kw)
    if top:
        out[:, :top, :] *= torch.tensor([(y + 1) / top for y in range(top)], **kw)[:, None]
    if bottom:
        out[:, h - bottom:, :] *= torch.tensor([(y + 1) / bottom for y in range(bottom)][::-1], **kw)[:, None]
    return out


def _node_output(value: Any) -> Any:
    """A V3 NodeOutput, as the original V3 node returns (a plain tuple outside ComfyUI, for the tests)."""
    try:
        from comfy_api.latest import io  # noqa: WPS433 - core only exists inside ComfyUI
    except Exception:  # noqa: BLE001
        return (value,)
    return io.NodeOutput(value)


def _first_tensor(result: Any) -> torch.Tensor | None:
    """The first output tensor of a V3 NodeOutput / V1 tuple."""
    for attr in ("result", "args"):
        val = getattr(result, attr, None)
        if isinstance(val, (tuple, list)) and val and torch.is_tensor(val[0]):
            return val[0]
    if isinstance(result, (tuple, list)) and result and torch.is_tensor(result[0]):
        return result[0]
    return None


@dataclass
class Adapter:
    class_id: str
    version: int
    supports: Callable[[Mapping[str, Any]], bool]
    run: Callable[[Mapping[str, Any]], torch.Tensor]
    wrap: Callable[[torch.Tensor], Any]
    selftest_cases: Callable[[], list[dict[str, Any]]]
    frame_input: str
    exact: bool = True
    state: dict[str, Any] = field(default_factory=dict)


def _grow_cases() -> list[dict[str, Any]]:
    g = torch.Generator().manual_seed(7)
    cases = []
    for (n, h, w) in ((3, 9, 11), (2, 1, 5), (2, 6, 1)):
        m = (torch.rand(n, h, w, generator=g) > 0.6).float() * torch.rand(n, h, w, generator=g)
        m[:, 0, :] = torch.rand(n, w, generator=g)
        for expand in (0, 1, 3, -1, -2):
            for tapered in (True, False):
                cases.append({"mask": m.clone(), "expand": expand, "tapered_corners": tapered})
    return cases


def _grow_supports(args: Mapping[str, Any]) -> bool:
    m = args.get("mask")
    return (torch.is_tensor(m) and m.dtype in (torch.float32, torch.float16, torch.float64) and m.ndim in (2, 3)
            and isinstance(args.get("expand"), int) and isinstance(args.get("tapered_corners"), bool)
            and m.device.type == "cpu")   # core's own path is CPU-only (m.numpy()); a GPU mask runs the original


_SIDES = ("left", "top", "right", "bottom")


def _feather_cases() -> list[dict[str, Any]]:
    g = torch.Generator().manual_seed(11)
    cases = []
    for (n, h, w) in ((3, 9, 11), (2, 1, 5), (2, 6, 1)):
        m = torch.rand(n, h, w, generator=g)
        for dt in (torch.float32, torch.float64):
            for sides in ((0, 0, 0, 0), (1, 1, 1, 1), (3, 0, 5, 2), (8, 8, 8, 8), (40, 2, 40, 2)):
                cases.append({"mask": m.to(dt), **dict(zip(_SIDES, sides))})
    return cases


def _feather_supports(args: Mapping[str, Any]) -> bool:
    m = args.get("mask")
    return (torch.is_tensor(m) and m.dtype in (torch.float32, torch.float64) and m.ndim in (2, 3)
            and all(isinstance(args.get(k), int) and not isinstance(args.get(k), bool) and args[k] >= 0 for k in _SIDES)
            # half precision: core multiplies in float and rounds once, a half ramp would round twice. GPU: not yet
            # validated on a GPU (design slice 3), so a CUDA mask runs the original.
            and m.device.type == "cpu")


ADAPTERS: dict[str, Adapter] = {
    "GrowMask": Adapter(
        class_id="GrowMask", version=1, supports=_grow_supports,
        run=lambda a: grow_mask_batched(a["mask"], a["expand"], a["tapered_corners"]),
        wrap=_node_output, selftest_cases=_grow_cases, frame_input="mask",
    ),
    "FeatherMask": Adapter(
        class_id="FeatherMask", version=1, supports=_feather_supports,
        run=lambda a: feather_mask_strips(a["mask"], a["left"], a["top"], a["right"], a["bottom"]),
        wrap=_node_output, selftest_cases=_feather_cases, frame_input="mask",
    ),
}


# ── validation, measured choice, persistence ────────────────────────────────────────────────────────────────────────

_LOCK = threading.Lock()
_PROFILES: dict[str, dict[str, Any]] | None = None
_PROFILE_PATH: Path | None = None
DECISIONS: list[dict[str, Any]] = []          # the last calls, for the diagnostics (bounded)


def fingerprint(func: Callable[..., Any]) -> str:
    try:
        src = inspect.getsource(func)
    except (OSError, TypeError):
        src = repr(getattr(func, "__code__", func))
    return hashlib.sha256(src.encode("utf-8")).hexdigest()


def _profile_path() -> Path | None:
    override = os.environ.get("C2C_AUTOBATCH_PROFILES", "").strip()
    if override:
        return Path(override)
    try:
        import folder_paths  # noqa: WPS433

        return Path(folder_paths.get_user_directory()) / "default" / "c2c" / "autobatch_profiles.json"
    except Exception:
        return None


def _profiles() -> dict[str, dict[str, Any]]:
    global _PROFILES, _PROFILE_PATH
    if _PROFILES is None:
        _PROFILE_PATH = _profile_path()
        _PROFILES = {}
        if _PROFILE_PATH and _PROFILE_PATH.is_file():
            try:
                data = json.loads(_PROFILE_PATH.read_text(encoding="utf-8"))
                if data.get("version") == PROFILE_VERSION:
                    _PROFILES = dict(data.get("profiles") or {})
            except Exception:  # noqa: BLE001 - an unreadable file only costs a re-measure
                _PROFILES = {}
    return _PROFILES


def _save_profiles() -> None:
    path = _PROFILE_PATH
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"version": PROFILE_VERSION, "profiles": _PROFILES}, indent=1), encoding="utf-8")
        os.replace(tmp, path)
    except Exception as exc:  # noqa: BLE001
        log.debug("[C2C autobatch] profiles not saved: %s", exc)


def bucket(t: torch.Tensor) -> str:
    frames = int(t.shape[0]) if t.ndim >= 3 else 1
    pixels = int(t.shape[-1]) * int(t.shape[-2]) if t.ndim >= 2 else int(t.numel())
    return f"{str(t.dtype).replace('torch.', '')}|px2^{max(0, int(math.log2(max(1, pixels))))}|f2^{max(0, int(math.log2(max(1, frames))))}"


def profile_key(adapter: Adapter, fp: str, t: torch.Tensor) -> str:
    return "|".join([adapter.class_id, f"v{adapter.version}", fp[:16], t.device.type,
                     f"torch{torch.__version__.split('+')[0]}", bucket(t)])


def _same(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.shape == b.shape and a.dtype == b.dtype and bool(torch.equal(a, b))


def validate(adapter: Adapter, original: Callable[..., Any]) -> tuple[bool, str]:
    """Self-test against the original, once per original fingerprint. Inputs must come back unmodified."""
    fp = fingerprint(original)
    st = adapter.state
    if st.get("fp") == fp and "ok" in st:
        return st["ok"], st["reason"]
    ok, reason = True, "self-test passed"
    try:
        for case in adapter.selftest_cases():
            before = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in case.items()}
            ref = _first_tensor(original(**case))
            got = adapter.run(case)
            if ref is None or not _same(ref, got):
                ok, reason = False, f"differs from the original for {{{', '.join(f'{k}={v}' for k, v in case.items() if not torch.is_tensor(v))}}}"
                break
            if any(torch.is_tensor(v) and not torch.equal(v, case[k]) for k, v in before.items()):
                ok, reason = False, "mutated its input"
                break
    except Exception as exc:  # noqa: BLE001
        ok, reason = False, f"self-test failed: {exc}"
    st.update(fp=fp, ok=ok, reason=reason)
    if not ok:
        log.warning("[C2C autobatch] adapter for %s disabled: %s", adapter.class_id, reason)
    return ok, reason


def emit_decision(entry: dict[str, Any]) -> None:
    """Tag a decision with the node that is running (core sets server.last_node_id before each node) and send it to
    the browser as ``c2c.autobatch.decision`` for the node badge + the sidebar table. Never raises."""
    try:
        from server import PromptServer  # noqa: WPS433

        srv = getattr(PromptServer, "instance", None)
        if srv is None:
            return
        entry.setdefault("node", getattr(srv, "last_node_id", None))
        if getattr(srv, "client_id", None) is not None and entry.get("node") is not None:
            srv.send_sync("c2c.autobatch.decision", dict(entry), srv.client_id)
    except Exception:  # noqa: BLE001 - diagnostics must never break a run
        pass


def _record(entry: dict[str, Any]) -> None:
    emit_decision(entry)
    DECISIONS.append(entry)
    del DECISIONS[:-50]


def try_adapter(class_id: str, original: Callable[..., Any], args: Mapping[str, Any]) -> tuple[bool, Any]:
    """(True, result) when the adapter produced the result; (False, None) -> the caller runs its normal path.
    ``original(**kwargs)`` must call the node's ORIGINAL function (not the wrapper)."""
    adapter = ADAPTERS.get(class_id)
    if adapter is None or not adapters_enabled() or not adapter.supports(args):
        return False, None
    ok, reason = validate(adapter, original)
    t = args[adapter.frame_input]
    if not ok:
        _record({"label": class_id, "strategy": "original", "reason": f"adapter off: {reason}"})
        return False, None
    fp = adapter.state["fp"]
    key = profile_key(adapter, fp, t)
    frames = int(t.shape[0]) if t.ndim >= 3 else 1
    with _LOCK:
        prof = _profiles().get(key)
    if prof is None:
        # measure once on a slice of at most SLICE_FRAMES frames: original vs adapter, same inputs, same result
        k = min(frames, SLICE_FRAMES)
        sl = dict(args)
        sl[adapter.frame_input] = t[:k] if t.ndim >= 3 else t
        t0 = time.perf_counter()
        ref = _first_tensor(original(**sl))
        t1 = time.perf_counter()
        got = adapter.run(sl)
        t2 = time.perf_counter()
        if ref is None or not _same(ref, got):
            adapter.state.update(ok=False, reason="differed from the original on a real input")
            _record({"label": class_id, "strategy": "original", "reason": "adapter off: differed on a real input"})
            return False, None
        orig_pf, adap_pf = (t1 - t0) / k, (t2 - t1) / k
        prof = {"choice": "adapter" if adap_pf < orig_pf else "original",
                "original_s_per_frame": orig_pf, "adapter_s_per_frame": adap_pf, "measured_frames": k,
                "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
        with _LOCK:
            _profiles()[key] = prof
            _save_profiles()
    if prof["choice"] != "adapter":
        _record({"label": class_id, "strategy": "original", "frames": frames, "reason": "original measured faster here",
                 "speedup_predicted": round(prof["original_s_per_frame"] / max(1e-12, prof["adapter_s_per_frame"]), 2)})
        return False, None
    t0 = time.perf_counter()
    try:
        out = adapter.run(args)
    except Exception as exc:  # noqa: BLE001 - no side effects happened: the original runs instead
        _record({"label": class_id, "strategy": "original", "reason": f"adapter error, fell back: {exc}"})
        return False, None
    took = time.perf_counter() - t0
    predicted = prof["adapter_s_per_frame"] * frames
    if predicted > 0 and took > REMEASURE_IF_SLOWER * predicted and took > 0.05:
        with _LOCK:
            _profiles().pop(key, None)        # measure again next time
    _record({"label": class_id, "strategy": "adapter", "frames": frames, "device": t.device.type,
             "seconds": round(took, 4),
             "speedup_predicted": round(prof["original_s_per_frame"] / max(1e-12, prof["adapter_s_per_frame"]), 2)})
    return True, adapter.wrap(out)


def status() -> dict[str, Any]:
    return {
        "enabled": adapters_enabled(),
        "adapters": {k: {"version": a.version, "validated": a.state.get("ok"), "reason": a.state.get("reason")}
                     for k, a in ADAPTERS.items()},
        "profiles": dict(_profiles()),
        "decisions": list(DECISIONS),
    }
