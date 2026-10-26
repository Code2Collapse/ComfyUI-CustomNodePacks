"""Built-in AV handles for Load Video (C2C) and Save Video (C2C) (L9.40, docs/save-video-v2.md).

Load adds them: the count comes from AV Handles' own planner (fit / add, the model frame grids, the record it keeps),
run on a stand-in batch of the right length, and the frames are added LAZILY as repeated edge-frame indices of the
video handle - so IMAGE, mask and the VAE (LATENT) path all get them without decoding the clip first. Audio moves by
the same duration. The count travels in video_info["c2c_handles"].

Save trims them: auto = exactly what video_info says (fit's target length wins over the count, as in AV Handles);
manual = a count and a side; off = nothing. auto never reads AV Handles' remembered count: an implicit trim must not
act on a record from an unrelated earlier run.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import OrderedDict
from typing import Any

import torch

try:
    from .. import av_handles as AV
    from .._frame_grid import CUSTOM, GRID_NAMES, resolve, trim_to
except ImportError:  # tests: the pack root on sys.path, nodes imported as a package
    from nodes import av_handles as AV  # type: ignore
    from nodes._frame_grid import CUSTOM, GRID_NAMES, resolve, trim_to  # type: ignore

KILL_SWITCH = "C2C_VIDEO_HANDLES"
HANDLE_MODES = ("off", "fit", "add")
FROM_FORMAT = "from format"
GRID_CHOICES = (FROM_FORMAT,) + tuple(GRID_NAMES)
TRIM_MODES = ("auto", "manual", "off")
SIDES = AV.SIDES
INFO_KEY = "c2c_handles"


def disabled() -> bool:
    return os.environ.get(KILL_SWITCH, "").strip().lower() in ("0", "off", "false", "no")


def grid_for(handle_grid: str, format_preset: dict, format_name: str, step: int, plus: int) -> tuple[str, int, int, str]:
    """(AV Handles model name, custom step, custom plus, label). 'from format' uses Load Video's format preset grid
    (one grid, not two); a format without a frame rule means no grid."""
    if handle_grid == FROM_FORMAT:
        frames = (format_preset or {}).get("frames")
        if frames:
            s, p = int(frames[0]), int(frames[1])
            return CUSTOM, s, p, f"{format_name} ({s}n+{p}), from format"
        return "none (any frame count)", 1, 0, "no grid (format has no frame rule)"
    if handle_grid not in GRID_NAMES:
        raise AV.AVHandlesError(f"Unknown handle grid {handle_grid!r}; pick one of {', '.join(GRID_CHOICES)}.")
    g = resolve(handle_grid, int(step), int(plus))
    return handle_grid, int(step), int(plus), (g.note if handle_grid == CUSTOM else handle_grid)


def plan_handles(n: int, *, mode: str, frames: int, target: int, side: str, model: str, step: int,
                 plus: int) -> dict:
    """What add_on_load will put on an n-frame clip, without touching anything (no AV Handles memory, no disk): the
    probe shows it before a run (L9.52 - owner: the count was 'not what I typed' and nothing said why).
    -> {head, tail, total, target, asked, note}. Mirrors AVHandlesMEC.run for fit and add (padding = the model grid)."""
    n = max(0, int(n))
    if mode == "fit":
        grid = AV.resolve(model, int(step), int(plus))
        want = int(target) or n
        plan = AV.plan_fit(want, grid)
        used = plan.generate - n
        if used < 0:
            return {"head": 0, "tail": 0, "total": n, "target": want, "asked": want, "error": True,
                    "note": f"the clip is already {n} frames but the target is {want}: fit only pads"}
        head, tail = AV.split_ends(used, side)
        g = "" if grid.step <= 1 else f" on {model}"
        return {"head": head, "tail": tail, "total": n + used, "target": plan.target, "asked": want,
                "note": f"fit {want} frames{g}: generate {plan.generate}, the trim keeps {plan.target}"}
    want = max(0, int(frames))
    grid, label = AV.add_grid(AV.USE_MODEL, model, int(step), int(plus))
    label = label.replace(" (model)", "")                # "Wan 2.x (4n+1) (model)" reads as a typo in the note

    def planned(k: int) -> int:
        return k if grid is None else grid.next_at_or_above(n + k) - n

    if side == "both":
        needed = planned(2 * want)
        head, tail = want, needed - want
        asked = 2 * want
    else:
        head, tail = AV.split_ends(planned(want), side)
        asked = want
    extra = head + tail - asked
    why = f" ({asked} asked, +{extra} to land on {label})" if extra > 0 else ""
    return {"head": head, "tail": tail, "total": n + head + tail, "target": 0, "asked": asked,
            "note": f"adds {head} head + {tail} tail -> {n + head + tail} frames{why}"}


def add_on_load(loaded, audio, *, mode: str, frames: int, target: int, side: str, model: str, step: int, plus: int,
                fps: float, memo_key: str = "") -> tuple[Any, Any, dict, str]:
    """(handle with the handles, audio moved to match, record, report). mode = fit | add."""
    if mode not in ("fit", "add"):
        raise AV.AVHandlesError(f"handles must be off, fit or add; got {mode!r}.")
    n = len(loaded)
    proxy = torch.zeros(n, 1, 1, 1)                      # AV Handles plans from the length only
    _img, out_aud, total, rec, info = AV.AVHandlesMEC().run(
        mode=mode, handle_frames=int(frames), target_frames=int(target), model=model, custom_step=int(step),
        custom_plus=int(plus), images=proxy, audio=audio, side=side,
        padding_mode=AV.USE_MODEL if mode == "add" else "disabled", manual_fps=float(fps), memo_key=memo_key)
    head, tail = int(rec["head"]), int(rec["tail"])
    sel = list(loaded.selected_indices())
    if head or tail:
        from .handle import TransformOp
        loaded = loaded._selected([sel[0]] * head + sel + [sel[-1]] * tail, TransformOp("handles", (head, tail)))
    assert len(loaded) == total == n + head + tail
    rec = dict(rec, fps=float(fps))
    return loaded, out_aud, rec, info


class FrameWindow:
    """frames [start, start + count) of a tensor batch or a FrameStream, read front to back like the encoders do."""

    def __init__(self, frames, start: int, count: int):
        self._f, self._start = frames, int(start)
        n, h, w, c = frames.shape
        self.shape = (int(count), int(h), int(w), int(c))
        self.ndim = 4
        self.dtype = getattr(frames, "dtype", torch.float32)

    def __getitem__(self, key):
        if isinstance(key, slice):
            a = 0 if key.start is None else int(key.start)
            b = self.shape[0] if key.stop is None else min(int(key.stop), self.shape[0])
            return self._f[self._start + a:self._start + b]
        if isinstance(key, int):
            return self._f[self._start + key]
        raise TypeError("frames are read by index or slice")

    def __getattr__(self, name):                       # last_frame, delivered, close, _transform ...
        return getattr(self._f, name)


def window(frames, start: int, count: int):
    if start == 0 and count == frames.shape[0]:
        return frames
    if torch.is_tensor(frames):
        return frames[start:start + count]
    return FrameWindow(frames, start, count)


def trim_count(n: int, video_info: Any, mode: str, frames: int, side: str) -> tuple[int, int, str]:
    """(head, tail, where it came from) for Save Video's trim."""
    if mode == "off" or disabled():
        return 0, 0, "trim off" if mode == "off" else f"handles switched off ({KILL_SWITCH}=0)"
    if mode == "manual":
        k = max(0, int(frames))
        head, tail = (k, k) if side == "both" else AV.split_ends(k, side)
        return head, tail, f"manual: {k} frame(s) on {'each end' if side == 'both' else 'the ' + side}"
    rec = video_info.get(INFO_KEY) if isinstance(video_info, dict) else None
    if not isinstance(rec, dict):
        return 0, 0, "no handles in video_info - nothing trimmed (wire Load Video's video_info, or trim manually)"
    head, tail = int(rec.get("head", 0)), int(rec.get("tail", 0))
    tgt = int(rec.get("target") or 0)
    if tgt:
        excess = trim_to(n, tgt)
        if excess != head + tail:
            head, tail = AV.split_ends(excess, str(rec.get("side") or "head"))
        return head, tail, f"auto: fit's target of {tgt} frames ({n} arrived)"
    return head, tail, f"auto: {head} head + {tail} tail frame(s) added by Load Video"


def trim_on_save(images, audio, video_info, *, mode: str, frames: int, side: str, fps: float):
    """(frames, audio, report). Refuses a trim that would leave nothing, with the numbers."""
    n = int(images.shape[0])
    head, tail, origin = trim_count(n, video_info, mode, frames, side)
    if head + tail >= n and head + tail > 0:
        raise AV.AVHandlesError(f"Trimming {head} + {tail} frames off a {n}-frame clip would leave nothing; "
                                f"the handles ({origin}) do not belong to these frames.")
    out = window(images, head, n - head - tail)
    out_aud = AV.shift_audio_ends(audio, head, tail, fps, add=False) if audio is not None else None
    return out, out_aud, {"head": head, "tail": tail, "from": origin, "frames_in": n, "frames_out": n - head - tail}


def export_range(images, audio, range_in: int, range_out: int, fps: float):
    """Frames [range_in, range_out] (inclusive, of the trimmed clip; 0 / 0 = all), clamped and reported."""
    n = int(images.shape[0])
    a, b = int(range_in or 0), int(range_out or 0)
    if a <= 0 and b <= 0:
        return images, audio, None, (0, n - 1)
    a = max(0, min(a, n - 1))
    b = n - 1 if b <= 0 else max(a, min(b, n - 1))
    note = f"frames {a}-{b} of {n}" + (" (clamped)" if (int(range_in) != a or (int(range_out) and int(range_out) != b))
                                        else "")
    out = window(images, a, b - a + 1)
    if audio is not None and fps > 0:
        head, tail = a, n - 1 - b
        audio = AV.shift_audio_ends(audio, head, tail, fps, add=False)
    return out, audio, note, (a, b)


# ── Save Video without a video_info wire (L9.52) ────────────────────────────────────────────────────────────────
# owner 2026-10-25: "handles also it should automatically detect from load video and trim". Save follows its images
# input back to the Load Video it came from and uses the video_info that Load produced. A Load records its video_info
# here when it runs, with a signature of its own prompt inputs: a Load that is cached (not re-run) still has the
# record of the run that produced the cached frames, and a record of different settings is never used.

LOADER_CLASSES = ("LoadVideoC2C", "LoadVideoPathC2C")
_LOADS: "OrderedDict[str, tuple[str, dict]]" = OrderedDict()
_LOADS_MAX = 64
MAX_HOPS = 256


def inputs_signature(inputs: Any) -> str:
    return hashlib.sha1(json.dumps(inputs, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def remember_load(unique_id: Any, prompt: Any, video_info: dict) -> None:
    """Called by the loaders after building video_info (bounded: the last 64 loader nodes)."""
    node = prompt.get(str(unique_id)) if isinstance(prompt, dict) else None
    if unique_id is None or not isinstance(node, dict):
        return
    key = str(unique_id)
    _LOADS[key] = (inputs_signature(node.get("inputs") or {}), dict(video_info or {}))
    _LOADS.move_to_end(key)
    while len(_LOADS) > _LOADS_MAX:
        _LOADS.popitem(last=False)


def _is_link(v: Any) -> bool:
    return isinstance(v, (list, tuple)) and len(v) == 2 and isinstance(v[0], (str, int)) and isinstance(v[1], int)


def _typed_input_names(class_type: str, kind: str) -> list:
    """Input names of a class in declaration order whose type carries `kind` ("IMAGE" also matches "IMAGE,LATENT"),
    then its wildcard ("*") inputs. [] when the class is unknown."""
    try:
        import nodes as _core_nodes
        cls = _core_nodes.NODE_CLASS_MAPPINGS.get(class_type)
        spec = cls.INPUT_TYPES() if cls is not None else None
    except Exception:  # noqa: BLE001 - a class whose INPUT_TYPES fails is simply not followed by type
        spec = None
    if not isinstance(spec, dict):
        return []
    typed, wild = [], []
    for sec in ("required", "optional"):
        for name, s in (spec.get(sec) or {}).items():
            t = s[0] if isinstance(s, (list, tuple)) and s else None
            if not isinstance(t, str):
                continue
            if kind in [x.strip() for x in t.split(",")]:
                typed.append(name)
            elif t.strip() == "*":
                wild.append(name)
    return typed + wild


def trace_loader(prompt: dict, start_id: Any, input_name: str, kind: str = "IMAGE") -> tuple[str | None, str]:
    """(Load Video id, "") reached from `start_id`'s `input_name` through each node's FIRST linked input of `kind`,
    or (None, why). The API prompt is flat (subgraph nodes are "outer:inner" ids), so subgraphs need nothing extra."""
    if not isinstance(prompt, dict):
        return None, "no prompt"
    node = prompt.get(str(start_id)) or {}
    link = (node.get("inputs") or {}).get(input_name)
    seen: set = set()
    for _ in range(MAX_HOPS):
        if not _is_link(link):
            return None, f"{input_name} is not wired to a Load Video"
        src, slot = str(link[0]), int(link[1])
        if src in seen:
            return None, "the graph loops"
        seen.add(src)
        n = prompt.get(src)
        if not isinstance(n, dict):
            return None, f"node {src} is not in the prompt"
        cls = str(n.get("class_type") or "")
        if cls in LOADER_CLASSES:
            return (src, "") if slot == 0 else (None, f"Load Video #{src} output {slot} is not its frames")
        ins = n.get("inputs") or {}
        nxt = None
        for name in _typed_input_names(cls, kind):
            if _is_link(ins.get(name)):
                nxt = ins[name]
                break
        if nxt is None:
            return None, f"no Load Video upstream (stopped at {cls} #{src})"
        link = nxt
    return None, "the chain to a Load Video is too long"


FOLLOW_AUTO = "auto"


def upstream_loads(prompt: dict, start_id: Any, input_name: str, kind: str = "IMAGE") -> list[str]:
    """Every Load Video whose frames reach `start_id`'s `input_name` through any input of `kind` (or a wildcard), in
    first-input-first order. Bounded by MAX_HOPS nodes."""
    if not isinstance(prompt, dict):
        return []
    first = ((prompt.get(str(start_id)) or {}).get("inputs") or {}).get(input_name)
    stack, seen, found = [first], set(), []
    while stack and len(seen) < MAX_HOPS:
        link = stack.pop()
        if not _is_link(link):
            continue
        src = str(link[0])
        if src in seen:
            continue
        seen.add(src)
        n = prompt.get(src)
        if not isinstance(n, dict):
            continue
        cls = str(n.get("class_type") or "")
        if cls in LOADER_CLASSES:
            if int(link[1]) == 0 and src not in found:
                found.append(src)
            continue
        ins = n.get("inputs") or {}
        links = [ins[name] for name in _typed_input_names(cls, kind) if _is_link(ins.get(name))]
        stack.extend(reversed(links))                       # the first input is explored first
    return found


def resolve_follow(prompt: Any, unique_id: Any, input_name: str = "images", kind: str = "IMAGE",
                   follow: Any = FOLLOW_AUTO) -> dict[str, Any]:
    """L9.55 (owner 2026-10-25): which Load Video a Save follows. The primary one is reached through each node's FIRST
    image input (never execution order). Other Loads upstream make it ambiguous: a warning until the user picks one
    (`follow_load` = its node id); a pick that is no longer upstream falls back to auto, with a warning.
    -> {load, candidates, ambiguous, chosen, note, warning}."""
    primary, why = trace_loader(prompt, unique_id, input_name, kind)
    cands = upstream_loads(prompt, unique_id, input_name, kind)
    if primary is not None and primary not in cands:
        cands.insert(0, primary)
    out: dict[str, Any] = {"load": None, "candidates": cands, "ambiguous": False, "chosen": False, "note": why,
                           "warning": None}
    want = str(follow or FOLLOW_AUTO).strip()
    if want and want != FOLLOW_AUTO:
        if want in cands:
            out.update(load=want, chosen=True, note=f"Load Video #{want} (chosen in follow_load)")
            return out
        out["warning"] = (f"follow_load names Load Video #{want}, which is not upstream of this Save any more - "
                          "following the automatic choice")
    others = [c for c in cands if c != primary]
    if primary is not None:
        out.update(load=primary, note="")
        if others:
            out["ambiguous"] = True
            out["warning"] = out["warning"] or (
                f"Load Video #{primary} is followed (first image input); "
                + ", ".join(f"#{c}" for c in others) + " also upstream - pick one in follow_load to confirm")
    elif len(cands) == 1:
        out.update(load=cands[0], note=f"Load Video #{cands[0]} (the only one upstream)")
    elif len(cands) > 1:
        out["ambiguous"] = True
        out["warning"] = out["warning"] or (
            "several Load Videos are upstream (" + ", ".join(f"#{c}" for c in cands)
            + ") and none is on the first-image path - pick one in follow_load")
    return out


def traced_video_info(prompt: Any, unique_id: Any, input_name: str = "images",
                      kind: str = "IMAGE", follow: Any = FOLLOW_AUTO,
                      link: dict | None = None) -> tuple[dict | None, str]:
    """(the video_info the upstream Load Video produced, note). None + why when there is no usable record. `link`, when
    given, receives resolve_follow's answer (candidates / ambiguity) for the node's report."""
    res = resolve_follow(prompt, unique_id, input_name, kind, follow)
    if link is not None:
        link.update(res)
    src, why = res["load"], res["note"]
    if src is None:
        return None, why
    got = _LOADS.get(src)
    sig = inputs_signature((prompt.get(src) or {}).get("inputs") or {})
    if got is None or got[0] != sig:
        return None, (f"Load Video #{src} is upstream, but it has not run with these settings since ComfyUI started "
                      f"- wire its video_info, or trim manually")
    return dict(got[1]), (res["note"] if res["chosen"] else f"Load Video #{src} (traced - no video_info wire)")
