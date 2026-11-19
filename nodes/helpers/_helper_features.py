"""L9.170 (A25 Part 5, 5.4 General utilities): the pure logic behind the features appended to the C2C Helpers nodes
(Size, Conditional Switch, Execution Timer, Batch Range, Probe, Number Lerp, Seed List, Text Template).

Kept apart from helpers.py so every piece can be unit-tested without ComfyUI. No function here evaluates user code:
templates, comparisons and range lists are parsed by small hand-written parsers.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
from typing import Any

import torch

# ─────────────────────────────── Size: model profiles, named profiles ───────────────────────────────

# Behaviour idea (no code copied) from ComfyUI-AusBoss (MIT) Latent Size: per-model-family size constraints. The
# multiples are the architectures' own pixel-to-token factors (VAE downscale x patch size), not tuning values.
MODEL_PROFILE_NONE = "none (use multiple)"
MODEL_PROFILES = {
    MODEL_PROFILE_NONE: None,
    "SD 1.5 / SDXL (x64)": 64,                      # VAE /8; x64 is the size these UNets were trained on
    "SD3 / Flux / Qwen Image (x16)": 16,            # VAE /8 x patch 2
    "Wan 2.1 / 2.2, HunyuanVideo (x16)": 16,        # VAE /8 x patch 2
    "LTX-Video (x32)": 32,                          # VAE /32
}

_PROFILE_LINE = re.compile(r"^\s*(?P<name>[^=#]+?)\s*=\s*(?P<a>\d+(?:\.\d+)?)\s*(?P<sep>[x×:])\s*(?P<b>\d+(?:\.\d+)?)\s*$",
                           re.IGNORECASE)


def parse_size_profiles(text: str) -> dict[str, tuple[str, float, float]]:
    """'Name = 1920x1080' (exact size) or 'Name = 2.39:1' (ratio scaled to the long edge), one per line;
    '#' starts a comment. Returns {name: (kind, a, b)} with kind 'size' or 'ratio'."""
    out: dict[str, tuple[str, float, float]] = {}
    for n, raw in enumerate(str(text or "").splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        m = _PROFILE_LINE.match(line)
        if not m:
            raise ValueError(f"[Size] custom_profiles line {n} {raw.strip()!r}: write 'Name = 1920x1080' or "
                             f"'Name = 2.39:1'.")
        a, b = float(m["a"]), float(m["b"])
        if a <= 0 or b <= 0:
            raise ValueError(f"[Size] custom_profiles line {n}: sizes and ratios must be above 0.")
        out[m["name"].strip()] = ("ratio" if m["sep"] == ":" else "size", a, b)
    return out


# ─────────────────────────────── Conditional Switch: condition builder ──────────────────────────────

COMPARE_OPERATORS = ("==", "!=", "<", "<=", ">", ">=", "contains", "starts with", "ends with", "matches regex",
                     "is empty", "is not empty")
COMPARE_AS = ("auto", "number", "text", "boolean")
_TRUE_WORDS = {"1", "true", "yes", "on", "y", "t"}
_FALSE_WORDS = {"0", "false", "no", "off", "n", "f", ""}


def _is_empty(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, torch.Tensor):
        return v.numel() == 0
    if isinstance(v, (str, list, tuple, dict)):
        return len(v) == 0 if not isinstance(v, str) else v.strip() == ""
    return False


def _to_bool(v: Any, side: str) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return v != 0
    s = str(v).strip().lower()
    if s in _TRUE_WORDS:
        return True
    if s in _FALSE_WORDS:
        return False
    raise ValueError(f"Conditional Switch: {side} {v!r} is not a boolean (use true/false, yes/no, on/off, 1/0).")


def _to_number(v: Any, side: str) -> float:
    if isinstance(v, bool):
        return float(v)
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, torch.Tensor) and v.numel() == 1:
        return float(v.item())
    try:
        return float(str(v).strip())
    except ValueError:
        raise ValueError(f"Conditional Switch: {side} {v!r} is not a number; set compare_as to 'text' or wire a "
                         f"number.") from None


def evaluate_comparison(a: Any, operator: str, b: str, compare_as: str = "auto") -> tuple[bool, str]:
    """Typed comparison a <operator> b. Returns (result, a one-line explanation)."""
    if operator not in COMPARE_OPERATORS:
        raise ValueError(f"Conditional Switch: unknown operator {operator!r}; choose one of "
                         f"{', '.join(COMPARE_OPERATORS)}.")
    if operator in ("is empty", "is not empty"):
        r = _is_empty(a) == (operator == "is empty")
        return r, f"compare_a {operator} -> {r}"
    if a is None:
        raise ValueError(f"Conditional Switch: source 'compare' with operator {operator!r} needs compare_a wired.")
    kind = compare_as
    if kind == "auto":
        kind = ("boolean" if isinstance(a, bool) else
                "number" if isinstance(a, (int, float)) or (isinstance(a, torch.Tensor) and a.numel() == 1)
                else "text")
    if operator in ("contains", "starts with", "ends with", "matches regex"):
        kind = "text"
    if kind == "boolean":
        x, y = _to_bool(a, "compare_a"), _to_bool(b, "compare_b")
    elif kind == "number":
        x, y = _to_number(a, "compare_a"), _to_number(b, "compare_b")
    elif kind == "text":
        x, y = str(a), str(b)
    else:
        raise ValueError(f"Conditional Switch: unknown compare_as {compare_as!r}; choose one of {', '.join(COMPARE_AS)}.")
    if operator in ("<", "<=", ">", ">=") and kind == "boolean":
        raise ValueError(f"Conditional Switch: {operator!r} needs numbers or text, not booleans.")
    if operator == "==":
        r = x == y
    elif operator == "!=":
        r = x != y
    elif operator == "<":
        r = x < y
    elif operator == "<=":
        r = x <= y
    elif operator == ">":
        r = x > y
    elif operator == ">=":
        r = x >= y
    elif operator == "contains":
        r = y in x
    elif operator == "starts with":
        r = x.startswith(y)
    elif operator == "ends with":
        r = x.endswith(y)
    else:  # matches regex
        if len(y) > 256:
            raise ValueError("Conditional Switch: the regex in compare_b is longer than 256 characters.")
        try:
            r = re.search(y, x) is not None
        except re.error as exc:
            raise ValueError(f"Conditional Switch: compare_b is not a valid regex: {exc}") from None
    shown = x if kind != "text" else (x if len(x) <= 40 else x[:37] + "...")
    return bool(r), f"({kind}) {shown!r} {operator} {y!r} -> {bool(r)}"


# ─────────────────────────────── Execution Timer: history, stats, memory ────────────────────────────

def p95(values: list[float]) -> float:
    """Nearest-rank 95th percentile."""
    s = sorted(values)
    return float(s[max(0, math.ceil(0.95 * len(s)) - 1)]) if s else 0.0


def stage_stats(values: list[float]) -> dict:
    if not values:
        return {"count": 0}
    return {"count": len(values), "last": round(values[-1], 6), "median": round(statistics.median(values), 6),
            "p95": round(p95(values), 6), "min": round(min(values), 6), "max": round(max(values), 6),
            "total": round(sum(values), 6)}


def stage_tree(history: dict[str, list[float]]) -> dict:
    """Group stages by their '/'-separated label path; each level sums its children's last and total seconds."""
    root: dict = {}
    for stage, vals in history.items():
        if not vals:
            continue
        node = root
        for part in [p for p in stage.split("/") if p] or [stage]:
            node = node.setdefault(part, {"last": 0.0, "total": 0.0, "children": {}})
            node["last"] = round(node["last"] + vals[-1], 6)
            node["total"] = round(node["total"] + sum(vals), 6)
            node = node["children"]
    return root


def memory_snapshot() -> dict:
    out: dict = {}
    try:
        import psutil
        out["ram_rss_mb"] = round(psutil.Process().memory_info().rss / 2**20, 1)
    except Exception:  # psutil is optional: report what is available
        out["ram_rss_mb"] = None
    try:
        if torch.cuda.is_available():
            out["vram_allocated_mb"] = round(torch.cuda.memory_allocated() / 2**20, 1)
            out["vram_peak_mb"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
    except Exception:
        pass
    return out


# ─────────────────────────────── Batch Range: range lists, timecode, resample ───────────────────────

_TC = re.compile(r"^(?:(\d+):)?(\d+):(\d+)(?:[:;.](\d+))?$")   # [HH:]MM:SS[:FF]


def _parse_position(tok: str, fps: float, total: int, what: str) -> int:
    """One frame position: an integer frame (negative counts from the end), '<seconds>s', or a [HH:]MM:SS[:FF]
    timecode. Times need fps > 0."""
    t = tok.strip().lower()
    if re.fullmatch(r"-?\d+", t):
        v = int(t)
        return total + v if v < 0 else v
    if t.endswith("s") and re.fullmatch(r"\d+(?:\.\d+)?s", t):
        if fps <= 0:
            raise ValueError(f"Batch Range: {what} {tok!r} is in seconds; set fps above 0.")
        return int(math.floor(float(t[:-1]) * fps + 1e-6))
    m = _TC.match(t)
    if m:
        if fps <= 0:
            raise ValueError(f"Batch Range: {what} {tok!r} is a timecode; set fps above 0.")
        hh, mm, ss, ff = (int(x) if x else 0 for x in m.groups())
        return int(math.floor((hh * 3600 + mm * 60 + ss) * fps + 1e-6)) + ff
    raise ValueError(f"Batch Range: cannot read {what} {tok!r}: use a frame number, seconds like '1.5s' or a "
                     f"timecode like '00:01:12'.")


def parse_frame_ranges(text: str, total: int, fps: float = 0.0) -> list[list[int]]:
    """'0..9, 20..29/2, -5.., 1.5s..3s, 00:00:02:00..00:00:03:00, 42' -> one list of source indices per range.
    Ends are INCLUSIVE; an open end runs to the last frame; '/N' keeps every Nth frame. 'a-b' with plain
    non-negative frame numbers is also accepted. Out-of-batch ranges are an error (never silently empty)."""
    tokens = [t.strip() for t in re.split(r"[,;\n]", str(text or "")) if t.strip()]
    if not tokens:
        raise ValueError("Batch Range: 'ranges' is empty; write ranges like '0..9, 20..29/2'.")
    out: list[list[int]] = []
    for tok in tokens:
        body, step = tok, 1
        if "/" in tok:
            body, s = tok.rsplit("/", 1)
            if not s.strip().isdigit() or int(s) < 1:
                raise ValueError(f"Batch Range: step in {tok!r} must be a whole number of 1 or more.")
            step = int(s)
        if ".." in body:
            a, b = body.split("..", 1)
        elif re.fullmatch(r"\s*\d+\s*-\s*\d+\s*", body):
            a, b = body.split("-", 1)
        else:
            a = b = body
        s_i = _parse_position(a, fps, total, "start") if a.strip() else 0
        e_i = _parse_position(b, fps, total, "end") if b.strip() else total - 1
        if not (0 <= s_i < total) or not (0 <= e_i < total) or e_i < s_i:
            raise ValueError(f"Batch Range: range {tok!r} gives frames {s_i}..{e_i}, outside the batch's 0..{total - 1}"
                             f" (or start after end).")
        out.append(list(range(s_i, e_i + 1, step)))
    return out


def resample_indices(total: int, src_fps: float, dst_fps: float) -> list[int]:
    """Source frame nearest to each output time when a clip at src_fps is re-sampled to dst_fps (same duration)."""
    if src_fps <= 0 or dst_fps <= 0:
        raise ValueError("Batch Range: 'resample to fps' needs fps and target_fps above 0.")
    n_out = max(1, int(math.floor(total * dst_fps / src_fps + 1e-6)))
    return [min(total - 1, int(math.floor(k * src_fps / dst_fps + 0.5 - 1e-9))) for k in range(n_out)]


# ─────────────────────────────── Probe: per-frame stats, anomalies, scopes ──────────────────────────

_LUMA = (0.2126, 0.7152, 0.0722)


def frame_stats(images: torch.Tensor | None, mask: torch.Tensor | None) -> list[dict]:
    rows: list[dict] = []
    n = images.shape[0] if images is not None else (mask.reshape(-1, *mask.shape[-2:]).shape[0] if mask is not None
                                                     else 0)
    m3 = mask.reshape(-1, *mask.shape[-2:]) if mask is not None else None
    for i in range(n):
        r: dict = {"frame": i}
        if images is not None:
            f = images[i].float()
            finite = torch.isfinite(f)
            r["nan"] = int(torch.isnan(f).sum().item())
            r["inf"] = int(torch.isinf(f).sum().item())
            fz = torch.where(finite, f, torch.zeros_like(f))
            ch = fz.reshape(-1, fz.shape[-1]).mean(0)
            r["channel_mean"] = [round(float(c), 6) for c in ch[:3]]
            r["luma"] = round(float(sum(w * float(ch[k]) for k, w in enumerate(_LUMA) if k < ch.shape[0])), 6) \
                if ch.shape[0] >= 3 else round(float(ch.mean()), 6)
            r["clipped_pct"] = round(float(((fz >= 0.999) | (fz <= 0.001)).float().mean()) * 100.0, 4)
        if m3 is not None and i < m3.shape[0]:
            r["alpha_mean"] = round(float(m3[i].float().mean()), 6)
        rows.append(r)
    return rows


def find_anomalies(rows: list[dict], threshold: float) -> list[dict]:
    """A frame is flagged when it holds NaN/Inf, or when luma, any channel mean, alpha (mask mean) or the clipped
    fraction jumps by more than `threshold` (0..1 scale) from the previous frame."""
    out: list[dict] = []
    for i, r in enumerate(rows):
        why = []
        if r.get("nan") or r.get("inf"):
            why.append(f"NaN={r.get('nan', 0)} Inf={r.get('inf', 0)}")
        if i > 0:
            p = rows[i - 1]
            if "luma" in r and abs(r["luma"] - p["luma"]) > threshold:
                why.append(f"exposure jump {r['luma'] - p['luma']:+.4f}")
            if "channel_mean" in r:
                d = max(abs(a - b) for a, b in zip(r["channel_mean"], p["channel_mean"]))
                if d > threshold:
                    why.append(f"colour jump {d:.4f}")
                dc = abs(r["clipped_pct"] - p["clipped_pct"]) / 100.0
                if dc > threshold:
                    why.append(f"saturation/clipping jump {dc * 100:.2f}%")
            if "alpha_mean" in r and "alpha_mean" in p and abs(r["alpha_mean"] - p["alpha_mean"]) > threshold:
                why.append(f"alpha jump {r['alpha_mean'] - p['alpha_mean']:+.4f}")
        if why:
            out.append({"frame": r["frame"], "why": why})
    return out


SCOPES = ("none", "histogram", "waveform", "rgb parade")
_SCOPE_H = 256


def _limit(images: torch.Tensor, max_frames: int = 16, max_side: int = 512) -> torch.Tensor:
    b, h, w = images.shape[:3]
    fs = max(1, math.ceil(b / max_frames))
    ys, xs = max(1, math.ceil(h / max_side)), max(1, math.ceil(w / max_side))
    return images[::fs, ::ys, ::xs, :3].float()


def _waveform(chan: torch.Tensor, cols: int) -> torch.Tensor:
    """chan [B,H,W] in 0..1 -> [256, cols] log-scaled density (row 0 = value 1.0)."""
    b, h, w = chan.shape
    x = (torch.arange(w) * cols // max(1, w)).view(1, 1, w).expand(b, h, w)
    v = (torch.nan_to_num(chan, 0.0).clamp(0, 1) * (_SCOPE_H - 1)).round().long()
    counts = torch.bincount((v * cols + x).reshape(-1), minlength=_SCOPE_H * cols).float().view(_SCOPE_H, cols)
    d = torch.log1p(counts)
    d = d / d.max().clamp_min(1e-6)
    return d.flip(0)


def render_scope(images: torch.Tensor, kind: str) -> torch.Tensor:
    """Histogram / luma waveform / RGB parade of the whole batch as an IMAGE [1, 256, W, 3]."""
    if kind not in SCOPES:
        raise ValueError(f"Probe: unknown scope {kind!r}; choose one of {', '.join(SCOPES)}.")
    img = _limit(images)
    if img.shape[-1] < 3:
        img = img[..., :1].expand(*img.shape[:-1], 3)
    if kind == "histogram":
        out = torch.zeros(_SCOPE_H, 256, 3)
        for c in range(3):
            v = (torch.nan_to_num(img[..., c], 0.0).clamp(0, 1) * 255).round().long().reshape(-1)
            hist = torch.log1p(torch.bincount(v, minlength=256).float())
            hist = hist / hist.max().clamp_min(1e-6)
            heights = (hist * (_SCOPE_H - 1)).round().long()
            rows = torch.arange(_SCOPE_H).view(_SCOPE_H, 1)
            out[..., c] = (rows >= (_SCOPE_H - 1 - heights).view(1, 256)).float() * 0.85
        return out.unsqueeze(0)
    cols = 256
    if kind == "waveform":
        luma = sum(w * img[..., k] for k, w in enumerate(_LUMA))
        d = _waveform(luma, cols)
        return torch.stack([d * 0.6, d, d * 0.6], -1).unsqueeze(0)
    parts = []
    for c in range(3):
        d = _waveform(img[..., c], cols)
        col = torch.zeros(_SCOPE_H, cols, 3)
        col[..., c] = d
        parts.append(col)
        parts.append(torch.zeros(_SCOPE_H, 4, 3))
    return torch.cat(parts[:-1], 1).unsqueeze(0)


# ─────────────────────────────── Number Lerp: keyframes, time domain ────────────────────────────────

TIME_MODES = ("normalised t", "frame", "seconds")
LOOP_MODES = ("clamp", "loop", "ping-pong")


def ease(t: float, curve: str) -> float:
    if curve == "smoothstep":
        return t * t * (3.0 - 2.0 * t)
    if curve == "cosine":
        return (1.0 - math.cos(t * math.pi)) * 0.5
    if curve == "hold":
        return 0.0 if t < 1.0 else 1.0
    return t


def parse_keyframes(text: str) -> list[tuple[float, float]]:
    """'0:0, 0.5:1, 1:0' -> sorted [(position, value)]. Separators: comma, semicolon or newline."""
    keys = []
    for tok in [t.strip() for t in re.split(r"[,;\n]", str(text or "")) if t.strip()]:
        if ":" not in tok:
            raise ValueError(f"Number Lerp: keyframe {tok!r} must be 'position:value' (e.g. '24:1.5').")
        a, b = tok.split(":", 1)
        try:
            keys.append((float(a), float(b)))
        except ValueError:
            raise ValueError(f"Number Lerp: keyframe {tok!r} must hold two numbers.") from None
    keys.sort(key=lambda k: k[0])
    return keys


def wrap_time(pos: float, period: float, loop: str) -> float:
    if loop not in LOOP_MODES:
        raise ValueError(f"Number Lerp: unknown loop {loop!r}; choose one of {', '.join(LOOP_MODES)}.")
    if period <= 0 or loop == "clamp":
        return pos
    if loop == "loop":
        return pos % period
    k = pos % (2 * period)
    return k if k <= period else 2 * period - k


def eval_keyframes(keys: list[tuple[float, float]], pos: float, curve: str) -> float:
    if not keys:
        raise ValueError("Number Lerp: no keyframes.")
    if pos <= keys[0][0]:
        return keys[0][1]
    if pos >= keys[-1][0]:
        return keys[-1][1]
    for (p0, v0), (p1, v1) in zip(keys, keys[1:]):
        if p0 <= pos < p1:
            return v0 + (v1 - v0) * ease((pos - p0) / (p1 - p0), curve)
    return keys[-1][1]  # pragma: no cover - unreachable


# ─────────────────────────────── Seed List: shot seed maps, manifest ────────────────────────────────

def seeds_for(base_seed: int, count: int, mode: str) -> list[int]:
    """The original Seed List algorithm (unchanged; moved here so shot maps reuse it)."""
    if mode == "increment":
        return [(base_seed + i) & 0xFFFFFFFF for i in range(count)]
    if mode == "hash":
        return [int.from_bytes(hashlib.sha256(f"{base_seed}-{i}".encode()).digest()[:4], "big") for i in range(count)]
    import random
    r = random.Random(base_seed)
    return [r.randint(0, 0xFFFFFFFF) for _ in range(count)]


def shot_base(base_seed: int, shot: str) -> int:
    """A shot's own base seed: depends only on the base seed and the shot's name (never on list order)."""
    return int.from_bytes(hashlib.sha256(f"{base_seed}:shot:{shot}".encode()).digest()[:4], "big")


def parse_shot_ids(text: str) -> list[str]:
    ids = [t.strip() for t in re.split(r"[,;\n]", str(text or "")) if t.strip()]
    dup = sorted({s for s in ids if ids.count(s) > 1})
    if dup:
        raise ValueError(f"Seed List: shot ids must be unique; repeated: {', '.join(dup)}.")
    return ids


def prompt_fingerprint(prompt: Any) -> str | None:
    if not prompt:
        return None
    try:
        return hashlib.sha256(json.dumps(prompt, sort_keys=True, default=str).encode()).hexdigest()
    except Exception:
        return None


# ─────────────────────────────── Text Template: schema, sections, filters ───────────────────────────

_SCHEMA_LINE = re.compile(r"^\s*(?P<name>[A-Za-z_]\w*)\s*:\s*(?P<type>text|int|float|bool|choice\((?P<opts>[^)]*)\))"
                          r"\s*(?:=\s*(?P<default>.*?))?\s*(?P<req>required)?\s*$", re.IGNORECASE)
_VAR_LINE = re.compile(r"^\s*(?P<name>[A-Za-z_]\w*)\s*=\s?(?P<value>.*)$")


def parse_variables(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for n, raw in enumerate(str(text or "").splitlines(), 1):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        m = _VAR_LINE.match(raw)
        if not m:
            raise ValueError(f"Text Template: variables line {n} {raw.strip()!r}: write 'name = value'.")
        out[m["name"]] = m["value"].rstrip()
    return out


def parse_schema(text: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for n, raw in enumerate(str(text or "").splitlines(), 1):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        m = _SCHEMA_LINE.match(raw)
        if not m:
            raise ValueError(f"Text Template: schema line {n} {raw.strip()!r}: write 'name: text|int|float|bool|"
                             f"choice(a|b) [= default] [required]'.")
        t = m["type"].lower()
        opts = [o.strip() for o in m["opts"].split("|")] if m["opts"] is not None else None
        out[m["name"]] = {"type": "choice" if opts is not None else t, "options": opts,
                          "default": m["default"], "required": bool(m["req"])}
    return out


def apply_schema(values: dict[str, str], schema: dict[str, dict]) -> tuple[dict[str, str], list[str]]:
    """Fill defaults, check required fields and types. Returns (values, problems)."""
    vals = dict(values)
    problems: list[str] = []
    for name, rule in schema.items():
        v = vals.get(name, "")
        if (v is None or str(v) == "") and rule["default"] is not None:
            v = rule["default"]
        v = "" if v is None else str(v)
        if v == "":
            if rule["required"]:
                problems.append(f"{name} is required")
            vals[name] = v
            continue
        t = rule["type"]
        try:
            if t == "int":
                v = str(int(v.strip()))
            elif t == "float":
                v = repr(float(v.strip()))
            elif t == "bool":
                s = v.strip().lower()
                if s not in _TRUE_WORDS | _FALSE_WORDS:
                    raise ValueError
                v = "true" if s in _TRUE_WORDS else "false"
            elif t == "choice" and v not in rule["options"]:
                problems.append(f"{name}={v!r} is not one of {rule['options']}")
        except ValueError:
            problems.append(f"{name}={v!r} is not a valid {t}")
        vals[name] = v
    return vals, problems


_SECTION = re.compile(r"\{([?!])([A-Za-z_]\w*)(?:=([^{}]*))?\}((?:(?!\{[?!][A-Za-z_]).)*?)\{/\2\}", re.DOTALL)
_PLACEHOLDER = re.compile(r"\{([A-Za-z_]\w*)((?:\|[^{}|]+)*)\}")
_FILTERS = ("upper", "lower", "title", "capitalize", "strip", "default", "prefix", "suffix", "truncate")


def _truthy(v: str) -> bool:
    return str(v).strip().lower() not in _FALSE_WORDS


def _apply_filters(name: str, v: str, chain: str) -> str:
    for f in [x for x in chain.split("|") if x]:
        fname, _, arg = f.partition(":")
        fname = fname.strip()
        if fname == "upper":
            v = v.upper()
        elif fname == "lower":
            v = v.lower()
        elif fname == "title":
            v = v.title()
        elif fname == "capitalize":
            v = v[:1].upper() + v[1:]
        elif fname == "strip":
            v = v.strip()
        elif fname == "default":
            v = v if v != "" else arg
        elif fname == "prefix":
            v = arg + v if v != "" else v
        elif fname == "suffix":
            v = v + arg if v != "" else v
        elif fname == "truncate":
            if not arg.strip().isdigit():
                raise ValueError(f"Text Template: filter truncate on {{{name}}} needs a number, e.g. truncate:40.")
            v = v[:int(arg)]
        else:
            raise ValueError(f"Text Template: unknown filter {fname!r} on {{{name}}}; filters: {', '.join(_FILTERS)}.")
    return v


def render_extended(template: str, values: dict[str, str]) -> tuple[str, list[str]]:
    """{name|filter:arg}, {?name}..{/name} (kept when name is set and not false/0/no/off), {?name=value}..{/name},
    {!name}..{/name} (kept when name is empty or false); '{{' and '}}' are literal braces. Unknown names are an error.
    Returns (text, notes)."""
    notes: list[str] = []
    s = str(template).replace("{{", "\x00").replace("}}", "\x01")

    def known(name: str) -> str:
        if name not in values:
            raise ValueError(f"Text Template: unknown placeholder {{{name}}}; known: {', '.join(sorted(values))}.")
        return str(values[name])

    def section(m: re.Match) -> str:
        kind, name, want, body = m.group(1), m.group(2), m.group(3), m.group(4)
        v = known(name)
        keep = (v == want) if want is not None else _truthy(v)
        if kind == "!":
            keep = not keep
        notes.append(f"section {kind}{name}{'=' + want if want is not None else ''}: {'kept' if keep else 'dropped'}")
        return body if keep else ""

    for _ in range(64):
        s2 = _SECTION.sub(section, s)
        if s2 == s:
            break
        s = s2
    if re.search(r"\{[?!/][A-Za-z_]", s):
        raise ValueError("Text Template: a {?name} / {!name} section has no matching {/name} (or the names differ).")
    s = _PLACEHOLDER.sub(lambda m: _apply_filters(m.group(1), known(m.group(1)), m.group(2)), s)
    if "{" in s or "}" in s:
        raise ValueError("Text Template: a stray '{' or '}' is left; write '{{' or '}}' for a literal brace.")
    return s.replace("\x00", "{").replace("\x01", "}"), notes
