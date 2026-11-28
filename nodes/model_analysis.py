"""
Model-analysis node (C2C): VAE Inspect (VAEBlockInspectorMEC), two modes -
  - inspect: per-block weight statistics and outlier tensors for one VAE;
  - compare: tensor-by-tensor cosine similarity between two VAEs (or any two
    state-dict-bearing model objects). This was the VAE Similarity Analyser
    node, merged in L7.65 P27; its saved workflows migrate to mode "compare".

Diagnostic only; it never modifies the input models. It works on whatever
``state_dict()`` the wrapped object exposes; a plain ``dict`` of tensors works too.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

import torch
try:                                                       # A28 layout declarations (display only)
    from ._layout_decl import with_layout
except ImportError:  # loaded as a plain file (tests): the same module by path, never an absolute import
    try:
        import importlib.util as _c2c_ilu
        import os as _c2c_os
        _c2c_spec = _c2c_ilu.spec_from_file_location("c2c_layout_by_path", _c2c_os.path.join(
            _c2c_os.path.dirname(_c2c_os.path.abspath(__file__)), '_layout_decl.py'))
        _c2c_mod = _c2c_ilu.module_from_spec(_c2c_spec)
        _c2c_spec.loader.exec_module(_c2c_mod)
        with_layout = _c2c_mod.with_layout
    except Exception:  # pragma: no cover - layout keys are display only: never take the node down
        def with_layout(spec, layout):
            return spec

logger = logging.getLogger("MEC.ModelAnalysis")

try:                                                       # L9.174: heatmap + benchmark helpers
    from . import _vae_bench as _VB
except ImportError:  # loaded as a plain file (tests): the same module by path
    import importlib.util as _vb_ilu
    import os as _vb_os
    _vb_spec = _vb_ilu.spec_from_file_location("c2c_vae_bench_by_path", _vb_os.path.join(
        _vb_os.path.dirname(_vb_os.path.abspath(__file__)), "_vae_bench.py"))
    _VB = _vb_ilu.module_from_spec(_vb_spec)
    _vb_spec.loader.exec_module(_VB)


_BLOCK_RE = re.compile(
    r"^(encoder|decoder|quant_conv|post_quant_conv)"
    r"(?:\.(?:down_blocks|up_blocks|mid_block|conv_in|conv_out|norm_out))?",
)


def _to_state_dict(obj: Any) -> dict[str, torch.Tensor]:
    if obj is None:
        raise ValueError("VAE input is None.")
    if isinstance(obj, dict):
        return {k: v for k, v in obj.items() if isinstance(v, torch.Tensor)}
    sd_fn = getattr(obj, "state_dict", None)
    if callable(sd_fn):
        sd = sd_fn()
        return {k: v for k, v in sd.items() if isinstance(v, torch.Tensor)}
    inner = getattr(obj, "model", None) or getattr(obj, "first_stage_model", None)
    if inner is not None and hasattr(inner, "state_dict"):
        return {k: v for k, v in inner.state_dict().items() if isinstance(v, torch.Tensor)}
    raise ValueError(f"Cannot extract state_dict from {type(obj).__name__}")


def _block_of(key: str) -> str:
    m = _BLOCK_RE.match(key)
    if not m:
        return "other"
    parts = key.split(".")
    # audit 2026-11-27: encoder.down.0 / .1 / .2 (and up_blocks.N, Wan's downsamples.N...) were one bucket each
    # ("encoder.down"), so a divergence in one resolution level hid among the others. Keep the level index.
    if len(parts) >= 3 and parts[2].isdigit():
        return ".".join(parts[:3])
    if len(parts) >= 2:
        return ".".join(parts[:2])
    return parts[0]


def _compare(vae_a, vae_b, include_per_tensor: bool = False) -> tuple[str, float, str]:
    """Cosine similarity between two VAE state dicts, per tensor & per block: (report_json, global_cosine,
    most_divergent_blocks_json). Was VAESimilarityAnalyserMEC.analyse (merged into VAE Inspect, L7.65 P27)."""
    sd_a = _to_state_dict(vae_a)
    sd_b = _to_state_dict(vae_b)
    common = sorted(set(sd_a) & set(sd_b))
    only_a = sorted(set(sd_a) - set(sd_b))
    only_b = sorted(set(sd_b) - set(sd_a))
    per_tensor: list[dict] = []
    block_dot: dict[str, float] = {}
    block_norm_a: dict[str, float] = {}
    block_norm_b: dict[str, float] = {}
    global_dot = 0.0
    global_na = 0.0
    global_nb = 0.0
    for k in common:
        ta = sd_a[k].detach().to(torch.float32).flatten()
        tb = sd_b[k].detach().to(torch.float32).flatten()
        if ta.shape != tb.shape:
            continue
        dot = float((ta * tb).sum().item())
        na = float((ta * ta).sum().item())
        nb = float((tb * tb).sum().item())
        global_dot += dot
        global_na += na
        global_nb += nb
        blk = _block_of(k)
        block_dot[blk] = block_dot.get(blk, 0.0) + dot
        block_norm_a[blk] = block_norm_a.get(blk, 0.0) + na
        block_norm_b[blk] = block_norm_b.get(blk, 0.0) + nb
        if include_per_tensor:
            cos = dot / max((na ** 0.5) * (nb ** 0.5), 1e-12)
            per_tensor.append({"key": k, "cos": cos, "shape": list(ta.shape)})
    block_cos = {
        blk: block_dot[blk] / max((block_norm_a[blk] ** 0.5) * (block_norm_b[blk] ** 0.5), 1e-12)
        for blk in block_dot
    }
    global_cos = global_dot / max((global_na ** 0.5) * (global_nb ** 0.5), 1e-12)
    # MANUAL bug-fix (Apr 2026): expose the lowest-cosine blocks for
    # quick triage; ascending cosine == most divergent first.
    divergent_sorted = sorted(block_cos.items(), key=lambda kv: kv[1])
    most_divergent_blocks = [
        {"block": blk, "cosine": cos} for blk, cos in divergent_sorted[:10]
    ]
    report = {
        "global_cosine": global_cos,
        "common_tensors": len(common),
        "only_in_a": only_a[:50],
        "only_in_b": only_b[:50],
        "per_block_cosine": block_cos,
        "most_divergent_blocks": most_divergent_blocks,
    }
    if include_per_tensor:
        report["per_tensor"] = per_tensor
    return (
        json.dumps(report, indent=2),
        float(global_cos),
        json.dumps(most_divergent_blocks, indent=2),
    )


# A28 (docs/audit/A28/PARAM_CNP_B.md, VAEBlockInspectorMEC): each mode's own control only. Display only. Replaces the
# c2c_mode_widgets.js rule (one visibility owner per node).
_INSPECT_LAYOUT = {
    "anomaly_threshold": {"c2c_show_if": {"mode": ["inspect"]}},
    "include_per_tensor": {"c2c_show_if": {"mode": ["compare"]}},
    "benchmark_resolutions": {"c2c_show_if": {"benchmark": [True]}},
}


def _inspect_heatmap(report: dict):
    """Inspect: one row per block, columns mean / std / abs_mean / outliers, each normalised over the blocks."""
    blocks = report.get("blocks", {})
    names = sorted(blocks)
    out_count = {}
    for o in report.get("outliers_detail", []):
        out_count[o.get("block")] = out_count.get(o.get("block"), 0) + 1
    cols = ["mean", "std", "abs_mean", "outliers"]
    raw = [[blocks[n]["mean"], blocks[n]["std"], blocks[n]["abs_mean"], float(out_count.get(n, 0))] for n in names]
    norm = []
    for j in range(len(cols)):
        col = [r[j] for r in raw]
        lo, hi = (min(col), max(col)) if col else (0.0, 0.0)
        norm.append([(v - lo) / (hi - lo) if hi > lo else 0.0 for v in col])
    values = [[norm[j][i] for j in range(len(cols))] for i in range(len(names))]
    return _VB.render_grid(names, ["mn", "sd", "abs", "out"], values, marks=[out_count.get(n, 0) > 0 for n in names])


def _compare_heatmap(report_json: str):
    """Compare: one row per block, cell = 1 - cosine (red = divergent)."""
    rep = json.loads(report_json)
    cos = rep.get("per_block_cosine", {})
    names = sorted(cos)
    red = lambda v: (0.15 + 0.80 * v, 0.55 * (1.0 - v) + 0.10, 0.20 * (1.0 - v) + 0.10)   # noqa: E731
    values = [[max(0.0, min(1.0, 1.0 - float(cos[n])))] for n in names]
    return _VB.render_grid(names, ["1-c"], values, colour=red)


def _benchmark(vae, vae_b, mode: str, resolutions: str) -> str:
    res = _VB.parse_resolutions(resolutions)
    out = {"resolutions": res, "image_types": list(_VB.IMAGE_TYPES),
           "not_included": ["precision modes (would change the dtype of the connected VAE)"],
           "vae": {"reconstruction": _VB.reconstruction_suite(vae, res), "latents": _VB.latent_suite(vae, res[0])}}
    if mode == "compare":
        out["vae_b"] = {"reconstruction": _VB.reconstruction_suite(vae_b, res), "latents": _VB.latent_suite(vae_b, res[0])}
        delta = []
        for ca, cb in zip(out["vae"]["reconstruction"]["cases"], out["vae_b"]["reconstruction"]["cases"]):
            pa, pb = ca.get("psnr_db"), cb.get("psnr_db")
            delta.append({"type": ca["type"], "resolution": ca["resolution"],
                          "psnr_delta_db_b_minus_a": None if pa is None or pb is None else round(pb - pa, 3)})
        out["psnr_delta"] = delta
    return json.dumps(out, indent=2)


class VAEBlockInspectorMEC:
    """VAE Inspect: per-block parameter statistics of one VAE, or how similar two VAEs are."""

    MODES = ("inspect", "compare")

    @classmethod
    def INPUT_TYPES(cls):
        return with_layout({
            "required": {"vae": ("VAE", {"tooltip": "VAE to inspect (compare mode: the first VAE)."})},
            "optional": {
                # Phase 3b v2 - Feature A: anomaly scoring vs reference dist.
                "anomaly_threshold": ("FLOAT", {
                    "default": 5.0, "min": 1.5, "max": 50.0, "step": 0.5,
                    "tooltip": (
                        "Inspect: tensors whose abs_mean exceeds this multiple of "
                        "the cohort median are flagged as magnitude outliers. "
                        "Lower => more sensitive (more flags)."
                    ),
                }),
                # appended after anomaly_threshold so saved Block Inspector values keep their positions
                "mode": (list(cls.MODES), {"default": "inspect",
                         "tooltip": "inspect = weight statistics of one VAE; compare = cosine similarity to vae_b."}),
                "vae_b": ("VAE", {"tooltip": "Compare: the second VAE."}),
                "include_per_tensor": ("BOOLEAN", {"default": False,
                                       "tooltip": "Compare: include per-tensor cosine entries in the JSON report "
                                                  "(verbose)."}),
                # L9.174 (A25), appended: defaults = the behaviour of every saved workflow
                "benchmark": ("BOOLEAN", {"default": False, "tooltip": "Run the reconstruction benchmark: five "
                              "fixed synthetic images (flat grey, gradient, 2 px checker, colour bars, noise) at each "
                              "benchmark resolution through encode -> decode, plus decodes of random latents; PSNR, "
                              "max error, colour shift and clipping per case (compare: both VAEs and the PSNR "
                              "difference). Runs the VAE: seconds on CPU per case. Result in benchmark_json."}),
                "benchmark_resolutions": ("STRING", {"default": "64,128", "tooltip": "Benchmark sizes in pixels, "
                                          "comma-separated: multiples of 8 from 32 to 1024, at most 4."}),
            },
        }, _INSPECT_LAYOUT)

    RETURN_TYPES = ("STRING", "STRING", "FLOAT", "FLOAT", "STRING", "IMAGE", "STRING")
    RETURN_NAMES = ("report_json", "outlier_tensor_names", "anomaly_score", "global_cosine", "most_divergent_blocks",
                    "block_heatmap", "benchmark_json")
    OUTPUT_TOOLTIPS = (
        "Inspect: per-block weight statistics plus outlier details. Compare: per-block cosine, missing keys and "
        "optional per-tensor entries. JSON.",
        "Inspect: JSON list of tensor names flagged as outliers.",
        "Inspect: aggregate anomaly score in [0, 1] (higher means more outliers detected).",
        "Compare: global cosine similarity across all common tensors.",
        "Compare: JSON list of the 10 most divergent blocks (lowest cosine first).",
        "Heatmap image. Inspect: one row per block, columns mean / std / abs mean / outlier count (each scaled over "
        "the blocks, dark = low, bright = high), red marker = block with outliers. Compare: 1 - cosine per block "
        "(red = divergent).",
        "benchmark on: reconstruction and latent-decode results as JSON. Empty when off.",
    )
    FUNCTION = "inspect"
    CATEGORY = "C2C/ModelAnalysis"
    DESCRIPTION = ("Inspect a VAE's per-block weight statistics and outlier tensors, or compare two VAEs block by "
                   "block (cosine similarity) - for checking merges and broken checkpoints.")

    def inspect(self, vae, anomaly_threshold: float = 5.0, mode: str = "inspect", vae_b=None,
                include_per_tensor: bool = False, benchmark: bool = False, benchmark_resolutions: str = "64,128"):
        if benchmark:
            _VB.parse_resolutions(benchmark_resolutions)          # fail fast, before any analysis
        if mode == "compare":
            if vae_b is None:
                raise ValueError("VAE Inspect (compare): connect the second VAE to vae_b.")
            report, global_cos, divergent = _compare(vae, vae_b, include_per_tensor)
            bench = _benchmark(vae, vae_b, mode, benchmark_resolutions) if benchmark else ""
            return (report, "[]", 0.0, global_cos, divergent, _compare_heatmap(report), bench)
        if mode != "inspect":
            raise ValueError(f"VAE Inspect: unknown mode {mode!r}; choose inspect or compare")
        res = self._inspect(vae, anomaly_threshold)
        heat = _inspect_heatmap(json.loads(res[0]))
        bench = _benchmark(vae, None, mode, benchmark_resolutions) if benchmark else ""
        return (*res, 0.0, "[]", heat, bench)

    def _inspect(self, vae, anomaly_threshold: float = 5.0):
        sd = _to_state_dict(vae)
        agg: dict[str, dict[str, float]] = {}
        # MANUAL bug-fix (Apr 2026): track per-tensor stats too so we can
        # flag outliers (NaN/Inf, or whose abs_mean is >5x the median).
        per_tensor_abs_mean: dict[str, float] = {}
        cohort_val: dict[str, tuple] = {}
        outliers: list[dict] = []
        for k, v in sd.items():
            blk = _block_of(k)
            t = v.detach().to(torch.float32).flatten()
            n = t.numel()
            if n == 0:
                continue
            # NaN/Inf check first -- always an outlier.
            if not torch.isfinite(t).all().item():
                outliers.append({
                    "name": k,
                    "block": blk,
                    "reason": "non_finite",
                    "shape": list(v.shape),
                })
                continue
            t_abs_mean = float(t.abs().mean().item())
            per_tensor_abs_mean[k] = t_abs_mean
            # L9.264 (CNP-NODES-2 #18): compare like with like - a tensor's cohort is its name ending and rank, and a
            # weight's magnitude is scaled by sqrt(fan-in) so wide and narrow layers compare (measured on core's SD
            # VAE layout with torch's init: the model-wide median flagged 60 healthy tensors, 52 of them norm layers,
            # score 0.242 - a real x40 conv weight was buried among them; per cohort: 0 flagged, and that one found)
            src = v if v.ndim >= 2 else sd.get(k.rsplit(".", 1)[0] + ".weight")   # a bias: its layer's weight
            fan_in = int(torch.tensor(src.shape[1:]).prod().item()) if src is not None and src.ndim >= 2 else 1
            cohort_val[k] = (f"{k.rsplit('.', 1)[-1]}/{v.ndim}d", t_abs_mean * (fan_in ** 0.5))
            stat = agg.setdefault(blk, {"sum": 0.0, "sumsq": 0.0, "abssum": 0.0, "count": 0})
            stat["sum"] += float(t.sum().item())
            stat["sumsq"] += float((t * t).sum().item())
            stat["abssum"] += float(t.abs().sum().item())
            stat["count"] += int(n)
        # Magnitude outliers: tensors whose (fan-in scaled) abs_mean is >Nx the median of their own cohort (same name
        # ending and rank). Cohorts of fewer than 4 tensors have no meaningful median and are not judged.
        cohorts: dict[str, list] = {}
        for k, (kind, val) in cohort_val.items():
            cohorts.setdefault(kind, []).append(val)
        medians = {kind: sorted(vs)[len(vs) // 2] for kind, vs in cohorts.items() if len(vs) >= 4}
        for k, (kind, val) in cohort_val.items():
            median = medians.get(kind, 0.0)
            if median > 0 and val > median * float(anomaly_threshold):
                outliers.append({
                    "name": k,
                    "block": _block_of(k),
                    "reason": "high_abs_mean",
                    "abs_mean": per_tensor_abs_mean[k],
                    "cohort": kind,
                    "ratio_to_cohort_median": round(val / median, 3),
                })
        out = {}
        for blk, s in agg.items():
            n = max(s["count"], 1)
            mean = s["sum"] / n
            var = max(s["sumsq"] / n - mean * mean, 0.0)
            out[blk] = {
                "mean": mean,
                "std": var ** 0.5,
                "abs_mean": s["abssum"] / n,
                "count": s["count"],
            }
        report = {
            "blocks": out,
            "total_tensors": len(sd),
            "outlier_tensor_names": [o["name"] for o in outliers],
            "outliers_detail": outliers,
        }
        # Phase 3b v2 - Feature A: anomaly_score in [0, 1].
        # Combines fraction-of-outlier-tensors with a hard NaN/Inf penalty.
        # 0.0 == healthy; >0.05 ~= concerning; >0.20 ~= probably corrupt.
        total_inspected = max(len(sd), 1)
        non_finite_count = sum(1 for o in outliers if o.get("reason") == "non_finite")
        magnitude_count = len(outliers) - non_finite_count
        anomaly_score = min(
            1.0,
            (magnitude_count / total_inspected)
            + (non_finite_count / total_inspected) * 5.0,
        )
        report["anomaly_score"] = anomaly_score
        report["anomaly_threshold"] = float(anomaly_threshold)
        return (
            json.dumps(report, indent=2),
            json.dumps([o["name"] for o in outliers], indent=2),
            float(anomaly_score),
        )


NODE_CLASS_MAPPINGS = {
    "VAEBlockInspectorMEC": VAEBlockInspectorMEC,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "VAEBlockInspectorMEC": "VAE Inspect",
}
