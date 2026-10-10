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

logger = logging.getLogger("MEC.ModelAnalysis")


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


class VAEBlockInspectorMEC:
    """VAE Inspect: per-block parameter statistics of one VAE, or how similar two VAEs are."""

    MODES = ("inspect", "compare")

    @classmethod
    def INPUT_TYPES(cls):
        return {
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
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "FLOAT", "FLOAT", "STRING")
    RETURN_NAMES = ("report_json", "outlier_tensor_names", "anomaly_score", "global_cosine", "most_divergent_blocks")
    OUTPUT_TOOLTIPS = (
        "Inspect: per-block weight statistics plus outlier details. Compare: per-block cosine, missing keys and "
        "optional per-tensor entries. JSON.",
        "Inspect: JSON list of tensor names flagged as outliers.",
        "Inspect: aggregate anomaly score in [0, 1] (higher means more outliers detected).",
        "Compare: global cosine similarity across all common tensors.",
        "Compare: JSON list of the 10 most divergent blocks (lowest cosine first).",
    )
    FUNCTION = "inspect"
    CATEGORY = "C2C/ModelAnalysis"
    DESCRIPTION = ("Inspect a VAE's per-block weight statistics and outlier tensors, or compare two VAEs block by "
                   "block (cosine similarity) - for checking merges and broken checkpoints.")

    def inspect(self, vae, anomaly_threshold: float = 5.0, mode: str = "inspect", vae_b=None,
                include_per_tensor: bool = False):
        if mode == "compare":
            if vae_b is None:
                raise ValueError("VAE Inspect (compare): connect the second VAE to vae_b.")
            report, global_cos, divergent = _compare(vae, vae_b, include_per_tensor)
            return (report, "[]", 0.0, global_cos, divergent)
        if mode != "inspect":
            raise ValueError(f"VAE Inspect: unknown mode {mode!r}; choose inspect or compare")
        return (*self._inspect(vae, anomaly_threshold), 0.0, "[]")

    def _inspect(self, vae, anomaly_threshold: float = 5.0):
        sd = _to_state_dict(vae)
        agg: dict[str, dict[str, float]] = {}
        # MANUAL bug-fix (Apr 2026): track per-tensor stats too so we can
        # flag outliers (NaN/Inf, or whose abs_mean is >5x the median).
        per_tensor_abs_mean: dict[str, float] = {}
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
            stat = agg.setdefault(blk, {"sum": 0.0, "sumsq": 0.0, "abssum": 0.0, "count": 0})
            stat["sum"] += float(t.sum().item())
            stat["sumsq"] += float((t * t).sum().item())
            stat["abssum"] += float(t.abs().sum().item())
            stat["count"] += int(n)
        # Magnitude outliers: tensors whose abs_mean is >Nx the median
        # across the model. Cheap heuristic, surfaces corrupted layers.
        if per_tensor_abs_mean:
            vals = sorted(per_tensor_abs_mean.values())
            median = vals[len(vals) // 2] if vals else 0.0
            threshold = median * float(anomaly_threshold)
            for k, am in per_tensor_abs_mean.items():
                if median > 0 and am > threshold:
                    outliers.append({
                        "name": k,
                        "block": _block_of(k),
                        "reason": "high_abs_mean",
                        "abs_mean": am,
                        "median": median,
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
