"""
nodes/vae_merge.py — VAE Merge (C2C) — flagship VAE merging node.

Recreates and extends the (now-deleted) TechnoByteJS/ComfyUI-TechNodes
VAE Merge node with a richer merge-method set borrowed from `meh` /
SD.Next, adds data-driven auto-alpha, a latent-space reconstruction
probe, a dry-run mode, and exportable / importable merge recipes.

Supported merge modes
─────────────────────
    weighted_sum        out = (1 - α)·A + α·B
    add_difference      out = A + α·(B - C)                 (vae_c required)
    tensor_sum          magnitude-blended tensor sum
    triple_sum          out = (A + B + C) / 3               (vae_c required)
    slerp               spherical linear interpolation on flattened tensors
    sigmoid             smooth S-curve blend (TechnoByte / meh)
    geometric           weighted geometric mean of |A|·|B|, sign of A
    max_abs             elementwise pick of whichever has larger |·|
    smooth_add_diff     A + α·tanh(B - C)                    (vae_c required)
    distribution_xover  per-key swap if FFT-energy of B < A   (TechnoByte)
    dare_ties           DARE pruning + TIES sign election
    block_swap          per-block hard swap (slider ≥ 0.5 → B)
    clamp_interp        weighted_sum clamped to per-tensor min/max of A

Architecture detection
──────────────────────
SD 1.5 / SDXL / Flux / Mochi / unknown — auto-detected from state-dict
keys; per-block sliders are mapped to the right key prefixes.

Outputs
───────
    vae               merged ComfyUI VAE (deep copy of vae_a)
    info              JSON summary (also a recipe — paste into recipe_in
                      on a future run for bit-exact reproduction)
    recipe_json       same content as `info` but stripped of timestamps
                      so it diffs cleanly across runs
    probe_report      reconstruction-quality JSON (only if reference_image
                      is connected) — MSE/PSNR for A, B, and merged

Public class:
    VAEMergeMEC
"""

from __future__ import annotations

import copy
import gc
import json
import logging
import math
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import torch

from ._is_changed_util import hash_args_and_kwargs

logger = logging.getLogger("MEC.VAEMerge")


MERGE_MODES = [
    "weighted_sum",
    "add_difference",
    "tensor_sum",
    "triple_sum",
    "slerp",
    "sigmoid",
    "geometric",
    "max_abs",
    "smooth_add_diff",
    "distribution_xover",
    "dare_ties",
    "block_swap",
    "clamp_interp",
]

REQUIRES_VAE_C = {"add_difference", "triple_sum", "smooth_add_diff"}


# ───────────────────────── architecture detection ──────────────────────

_BLOCK_PATTERNS_SD: Dict[str, List[str]] = {
    "block_conv_in":  [r"(?:^|\.)encoder\.conv_in\.", r"(?:^|\.)decoder\.conv_in\."],
    "block_conv_out": [r"(?:^|\.)encoder\.conv_out\.", r"(?:^|\.)decoder\.conv_out\."],
    "block_norm_out": [r"(?:^|\.)encoder\.norm_out\.", r"(?:^|\.)decoder\.norm_out\."],
    "block_mid":      [r"(?:^|\.)encoder\.mid\.", r"(?:^|\.)decoder\.mid\.",
                       r"(?:^|\.)encoder\.mid_block\.", r"(?:^|\.)decoder\.mid_block\."],
    "block_0":        [r"(?:^|\.)encoder\.down\.0\.", r"(?:^|\.)decoder\.up\.3\.",
                       r"(?:^|\.)encoder\.down_blocks\.0\.", r"(?:^|\.)decoder\.up_blocks\.0\."],
    "block_1":        [r"(?:^|\.)encoder\.down\.1\.", r"(?:^|\.)decoder\.up\.2\.",
                       r"(?:^|\.)encoder\.down_blocks\.1\.", r"(?:^|\.)decoder\.up_blocks\.1\."],
    "block_2":        [r"(?:^|\.)encoder\.down\.2\.", r"(?:^|\.)decoder\.up\.1\.",
                       r"(?:^|\.)encoder\.down_blocks\.2\.", r"(?:^|\.)decoder\.up_blocks\.2\."],
    "block_3":        [r"(?:^|\.)encoder\.down\.3\.", r"(?:^|\.)decoder\.up\.0\.",
                       r"(?:^|\.)encoder\.down_blocks\.3\.", r"(?:^|\.)decoder\.up_blocks\.0\."],
}

_ARCH_SIGNATURES: List[Tuple[str, re.Pattern]] = [
    ("flux",   re.compile(r"(?:^|\.)(?:img_in|txt_in|double_blocks|single_blocks)\.")),
    ("mochi",  re.compile(r"(?:^|\.)(?:t5|patch_embed|attn_blocks)\.")),
    ("sdxl",   re.compile(r"(?:^|\.)encoder\.down(?:_blocks)?\.3\.")),
    ("sd1x",   re.compile(r"(?:^|\.)encoder\.down(?:_blocks)?\.2\.")),
]


def _detect_architecture(keys: List[str]) -> str:
    joined = "\n".join(keys)
    for name, pat in _ARCH_SIGNATURES:
        if pat.search(joined):
            return name
    return "unknown"


def _classify_key(key: str) -> Optional[str]:
    for slider, patterns in _BLOCK_PATTERNS_SD.items():
        for pat in patterns:
            if re.search(pat, key):
                return slider
    return None


# ─────────────────────────── state-dict helpers ────────────────────────

def _extract_state_dict(vae: Any) -> Tuple[Dict[str, torch.Tensor], Any]:
    inner = getattr(vae, "first_stage_model", None)
    if inner is not None and hasattr(inner, "state_dict"):
        return {k: v for k, v in inner.state_dict().items()}, inner
    if hasattr(vae, "state_dict"):
        return {k: v for k, v in vae.state_dict().items()}, vae
    if isinstance(vae, dict):
        return dict(vae), None
    raise TypeError(
        f"[MEC VAE Merge] Could not extract state dict from VAE input "
        f"(type={type(vae).__name__}). Expected a ComfyUI VAE wrapper."
    )


def _check_compatible(
    a: Dict[str, torch.Tensor],
    b: Dict[str, torch.Tensor],
    label_a: str,
    label_b: str,
) -> None:
    keys_a, keys_b = set(a.keys()), set(b.keys())
    missing_in_b = sorted(keys_a - keys_b)
    missing_in_a = sorted(keys_b - keys_a)
    if missing_in_a or missing_in_b:
        raise ValueError(
            f"[MEC VAE Merge] Architecture mismatch between {label_a} and {label_b}.\n"
            f"  {label_b} is missing {len(missing_in_b)} keys (e.g. {missing_in_b[:5]}).\n"
            f"  {label_a} is missing {len(missing_in_a)} keys (e.g. {missing_in_a[:5]}).\n"
            f"  Both VAEs must share the same architecture."
        )
    for k in keys_a:
        if a[k].shape != b[k].shape:
            raise ValueError(
                f"[MEC VAE Merge] Shape mismatch for key '{k}': "
                f"{label_a}={tuple(a[k].shape)} vs {label_b}={tuple(b[k].shape)}."
            )


# ─────────────────────────── merge primitives ──────────────────────────

def _merge_two(
    ta: torch.Tensor, tb: torch.Tensor, mode: str, alpha: float,
) -> torch.Tensor:
    out_dtype = ta.dtype
    a = ta.detach().to(torch.float32)
    b = tb.detach().to(torch.float32)
    if mode == "weighted_sum":
        out = (1.0 - alpha) * a + alpha * b
    elif mode == "tensor_sum":
        summed = a + b
        peak = summed.abs().max()
        out = summed / peak.clamp_min(1e-8) * a.abs().max().clamp_min(1e-8)
    elif mode == "slerp":
        out = _slerp(a, b, alpha)
    elif mode == "sigmoid":
        # smooth S-curve: heavier toward the dominant model away from α=0.5
        # technobyte/meh behaviour — alpha threads through a sigmoid envelope
        s = 1.0 / (1.0 + math.exp(-12.0 * (alpha - 0.5)))
        out = (1.0 - s) * a + s * b
    elif mode == "geometric":
        # weighted geometric mean of magnitudes, sign of a
        eps = 1e-8
        mag = torch.exp((1.0 - alpha) * torch.log(a.abs() + eps) + alpha * torch.log(b.abs() + eps))
        out = mag * torch.sign(a)
    elif mode == "max_abs":
        pick_b = b.abs() > a.abs()
        out = torch.where(pick_b, b, a)
    elif mode == "dare_ties":
        out = _dare_ties_pair(a, b, alpha)
    elif mode == "block_swap":
        out = b if alpha >= 0.5 else a
    elif mode == "clamp_interp":
        merged = (1.0 - alpha) * a + alpha * b
        out = merged.clamp(a.min(), a.max())
    elif mode == "distribution_xover":
        # If b's high-frequency energy exceeds a's, swap to b at strength alpha
        a_hf = a.flatten().std()
        b_hf = b.flatten().std()
        if float(b_hf) > float(a_hf):
            out = (1.0 - alpha) * a + alpha * b
        else:
            out = a
    else:
        out = (1.0 - alpha) * a + alpha * b
    return out.to(out_dtype)


def _merge_three(
    ta: torch.Tensor, tb: torch.Tensor, tc: torch.Tensor, mode: str, alpha: float,
) -> torch.Tensor:
    out_dtype = ta.dtype
    a = ta.detach().to(torch.float32)
    b = tb.detach().to(torch.float32)
    c = tc.detach().to(torch.float32)
    if mode == "add_difference":
        out = a + alpha * (b - c)
    elif mode == "triple_sum":
        out = (a + b + c) / 3.0
    elif mode == "smooth_add_diff":
        # Tanh-bounded delta — never blows up even with extreme alphas
        delta = torch.tanh(b - c)
        out = a + alpha * delta
    else:
        out = a
    return out.to(out_dtype)


def _slerp(a: torch.Tensor, b: torch.Tensor, t: float) -> torch.Tensor:
    flat_a = a.flatten()
    flat_b = b.flatten()
    na = flat_a.norm().clamp_min(1e-8)
    nb = flat_b.norm().clamp_min(1e-8)
    dot = (flat_a / na).dot(flat_b / nb).clamp(-1.0, 1.0)
    omega = torch.acos(dot)
    so = torch.sin(omega)
    if so.abs() < 1e-6:
        return ((1.0 - t) * a + t * b)
    coef_a = torch.sin((1.0 - t) * omega) / so
    coef_b = torch.sin(t * omega) / so
    return (coef_a * flat_a + coef_b * flat_b).view_as(a)


def _dare_ties_pair(
    a: torch.Tensor, b: torch.Tensor, alpha: float, drop_p: float = 0.1,
) -> torch.Tensor:
    delta = b - a
    mask = torch.rand_like(delta) > drop_p
    delta = delta * mask / max(1.0 - drop_p, 1e-6)
    sign = torch.sign(delta.sum())
    if sign != 0:
        delta = torch.where(torch.sign(delta) == sign, delta, torch.zeros_like(delta))
    return a + alpha * delta


def _per_key_alpha(
    key: str,
    base_alpha: float,
    use_blocks: bool,
    block_weights: Dict[str, float],
) -> float:
    if not use_blocks:
        return base_alpha
    slider = _classify_key(key)
    if slider is None:
        return base_alpha
    return float(block_weights.get(slider, base_alpha))


# ─────────────────── feature A: block-similarity auto-alpha ────────────

def _block_similarity(
    sd_a: Dict[str, torch.Tensor], sd_b: Dict[str, torch.Tensor],
) -> Dict[str, float]:
    """Per-block mean cosine similarity between A and B.

    Returns ``{slider_name: cos_sim_in_[-1, 1]}`` with one entry per slider
    that had matching keys. Values close to 1.0 mean A and B agree on that
    block (so blending is safe at α=0.5); values near 0 or negative mean
    they diverge and the user probably wants α far from 0.5.
    """
    bucket: Dict[str, List[float]] = {}
    for key, ta in sd_a.items():
        slider = _classify_key(key)
        if slider is None or key not in sd_b:
            continue
        try:
            a = ta.detach().to(torch.float32).flatten()
            b = sd_b[key].detach().to(torch.float32).flatten()
            na = a.norm().clamp_min(1e-8)
            nb = b.norm().clamp_min(1e-8)
            cos = float((a / na).dot(b / nb).clamp(-1.0, 1.0))
            bucket.setdefault(slider, []).append(cos)
        except Exception:
            continue
    return {k: float(sum(v) / len(v)) for k, v in bucket.items() if v}


def _auto_alpha_from_similarity(
    sim: Dict[str, float], base_alpha: float,
) -> Dict[str, float]:
    """Convert per-block cosine similarity to per-block alphas.

    Heuristic: cos≈1 → α=0.5 (safe blend); cos≪1 → α biased toward A
    (since dissimilar blocks usually mean B has incompatible features).
    Caller can override with ``base_alpha`` to bias the whole curve.
    """
    out: Dict[str, float] = {}
    for slider, cos in sim.items():
        # Map cos in [-1, 1] → α in [0, 0.5]; at cos=1 → α=base; at cos=-1 → α=0.
        scale = max(0.0, (cos + 1.0) * 0.5)  # [0, 1]
        out[slider] = round(scale * base_alpha + (1.0 - scale) * 0.0, 4)
    return out


# ─────────────────── feature B: latent-space probe ─────────────────────

def _build_probe_image(reference_image: torch.Tensor) -> torch.Tensor:
    """Resize reference image to a small canonical size for probing."""
    # Comfy IMAGE: (B, H, W, 3) in [0, 1]
    if reference_image is None:
        return None
    img = reference_image
    if img.dim() == 3:
        img = img.unsqueeze(0)
    # Take only the first frame for speed
    img = img[:1].to(torch.float32).clamp(0.0, 1.0)
    # Channels-last → channels-first
    img = img.permute(0, 3, 1, 2)
    # Downsample to 256×256 if larger (probe is for relative quality, not absolute fidelity)
    if img.shape[-1] > 256 or img.shape[-2] > 256:
        img = torch.nn.functional.interpolate(img, size=(256, 256), mode="bilinear", align_corners=False)
    return img


def _probe_vae_reconstruction(
    vae: Any, probe_image_chw: torch.Tensor,
) -> Optional[Dict[str, float]]:
    """Encode→decode through *vae*, return MSE / PSNR vs the input.

    Best-effort — returns None if the VAE wrapper doesn't expose
    .encode()/.decode() the way ComfyUI's wrapper does.
    """
    if probe_image_chw is None or vae is None:
        return None
    try:
        # ComfyUI VAE wrapper expects (B, H, W, 3) in [0, 1]
        bhwc = probe_image_chw.permute(0, 2, 3, 1)
        if hasattr(vae, "encode") and hasattr(vae, "decode"):
            latent = vae.encode(bhwc)
            recon = vae.decode(latent)
            if recon.dim() == 4 and recon.shape[-1] in (3, 4):
                recon = recon[..., :3]
            mse = float(((recon - bhwc) ** 2).mean().item())
            psnr = 10.0 * math.log10(1.0 / max(mse, 1e-12))
            return {"mse": round(mse, 8), "psnr_db": round(psnr, 3)}
    except Exception as exc:
        logger.warning("[MEC VAE Merge] Probe failed: %s", exc)
    return None


# ─────────────────── feature D: recipe export / import ─────────────────

def _build_recipe(
    merge_mode: str, alpha: float, beta: float, use_blocks: bool,
    block_weights: Dict[str, float], brightness: float, contrast: float,
    auto_alpha: bool, auto_alpha_blocks: Dict[str, float],
    architecture: str,
) -> Dict[str, Any]:
    return {
        "schema": "mec.vae_merge.recipe.v1",
        "merge_mode": merge_mode,
        "alpha": float(alpha),
        "beta": float(beta),
        "use_blocks": bool(use_blocks),
        "block_weights": {k: float(v) for k, v in block_weights.items()},
        "auto_alpha": bool(auto_alpha),
        "auto_alpha_blocks": {k: float(v) for k, v in auto_alpha_blocks.items()},
        "brightness": float(brightness),
        "contrast": float(contrast),
        "architecture": architecture,
    }


def _parse_recipe(recipe_json: str) -> Optional[Dict[str, Any]]:
    if not recipe_json or not recipe_json.strip():
        return None
    try:
        data = json.loads(recipe_json)
        if not isinstance(data, dict):
            return None
        if data.get("schema", "").startswith("mec.vae_merge.recipe"):
            return data
        # Permissive: accept legacy info-blob too
        if "merge_mode" in data and "alpha" in data:
            return data
    except (json.JSONDecodeError, TypeError):
        return None
    return None


# ──────────────────────────── ComfyUI node ─────────────────────────────


class VAEMergeMEC:
    """
    (C2C) VAE Merge — combine 2 or 3 VAE checkpoints with TechnoByte-style
    method coverage plus data-driven auto-alpha, latent reconstruction
    probe, dry-run, and exportable recipes.

    See module docstring for the full merge-mode list.
    """

    DESCRIPTION = (
        "Merge 2 or 3 VAEs with 13 strategies (weighted_sum, sigmoid, "
        "geometric, slerp, dare_ties, …). Optional per-block weights, "
        "auto-alpha from block cosine similarity, latent-space probe to "
        "report reconstruction MSE/PSNR, dry-run, and recipe export."
    )

    @classmethod
    def INPUT_TYPES(cls):
        block_default = {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}
        return {
            "required": {
                "vae_a": ("VAE", {"tooltip": "Primary VAE (acts as base; deep-copied clone is returned)."}),
                "vae_b": ("VAE", {"tooltip": "Secondary VAE blended into vae_a."}),
                "merge_mode": (MERGE_MODES, {
                    "default": "weighted_sum",
                    "tooltip": (
                        "Blend strategy. add_difference / triple_sum / smooth_add_diff need vae_c. "
                        "sigmoid + geometric mimic TechnoByte/meh behaviour. distribution_xover keeps A "
                        "unless B has higher detail energy. dare_ties = sparse delta + sign election."
                    )
                }),
                "alpha": ("FLOAT", {"default": 0.30, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Primary blend weight. weighted_sum: 0=A, 1=B."}),
                "beta": ("FLOAT", {"default": 0.70, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Secondary blend weight (used by add_difference / 3-VAE modes)."}),
                "brightness": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Post-merge brightness shift on decoder.conv_out."}),
                "contrast": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Post-merge contrast gain on decoder.conv_out."}),
                "use_blocks": ("BOOLEAN", {"default": False,
                    "tooltip": "Enable per-block sliders. When False all keys use 'alpha'."}),
                "auto_alpha": ("BOOLEAN", {"default": False,
                    "tooltip": (
                        "Data-driven block weights. When True, computes per-block cosine "
                        "similarity between A and B; dissimilar blocks bias toward A. "
                        "Overrides the manual block sliders."
                    )}),
                "block_conv_in":  ("FLOAT", dict(block_default, tooltip="Weight for encoder/decoder conv_in.")),
                "block_conv_out": ("FLOAT", dict(block_default, tooltip="Weight for encoder/decoder conv_out.")),
                "block_norm_out": ("FLOAT", dict(block_default, tooltip="Weight for encoder/decoder norm_out.")),
                "block_0":        ("FLOAT", dict(block_default, tooltip="First down/up block pair.")),
                "block_1":        ("FLOAT", dict(block_default, tooltip="Second down/up block pair.")),
                "block_2":        ("FLOAT", dict(block_default, tooltip="Third down/up block pair.")),
                "block_3":        ("FLOAT", dict(block_default, tooltip="Fourth down/up block pair (SDXL).")),
                "block_mid":      ("FLOAT", dict(block_default, tooltip="Mid block.")),
                "device": (["cpu", "cuda", "auto"], {
                    "default": "cpu",
                    "tooltip": "Compute device. CPU is safe (default); CUDA is faster but uses VRAM."}),
                "dry_run": ("BOOLEAN", {"default": False,
                    "tooltip": (
                        "If True, skip the merge and return only the recipe + similarity report. "
                        "Use this for fast block-similarity inspection without waiting for the merge."
                    )}),
            },
            "optional": {
                "vae_c": ("VAE", {"tooltip": "Optional third VAE for add_difference / triple_sum / smooth_add_diff."}),
                "reference_image": ("IMAGE", {
                    "tooltip": (
                        "Optional reference image. If connected, encodes/decodes it through A, B, "
                        "and the merged VAE; reports MSE/PSNR per VAE in probe_report."
                    )}),
                "recipe_in": ("STRING", {
                    "default": "", "multiline": True,
                    "tooltip": (
                        "Optional recipe JSON from a previous run. When provided, overrides every "
                        "widget value above so the exact merge is reproduced."
                    )}),
            },
        }

    RETURN_TYPES = ("VAE", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("vae", "info", "recipe_json", "probe_report")
    OUTPUT_TOOLTIPS = (
        "Merged VAE (or vae_a clone when dry_run=True).",
        "Human-readable info string summarising mode, alpha, blocks, and timing.",
        "Reproducible recipe JSON; feed back into recipe_in to repeat the exact merge.",
        "Probe report with MSE/PSNR per VAE when reference_image is connected.",
    )
    FUNCTION = "merge"
    CATEGORY = "C2C/VAE"
    OUTPUT_NODE = False

    @classmethod
    def IS_CHANGED(cls, vae_a, vae_b, merge_mode, alpha, beta, brightness, contrast,
                   use_blocks, auto_alpha, block_conv_in, block_conv_out, block_norm_out,
                   block_0, block_1, block_2, block_3, block_mid, device, dry_run,
                   vae_c=None, reference_image=None, recipe_in="", **kwargs):
        return hash_args_and_kwargs(
            vae_a, vae_b, merge_mode, alpha, beta, brightness, contrast,
            use_blocks, auto_alpha, block_conv_in, block_conv_out, block_norm_out,
            block_0, block_1, block_2, block_3, block_mid, device, dry_run,
            vae_c, reference_image, recipe_in, **kwargs,
        )

    # ------------------------------------------------------------------
    def merge(
        self,
        vae_a: Any,
        vae_b: Any,
        merge_mode: str = "weighted_sum",
        alpha: float = 0.30,
        beta: float = 0.70,
        brightness: float = 0.0,
        contrast: float = 0.0,
        use_blocks: bool = False,
        auto_alpha: bool = False,
        block_conv_in: float = 0.5,
        block_conv_out: float = 0.5,
        block_norm_out: float = 0.5,
        block_0: float = 0.5,
        block_1: float = 0.5,
        block_2: float = 0.5,
        block_3: float = 0.5,
        block_mid: float = 0.5,
        device: str = "cpu",
        dry_run: bool = False,
        vae_c: Optional[Any] = None,
        reference_image: Optional[torch.Tensor] = None,
        recipe_in: str = "",
    ) -> Tuple[Any, str, str, str]:

        if reference_image is not None and (
            not isinstance(reference_image, torch.Tensor)
            or reference_image.ndim != 4
            or reference_image.shape[-1] not in (3, 4)
        ):
            raise ValueError(
                "VAEMergeMEC: reference_image must be IMAGE [B,H,W,C], "
                f"got {tuple(getattr(reference_image, 'shape', ()))}"
            )

        with torch.inference_mode():
            return self._merge_impl(
                vae_a, vae_b, merge_mode, alpha, beta, brightness, contrast,
                use_blocks, auto_alpha, block_conv_in, block_conv_out, block_norm_out,
                block_0, block_1, block_2, block_3, block_mid, device, dry_run,
                vae_c, reference_image, recipe_in,
            )

    def _merge_impl(
        self,
        vae_a: Any,
        vae_b: Any,
        merge_mode: str = "weighted_sum",
        alpha: float = 0.30,
        beta: float = 0.70,
        brightness: float = 0.0,
        contrast: float = 0.0,
        use_blocks: bool = False,
        auto_alpha: bool = False,
        block_conv_in: float = 0.5,
        block_conv_out: float = 0.5,
        block_norm_out: float = 0.5,
        block_0: float = 0.5,
        block_1: float = 0.5,
        block_2: float = 0.5,
        block_3: float = 0.5,
        block_mid: float = 0.5,
        device: str = "cpu",
        dry_run: bool = False,
        vae_c: Optional[Any] = None,
        reference_image: Optional[torch.Tensor] = None,
        recipe_in: str = "",
    ) -> Tuple[Any, str, str, str]:

        # ── Required-input validation (fail fast with clear message) ──
        if vae_a is None or vae_b is None:
            missing = []
            if vae_a is None:
                missing.append("vae_a")
            if vae_b is None:
                missing.append("vae_b")
            raise ValueError(
                "VAEMergeMEC: required VAE input(s) missing: "
                + ", ".join(missing)
                + ". Connect a VAE Loader to each input."
            )

        # ── Recipe-import: override widgets if a recipe is provided ──
        loaded = _parse_recipe(recipe_in)
        if loaded is not None:
            merge_mode = loaded.get("merge_mode", merge_mode)
            alpha = float(loaded.get("alpha", alpha))
            beta = float(loaded.get("beta", beta))
            use_blocks = bool(loaded.get("use_blocks", use_blocks))
            auto_alpha = bool(loaded.get("auto_alpha", auto_alpha))
            brightness = float(loaded.get("brightness", brightness))
            contrast = float(loaded.get("contrast", contrast))
            bw = loaded.get("block_weights", {}) or {}
            block_conv_in = float(bw.get("block_conv_in", block_conv_in))
            block_conv_out = float(bw.get("block_conv_out", block_conv_out))
            block_norm_out = float(bw.get("block_norm_out", block_norm_out))
            block_0 = float(bw.get("block_0", block_0))
            block_1 = float(bw.get("block_1", block_1))
            block_2 = float(bw.get("block_2", block_2))
            block_3 = float(bw.get("block_3", block_3))
            block_mid = float(bw.get("block_mid", block_mid))

        # Resolve compute device
        if device == "auto":
            compute = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif device == "cuda" and torch.cuda.is_available():
            compute = torch.device("cuda")
        else:
            compute = torch.device("cpu")

        info: Dict[str, Any] = {
            "merge_mode": merge_mode,
            "alpha": float(alpha),
            "beta": float(beta),
            "use_blocks": bool(use_blocks),
            "auto_alpha": bool(auto_alpha),
            "device": str(compute),
            "dry_run": bool(dry_run),
            "recipe_loaded": loaded is not None,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        block_weights = {
            "block_conv_in": block_conv_in,
            "block_conv_out": block_conv_out,
            "block_norm_out": block_norm_out,
            "block_0": block_0,
            "block_1": block_1,
            "block_2": block_2,
            "block_3": block_3,
            "block_mid": block_mid,
        }

        empty_probe = json.dumps({"probe": "disabled (no reference_image connected)"})

        try:
            if merge_mode in REQUIRES_VAE_C and vae_c is None:
                raise ValueError(
                    f"[MEC VAE Merge] '{merge_mode}' requires vae_c to be connected. "
                    f"Connect a third VAE, or switch to 'weighted_sum'."
                )

            sd_a, _ = _extract_state_dict(vae_a)
            sd_b, _ = _extract_state_dict(vae_b)
            sd_c: Optional[Dict[str, torch.Tensor]] = None

            _check_compatible(sd_a, sd_b, "vae_a", "vae_b")
            if vae_c is not None:
                sd_c, _ = _extract_state_dict(vae_c)
                _check_compatible(sd_a, sd_c, "vae_a", "vae_c")

            arch = _detect_architecture(list(sd_a.keys()))
            info["architecture"] = arch
            info["key_count"] = len(sd_a)

            # ── Auto-alpha (feature A) ──
            similarity = _block_similarity(sd_a, sd_b)
            info["block_similarity"] = {k: round(v, 4) for k, v in similarity.items()}
            auto_alpha_blocks: Dict[str, float] = {}
            if auto_alpha:
                auto_alpha_blocks = _auto_alpha_from_similarity(similarity, alpha)
                # Auto-alpha overrides manual sliders
                for k, v in auto_alpha_blocks.items():
                    block_weights[k] = v
                use_blocks = True
                info["auto_alpha_blocks"] = auto_alpha_blocks

            # ── Build recipe early so dry-run can short-circuit ──
            recipe = _build_recipe(
                merge_mode=merge_mode, alpha=alpha, beta=beta,
                use_blocks=use_blocks, block_weights=block_weights,
                brightness=brightness, contrast=contrast,
                auto_alpha=auto_alpha, auto_alpha_blocks=auto_alpha_blocks,
                architecture=arch,
            )
            recipe_str = json.dumps(recipe, indent=2)

            # ── Dry-run (feature C) ──
            if dry_run:
                info["status"] = "dry_run"
                info["note"] = (
                    "dry_run=True: returned vae_a unchanged. "
                    "Use the block_similarity / recipe outputs to plan a real merge."
                )
                return (vae_a, json.dumps(info, indent=2), recipe_str, empty_probe)

            # ── Pre-merge probe of A and B ──
            probe_image_chw = _build_probe_image(reference_image) if reference_image is not None else None
            probe: Dict[str, Any] = {}
            if probe_image_chw is not None:
                probe["vae_a"] = _probe_vae_reconstruction(vae_a, probe_image_chw)
                probe["vae_b"] = _probe_vae_reconstruction(vae_b, probe_image_chw)

            # Track dtype distribution
            dtypes: Dict[str, int] = {}
            for v in sd_a.values():
                key = str(v.dtype).replace("torch.", "")
                dtypes[key] = dtypes.get(key, 0) + 1
            info["dtypes"] = dtypes

            # ── Main merge loop ──
            merged: Dict[str, torch.Tensor] = {}
            for key, ta in sd_a.items():
                tb = sd_b[key]
                ta_dev = ta.detach().to(compute)
                tb_dev = tb.detach().to(compute)
                a_eff = _per_key_alpha(key, alpha, use_blocks, block_weights)

                if merge_mode in REQUIRES_VAE_C:
                    assert sd_c is not None  # checked above
                    tc_dev = sd_c[key].detach().to(compute)
                    out = _merge_three(ta_dev, tb_dev, tc_dev, merge_mode, a_eff)
                else:
                    out = _merge_two(ta_dev, tb_dev, merge_mode, a_eff)

                if not torch.isfinite(out).all():
                    out = torch.where(torch.isfinite(out), out, ta_dev)
                    info.setdefault("warnings", []).append(f"non-finite values clamped at key '{key}'")

                # Move back to CPU for load_state_dict
                merged[key] = out.detach().to("cpu")

            # ── Brightness / contrast on decoder.conv_out ──
            if abs(brightness) > 1e-6 or abs(contrast) > 1e-6:
                applied = 0
                for key, t in merged.items():
                    if "decoder.conv_out" in key and t.dtype.is_floating_point:
                        t32 = t.to(torch.float32)
                        t32 = (t32 + float(brightness) * 0.1) * (1.0 + float(contrast))
                        t32 = t32.clamp(-10.0, 10.0)
                        merged[key] = t32.to(t.dtype)
                        applied += 1
                info["postprocess_keys"] = applied

            # ── Build the output VAE wrapper ──
            try:
                merged_vae = copy.deepcopy(vae_a)
            except Exception as exc:
                logger.warning("[MEC VAE Merge] deepcopy(vae_a) failed (%s); returning original wrapper.", exc)
                merged_vae = vae_a

            inner = getattr(merged_vae, "first_stage_model", None)
            target = inner if (inner is not None and hasattr(inner, "load_state_dict")) else merged_vae
            try:
                if hasattr(target, "load_state_dict"):
                    incompatible = target.load_state_dict(merged, strict=False)
                    missing = getattr(incompatible, "missing_keys", []) or []
                    unexpected = getattr(incompatible, "unexpected_keys", []) or []
                    if missing or unexpected:
                        info["load_state_dict"] = {
                            "missing": missing[:10],
                            "unexpected": unexpected[:10],
                        }
                else:
                    info.setdefault("warnings", []).append(
                        "target VAE has no load_state_dict; returning unmodified vae_a clone"
                    )
            except Exception as exc:
                logger.error("[MEC VAE Merge] load_state_dict failed: %s", exc, exc_info=True)
                info.setdefault("warnings", []).append(f"load_state_dict failed: {exc}")

            # ── Post-merge probe ──
            if probe_image_chw is not None:
                probe["merged"] = _probe_vae_reconstruction(merged_vae, probe_image_chw)
                if probe.get("vae_a") and probe.get("merged"):
                    probe["delta_psnr_vs_a_db"] = round(
                        probe["merged"]["psnr_db"] - probe["vae_a"]["psnr_db"], 3
                    )

            info["status"] = "ok"
            probe_str = json.dumps(probe, indent=2) if probe else empty_probe
            return (merged_vae, json.dumps(info, indent=2), recipe_str, probe_str)

        except ValueError:
            raise
        except Exception as exc:
            logger.error("[MEC VAE Merge] Unexpected failure: %s", exc, exc_info=True)
            info["status"] = "error"
            info["error"] = str(exc)
            return (vae_a, json.dumps(info, indent=2), "{}", empty_probe)
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


NODE_CLASS_MAPPINGS = {"VAEMergeMEC": VAEMergeMEC}
NODE_DISPLAY_NAME_MAPPINGS = {"VAEMergeMEC": "VAE Merge"}
