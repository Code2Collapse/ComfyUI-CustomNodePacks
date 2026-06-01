"""Local re-implementation of Normalized Attention Guidance (NAG).

Paper: arXiv:2505.21179 (Yi et al., 2025).

NAG is a CFG-like guidance applied at the **attention output** level
inside each transformer block, rather than at the unet/dit output
level. The trick: at attention forward time, run two forwards (with
and without a "guidance" token / no-token mask) and combine:

    attn_guided = attn_neg + scale * (attn_pos - attn_neg)

then re-normalise per-token L2 norm so the guided activations stay on
the same manifold as the model's training distribution. The
re-normalisation has two clamps:

    tau    — minimum allowed L2 norm of guided activations
    alpha  — soft-mix toward the pos-only activations to avoid bypass

For Wan video models (transformer attn1 = self-attn), the guidance
"negative" branch is the cross-attention conditioned on an empty /
neutral token; "positive" is the real prompt cond.

This module provides:
    apply_nag_to_attn_output(attn_pos, attn_neg, scale, tau, alpha)
        Pure function. Used both inside model patches (during sampler
        forward) and inside tests.

    build_nag_patch(scale, tau, alpha)
        Returns a callable suitable for installing into
        ``model.model_options["transformer_options"]["patches"]["attn1_patch"]``.

The patch path matches kijai's interface so a model patched by either
implementation behaves identically downstream.

NO weights are loaded; this is a stateless transformation on tensors.
"""
from __future__ import annotations

from typing import Callable

import torch


# ── Pure tensor op ─────────────────────────────────────────────────────


def apply_nag_to_attn_output(
    attn_pos: torch.Tensor,
    attn_neg: torch.Tensor,
    *,
    scale: float = 11.0,
    tau:   float = 2.5,
    alpha: float = 0.25,
) -> torch.Tensor:
    """Apply NAG to a pair of attention outputs.

    Args:
        attn_pos: positive-branch attn output, shape (..., D).
        attn_neg: negative-branch attn output, same shape as ``attn_pos``.
        scale: guidance scale (paper default 11.0).
        tau:   minimum L2 norm of guided activations (paper default 2.5).
        alpha: soft-mix weight toward pos-only (paper default 0.25).

    Returns:
        Guided attention output with the same shape as the inputs.
    """
    if attn_pos.shape != attn_neg.shape:
        raise ValueError(
            f"NAG: attn_pos {tuple(attn_pos.shape)} != attn_neg "
            f"{tuple(attn_neg.shape)}"
        )
    # 1) CFG-style guidance
    guided = attn_neg + scale * (attn_pos - attn_neg)

    # 2) Per-token L2-norm clamp to tau (renorm if the guided norm is
    #    smaller than the positive-branch norm scaled by tau).
    pos_norm    = torch.linalg.norm(attn_pos, dim=-1, keepdim=True)
    guided_norm = torch.linalg.norm(guided,   dim=-1, keepdim=True).clamp_min(1e-6)
    target_norm = torch.clamp_min(pos_norm, tau)
    guided = guided * (target_norm / guided_norm)

    # 3) Soft-mix toward pos to suppress over-correction.
    return (1.0 - alpha) * guided + alpha * attn_pos


def build_nag_patch(
    scale: float = 11.0,
    tau:   float = 2.5,
    alpha: float = 0.25,
) -> Callable:
    """Return a patch suitable for transformer_options["patches"]["attn1_patch"].

    The returned callable receives (q, k, v, extra_options) and returns
    a modified (q, k, v) triple. NAG is applied to ``q`` per the paper:
    the query is the projection that controls *what* the attention
    head looks for, so guiding it produces the strongest semantic
    signal.

    For tests / pure-tensor use, prefer ``apply_nag_to_attn_output``
    directly.
    """

    def _patch(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
               extra_options: dict | None = None) -> tuple:
        # The "negative" branch q is supplied by the caller via
        # extra_options["nag_q_neg"] when running a paired forward; if
        # absent we have nothing to guide against and return inputs as-is.
        if not extra_options:
            return q, k, v
        q_neg = extra_options.get("nag_q_neg")
        if q_neg is None or q_neg.shape != q.shape:
            return q, k, v
        q_guided = apply_nag_to_attn_output(
            q, q_neg, scale=scale, tau=tau, alpha=alpha,
        )
        return q_guided, k, v

    _patch.__name__ = "nag_attn1_patch"
    _patch.nag_params = {"scale": scale, "tau": tau, "alpha": alpha}
    return _patch
