"""Local re-implementation of Uni3C (Unified 3D Camera Control).

Reference: kijai's WanVideoUni3CController node and the upstream Uni3C
3-D camera-control conditioning approach. Uni3C encodes a per-frame
camera pose (translation + rotation) as a continuous embedding vector
that is added to the model's positional embedding stream.

For Wan video models the camera pose is supplied as a sequence of
4×4 extrinsic matrices, one per frame. This module converts that
sequence into an embedding-vector sequence suitable for injection
alongside the temporal RoPE.

Pure tensor; no model weights involved.

Encoding:
    For each frame:
        translation t ∈ R³           → identity
        rotation R ∈ SO(3)           → 6-D continuous rep
                                       (first 2 columns of R, flattened)
    Concatenate → 9-D per-frame camera embedding.
Optionally project to ``embed_dim`` via a fixed sinusoidal expansion
(parameter-free) so the embedding lives in the model's hidden space.
"""
from __future__ import annotations

import torch


def rot6d_from_matrix(R: torch.Tensor) -> torch.Tensor:
    """Convert (..., 3, 3) rotation matrices to (..., 6) continuous rep.

    Uses the first two columns of R, flattened. This is the standard
    "continuous 6D" parametrisation (Zhou et al., 2019, arXiv:1812.07035)
    that avoids the singularities of Euler angles / quaternions.
    """
    if R.shape[-2:] != (3, 3):
        raise ValueError(f"rot6d_from_matrix: expected (...,3,3); got {tuple(R.shape)}")
    return torch.cat((R[..., :, 0], R[..., :, 1]), dim=-1)


def encode_camera_poses(
    extrinsics: torch.Tensor,
    *,
    embed_dim: int | None = None,
) -> torch.Tensor:
    """Encode a per-frame camera-pose sequence as an embedding sequence.

    Args:
        extrinsics: shape ``(F, 4, 4)`` — per-frame 4×4 extrinsic
            matrix (rotation+translation). May also be ``(B, F, 4, 4)``.
        embed_dim:  if not None, expand the 9-D base embedding to
            ``embed_dim`` via a fixed sinusoidal projection.

    Returns:
        ``(..., F, 9)`` or ``(..., F, embed_dim)`` tensor.
    """
    if extrinsics.shape[-2:] != (4, 4):
        raise ValueError(
            f"encode_camera_poses: need (..., F, 4, 4); got "
            f"{tuple(extrinsics.shape)}"
        )
    R = extrinsics[..., :3, :3]
    t = extrinsics[..., :3,  3]
    r6 = rot6d_from_matrix(R)               # (..., F, 6)
    base = torch.cat((r6, t), dim=-1)       # (..., F, 9)
    if embed_dim is None:
        return base
    if embed_dim < 9:
        raise ValueError(
            f"encode_camera_poses: embed_dim={embed_dim} < 9 (base dim)"
        )
    return _sinusoidal_project(base, embed_dim)


def _sinusoidal_project(x: torch.Tensor, out_dim: int) -> torch.Tensor:
    """Project the last dim from D to ``out_dim`` via fixed sin/cos features.

    Parameter-free — uses a deterministic frequency band so the same
    pose always maps to the same embedding.
    """
    D = x.shape[-1]
    n_extra = out_dim - D
    if n_extra <= 0:
        return x
    half = (n_extra + 1) // 2
    # Geometric frequency band 1 .. 2**(half-1), broadcast over the D dims.
    freqs = torch.arange(1, half + 1, device=x.device, dtype=x.dtype)
    # Project each input dim through the band → (..., D, half)
    angles = x.unsqueeze(-1) * freqs       # (..., D, half)
    sin = torch.sin(angles)
    cos = torch.cos(angles)
    extra = torch.cat((sin, cos), dim=-1)  # (..., D, 2*half)
    extra = extra.reshape(*x.shape[:-1], D * 2 * half)
    # Concatenate raw + sinusoidal, then truncate to out_dim.
    return torch.cat((x, extra), dim=-1)[..., :out_dim]
