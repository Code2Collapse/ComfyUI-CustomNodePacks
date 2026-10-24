"""Sequence-parallel attention: Ulysses, Ring, and their USP hybrid. Exact, not approximate.

Layout everywhere: [B, H, L, D] (batch, heads, tokens, head dim). Every rank holds a contiguous shard of the token
sequence (its own length; shards may be uneven). The result on each rank is the attention output for its own shard,
bit-for-bit the same maths as attention over the whole sequence on one device (up to float rounding order).

* Ulysses (DeepSpeed-Ulysses, arXiv 2309.14509): all-to-all turns "my tokens, all heads" into "all tokens, my heads",
  full attention per head group, all-to-all back. Needs heads % degree == 0.
* Ring (arXiv 2310.01889): K/V blocks travel around the ring; each rank folds every block into its output with the
  online-softmax merge (running log-sum-exp), so no rank ever holds the whole sequence.
* USP (arXiv 2405.07719): ranks laid out as ring x ulysses; Ulysses inside a group (fast links), Ring across groups.

The attention mask, when given, is the GLOBAL one (broadcastable to [B, H, Lq, Lk] over the whole sequence, bool =
keep, float = additive bias); each rank slices what it needs. (raylight's USP path drops the mask - research.md.)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import torch

from .comm import Comm


# ── single-block attention with log-sum-exp ─────────────────────────────────────────────────────────────────

def _bias(mask: Optional[torch.Tensor], dtype) -> Optional[torch.Tensor]:
    if mask is None:
        return None
    if mask.dtype == torch.bool:
        return torch.zeros(mask.shape, dtype=dtype, device=mask.device).masked_fill(~mask, float("-inf"))
    return mask.to(dtype)


def block_attention(q, k, v, mask=None, scale=None) -> Tuple[torch.Tensor, torch.Tensor]:
    """softmax(q k^T * scale + mask) v for one K/V block, plus the per-row log-sum-exp.

    Returns (out [B,H,Lq,D] in q's dtype, lse [B,H,Lq] float32, float64 for float64 input). Rows whose keys are all masked get out 0, lse -inf,
    so merging them is a no-op.
    """
    scale = (1.0 / math.sqrt(q.shape[-1])) if scale is None else float(scale)
    acc = torch.float64 if q.dtype == torch.float64 else torch.float32
    s = torch.matmul(q.to(acc), k.to(acc).transpose(-1, -2)) * scale
    b = _bias(mask, acc)
    if b is not None:
        s = s + b
    lse = torch.logsumexp(s, dim=-1)
    p = torch.exp(s - torch.where(torch.isinf(lse), torch.zeros_like(lse), lse).unsqueeze(-1))
    out = torch.matmul(p, v.to(acc))
    out = torch.where(torch.isinf(lse).unsqueeze(-1), torch.zeros_like(out), out)
    return out.to(q.dtype), lse


def merge(out_a, lse_a, out_b, lse_b):
    """Fold two partial attention results over disjoint key sets into one (online-softmax merge)."""
    lse = torch.logaddexp(lse_a, lse_b)
    dead = torch.isinf(lse)
    safe = torch.where(dead, torch.zeros_like(lse), lse)
    wa = torch.where(dead, torch.zeros_like(lse), torch.exp(lse_a - safe)).unsqueeze(-1)
    wb = torch.where(dead, torch.zeros_like(lse), torch.exp(lse_b - safe)).unsqueeze(-1)
    acc = torch.float64 if out_a.dtype == torch.float64 else torch.float32
    out = out_a.to(acc) * wa.to(acc) + out_b.to(acc) * wb.to(acc)
    return out.to(out_a.dtype), lse


def full_attention(q, k, v, mask=None, scale=None) -> torch.Tensor:
    """Reference: plain attention on one device (what the parallel versions must reproduce)."""
    return block_attention(q, k, v, mask, scale)[0]


def _mask_block(mask, rows: Tuple[int, int], cols: Tuple[int, int], heads: Optional[Tuple[int, int]] = None):
    """Slice a global (broadcastable) mask to the given query rows / key columns / head range."""
    if mask is None:
        return None
    m = mask
    while m.dim() < 4:
        m = m.unsqueeze(0)
    if m.shape[-2] != 1:
        m = m[..., rows[0]:rows[1], :]
    if m.shape[-1] != 1:
        m = m[..., cols[0]:cols[1]]
    if heads is not None and m.shape[1] != 1:
        m = m[:, heads[0]:heads[1]]
    return m


# ── mesh: who talks to whom ─────────────────────────────────────────────────────────────────────────────────

@dataclass
class Mesh:
    """ring x ulysses layout of the sequence-parallel ranks. rank = ring_index * ulysses + ulysses_index."""

    world: Comm
    ulysses: Comm
    ring: Comm
    ulysses_degree: int
    ring_degree: int
    lengths: List[int] = field(default_factory=list)     # token count of every world rank's shard (set per call)

    @classmethod
    def build(cls, comm: Comm, ulysses: int, ring: int) -> "Mesh":
        if ulysses * ring != comm.world:
            raise ValueError(f"ulysses ({ulysses}) x ring ({ring}) must equal the world size ({comm.world})")
        u_groups = [[g * ulysses + i for i in range(ulysses)] for g in range(ring)]
        r_groups = [[g * ulysses + i for g in range(ring)] for i in range(ulysses)]
        return cls(comm, comm.split(u_groups), comm.split(r_groups), ulysses, ring)

    @property
    def ring_index(self) -> int:
        return self.world.rank // self.ulysses_degree

    def shard_range(self, rank: int) -> Tuple[int, int]:
        start = sum(self.lengths[:rank])
        return start, start + self.lengths[rank]

    def block_range(self, ring_index: int) -> Tuple[int, int]:
        """Global token range held (after the Ulysses exchange) by ring position `ring_index`."""
        first = ring_index * self.ulysses_degree
        a, _ = self.shard_range(first)
        _, b = self.shard_range(first + self.ulysses_degree - 1)
        return a, b


def _gather_lengths(comm: Comm, n: int, device) -> List[int]:
    got = comm.all_gather(torch.tensor([n], dtype=torch.long, device=device))
    return [int(x.item()) for x in got]


# ── Ulysses exchange ────────────────────────────────────────────────────────────────────────────────────────

def _seq_to_heads(comm: Comm, t: torch.Tensor) -> torch.Tensor:
    """[B, H, Ls, D] (my tokens, all heads) -> [B, H/U, sum Ls, D] (group tokens, my heads)."""
    u = comm.world
    if u == 1:
        return t
    h = t.shape[1]
    if h % u:
        raise ValueError(f"{h} heads cannot be split over a Ulysses degree of {u}")
    chunks = list(t.chunk(u, dim=1))
    return torch.cat(comm.all_to_all(chunks), dim=2)


def _heads_to_seq(comm: Comm, t: torch.Tensor, lengths: Sequence[int]) -> torch.Tensor:
    """Inverse of _seq_to_heads: split tokens back to their owners, gather all head groups."""
    if comm.world == 1:
        return t
    chunks = list(torch.split(t, list(lengths), dim=2))
    return torch.cat(comm.all_to_all(chunks), dim=1)


# ── ring pass ───────────────────────────────────────────────────────────────────────────────────────────────

def _pad_tokens(t: torch.Tensor, n: int) -> torch.Tensor:
    if t.shape[2] == n:
        return t
    pad = torch.zeros(t.shape[0], t.shape[1], n - t.shape[2], t.shape[3], dtype=t.dtype, device=t.device)
    return torch.cat([t, pad], dim=2)


def _ring_pass(mesh: Mesh, q, k, v, mask, scale, heads: Tuple[int, int]):
    """q, k, v: this rank's block (after the Ulysses exchange). Returns this block's output."""
    r = mesh.ring_degree
    me = mesh.ring_index
    rows = mesh.block_range(me)
    if r == 1:
        return block_attention(q, k, v, _mask_block(mask, rows, rows, heads), scale)[0]
    sizes = [mesh.block_range(i)[1] - mesh.block_range(i)[0] for i in range(r)]
    n = max(sizes)
    kv = torch.stack([_pad_tokens(k, n), _pad_tokens(v, n)])     # one message per step
    out = lse = None
    for step in range(r):
        src = (me - step) % r                                       # whose block I hold now
        cols = mesh.block_range(src)
        kk, vv = kv[0][:, :, :sizes[src]], kv[1][:, :, :sizes[src]]
        o, l = block_attention(q, kk, vv, _mask_block(mask, rows, cols, heads), scale)
        out, lse = (o, l) if out is None else merge(out, lse, o, l)
        if step + 1 < r:
            kv = mesh.ring.sendrecv_ring(kv)
    return out


# ── public entry points ─────────────────────────────────────────────────────────────────────────────────────

def usp_attention(mesh: Mesh, q, k, v, mask=None, scale=None, local_fn=None) -> torch.Tensor:
    """Sequence-parallel self-attention. q, k, v: [B, H, Ls, D], this rank's token shard. Returns [B, H, Ls, D].

    local_fn(q, k, v) -> out ([B, h, L, D] in and out), when given, computes the whole-sequence attention of the
    pure-Ulysses case (ring degree 1, no mask) - so the model keeps whatever fast kernel it was using (SDPA, flash,
    sage). Ring steps need the per-block log-sum-exp and always use block_attention.
    """
    if not (q.shape[2] == k.shape[2] == v.shape[2]):
        raise ValueError("sequence-parallel attention shards self-attention; q, k and v must hold the same tokens "
                         "(cross-attention to a replicated context needs no exchange - call block_attention)")
    mesh.lengths = _gather_lengths(mesh.world, q.shape[2], q.device)
    u = mesh.ulysses_degree
    h = q.shape[1]
    if h % u:
        raise ValueError(f"{h} heads cannot be split over a Ulysses degree of {u}; plan a larger ring degree")
    hu = h // u
    my_heads = (mesh.ulysses.rank * hu, (mesh.ulysses.rank + 1) * hu)
    if q.shape == k.shape == v.shape:                       # one all-to-all for q, k and v together
        qq, kk, vv = _seq_to_heads(mesh.ulysses, torch.cat([q, k, v], dim=0)).chunk(3, dim=0)
    else:
        qq, kk, vv = (_seq_to_heads(mesh.ulysses, t) for t in (q, k, v))
    if mesh.ring_degree == 1 and mask is None and local_fn is not None:
        out = local_fn(qq, kk, vv)
    else:
        out = _ring_pass(mesh, qq, kk, vv, mask, scale, my_heads)
    first = mesh.ring_index * u
    group_lengths = mesh.lengths[first:first + u]
    return _heads_to_seq(mesh.ulysses, out, group_lengths)


def ulysses_attention(comm: Comm, q, k, v, mask=None, scale=None) -> torch.Tensor:
    return usp_attention(Mesh.build(comm, comm.world, 1), q, k, v, mask, scale)


def ring_attention(comm: Comm, q, k, v, mask=None, scale=None) -> torch.Tensor:
    return usp_attention(Mesh.build(comm, 1, comm.world), q, k, v, mask, scale)


# ── token sharding helpers ──────────────────────────────────────────────────────────────────────────────────

def shard_sizes(total: int, world: int) -> List[int]:
    """Even split of `total` tokens over `world` ranks (the first ranks take the remainder)."""
    base, extra = divmod(int(total), int(world))
    return [base + (1 if r < extra else 0) for r in range(world)]


def shard(t: torch.Tensor, rank: int, world: int, dim: int = 2) -> torch.Tensor:
    sizes = shard_sizes(t.shape[dim], world)
    return torch.split(t, sizes, dim=dim)[rank]


def unshard(comm: Comm, t: torch.Tensor, dim: int = 2) -> torch.Tensor:
    """All-gather token shards (uneven allowed) back into the whole sequence on every rank."""
    lengths = _gather_lengths(comm, t.shape[dim], t.device)
    n = max(lengths)
    if t.shape[dim] < n:
        pad_shape = list(t.shape)
        pad_shape[dim] = n - t.shape[dim]
        t = torch.cat([t, torch.zeros(pad_shape, dtype=t.dtype, device=t.device)], dim=dim)
    parts = comm.all_gather(t)
    return torch.cat([p.narrow(dim, 0, lengths[i]) for i, p in enumerate(parts)], dim=dim)


# ── ComfyUI layout ──────────────────────────────────────────────────────────────────────────────────────────

def to_bhld(t: torch.Tensor, heads: int, skip_reshape: bool) -> torch.Tensor:
    """ComfyUI optimized-attention input ([B, L, H*D], or [B, H, L, D] with skip_reshape) -> [B, H, L, D]."""
    if skip_reshape:
        return t
    b, n, hd = t.shape
    return t.view(b, n, heads, hd // heads).transpose(1, 2)


def from_bhld(t: torch.Tensor, skip_output_reshape: bool) -> torch.Tensor:
    """[B, H, L, D] -> ComfyUI optimized-attention output ([B, L, H*D], or [B, H, L, D] with skip_output_reshape)."""
    if skip_output_reshape:
        return t
    b, h, n, d = t.shape
    return t.transpose(1, 2).reshape(b, n, h * d)
