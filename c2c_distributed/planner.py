"""Pick a sequence-parallel layout (Ulysses x Ring) for a model on the GPUs at hand, and say why.

Rules (DeepSpeed-Ulysses arXiv 2309.14509, USP arXiv 2405.07719):

* Ulysses moves every token's heads in one all-to-all per attention: cheap on fast links (NVLink / NVSwitch, one
  box), and the head count must divide by its degree.
* Ring passes K/V blocks around the ranks and overlaps the transfer with compute: the better choice across slow
  links (PCIe without P2P, between boxes) and the fallback when the heads do not divide.
* So: Ulysses as large as the heads and the fast-link domain allow, Ring for the rest.
* Too few tokens per rank and the exchange costs more than it saves: below `min_tokens_per_rank` the plan says so
  (the sampler then runs that call on one GPU).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import List, Optional

MIN_TOKENS_PER_RANK = 1024


@dataclass
class Plan:
    world: int
    ulysses: int
    ring: int
    heads: Optional[int] = None
    tokens: Optional[int] = None
    notes: List[str] = field(default_factory=list)

    @property
    def mode(self) -> str:
        if self.world == 1:
            return "single"
        if self.ring == 1:
            return "ulysses"
        if self.ulysses == 1:
            return "ring"
        return "usp"

    def as_dict(self) -> dict:
        d = asdict(self)
        d["mode"] = self.mode
        return d


def _divisors(n: int) -> List[int]:
    return [d for d in range(1, n + 1) if n % d == 0]


def plan_sequence_parallel(world: int, heads: Optional[int] = None, tokens: Optional[int] = None,
                           fast_domain: Optional[int] = None, prefer: str = "auto",
                           min_tokens_per_rank: int = MIN_TOKENS_PER_RANK) -> Plan:
    """world: GPUs taking part. heads: attention heads of the model (None = unknown, Ring only is safe).
    tokens: sequence length of the call (None = unknown). fast_domain: how many of the GPUs share fast links
    (NVLink / same box with P2P); None = all of them. prefer: auto | ulysses | ring."""
    world = max(1, int(world))
    notes: List[str] = []
    if world == 1:
        return Plan(1, 1, 1, heads, tokens, ["one GPU: nothing to split"])
    domain = world if not fast_domain else max(1, min(int(fast_domain), world))
    usable = [d for d in _divisors(world) if (heads is None or heads % d == 0) and d <= domain]
    if heads is None:
        usable = [1]
        notes.append("head count unknown: Ring only (Ulysses needs heads divisible by its degree)")
    if prefer == "ring":
        u = 1
    elif prefer == "ulysses":
        u = max(usable)
        if u != world:
            notes.append(f"Ulysses alone needs {world} to divide {heads} heads within one fast-link domain; "
                         f"using Ulysses {u} x Ring {world // u}")
    else:
        u = max(usable)
    r = world // u
    if heads is not None and u < world and prefer != "ring":
        by_heads = max(d for d in _divisors(world) if heads % d == 0)
        if by_heads > u:
            notes.append(f"Ulysses limited to {u} by the fast-link domain ({domain} GPUs); Ring {r} crosses it")
        else:
            notes.append(f"{heads} heads divide by {u} at most among the divisors of {world}: Ring {r} covers the rest")
    if tokens is not None and tokens // world < min_tokens_per_rank:
        notes.append(f"{tokens} tokens over {world} GPUs is {tokens // world} per GPU (< {min_tokens_per_rank}): "
                     "the exchange outweighs the work - this call runs on one GPU")
    return Plan(world, u, r, heads, tokens, notes)


def worth_splitting(tokens: int, world: int, min_tokens_per_rank: int = MIN_TOKENS_PER_RANK) -> bool:
    return world > 1 and tokens // world >= min_tokens_per_rank
