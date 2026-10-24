"""Sequence parallelism inside one ComfyUI process (Ulysses / Ring / USP), any OS.

How a model call runs (docs/c2c-distributed.md):

1. ComfyUI core's MultiGPU clones give one copy of the model per GPU (`comfy.multigpu.create_multigpu_deepclones`,
   reloaded from the loader, LoRA and patches re-applied by core every run).
2. A CALC_COND_BATCH wrapper takes the clones away from core's own cond splitter for this model (the two would fight
   over the GPUs) and hands them to:
3. a DIFFUSION_MODEL wrapper. The call's batch items (cond / uncond, images) go to GPU groups first - that needs no
   exchange at all. Inside a group the SAME call runs on every GPU at once: rank 0 in the sampler's thread, the
   others on core's per-device worker threads, each with the inputs moved to its GPU.
4. Inside the model, a block-replace patch on every transformer block cuts the token sequence into one shard per GPU
   at the first block and gathers it back after the last one, so everything outside the blocks (patch embedding,
   head, unpatchify) is untouched. RoPE tables / per-token modulation are cut the same way.
5. Self-attention inside the blocks goes through the attention override: Ulysses all-to-all and / or Ring passes
   (attention.py), exact. Cross-attention to the (replicated) text needs no exchange and runs locally.

No model code is copied (ComfyUI is GPL-3.0; this pack is Apache-2.0): only core's documented extension points -
wrappers, `patches_replace["dit"]`, `optimized_attention_override` - are used.

Supported model classes are listed in ADAPTERS; anything else is refused with the reason, never run wrong.
"""
from __future__ import annotations

import dataclasses
import logging
import threading
from typing import Any, Callable, List, Optional, Sequence, Tuple

import torch

from . import attention as A
from .comm import Comm, run_ranks
from .planner import Plan, plan_sequence_parallel, worth_splitting

log = logging.getLogger("c2c.distributed")

STATE_KEY = "c2c_usp"                 # transformer_options: UspState of the node that set this model up
RANKS_KEY = "c2c_usp_ranks"           # transformer_options: [(device, ModelPatcher)] for this sampling run
POOL_KEY = "c2c_usp_pool"             # transformer_options: core's MultiGPUThreadPool, when it made one
WRAPPER_KEY = "c2c_usp"
SIM_KEY = "c2c_usp_sim"               # additional models: extra copies on ONE device (C2C_DISTRIBUTED_SIMULATE)


def simulate_ranks() -> int:
    """C2C_DISTRIBUTED_SIMULATE=N: run N ranks on the one device there is - a test of the whole path (exchanges,
    clones, sampler) on a single-GPU or CPU box. Slower than no split and N x the model memory; never for speed."""
    import os
    try:
        return max(0, int(os.environ.get("C2C_DISTRIBUTED_SIMULATE", "0") or 0))
    except ValueError:
        return 0

_tls = threading.local()


# ── model adapters ──────────────────────────────────────────────────────────────────────────────────────────

class Adapter:
    """How one model family lays out tokens inside its blocks."""

    name = "?"
    classes: Tuple[str, ...] = ()
    img_dim = 1                         # token dim of the block stream ("img")
    self_attention_first_only = True    # True: only the first attention call per block is self-attention
    batch_split = False                 # True: a call's batch items (cond / uncond, images) may go to different GPUs

    def heads(self, dm) -> Optional[int]:
        return None

    def tokens_estimate(self, dm, args, kwargs) -> Optional[int]:
        return None

    def shard_args(self, args: dict, sl: slice, total: int) -> dict:
        raise NotImplementedError


def _cut(t, dim: int, sl: slice, total: int):
    """Slice `t` along `dim` when that dim is the token axis (length == total); otherwise leave it."""
    if isinstance(t, torch.Tensor) and t.dim() > dim and t.shape[dim] == total:
        idx = [slice(None)] * t.dim()
        idx[dim] = sl
        return t[tuple(idx)]
    return t


class WanAdapter(Adapter):
    name = "Wan 2.1 / 2.2"
    classes = ("WanModel",)
    img_dim = 1
    batch_split = True

    def batch(self, args, kwargs):
        x = args[0] if args else kwargs.get("x")
        return int(x.shape[0]) if isinstance(x, torch.Tensor) else 1

    def heads(self, dm):
        return int(getattr(dm, "num_heads", 0)) or None

    def tokens_estimate(self, dm, args, kwargs):
        x = args[0] if args else kwargs.get("x")
        ps = getattr(dm, "patch_size", None)
        if not isinstance(x, torch.Tensor) or x.dim() != 5 or not ps:
            return None
        t, h, w = x.shape[2:]
        return -(-t // ps[0]) * -(-h // ps[1]) * -(-w // ps[2])

    def shard_args(self, args, sl, total):
        out = dict(args)
        out["img"] = _cut(args["img"], 1, sl, total)
        out["pe"] = _cut(args.get("pe"), 1, sl, total)
        out["vec"] = _cut(args.get("vec"), 1, sl, total)        # Wan 2.2 TI2V: per-token timestep modulation
        return out


class MiniMaxH3Adapter(Adapter):
    name = "MiniMax H3"
    classes = ("MiniMaxH3Model",)
    img_dim = 0                         # packed stream [N, C], batch 1
    self_attention_first_only = False   # the block's only attention is self-attention

    def heads(self, dm):
        try:
            return int(dm.blocks[0].attn.heads)
        except Exception:  # noqa: BLE001
            return None

    def shard_args(self, args, sl, total):
        out = dict(args)
        out["img"] = _cut(args["img"], 0, sl, total)
        out["rope_freqs"] = _cut(args.get("rope_freqs"), 1, sl, total)
        segs = []
        for a, b, row in args.get("mod_segments") or []:   # (start, stop, row) over the packed tokens
            lo, hi = max(a, sl.start), min(b, sl.stop)
            if lo >= hi:
                continue
            if isinstance(row, torch.Tensor) and row.dim() > 0:
                row = row[lo - a:hi - a]
            segs.append((lo - sl.start, hi - sl.start, row))
        out["mod_segments"] = segs
        return out


ADAPTERS: Tuple[Adapter, ...] = (WanAdapter(), MiniMaxH3Adapter())


def adapter_for(dm) -> Optional[Adapter]:
    name = type(dm).__name__
    for a in ADAPTERS:
        if name in a.classes:
            return a
    return None


def supported_models() -> List[str]:
    return [f"{a.name} ({', '.join(a.classes)})" for a in ADAPTERS]


# ── per-rank state (thread-local) ───────────────────────────────────────────────────────────────────────────

@dataclasses.dataclass
class RankCtx:
    comm: Comm
    mesh: A.Mesh
    adapter: Adapter
    n_blocks: int
    prev_override: Optional[Callable] = None
    total: int = 0                      # full token count of the block stream
    sl: slice = slice(0, 0)
    sharded: bool = False
    attn_calls: int = 0


def _ctx() -> Optional[RankCtx]:
    return getattr(_tls, "ctx", None)


# ── block patch: shard at the first block, gather after the last ─────────────────────────────────────────────

def block_patch(args: dict, extra: dict) -> dict:
    ctx = _ctx()
    original = extra["original_block"]
    if ctx is None:
        return original(args)
    i = int(args["transformer_options"].get("block_index", 0))
    ad = ctx.adapter
    img = args["img"]
    if not ctx.sharded:
        ctx.total = int(img.shape[ad.img_dim])
        sizes = A.shard_sizes(ctx.total, ctx.comm.world)
        start = sum(sizes[:ctx.comm.rank])
        ctx.sl = slice(start, start + sizes[ctx.comm.rank])
        ctx.sharded = True
        local = ad.shard_args(args, ctx.sl, ctx.total)
    else:
        local = ad.shard_args({**args, "img": None}, ctx.sl, ctx.total)
        local["img"] = img                                  # already this rank's shard
    ctx.attn_calls = 0
    out = original(local)
    if i >= ctx.n_blocks - 1:
        out = dict(out)
        out["img"] = A.unshard(ctx.comm, out["img"], dim=ad.img_dim)
        ctx.sharded = False
    return out


# ── attention override ──────────────────────────────────────────────────────────────────────────────────────

def _local(func, prev, q, k, v, heads, **kw):
    if prev is not None:
        return prev(func, q, k, v, heads, **kw)
    return func(q, k, v, heads, **kw)


def attention_override(func, q, k, v, heads, mask=None, skip_reshape=False, skip_output_reshape=False, **kw):
    ctx = _ctx()
    if ctx is not None:
        prev = ctx.prev_override
    else:                                                   # not inside a parallel call: keep the user's override
        prev = getattr((kw.get("transformer_options") or {}).get(STATE_KEY), "prev_override", None)
    if ctx is None or not ctx.sharded:
        return _local(func, prev, q, k, v, heads, mask=mask, skip_reshape=skip_reshape,
                      skip_output_reshape=skip_output_reshape, **kw)
    ctx.attn_calls += 1
    tok_dim = 2 if skip_reshape else 1
    local_len = ctx.sl.stop - ctx.sl.start
    is_self = (q.shape[tok_dim] == k.shape[tok_dim] == local_len
               and (not ctx.adapter.self_attention_first_only or ctx.attn_calls == 1))
    if not is_self:                                         # cross-attention to replicated context: no exchange
        return _local(func, prev, q, k, v, heads, mask=mask, skip_reshape=skip_reshape,
                      skip_output_reshape=skip_output_reshape, **kw)
    if mask is not None:
        raise RuntimeError("C2C Sequence Parallel: this model passed an attention mask to self-attention, which the "
                           "sequence-parallel path does not support yet - remove the node for this model.")
    qb, kb, vb = (A.to_bhld(t, heads, skip_reshape) for t in (q, k, v))

    def local_fn(qq, kk, vv):
        out = _local(func, prev, qq, kk, vv, qq.shape[1], mask=None, skip_reshape=True,
                     skip_output_reshape=True, **kw)
        if out.dim() == 3:                                  # a kernel that ignores skip_output_reshape
            b, n, hd = out.shape
            out = out.view(b, n, qq.shape[1], hd // qq.shape[1]).transpose(1, 2)
        return out

    out = A.usp_attention(ctx.mesh, qb, kb, vb, local_fn=local_fn)
    return A.from_bhld(out, skip_output_reshape)



# ── dispatch: one call, every GPU ────────────────────────────────────────────────────────────────────────────

def _move(obj, device, memo=None):
    """Copy containers and move every tensor inside to `device` (dicts, lists, tuples, dataclasses)."""
    if isinstance(obj, torch.Tensor):
        return obj.to(device) if obj.device != device else obj
    if isinstance(obj, dict):
        return {k: _move(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_move(v, device) for v in obj]
    if isinstance(obj, tuple):
        moved = [_move(v, device) for v in obj]
        return type(obj)(*moved) if hasattr(obj, "_fields") else tuple(moved)
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        try:
            return dataclasses.replace(obj, **{f.name: _move(getattr(obj, f.name), device)
                                               for f in dataclasses.fields(obj) if f.init})
        except Exception:  # noqa: BLE001 - frozen / custom init: leave it
            return obj
    return obj


def _move_options(to: dict, device) -> dict:
    """transformer_options for one rank: its own dict (models write into it per block), tensors on its device."""
    out = {}
    for k, v in to.items():
        if k in (RANKS_KEY, POOL_KEY, STATE_KEY, "wrappers", "callbacks", "patches", "patches_replace"):
            out[k] = v
        else:
            out[k] = _move(v, device)
    return out


def _find_options(args, kwargs) -> Tuple[Optional[dict], Optional[int]]:
    to = kwargs.get("transformer_options")
    if isinstance(to, dict):
        return to, None
    for i, a in enumerate(args):
        if isinstance(a, dict) and ("patches" in a or "wrappers" in a or STATE_KEY in a or RANKS_KEY in a):
            return a, i
    return None, None


def foreign_patches(to: dict) -> List[str]:
    """Patches that would see sharded tokens without knowing it -> refuse rather than produce garbage."""
    bad = []
    for key, fn in ((to.get("patches_replace") or {}).get("dit") or {}).items():
        if fn is not block_patch:
            bad.append(f"block replace {key}")
    for kind in ("double_block", "single_block", "attn1_patch", "attn1_output_patch"):
        if (to.get("patches") or {}).get(kind):
            bad.append(f"patch '{kind}'")
    if to.get("optimized_attention_override") is not attention_override:
        bad.append("another attention override was set after C2C Sequence Parallel")
    return bad


_PER_CHUNK_KEYS = ("cond_or_uncond", "uuids")          # transformer_options lists with one entry per cond chunk


def _slice_batch(obj, total: int, lo: int, hi: int):
    """Copy containers, cutting every tensor whose first dim is the call's batch to items [lo, hi)."""
    if isinstance(obj, torch.Tensor):
        return obj[lo:hi] if obj.dim() > 0 and obj.shape[0] == total else obj
    if isinstance(obj, dict):
        return {k: _slice_batch(v, total, lo, hi) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_slice_batch(v, total, lo, hi) for v in obj]
    if isinstance(obj, tuple) and not hasattr(obj, "_fields"):
        return tuple(_slice_batch(v, total, lo, hi) for v in obj)
    return obj


def _slice_options(to: dict, total: int, lo: int, hi: int) -> dict:
    out = dict(to)
    sig = to.get("sigmas")
    if isinstance(sig, torch.Tensor) and sig.dim() > 0 and sig.shape[0] == total:
        out["sigmas"] = sig[lo:hi]
    for key in _PER_CHUNK_KEYS:
        v = to.get(key)
        if isinstance(v, list) and v and total % len(v) == 0:
            chunk = total // len(v)
            out[key] = [v[c] for c in range(lo // chunk, -(-hi // chunk))]
    return out


def _layout(world: int, batch: int, tokens, adapter: Adapter, min_tokens: int) -> Tuple[int, int]:
    """(batch degree, sequence degree): batch items first (no exchange at all), tokens for the rest."""
    bdeg = 1
    if adapter.batch_split and batch > 1:
        bdeg = max(d for d in range(1, world + 1) if world % d == 0 and batch % d == 0)
    seq = world // bdeg
    if tokens is not None and seq > 1 and not worth_splitting(tokens, seq, min_tokens):
        seq = max([d for d in range(1, seq + 1) if (world // bdeg) % d == 0 and (d == 1 or tokens // d >= min_tokens)])
    return bdeg, seq


def diffusion_model_wrapper(executor, *args, **kwargs):
    to, pos = _find_options(args, kwargs)
    ranks = (to or {}).get(RANKS_KEY)
    state = (to or {}).get(STATE_KEY)
    if not to or state is None or not ranks or len(ranks) < 2 or not state.enabled():
        return executor(*args, **kwargs)
    dm = executor.class_obj
    ad = adapter_for(dm)
    if ad is None:
        raise RuntimeError(f"C2C Sequence Parallel does not support {type(dm).__name__} yet "
                           f"(supported: {'; '.join(supported_models())}).")
    bad = foreign_patches(to)
    if bad:
        raise RuntimeError("C2C Sequence Parallel cannot run with: " + "; ".join(bad)
                           + ". Put C2C Sequence Parallel last in the model chain, or remove the conflicting node.")
    world = len(ranks)
    tokens = ad.tokens_estimate(dm, args, kwargs)
    batch = ad.batch(args, kwargs) if ad.batch_split else 1
    bdeg, seq = _layout(world, batch, tokens, ad, state.min_tokens_per_gpu)
    if bdeg * seq < 2:
        state.note(f"{tokens} tokens: below {state.min_tokens_per_gpu} per GPU, ran on one GPU")
        return executor(*args, **kwargs)
    used = bdeg * seq
    plan = plan_sequence_parallel(seq, heads=ad.heads(dm), tokens=tokens, prefer=state.prefer,
                                  fast_domain=state.fast_domain or None, min_tokens_per_rank=0)
    what = []
    if bdeg > 1:
        what.append(f"batch {bdeg} ({batch // bdeg} of {batch} items per group)")
    if seq > 1:
        what.append(f"{plan.mode}: Ulysses {plan.ulysses} x Ring {plan.ring}")
    state.note(" x ".join(what) + f" over {used} GPUs" + (f", {tokens} tokens" if tokens else "")
               + (f" ({world - used} idle)" if used < world else ""))
    devices = [torch.device(d) for d, _ in ranks[:used]]
    pool = to.get(POOL_KEY)
    original_name = getattr(executor.original, "__name__", "_forward")
    per = batch // bdeg

    def rank_fn(comm: Comm):
        r = comm.rank
        g = r // seq
        dev = devices[r]
        a = list(args)
        kw = dict(kwargs)
        own_to = to
        if bdeg > 1:
            lo, hi = g * per, (g + 1) * per
            a = [x if i == pos else _slice_batch(x, batch, lo, hi) for i, x in enumerate(a)]
            kw = {k: (v if k == "transformer_options" else _slice_batch(v, batch, lo, hi)) for k, v in kw.items()}
            own_to = _slice_options(to, batch, lo, hi)
        if r > 0:
            a = [x if i == pos else _move(x, dev) for i, x in enumerate(a)]
            kw = {k: (v if k == "transformer_options" else _move(v, dev)) for k, v in kw.items()}
        own = _move_options(own_to, dev)
        if pos is None:
            kw["transformer_options"] = own
        else:
            a[pos] = own
        sub = comm.split([[gg * seq + i for i in range(seq)] for gg in range(bdeg)]) if bdeg > 1 else comm
        if seq > 1:
            _tls.ctx = RankCtx(comm=sub, mesh=A.Mesh.build(sub, plan.ulysses, plan.ring), adapter=ad,
                               n_blocks=len(dm.blocks), prev_override=state.prev_override)
        try:
            if r == 0:
                out = executor(*a, **kw)
            else:
                dm_r = ranks[r][1].model.diffusion_model
                fn = getattr(dm_r, original_name)
                cls = type(executor)
                out = cls.new_class_executor(fn, dm_r, executor.wrappers, executor.idx + 1).execute(*a, **kw)
        finally:
            _tls.ctx = None
        if bdeg == 1:
            return out
        # batch groups: each group's first rank hands its slice to rank 0, which puts the batch back together
        if r == 0:
            parts = [out] + [comm.recv(out, gg * seq) for gg in range(1, bdeg)]
            return torch.cat(parts, dim=0)
        if r % seq == 0:
            comm.send(out, 0)
        return None

    use_pool = pool is not None and len(set(devices)) == used and all(d in pool.devices for d in devices[1:])
    submit = (lambda r, job: pool.submit(devices[r], job)) if use_pool else None
    try:
        results = run_ranks(used, rank_fn, devices=devices, submit=submit, timeout=state.timeout)
    finally:
        if use_pool:
            for r in range(1, used):
                pool.get_result(devices[r])     # drain the pool's (None, None) acknowledgements
    return results[0]


def calc_cond_batch_wrapper(executor, model, conds, x_in, timestep, model_options):
    """Take core's MultiGPU clones for sequence parallelism instead of core's cond split."""
    to = model_options.get("transformer_options") or {}
    state = to.get(STATE_KEY)
    clones = model_options.get("multigpu_clones")
    if state is None:
        return executor(model, conds, x_in, timestep, model_options)
    if clones:
        ranks = list(clones.items())
    elif state.sim:
        ranks = [(x_in.device, None)] + [(c.load_device, c) for c in state.sim]
    else:
        return executor(model, conds, x_in, timestep, model_options)
    for cond in conds:
        for c in cond or []:
            if isinstance(c, dict) and c.get("control") is not None:
                raise RuntimeError("C2C Sequence Parallel: ControlNet / control conditioning is not supported "
                                   "with sequence parallelism yet - remove C2C Sequence Parallel for this run.")
    mo = dict(model_options)
    mo.pop("multigpu_clones", None)
    to = dict(to)
    to[RANKS_KEY] = ranks
    to[POOL_KEY] = model_options.get("multigpu_thread_pool")
    mo["transformer_options"] = to
    return executor(model, conds, x_in, timestep, mo)


# ── node-side setup ─────────────────────────────────────────────────────────────────────────────────────────

class UspState:
    """Per-node settings carried in transformer_options (deep-copied by core with the model options)."""

    def __init__(self, prefer="auto", min_tokens_per_gpu=1024, fast_domain=0, prev_override=None,
                 timeout=600.0):
        self.prefer = prefer
        self.min_tokens_per_gpu = int(min_tokens_per_gpu)
        self.fast_domain = int(fast_domain)
        self.prev_override = prev_override
        self.timeout = float(timeout)
        self.last: List[str] = []
        self.sim: list = []                 # simulated rank copies (ModelPatchers on the main device)

    def __deepcopy__(self, memo):
        n = UspState(self.prefer, self.min_tokens_per_gpu, self.fast_domain, self.prev_override, self.timeout)
        n.last = self.last                  # shared log, so the node can read what the runs did
        n.sim = self.sim
        return n

    @staticmethod
    def enabled() -> bool:
        from . import enabled
        return enabled()

    def note(self, msg: str):
        if not self.last or self.last[-1] != msg:
            self.last.append(msg)
            del self.last[:-20]
            log.info("C2C Sequence Parallel: %s", msg)


def apply(model, gpus: int = 0, prefer: str = "auto", min_tokens_per_gpu: int = 1024, fast_domain: int = 0,
          make_clones: bool = True):
    """MODEL -> MODEL with sequence parallelism set up. Returns (model, report lines)."""
    dm = model.model.diffusion_model
    ad = adapter_for(dm)
    if ad is None:
        raise ValueError(f"C2C Sequence Parallel does not support {type(dm).__name__} yet. "
                         f"Supported: {'; '.join(supported_models())}.")
    report = [f"model: {type(dm).__name__} ({ad.name}), heads {ad.heads(dm)}"]
    m = model.clone()
    if make_clones:
        import comfy.model_management as mm
        import comfy.multigpu
        available = len(mm.get_all_torch_devices(exclude_current=False))
        want = available if int(gpus) <= 0 else min(int(gpus), available)
        if want >= 2:
            try:
                m = comfy.multigpu.create_multigpu_deepclones(m, want, reuse_loaded=True)
            except RuntimeError as exc:     # loader without a reload factory (core's own requirement)
                raise ValueError(f"C2C Sequence Parallel needs one model copy per GPU and this loader cannot make "
                                 f"them: {exc}") from exc
        n = 1 + len(m.get_additional_models_with_key("multigpu"))
        sim = simulate_ranks()
        sim_copies = []
        if n < 2 and sim >= 2:
            try:
                sim_copies = [m.deepclone_multigpu(new_load_device=m.load_device) for _ in range(sim - 1)]
            except RuntimeError as exc:
                raise ValueError(f"C2C_DISTRIBUTED_SIMULATE needs a loader that can reload the model: {exc}") from exc
            m.set_additional_models(SIM_KEY, sim_copies)
            report.append(f"SIMULATION (C2C_DISTRIBUTED_SIMULATE={sim}): {sim} ranks on {m.load_device} - tests the "
                          f"whole path, slower than one GPU and {sim} x the model memory")
            n = sim
        elif n < 2:
            report.append(f"GPUs: 1 of {available} visible - nothing to split, the model runs as before")
        if n >= 2 and not sim_copies:
            report.append(f"GPUs: {n} of {available} visible")
        p = plan_sequence_parallel(n, heads=ad.heads(dm), prefer=prefer, fast_domain=fast_domain or None)
        report.append(f"plan: {p.mode} (Ulysses {p.ulysses} x Ring {p.ring})" + "".join(f"; {x}" for x in p.notes))
    to = m.model_options.setdefault("transformer_options", {})
    prev = to.get("optimized_attention_override")
    if prev is attention_override:
        prev = getattr(to.get(STATE_KEY), "prev_override", None)
    state = UspState(prefer, min_tokens_per_gpu, fast_domain, prev)
    if make_clones:
        state.sim = sim_copies
    to[STATE_KEY] = state
    to["optimized_attention_override"] = attention_override
    for i in range(len(dm.blocks)):
        m.set_model_patch_replace(block_patch, "dit", "double_block", i)
    m.remove_wrappers_with_key("diffusion_model", WRAPPER_KEY)
    m.remove_wrappers_with_key("calc_cond_batch", WRAPPER_KEY)
    m.add_wrapper_with_key("diffusion_model", WRAPPER_KEY, diffusion_model_wrapper)
    m.add_wrapper_with_key("calc_cond_batch", WRAPPER_KEY, calc_cond_batch_wrapper)
    return m, report, state
