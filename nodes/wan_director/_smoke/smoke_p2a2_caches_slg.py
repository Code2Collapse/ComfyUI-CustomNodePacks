"""P2.A.2 smoke — TeaCache, MagCache, EasyCache, SLG (+ adapter dispatch).

All pure-tensor / pure-state checks; no model, no server.

Phase A:
  1. Import the 4 new local modules + adapter.
  2. probe() now lists kijai|local for {teacache,magcache,easycache,slg}.
  3. TeaCache:  miss → record → repeated should_skip True (under thresh)
                → record bump → should_skip False (over thresh)
                → max_skips clamp works.
  4. MagCache:  miss → record → predicted small mag → skip
                → predicted big mag → refresh.
  5. EasyCache: identical latent twice → skip; very different latent → refresh.
  6. SLG:  build_slg([3,7,11], 0.5, 0.2, 0.8) → in-window step→guides,
           out-of-window step→pass-through. combine_slg shape preserved.
           make_layer_skip_predicate maps 3/7/11 → True, others → False.
  7. adapter.make_cache("none") → None; "teacache" → TeaCache instance.

Exit 0 on full pass, 2 on any FAIL.
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]          # ComfyUI-CustomNodePacks/
WD   = ROOT / "nodes" / "wan_director"
OK = "[OK]  "
NO = "[FAIL]"


def _import_submodule(qualname: str, file_path: Path):
    parts = qualname.split(".")
    for i in range(1, len(parts)):
        pkg_name = ".".join(parts[:i])
        if pkg_name in sys.modules:
            continue
        pkg = types.ModuleType(pkg_name)
        rel = Path(*parts[:i])
        pkg.__path__ = [str(ROOT / rel)]
        sys.modules[pkg_name] = pkg
    spec = importlib.util.spec_from_file_location(qualname, str(file_path))
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[qualname] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    all_ok = True
    print("\n=== P2.A.2 / 1: imports ===")
    try:
        tc = _import_submodule(
            "nodes.wan_director.features._local_teacache",
            WD / "features" / "_local_teacache.py")
        mc = _import_submodule(
            "nodes.wan_director.features._local_magcache",
            WD / "features" / "_local_magcache.py")
        ec = _import_submodule(
            "nodes.wan_director.features._local_easycache",
            WD / "features" / "_local_easycache.py")
        slg = _import_submodule(
            "nodes.wan_director.features._local_slg",
            WD / "features" / "_local_slg.py")
        # _local_nag and _local_freeinit must be importable for adapter.
        _import_submodule(
            "nodes.wan_director.features._local_nag",
            WD / "features" / "_local_nag.py")
        _import_submodule(
            "nodes.wan_director.features._local_freeinit",
            WD / "features" / "_local_freeinit.py")
        adapter = _import_submodule(
            "nodes.wan_director._kijai_adapter",
            WD / "_kijai_adapter.py")
        print(f"{OK} imported 4 cache/guidance modules + adapter")
    except Exception as ex:
        print(f"{NO} import failed: {ex!r}")
        return 2

    print("\n=== P2.A.2 / 2: probe lists the 4 new features ===")
    try:
        p = adapter.probe()
        for f in ("teacache", "magcache", "easycache", "slg"):
            assert p.get(f) in ("kijai", "local"), f"{f}={p.get(f)!r}"
        print(f"{OK} probe entries OK: " + ", ".join(
            f"{f}={p[f]}" for f in ("teacache", "magcache", "easycache", "slg")))
    except AssertionError as ex:
        print(f"{NO} {ex}"); all_ok = False

    print("\n=== P2.A.2 / 3: TeaCache gating ===")
    try:
        cache = tc.TeaCache(rel_l1_thresh=0.10, max_skips=3)
        emb_a  = torch.tensor([1.0, 1.0, 1.0])
        emb_a2 = torch.tensor([1.02, 1.0, 1.0])   # rel_l1 ≈ 0.0067 < 0.10
        emb_b  = torch.tensor([1.0, 5.0, 1.0])    # huge change
        res_a  = torch.randn(8)
        # First call: miss (no cache).
        assert cache.should_skip(emb_a) is False
        cache.record(emb_a, res_a)
        # Tiny embedding change → should skip, returns cached residual.
        assert cache.should_skip(emb_a2) is True, "expected skip on small Δemb"
        _ = cache.cached_residual
        # max_skips=3 → after 3 skips consecutive, gate clamps to False.
        cache.should_skip(emb_a2); _ = cache.cached_residual
        cache.should_skip(emb_a2); _ = cache.cached_residual
        assert cache.should_skip(emb_a2) is False, "max_skips not enforced"
        # Big embedding change → refresh.
        cache.record(emb_b, torch.randn(8))
        assert cache.should_skip(emb_a) is False
        print(f"{OK} {cache.report()}")
    except AssertionError as ex:
        print(f"{NO} TeaCache: {ex}"); all_ok = False
    except Exception as ex:
        print(f"{NO} TeaCache failed: {ex!r}"); all_ok = False

    print("\n=== P2.A.2 / 4: MagCache gating ===")
    try:
        cache = mc.MagCache(mag_thresh=1.2, ema_alpha=0.5, max_skips=3)
        big   = torch.ones(4) * 10.0   # mag ~20
        small = torch.ones(4) * 0.1    # mag ~0.2
        assert cache.should_skip() is False                # empty
        cache.record(big)                                  # ema_mag ≈ 20
        # predicted small → skip (under threshold).
        assert cache.should_skip(predictor=small) is True
        _ = cache.cached_residual
        # predicted huge → refresh.
        huge = torch.ones(4) * 100.0
        assert cache.should_skip(predictor=huge) is False
        print(f"{OK} {cache.report()}")
    except AssertionError as ex:
        print(f"{NO} MagCache: {ex}"); all_ok = False
    except Exception as ex:
        print(f"{NO} MagCache failed: {ex!r}"); all_ok = False

    print("\n=== P2.A.2 / 5: EasyCache gating ===")
    try:
        cache = ec.EasyCache(l1_thresh=0.05, max_skips=3)
        lat_a = torch.zeros(2, 4, 4)
        lat_b = torch.zeros(2, 4, 4) + 0.001
        lat_c = torch.zeros(2, 4, 4) + 1.0
        res   = torch.randn(2, 4, 4)
        assert cache.should_skip(lat_a) is False
        cache.record(lat_a, res)
        assert cache.should_skip(lat_b) is True, "tiny diff should skip"
        _ = cache.cached_residual
        assert cache.should_skip(lat_c) is False, "big diff should refresh"
        print(f"{OK} {cache.report()}")
    except AssertionError as ex:
        print(f"{NO} EasyCache: {ex}"); all_ok = False
    except Exception as ex:
        print(f"{NO} EasyCache failed: {ex!r}"); all_ok = False

    print("\n=== P2.A.2 / 6: SLG ===")
    try:
        cfg = slg.SLGConfig(skip_layers=(3, 7, 11, 11),  # dedupe check
                            slg_scale=0.5, start_pct=0.2, end_pct=0.8)
        assert cfg.skip_layers == (3, 7, 11), f"dedupe broke: {cfg.skip_layers}"
        # Step window: 10 steps; step 0 (pct 0) out, step 5 (pct 0.56) in,
        # step 9 (pct 1.0) out.
        assert not slg.should_apply_slg(0, 10, cfg)
        assert     slg.should_apply_slg(5, 10, cfg)
        assert not slg.should_apply_slg(9, 10, cfg)
        # Empty skip_layers → never active.
        empty = slg.SLGConfig(skip_layers=())
        assert not slg.should_apply_slg(5, 10, empty)
        # Predicate.
        pred = slg.make_layer_skip_predicate(cfg)
        for i in (3, 7, 11):
            assert pred(i) is True, f"predicate missed layer {i}"
        for i in (0, 1, 2, 4, 6, 10, 12):
            assert pred(i) is False, f"predicate hit layer {i}"
        # combine_slg.
        eps_pos  = torch.randn(1, 4, 8, 16, 16)
        eps_skip = torch.randn(1, 4, 8, 16, 16)
        out = slg.combine_slg(eps_pos, eps_skip, 0.5)
        assert out.shape == eps_pos.shape
        expected = eps_pos + 0.5 * (eps_pos - eps_skip)
        assert torch.allclose(out, expected, atol=1e-6)
        # adapter.apply_slg with step-window gating.
        gated_in  = adapter.apply_slg(eps_pos, eps_skip, cfg, step=5, n_steps=10)
        gated_out = adapter.apply_slg(eps_pos, eps_skip, cfg, step=0, n_steps=10)
        assert torch.allclose(gated_in,  expected, atol=1e-6)
        assert torch.allclose(gated_out, eps_pos,  atol=1e-6), \
            "outside step window must return eps_pos unchanged"
        print(f"{OK} SLG: dedupe, window gating, predicate, combine, "
              f"adapter dispatch all correct")
    except AssertionError as ex:
        print(f"{NO} SLG: {ex}"); all_ok = False
    except Exception as ex:
        print(f"{NO} SLG failed: {ex!r}"); all_ok = False

    print("\n=== P2.A.2 / 7: adapter.make_cache factory ===")
    try:
        assert adapter.make_cache("none") is None
        c1 = adapter.make_cache("teacache",  rel_l1_thresh=0.05)
        c2 = adapter.make_cache("magcache",  mag_thresh=1.5)
        c3 = adapter.make_cache("easycache", l1_thresh=0.01)
        assert isinstance(c1, tc.TeaCache)
        assert isinstance(c2, mc.MagCache)
        assert isinstance(c3, ec.EasyCache)
        assert c1.rel_l1_thresh == 0.05
        assert c2.mag_thresh    == 1.5
        assert c3.l1_thresh     == 0.01
        # Unknown raises.
        try:
            adapter.make_cache("nope")
            raise AssertionError("expected ValueError for unknown cache")
        except ValueError:
            pass
        print(f"{OK} make_cache factory dispatches + validates")
    except AssertionError as ex:
        print(f"{NO} make_cache: {ex}"); all_ok = False
    except Exception as ex:
        print(f"{NO} make_cache failed: {ex!r}"); all_ok = False

    print()
    print("=" * 60)
    print(f"P2.A.2 overall: {'PASS' if all_ok else 'FAIL'}")
    print("=" * 60)
    return 0 if all_ok else 2


if __name__ == "__main__":
    sys.exit(main())
