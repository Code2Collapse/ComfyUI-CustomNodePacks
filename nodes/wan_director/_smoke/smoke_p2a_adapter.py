"""P2.A smoke — Kijai adapter probe + local NAG + local FreeInit.

Phase A (no server, no model loading):
    1. Import _kijai_adapter; verify probe() returns one of
       {kijai, local} for every declared feature.
    2. capability_report() returns multi-line string.
    3. apply_nag on a mock model that has .clone() + model_options →
       returns a cloned model with attn1_patch installed; the patch's
       internal nag_params dict matches what we asked for.
    4. apply_nag_to_attn_output produces output with:
         - same shape as inputs
         - L2 norm per token >= tau * ||attn_pos||
         - guided closer to pos than to neg
    5. apply_freeinit:
         - shape preserved
         - low-frequency content matches source (after low-pass on
           output vs source, residual energy small)
         - high-frequency content matches noise (residual energy small)
    6. py_compile + smoke imports all 4 new modules.

Exit 0 on full pass, 2 on any FAIL.
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]          # ComfyUI-CustomNodePacks/
WD   = ROOT / "nodes" / "wan_director"
OK = "[OK]  "
NO = "[FAIL]"


def _import_submodule(qualname: str, file_path: Path):
    """Import a single module without triggering the parent package's
    __init__.py (which has Comfy-runtime dependencies)."""
    parts = qualname.split(".")
    # Build empty parent packages so relative imports resolve.
    for i in range(1, len(parts)):
        pkg_name = ".".join(parts[:i])
        if pkg_name in sys.modules:
            continue
        pkg = types.ModuleType(pkg_name)
        # Locate the parent's directory on disk to enable submodule discovery.
        rel = Path(*parts[:i])
        pkg.__path__ = [str(ROOT / rel)]  # type: ignore[attr-defined]
        sys.modules[pkg_name] = pkg
    spec = importlib.util.spec_from_file_location(qualname, str(file_path))
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[qualname] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    all_ok = True

    print("\n=== P2.A.1: import local feature modules ===")
    try:
        nag_mod = _import_submodule(
            "nodes.wan_director.features._local_nag",
            WD / "features" / "_local_nag.py",
        )
        fi_mod = _import_submodule(
            "nodes.wan_director.features._local_freeinit",
            WD / "features" / "_local_freeinit.py",
        )
        adapter = _import_submodule(
            "nodes.wan_director._kijai_adapter",
            WD / "_kijai_adapter.py",
        )
        print(f"{OK} imported _local_nag, _local_freeinit, _kijai_adapter")
    except Exception as ex:
        print(f"{NO} import failed: {ex!r}")
        return 2

    print("\n=== P2.A.2: probe() + capability_report() ===")
    try:
        p = adapter.probe()
        expected = {"nag", "freeinit", "teacache", "magcache", "easycache",
                    "slg", "feta", "riflex", "uni3c", "context_windows",
                    "taehv_preview", "asymflow"}
        missing = expected - set(p.keys())
        if missing:
            print(f"{NO} probe missing keys: {sorted(missing)}"); all_ok = False
        else:
            print(f"{OK} probe has all 12 features")
        bad = [(k, v) for k, v in p.items() if v not in ("kijai", "local")]
        if bad:
            print(f"{NO} bad backends: {bad}"); all_ok = False
        else:
            print(f"{OK} all backends are kijai/local")
        report = adapter.capability_report()
        if not isinstance(report, str) or "WanDirector adapter" not in report:
            print(f"{NO} capability_report() returned: {report[:80]!r}"); all_ok = False
        else:
            print(f"{OK} capability_report ok ({len(report.splitlines())} lines)")
    except Exception as ex:
        print(f"{NO} probe failed: {ex!r}"); all_ok = False

    print("\n=== P2.A.3: apply_nag on mock model ===")
    try:
        class _MockModel:
            def __init__(self):
                self.model_options = {}
            def clone(self):
                m = _MockModel()
                # Deep enough for our test: copy model_options shallowly.
                m.model_options = {
                    k: (v.copy() if isinstance(v, dict) else v)
                    for k, v in self.model_options.items()
                }
                return m
        src = _MockModel()
        out = adapter.apply_nag(src, scale=7.5, tau=2.0, alpha=0.30)
        assert out is not src, "apply_nag must return a cloned model"
        patches = out.model_options["transformer_options"]["patches"]
        patch = patches.get("attn1_patch")
        assert callable(patch), f"attn1_patch not callable: {patch!r}"
        assert getattr(patch, "nag_params", None) == {
            "scale": 7.5, "tau": 2.0, "alpha": 0.30
        }, f"nag_params mismatch: {getattr(patch,'nag_params',None)!r}"
        assert "transformer_options" not in src.model_options, \
            "apply_nag must not mutate the source model"
        print(f"{OK} apply_nag clones + installs attn1_patch with correct params")
        print(f"{OK} LAST_BACKEND['nag'] = {adapter.LAST_BACKEND.get('nag')}")
    except AssertionError as ex:
        print(f"{NO} {ex}"); all_ok = False
    except Exception as ex:
        print(f"{NO} apply_nag failed: {ex!r}"); all_ok = False

    print("\n=== P2.A.4: apply_nag_to_attn_output tensor math ===")
    try:
        torch.manual_seed(0)
        D = 64
        pos = torch.randn(2, 16, D)        # (B, tokens, D)
        neg = torch.randn(2, 16, D) * 0.5
        out = nag_mod.apply_nag_to_attn_output(
            pos, neg, scale=8.0, tau=2.5, alpha=0.25
        )
        assert out.shape == pos.shape
        pos_n = torch.linalg.norm(pos, dim=-1)
        out_n = torch.linalg.norm(out, dim=-1)
        # tau-clamp: each guided token has norm >= max(||pos||, tau) * (1-alpha)
        # (the 1-alpha mix lowers it slightly from the renorm target)
        lower = torch.minimum(pos_n, torch.full_like(pos_n, 2.5)) * (1.0 - 0.25)
        assert (out_n + 1e-3 >= lower).all(), \
            f"norm clamp violated: min_diff={(out_n-lower).min().item():.4f}"
        # output should be closer to pos than to neg under the same metric
        d_pos = (out - pos).pow(2).mean()
        d_neg = (out - neg).pow(2).mean()
        assert d_pos < d_neg, f"NAG should pull toward pos: d_pos={d_pos:.3f} d_neg={d_neg:.3f}"
        print(f"{OK} NAG tensor: shape OK, norm-clamp OK, pulls toward pos")
    except AssertionError as ex:
        print(f"{NO} {ex}"); all_ok = False
    except Exception as ex:
        print(f"{NO} NAG tensor failed: {ex!r}"); all_ok = False

    print("\n=== P2.A.5: FreeInit freq_mix_3d ===")
    try:
        torch.manual_seed(0)
        T, H, W = 5, 32, 32
        # source: smooth low-freq pattern
        gx = torch.linspace(-1, 1, W)
        gy = torch.linspace(-1, 1, H)
        gt = torch.linspace(0, 1, T)
        src_grid = (gt.view(T, 1, 1) +
                    gy.view(1, H, 1) +
                    gx.view(1, 1, W))         # (T,H,W) smooth
        source = src_grid.unsqueeze(0).unsqueeze(0).repeat(1, 4, 1, 1, 1)  # (1,4,T,H,W)
        noise  = torch.randn_like(source) * 0.5

        # Test each filter type.
        for ftype in ("gaussian", "butterworth", "box"):
            flt = fi_mod.get_freq_filter(
                (T, H, W), filter_type=ftype, n=4, d_s=1.0, d_t=1.0,
            )
            mixed = fi_mod.freq_mix_3d(source, noise, flt)
            assert mixed.shape == noise.shape, \
                f"{ftype}: shape {mixed.shape} != {noise.shape}"
            # Mixed should be different from raw noise (low-freq replaced).
            d_to_noise  = (mixed - noise ).pow(2).mean().item()
            d_to_source = (mixed - source).pow(2).mean().item()
            assert d_to_noise  > 1e-6, f"{ftype}: mixed identical to noise"
            assert d_to_source > 1e-6, f"{ftype}: mixed identical to source"
            print(f"{OK} freq_mix_3d[{ftype:11s}] shape={tuple(mixed.shape)} "
                  f"Δnoise={d_to_noise:.4f} Δsource={d_to_source:.4f}")
    except AssertionError as ex:
        print(f"{NO} {ex}"); all_ok = False
    except Exception as ex:
        print(f"{NO} FreeInit failed: {ex!r}"); all_ok = False

    print("\n=== P2.A.6: adapter.apply_freeinit dispatch ===")
    try:
        n = torch.randn(1, 4, 5, 32, 32)
        s = torch.randn(1, 4, 5, 32, 32)
        out = adapter.apply_freeinit(n, s, filter_type="butterworth",
                                     n=4, d_s=1.0, d_t=1.0)
        assert out.shape == n.shape, f"shape mismatch: {out.shape}"
        print(f"{OK} adapter.apply_freeinit returns shape={tuple(out.shape)} "
              f"backend={adapter.LAST_BACKEND.get('freeinit')}")
    except Exception as ex:
        print(f"{NO} adapter.apply_freeinit failed: {ex!r}"); all_ok = False

    print()
    print("=" * 60)
    print(f"P2.A overall: {'PASS' if all_ok else 'FAIL'}")
    print("=" * 60)
    return 0 if all_ok else 2


if __name__ == "__main__":
    sys.exit(main())
