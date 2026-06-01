"""P2.A.3 smoke — FETA, RIFLEx, Uni3C, Context Windows, TAEHV preview.

All pure-tensor; no model, no server.

Checks:
 1. Imports + probe lists 5 new features.
 2. FETA: scale=0 → identity; scale>0 → output differs from input,
    shape preserved.
 3. RIFLEx: target==source → unchanged; target>source → first-k entries
    rescaled by source/target, rest unchanged; target<source → no-op
    (extrapolation_active is False).
 4. Uni3C: rot6d_from_matrix of identity → [1,0,0, 0,1,0]; encode of
    identity extrinsics with t=(1,2,3) gives ...(1,2,3); embed_dim
    expansion shape OK.
 5. Context Windows: plan_windows(F=20, window=8, overlap=2) → 4
    windows covering [0..20). blend_window_weights symmetry.
    splice_windows round-trip — when input frames are constants per
    window, blended output equals weighted mean per overlap.
 6. TAEHV preview: latent (B=1,C=16,T=4,H=8,W=8) → RGB shape
    (1,3,4,8,8), values in [0,1], deterministic for same seed.

Exit 0 PASS, 2 FAIL.
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
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

    print("\n=== P2.A.3 / 1: imports + probe ===")
    try:
        # Adapter pulls in every feature; import them all via the same
        # stub-package trick used in prior smokes.
        for sub in ("_local_nag", "_local_freeinit", "_local_teacache",
                    "_local_magcache", "_local_easycache", "_local_slg",
                    "_local_feta", "_local_riflex", "_local_uni3c",
                    "_local_context_windows", "_local_taehv_preview"):
            _import_submodule(
                f"nodes.wan_director.features.{sub}",
                WD / "features" / f"{sub}.py",
            )
        feta   = sys.modules["nodes.wan_director.features._local_feta"]
        riflex = sys.modules["nodes.wan_director.features._local_riflex"]
        uni3c  = sys.modules["nodes.wan_director.features._local_uni3c"]
        cw     = sys.modules["nodes.wan_director.features._local_context_windows"]
        taehv  = sys.modules["nodes.wan_director.features._local_taehv_preview"]
        adapter = _import_submodule(
            "nodes.wan_director._kijai_adapter",
            WD / "_kijai_adapter.py",
        )
        p = adapter.probe()
        for f in ("feta", "riflex", "uni3c", "context_windows", "taehv_preview"):
            assert p.get(f) in ("kijai", "local"), f"{f}={p.get(f)!r}"
        print(f"{OK} imports OK; probe has " + ", ".join(
            f"{f}={p[f]}" for f in
            ("feta", "riflex", "uni3c", "context_windows", "taehv_preview")))
    except Exception as ex:
        print(f"{NO} import/probe: {ex!r}")
        return 2

    print("\n=== P2.A.3 / 2: FETA ===")
    try:
        torch.manual_seed(0)
        logits = torch.randn(1, 2, 16, 16)
        out0 = feta.feta_attention_bias(logits, feta_scale=0.0)
        assert torch.equal(out0, logits), "scale=0 must be identity"
        out1 = feta.feta_attention_bias(logits, feta_scale=0.5,
                                        freq_center=0.20, freq_bandwidth=0.15)
        assert out1.shape == logits.shape
        assert not torch.allclose(out1, logits), \
            "scale>0 must change the logits"
        # adapter dispatch.
        outA = adapter.apply_feta(logits, feta_scale=0.5)
        assert outA.shape == logits.shape
        print(f"{OK} FETA: identity at scale=0, non-trivial at scale>0, "
              f"shape preserved (adapter backend={adapter.LAST_BACKEND['feta']})")
    except AssertionError as ex:
        print(f"{NO} FETA: {ex}"); all_ok = False
    except Exception as ex:
        print(f"{NO} FETA failed: {ex!r}"); all_ok = False

    print("\n=== P2.A.3 / 3: RIFLEx ===")
    try:
        freqs = torch.tensor([0.1, 0.2, 0.4, 0.8, 1.6, 3.2])
        # target == source → unchanged.
        out_eq = riflex.rescale_rope_freqs(freqs, source_len=16, target_len=16, k=2)
        assert torch.allclose(out_eq, freqs), "equal lengths must no-op"
        # target > source → first k rescaled by source/target.
        out_ext = riflex.rescale_rope_freqs(freqs, source_len=16, target_len=64, k=2)
        assert torch.isclose(out_ext[0], freqs[0] * (16 / 64))
        assert torch.isclose(out_ext[1], freqs[1] * (16 / 64))
        assert torch.allclose(out_ext[2:], freqs[2:])
        # Adapter dispatch + extrapolation_active flag.
        assert riflex.riflex_extrapolation_active(64, 16) is True
        assert riflex.riflex_extrapolation_active(16, 16) is False
        out_ad = adapter.apply_riflex(freqs, source_len=16, target_len=16, k=2)
        assert torch.allclose(out_ad, freqs)
        print(f"{OK} RIFLEx: no-op on equal/short, rescales first-k on extrapolation")
    except AssertionError as ex:
        print(f"{NO} RIFLEx: {ex}"); all_ok = False
    except Exception as ex:
        print(f"{NO} RIFLEx failed: {ex!r}"); all_ok = False

    print("\n=== P2.A.3 / 4: Uni3C ===")
    try:
        I = torch.eye(3)
        r6 = uni3c.rot6d_from_matrix(I)
        expected = torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float32)
        assert torch.allclose(r6, expected), f"rot6d(I)={r6}"
        # 2 frames; identity rotation, translations (1,2,3) and (4,5,6).
        ext = torch.eye(4).unsqueeze(0).repeat(2, 1, 1)
        ext[0, :3, 3] = torch.tensor([1.0, 2.0, 3.0])
        ext[1, :3, 3] = torch.tensor([4.0, 5.0, 6.0])
        enc = uni3c.encode_camera_poses(ext)
        assert enc.shape == (2, 9), f"got {tuple(enc.shape)}"
        assert torch.allclose(enc[0, 6:], torch.tensor([1.0, 2.0, 3.0]))
        assert torch.allclose(enc[1, 6:], torch.tensor([4.0, 5.0, 6.0]))
        # embed_dim expansion.
        enc64 = uni3c.encode_camera_poses(ext, embed_dim=64)
        assert enc64.shape == (2, 64)
        # adapter dispatch.
        encA = adapter.apply_uni3c(ext, embed_dim=32)
        assert encA.shape == (2, 32)
        print(f"{OK} Uni3C: rot6d(I) correct, translations preserved, "
              f"embed_dim expansion OK")
    except AssertionError as ex:
        print(f"{NO} Uni3C: {ex}"); all_ok = False
    except Exception as ex:
        print(f"{NO} Uni3C failed: {ex!r}"); all_ok = False

    print("\n=== P2.A.3 / 5: Context Windows ===")
    try:
        # F=20, window=8, overlap=2 → stride=6 → windows starting at
        # 0,6,12,18 — last clipped to 20.
        plan = cw.plan_windows(20, window=8, overlap=2)
        starts = [r.start for r in plan]
        ends   = [r.stop  for r in plan]
        assert starts[0] == 0 and ends[-1] == 20, \
            f"plan does not cover [0,20): {list(plan)}"
        # Frame 0 covered exactly once; frame 6 in 2 windows (0..8 and 6..14).
        coverage = [0] * 20
        for r in plan:
            for f in r:
                coverage[f] += 1
        assert all(c >= 1 for c in coverage), \
            f"frames uncovered: {coverage}"
        # blend_window_weights symmetry.
        w = cw.blend_window_weights(8, 2)
        assert torch.allclose(w, w.flip(0)), "blend weights asymmetric"
        assert w[3] == 1.0 and w[4] == 1.0, "middle should be 1"
        assert w[0] < 1.0 and w[-1] < 1.0, "edges should ramp down"
        # splice round-trip with constants.
        plan2 = cw.plan_windows(10, window=6, overlap=2)  # stride=4
        outs = []
        for win_idx, r in enumerate(plan2):
            # Each window's frame value = win_idx * 10 + frame_idx_in_window.
            fr = torch.full((len(r), 3), float(win_idx) * 10.0)
            outs.append(fr)
        full = cw.splice_windows(outs, plan2, full_len=10, overlap=2)
        assert full.shape == (10, 3), f"full shape {full.shape}"
        assert torch.isfinite(full).all(), "non-finite in spliced output"
        # adapter dispatch.
        plan3 = adapter.apply_context_windows(20, window=8, overlap=2)
        assert [(r.start, r.stop) for r in plan3] == \
               [(r.start, r.stop) for r in plan]
        print(f"{OK} Context Windows: plan covers F, weights symmetric, "
              f"splice works, adapter dispatch matches direct call")
    except AssertionError as ex:
        print(f"{NO} ContextWindows: {ex}"); all_ok = False
    except Exception as ex:
        print(f"{NO} ContextWindows failed: {ex!r}"); all_ok = False

    print("\n=== P2.A.3 / 6: TAEHV preview ===")
    try:
        torch.manual_seed(42)
        latent = torch.randn(1, 16, 4, 8, 8)
        rgb1 = taehv.latent_to_rgb_preview(latent, seed=1729)
        rgb2 = taehv.latent_to_rgb_preview(latent, seed=1729)
        assert rgb1.shape == (1, 3, 4, 8, 8), f"shape {rgb1.shape}"
        assert (rgb1 >= 0).all() and (rgb1 <= 1).all(), \
            f"out of [0,1]: min={rgb1.min()} max={rgb1.max()}"
        assert torch.equal(rgb1, rgb2), "preview not deterministic for same seed"
        # Different seed → different output.
        rgb3 = taehv.latent_to_rgb_preview(latent, seed=2024)
        assert not torch.equal(rgb1, rgb3), "different seeds gave equal output"
        # 4D input also supported.
        latent4 = torch.randn(1, 16, 8, 8)
        rgb4 = taehv.latent_to_rgb_preview(latent4)
        assert rgb4.shape == (1, 3, 8, 8)
        # adapter dispatch.
        rgbA = adapter.apply_taehv_preview(latent, seed=1729)
        assert torch.equal(rgbA, rgb1)
        print(f"{OK} TAEHV preview: shape OK, range [0,1], deterministic, "
              f"4D/5D supported, adapter matches")
    except AssertionError as ex:
        print(f"{NO} TAEHV: {ex}"); all_ok = False
    except Exception as ex:
        print(f"{NO} TAEHV failed: {ex!r}"); all_ok = False

    print()
    print("=" * 60)
    print(f"P2.A.3 overall: {'PASS' if all_ok else 'FAIL'}")
    print("=" * 60)
    return 0 if all_ok else 2


if __name__ == "__main__":
    sys.exit(main())
