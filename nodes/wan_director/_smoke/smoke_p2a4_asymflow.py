"""P2.A.4 smoke — AsymFlow adapter wrapper (pure-tensor side).

The model-patch path requires the ComfyUI runtime (``comfy`` package)
so it can't run in this isolated smoke. We do verify:

  1. adapter.asymflow_time_shift(shift, t) matches the underlying
     pure function in nodes.asymflow_sampler for tensor + scalar
     inputs.
  2. probe()['asymflow'] == 'local' (no kijai mirror exists).
  3. The mathematical contract:
       shift = 1.0 → sigma(t) = sqrt(t/(1-t)) / (1 + sqrt(t/(1-t)))
                    = sqrt(t) / (sqrt(t)+sqrt(1-t))
       At t = 0.5, sigma = 0.5  for any shift > 0.
       Monotone increasing in t.
  4. apply_asymflow lazily imports the patch class without crashing
     on missing comfy (we expect a controlled RuntimeError, not an
     ImportError, when the runtime is absent — confirming the
     adapter does not blow up at import time).

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

    print("\n=== P2.A.4 / 1: imports ===")
    try:
        for sub in ("_local_nag", "_local_freeinit", "_local_teacache",
                    "_local_magcache", "_local_easycache", "_local_slg",
                    "_local_feta", "_local_riflex", "_local_uni3c",
                    "_local_context_windows", "_local_taehv_preview"):
            _import_submodule(
                f"nodes.wan_director.features.{sub}",
                WD / "features" / f"{sub}.py")
        adapter = _import_submodule(
            "nodes.wan_director._kijai_adapter",
            WD / "_kijai_adapter.py")
        # The asymflow sibling module needs to be loaded so the late
        # imports inside adapter.apply_asymflow / asymflow_time_shift
        # resolve correctly.
        asym = _import_submodule(
            "nodes.asymflow_sampler",
            ROOT / "nodes" / "asymflow_sampler.py")
        print(f"{OK} imported adapter + asymflow_sampler")
    except Exception as ex:
        print(f"{NO} import failed: {ex!r}")
        return 2

    print("\n=== P2.A.4 / 2: probe registers asymflow=local ===")
    try:
        p = adapter.probe()
        assert p.get("asymflow") == "local", f"asymflow={p.get('asymflow')!r}"
        print(f"{OK} probe['asymflow'] = local")
    except AssertionError as ex:
        print(f"{NO} {ex}"); all_ok = False

    print("\n=== P2.A.4 / 3: pure-tensor asymflow_time_shift ===")
    try:
        # Scalar: matches direct implementation.
        s_adap = adapter.asymflow_time_shift(3.0, 0.25)
        s_pure = asym.asymflow_time_shift(3.0, 0.25)
        assert abs(s_adap - s_pure) < 1e-9, f"adap={s_adap} pure={s_pure}"
        # Tensor: matches direct implementation.
        t = torch.linspace(0.01, 0.99, 10)
        v_adap = adapter.asymflow_time_shift(3.0, t)
        v_pure = asym.asymflow_time_shift(3.0, t)
        assert torch.allclose(v_adap, v_pure)
        # At t=0.5, shift=1.0 → sigma=0.5.
        half = adapter.asymflow_time_shift(1.0, 0.5)
        assert abs(half - 0.5) < 1e-6, f"sigma(0.5;shift=1)={half}"
        # Monotone increasing in t for shift=3.
        v3 = adapter.asymflow_time_shift(3.0, t)
        assert (v3[1:] - v3[:-1] > 0).all(), "AsymFlow not monotone"
        # Increasing shift → sigma decreases at fixed t (more steps high-noise).
        v_shift_low  = adapter.asymflow_time_shift(0.5, 0.5)
        v_shift_high = adapter.asymflow_time_shift(5.0, 0.5)
        assert v_shift_low > v_shift_high, \
            f"shift monotonicity broken: low={v_shift_low} high={v_shift_high}"
        print(f"{OK} asymflow_time_shift: adapter==pure, t=0.5/shift=1 gives "
              f"sigma=0.5, monotone in t, decreasing in shift")
    except AssertionError as ex:
        print(f"{NO} math: {ex}"); all_ok = False
    except Exception as ex:
        print(f"{NO} math failed: {ex!r}"); all_ok = False

    print("\n=== P2.A.4 / 4: apply_asymflow controlled failure on missing comfy ===")
    try:
        # We do NOT have the comfy runtime in this sys.path → the patch
        # class will raise RuntimeError("comfy.model_sampling is unavailable").
        # The important contract: the adapter must propagate a clean
        # RuntimeError, not crash at import time.
        try:
            adapter.apply_asymflow(object(), shift=3.0, multiplier=1000)
            # If somehow the runtime IS available, we accept that too —
            # just record the backend.
            print(f"{OK} apply_asymflow ran (comfy runtime present); "
                  f"backend={adapter.LAST_BACKEND.get('asymflow')}")
        except RuntimeError as re:
            # Expected when comfy is absent.
            assert "comfy" in str(re).lower() or "model_sampling" in str(re).lower() \
                or "clone" in str(re).lower(), \
                f"unexpected RuntimeError: {re}"
            print(f"{OK} apply_asymflow raises controlled RuntimeError "
                  f"when comfy runtime missing: {re}")
        except AttributeError as ae:
            # object() has no .clone() — equally acceptable controlled error.
            print(f"{OK} apply_asymflow raises controlled AttributeError "
                  f"on stub-object input: {ae}")
    except Exception as ex:
        print(f"{NO} apply_asymflow failed unexpectedly: {ex!r}"); all_ok = False

    print()
    print("=" * 60)
    print(f"P2.A.4 overall: {'PASS' if all_ok else 'FAIL'}")
    print("=" * 60)
    return 0 if all_ok else 2


if __name__ == "__main__":
    sys.exit(main())
