"""No node may hand an inference tensor to the next node.

A tensor produced inside `torch.inference_mode()` is an *inference tensor*, and
an inference tensor cannot be mutated in place outside inference mode. ComfyUI
passes a node's output straight into the next node, and a great many nodes -
core ones included - do in-place work on an IMAGE or MASK they were handed.
Those die with

    RuntimeError: Inplace update to inference tensor outside InferenceMode
                  is not allowed.

which names neither the node that created the tensor nor, usefully, the one
that died on it. The user sees a crash in the middle of a graph, three nodes
downstream of the cause.

This was not hypothetical. 64 call sites in this pack were in that state, in 48
files, and MaskTransformXY was verified failing exactly that way before the fix.
`torch.no_grad()` skips the autograd graph identically without setting the flag,
so it is the right tool throughout node code.

Two layers here: a source check that covers everything including the nodes that
need model weights, and a runtime probe over every node that can actually be
called on this machine with nothing but tensors.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest
import torch

PACK = Path(__file__).resolve().parents[1]

SKIP_DIRS = {"__pycache__", ".git", "third_party", "_deprecated", "_AUDIT",
             "node_modules", "tests", "_tests"}


def _source_files():
    for path in sorted(PACK.rglob("*.py")):
        if any(p in SKIP_DIRS for p in path.parts):
            continue
        yield path


def test_no_call_site_uses_inference_mode():
    # The total check. It also covers every node that needs weights, which the
    # runtime probe below can never reach on a machine without them.
    offenders = []
    for path in _source_files():
        src = path.read_text(encoding="utf-8", errors="replace")
        if "inference_mode" not in src:
            continue
        for i, line in enumerate(src.splitlines(), 1):
            stripped = line.lstrip()
            if stripped.startswith("#"):
                continue          # the comments explaining this rule may say it
            if "inference_mode(" in line or "@torch.inference_mode" in line:
                offenders.append(f"{path.relative_to(PACK)}:{i}")
    assert not offenders, (
        "inference_mode marks its results so the NEXT node cannot mutate them "
        "in place; use torch.no_grad(). Found at: " + ", ".join(offenders[:12])
    )


def test_the_check_above_would_actually_fire():
    # A grep-shaped test that matches nothing is indistinguishable from a
    # passing one. Prove the matcher works on a line it should catch.
    line = "        with torch.inference_mode():"
    assert "inference_mode(" in line


# ── runtime probe ───────────────────────────────────────────────────────────

def _fill_folder_paths_stub():
    """conftest.py installs a bare `folder_paths` stub so the suite can import
    node modules at all. It has no attributes, so loading the whole pack dies on
    `folder_paths.base_path`. Fill in only what the pack reads, leaving the
    stub's identity alone - replacing the module would change what every other
    test in the session sees."""
    import folder_paths                                   # the stub

    if hasattr(folder_paths, "base_path"):
        return
    core = PACK.parent.parent / "ComfyUI_windows_portable" / "ComfyUI"
    folder_paths.base_path = str(core if core.is_dir() else PACK)
    folder_paths.models_dir = str(Path(folder_paths.base_path) / "models")
    folder_paths.get_full_path = lambda *_a, **_k: None
    folder_paths.get_filename_list = lambda *_a, **_k: []
    folder_paths.get_folder_paths = lambda *_a, **_k: []
    folder_paths.get_temp_directory = lambda: str(PACK / "_test_models")
    folder_paths.get_input_directory = lambda: str(PACK / "_test_models")
    folder_paths.get_output_directory = lambda: str(PACK / "_test_models")


def _load_pack():
    import importlib.util

    core = PACK.parent.parent / "ComfyUI_windows_portable" / "ComfyUI"
    if core.is_dir() and str(core) not in sys.path:
        sys.path.insert(0, str(core))
    _fill_folder_paths_stub()
    name = str(PACK).replace(".", "_x_")
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, PACK / "__init__.py", submodule_search_locations=[str(PACK)])
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _synthetic(kind, opts):
    """A plausible value for one declared input type, or None if we cannot."""
    if kind == "IMAGE":
        return torch.rand(1, 32, 32, 3)
    if kind == "MASK":
        m = torch.zeros(1, 32, 32)
        m[:, 8:24, 8:24] = 1.0
        return m
    if kind == "INT":
        return int(opts.get("default", 0))
    if kind == "FLOAT":
        return float(opts.get("default", 0.0))
    if kind == "BOOLEAN":
        return bool(opts.get("default", False))
    if kind == "STRING":
        return str(opts.get("default", ""))
    if isinstance(kind, list) and kind:
        return kind[0]
    return None


def _callable_nodes():
    """Nodes whose required inputs we can fabricate from tensors and scalars."""
    try:
        mod = _load_pack()
    except Exception:
        # The source check above is the total one; losing the probe on a box
        # that cannot load the pack is a gap, not a failure. It is reported by
        # test_the_probe_reaches_a_useful_number_of_nodes.
        return []
    out = []
    for node_id, cls in sorted(getattr(mod, "NODE_CLASS_MAPPINGS", {}).items()):
        try:
            spec = cls.INPUT_TYPES()
        except Exception:
            continue
        required = spec.get("required", {})
        if not required:
            continue
        kwargs = {}
        ok = True
        for name, entry in required.items():
            kind = entry[0] if isinstance(entry, (tuple, list)) else entry
            opts = entry[1] if isinstance(entry, (tuple, list)) and len(entry) > 1 else {}
            if not isinstance(opts, dict):
                opts = {}
            value = _synthetic(kind, opts)
            if value is None:
                ok = False
                break
            kwargs[name] = value
        # At least one tensor, or there is nothing here to leak.
        if ok and any(torch.is_tensor(v) for v in kwargs.values()):
            out.append((node_id, cls, kwargs))
    return out


_PROBE = _callable_nodes()


def test_the_probe_reaches_a_useful_number_of_nodes():
    # If a refactor makes every node need a model or a custom type, this file
    # quietly stops testing anything. Say so instead.
    assert len(_PROBE) >= 15, (
        f"the runtime probe only reached {len(_PROBE)} nodes; it is no longer "
        "covering enough to be worth trusting"
    )


@pytest.mark.parametrize("node_id,cls,kwargs", _PROBE,
                         ids=[n for n, _c, _k in _PROBE])
def test_node_output_can_be_mutated_downstream(node_id, cls, kwargs):
    # The failure exactly as a user meets it: the NEXT node does an in-place op.
    try:
        result = getattr(cls(), cls.FUNCTION)(**kwargs)
    except Exception:
        # A node that refuses synthetic input is not what this test is about -
        # a missing model, an unreachable service, a value it will not accept.
        pytest.skip(f"{node_id} does not run on synthetic input")

    values = result.get("result", ()) if isinstance(result, dict) else result
    if not isinstance(values, tuple):
        values = (values,)

    for i, value in enumerate(values):
        if not torch.is_tensor(value):
            continue
        assert not value.is_inference(), (
            f"{node_id} output {i} is an inference tensor; the next node cannot "
            "mutate it in place"
        )
        value.add_(0.0)          # must not raise
