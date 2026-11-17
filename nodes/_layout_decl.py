"""A28 (ORDERS 29) declarations: which inputs show when, read by js/_c2c_node_layout.js. Contract:
docs/audit/A28/README.md section 2; the per-node rules come from docs/audit/A28/PARAM_CNP_A.md.

Display only. `declare_layout(cls, layout)` wraps a V1 node's INPUT_TYPES so every listed input gets the c2c_* option
keys; names, order, types, defaults and tooltips stay exactly as they were, every input is still saved and still
reaches the node's function. Option dicts are COPIED (a spec another class shares is never changed), and an input put
in a c2c_section loses core's "advanced" flag (README section 3: Nodes 2.0 would fold it twice).
"""
from __future__ import annotations

from typing import Callable, Dict, Mapping, Union

Layout = Mapping[str, Mapping[str, object]]

KEYS = ("c2c_section", "c2c_show_if", "c2c_show_if_any", "c2c_hide_if_linked", "c2c_show_if_linked", "c2c_hidden",
        "c2c_show_if_status")


def apply_layout(spec: dict, layout: Layout) -> dict:
    """A copy of `spec` (an INPUT_TYPES dict) with `layout[name]` merged into each listed input's options."""
    out = dict(spec)
    for group in ("required", "optional"):
        g = spec.get(group)
        if not isinstance(g, dict):
            continue
        new = {}
        for name, s in g.items():
            keys = layout.get(name)
            if keys and isinstance(s, (tuple, list)) and s:
                opts = dict(s[1]) if len(s) > 1 and isinstance(s[1], dict) else {}
                opts.update(keys)
                if opts.get("c2c_section"):
                    opts.pop("advanced", None)
                s = (s[0], opts) + tuple(s[2:])
            new[name] = s
        out[group] = new
    return out


def declare_layout(cls, layout: Union[Layout, Callable[[dict], Layout]]):
    """Wrap `cls.INPUT_TYPES` (its own classmethod / staticmethod) with apply_layout. `layout` may be a function of
    the original spec (for inputs generated from tables). Returns cls."""
    raw = cls.__dict__.get("INPUT_TYPES")
    if raw is None:
        raise TypeError(f"{cls.__name__} defines no INPUT_TYPES of its own")
    orig = raw.__func__ if isinstance(raw, (classmethod, staticmethod)) else raw
    static = isinstance(raw, staticmethod)

    def INPUT_TYPES(c):
        spec = orig() if static else orig(c)
        lay = layout(spec) if callable(layout) else layout
        return apply_layout(spec, lay)

    INPUT_TYPES.__doc__ = getattr(orig, "__doc__", None)
    INPUT_TYPES.__c2c_layout__ = True
    cls.INPUT_TYPES = classmethod(INPUT_TYPES)
    return cls


def sections(**groups) -> Dict[str, Dict[str, object]]:
    """sections(Performance=["device", "dtype"]) -> {"device": {"c2c_section": "Performance"}, ...}"""
    out: Dict[str, Dict[str, object]] = {}
    for name, members in groups.items():
        for m in members:
            out.setdefault(m, {})["c2c_section"] = name.replace("_", " ")
    return out


def merge(*layouts: Layout) -> Dict[str, Dict[str, object]]:
    """Several layouts in one (later keys win per input)."""
    out: Dict[str, Dict[str, object]] = {}
    for lay in layouts:
        for name, keys in lay.items():
            out.setdefault(name, {}).update(keys)
    return out
