"""A28 (ORDERS 29) declarations: which inputs show when, read by js/_c2c_node_layout.js. Contract:
docs/audit/A28/README.md section 2; the per-node rules come from docs/audit/A28/PARAM_CNP_A.md.

One module (L9.285): `declare_layout` / `apply_layout` (part A) and `with_layout` / `section` (part B, formerly
nodes/_c2c_layout.py) share the per-input merge `_merged`; each keeps its own behaviour exactly.

Display only. `declare_layout(cls, layout)` wraps a V1 node's INPUT_TYPES so every listed input gets the c2c_* option
keys; names, order, types, defaults and tooltips stay exactly as they were, every input is still saved and still
reaches the node's function. Option dicts are COPIED (a spec another class shares is never changed), and an input put
in a c2c_section loses core's "advanced" flag (README section 3: Nodes 2.0 would fold it twice).
"""
from __future__ import annotations

import logging
from typing import Callable, Dict, Mapping, Union

LOG = logging.getLogger("C2C.Layout")

Layout = Mapping[str, Mapping[str, object]]

KEYS = ("c2c_section", "c2c_show_if", "c2c_show_if_any", "c2c_hide_if_linked", "c2c_show_if_linked", "c2c_hidden",
        "c2c_show_if_status")


def _merged(s, rule: Mapping[str, object], drop_advanced: bool):
    """One input spec with `rule` merged into a COPY of its options dict (a dict another input or class shares never
    picks up the keys); core's "advanced" flag dropped when `drop_advanced`."""
    has_opts = isinstance(s, (tuple, list)) and len(s) >= 2 and isinstance(s[1], dict)
    opts = dict(s[1]) if has_opts else {}
    opts.update(rule)
    if drop_advanced:
        opts.pop("advanced", None)
    head = s[0] if isinstance(s, (tuple, list)) and s else s
    tail = tuple(s[2:]) if isinstance(s, (tuple, list)) and len(s) > 2 else ()
    return (head, opts, *tail)


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
                s = _merged(s, keys, bool({**(s[1] if len(s) > 1 and isinstance(s[1], dict) else {}), **keys}
                                          .get("c2c_section")))
            new[name] = s
        out[group] = new
    return out


def with_layout(spec: dict, layout: Dict[str, dict]) -> dict:
    """Merge ``layout`` (input name -> rule keys) into ``spec`` (an INPUT_TYPES dict) IN PLACE and return it.

    Idempotent (INPUT_TYPES may be called many times on the same cached dict). Unknown keys and undeclared names are
    logged and ignored, never raised: a typo must not take the node (or the pack) down.
    """
    seen = set()
    for group in ("required", "optional"):
        inputs = spec.get(group)
        if not isinstance(inputs, dict):
            continue
        for name, s in list(inputs.items()):
            rule = layout.get(name)
            if rule is None:
                continue
            seen.add(name)
            bad = [k for k in rule if k not in KEYS]
            if bad:
                LOG.warning("A28 layout: %s has unknown keys %s (ignored)", name, bad)
            rule = {k: v for k, v in rule.items() if k in KEYS}
            inputs[name] = _merged(s, rule, "c2c_section" in rule)
    missing = set(layout) - seen
    if missing:
        LOG.warning("A28 layout: undeclared inputs %s (ignored)", sorted(missing))
    return spec


def section(name: str, **extra) -> dict:
    """``{"c2c_section": name, ...extra}`` - shorthand for the common case."""
    return {"c2c_section": name, **extra}


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
