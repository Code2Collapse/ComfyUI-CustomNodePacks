"""A28 (ORDERS 29) layout declarations: merge the declarative visibility keys into a node's INPUT_TYPES.

The keys (c2c_section, c2c_show_if, c2c_show_if_any, c2c_hide_if_linked, c2c_show_if_linked, c2c_hidden,
c2c_show_if_status) are read only by js/_c2c_node_layout.js. Contract: docs/audit/A28/README.md section 2.

Display only: every input keeps its name, order, type, default and options; a hidden input is still saved and still
reaches the node's function with its value. Core's ``advanced`` flag is dropped from an input placed in a c2c_section
(README section 3: Nodes 2.0 would otherwise fold it twice, and the engine leaves such a node alone).
"""
from __future__ import annotations

import logging
from typing import Dict

LOG = logging.getLogger("C2C.Layout")

KEYS = ("c2c_section", "c2c_show_if", "c2c_show_if_any", "c2c_hide_if_linked", "c2c_show_if_linked",
        "c2c_hidden", "c2c_show_if_status")


def with_layout(spec: dict, layout: Dict[str, dict]) -> dict:
    """Merge ``layout`` (input name -> rule keys) into ``spec`` (an INPUT_TYPES dict) in place and return it.

    Idempotent (INPUT_TYPES may be called many times on the same cached dict). A name that is not declared is logged,
    never raised: a typo must not take the node (or the pack) down.
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
            # the options dict is COPIED: a spec dict another input or class shares never picks up these keys
            has_opts = isinstance(s, (tuple, list)) and len(s) >= 2 and isinstance(s[1], dict)
            opts = dict(s[1]) if has_opts else {}
            opts.update(rule)
            if "c2c_section" in rule:
                opts.pop("advanced", None)
            head = s[0] if isinstance(s, (tuple, list)) and s else s
            tail = tuple(s[2:]) if isinstance(s, (tuple, list)) and len(s) > 2 else ()
            inputs[name] = (head, opts, *tail)
    missing = set(layout) - seen
    if missing:
        LOG.warning("A28 layout: undeclared inputs %s (ignored)", sorted(missing))
    return spec


def section(name: str, **extra) -> dict:
    """``{"c2c_section": name, ...extra}`` - shorthand for the common case."""
    return {"c2c_section": name, **extra}
