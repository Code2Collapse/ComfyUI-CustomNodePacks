"""Promoted parameters for the C2C Vault.

WHY THE MAPPING IS INSIDE THE CIPHERTEXT
----------------------------------------
A promoted parameter has two halves:

  * which widget of which internal node it drives  -> SECRET
  * its label, type and current value              -> PUBLIC

They must be stored apart. Putting `{"node": "3", "widget": "seed"}` in the
clear-text manifest would tell a recipient there is a sampler in there, and
`{"node": "7", "widget": "lora_name"}` would name the LoRA stack - which is
most of what the vault exists to hide. So:

  ciphertext  subgraph["promoted"] = [{"id": "p0", "node": "3",
                                       "widget": "seed"}]
  manifest    vault_interface["params"] = [{"id": "p0", "name": "seed",
                                            "type": "INT", "value": 12345}]

The manifest leaks only what the owner chose to call the knob. `name` is a
label, not the internal widget name, and the owner can call it anything.

A promoted parameter can also be driven by a wire instead of a typed value:
`"socket": k` means "read param_k rather than `value`". That is the
convert-to-input half, and it reuses the same id, so flipping between the two
never re-points the parameter at a different widget.
"""

from __future__ import annotations

from typing import Any

MAX_VAULT_PARAMS = 8

# Types a promoted parameter may carry as a typed value. Anything else has to
# arrive over a wire, because we will not invent a widget for it.
VALUE_TYPES = {"INT", "FLOAT", "STRING", "BOOLEAN", "COMBO"}


class VaultPromotionError(RuntimeError):
    """Raised with a sentence a person can act on."""


def parse_params(interface: dict[str, Any]) -> list[dict[str, Any]]:
    """The public half, validated. Never trusts the manifest's shape."""
    raw = interface.get("params")
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise VaultPromotionError(
            "vault_interface.params must be a list of promoted parameters.")
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for i, p in enumerate(raw):
        if not isinstance(p, dict):
            raise VaultPromotionError(f"Promoted parameter {i} is not an object.")
        pid = str(p.get("id") or "").strip()
        if not pid:
            raise VaultPromotionError(f"Promoted parameter {i} has no id.")
        if pid in seen:
            raise VaultPromotionError(
                f"Two promoted parameters share the id {pid!r}. Ids address the "
                "widget inside the vault, so a duplicate would drive the wrong one.")
        seen.add(pid)
        entry: dict[str, Any] = {
            "id": pid,
            "name": str(p.get("name") or pid),
            "type": str(p.get("type") or "STRING").upper(),
            "value": p.get("value"),
        }
        sock = p.get("socket")
        if sock is not None:
            try:
                sock = int(sock)
            except (TypeError, ValueError):
                raise VaultPromotionError(
                    f"Promoted parameter {entry['name']!r} has a non-numeric socket.")
            if not 0 <= sock < MAX_VAULT_PARAMS:
                raise VaultPromotionError(
                    f"Promoted parameter {entry['name']!r} asks for param_{sock}, "
                    f"but the vault has param_0..param_{MAX_VAULT_PARAMS - 1}.")
            entry["socket"] = sock
        out.append(entry)
    return out


def parse_promoted(subgraph: dict[str, Any]) -> dict[str, tuple[str, str]]:
    """The secret half: id -> (node id, widget name)."""
    raw = subgraph.get("promoted") or []
    if not isinstance(raw, list):
        raise VaultPromotionError("Vault subgraph 'promoted' must be a list.")
    table: dict[str, tuple[str, str]] = {}
    for i, p in enumerate(raw):
        if not isinstance(p, dict):
            raise VaultPromotionError(f"Vault promotion {i} is not an object.")
        pid = str(p.get("id") or "").strip()
        node = str(p.get("node") or "").strip()
        widget = str(p.get("widget") or "").strip()
        if not (pid and node and widget):
            raise VaultPromotionError(
                f"Vault promotion {i} is incomplete; it needs id, node and widget.")
        table[pid] = (node, widget)
    return table


def resolve_overrides(
    subgraph: dict[str, Any],
    interface: dict[str, Any],
    socket_values: dict[int, Any],
) -> dict[str, dict[str, Any]]:
    """Fold promoted parameters into {node_id: {widget: value}}.

    `socket_values` is {k: value} for whatever param_k sockets are connected.
    Raises rather than guessing: a parameter the vault does not recognise, or a
    wired parameter with nothing on the wire, is a wiring mistake the person
    needs to see.
    """
    table = parse_promoted(subgraph)
    params = parse_params(interface)
    overrides: dict[str, dict[str, Any]] = {}

    for p in params:
        pid = p["id"]
        if pid not in table:
            raise VaultPromotionError(
                f"The vault has no parameter {p['name']!r} ({pid}). The manifest "
                "and the ciphertext disagree - re-lock the vault to rebuild both.")
        if "socket" in p:
            k = p["socket"]
            if k not in socket_values or socket_values[k] is None:
                raise VaultPromotionError(
                    f"Parameter {p['name']!r} is set to take a wire, but nothing "
                    f"is connected to param_{k}.")
            value = socket_values[k]
        else:
            if p["type"] not in VALUE_TYPES:
                raise VaultPromotionError(
                    f"Parameter {p['name']!r} is a {p['type']}, which cannot be "
                    "typed in - connect it to param_N instead.")
            value = p["value"]
        node, widget = table[pid]
        overrides.setdefault(node, {})[widget] = value

    return overrides


def describe(interface: dict[str, Any]) -> str:
    """One line per promoted parameter, for the node's report."""
    try:
        params = parse_params(interface)
    except VaultPromotionError as exc:
        return f"Promoted parameters could not be read: {exc}"
    if not params:
        return "No promoted parameters."
    lines = [f"{len(params)} promoted parameter(s):"]
    for p in params:
        if "socket" in p:
            lines.append(f"  {p['name']} <- param_{p['socket']} (wired)")
        else:
            lines.append(f"  {p['name']} = {p['value']!r}")
    return "\n".join(lines)
