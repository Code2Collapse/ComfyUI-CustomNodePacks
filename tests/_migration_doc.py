"""Generate docs/MIGRATION.md from the legacy replacement table and compat nodes."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
TABLE_PATH = REPO / "nodes" / "_legacy_replacements.json"
REMOVED_PATH = REPO / "nodes" / "_removed_nodes.json"
LEGACY_LIMITATION = (
    "Known core limitation (0.36.0): ``apply_replacements`` indexes "
    "``node_struct[\"inputs\"][old_id]`` directly, so an API prompt that omits "
    "an optional input of a legacy node raises KeyError (HTTP 500) instead of "
    'the 400 "node not found" it would get without a replacement - it failed '
    "either way."
)


def _load_table() -> list[dict[str, Any]]:
    with open(TABLE_PATH, encoding="utf-8") as f:
        return json.load(f)


def _load_removed() -> list[dict[str, Any]]:
    if not REMOVED_PATH.is_file():
        return []
    with open(REMOVED_PATH, encoding="utf-8") as f:
        return json.load(f)


def _translations(row: dict[str, Any]) -> str:
    out = []
    for v in row.get("c2c_values") or []:
        if "set" in v:
            out.append(f"{v['new_id']}={v['set']!r}")
        elif "map" in v:
            pairs = ", ".join(f"{k}→{val}" for k, val in v["map"].items())
            out.append(f"{v['old_id']}→{v['new_id']} ({pairs})")
        elif v.get("fn") == "log2":
            clamp = f", clamped to {v['clamp'][0]:g}..{v['clamp'][1]:g}" if v.get("clamp") else ""
            out.append(f"{v['old_id']}→{v['new_id']} (log2: multiplier to stops{clamp})")
    return "; ".join(out) or "—"


def _legacy_compat_rows() -> list[tuple[str, str, str]]:
    from nodes.legacy_compat import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

    rows: list[tuple[str, str, str]] = []
    for node_id, cls in NODE_CLASS_MAPPINGS.items():
        display = NODE_DISPLAY_NAME_MAPPINGS.get(node_id, node_id)
        desc = getattr(cls, "DESCRIPTION", "")
        rows.append((node_id, display, desc))
    return rows


def generate_migration_markdown() -> str:
    table = _load_table()
    lines = [
        "# Node migration guide (legacy → unified)",
        "",
        f"{len(table)} node ids have a unified successor. ComfyUI core's "
        "NodeReplaceManager (and this pack's server registration) maps saved "
        "workflows and API prompts from the old id to the new one. Core only "
        "rewrites an id that is no longer registered, so rows for deprecated "
        "nodes that still load change nothing until those nodes are removed.",
        "",
        "When a workflow loads, the pack migrates every node whose id is gone "
        "automatically (setting: C2C › Workflows › Migrate old C2C nodes on "
        "load; off = ComfyUI offers the replacement instead). Links and values "
        "move to the successor; one notice lists what was updated, what changes "
        "on the successor, and anything that could not be carried over - "
        "nothing is dropped silently. A successor from another pack (core, "
        "NukeMax) has to be installed; otherwise the node is left as it was "
        "and the notice says so.",
        "",
        "## Merged nodes (replacement table)",
        "",
        "| Old id | New id | Mode / pinned | Renamed inputs | Translated values | Dropped | Output remap | Note |",
        "|--------|--------|---------------|----------------|-------------------|---------|--------------|------|",
    ]

    for row in table:
        old_id = row["old_node_id"]
        new_id = row["new_node_id"]
        pinned = []
        renamed = []
        mapped_new_ids = set()
        for entry in row.get("input_mapping") or []:
            new_id_key = entry.get("new_id")
            if new_id_key:
                mapped_new_ids.add(new_id_key)
            if "set_value" in entry:
                pinned.append(f"{new_id_key}={entry['set_value']!r}")
            elif entry.get("old_id") != new_id_key:
                renamed.append(f"{entry.get('old_id')}→{new_id_key}")
        dropped = ", ".join(row.get("dropped") or []) or "—"
        out_remap = ", ".join(
            f"{m['old_idx']}→{m['new_idx']}" for m in (row.get("output_mapping") or [])
        ) or "—"
        note = " ".join(x for x in ((row.get("note") or "").strip(),
                                    ("What changes: " + row["c2c_note"]) if row.get("c2c_note") else "") if x) or "—"
        lines.append(
            f"| {old_id} | {new_id} | {', '.join(pinned) or '—'} | "
            f"{', '.join(renamed) or '—'} | {_translations(row)} | {dropped} | {out_remap} | {note} |"
        )

    removed = _load_removed()
    if removed:
        lines.extend([
            "",
            f"## Removed nodes with no successor ({len(removed)})",
            "",
            "A loading workflow drops these nodes and their links, and the notice "
            "names each one.",
            "",
            "| Node id | Where the information is now | Note |",
            "|---------|------------------------------|------|",
        ])
        for r in removed:
            lines.append(f"| {r['old_node_id']} | {r.get('c2c_note', '—')} | {r.get('note', '—')} |")

    lines.extend([
        "",
        "## Deprecated nodes kept as-is (7)",
        "",
        "These classes have no unified successor. They are registered with "
        "`DEPRECATED = True` (hidden from search; old workflows still load).",
        "",
        "| Node id | Display name | Description |",
        "|---------|--------------|-------------|",
    ])
    for node_id, display, desc in _legacy_compat_rows():
        lines.append(f"| {node_id} | {display} | {desc} |")

    lines.extend([
        "",
        "## Core limitation",
        "",
        LEGACY_LIMITATION,
        "",
    ])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate docs/MIGRATION.md")
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write docs/MIGRATION.md",
    )
    args = parser.parse_args(argv)
    text = generate_migration_markdown()
    if args.write:
        out = REPO / "docs" / "MIGRATION.md"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8", newline="\n")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
