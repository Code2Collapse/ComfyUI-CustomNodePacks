"""Load ComfyUI-CustomNodePacks exactly as ComfyUI does and export C2C_LOAD_SUMMARY.

Leading underscore so pytest does not collect this module.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO = Path(__file__).resolve().parents[1]
MARKER_BEGIN = "<!-- C2C:NODE-COUNTS:BEGIN -->"
MARKER_END = "<!-- C2C:NODE-COUNTS:END -->"


def _resolve_comfy_core() -> Path:
    candidates: List[Path] = []
    for env_name in ("COMFYUI_PATH", "C2C_COMFY_CORE"):
        raw = os.environ.get(env_name, "").strip()
        if raw:
            candidates.append(Path(raw))
    candidates.extend(
        (
            REPO.parent.parent / "ComfyUI_windows_portable" / "ComfyUI",
            REPO / "third_party" / "ComfyUI",
        )
    )
    for path in candidates:
        if (path / "main.py").is_file() and (path / "comfy").is_dir():
            return path.resolve()
    raise RuntimeError(
        "ComfyUI core not found. Set COMFYUI_PATH or C2C_COMFY_CORE, or install "
        "ComfyUI_windows_portable beside Custom_Nodes."
    )


def _stub_prompt_server() -> None:
    """Mirror tests/test_cross_pack_no_duplicate_ids.py _stub_prompt_server."""
    try:
        from server import PromptServer
    except Exception:
        return
    if getattr(PromptServer, "instance", None) is not None:
        return

    class _Routes:
        def __getattr__(self, _name):
            def _decorator(*_a, **_k):
                def _wrap(fn):
                    return fn

                return _wrap

            return _decorator

    class _Stub:
        routes = _Routes()

    PromptServer.instance = _Stub()  # type: ignore[attr-defined]


def load_pack_summary() -> Dict[str, Any]:
    core = _resolve_comfy_core()
    saved_path = list(sys.path)
    if str(core) not in sys.path:
        sys.path.insert(0, str(core))
    _stub_prompt_server()
    sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != REPO]
    name = str(REPO).replace(".", "_x_")
    try:
        spec = importlib.util.spec_from_file_location(
            name, REPO / "__init__.py", submodule_search_locations=[str(REPO)]
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not load spec for {REPO / '__init__.py'}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        summary = getattr(mod, "C2C_LOAD_SUMMARY", None)
        if not isinstance(summary, dict):
            raise RuntimeError("C2C_LOAD_SUMMARY missing after pack load")
        return summary
    finally:
        sys.path[:] = saved_path


def render_log_line(summary: Dict[str, Any]) -> str:
    fam_part = ", ".join(f"{label} {n}" for label, n in summary["families"])
    n_failed = len(summary["failed"])
    line = (
        f"[C2C] CustomNodePacks: {summary['total']} nodes loaded "
        f"({fam_part}) - {n_failed} failed"
    )
    if summary["failed"]:
        fail_part = "; ".join(
            f"{rec['key']} ({rec['error'][:80]})" for rec in summary["failed"]
        )
        line += f" : {fail_part}"
    return line


def render_counts_block(summary: Dict[str, Any]) -> str:
    lines = [
        f"**ComfyUI-CustomNodePacks** registers **{summary['total']}** nodes "
        "across these families:",
        "",
        "| Family | Nodes |",
        "|--------|------:|",
    ]
    for label, n in summary["families"]:
        lines.append(f"| {label} | {n} |")
    lines.extend(
        [
            "",
            "Startup log line to verify the pack loaded:",
            "",
            f"`{render_log_line(summary)}`",
            "",
        ]
    )
    return "\n".join(lines)


def _replace_marker_block(path: Path, block: str) -> None:
    text = path.read_text(encoding="utf-8")
    begin = text.find(MARKER_BEGIN)
    end = text.find(MARKER_END)
    if begin < 0 or end < 0 or end < begin:
        raise RuntimeError(f"Marker pair missing or out of order in {path}")
    new_text = (
        text[: begin + len(MARKER_BEGIN)]
        + "\n"
        + block.rstrip()
        + "\n"
        + text[end:]          # starts AT the END marker, so the marker survives
    )
    path.write_text(new_text, encoding="utf-8", newline="\n")


def write_docs(summary: Dict[str, Any]) -> None:
    block = render_counts_block(summary)
    _replace_marker_block(REPO / "README.md", block)
    _replace_marker_block(REPO / "NODE_REFERENCE.md", block)


def main(argv: List[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Export C2C_LOAD_SUMMARY from live pack load.")
    parser.add_argument(
        "--write-docs",
        action="store_true",
        help="Rewrite README.md and NODE_REFERENCE.md marker blocks.",
    )
    args = parser.parse_args(argv)
    summary = load_pack_summary()
    if args.write_docs:
        write_docs(summary)
    print(f"C2C_LOAD_SUMMARY_JSON={json.dumps(summary, ensure_ascii=True)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
