"""
BatchVersionManagerMEC – Shot/task hierarchy with atomic version reservation.

Layout:
    <root>/<show>/<shot>/<task>/v<NNN>/

Behaviour:
  - Discovers the next free version by scanning existing ``v###`` dirs.
  - When ``reserve=True``, atomically reserves the version by creating
    a `.lock` file using ``open(..., 'x')``. If the lock already
    exists (another process won the race), retries up to ``max_retries``
    with the next version number.
  - Always returns paths with forward slashes (vfx pipelines, render
    farms, and OS-agnostic asset DBs all expect this).

Outputs ``next_version_path`` (string), ``version_int``, and ``info_json``.
No pixel I/O; safe to call inside a workflow.
"""
from __future__ import annotations

import json
import logging
import re
import socket
import getpass
import hashlib
import datetime
from pathlib import Path

from ._is_changed_util import dir_version_fingerprint, hash_args_and_kwargs

logger = logging.getLogger("MEC.BatchVersionManager")

_VERSION_DIR_RE = re.compile(r"^v(\d{1,6})$")
_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_\-.]")


def _safe(token: str, fallback: str) -> str:
    """Sanitize a hierarchy token (show/shot/task)."""
    if not token or not token.strip():
        return fallback
    cleaned = _SAFE_NAME_RE.sub("_", token.strip())
    cleaned = cleaned.strip("._")
    return cleaned or fallback


def _scan_max_version(task_dir: Path) -> int:
    if not task_dir.is_dir():
        return 0
    max_v = 0
    for entry in task_dir.iterdir():
        if not entry.is_dir():
            continue
        m = _VERSION_DIR_RE.match(entry.name)
        if m:
            v = int(m.group(1))
            if v > max_v:
                max_v = v
    return max_v


class BatchVersionManagerMEC:
    """Allocate the next free ``v###`` directory under <root>/<show>/<shot>/<task>/."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "root": ("STRING", {
                    "default": "",
                    "tooltip": "Absolute output root (e.g. D:/projects/renders).",
                }),
                "show": ("STRING", {"default": "show", "tooltip": "Show / project name (top-level folder under root)"}),
                "shot": ("STRING", {"default": "sh010", "tooltip": "Shot identifier (folder under show)"}),
                "task": ("STRING", {"default": "comp", "tooltip": "Task name (folder under shot, e.g. comp, matte, render)"}),
            },
            "optional": {
                "reserve": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "Atomically reserve the version with a .lock file. "
                        "When False, only computes the path — no disk writes."
                    ),
                }),
                "padding": ("INT", {
                    "default": 3, "min": 1, "max": 6,
                    "tooltip": "Zero-pad width for v### (3 → v001, 4 → v0001).",
                }),
                "max_retries": ("INT", {
                    "default": 5, "min": 1, "max": 50,
                    "tooltip": "On lock-race contention, advance version and retry this many times.",
                }),
                "min_version": ("INT", {
                    "default": 1, "min": 1, "max": 999999,
                    "tooltip": "Floor for the first version when no v### exists yet.",
                }),
                "forward_slash": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "When True (default), output paths use forward slashes "
                        "for cross-platform compatibility. Set False to keep "
                        "native (Windows backslash) separators."
                    ),
                }),
                # Phase 3b v2 - Feature A: write a version_manifest.json
                # sidecar inside the reserved folder for full traceability.
                "write_manifest": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "When `reserve=True`, also write `version_manifest.json` "
                        "alongside the .lock containing workflow_hash + user + "
                        "host + timestamp + show/shot/task triple. Provides full "
                        "audit trail. Ignored when reserve=False."
                    ),
                }),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ("STRING", "INT", "STRING", "STRING")
    RETURN_NAMES = ("version_path", "version_int", "version_label", "info_json")
    OUTPUT_TOOLTIPS = (
        "Full path to the next-version directory (forward-slash by default).",
        "Integer version number that was allocated.",
        "Padded version label such as v001.",
        "JSON metadata: show, shot, task, user, host, timestamp, reservation status.",
    )
    FUNCTION = "allocate"
    CATEGORY = "C2C/IO"
    DESCRIPTION = (
        "Compute (and optionally atomically reserve) the next v### directory under "
        "<root>/<show>/<shot>/<task>/. Forward-slash output paths."
    )

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        h = hashlib.md5(hash_args_and_kwargs(**kwargs).encode())
        root = (kwargs.get("root") or "").strip()
        if root:
            show_s = _safe(kwargs.get("show", "show"), "show")
            shot_s = _safe(kwargs.get("shot", "sh010"), "shot")
            task_s = _safe(kwargs.get("task", "comp"), "comp")
            padding = int(kwargs.get("padding", 3))
            task_dir = Path(root).expanduser() / show_s / shot_s / task_s
            h.update(dir_version_fingerprint(task_dir, "v", padding).encode())
        return h.hexdigest()

    def allocate(
        self,
        root: str,
        show: str,
        shot: str,
        task: str,
        reserve: bool = False,
        padding: int = 3,
        max_retries: int = 5,
        min_version: int = 1,
        forward_slash: bool = True,
        write_manifest: bool = True,
        prompt=None,
        extra_pnginfo=None,
    ):
        if not root or not root.strip():
            raise ValueError("root path is required.")
        root_p = Path(root.strip()).expanduser()

        show_s = _safe(show, "show")
        shot_s = _safe(shot, "shot")
        task_s = _safe(task, "task")

        task_dir = root_p / show_s / shot_s / task_s
        next_v = max(_scan_max_version(task_dir) + 1, min_version)

        reserved = False
        attempts = 0
        target: Path = task_dir / f"v{next_v:0{padding}d}"
        if reserve:
            for attempts in range(max_retries):
                target = task_dir / f"v{next_v:0{padding}d}"
                try:
                    target.mkdir(parents=True, exist_ok=False)
                    lock = target / ".lock"
                    # 'x' = exclusive create; raises FileExistsError on race.
                    with open(lock, "x", encoding="utf-8") as fh:
                        fh.write(json.dumps({
                            "show": show_s, "shot": shot_s, "task": task_s,
                            "version": next_v,
                        }))
                    reserved = True
                    break
                except FileExistsError:
                    logger.info(
                        "[MEC] Version v%d already taken; advancing.", next_v,
                    )
                    next_v += 1
                except OSError as exc:
                    raise RuntimeError(
                        f"Failed to reserve version under {task_dir}: {exc}"
                    ) from exc
            else:
                # MANUAL bug-fix (Apr 2026): make exhaustion message actionable.
                raise OSError(
                    f"BatchVersionManagerMEC: {max_retries} retries failed for "
                    f"v{next_v} under {task_dir.as_posix()}. Another process is "
                    f"likely racing for the same task directory. Try increasing "
                    f"max_retries, or pick a unique show/shot/task triple."
                )

        label = f"v{next_v:0{padding}d}"
        # MANUAL bug-fix (Apr 2026): forward_slash toggle (default True for
        # back-compat). When False, paths come back with native separators.
        if forward_slash:
            path_out = target.as_posix()
            root_out = str(root_p).replace("\\", "/")
        else:
            path_out = str(target)
            root_out = str(root_p)
        info = {
            "root": root_out,
            "show": show_s,
            "shot": shot_s,
            "task": task_s,
            "version": next_v,
            "label": label,
            "reserved": reserved,
            "attempts": attempts + 1 if reserve else 0,
            "path": path_out,
        }
        # Phase 3b v2 - Feature A: write version_manifest.json sidecar.
        # Captured BEFORE the return so any IO problem gets logged but does
        # not poison the output payload.
        if reserve and reserved and write_manifest:
            try:
                wf_hash = ""
                if prompt is not None:
                    try:
                        canon = json.dumps(prompt, sort_keys=True, default=str).encode("utf-8")
                        wf_hash = hashlib.sha256(canon).hexdigest()
                    except (TypeError, ValueError) as exc:
                        logger.warning(
                            "[MEC] Workflow hash skipped (%s).", exc,
                        )
                manifest = {
                    "version": next_v,
                    "label": label,
                    "path": path_out,
                    "show": show_s,
                    "shot": shot_s,
                    "task": task_s,
                    "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "hostname": socket.gethostname(),
                    "user": getpass.getuser(),
                    "workflow_sha256": wf_hash,
                    "forward_slash": forward_slash,
                }
                manifest_path = target / "version_manifest.json"
                with open(manifest_path, "w", encoding="utf-8") as fh:
                    json.dump(manifest, fh, indent=2)
                info["manifest"] = (
                    manifest_path.as_posix() if forward_slash else str(manifest_path)
                )
                info["workflow_sha256"] = wf_hash
            except OSError as exc:
                logger.warning("[MEC] version_manifest.json write failed: %s", exc)
                info["manifest_error"] = str(exc)
        logger.info(
            "[MEC] BatchVersionManager %s/%s/%s -> %s (reserved=%s)",
            show_s, shot_s, task_s, label, reserved,
        )
        return (path_out, next_v, label, json.dumps(info, indent=2))


NODE_CLASS_MAPPINGS = {"BatchVersionManagerMEC": BatchVersionManagerMEC}
NODE_DISPLAY_NAME_MAPPINGS = {"BatchVersionManagerMEC": "Batch Version Manager"}
