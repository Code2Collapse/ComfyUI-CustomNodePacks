"""One version per run, shared by every output that belongs together.

THE BUG THIS FIXES. Put two Folder Version Incrementers in a workflow and one
run produces two versions - the .mov lands in v004 and the .exr in v005. Two
separate causes, both of them reasonable-looking in isolation:

  * `reserve_version=True` makes the first node CREATE its version directory.
    The second node then scans, sees it, and correctly reports the next one.
  * `suffix_mode="folder"` puts each suffix in its own TOP folder
    (`shot_a`, `shot_a_mask`), and each of those trees has its own independent
    v### counter. They drift apart over time and there is no run in which they
    agree except the first.

Neither is wrong on its own. What is wrong is that a single render's outputs -
the mov, the exr, the png, the masked mov - end up filed under different
version numbers, which makes them impossible to hand over as one deliverable.

THE FIX IS TWO PARTS, and the second is the one that matters.

1. A RUN-SCOPED LEASE. The first incrementer in an execution works out the
   version; every later one in the SAME execution, under the same key, is
   handed the same number instead of scanning again. ComfyUI gives each
   execution a `prompt_id`, which is what the lease is keyed on.

2. THE KEY IGNORES THE SUFFIX. This is what makes it correct rather than just
   consistent. `shot_a` and `shot_a_mask` are the same shot, so the suffix is
   stripped before keying and they share a lease. `shot_a` and `some_other_job`
   are not, so they keep separate counters and are never forced onto a shared
   number. The suffix is known exactly - it is a field on the node - so this is
   subtraction, not a guess.

AND THE VERSION IS THE MAX ACROSS THE FAMILY, which is what makes it safe in
any execution order. Suppose `shot_a` holds v001-v004 and `shot_a_mask` holds
only v001-v002. If the mask node happened to run first and leased "v003" from
its own tree, the beauty pass would then be told v003 - and would overwrite
existing work. So the lease scans every tree in the family and takes the
highest, giving v005 to all of them whichever node asks first. The mask tree
skips v003 and v004, which is correct: the version identifies the RENDER, not
how many times that particular output has been written.

Nothing here touches the filesystem beyond reading directory names.
"""

from __future__ import annotations

import threading
from collections import OrderedDict

# prompt_id -> {lease key: version}. An OrderedDict used as an LRU: a long
# session queues hundreds of prompts and every one of them would otherwise be
# remembered forever.
_LEASES: OrderedDict[str, dict[str, int]] = OrderedDict()
_LOCK = threading.Lock()

#: How many past executions to remember. Only the CURRENT run is ever read, so
#: this only needs to be big enough that a run's own nodes all find their
#: lease; the rest is slack for interleaved execution.
MAX_REMEMBERED_RUNS = 16


def current_run_id() -> str | None:
    """The id of the execution running right now, or None outside one.

    ComfyUI exposes this through a contextvar set around each node call. It is
    read defensively: on a build that does not have it, every call returns None
    and the lease degrades to "no sharing", which is the old behaviour rather
    than a crash.
    """
    try:
        from comfy_execution.utils import get_executing_context
    except Exception:  # noqa: BLE001 - older or stubbed ComfyUI
        return None
    try:
        ctx = get_executing_context()
    except Exception:  # noqa: BLE001
        return None
    return getattr(ctx, "prompt_id", None) if ctx is not None else None


def family_key(base_dir, folder_name: str, date_folder: str,
               suffix: str = "", group: str = "") -> str:
    """What decides whether two outputs share a version.

    An explicit `group` wins outright - it is the escape hatch for the case
    where two outputs belong together but their folder names do not say so.

    Otherwise the key is the base directory, the folder name WITH ITS SUFFIX
    REMOVED, and the date. Removing the suffix is the whole point: it is what
    puts `shot_a` and `shot_a_mask` on one lease while leaving an unrelated
    job on its own.
    """
    if group and group.strip():
        return "group:" + group.strip()
    stem = strip_suffix(folder_name, suffix)
    return f"{base_dir}|{stem}|{date_folder}"


def strip_suffix(folder_name: str, suffix: str) -> str:
    """`shot_a_mask` minus `_mask` -> `shot_a`.

    Only an exact trailing match is removed. A partial or fuzzy match would
    merge two families that merely look alike, and merging the wrong families
    is worse than not merging at all - it hands one shot's version number to
    another shot.
    """
    name = (folder_name or "").strip()
    suf = (suffix or "").strip()
    if suf and name.endswith(suf) and len(name) > len(suf):
        return name[:-len(suf)]
    return name


def scan_family_next(base_dir, folder_stem: str, date_folder: str,
                     prefix: str, padding: int, scan_next) -> int:
    """The next version across EVERY tree in the family.

    `shot_a`, `shot_a_mask` and `shot_a_png` are all scanned and the highest
    next-version wins, so the answer does not depend on which node ran first.
    Without this, a family whose trees are at different depths hands out a
    number that already exists in one of them - and the next render silently
    overwrites finished work.

    `scan_next` is injected so this module never imports the node file and can
    be tested with nothing on disk.
    """
    from pathlib import Path

    base = Path(base_dir)
    best = 1
    if not base.is_dir():
        return best

    stem = folder_stem.strip()
    for entry in base.iterdir():
        if not entry.is_dir():
            continue
        # The family is the exact stem plus anything suffixed onto it. A
        # startswith test on an EMPTY stem would match the whole output
        # directory, so that case is excluded.
        if not stem or not entry.name.startswith(stem):
            continue
        best = max(best, int(scan_next(entry / date_folder, prefix, padding)))
    return best


def lease(run_id: str | None, key: str, compute) -> tuple[int, bool]:
    """The version for this key in this run. Returns (version, was_shared).

    `was_shared` is True when an earlier node in the same run already decided
    it. That flag exists so the node can SAY so in its report: a user who set
    padding or a prefix on the second incrementer and saw it ignored needs to
    know the number came from somewhere else, not that their setting is broken.
    """
    if run_id is None:
        # No execution context - nothing to share within, so behave exactly as
        # the node did before leases existed.
        return int(compute()), False

    with _LOCK:
        run = _LEASES.get(run_id)
        if run is not None and key in run:
            _LEASES.move_to_end(run_id)
            return int(run[key]), True

    # Computed OUTSIDE the lock: it touches the filesystem, and holding a lock
    # across a directory scan would serialise every node in a parallel graph.
    value = int(compute())

    with _LOCK:
        run = _LEASES.setdefault(run_id, {})
        # Re-checked after the scan: another node may have leased the same key
        # while this one was reading the disk, and two different numbers for
        # one family is the exact bug this module exists to prevent.
        if key in run:
            _LEASES.move_to_end(run_id)
            return int(run[key]), True
        run[key] = value
        _LEASES.move_to_end(run_id)
        while len(_LEASES) > MAX_REMEMBERED_RUNS:
            _LEASES.popitem(last=False)
    return value, False


def describe(version: int, shared: bool, key: str, family: list[str]) -> str:
    """Why this number, in terms that answer 'why did I get two versions'."""
    lines = []
    if shared:
        lines.append(
            f"Version {version} was already decided by an earlier node in this "
            "same run, so this one reused it rather than taking the next "
            "number. That is what keeps the mov, the exr and the png of one "
            "render filed under one version.")
        lines.append(
            "Any prefix/padding set on THIS node was not used - the first "
            "incrementer in the run owns those.")
    else:
        lines.append(
            f"Version {version}: this is the first node in the run to ask, so "
            "it decided the number. Every other incrementer in the same family "
            "will now be handed the same one.")
    if family and len(family) > 1:
        lines.append(
            "Scanned the whole family so the answer does not depend on which "
            "node ran first: " + ", ".join(sorted(family)) + ".")
    return "\n".join(lines)


def reset_for_tests() -> None:
    """Drop every lease. Only for tests - a run must never call this."""
    with _LOCK:
        _LEASES.clear()
