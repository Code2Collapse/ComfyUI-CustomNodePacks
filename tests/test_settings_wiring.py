"""Every setting has a control, and every control does something.

Two failures this pins, both invisible:

  a setting READ but never REGISTERED has no control, so whatever its default
  is, that is what the user gets forever. `c2c.inspector.ai_widget_blurb` was
  in that state - it gates a NETWORK CALL, defaulted to on, and there was no
  way to say no.

  a setting REGISTERED but never READ is a control that does nothing. The user
  toggles it, nothing changes, and they reasonably conclude the pack is broken.

Getting this right took four passes, because the naive version of the check is
wrong in four different ways and every one of them produced confident false
positives:

  1. the accessor and the id are routinely on DIFFERENT LINES -
       getSettingValue?.(
           "c2c.statsPill.placement", "manager")
     a single-line regex called 27 live settings dead.
  2. a file may read through its own wrapper - c2c_node_explain.js has
     `_getSetting(id, def)` - so the accessor never appears next to the id.
  3. ids get BUILT: `getSettingValue("c2c.completion_fx." + id)` covers six
     settings with one call site and matches none of them literally.
  4. `onChange` in the registration IS consumption. The theme variant is
     applied entirely from its onChange and is never read back.

So the check below allows all four. It is deliberately generous: a false
negative here costs nothing, while a false positive sends someone hunting for
a bug that is not there - which is exactly what the first three versions did.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

WORKSPACE = Path(__file__).resolve().parents[2]
PACKS = [
    "ComfyUI-CustomNodePacks", "ComfyUI-NukeMaxNodes",
    "ComfyUI-WanNodeExperiments", "ComfyUI-MiniMaxSuite",
    "ComfyUI-WanAnimatePreprocessV2", "ComfyUI-GLM_Image",
]
SKIP = {"third_party", "node_modules", ".git", "__pycache__"}

# Both halves may wrap, and the accessor may be optional-chained.
READ = re.compile(
    r'(?:getSettingValue|setSettingValue|settingStore\.\w+)\s*\??\.?\s*\(\s*'
    r'["\']([\w.\-]+)["\']')
# An id assembled from a literal prefix plus something else - one call site
# standing in for a whole family.
DYNAMIC = re.compile(
    r'(?:getSettingValue|setSettingValue)\s*\??\.?\s*\(\s*'
    r'(?:["\']([\w.\-]*\.)["\']\s*\+|`|[A-Za-z_$][\w$]*\s*[,)])')


def js_files():
    for pack in PACKS:
        base = WORKSPACE / pack
        if not base.exists():
            continue
        for f in base.rglob("*.js"):
            if SKIP & set(f.parts) or f.name.endswith(".min.js"):
                continue
            if not ({"js", "web"} & set(f.parts)):
                continue
            yield pack, f


def _settings_arrays(text: str):
    """Each `settings: [ ... ]` literal, brace-matched rather than regexed -
    these arrays contain nested objects and a lazy regex stops at the first
    `]` inside an options list."""
    for m in re.finditer(r'settings:\s*\[', text):
        depth, i = 0, m.end() - 1
        while i < len(text):
            if text[i] == "[":
                depth += 1
            elif text[i] == "]":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        yield text[m.end():i]


def _entries(block: str):
    """(id, body) for each top-level object in a settings array."""
    out, depth, start = [], 0, None
    for i, ch in enumerate(block):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                out.append(block[start:i + 1])
                start = None
    for body in out:
        m = re.search(r'\bid:\s*["\']([\w.\-]+)["\']', body)
        if m:
            yield m.group(1), body


def scan():
    registered: dict[str, list[tuple[str, bool]]] = {}
    read: dict[str, set[str]] = {}
    prefixes: set[str] = set()
    wrapper_files: set[str] = set()
    for pack, f in js_files():
        text = f.read_text(encoding="utf-8", errors="ignore")
        rel = f"{pack}/{f.name}"
        for block in _settings_arrays(text):
            for sid, body in _entries(block):
                registered.setdefault(sid, []).append((rel, "onChange" in body))
        for sid in READ.findall(text):
            read.setdefault(sid, set()).add(rel)
        for m in DYNAMIC.finditer(text):
            if m.group(1):
                prefixes.add(m.group(1))
            wrapper_files.add(rel)
    return registered, read, prefixes, wrapper_files


REGISTERED, READ_IDS, PREFIXES, DYNAMIC_FILES = scan()


def _consumed(sid: str, places: list[tuple[str, bool]]) -> bool:
    if sid in READ_IDS:
        return True
    if any(on_change for _, on_change in places):
        return True
    if any(sid.startswith(p) for p in PREFIXES):
        return True
    # A file that reads through its own wrapper, or builds ids, covers the
    # settings it registers.
    return bool({f for f, _ in places} & DYNAMIC_FILES)


def test_the_scan_found_something():
    """A regex that silently matches nothing turns this whole file into a
    test that always passes."""
    assert len(REGISTERED) > 40, f"only found {len(REGISTERED)} settings"
    assert len(READ_IDS) > 20, f"only found {len(READ_IDS)} reads"


def test_every_setting_that_is_read_has_a_control():
    """Read but never registered: no UI exists, so the default is permanent.
    That is how c2c.inspector.ai_widget_blurb ended up making network calls
    with no way to decline."""
    orphans = {}
    for sid, files in READ_IDS.items():
        if sid in REGISTERED:
            continue
        # A bare prefix is the literal half of a built id, not a setting.
        if sid.endswith("."):
            continue
        # ComfyUI's own settings are read by us and registered by ComfyUI.
        if sid.split(".")[0] in {"Comfy", "LiteGraph", "pysssss", "Kj"}:
            continue
        orphans[sid] = sorted(files)
    assert not orphans, (
        "read with no control, so the default can never be changed: "
        + "; ".join(f"{k} (in {', '.join(v)})" for k, v in sorted(orphans.items())))


def test_every_control_does_something():
    """Registered but never consumed - the user toggles it and nothing
    happens, which reads as a broken pack."""
    inert = {sid: sorted({f for f, _ in places})
             for sid, places in REGISTERED.items()
             if not _consumed(sid, places)}
    assert not inert, (
        "these controls are inert: "
        + "; ".join(f"{k} ({', '.join(v)})" for k, v in sorted(inert.items())))


def test_shared_settings_are_registered_behind_a_guard():
    """Three settings are registered by more than one pack, because three
    packs ship their own copy of the theme and the i18n module. Whichever
    loads last would win and re-register, which ComfyUI logs as a duplicate.
    A module-level guard makes the second and third registration a no-op."""
    shared = {sid: sorted({f for f, _ in places})
              for sid, places in REGISTERED.items()
              if len({f.split("/")[0] for f, _ in places}) > 1}
    assert shared, "expected the theme settings to be shared across packs"
    for sid, files in shared.items():
        for rel in files:
            pack, name = rel.split("/", 1)
            matches = list((WORKSPACE / pack).rglob(name))
            assert matches, f"{rel} vanished"
            src = matches[0].read_text(encoding="utf-8", errors="ignore")
            # The guard need not be the FIRST condition: the i18n module gates
            # on a satellite flag before the window flag. Requiring
            # `if (!window.__X__)` verbatim reported a guarded file as
            # unguarded.
            assert re.search(r'!\s*window\.__\w+__', src), (
                f"{rel} registers {sid} with no duplicate guard - two packs "
                "registering the same id both run")


@pytest.mark.parametrize("field", ["name", "type"])
def test_every_control_is_presentable(field):
    """A setting with no name shows as a blank row; one with no type falls
    back to a text box, so a boolean becomes a field you type 'true' into."""
    missing = []
    for pack, f in js_files():
        text = f.read_text(encoding="utf-8", errors="ignore")
        for block in _settings_arrays(text):
            for sid, body in _entries(block):
                if not re.search(rf'\b{field}:\s*', body):
                    missing.append(f"{sid} ({pack}/{f.name})")
    assert not missing, f"settings with no {field}: " + ", ".join(sorted(missing))


def test_every_control_has_a_default():
    """Without one the control opens empty and the first read returns
    undefined, which every call site then has to guess a fallback for -
    and they will not all guess the same one."""
    missing = []
    for pack, f in js_files():
        text = f.read_text(encoding="utf-8", errors="ignore")
        for block in _settings_arrays(text):
            for sid, body in _entries(block):
                if "defaultValue" not in body:
                    missing.append(f"{sid} ({pack}/{f.name})")
    assert not missing, "settings with no defaultValue: " + ", ".join(sorted(missing))


def test_combo_defaults_are_one_of_the_options():
    """A default outside the list leaves the control showing nothing, and the
    user cannot get back to it once they change it."""
    bad = []
    for pack, f in js_files():
        text = f.read_text(encoding="utf-8", errors="ignore")
        for block in _settings_arrays(text):
            for sid, body in _entries(block):
                if not re.search(r'type:\s*["\']combo["\']', body):
                    continue
                opts = re.search(r'options:\s*\[(.*?)\]', body, re.S)
                dflt = re.search(r'defaultValue:\s*["\']([^"\']+)["\']', body)
                if not opts or not dflt:
                    continue
                values = re.findall(r'(?:value:\s*)?["\']([^"\']+)["\']', opts.group(1))
                if dflt.group(1) not in values:
                    bad.append(f"{sid}: default {dflt.group(1)!r} not in {values}")
    assert not bad, "; ".join(bad)
