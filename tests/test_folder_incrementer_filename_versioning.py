"""Folder Version Incrementer: filename versioning, version-in-filename, custom output name (L7.94, owner 2026-10-12).

  folder   (default)  shot_A/<date>/v001/shot_A.png          - unchanged with the new options off
           + version_in_filename + output_name=final_render -> shot_A/<date>/v001/final_render_v001.png
  filename            shot_A/<date>/shot_A_v001.png          - no v001 folder, next number no file of this name has

The savers are not importable here (comfy is stubbed); what they write is reproduced from their code:
core SaveImage / VHS add their own counter after the prefix (shot_A_v001_00001_.png, shot_A_v001_00001.mp4),
Save Video (C2C) with "Folder Version" naming writes <subfolder>/<basename(prefix)>.<ext> or a sequence folder
<subfolder>/<basename(prefix)>/. The live run through the real savers is docs/evidence/L7.94.
"""
from __future__ import annotations

import importlib.util
import json
from datetime import datetime
from pathlib import Path

import pytest

NAMES = ("version_string", "version_number", "folder_name", "subfolder_path", "filename_prefix", "output_filename",
         "source_stem", "current_version_token", "next_version_token", "next_version_stem", "version_path",
         "info_json")


def _load():
    root = Path(__file__).resolve().parents[1]
    import sys
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    spec = importlib.util.spec_from_file_location("folder_incrementer_l794", root / "folder_incrementer.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def fi():
    return _load()


@pytest.fixture
def out(fi, tmp_path, monkeypatch):
    monkeypatch.setattr(fi, "_get_output_dir", lambda: str(tmp_path))
    return tmp_path


@pytest.fixture
def day(fi):
    return datetime.now().strftime(fi.DATE_FORMAT_MAP["MM-DD-YYYY"])


def run(fi, **kw):
    kw.setdefault("path_style", "linux")
    kw.setdefault("source_choice", "custom")
    return dict(zip(NAMES, fi.FolderIncrementer().increment(**kw)))


def touch(path: Path, is_dir=False):
    if is_dir:
        path.mkdir(parents=True, exist_ok=True)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")


def test_the_new_inputs_are_optional_and_last(fi):
    req = fi.FolderIncrementer.INPUT_TYPES()
    opt = list(req["optional"])
    assert opt[-3:] == ["versioning", "version_in_filename", "output_name"]   # saved widget values keep positions
    assert req["optional"]["versioning"][0] == ["folder", "filename"]
    assert req["optional"]["versioning"][1]["default"] == "folder"
    assert req["optional"]["version_in_filename"][1]["default"] is False
    assert req["optional"]["output_name"][1]["default"] == ""


# ── Method A: folder versioning ───────────────────────────────────────────────────────────────────────────────────

def test_folder_default_is_the_old_path(fi, out, day):
    r = run(fi, custom_name="shot_A.png")
    assert r["subfolder_path"] == f"shot_A/{day}/v001"
    assert r["filename_prefix"] == f"shot_A/{day}/v001/shot_A"
    assert r["output_filename"] == f"shot_A/{day}/v001/shot_A.png"
    assert "versioning" not in json.loads(r["info_json"])           # the default output string is unchanged


def test_folder_explicit_defaults_equal_omitted(fi, out):
    a = run(fi, custom_name="shot_A.png", suffix="_mask")
    b = run(fi, custom_name="shot_A.png", suffix="_mask", versioning="folder", version_in_filename=False,
            output_name="")
    assert a == b


def test_folder_version_in_filename(fi, out, day):
    r = run(fi, custom_name="shot_A.png", version_in_filename=True)
    assert r["output_filename"] == f"shot_A/{day}/v001/shot_A_v001.png"
    assert r["filename_prefix"] == f"shot_A/{day}/v001/shot_A_v001"


def test_folder_output_name_and_version_owner_example(fi, out, day):
    r = run(fi, custom_name="shot_A.png", version_in_filename=True, output_name="final_render")
    assert r["folder_name"] == "shot_A"                               # the folder keeps the source name
    assert r["output_filename"] == f"shot_A/{day}/v001/final_render_v001.png"


def test_folder_output_name_alone_and_with_its_own_extension(fi, out, day):
    assert run(fi, custom_name="shot_A.png", output_name="final_render")["output_filename"] == \
        f"shot_A/{day}/v001/final_render.png"
    assert run(fi, custom_name="shot_A.png", output_name="final_render.mov")["output_filename"] == \
        f"shot_A/{day}/v001/final_render.mov"


def test_folder_version_in_filename_without_a_source_does_not_double(fi, out, day):
    r = run(fi, label="proj", version_in_filename=True)
    assert r["filename_prefix"] == f"proj/{day}/v001/v001"            # as before: the version IS the name


def test_folder_still_counts_folders_not_files(fi, out, day):
    touch(out / "shot_A" / day / "shot_A_v004.png")                   # a filename-versioned file is not a version dir
    assert run(fi, custom_name="shot_A.png")["version_string"] == "v001"
    touch(out / "shot_A" / day / "v001", is_dir=True)
    assert run(fi, custom_name="shot_A.png")["version_string"] == "v002"


# ── Method B: filename versioning ─────────────────────────────────────────────────────────────────────────────────

def test_filename_basic_no_version_folder(fi, out, day):
    r = run(fi, custom_name="shot_A.png", versioning="filename")
    assert r["subfolder_path"] == f"shot_A/{day}"
    assert r["filename_prefix"] == f"shot_A/{day}/shot_A_v001"
    assert r["output_filename"] == f"shot_A/{day}/shot_A_v001.png"
    assert r["version_path"].replace("\\", "/").endswith(f"shot_A/{day}")
    info = json.loads(r["info_json"])
    assert info["versioning"] == "filename" and info["file"] == "shot_A_v001.png"
    assert not (out / "shot_A" / day / "v001").exists()


@pytest.mark.parametrize("ext", [".png", ".mp4", ".exr", ".wav", ".webm"])
def test_filename_keeps_every_media_extension(fi, out, day, ext):
    assert run(fi, custom_name=f"shot_A{ext}", versioning="filename")["output_filename"] == \
        f"shot_A/{day}/shot_A_v001{ext}"


def test_filename_next_free_number_over_every_saver_naming(fi, out, day):
    d = out / "shot_A" / day
    touch(d / "shot_A_v001.png")                     # Save Video (C2C) Folder Version
    touch(d / "shot_A_v002_00001_.png")              # core SaveImage counter, batch frame 1
    touch(d / "shot_A_v002_00002_.png")              # ... frame 2
    touch(d / "shot_A_v003_00001.mp4")               # VHS counter, another extension
    touch(d / "shot_A_v004", is_dir=True)            # a Folder Version image sequence
    touch(d / "shot_AB_v009.png")                    # another name: does not count
    touch(d / "shot_A.final.png")                    # no version token: does not count
    assert run(fi, custom_name="shot_A.png", versioning="filename")["version_string"] == "v005"


def test_filename_counts_folder_method_versions_too(fi, out, day):
    touch(out / "shot_A" / day / "v007", is_dir=True)                 # switching methods never reuses a number
    assert run(fi, custom_name="shot_A.png", versioning="filename")["version_string"] == "v008"


def test_filename_each_output_name_has_its_own_counter(fi, out, day):
    d = out / "shot_A" / day
    touch(d / "shot_A_v003.png")
    assert run(fi, custom_name="shot_A.png", versioning="filename")["version_string"] == "v004"
    r = run(fi, custom_name="shot_A.png", versioning="filename", output_name="beauty")
    assert r["output_filename"] == f"shot_A/{day}/beauty_v001.png"
    touch(d / "beauty_v001.png")
    assert run(fi, custom_name="shot_A.png", versioning="filename", output_name="beauty")["version_string"] == "v002"


def test_filename_multiple_runs_write_v001_v002_v003(fi, out, day):
    got = []
    for _ in range(3):
        r = run(fi, custom_name="clip.mov", versioning="filename")
        got.append(r["output_filename"])
        touch(out / r["output_filename"])            # what the saver writes
    assert got == [f"clip/{day}/clip_v001.mov", f"clip/{day}/clip_v002.mov", f"clip/{day}/clip_v003.mov"]


def test_filename_never_duplicates_a_version_token(fi, out, day):
    r = run(fi, custom_name="shot_A_v003.mov", versioning="filename")
    assert r["folder_name"] == "shot_A_v003"                          # the folder is still source-derived
    assert r["output_filename"] == f"shot_A_v003/{day}/shot_A_v001.mov"
    r = run(fi, custom_name="shot_A.png", versioning="filename", output_name="comp_v012.exr")
    assert r["output_filename"] == f"shot_A/{day}/comp_v001.exr"
    r = run(fi, custom_name="shot_A_v003.mov", version_in_filename=True)
    assert r["output_filename"] == f"shot_A_v003/{day}/v001/shot_A_v001.mov"


def test_filename_without_a_source_is_named_after_the_folder(fi, out, day):
    r = run(fi, label="proj", versioning="filename")
    assert r["filename_prefix"] == f"proj/{day}/proj_v001"
    assert r["output_filename"] == r["filename_prefix"]                # no extension known: same as before


def test_filename_prefix_and_padding(fi, out, day):
    r = run(fi, custom_name="shot_A.png", versioning="filename", prefix="ver", padding=4)
    assert r["output_filename"] == f"shot_A/{day}/shot_A_ver0001.png"


@pytest.mark.parametrize("mode,expected", [
    ("filename", "shot_A/{day}/shot_A_mask_v001.png"),
    ("subfolder", "shot_A/{day}/mask/shot_A_v001.png"),
    ("folder", "shot_A_mask/{day}/shot_A_v001.png"),
])
def test_filename_suffix_modes(fi, out, day, mode, expected):
    r = run(fi, custom_name="shot_A.png", versioning="filename", suffix="_mask", suffix_mode=mode)
    assert r["output_filename"] == expected.format(day=day)


def test_filename_suffix_family_shares_the_highest_number(fi, out, day):
    touch(out / "shot_A" / day / "shot_A_v002.png")
    r = run(fi, custom_name="shot_A.png", versioning="filename", suffix="_mask")
    assert r["output_filename"] == f"shot_A/{day}/shot_A_mask_v003.png"
    touch(out / "shot_A" / day / "wan" / "shot_A_v006.png")           # subfolder mode, one level down
    r = run(fi, custom_name="shot_A.png", versioning="filename", suffix="_mask", suffix_mode="subfolder")
    assert r["output_filename"] == f"shot_A/{day}/mask/shot_A_v007.png"
    touch(out / "shot_A_mask" / day / "shot_A_v009.png")              # folder mode: the other tree of the family
    assert run(fi, custom_name="shot_A.png", versioning="filename")["version_string"] == "v010"


def test_filename_reserve_claims_the_number_without_a_folder(fi, out, day):
    a = run(fi, custom_name="shot_A.png", versioning="filename", reserve_version=True)
    b = run(fi, custom_name="shot_A.png", versioning="filename", reserve_version=True)
    assert (a["version_string"], b["version_string"]) == ("v001", "v002")
    d = out / "shot_A" / day
    assert (d / ".shot_A_v001.reserved").is_file() and not (d / "v001").exists()


def test_filename_run_lease_shares_one_version_per_name(fi, out, monkeypatch):
    monkeypatch.setattr(fi._lease, "current_run_id", lambda: "run-l794")
    png = run(fi, custom_name="shot_A.png", versioning="filename")
    touch(out / png["output_filename"])               # the first saver already wrote
    mov = run(fi, custom_name="shot_A.mov", versioning="filename")
    other = run(fi, custom_name="shot_A.png", versioning="filename", output_name="matte")
    assert png["version_string"] == mov["version_string"] == "v001"   # one deliverable, one number
    assert other["version_string"] == "v001" and other["output_filename"].endswith("matte_v001.png")


def test_filename_is_changed_tracks_new_files(fi, out, day):
    kw = {"source_filename": "shot_A.png", "versioning": "filename", "prefix": "v", "padding": 3}
    touch(out / "shot_A" / day / "keep.txt")
    before = fi.FolderIncrementer.IS_CHANGED(**kw)
    import os
    import time
    target = out / "shot_A" / day / "shot_A_v001.png"
    touch(target)
    d = out / "shot_A" / day
    os.utime(d, ns=(time.time_ns(), d.stat().st_mtime_ns + 1_000_000))   # coarse-mtime filesystems
    assert fi.FolderIncrementer.IS_CHANGED(**kw) != before


def test_core_saveimage_batch_naming_stays_intact(fi, out, day):
    """filename_prefix into core SaveImage: <prefix>_00001_.png per frame - one version, the saver's counter after
    it, never a second version token."""
    r = run(fi, custom_name="shot_A.png", versioning="filename")
    prefix = Path(r["filename_prefix"])
    frames = [f"{prefix.name}_{i:05}_.png" for i in range(1, 4)]
    assert frames == ["shot_A_v001_00001_.png", "shot_A_v001_00002_.png", "shot_A_v001_00003_.png"]
    for f in frames:
        touch(out / prefix.parent / f)
    assert run(fi, custom_name="shot_A.png", versioning="filename")["filename_prefix"].endswith("shot_A_v002")


@pytest.mark.parametrize("versioning", ["folder", "filename"])
def test_is_changed_follows_custom_name(fi, out, day, versioning):
    """An API run sets custom_name and leaves source_filename empty: the cache key must still follow <custom>/<date>,
    or the second queue is served from the cache and writes nothing (found by the L7.94 live check)."""
    kw = {"source_choice": "custom", "custom_name": "shot_A.png", "versioning": versioning}
    before = fi.FolderIncrementer.IS_CHANGED(**kw)
    touch(out / "shot_A" / day / ("v001" if versioning == "folder" else "shot_A_v001.png"),
          is_dir=versioning == "folder")
    assert fi.FolderIncrementer.IS_CHANGED(**kw) != before
