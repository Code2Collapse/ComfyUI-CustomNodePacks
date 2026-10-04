"""FolderIncrementer reads its source name out of ANY loader.

Reported 2026-09-29: "not taking names from load images as batch or whatever
folder ... it must take folder name of exr or folder name of images or file
name of exr or file name of video (whatever format)". The front-end kept a
widget value only if it contained a dot, from a fixed list of widget names on a
fixed list of loader classes, so every folder loader (Load Images From Dir,
VHS Load Images, WAS Load Image Batch, EXR sequence folders) was skipped.

The rules now live in js/_fi_source_name.js with a Python mirror in
folder_incrementer.py. One fixture table runs through BOTH here, so the
on-node preview and the name written to disk cannot disagree.
"""
from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
JS = ROOT / "js" / "_fi_source_name.js"
NODE = shutil.which("node")


@pytest.fixture(scope="module")
def fi():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    spec = importlib.util.spec_from_file_location("folder_incrementer_src_t", ROOT / "folder_incrementer.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# (value, folder_hint) -> (filename, kind) or None
CASES = [
    (("D:/shots/sh010_plate/", True), ("sh010_plate", "folder")),
    (("D:\\shots\\sh010_plate", False), ("sh010_plate", "folder")),
    (("/mnt/plates/sh010_plate/*.exr", True), ("sh010_plate.exr", "sequence")),
    (("D:/shots/sh010_plate/0001.exr", False), ("sh010_plate.exr", "sequence")),
    (("D:/shots/sh010/exr/", True), ("sh010", "folder")),
    (("D:/shots/sh010/v003/2048x1152/", True), ("sh010", "folder")),
    (("sh010.####.exr", False), ("sh010.exr", "sequence")),
    (("sh010_plate.%04d.exr", False), ("sh010_plate.exr", "sequence")),
    (("sh010.$F4.exr", False), ("sh010.exr", "sequence")),
    (("sh010.[1001-1100].exr", False), ("sh010.exr", "sequence")),
    (("sh010_plate.1001.exr", False), ("sh010_plate.1001.exr", "sequence")),
    (("ref_face.png", False), ("ref_face.png", "image")),
    (("subfolder/shot.png", False), ("shot.png", "image")),
    (("clip.mov [input]", False), ("clip.mov", "video")),
    (("A001_C003.R3D", False), ("A001_C003.R3D", "video")),
    (("interview.mxf", False), ("interview.mxf", "video")),
    (("https://cdn.example.com/media/clip.mp4?token=1", False), ("clip.mp4", "video")),
    (("voice.wav", False), ("voice.wav", "audio")),
    (("sh010", True), ("sh010", "folder")),
    (("sh010", False), None),
    (("a photo of a cat", False), None),
    (("", True), None),
    (("line one\nline2.png", False), None),
]


def test_python_rules(fi):
    for (value, hint), want in CASES:
        got = fi._source_from_value(value, hint)
        assert (tuple(got) if got else None) == want, (value, got)


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_js_rules_match_python(fi, tmp_path):
    probe = tmp_path / "probe.mjs"
    probe.write_text(
        f'import * as S from {json.dumps(JS.as_uri())};\n'
        f'const cases = {json.dumps([list(c[0]) for c in CASES])};\n'
        'const out = cases.map(([v, h]) => { const r = S.sourceFromValue(v, h); return r ? [r.filename, r.kind] : null; });\n'
        'process.stdout.write(JSON.stringify({ out, video: S.VIDEO_EXTS, image: S.IMAGE_EXTS,\n'
        '  audio: S.AUDIO_EXTS, seq: S.SEQ_EXTS, file: S.FILE_WIDGETS, folder: S.FOLDER_WIDGETS }));\n',
        encoding="utf-8")
    p = subprocess.run([NODE, str(probe)], capture_output=True, text=True, timeout=60)
    assert p.returncode == 0, p.stderr[:800]
    data = json.loads(p.stdout)
    for ((value, _), want), got in zip(CASES, data["out"]):
        assert (tuple(got) if got else None) == want, ("JS", value, got)
    # the two sides must know the same formats and the same widget names
    assert tuple(data["video"]) == fi._VIDEO_EXTS
    assert tuple(data["image"]) == fi._IMAGE_EXTS
    assert tuple(data["audio"]) == fi._AUDIO_EXTS
    assert tuple(data["seq"]) == fi._SEQ_EXTS
    assert tuple(data["file"]) == fi._FILE_WIDGETS
    assert tuple(data["folder"]) == fi._FOLDER_WIDGETS


# Real loaders' widget layouts (names as those packs ship them).
LOADERS = {
    "WAS Load Image Batch": ({"mode": "incremental_image", "index": 0, "label": "Batch 001",
                              "path": "D:/plates/sh020_bg/", "pattern": "*"}, "sh020_bg"),
    "VHS_LoadImagesPath": ({"directory": "/mnt/plates/sh030_fg", "image_load_cap": 0}, "sh030_fg"),
    "VHS_LoadImages (upload combo)": ({"directory": "sh031_plate", "image_load_cap": 0}, "sh031_plate"),
    "LoadImagesFromFolderKJ": ({"folder": "E:\\renders\\sh040\\exr", "width": 1024}, "sh040"),
    "LoadImage": ({"image": "hero_ref.png", "upload": "image"}, "hero_ref.png"),
    "VHS_LoadVideo": ({"video": "take_07.mov", "force_rate": 0}, "take_07.mov"),
    "NukeMax EXR sequence": ({"sequence_path": "D:/plates/sh050/sh050_plate.####.exr"}, "sh050_plate.exr"),
    "unknown pack, generic STRING": ({"value": "D:/plates/sh060_clean/", "text": "a woman"}, "sh060_clean"),
}


@pytest.mark.parametrize("name", sorted(LOADERS))
def test_any_loader_layout_yields_its_name(fi, name):
    inputs, want = LOADERS[name]
    got = fi._source_from_inputs(inputs)
    assert got and got[0] == want, (name, got)


def test_a_prompt_text_is_never_taken_for_a_name(fi):
    assert fi._source_from_inputs({"text": "cinematic shot of D:/ a city", "seed": 3}) is None


def _run(fi, tmp_path, prompt, choice="auto"):
    node = fi.FolderIncrementer()
    out = node.increment(base_path=str(tmp_path), source_choice=choice,
                         prompt=prompt, unique_id="9")
    return out[2]           # folder_name


def test_run_with_empty_source_reads_the_wired_folder_loader(fi, tmp_path):
    """The run itself finds the plate folder upstream - through a processing
    node - when the UI sent no name (API submit / unrecognised loader)."""
    prompt = {
        "1": {"class_type": "LoadImage", "inputs": {"image": "hero_ref.png"}},
        "2": {"class_type": "LoadImagesFromDir //Inspire", "inputs": {"directory": "D:/plates/sh070_plate/"}},
        "3": {"class_type": "ImageScale", "inputs": {"image": ["2", 0], "width": 512}},
        "9": {"class_type": "FolderIncrementer", "inputs": {"trigger_video": ["3", 0], "trigger_image": ["1", 0]}},
    }
    assert _run(fi, tmp_path, prompt) == "sh070_plate"
    assert _run(fi, tmp_path, prompt, "image") == "hero_ref"


def test_run_with_nothing_wired_scans_the_loaders(fi, tmp_path):
    prompt = {
        "4": {"class_type": "NukeMax_EXRSequenceLoad",
              "inputs": {"sequence_path": "/mnt/show/sh080/comp/sh080_comp_v003.%04d.exr"}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"text": "a quiet street"}},
        "9": {"class_type": "FolderIncrementer", "inputs": {}},
    }
    assert _run(fi, tmp_path, prompt, "exr") == "sh080_comp_v003"


def test_a_full_path_in_source_path_names_the_folder(fi):
    stem, ext = fi._resolve_stem_and_ext("D:/shots/sh090_plate/0001.exr")
    assert (stem, ext) == ("sh090_plate", ".exr")
    stem, ext = fi._resolve_stem_and_ext("D:/shots/sh091/exr/")
    assert (stem, ext) == ("sh091", "")
    # a plain filename is untouched by the path rules
    assert fi._resolve_stem_and_ext("C1799.MP4 Comp 1", ".mp4") == ("C1799.MP4 Comp 1", ".mp4")
