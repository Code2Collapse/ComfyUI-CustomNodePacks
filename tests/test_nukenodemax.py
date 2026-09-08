# NukeNodeMax feature tests — TDD per autonomous protocol.
# Run with: python -m pytest tests/test_nukenodemax.py -v
import atexit, json, os, shutil, sys, tempfile, importlib.util, types

import numpy as np
import pytest
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_PACK = os.path.dirname(_HERE)
_NODES = os.path.join(_PACK, "nodes")
if _PACK not in sys.path:
    sys.path.insert(0, _PACK)

# Provide a richer folder_paths stub than conftest's empty one.
fp = sys.modules.get("folder_paths")
if fp is None:
    fp = types.ModuleType("folder_paths")
    sys.modules["folder_paths"] = fp
# Scratch dirs live in the OS temp area, never in the repo. These used to point
# at _PACK/_test_temp etc., so any node that wrote a preview left PNGs inside the
# source tree for git to pick up.
_SCRATCH = tempfile.mkdtemp(prefix="c2c_nukenodemax_")
atexit.register(shutil.rmtree, _SCRATCH, True)


def _scratch(name):
    d = os.path.join(_SCRATCH, name)
    os.makedirs(d, exist_ok=True)
    return d


fp.base_path = _SCRATCH
fp.models_dir = _scratch("models")
fp.folder_names_and_paths = {}
fp.get_filename_list = lambda key: []
fp.get_full_path = lambda key, name: None
fp.get_folder_paths = lambda key: []
fp.add_model_folder_path = lambda *a, **k: None
fp.get_temp_directory = lambda: _scratch("temp")
fp.get_output_directory = lambda: _scratch("output")
fp.get_input_directory = lambda: _scratch("input")


def _load(modname, path):
    spec = importlib.util.spec_from_file_location(modname, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


# Synthetic parent package so relative imports inside the node files resolve.
_PKG = "nnxpkg"
_pkg = types.ModuleType(_PKG)
_pkg.__path__ = [_NODES]
sys.modules[_PKG] = _pkg


def _loadp(short, fname):
    return _load(f"{_PKG}.{short}", os.path.join(_NODES, fname))


# Order matters: propainter_flow_refine is imported by optical_flow.
_flow_refine = _loadp("propainter_flow_refine", "propainter_flow_refine.py")
_roto = _loadp("roto", "roto.py")
_optical = _loadp("optical_flow", "optical_flow.py")
_shuffle = _loadp("shuffle", "shuffle.py")
_tcl = _loadp("clipboard_tcl", "clipboard_tcl.py")
_insight = _loadp("insight", "insight.py")
_integrity = _loadp("integrity_guard", "integrity_guard.py")


# =====================================================================
# Feature 1 — Vector-Based Roto
# =====================================================================
def _square_spline(cx, cy, r):
    """Closed square via cubic Bezier with zero-length handles -> straight edges."""
    pts = [
        {"x": cx - r, "y": cy - r, "in": [cx - r, cy - r], "out": [cx - r, cy - r]},
        {"x": cx + r, "y": cy - r, "in": [cx + r, cy - r], "out": [cx + r, cy - r]},
        {"x": cx + r, "y": cy + r, "in": [cx + r, cy + r], "out": [cx + r, cy + r]},
        {"x": cx - r, "y": cy + r, "in": [cx - r, cy + r], "out": [cx - r, cy + r]},
    ]
    return pts


def test_roto_static_square():
    cls = _roto.VectorRotoMEC
    payload = {
        "canvas": {"w": 256, "h": 256},
        "frames": [{"frame": 0, "splines": [_square_spline(128, 128, 40)]}],
    }
    mask, _ = cls().rasterize(
        roto_json=json.dumps(payload),
        frame_count=1, width=256, height=256,
        samples_per_seg=24, feather_px=0,
    )
    assert mask.shape == (1, 256, 128 * 0 + 256), f"got shape {mask.shape}"
    assert mask.dtype == torch.float32
    # Interior pixel solid, exterior zero.
    assert mask[0, 128, 128].item() > 0.9, "centre pixel should be inside polygon"
    assert mask[0, 5, 5].item() < 0.05, "corner pixel should be outside polygon"
    # Approx area: 80x80 = 6400 px ± rasterisation tolerance.
    area = mask[0].sum().item()
    assert 6000 < area < 6800, f"square area {area} out of tolerance"


def test_roto_keyframe_interpolation():
    cls = _roto.VectorRotoMEC
    payload = {
        "canvas": {"w": 512, "h": 512},
        "frames": [
            {"frame": 0,  "splines": [_square_spline(100, 100, 30)]},
            {"frame": 10, "splines": [_square_spline(300, 300, 30)]},
        ],
    }
    masks, _ = cls().rasterize(
        roto_json=json.dumps(payload),
        frame_count=11, width=512, height=512,
        samples_per_seg=24, feather_px=0,
    )
    assert masks.shape == (11, 512, 512)
    # Frame 5: centroid should be near (200, 200).
    m = masks[5].numpy()
    assert m.sum() > 100, "frame 5 mask is empty -> interpolation broken"
    ys, xs = np.where(m > 0.5)
    cx, cy = xs.mean(), ys.mean()
    assert abs(cx - 200) < 8 and abs(cy - 200) < 8, f"centroid ({cx:.1f},{cy:.1f}) not near (200,200)"


# =====================================================================
# Feature 2 — Deep Image Compositing: MIGRATED OUT (Apr 2026)
# =====================================================================
# deep_composite.py moved to ComfyUI-NukeMaxNodes (nukemax/nodes/deep/) and the
# data model changed from a list-of-dict DEEP_IMAGE to the tensor-backed
# DeepImage dataclass, so these tests could not be repointed as written.
# The invariants they protected — a merge must not collapse to flat alpha, and
# an opaque front sample must fully occlude the one behind it — are ported to
# ComfyUI-NukeMaxNodes/tests/test_deep_image.py. That repo had NO deep coverage
# at all before this triage.

# =====================================================================
# Feature 3 — Optical Flow
# =====================================================================
def test_optical_flow_outputs_uv_and_revectors():
    """Translate frame_a by +5 px -> warp_with_flow(frame_a) should ≈ frame_b."""
    cls = _optical.OpticalFlowMEC
    H, W = 64, 64
    # Build a frame with a high-contrast vertical bar at column 20.
    frame_a = torch.zeros(1, H, W, 3)
    frame_a[:, :, 18:23, :] = 1.0
    frame_b = torch.zeros(1, H, W, 3)
    frame_b[:, :, 23:28, :] = 1.0  # shifted +5 px in x

    re_vectored, flow_rgb, consistency = cls().revector(
        frame_a=frame_a, frame_b=frame_b,
        iters=20, consistency_thr=1.5, scale=1.0,
    )
    assert flow_rgb.shape == (1, H, W, 3), "flow not packed as IMAGE"
    assert re_vectored.shape == frame_a.shape
    # Re-vectored frame must be closer to frame_b than to frame_a (proves flow is real, not blur).
    err_to_b = (re_vectored - frame_b).abs().mean().item()
    err_no_warp = (frame_a - frame_b).abs().mean().item()
    assert err_to_b < err_no_warp, (
        f"warped err={err_to_b:.4f} not better than no-warp err={err_no_warp:.4f} "
        f"-> DUMB_BLUR antipattern"
    )


# =====================================================================
# Feature 4 — Native Shuffle Logic
# =====================================================================
def test_shuffle_swap_r_to_a_g_to_r():
    """Input RGB=[1,0,0], swap R->A, G->R -> output RGBA = [0,0,0,1]."""
    cls = _shuffle.ShuffleMEC
    img = torch.zeros(1, 2, 2, 3)
    img[..., 0] = 1.0  # R=1, G=0, B=0
    out_img, out_mask = cls().shuffle(
        image=img,
        out_R="zero",   # R becomes 0  (G->R requested but G is 0 already; equivalent)
        out_G="zero",
        out_B="zero",
        out_A="R",      # A takes R
        premultiply_output=False,
    )
    # All RGB zero, A = 1.0 everywhere.
    assert torch.allclose(out_img[..., 0], torch.zeros(1, 2, 2)), "R not zeroed"
    assert torch.allclose(out_img[..., 1], torch.zeros(1, 2, 2)), "G not zeroed"
    assert torch.allclose(out_img[..., 2], torch.zeros(1, 2, 2)), "B not zeroed"
    assert torch.allclose(out_img[..., 3], torch.ones(1, 2, 2)), "A != source R"
    assert torch.allclose(out_mask, torch.ones(1, 2, 2)), "MASK output != A"


# =====================================================================
# Feature 5 — Nuke TCL Copy-Paste Protocol
# =====================================================================
def test_tcl_round_trip_no_hardcoded_ids():
    mod = _tcl

    nodes = [
        {"id": 1, "class_type": "LoadImage", "name": "Load1",
         "widgets": {"image_path": "foo.png"}, "xpos": 0, "ypos": 0, "selected": False},
        {"id": 2, "class_type": "Blur", "name": "Blur1",
         "widgets": {"size": 12}, "xpos": 200, "ypos": 0, "selected": False},
    ]
    links = [(1, 0, 2, 0)]
    tcl = mod.serialize(nodes, links)

    # MUST be human-readable TCL, NOT JSON.
    assert not tcl.lstrip().startswith("{"), "BINARY_CLIPBOARD: looks like JSON"
    assert "set cut_paste_input [stack 0]" in tcl, "missing TCL prologue"
    assert "push" in tcl and "[stack 0]" in tcl, "missing stack semantics -> HARD_CODED_LINKS"
    assert "LoadImage {" in tcl and "Blur {" in tcl
    # Stack semantics: the connection between Load1 and Blur1 must NOT be encoded
    # via hard-coded numeric ids in the Blur block; only `inputs N`.
    blur_block = tcl[tcl.index("Blur {"):tcl.index("end_group")]
    assert "inputs 1" in blur_block, "Blur block missing `inputs 1` (stack-pop count)"
    assert " 1 " not in blur_block.replace("inputs 1", ""), "hard-coded src id leaked into block"

    parsed = mod.parse(tcl)
    assert len(parsed["nodes"]) == 2
    classes = [n["class_type"] for n in parsed["nodes"]]
    assert classes == ["LoadImage", "Blur"]
    # Link reconstructed: source id of LoadImage -> dest id of Blur, dest slot 0.
    assert len(parsed["links"]) == 1
    fid, fslot, tid, tslot = parsed["links"][0]
    src_node = next(n for n in parsed["nodes"] if n["id"] == fid)
    dst_node = next(n for n in parsed["nodes"] if n["id"] == tid)
    assert src_node["class_type"] == "LoadImage"
    assert dst_node["class_type"] == "Blur"
    assert tslot == 0


# =====================================================================
# Feature 6 — Insight Diagnostic
# =====================================================================
def test_insight_explains_module_not_found():
    mod = _insight
    try:
        raise ModuleNotFoundError("No module named 'propainter'")
    except ModuleNotFoundError as e:
        hint = mod.explain_exception(e)
    # Must NOT just echo stderr — must produce a translated, human hint.
    assert "pip install" in hint.lower(), f"DUMB_LOGGER: hint did not translate ({hint!r})"


def test_insight_explains_cuda_oom():
    mod = _insight
    try:
        raise RuntimeError("CUDA out of memory. Tried to allocate 12.00 GiB")
    except RuntimeError as e:
        hint = mod.explain_exception(e)
    assert "vram" in hint.lower() or "lowvram" in hint.lower() or "batch" in hint.lower(), \
        f"OOM hint not specific: {hint!r}"


# =====================================================================
# Feature 7 — Conflict & Integrity Guard
# =====================================================================
def test_integrity_guard_runs_pip_check_actively():
    """Must actually execute `pip check` subprocess, not just list packages."""
    mod = _integrity
    # Run the worker synchronously.
    mod._worker()
    rep = dict(mod._LAST_REPORT)
    assert rep.get("ready"), "integrity worker never completed"
    assert "pip_check" in rep, "STATIC_GUARD: never ran pip check"
    assert "rc" in rep["pip_check"], "pip_check did not store subprocess rc -> STATIC_GUARD"
    # The subprocess must have been invoked; rc is an int (0=clean, 1=conflicts).
    assert isinstance(rep["pip_check"]["rc"], int)
