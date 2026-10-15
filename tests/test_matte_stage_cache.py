"""L7.114 (owner answer 2026-10-15: "inside the node + stage cache"): viewer-driven Mask + Matting runs keep the
TRACK and EDGE stages in a byte-bounded cache, so a late setting reruns only the late stages. Counting stand-ins
around the real (model-free) segmenter, tracker and matter show which stage ran; a cached run must give the same
tensors as a fresh one."""
from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from nodes.mask_matting import _stage_cache as sc

H, W, CX, CY, R = 64, 80, 40, 32, 14


def plate(b=3, seed=5):
    rng = np.random.default_rng(seed)
    frames = []
    for i in range(b):
        g = (0.45 + 0.06 * rng.standard_normal((H, W, 1))).clip(0, 1).repeat(3, -1)
        yy, xx = np.mgrid[:H, :W]
        g[(xx - CX - i) ** 2 + (yy - CY) ** 2 <= R ** 2] = [0.85, 0.12, 0.10]
        frames.append(g)
    return torch.from_numpy(np.stack(frames)).float()


PICKS = json.dumps({"v": 1, "frame": 0, "subject": {"x": CX / W, "y": CY / H, "box": [0.25, 0.2, 0.75, 0.8]}})
PICKS2 = json.dumps({"v": 1, "frame": 0, "subject": {"x": (CX + 3) / W, "y": CY / H}})


@pytest.fixture(autouse=True)
def fresh(monkeypatch):
    monkeypatch.setattr(sc, "_CACHE", None)
    monkeypatch.delenv("C2C_MATTE_CACHE_MB", raising=False)


@pytest.fixture(scope="module")
def maskops():
    node = pytest.importorskip("nodes.mask_matting.node", reason="MaskOps unavailable")
    return node.MaskOpsMEC


def defaults(cls):
    spec = cls.INPUT_TYPES()
    out = {}
    for name, s in {**spec["required"], **spec.get("optional", {})}.items():
        opts = s[1] if len(s) > 1 and isinstance(s[1], dict) else {}
        if "default" in opts:
            out[name] = opts["default"]
        elif isinstance(s[0], (list, tuple)) and s[0]:
            out[name] = s[0][0]
    return out


def run(cls, **kw):
    args = defaults(cls)
    args.update(kw)
    return cls().execute(**args)


def counting(monkeypatch, owner, name):
    calls = []
    orig = getattr(owner, name)

    def wrapped(*a, **k):
        calls.append(1)
        return orig(*a, **k)

    monkeypatch.setattr(owner, name, wrapped)
    return calls


def same(a, b):
    if torch.is_tensor(a):
        return torch.is_tensor(b) and a.shape == b.shape and torch.equal(a, b)
    return a == b


# ── the cache itself ─────────────────────────────────────────────────────────────────────────────────────────────

def test_bounded_in_bytes_least_recently_used_first():
    c = sc.StageCache(3 * 4000)                       # room for three 1000-float tensors
    for k in "abc":
        assert c.put(k, {"t": torch.zeros(1000)})
    assert c.get("a") is not None                     # a is now the most recent
    c.put("d", {"t": torch.zeros(1000)})
    assert c.get("b") is None and c.get("a") is not None and c.get("d") is not None
    assert c.bytes <= c.budget and len(c) == 3
    assert not c.put("huge", torch.zeros(4000))       # bigger than the whole budget: not kept
    assert c.get("huge") is None


def test_budget_from_the_env(monkeypatch):
    assert sc.budget_bytes() == 1024 * 2**20
    monkeypatch.setenv("C2C_MATTE_CACHE_MB", "16")
    assert sc.get_cache().budget == 16 * 2**20
    monkeypatch.setenv("C2C_MATTE_CACHE_MB", "0")
    assert sc.get_cache() is None
    monkeypatch.setenv("C2C_MATTE_CACHE_MB", "lots")
    assert sc.budget_bytes() == 1024 * 2**20


def test_the_fingerprint_sees_every_pixel():
    a = plate(2)
    b = a.clone()
    b[1, 63, 79, 2] += 1e-6                            # one channel of one far-corner pixel
    assert sc.fingerprint(a) == sc.fingerprint(a.clone())
    assert sc.fingerprint(a) != sc.fingerprint(b)
    assert sc.fingerprint(a) != sc.fingerprint(a.reshape(2, W, H, 3))   # same bytes, other shape
    assert sc.fingerprint(None) == "none"


def test_entries_live_on_the_cpu():
    c = sc.StageCache(2**20)
    t = torch.ones(4)
    c.put("k", {"x": [t]})
    assert c.get("k")["x"][0].device.type == "cpu"


# ── cascade: TRACK / EDGE / FINISH ───────────────────────────────────────────────────────────────────────────────

CASCADE = dict(segmenter="classic", matter="guided_filter", pipeline="cascade (legacy)")


def _cascade_counters(monkeypatch):
    from nodes.mask_matting.segmenters import get_segmenter_cls
    from nodes.mask_matting.matters import get_matter_cls
    return (counting(monkeypatch, get_segmenter_cls("classic"), "segment"),
            counting(monkeypatch, get_matter_cls("guided_filter"), "matte"))


def test_cascade_late_settings_rerun_only_late_stages(maskops, monkeypatch):
    seg, mat = _cascade_counters(monkeypatch)
    image = plate(3)
    first = run(maskops, image=image, picks=PICKS, **CASCADE)
    assert (len(seg), len(mat)) == (1, 1)
    assert json.loads(first[7])["viewer"]["stage_cache"] == {"track": "ran", "edge": "ran"}

    # a FINISH setting: neither SAM nor the matter runs again
    out = run(maskops, image=image, picks=PICKS, despill="red", **CASCADE)
    assert (len(seg), len(mat)) == (1, 1)
    assert json.loads(out[7])["viewer"]["stage_cache"] == {"track": "reused", "edge": "reused"}

    # an EDGE setting: only the matter runs again
    out = run(maskops, image=image, picks=PICKS, trimap_dilate=12, **CASCADE)
    assert (len(seg), len(mat)) == (1, 2)
    assert json.loads(out[7])["viewer"]["stage_cache"] == {"track": "reused", "edge": "ran"}

    # another selection or another plate: everything runs
    run(maskops, image=image, picks=PICKS2, **CASCADE)
    assert (len(seg), len(mat)) == (2, 3)
    run(maskops, image=plate(3, seed=6), picks=PICKS, **CASCADE)
    assert (len(seg), len(mat)) == (3, 4)


def test_a_cached_cascade_run_gives_the_fresh_result(maskops, monkeypatch):
    # the model-free segmenter (GrabCut) draws from OpenCV's random generator: two fresh runs differ unless it is
    # seeded, so both reference runs start from the same seed
    cv2 = pytest.importorskip("cv2")
    image = plate(3)
    cv2.setRNGSeed(7)
    run(maskops, image=image, picks=PICKS, **CASCADE)                       # fills the cache
    cached = run(maskops, image=image, picks=PICKS, despill="red", **CASCADE)
    monkeypatch.setenv("C2C_MATTE_CACHE_MB", "0")
    cv2.setRNGSeed(7)
    fresh = run(maskops, image=image, picks=PICKS, despill="red", **CASCADE)
    for k in range(19):
        if k == 7:
            a, b = json.loads(cached[7]), json.loads(fresh[7])
            a.pop("viewer"), b.pop("viewer")
            assert a == b
        else:
            assert same(cached[k], fresh[k]), maskops.RETURN_NAMES[k]


def test_cache_off_reruns_everything(maskops, monkeypatch):
    monkeypatch.setenv("C2C_MATTE_CACHE_MB", "0")
    seg, mat = _cascade_counters(monkeypatch)
    image = plate(2)
    run(maskops, image=image, picks=PICKS, **CASCADE)
    out = run(maskops, image=image, picks=PICKS, despill="red", **CASCADE)
    assert (len(seg), len(mat)) == (2, 2)
    assert json.loads(out[7])["viewer"]["cache"] == "off"


def test_plain_runs_never_cache(maskops, monkeypatch):
    seg, mat = _cascade_counters(monkeypatch)
    image = plate(2)
    for despill in ("off", "red"):
        run(maskops, image=image, positive_coords=json.dumps([[CX, CY]]), despill=despill, **CASCADE)
    assert (len(seg), len(mat)) == (2, 2)
    assert sc._CACHE is None


# ── ONYX ─────────────────────────────────────────────────────────────────────────────────────────────────────────

def test_onyx_late_settings_rerun_only_late_stages(maskops, monkeypatch):
    from nodes.mask_matting import _onyx_pipeline as op
    track = counting(monkeypatch, op, "propagate_objects")
    edge = counting(monkeypatch, op, "stabilise_alpha_unknown")
    image = plate(3)
    kw = dict(image=image, matter="guided_filter", pipeline="onyx", picks=PICKS)
    first = run(maskops, **kw)
    assert (len(track), len(edge)) == (1, 1)
    run(maskops, **kw, lightwrap_strength=0.4)            # finishing
    assert (len(track), len(edge)) == (1, 1)
    out = run(maskops, **kw, band_scale=2.0)               # edge stage: trimap band
    assert (len(track), len(edge)) == (1, 2)
    assert json.loads(out[7])["onyx"]["stage_cache"] == {"track": "reused", "edge": "ran"}
    again = run(maskops, **kw)                             # back to the first settings: all from the cache
    assert (len(track), len(edge)) == (1, 2)
    for k in (0, 1, 3, 17):
        assert same(first[k], again[k])


# ── the three engines (one stage each) ───────────────────────────────────────────────────────────────────────────

def test_an_engine_runs_once_for_finishing_changes(maskops, monkeypatch):
    from nodes.mask_matting import _engines
    calls = []

    def fake(image, kw):
        calls.append(1)
        a = torch.zeros(image.shape[0], H, W)
        a[:, 10:40, 20:60] = 1
        return _engines.EngineResult(mask=a, alpha=a, preview=None, score=0.5, bbox=None, edge_mask=None,
                                     foreground=None, info={"engine": "fake"})

    monkeypatch.setitem(_engines._RUNNERS, _engines.SAM_VITMATTE, fake)
    sam = object()
    kw = dict(image=plate(2), backend=_engines.SAM_VITMATTE, sam_model=sam, picks=PICKS)
    run(maskops, **kw)
    out = run(maskops, **kw, edge_mode="hard", despill="red")
    assert len(calls) == 1
    assert json.loads(out[7])["viewer"]["stage_cache"] == {"engine": "reused"}
    run(maskops, **kw, sam_edge_radius=20)                 # the engine's own setting
    assert len(calls) == 2
    run(maskops, **{**kw, "sam_model": object()})           # another SAM model object: never the cached one
    assert len(calls) == 3
