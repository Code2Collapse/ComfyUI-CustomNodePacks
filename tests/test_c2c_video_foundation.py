"""CPU tests for c2c_video foundation (S0)."""

from __future__ import annotations

import importlib.util
import sys
import threading
from pathlib import Path

import numpy as np
import pytest

PACK = Path(__file__).resolve().parents[1]
C2C_VIDEO = PACK / "nodes" / "c2c_video"


def _load_c2c_video():
    name = "c2c_video"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name,
        C2C_VIDEO / "__init__.py",
        submodule_search_locations=[str(C2C_VIDEO)],
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


cv = _load_c2c_video()


@pytest.fixture(autouse=True)
def _clean_video_state():
    cv.reset_demux_call_count()
    cv.drain_container_pool()
    yield
    cv.drain_container_pool()



@pytest.fixture(autouse=True)
def _private_index_cache(tmp_path, monkeypatch):
    """Persisted exact indexes go to the test's own folder, never the user's cache."""
    monkeypatch.setenv("C2C_VIDEO_INDEX_DIR", str(tmp_path / "index_cache"))


def test_exact_index_persists_across_sessions(ffv1_clip):
    """A second process (simulated: memory cache dropped) reads the index
    from disk instead of demuxing the file again."""
    from importlib import import_module
    probe_mod = import_module(cv.__name__ + ".probe")
    h = cv.probe_file(ffv1_clip, index="exact")
    probe_mod._INDEX_CACHE.clear()
    cv.reset_demux_call_count()
    h2 = cv.probe_file(ffv1_clip, index="exact")
    assert cv.demux_call_count() == 0
    assert h2.frame_count == h.frame_count
    batch = cv.read(h2, [0, h2.frame_count - 1], fmt="uint8")
    assert [cv.decode_index(b) for b in batch] == [0, h2.frame_count - 1]


@pytest.fixture
def cfr_clip(tmp_path):
    path = str(tmp_path / "cfr.mp4")
    cv.make_test_clip("h264_cfr_b", path, frames=60, width=256, height=256)
    return path


@pytest.fixture
def vfr_clip(tmp_path):
    path = str(tmp_path / "vfr.mp4")
    cv.make_test_clip("h264_vfr", path, frames=30, width=256, height=256)
    return path


@pytest.fixture
def mislabel_clip(tmp_path):
    path = str(tmp_path / "mislabel.mp4")
    cv.make_test_clip("h264_cfr_mislabel", path, frames=40, width=256, height=256)
    return path


@pytest.fixture
def prores_clip(tmp_path):
    path = str(tmp_path / "prores.mov")
    cv.make_test_clip("prores4444", path, frames=20, width=256, height=256)
    return path


@pytest.fixture
def ten_bit_clip(tmp_path):
    path = str(tmp_path / "10bit.mp4")
    cv.make_test_clip("h264_10bit", path, frames=20, width=256, height=256)
    return path


@pytest.fixture
def ffv1_clip(tmp_path):
    path = str(tmp_path / "ffv1.mkv")
    cv.make_test_clip("ffv1", path, frames=30, width=256, height=256)
    return path


@pytest.fixture
def exr_seq(tmp_path):
    d = str(tmp_path / "exr_seq")
    cv.make_test_sequence(d, frames=8, width=256, height=64)
    return d


def test_probe_counts_all_variants(cfr_clip, vfr_clip, prores_clip, ten_bit_clip, ffv1_clip, exr_seq):
    assert cv.probe_file(cfr_clip).frame_count == 60
    assert cv.probe_file(vfr_clip).frame_count == 30
    assert cv.probe_file(prores_clip).frame_count == 20
    assert cv.probe_file(ten_bit_clip).frame_count == 20
    assert cv.probe_file(ffv1_clip).frame_count == 30
    assert cv.probe_sequence(exr_seq).frame_count == 8


def test_auto_probe_skips_demux(cfr_clip):
    cv.reset_demux_call_count()
    h = cv.probe_file(cfr_clip, index="auto")
    assert cv.demux_call_count() == 0
    tbl = cv.get_index_table(h.source.stat_key)
    assert not tbl.is_exact


def test_vfr_falls_to_exact(vfr_clip):
    cv.reset_demux_call_count()
    h = cv.probe_file(vfr_clip, index="auto")
    assert cv.demux_call_count() >= 1
    assert cv.get_index_table(h.source.stat_key).is_exact


def test_mislabel_rebuild_on_read(mislabel_clip, monkeypatch):
    """Metadata that CLAIMS constant frame rate on a clip whose timestamps are
    irregular: the fast arithmetic table is wrong, and the first read must
    notice (decoded pts != expected), rebuild the exact table and still return
    exactly the requested frames."""
    from importlib import import_module
    probe_mod = import_module(cv.__name__ + ".probe")
    monkeypatch.setattr(probe_mod, "_guess_cfr", lambda vs, n, d: (True, probe_mod._stream_fps(vs)))
    # ...and the irregularity sits where the head/tail sample does not look
    monkeypatch.setattr(probe_mod, "_cfr_confirmed", lambda path, si, table, claimed: claimed)
    h = cv.probe_file(mislabel_clip, index="auto")
    assert not cv.get_index_table(h.source.stat_key).is_exact
    positions = list(range(len(h)))
    batch = cv.read(h, positions)
    assert batch.shape[0] == len(h)
    for i, pos in enumerate(positions):
        assert cv.decode_index(batch[i]) == h.selected_indices()[pos]
    assert cv.get_index_table(h.source.stat_key).is_exact


def test_index_table_matches_full_decode(ffv1_clip):
    h = cv.probe_file(ffv1_clip, index="exact")
    tbl = cv.get_index_table(h.source.stat_key)
    all_idx = list(range(h.frame_count))
    batch = cv.read(h, all_idx)
    for i in range(h.frame_count):
        assert tbl.pts_of(i) is not None
        assert cv.decode_index(batch[i]) == i


def _assert_read_patterns(handle, expected_count):
    patterns = [
        list(range(expected_count)),
        list(reversed(range(expected_count))),
        [0, 0, 1, 1, 2],
        list(range(0, expected_count, 7)),
        [0],
        [expected_count - 1],
    ]
    for pos_list in patterns:
        batch = cv.read(handle, pos_list)
        assert batch.shape[0] == len(pos_list)
        sel = list(handle.selected_indices())
        for j, pos in enumerate(pos_list):
            assert cv.decode_index(batch[j]) == sel[pos]


def test_read_cfr_bframes_exact(cfr_clip):
    h = cv.probe_file(cfr_clip, index="exact")
    _assert_read_patterns(h, len(h))


def test_read_vfr_exact(vfr_clip):
    h = cv.probe_file(vfr_clip, index="exact")
    _assert_read_patterns(h, len(h))


def test_read_prores_alpha(prores_clip):
    h = cv.probe_file(prores_clip, index="exact")
    batch = cv.read(h, [0, 5, 10])
    assert batch.shape[-1] == 4
    assert batch[..., 3].shape[-1] == batch.shape[-1] or batch[..., 3].ndim == 3
    assert float(batch[..., 3].mean()) > 0.2


def test_read_10bit_uint16_preserves(ten_bit_clip):
    h = cv.probe_file(ten_bit_clip, index="exact")
    batch = cv.read(h, [0], fmt="uint16")
    assert batch.dtype == np.uint16
    assert int(batch.max()) > 255


def test_read_10bit_uint8_rounds_not_truncates(ten_bit_clip):
    h = cv.probe_file(ten_bit_clip, index="exact")
    u16 = cv.read(h, [0], fmt="uint16")
    u8 = cv.read(h, [0], fmt="uint8")
    expected = np.clip(np.rint(u16[0].astype(np.float32) / 65535.0 * 255.0), 0, 255).astype(np.uint8)
    np.testing.assert_array_equal(u8[0], expected)


def test_read_exr_hdr(exr_seq):
    """Float EXR keeps values above 1.0 (the pattern's white blocks carry
    red = 2.5) and still decodes to the right frame; integer reads clamp."""
    h = cv.probe_sequence(exr_seq)
    batch = cv.read(h, [3], fmt="float32")
    assert batch.dtype == np.float32
    assert float(batch[0, ..., 0].max()) == pytest.approx(2.5)
    assert cv.decode_index(batch[0]) == 3
    u8 = cv.read(h, [3], fmt="uint8")
    assert u8.dtype == np.uint8 and int(u8[0, ..., 0].max()) == 255
    assert cv.decode_index(u8[0]) == 3


def test_iter_chunks_once_in_order(cfr_clip):
    h = cv.probe_file(cfr_clip, index="exact")
    sel = list(h.selected_indices())
    parts = list(cv.iter_chunks(h, 13))
    got = np.concatenate(parts, axis=0)
    assert got.shape[0] == len(sel)
    full = cv.read(h, list(range(len(h))))
    for i in range(len(sel)):
        assert cv.decode_index(got[i]) == cv.decode_index(full[i])


def test_handle_trimmed_every(cfr_clip):
    h = cv.probe_file(cfr_clip)
    t = h.trimmed(10, 40).every(2)
    assert list(t.selected_indices()) == list(range(10, 40, 2))
    assert len(t) == 15


def test_fingerprint_stable_and_sensitive(cfr_clip):
    h = cv.probe_file(cfr_clip)
    assert h.fingerprint() == h.fingerprint()
    assert h.trimmed(0, 10).fingerprint() != h.fingerprint()
    assert h.every(2).fingerprint() != h.fingerprint()
    assert h.with_colorspace("ACEScg").fingerprint() != h.fingerprint()


def test_budget_math():
    fits = cv.frames_that_fit(1920, 1080, 3, "float32", fraction=0.25, available=8_000_000_000)
    bpf = cv.bytes_per_frame(1920, 1080, 3, "float32")
    assert fits * bpf <= 8_000_000_000 * 0.25 + bpf


def test_container_pool_closes(tmp_path, cfr_clip):
    pytest.importorskip("psutil")
    import psutil

    proc = psutil.Process()
    baseline = len(proc.open_files())
    paths = []
    for i in range(5):
        p = str(tmp_path / f"clip{i}.mp4")
        cv.make_test_clip("h264_cfr_b", p, frames=5, width=256, height=64)
        paths.append(p)
    for p in paths:
        h = cv.probe_file(p)
        cv.read(h, [0])
    cv.drain_container_pool()
    assert len(proc.open_files()) <= baseline + 2


def test_concurrent_reads(cfr_clip):
    h = cv.probe_file(cfr_clip, index="exact")
    positions_a = list(range(0, len(h), 3))
    positions_b = list(reversed(range(len(h))))
    out = {}

    def worker(key, pos):
        out[key] = cv.read(h, pos)

    t1 = threading.Thread(target=worker, args=("a", positions_a))
    t2 = threading.Thread(target=worker, args=("b", positions_b))
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    sel = list(h.selected_indices())
    for j, pos in enumerate(positions_a):
        assert cv.decode_index(out["a"][j]) == sel[pos]
    for j, pos in enumerate(positions_b):
        assert cv.decode_index(out["b"][j]) == sel[pos]


def test_colour_bt709_limited(tmp_path):
    pytest.importorskip("av")
    import av

    path = str(tmp_path / "colour.mp4")
    cv.make_colour_test_clip(path, frames=4, width=320, height=240)
    h = cv.probe_file(path, index="exact")
    assert h.colour.matrix == "bt709"
    assert h.colour.range == "tv"
    for fi in range(4):
        batch = cv.read(h, [fi], fmt="uint8")
        mid_y, mid_x = h.height // 2, h.width // 2
        got = batch[0, mid_y, mid_x]
        exp = np.array(cv.expected_flat_rgb_patch(fi), dtype=np.uint8)
        assert np.all(np.abs(got[:3].astype(np.int16) - exp.astype(np.int16)) <= 2), (
            f"frame {fi}: got {got[:3]} expected {exp}"
        )

    # Sanity: wrong matrix (BT.601) would miss by > 10 on red patch
    container = av.open(path)
    vs = container.streams.video[0]
    container.seek(0, stream=vs, backward=True)
    frame = next(container.decode(vs))
    import av.video.reformatter as vr

    ref = vr.VideoReformatter().reformat(
        frame,
        format="rgb24",
        src_colorspace=vr.Colorspace.ITU709,
        dst_colorspace=vr.Colorspace.ITU709,
        src_color_range=vr.ColorRange.MPEG,
        dst_color_range=vr.ColorRange.JPEG,
    ).to_ndarray(format="rgb24")
    wrong = vr.VideoReformatter().reformat(
        frame,
        format="rgb24",
        src_colorspace=vr.Colorspace.ITU601,
        dst_colorspace=vr.Colorspace.ITU601,
        src_color_range=vr.ColorRange.MPEG,
        dst_color_range=vr.ColorRange.JPEG,
    ).to_ndarray(format="rgb24")
    container.close()
    exp_r = cv.expected_flat_rgb_patch(0)[0]
    err_ref = abs(int(ref[120, 160, 0]) - exp_r)
    err_wrong = abs(int(wrong[120, 160, 0]) - exp_r)
    assert err_ref <= 2
    assert err_wrong > 10


@pytest.mark.slow
def test_chunk_1080p600_memory(tmp_path):
    pytest.importorskip("psutil")
    import psutil

    path = str(tmp_path / "big.mp4")
    cv.make_test_clip("h264_cfr_b", path, frames=600, width=1920, height=1080)
    h = cv.probe_file(path, index="auto")
    proc = psutil.Process()
    before = proc.memory_info().rss
    peak = before
    total = 0
    for chunk in cv.iter_chunks(h, 32):
        total += chunk.shape[0]
        peak = max(peak, proc.memory_info().rss)
    cv.drain_container_pool()
    assert total == len(h)
    assert (peak - before) < 1_500_000_000


# ── handle semantics the loaders rely on (VHS order: rate -> skip -> nth -> cap) ──

def test_selection_ops_compose_as_slices(cfr_clip):
    h = cv.probe_file(cfr_clip)
    assert list(h.every(2).trimmed(0, 10).selected_indices()) == list(range(0, 20, 2))
    assert list(h.trimmed(5).every(3).capped(4).selected_indices()) == [5, 8, 11, 14]
    assert list(h.capped(0).selected_indices()) == list(range(len(h)))


def test_retime_matches_vhs_rule_on_cfr(cfr_clip):
    """VHS force_rate on a CFR source: output j is source ceil(j * fps / rate)."""
    import math
    h = cv.probe_file(cfr_clip)             # 24 fps, 60 frames
    times = [i / 24 for i in range(h.frame_count)]
    for rate in (8, 12, 16, 30, 48):
        got = list(h.retimed(rate, times).selected_indices())
        want = []
        j = 0
        while math.ceil(j * 24 / rate - 1e-9) <= h.frame_count - 1:
            want.append(math.ceil(j * 24 / rate - 1e-9))
            j += 1
        assert got == want, rate
        assert h.retimed(rate, times).fps == rate


def test_retime_follows_real_timestamps(vfr_clip):
    """On a VFR clip the frame shown at each output time is judged by the
    file's own timestamps, not by index / nominal fps."""
    h = cv.probe_file(vfr_clip, index="exact")
    tbl = cv.get_index_table(h.source.stat_key)
    times = [float(tbl.pts_of(i) * tbl.time_base) for i in range(h.frame_count)]
    r = h.retimed(10, times)
    for j, src in enumerate(r.selected_indices()):
        t = times[0] + j / 10
        assert times[src] >= t - 1e-6
        assert src == 0 or times[src - 1] < t - 1e-6
    batch = cv.read(r, range(len(r)), fmt="uint8")
    assert [cv.decode_index(b) for b in batch] == list(r.selected_indices())


def test_scaled_reads_resize_at_decode(cfr_clip, exr_seq):
    h = cv.probe_file(cfr_clip).scaled(128, 96)
    b = cv.read(h, [0, 1], fmt="float32")
    assert b.shape == (2, 96, 128, 3)
    s = cv.probe_sequence(exr_seq).scaled(128, 32)
    e = cv.read(s, [3], fmt="float32")   # frame 0 is all-black blocks (index 0, CRC 0)
    assert e.shape == (1, 32, 128, 3) and float(e[0, ..., 0].max()) > 1.0   # HDR survives the resize
    assert h.fingerprint() != cv.probe_file(cfr_clip).fingerprint()
