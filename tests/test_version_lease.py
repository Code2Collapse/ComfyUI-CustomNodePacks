"""One version per run, shared by the outputs that belong together.

The reported bug: two Folder Version Incrementers in one workflow, one run,
two different versions - the mov in v004 and the exr in v005 - which makes the
render impossible to hand over as one deliverable.

The three properties that have to hold:

  * SHARING. Two incrementers in one run, same family, get the same number.
  * SEPARATION. Two incrementers in one run, DIFFERENT families, do not.
    This is the part that keeps "one version per run" from forcing an
    unrelated job onto a shared number.
  * ORDER INDEPENDENCE. The number must not depend on which node happened to
    run first. This is the one that prevents silent overwrites: a family whose
    trees sit at different depths would otherwise hand out a version that
    already exists in one of them.

CPU-only, no ComfyUI.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PACK = Path(__file__).resolve().parents[1]
if str(PACK) not in sys.path:
    sys.path.insert(0, str(PACK))

import _version_lease as vl  # noqa: E402


@pytest.fixture(autouse=True)
def clean():
    vl.reset_for_tests()
    yield
    vl.reset_for_tests()


def counting(value):
    """A compute function that records how often it was actually called."""
    calls = []

    def compute():
        calls.append(1)
        return value

    compute.calls = calls
    return compute


# ── sharing ─────────────────────────────────────────────────────────────────

def test_a_second_ask_in_the_same_run_reuses_the_first_answer():
    """THE reported bug: one run must not produce two versions."""
    first, shared_a = vl.lease("run-1", "k", counting(4))
    second, shared_b = vl.lease("run-1", "k", counting(9))
    assert first == 4 and second == 4
    assert shared_a is False and shared_b is True


def test_the_second_ask_does_not_even_scan():
    """Rescanning is how the second node saw the first node's reserved
    directory and took the next number."""
    c = counting(4)
    vl.lease("run-1", "k", c)
    vl.lease("run-1", "k", c)
    assert len(c.calls) == 1


def test_four_outputs_in_one_run_all_get_one_version():
    """The actual shape of the request: mov, exr, png, masked mov."""
    versions = [vl.lease("run-1", "shot_a", counting(7))[0] for _ in range(4)]
    assert versions == [7, 7, 7, 7]


def test_a_new_run_gets_a_new_number():
    assert vl.lease("run-1", "k", counting(4))[0] == 4
    assert vl.lease("run-2", "k", counting(5))[0] == 5


# ── separation ──────────────────────────────────────────────────────────────

def test_different_families_in_one_run_keep_their_own_versions():
    """Without this, 'one version per run' would force an unrelated job onto
    the same number - the risk that had to be removed."""
    a, _ = vl.lease("run-1", "shot_a", counting(4))
    b, _ = vl.lease("run-1", "other_job", counting(11))
    assert a == 4 and b == 11


def test_no_execution_context_means_no_sharing():
    """Outside a run there is nothing to share within, so it behaves exactly
    as it did before leases existed rather than crashing."""
    a, shared_a = vl.lease(None, "k", counting(4))
    b, shared_b = vl.lease(None, "k", counting(5))
    assert (a, b) == (4, 5)
    assert shared_a is False and shared_b is False


# ── the family key ──────────────────────────────────────────────────────────

def test_a_suffixed_folder_shares_the_family_of_its_base():
    """shot_a and shot_a_mask are the same shot. This is what makes the
    sharing correct rather than merely consistent."""
    base = vl.family_key("/out", "shot_a", "09-22-2026", suffix="")
    mask = vl.family_key("/out", "shot_a_mask", "09-22-2026", suffix="_mask")
    assert base == mask


def test_an_unrelated_folder_is_a_different_family():
    a = vl.family_key("/out", "shot_a", "09-22-2026", suffix="")
    b = vl.family_key("/out", "some_other_job", "09-22-2026", suffix="")
    assert a != b


def test_a_different_date_is_a_different_family():
    a = vl.family_key("/out", "shot_a", "09-22-2026", suffix="")
    b = vl.family_key("/out", "shot_a", "09-23-2026", suffix="")
    assert a != b


def test_a_different_base_directory_is_a_different_family():
    a = vl.family_key("/out", "shot_a", "09-22-2026")
    b = vl.family_key("/other", "shot_a", "09-22-2026")
    assert a != b


def test_an_explicit_group_overrides_everything():
    """The escape hatch: two outputs that belong together but whose folder
    names do not say so."""
    a = vl.family_key("/out", "wildly_different", "09-22-2026", group="deliver")
    b = vl.family_key("/other", "nothing_alike", "01-01-2020", group="deliver")
    assert a == b


def test_a_blank_group_falls_back_to_the_derived_key():
    assert vl.family_key("/out", "shot_a", "d", group="   ") == \
        vl.family_key("/out", "shot_a", "d")


# ── stripping the suffix ────────────────────────────────────────────────────

def test_an_exact_trailing_suffix_is_removed():
    assert vl.strip_suffix("shot_a_mask", "_mask") == "shot_a"


def test_a_suffix_that_is_not_there_changes_nothing():
    assert vl.strip_suffix("shot_a", "_mask") == "shot_a"


def test_a_suffix_in_the_middle_is_not_removed():
    """Only a trailing match. Removing it anywhere would merge families that
    merely look alike, and handing one shot's version to another is worse than
    not merging at all."""
    assert vl.strip_suffix("shot_mask_final", "_mask") == "shot_mask_final"


def test_a_folder_that_is_entirely_the_suffix_is_left_alone():
    """Stripping it would leave an empty stem, which matches every folder."""
    assert vl.strip_suffix("_mask", "_mask") == "_mask"


def test_no_suffix_is_a_no_op():
    assert vl.strip_suffix("shot_a", "") == "shot_a"


# ── order independence ──────────────────────────────────────────────────────

def _tree(tmp_path, name, date, versions):
    for v in versions:
        (tmp_path / name / date / f"v{v:03d}").mkdir(parents=True, exist_ok=True)


def fake_scan(scan_dir, prefix, padding):
    """The node's own scanner, reimplemented minimally for the test."""
    import re
    p = Path(scan_dir)
    if not p.is_dir():
        return 1
    pat = re.compile(rf"^{re.escape(prefix)}(\d{{{padding},}})$")
    best = 0
    for e in p.iterdir():
        m = pat.match(e.name) if e.is_dir() else None
        if m:
            best = max(best, int(m.group(1)))
    return best + 1


def test_the_family_max_is_taken_not_just_one_tree(tmp_path):
    """THE overwrite guard. shot_a holds v001-v004 and shot_a_mask only
    v001-v002. If the mask node runs first and leases v003 from its own tree,
    the beauty pass is then handed v003 - and overwrites finished work."""
    date = "09-22-2026"
    _tree(tmp_path, "shot_a", date, [1, 2, 3, 4])
    _tree(tmp_path, "shot_a_mask", date, [1, 2])

    assert fake_scan(tmp_path / "shot_a_mask" / date, "v", 3) == 3, \
        "the fixture does not reproduce the mismatch"

    nxt = vl.scan_family_next(tmp_path, "shot_a", date, "v", 3, fake_scan)
    assert nxt == 5, f"took {nxt} - the deeper tree was not considered"


def test_the_answer_does_not_depend_on_who_asks_first(tmp_path):
    date = "09-22-2026"
    _tree(tmp_path, "shot_a", date, [1, 2, 3, 4])
    _tree(tmp_path, "shot_a_mask", date, [1, 2])

    from_beauty = vl.scan_family_next(tmp_path, "shot_a", date, "v", 3, fake_scan)
    from_mask = vl.scan_family_next(tmp_path, "shot_a", date, "v", 3, fake_scan)
    assert from_beauty == from_mask == 5


def test_an_unrelated_tree_is_not_counted(tmp_path):
    """some_other_job holding v099 must not push shot_a to v100."""
    date = "09-22-2026"
    _tree(tmp_path, "shot_a", date, [1])
    _tree(tmp_path, "some_other_job", date, [99])
    assert vl.scan_family_next(tmp_path, "shot_a", date, "v", 3, fake_scan) == 2


def test_an_empty_stem_does_not_match_every_folder(tmp_path):
    """startswith('') is True for everything, which would make one job's
    version depend on every unrelated folder in the output directory."""
    date = "09-22-2026"
    _tree(tmp_path, "anything", date, [50])
    assert vl.scan_family_next(tmp_path, "", date, "v", 3, fake_scan) == 1


def test_a_missing_base_directory_starts_at_one(tmp_path):
    assert vl.scan_family_next(tmp_path / "nope", "shot_a", "d", "v", 3,
                               fake_scan) == 1


def test_a_family_with_no_versions_yet_starts_at_one(tmp_path):
    (tmp_path / "shot_a").mkdir()
    assert vl.scan_family_next(tmp_path, "shot_a", "09-22-2026", "v", 3,
                               fake_scan) == 1


# ── housekeeping ────────────────────────────────────────────────────────────

def test_old_runs_are_forgotten():
    """A long session queues hundreds of prompts; remembering every one would
    grow without limit."""
    for i in range(vl.MAX_REMEMBERED_RUNS + 8):
        vl.lease(f"run-{i}", "k", counting(i))
    assert len(vl._LEASES) <= vl.MAX_REMEMBERED_RUNS


def test_the_current_run_survives_housekeeping():
    vl.lease("keep-me", "k", counting(1))
    for i in range(vl.MAX_REMEMBERED_RUNS - 1):
        vl.lease(f"run-{i}", "k", counting(i))
    # touched most recently of the old ones, but "keep-me" was first...
    # re-asking must still find it, which is what the LRU ordering is for
    again, shared = vl.lease("keep-me", "k", counting(999))
    assert (again, shared) == (1, True)


def test_the_run_id_reader_never_raises_outside_an_execution():
    """It is read defensively: on a build without the contextvar every call
    returns None and the lease degrades to the old behaviour."""
    assert vl.current_run_id() is None or isinstance(vl.current_run_id(), str)


# ── the report ──────────────────────────────────────────────────────────────

def test_the_report_says_when_a_number_was_inherited():
    """A user who set padding on the second incrementer and saw it ignored
    needs to know the number came from elsewhere, not that their setting is
    broken."""
    text = vl.describe(4, shared=True, key="shot_a", family=[])
    assert "already decided by an earlier node" in text
    assert "was not used" in text


def test_the_report_says_when_this_node_decided():
    text = vl.describe(4, shared=False, key="shot_a", family=[])
    assert "first node in the run to ask" in text


def test_the_report_names_the_family_it_scanned():
    text = vl.describe(5, shared=False, key="shot_a",
                       family=["shot_a", "shot_a_mask"])
    assert "shot_a_mask" in text
    assert "which node ran first" in text
