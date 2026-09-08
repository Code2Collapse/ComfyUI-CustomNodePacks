"""Tests for dependency_checker.py — clean install, breaking torch, risky numpy."""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from dependency_checker import (
    CRITICAL_PACKAGES,
    check_conflicts,
    format_warning_message,
)


def _write_reqs(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "requirements.txt"
    p.write_text(textwrap.dedent(body).strip() + "\n", encoding="utf-8")
    return p


# ── 1. clean install — nothing currently installed ──────────────────
def test_clean_install_is_safe(tmp_path: Path) -> None:
    reqs = _write_reqs(tmp_path, """
        # fresh env: no packages installed
        opencv-python>=4.8,<5.0
        pillow>=10.0,<12.0
    """)
    report = check_conflicts(reqs, installed={})
    assert report.breaking == []
    assert report.risky == []
    assert {e.package for e in report.safe} == {"opencv-python", "pillow"}
    msg = format_warning_message(report)
    assert "BREAKING" not in msg
    assert "2 requirement(s) already satisfied" in msg or "satisfied" in msg


# ── 2. breaking torch conflict ──────────────────────────────────────
def test_breaking_torch_conflict_is_flagged(tmp_path: Path) -> None:
    reqs = _write_reqs(tmp_path, """
        torch==2.4.0
        numpy>=1.24,<2.0
    """)
    # User has torch 2.1.0 installed — incompatible with ==2.4.0
    installed = {"torch": "2.1.0", "numpy": "1.26.4"}
    report = check_conflicts(reqs, installed=installed)

    assert "torch" in CRITICAL_PACKAGES
    assert len(report.breaking) == 1
    e = report.breaking[0]
    assert e.package == "torch"
    assert e.installed == "2.1.0"
    assert "2.4.0" in e.required
    assert "CRITICAL" in e.reason

    # numpy 1.26.4 satisfies >=1.24,<2.0 → safe
    assert any(s.package == "numpy" for s in report.safe)
    msg = format_warning_message(report)
    assert "BREAKING" in msg
    assert "torch" in msg
    assert "do NOT auto-install" in msg


# ── 3. risky numpy conflict — unbounded critical package ────────────
def test_risky_unbounded_numpy_is_flagged(tmp_path: Path) -> None:
    reqs = _write_reqs(tmp_path, """
        # No upper bound on a CRITICAL package → risky even if currently satisfies
        numpy>=1.24
    """)
    installed = {"numpy": "1.26.4"}
    report = check_conflicts(reqs, installed=installed)

    assert report.breaking == []
    assert len(report.risky) == 1
    assert report.risky[0].package == "numpy"
    assert "no upper bound" in report.risky[0].reason

    msg = format_warning_message(report)
    assert "RISKY" in msg
    assert "numpy" in msg


# ── extra: non-critical mismatch is risky, not breaking ─────────────
def test_non_critical_mismatch_is_risky_not_breaking(tmp_path: Path) -> None:
    reqs = _write_reqs(tmp_path, "matplotlib>=4.0\n")
    installed = {"matplotlib": "3.8.0"}     # below required minimum
    report = check_conflicts(reqs, installed=installed)
    assert report.breaking == []
    assert len(report.risky) == 1
    assert report.risky[0].package == "matplotlib"


if __name__ == "__main__":           # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
