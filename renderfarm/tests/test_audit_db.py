"""Audit DB: async writer, schema, duration math, query filters, stats."""

import time

from renderfarm.logging.audit_db import AuditDB


def _mk(tmp_path):
    return AuditDB(path=str(tmp_path / "audit.db"))


def test_submit_and_query(tmp_path):
    db = _mk(tmp_path)
    db.record_submitted("j1", "aks", "alice", "projA", 7, "heavy_94gb", submitted_at=100.0)
    db.record_submitted("j2", "runpod", "bob", "projB", 3, "light_30gb", submitted_at=200.0)
    assert db.flush()
    rows = db.query()
    assert {r["job_id"] for r in rows} == {"j1", "j2"}
    assert rows[0]["job_id"] == "j2"  # newest first
    only_alice = db.query(user="alice")
    assert len(only_alice) == 1 and only_alice[0]["priority"] == 7
    assert db.query(project="projB")[0]["compute_profile"] == "light_30gb"


def test_completion_duration(tmp_path):
    db = _mk(tmp_path)
    db.record_submitted("j1", "aks", "alice", "p", 5, "heavy_94gb", submitted_at=1000.0)
    db.mark_completed("j1", "complete", completed_at=1042.5)
    assert db.flush()
    row = db.query()[0]
    assert row["status"] == "complete"
    assert abs(row["duration_seconds"] - 42.5) < 1e-6
    assert row["completed_at"] == 1042.5


def test_status_and_error(tmp_path):
    db = _mk(tmp_path)
    db.record_submitted("j1", "aks", "alice", "p", 5, "x")
    db.update_status("j1", "running")
    db.mark_completed("j1", "failed", error="CUDA out of memory")
    assert db.flush()
    row = db.query(status="failed")[0]
    assert row["error"] == "CUDA out of memory"


def test_writes_are_nonblocking(tmp_path):
    db = _mk(tmp_path)
    t0 = time.perf_counter()
    for i in range(500):
        db.record_submitted(f"j{i}", "aks", "alice", "p", 5, "x")
    enqueue_time = time.perf_counter() - t0
    # 500 enqueues must not wait on disk (writer thread drains later).
    assert enqueue_time < 0.5
    assert db.flush(timeout=10)
    assert len(db.query(limit=1000)) == 500


def test_stats(tmp_path):
    db = _mk(tmp_path)
    db.record_submitted("j1", "aks", "alice", "p", 5, "x", submitted_at=10)
    db.record_submitted("j2", "aks", "bob", "p", 5, "x", submitted_at=20)
    db.mark_completed("j1", "complete", completed_at=70)
    db.flush()
    s = db.stats()
    assert s["total_jobs"] == 2
    assert s["by_user"] == {"alice": 1, "bob": 1}
    assert s["avg_duration_seconds"] == 60.0
