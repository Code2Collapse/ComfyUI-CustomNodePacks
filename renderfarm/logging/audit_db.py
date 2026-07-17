"""RIB audit log — SQLite at renderfarm/config/audit.db (gitignored).

Writes are ASYNC: every mutation is queued to a single daemon writer thread
(SQLite wants one writing connection anyway), so node execution and the
spooler never block on disk I/O. Reads open a short-lived connection per
call — WAL mode makes concurrent read-while-write safe.
"""

from __future__ import annotations

import logging
import os
import queue
import sqlite3
import threading
import time

log = logging.getLogger("RIB.audit")

_CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config")
DB_PATH = os.path.join(_CONFIG_DIR, "audit.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id           TEXT PRIMARY KEY,
    backend_name     TEXT NOT NULL,
    user             TEXT NOT NULL,
    project_name     TEXT NOT NULL DEFAULT '',
    priority         INTEGER NOT NULL DEFAULT 5,
    compute_profile  TEXT NOT NULL DEFAULT '',
    submitted_at     REAL NOT NULL,
    completed_at     REAL,
    duration_seconds REAL,
    status           TEXT NOT NULL,
    error            TEXT
);
CREATE INDEX IF NOT EXISTS idx_jobs_user      ON jobs(user);
CREATE INDEX IF NOT EXISTS idx_jobs_project   ON jobs(project_name);
CREATE INDEX IF NOT EXISTS idx_jobs_submitted ON jobs(submitted_at);
"""

_SENTINEL = object()


class AuditDB:
    """Async-write / sync-read audit store. One instance per db path."""

    def __init__(self, path: str = DB_PATH):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Create schema synchronously once so readers never race the writer.
        with self._connect() as con:
            con.executescript(_SCHEMA)
        self._q: "queue.Queue" = queue.Queue()
        self._writer = threading.Thread(target=self._writer_loop, name="rib-audit-writer", daemon=True)
        self._writer.start()

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.path, timeout=10)
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA synchronous=NORMAL")
        return con

    def _writer_loop(self):
        con = self._connect()
        while True:
            item = self._q.get()
            try:
                if item is _SENTINEL:
                    self._q.task_done()
                    continue  # flush() marker — task_done is the signal
                sql, params = item
                con.execute(sql, params)
                con.commit()
            except Exception as exc:  # noqa: BLE001 — audit must never kill the farm
                log.error("audit write failed: %s (sql=%s)", exc, item)
            finally:
                if item is not _SENTINEL:
                    self._q.task_done()

    def _enqueue(self, sql: str, params: tuple):
        self._q.put((sql, params))

    def flush(self, timeout: float = 5.0) -> bool:
        """Block until all queued writes hit disk (tests / shutdown)."""
        deadline = time.time() + timeout
        while not self._q.empty():
            if time.time() > deadline:
                return False
            time.sleep(0.02)
        # one join-ish settle for the in-flight item
        time.sleep(0.05)
        return True

    # ── mutations (async) ────────────────────────────────────────────
    def record_submitted(self, job_id: str, backend_name: str, user: str,
                         project_name: str, priority: int, compute_profile: str,
                         submitted_at: float | None = None):
        self._enqueue(
            "INSERT OR REPLACE INTO jobs (job_id, backend_name, user, project_name, priority, "
            "compute_profile, submitted_at, status) VALUES (?,?,?,?,?,?,?,?)",
            (job_id, backend_name, user, project_name, int(priority), compute_profile,
             float(submitted_at if submitted_at is not None else time.time()), "queued"),
        )

    def update_status(self, job_id: str, status: str, error: str | None = None):
        self._enqueue("UPDATE jobs SET status=?, error=COALESCE(?, error) WHERE job_id=?",
                      (status, error, job_id))

    def mark_completed(self, job_id: str, status: str, error: str | None = None,
                       completed_at: float | None = None):
        t = float(completed_at if completed_at is not None else time.time())
        self._enqueue(
            "UPDATE jobs SET status=?, error=?, completed_at=?, "
            "duration_seconds=? - submitted_at WHERE job_id=?",
            (status, error, t, t, job_id),
        )

    def set_priority(self, job_id: str, priority: int):
        self._enqueue("UPDATE jobs SET priority=? WHERE job_id=?", (int(priority), job_id))

    # ── reads (sync, own connection) ─────────────────────────────────
    def query(self, user: str | None = None, project: str | None = None,
              status: str | None = None, limit: int = 100) -> list[dict]:
        sql = "SELECT * FROM jobs WHERE 1=1"
        params: list = []
        if user:
            sql += " AND user=?"; params.append(user)
        if project:
            sql += " AND project_name=?"; params.append(project)
        if status:
            sql += " AND status=?"; params.append(status)
        sql += " ORDER BY submitted_at DESC LIMIT ?"
        params.append(int(limit))
        with self._connect() as con:
            con.row_factory = sqlite3.Row
            return [dict(r) for r in con.execute(sql, params).fetchall()]

    def stats(self) -> dict:
        with self._connect() as con:
            con.row_factory = sqlite3.Row
            by_status = {r["status"]: r["n"] for r in
                         con.execute("SELECT status, COUNT(*) n FROM jobs GROUP BY status")}
            by_user = {r["user"]: r["n"] for r in
                       con.execute("SELECT user, COUNT(*) n FROM jobs GROUP BY user")}
            row = con.execute(
                "SELECT COUNT(*) n, AVG(duration_seconds) avg_s, SUM(duration_seconds) sum_s "
                "FROM jobs WHERE duration_seconds IS NOT NULL").fetchone()
        return {
            "total_jobs": sum(by_status.values()),
            "by_status": by_status,
            "by_user": by_user,
            "completed_with_duration": row["n"],
            "avg_duration_seconds": round(row["avg_s"], 2) if row["avg_s"] else None,
            "total_gpu_seconds": round(row["sum_s"], 2) if row["sum_s"] else None,
        }


_db_lock = threading.Lock()
_db: AuditDB | None = None


def get_audit_db(path: str = DB_PATH) -> AuditDB:
    """Process-wide singleton for the default path; tests pass their own path."""
    global _db
    if path != DB_PATH:
        return AuditDB(path)
    with _db_lock:
        if _db is None:
            _db = AuditDB(path)
        return _db
