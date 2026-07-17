"""Spooler: priority ordering, per-user + per-backend caps, bump/pause/cancel,
and the full dispatch lifecycle against a fake adapter."""

import threading
import time

from renderfarm.spooler.queue_manager import Job, QueueManager


class FakeAdapter:
    """Deterministic backend: completes after `steps` status polls."""

    def __init__(self, steps=2, fail=False):
        self.steps = steps
        self.fail = fail
        self.submitted = []
        self.cancelled = []
        self._polls = {}

    def submit(self, prompt_json, compute_profile, priority):
        rid = f"remote-{len(self.submitted)}"
        self.submitted.append((rid, compute_profile, priority))
        self._polls[rid] = 0
        return rid

    def get_status(self, rid):
        self._polls[rid] += 1
        if self._polls[rid] >= self.steps:
            if self.fail:
                return {"status": "error", "progress_pct": None, "error": "boom"}
            return {"status": "complete", "progress_pct": 1.0, "error": None}
        return {"status": "running", "progress_pct": 0.5, "error": None}

    def get_result(self, rid):
        return [f"/tmp/{rid}.png"]

    def cancel(self, rid):
        self.cancelled.append(rid)
        return True


def _qm(adapter, user_cap=2, backend_cap=2, autostart=False):
    return QueueManager(lambda name: adapter, lambda u: user_cap,
                        lambda b: backend_cap, audit=None,
                        poll_interval=0.01, autostart=autostart)


def _job(user="alice", prio=5, backend="aks", submitted_at=None, **kw):
    j = Job(prompt_json={"1": {"class_type": "X", "inputs": {}}},
            backend_name=backend, user=user, priority=prio, **kw)
    if submitted_at is not None:
        j.submitted_at = submitted_at
    return j


# ── ordering ─────────────────────────────────────────────────────────
def test_priority_desc_then_fifo():
    qm = _qm(FakeAdapter(), backend_cap=0)  # capacity 0: nothing dispatches
    a = qm.submit(_job(prio=5, submitted_at=100))
    b = qm.submit(_job(prio=9, submitted_at=200))
    c = qm.submit(_job(prio=5, submitted_at=50))
    order = [j["job_id"] for j in qm.snapshot()["pending"]]
    assert order == [b.job_id, c.job_id, a.job_id]  # 9 first, then older 5


def test_bump_reorders_queue():
    qm = _qm(FakeAdapter(), backend_cap=0)
    a = qm.submit(_job(prio=5, submitted_at=100))
    b = qm.submit(_job(prio=5, submitted_at=200))
    assert qm.bump(b.job_id, 8)
    assert [j["job_id"] for j in qm.snapshot()["pending"]][0] == b.job_id
    assert not qm.bump("nope", 9)


# ── capacity enforcement ─────────────────────────────────────────────
def test_per_user_cap_holds_jobs_locally():
    ad = FakeAdapter(steps=10_000)  # never completes during test
    qm = _qm(ad, user_cap=1, backend_cap=10)
    j1 = qm.submit(_job(user="alice"))
    j2 = qm.submit(_job(user="alice"))
    j3 = qm.submit(_job(user="bob"))
    dispatched = qm.schedule_once()
    ids = {j.job_id for j in dispatched}
    assert j1.job_id in ids and j3.job_id in ids and j2.job_id not in ids
    # second pass: alice still at her cap of 1
    assert qm.schedule_once() == []
    assert [j["job_id"] for j in qm.snapshot()["pending"]] == [j2.job_id]


def test_backend_cap_holds_jobs_locally():
    ad = FakeAdapter(steps=10_000)
    qm = _qm(ad, user_cap=10, backend_cap=1)
    qm.submit(_job(user="alice"))
    qm.submit(_job(user="bob"))
    assert len(qm.schedule_once()) == 1
    assert len(qm.snapshot()["pending"]) == 1


# ── lifecycle ────────────────────────────────────────────────────────
def test_dispatch_to_completion():
    ad = FakeAdapter(steps=2)
    qm = _qm(ad, autostart=True)
    job = qm.submit(_job(prio=7))
    done = qm.wait(job.job_id, timeout=5)
    assert done.status == "complete"
    assert done.result_paths == ["/tmp/remote-0.png"]
    assert ad.submitted[0][2] == 7  # priority forwarded to the backend


def test_failure_propagates_error():
    qm = _qm(FakeAdapter(steps=1, fail=True), autostart=True)
    job = qm.submit(_job())
    done = qm.wait(job.job_id, timeout=5)
    assert done.status == "failed" and done.error == "boom"


def test_pause_resume_cancel_pending():
    qm = _qm(FakeAdapter(), backend_cap=0)
    j = qm.submit(_job())
    assert qm.pause(j.job_id)
    assert qm.snapshot()["pending"][0]["status"] == "paused"
    assert qm.schedule_once() == []  # paused jobs never dispatch
    assert qm.resume(j.job_id)
    assert qm.cancel(j.job_id)
    assert qm.snapshot()["pending"] == []
    assert qm.get(j.job_id).status == "cancelled"
    assert j.done.is_set()


def test_cancel_running_calls_backend():
    ad = FakeAdapter(steps=10_000)
    qm = _qm(ad, autostart=False)
    j = qm.submit(_job())
    qm.schedule_once()
    deadline = time.time() + 2
    while j.remote_id is None and time.time() < deadline:
        time.sleep(0.01)
    assert j.remote_id is not None
    assert qm.cancel(j.job_id)
    assert ad.cancelled == [j.remote_id]
    assert j.status == "cancelled"
