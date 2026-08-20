"""Resilience of the spooler and courier: job timeout, submit retry, restart
reconciliation, and the `mounted` media-rewrite mode.

Each test drives the REAL code path with a fake that reproduces the specific
production failure, so a regression fails here rather than on the farm.
"""

import time

from renderfarm.courier import prepare_prompt
from renderfarm.spooler.queue_manager import Job, QueueManager


class HangingAdapter:
    """Accepts the job, then never reaches a terminal status.

    This is the backend that quietly wedges: the pod is alive and answering,
    but the render never finishes and never errors.
    """

    def __init__(self):
        self.polls = 0

    def submit(self, prompt_json, compute_profile, priority):
        return "remote-hang"

    def get_status(self, rid):
        self.polls += 1
        return {"status": "running", "progress_pct": 0.5, "error": None}

    def get_result(self, rid):
        return []

    def cancel(self, rid):
        return True


class FlakyAdapter:
    """Fails `fail_times` submits, then succeeds and completes immediately."""

    def __init__(self, fail_times=1):
        self.fail_times = fail_times
        self.attempts = 0

    def submit(self, prompt_json, compute_profile, priority):
        self.attempts += 1
        if self.attempts <= self.fail_times:
            raise ConnectionError("backend momentarily unreachable")
        return "remote-ok"

    def get_status(self, rid):
        return {"status": "complete", "progress_pct": 1.0, "error": None}

    def get_result(self, rid):
        return ["/tmp/out.png"]

    def cancel(self, rid):
        return True


def _job(**kw):
    return Job(prompt_json={"1": {"class_type": "X", "inputs": {}}},
               backend_name="aks", user="alice", **kw)


def test_hanging_backend_times_out_instead_of_pinning_the_thread():
    ad = HangingAdapter()
    qm = QueueManager(lambda n: ad, lambda u: 2, lambda b: 2, audit=None,
                      poll_interval=0.01, autostart=False, job_timeout=0.15)
    job = _job()
    qm._run_job(job)
    assert job.status == "failed"
    assert "timed out" in (job.error or "")
    # It must also warn that the remote job may still be alive — silently
    # marking it failed would invite a duplicate resubmit.
    assert "may still be running" in (job.error or "")
    assert ad.polls > 0


def test_submit_retries_a_transient_failure():
    ad = FlakyAdapter(fail_times=1)
    qm = QueueManager(lambda n: ad, lambda u: 2, lambda b: 2, audit=None,
                      poll_interval=0.01, autostart=False,
                      submit_attempts=2, retry_wait=0.01)
    job = _job()
    qm._run_job(job)
    assert ad.attempts == 2, "should have retried exactly once"
    assert job.status == "complete"


def test_submit_gives_up_after_the_attempt_budget():
    ad = FlakyAdapter(fail_times=99)
    qm = QueueManager(lambda n: ad, lambda u: 2, lambda b: 2, audit=None,
                      poll_interval=0.01, autostart=False,
                      submit_attempts=2, retry_wait=0.01)
    job = _job()
    qm._run_job(job)
    assert ad.attempts == 2
    assert job.status == "failed"
    assert "unreachable" in (job.error or "")


def test_polling_is_never_retried_after_a_successful_submit():
    """A submit that succeeded must not be re-sent — that would double-bill."""

    class DiesWhilePolling:
        def __init__(self):
            self.submits = 0

        def submit(self, p, c, pr):
            self.submits += 1
            return "remote-1"

        def get_status(self, rid):
            raise ConnectionError("lost the backend mid-render")

        def get_result(self, rid):
            return []

        def cancel(self, rid):
            return True

    ad = DiesWhilePolling()
    qm = QueueManager(lambda n: ad, lambda u: 2, lambda b: 2, audit=None,
                      poll_interval=0.01, autostart=False,
                      submit_attempts=3, retry_wait=0.01)
    job = _job()
    qm._run_job(job)
    assert ad.submits == 1, "the prompt must only ever be submitted once"
    assert job.status == "failed"


class FakeAudit:
    """Just enough AuditDB to exercise reconciliation."""

    def __init__(self, rows):
        self.rows = rows
        self.updates = []

    def query(self, user=None, project=None, status=None, limit=100):
        return [r for r in self.rows if status is None or r["status"] == status]

    def update_status(self, job_id, status, error=None):
        self.updates.append((job_id, status, error))


def test_restart_closes_out_ghost_jobs_without_resubmitting():
    audit = FakeAudit([
        {"job_id": "a1", "status": "queued", "backend_name": "aks"},
        {"job_id": "a2", "status": "running", "backend_name": "aks"},
        {"job_id": "a3", "status": "complete", "backend_name": "aks"},
    ])
    QueueManager(lambda n: None, lambda u: 2, lambda b: 2, audit=audit,
                 poll_interval=0.01, autostart=False)
    touched = {j for j, _, _ in audit.updates}
    assert touched == {"a1", "a2"}, "only mid-flight rows should be closed out"
    assert all(s == "failed" for _, s, _ in audit.updates)
    assert all("restart" in (e or "") for _, _, e in audit.updates)


def test_mounted_mode_passes_the_filename_not_a_url(tmp_path):
    """BlobFUSE/NFS backends already see the file; a URL is wrong there."""
    media = tmp_path / "plate.0001.exr"
    media.write_bytes(b"x")

    class FakeStorage:
        def __init__(self):
            self.uploaded = []

        def upload(self, p):
            self.uploaded.append(p)
            return "https://blob.example/c2c-farm/abc/plate.0001.exr?sig=xyz"

    prompt = {"1": {"class_type": "LoadImage",
                    "inputs": {"image": str(media)}}}
    st = FakeStorage()
    out, uploads = prepare_prompt(
        prompt, {"name": "aks", "media_rewrite": "mounted"}, storage=st,
        resolver=lambda v: str(media) if v == str(media) else None)

    assert out["1"]["inputs"]["image"] == "plate.0001.exr"
    assert out["1"]["class_type"] == "LoadImage", "node type must not change"
    # The upload still has to happen — that is what puts the bytes where the
    # mount can see them.
    assert st.uploaded == [str(media)]
    assert uploads


def test_in_place_mode_still_rewrites_to_a_url(tmp_path):
    media = tmp_path / "plate.exr"
    media.write_bytes(b"x")
    url = "https://blob.example/plate.exr?sig=xyz"

    class FakeStorage:
        def upload(self, p):
            return url

    prompt = {"1": {"class_type": "LoadImage", "inputs": {"image": str(media)}}}
    out, _ = prepare_prompt(prompt, {"name": "lan"}, storage=FakeStorage(),
                            resolver=lambda v: str(media) if v == str(media) else None)
    assert out["1"]["inputs"]["image"] == url


def test_unknown_rewrite_mode_fails_loudly(tmp_path):
    media = tmp_path / "p.exr"
    media.write_bytes(b"x")

    class FakeStorage:
        def upload(self, p):
            return "u"

    prompt = {"1": {"class_type": "LoadImage", "inputs": {"image": str(media)}}}
    try:
        prepare_prompt(prompt, {"name": "aks", "media_rewrite": "blobfuse"},
                       storage=FakeStorage(),
                       resolver=lambda v: str(media) if v == str(media) else None)
    except RuntimeError as exc:
        assert "media_rewrite" in str(exc) and "mounted" in str(exc)
    else:
        raise AssertionError("a typo'd rewrite mode must not be silently ignored")


def test_progress_dict_is_bounded():
    from renderfarm.backends.comfyui_native_adapter import _WsTap
    tap = _WsTap.__new__(_WsTap)
    tap.progress = {}
    tap.previews = {}
    tap._current_prompt = None
    tap.latest_preview = None
    import json
    for i in range(400):
        tap._on_text(json.dumps({"type": "progress",
                                 "data": {"prompt_id": f"p{i}", "value": 1, "max": 2}}))
    assert len(tap.progress) <= 256, "unbounded progress dict leaks on a long-lived tap"
