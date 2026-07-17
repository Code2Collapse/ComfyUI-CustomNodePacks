"""C2C Farm render-farm nodes — Submit / ClusterStatus / JobHistory.

The graph-facing face of the renderfarm/ subsystem (Tractor-style spooler
dispatching full workflows to remote ComfyUI Docker backends). All inputs
are native Python INPUT_TYPES — no DOM/HTML parameters.
"""

from __future__ import annotations

import json
import logging
import time

log = logging.getLogger("C2C.Farm.nodes")

# Dashboard REST routes ride this module's import (guarded — a route failure
# must never take the node registrations down with it).
try:
    from ..renderfarm.api_routes import register_routes as _farm_register_routes
    _farm_register_routes()
except Exception as _exc:  # noqa: BLE001
    log.warning("C2C Farm dashboard routes not registered: %s", _exc)


def _backend_choices() -> list[str]:
    try:
        from ..renderfarm.user_config import list_backends
        names = [b["name"] for b in list_backends(enabled_only=False)]
        return names or ["<no backends configured>"]
    except Exception as exc:  # noqa: BLE001
        log.warning("C2C Farm: could not read backends.json: %s", exc)
        return ["<no backends configured>"]


def _profile_choices() -> list[str]:
    try:
        from ..renderfarm.user_config import compute_profiles
        profiles = list(compute_profiles().keys())
        return profiles or ["heavy_94gb", "light_30gb"]
    except Exception:  # noqa: BLE001
        return ["heavy_94gb", "light_30gb"]


def _progress_event(node_id, pct, label):
    try:
        from server import PromptServer
        PromptServer.instance.send_sync(
            "c2c.farm.progress",
            {"node": str(node_id) if node_id is not None else "",
             "pct": float(pct), "label": label})
    except Exception:  # noqa: BLE001 — progress must never break dispatch
        pass


class C2C_Submit:
    """Spool a full workflow JSON to a remote ComfyUI backend."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt_json": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": "The workflow to run remotely, in ComfyUI API format "
                               "(Save (API Format) / the {'node_id': {class_type, inputs}} map). "
                               "Accepts a bare prompt map or {'prompt': {...}}."}),
                "backend": (_backend_choices(), {
                    "tooltip": "Registered compute backend (renderfarm/config/backends.json)."}),
                "compute_profile": (_profile_choices(), {
                    "tooltip": "Gateway routing profile: heavy_94gb = 1 replica per 94GB GPU, "
                               "light_30gb = 3 replicas per GPU."}),
                "priority": ("INT", {"default": 5, "min": 1, "max": 10,
                                     "tooltip": "Spool priority — higher dispatches first."}),
                "project_name": ("STRING", {"default": "", "tooltip": "Audit-log project tag."}),
                "wait_for_completion": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "On: block until the remote render finishes and pull outputs "
                               "back into input/c2c_farm_results/<job>/. Off: return the job id "
                               "immediately and track it in the C2C Farm dashboard."}),
                "timeout_minutes": ("INT", {"default": 120, "min": 1, "max": 1440}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("job_id", "result_paths", "info")
    FUNCTION = "submit"
    CATEGORY = "MEC/RenderFarm"
    DESCRIPTION = "Dispatch a full workflow JSON to a remote ComfyUI render-farm backend."

    def submit(self, prompt_json, backend, compute_profile, priority, project_name,
               wait_for_completion, timeout_minutes, unique_id=None):
        from ..renderfarm import courier, preflight
        from ..renderfarm.backends import get_adapter
        from ..renderfarm.spooler.queue_manager import Job, get_queue_manager
        from ..renderfarm.user_config import get_backend, get_user, validate_profile

        user = get_user()  # fail-loud: C2C_USER_NAME must be set + declared
        if backend.startswith("<"):
            raise RuntimeError(
                "C2C Farm: no backends configured — register one in "
                "renderfarm/config/backends.json and set \"enabled\": true.")
        backend_cfg = get_backend(backend)   # fail-loud: unknown/disabled
        validate_profile(backend_cfg, compute_profile)
        projects = user.get("projects", [])
        if project_name and projects and "*" not in projects and project_name not in projects:
            raise RuntimeError(
                f"C2C Farm: user '{user['name']}' may not submit under project "
                f"'{project_name}'. Allowed: {projects} (config/users.json).")

        prompt = json.loads(prompt_json) if isinstance(prompt_json, str) else prompt_json
        if isinstance(prompt, dict) and "prompt" in prompt and isinstance(prompt["prompt"], dict):
            prompt = prompt["prompt"]
        if not isinstance(prompt, dict) or not prompt:
            raise RuntimeError(
                "C2C Farm: prompt_json is not a workflow. Export with 'Save (API Format)' "
                "and paste/pipe the JSON object here.")

        adapter = get_adapter(backend)
        _progress_event(unique_id, 0.05, "pre-flight node check")
        preflight.validate_backend_nodes(prompt, adapter, backend)  # fatality preventer

        _progress_event(unique_id, 0.10, "courier: media upload")
        prompt, uploads = courier.prepare_prompt(prompt, backend_cfg)

        job = Job(prompt_json=prompt, backend_name=backend, user=user["name"],
                  project_name=project_name, priority=int(priority),
                  compute_profile=compute_profile)
        qm = get_queue_manager()
        qm.submit(job)
        info = {"job_id": job.job_id, "backend": backend, "user": user["name"],
                "compute_profile": compute_profile, "priority": int(priority),
                "uploaded_media": len(uploads)}

        if not wait_for_completion:
            _progress_event(unique_id, 1.0, f"spooled {job.job_id}")
            return (job.job_id, "", json.dumps({**info, "status": "spooled"}, indent=2))

        deadline = time.time() + timeout_minutes * 60
        while not job.done.wait(timeout=2.0):
            if time.time() > deadline:
                raise RuntimeError(
                    f"C2C Farm: job {job.job_id} exceeded timeout_minutes={timeout_minutes} "
                    f"(status={job.status}). It keeps running — watch it in the C2C Farm "
                    f"dashboard or re-fetch with C2C_JobHistory.")
            pct = 0.15 + 0.8 * float(job.progress or 0.0)
            _progress_event(unique_id, pct, f"{job.status} on {backend}")
        if job.status != "complete":
            raise RuntimeError(f"C2C Farm: job {job.job_id} {job.status}: {job.error}")
        _progress_event(unique_id, 1.0, "complete")
        return (job.job_id, "\n".join(job.result_paths),
                json.dumps({**info, "status": job.status,
                            "results": job.result_paths}, indent=2))


class C2C_ClusterStatus:
    """Live capacity of every registered backend, as JSON."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "backend": (["all"] + _backend_choices(), {
                "tooltip": "One backend, or 'all' for the whole cluster."}),
        }}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("cluster_json",)
    FUNCTION = "status"
    CATEGORY = "MEC/RenderFarm"
    DESCRIPTION = "Live render-farm capacity (queue depth, VRAM, reachability) as JSON."

    @classmethod
    def IS_CHANGED(cls, backend):
        # Live cluster state genuinely changes — refresh every ~5s bucket.
        return int(time.time() // 5)

    def status(self, backend):
        from ..renderfarm.backends import get_adapter
        from ..renderfarm.user_config import list_backends
        report = []
        for cfg in list_backends(enabled_only=False):
            if backend != "all" and cfg["name"] != backend:
                continue
            if not cfg.get("enabled"):
                report.append({"backend": cfg["name"], "reachable": False, "disabled": True})
                continue
            try:
                report.append(get_adapter(cfg["name"]).capacity())
            except Exception as exc:  # noqa: BLE001
                report.append({"backend": cfg["name"], "reachable": False,
                               "error": str(exc)[:200]})
        return (json.dumps({"backends": report}, indent=2),)


class C2C_JobHistory:
    """Query the SQLite audit log (who ran what, where, how long)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "limit": ("INT", {"default": 50, "min": 1, "max": 1000}),
            "user_filter": ("STRING", {"default": "", "tooltip": "Empty = all users."}),
            "project_filter": ("STRING", {"default": "", "tooltip": "Empty = all projects."}),
        }}

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("stats_json", "jobs_table")
    FUNCTION = "history"
    CATEGORY = "MEC/RenderFarm"
    DESCRIPTION = "Render-farm audit log: per-user/status stats + recent jobs table."

    @classmethod
    def IS_CHANGED(cls, limit, user_filter, project_filter):
        # Re-run only when the audit DB actually changed on disk.
        import os
        from ..renderfarm.logging.audit_db import DB_PATH
        try:
            return os.path.getmtime(DB_PATH)
        except OSError:
            return 0

    def history(self, limit, user_filter, project_filter):
        from ..renderfarm.logging.audit_db import get_audit_db
        db = get_audit_db()
        rows = db.query(user=user_filter or None, project=project_filter or None,
                        limit=int(limit))
        header = f"{'job_id':<14}{'user':<12}{'project':<14}{'backend':<16}" \
                 f"{'prio':<5}{'status':<10}{'dur(s)':<8}"
        lines = [header, "-" * len(header)]
        for r in rows:
            dur = f"{r['duration_seconds']:.0f}" if r.get("duration_seconds") else "-"
            lines.append(f"{r['job_id']:<14}{r['user']:<12}{r['project_name']:<14}"
                         f"{r['backend_name']:<16}{r['priority']:<5}{r['status']:<10}{dur:<8}")
        return (json.dumps(db.stats(), indent=2), "\n".join(lines))


NODE_CLASS_MAPPINGS = {
    "C2C_Submit": C2C_Submit,
    "C2C_ClusterStatus": C2C_ClusterStatus,
    "C2C_JobHistory": C2C_JobHistory,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "C2C_Submit": "C2C Farm Submit — Remote Render",
    "C2C_ClusterStatus": "C2C Farm Cluster Status",
    "C2C_JobHistory": "C2C Farm Job History (Audit Log)",
}
