"""C2C Farm render-farm nodes — Submit / ClusterStatus / JobHistory.

The graph-facing face of the renderfarm/ subsystem (Tractor-style spooler
dispatching full workflows to remote ComfyUI Docker backends). All inputs
are native Python INPUT_TYPES — no DOM/HTML parameters.
"""

from __future__ import annotations

import json
import logging
import time
try:                                                       # A28 layout declarations (display only)
    from ._layout_decl import with_layout
except ImportError:  # loaded as a plain file (tests): the same module by path, never an absolute import
    try:
        import importlib.util as _c2c_ilu
        import os as _c2c_os
        _c2c_spec = _c2c_ilu.spec_from_file_location("c2c_layout_by_path", _c2c_os.path.join(
            _c2c_os.path.dirname(_c2c_os.path.abspath(__file__)), '_layout_decl.py'))
        _c2c_mod = _c2c_ilu.module_from_spec(_c2c_spec)
        _c2c_spec.loader.exec_module(_c2c_mod)
        with_layout = _c2c_mod.with_layout
    except Exception:  # pragma: no cover - layout keys are display only: never take the node down
        def with_layout(spec, layout):
            return spec

log = logging.getLogger("C2C.Farm.nodes")

# Dashboard REST routes ride this module's import (guarded — a route failure
# must never take the node registrations down with it).
try:
    from ..renderfarm.api_routes import register_routes as _farm_register_routes
    _farm_register_routes()
except Exception as _exc:  # noqa: BLE001
    log.warning("C2C Farm dashboard routes not registered: %s", _exc)


def _cancel_requested() -> bool:
    """ComfyUI's Cancel / Interrupt flag (False outside ComfyUI)."""
    try:
        import comfy.model_management as mm
        return bool(mm.processing_interrupted())
    except Exception:  # noqa: BLE001 - no ComfyUI (tests, tools)
        return False


def _raise_interrupted() -> None:
    """Raise ComfyUI's own interrupt exception (clears the flag, ends the prompt as 'cancelled', not as an error)."""
    import comfy.model_management as mm
    mm.throw_exception_if_processing_interrupted()
    raise RuntimeError("C2C Farm: cancelled")                 # the flag was cleared in between: still stop


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


# ── L9.174 (A25): dependency / model manifest + node-version check (Submit) ────────────────────────────────────────
_MODEL_EXTS = (".safetensors", ".ckpt", ".pt", ".pth", ".bin", ".gguf", ".sft", ".onnx")
DEPENDENCY_CHECKS = ["off", "report", "block"]


def _is_link(v) -> bool:
    return isinstance(v, list) and len(v) == 2 and isinstance(v[1], int) and isinstance(v[0], (str, int))


def build_manifest(prompt: dict) -> dict:
    """What the workflow needs, read locally from the prompt (no network)."""
    classes = sorted({n["class_type"] for n in prompt.values() if isinstance(n, dict) and n.get("class_type")})
    models = []
    for nid, node in prompt.items():
        if not isinstance(node, dict):
            continue
        for name, val in (node.get("inputs") or {}).items():
            if isinstance(val, str) and val.lower().endswith(_MODEL_EXTS):
                models.append({"node_id": str(nid), "class_type": node.get("class_type"), "input": name,
                               "value": val})
    return {"node_classes": classes, "model_files": models, "checked": False}


def _backend_object_info(adapter):
    """The backend's full /object_info, or None when the adapter cannot fetch it."""
    if hasattr(adapter, "object_info"):
        return adapter.object_info()
    if hasattr(adapter, "_request"):
        return adapter._request("GET", "/object_info", timeout=(5, 60)).json()
    return None


def _combo_options(spec):
    """Options of a combo input spec, V1 [[...], {...}] or V3 ["COMBO", {"options": [...]}]; None = not a combo."""
    if not isinstance(spec, (list, tuple)) or not spec:
        return None
    if isinstance(spec[0], (list, tuple)):
        return [str(x) for x in spec[0]]
    if spec[0] == "COMBO" and len(spec) > 1 and isinstance(spec[1], dict):
        opts = spec[1].get("options")
        return [str(x) for x in opts] if isinstance(opts, (list, tuple)) else None
    return None


def check_dependencies(prompt: dict, object_info: dict, manifest: dict) -> dict:
    """Compare the workflow with the backend's /object_info: missing node classes, model files / combo values the
    backend does not offer, inputs its version of a node does not declare, required inputs the workflow leaves out."""
    missing_classes = sorted(c for c in manifest["node_classes"] if c not in object_info)
    missing_values, unknown_inputs, missing_required = [], [], []
    for nid, node in prompt.items():
        if not isinstance(node, dict):
            continue
        ct = node.get("class_type")
        info = object_info.get(ct)
        if not isinstance(info, dict):
            continue
        spec = info.get("input") or {}
        req, opt, hid = spec.get("required") or {}, spec.get("optional") or {}, spec.get("hidden") or {}
        declared = {**hid, **opt, **req}
        inputs = node.get("inputs") or {}
        for name, val in inputs.items():
            if name not in declared:
                unknown_inputs.append({"node_id": str(nid), "class_type": ct, "input": name})
                continue
            opts = _combo_options(declared[name])
            if opts is not None and not _is_link(val) and str(val) not in opts:
                missing_values.append({"node_id": str(nid), "class_type": ct, "input": name, "value": val})
        for name in req:
            if name not in inputs:
                missing_required.append({"node_id": str(nid), "class_type": ct, "input": name})
    ok = not (missing_classes or missing_values or unknown_inputs or missing_required)
    return {**manifest, "checked": True, "ok": ok, "missing_classes": missing_classes,
            "missing_values": missing_values, "unknown_inputs": unknown_inputs,
            "missing_required_inputs": missing_required}


def _dependency_error(backend: str, res: dict) -> str:
    parts = []
    if res["missing_classes"]:
        parts.append(f"node classes not installed: {', '.join(res['missing_classes'])}")
    if res["missing_values"]:
        parts.append("files / options the backend does not have: " + "; ".join(
            f"{m['class_type']}.{m['input']}={m['value']!r} (node {m['node_id']})" for m in res["missing_values"][:10]))
    if res["unknown_inputs"]:
        parts.append("inputs this backend's version of the node does not know (older / newer node version): " +
                     "; ".join(f"{m['class_type']}.{m['input']} (node {m['node_id']})"
                               for m in res["unknown_inputs"][:10]))
    if res["missing_required_inputs"]:
        parts.append("required inputs the workflow does not set (newer node version on the backend): " +
                     "; ".join(f"{m['class_type']}.{m['input']} (node {m['node_id']})"
                               for m in res["missing_required_inputs"][:10]))
    return (f"C2C Farm: backend '{backend}' cannot run this workflow as it is - " + " | ".join(parts) +
            ". Nothing was uploaded or spooled. Update the backend image / install the models, or set "
            "dependency_check to 'report' to submit anyway.")


# ── L9.174 (A25): cluster telemetry + capacity forecast (Cluster Status) ────────────────────────────────────────────
_GB = float(2 ** 30)


def _telemetry(entry: dict) -> dict:
    gpus = entry.get("gpus") or []
    totals = [g.get("vram_total") for g in gpus if isinstance(g.get("vram_total"), (int, float))]
    frees = [g.get("vram_free") for g in gpus if isinstance(g.get("vram_free"), (int, float))]
    running, pending = entry.get("running"), entry.get("pending")
    mcj = entry.get("max_concurrent_jobs")
    t = {
        "queue_depth": (running or 0) + (pending or 0) if running is not None or pending is not None else None,
        "free_workers": max(0, int(mcj) - int(running or 0)) if isinstance(mcj, int) and running is not None
        else None,
        "vram_total_gb": round(sum(totals) / _GB, 3) if totals else None,
        "vram_free_gb": round(sum(frees) / _GB, 3) if frees else None,
        "largest_gpu_free_gb": round(max(frees) / _GB, 3) if frees else None,
    }
    t["vram_used_pct"] = (round(100.0 * (1.0 - sum(frees) / sum(totals)), 1)
                          if totals and frees and sum(totals) > 0 else None)
    return t


def _forecast(entry: dict, need_gb: float, workers: int) -> dict:
    if entry.get("disabled"):
        return {"fits": False, "reason": "backend disabled"}
    if not entry.get("reachable"):
        return {"fits": False, "reason": "backend unreachable"}
    t = entry.get("telemetry") or _telemetry(entry)
    pending = entry.get("pending")
    if t["largest_gpu_free_gb"] is None:
        return {"fits": None, "reason": "the backend does not report VRAM", "queue_ahead": pending,
                "free_workers": t["free_workers"]}
    if t["largest_gpu_free_gb"] < need_gb:
        return {"fits": False, "reason": f"largest free GPU memory {t['largest_gpu_free_gb']} GB < {need_gb} GB",
                "queue_ahead": pending, "free_workers": t["free_workers"]}
    fw = t["free_workers"]
    if fw is not None and fw < workers:
        return {"fits": False, "fits_after_queue": True, "reason": f"{fw} free worker(s) < {workers}: it would wait",
                "queue_ahead": pending, "free_workers": fw}
    return {"fits": True, "reason": f"a GPU with {t['largest_gpu_free_gb']} GB free and {fw} free worker(s)",
            "queue_ahead": pending, "free_workers": fw}


def _cluster_summary(report: list) -> str:
    lines = []
    for e in report:
        if e.get("disabled"):
            lines.append(f"{e.get('backend')}: disabled")
            continue
        if not e.get("reachable"):
            lines.append(f"{e.get('backend')}: UNREACHABLE {e.get('error', '')}".rstrip())
            continue
        t = e.get("telemetry") or _telemetry(e)
        line = (f"{e.get('backend')}: running {e.get('running')}, pending {e.get('pending')}, free workers "
                f"{t['free_workers']}, VRAM free {t['vram_free_gb']} / {t['vram_total_gb']} GB")
        if "forecast" in e:
            f = e["forecast"]
            line += f" | forecast: {'fits' if f['fits'] else ('unknown' if f['fits'] is None else 'no')} ({f['reason']})"
        lines.append(line)
    up = sum(1 for e in report if e.get("reachable"))
    lines.append(f"cluster: {up} of {len(report)} backend(s) reachable")
    return "\n".join(lines)


# ── L9.174 (A25): execution trace + run comparison (Job History) ────────────────────────────────────────────────────
_CONFIG_FIELDS = ("backend_name", "compute_profile", "priority", "project_name", "user")


def _iso(t):
    import datetime
    return None if t is None else datetime.datetime.fromtimestamp(float(t)).isoformat(timespec="seconds")


def _audit_row(db_path: str, job_id: str):
    import sqlite3
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=10)
    try:
        con.row_factory = sqlite3.Row
        r = con.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        return dict(r) if r else None
    finally:
        con.close()


def _live_job(job_id: str):
    """The in-memory Job when the production queue manager already runs (never started from here)."""
    try:
        from ..renderfarm.spooler import queue_manager as _qmod
    except Exception:  # noqa: BLE001
        return None
    qm = getattr(_qmod, "_qm", None)
    return qm.get(job_id) if qm is not None else None


def _job_trace(db_path: str, job_id: str) -> dict:
    row = _audit_row(db_path, job_id)
    live = _live_job(job_id)
    if row is None and live is None:
        raise ValueError(f"C2C Job History: no job {job_id!r} in the audit log. Job ids come from C2C Farm Submit's "
                         f"job_id output, the jobs_table output or the C2C Farm dashboard.")
    row = dict(row or {})
    if live is not None:
        for k in ("backend_name", "user", "project_name", "priority", "compute_profile", "submitted_at", "status",
                  "error"):
            row.setdefault(k, getattr(live, k, None))
    stages = [{"stage": "submitted", "at": row.get("submitted_at"), "iso": _iso(row.get("submitted_at"))}]
    if row.get("completed_at") is not None:
        stages.append({"stage": row.get("status"), "at": row["completed_at"], "iso": _iso(row["completed_at"])})
    else:
        stages.append({"stage": row.get("status"), "at": None, "iso": None})
    out = {"job": {k: row.get(k) for k in ("job_id", *_CONFIG_FIELDS, "status")}, "stages": stages,
           "duration_seconds": row.get("duration_seconds"), "error": row.get("error")}
    if row.get("completed_at") is None and row.get("submitted_at") is not None:
        out["elapsed_seconds"] = round(time.time() - float(row["submitted_at"]), 1)
    if live is not None:
        prompt = getattr(live, "prompt_json", None) or {}
        hist = {}
        for n in prompt.values():
            if isinstance(n, dict):
                hist[n.get("class_type")] = hist.get(n.get("class_type"), 0) + 1
        out["live"] = {"remote_id": getattr(live, "remote_id", None), "progress": getattr(live, "progress", None),
                       "prompt_nodes": len(prompt), "class_types": hist}
        out["outputs"] = {"count": len(live.result_paths or []), "paths": list(live.result_paths or [])}
    else:
        out["outputs"] = None
    out["not_recorded"] = ["per-node timings", "remote logs", "resource use per job"] + (
        [] if live is not None else ["outputs and prompt (job not in this session's memory)"])
    return out


def _prompt_diff(pa: dict, pb: dict) -> dict:
    added = sorted(set(pb) - set(pa), key=str)
    removed = sorted(set(pa) - set(pb), key=str)
    changed = {}
    for nid in sorted(set(pa) & set(pb), key=str):
        a, b = pa[nid] or {}, pb[nid] or {}
        if a.get("class_type") != b.get("class_type"):
            changed[str(nid)] = {"class_type": [a.get("class_type"), b.get("class_type")]}
            continue
        ia, ib = a.get("inputs") or {}, b.get("inputs") or {}
        d = {k: [ia.get(k), ib.get(k)] for k in sorted(set(ia) | set(ib)) if ia.get(k) != ib.get(k)}
        if d:
            changed[str(nid)] = d
    return {"added": [str(x) for x in added], "removed": [str(x) for x in removed], "changed_inputs": changed}


def _job_compare(db_path: str, job_a: str, job_b: str) -> dict:
    a, b = _job_trace(db_path, job_a), _job_trace(db_path, job_b)
    cfg = {f: [a["job"].get(f), b["job"].get(f)] for f in _CONFIG_FIELDS if a["job"].get(f) != b["job"].get(f)}
    da, dbb = a.get("duration_seconds"), b.get("duration_seconds")
    out = {"a": a, "b": b, "config_diff": cfg, "status": [a["job"].get("status"), b["job"].get("status")],
           "duration_delta_s": round(dbb - da, 3) if da is not None and dbb is not None else None,
           "duration_ratio": round(dbb / da, 4) if da and dbb is not None else None,
           "outputs": [None if a["outputs"] is None else a["outputs"]["count"],
                       None if b["outputs"] is None else b["outputs"]["count"]]}
    la, lb = _live_job(job_a), _live_job(job_b)
    if la is not None and lb is not None:
        out["prompt_diff"] = _prompt_diff(la.prompt_json or {}, lb.prompt_json or {})
    else:
        out["prompt_diff"] = "unavailable (job not in this session's memory)"
    return out


# A28 (docs/audit/A28/PARAM_CNP_B.md, C2C_Submit): the job bookkeeping sits in a "Farm" section; timeout_minutes is
# read only while waiting (_wait). Display only.
_SUBMIT_LAYOUT = {
    "priority": {"c2c_section": "Farm"},
    "project_name": {"c2c_section": "Farm"},
    "force_resubmit": {"c2c_section": "Farm"},
    "timeout_minutes": {"c2c_show_if": {"wait_for_completion": [True]}},
    "dependency_check": {"c2c_section": "Farm"},
}


class C2C_Submit:
    """Spool a full workflow JSON to a remote ComfyUI backend."""

    @classmethod
    def INPUT_TYPES(cls):
        return with_layout({
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
                "timeout_minutes": ("INT", {"default": 120, "min": 1, "max": 1440, "tooltip": "How long to wait before the node fails. The job itself keeps running - follow it in the C2C Farm dashboard."}),
            },
            "optional": {
                # A22 L9.135, appended: an identical submission reuses the earlier job (no second paid render)
                "force_resubmit": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Off: the same workflow on the same backend / profile / project reuses the job "
                               "already running or finished (no second upload or paid render). On: always "
                               "spool a new job."}),
                # L9.174 (A25), appended: "off" = the behaviour of every saved workflow
                "dependency_check": (DEPENDENCY_CHECKS, {
                    "default": "off",
                    "tooltip": "Before upload, compare the workflow with the backend's node list: missing node "
                               "classes, model files / options it does not have, inputs its version of a node does "
                               "not know and required inputs the workflow leaves out. off: no check (manifest only). "
                               "report: results in manifest_json, submit anyway. block: stop before anything is "
                               "uploaded or spooled when something is missing."}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }, _SUBMIT_LAYOUT)

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("job_id", "result_paths", "info", "manifest_json")
    OUTPUT_TOOLTIPS = ("The farm job id (look it up in the C2C Farm dashboard or Job History).",
                       "Local paths of the fetched outputs, one per line (empty when not waiting).",
                       "Backend, profile, status and results as JSON.",
                       "What the workflow needs (node classes, model files) and, with dependency_check on, what the "
                       "backend lacks, as JSON.")
    FUNCTION = "submit"
    CATEGORY = "MEC/RenderFarm"
    DESCRIPTION = "Dispatch a full workflow JSON to a remote ComfyUI render-farm backend."

    def submit(self, prompt_json, backend, compute_profile, priority, project_name,
               wait_for_completion, timeout_minutes, unique_id=None, force_resubmit=False,
               dependency_check="off"):
        if dependency_check not in DEPENDENCY_CHECKS:
            raise ValueError(f"C2C Farm: dependency_check must be one of {DEPENDENCY_CHECKS}, got "
                             f"{dependency_check!r}")
        res = self._submit(prompt_json, backend, compute_profile, priority, project_name, wait_for_completion,
                           timeout_minutes, unique_id, force_resubmit, dependency_check)
        return tuple(res[:3]) + (json.dumps(self._manifest, indent=2, default=str),)

    def _submit(self, prompt_json, backend, compute_profile, priority, project_name,
                wait_for_completion, timeout_minutes, unique_id, force_resubmit, dependency_check):
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

        self._manifest = build_manifest(prompt)
        qm = get_queue_manager()
        fp = Job.fingerprint_of(prompt, backend, compute_profile, project_name, user["name"])
        reused = None if force_resubmit else qm.find_reusable(fp)
        if reused is not None:
            # A22 L9.135: the same work again (core dropped the node's cache) - no second upload, no second paid job
            _progress_event(unique_id, 0.15, f"same job as {reused.job_id}: reusing it")
            job = reused
            if dependency_check != "off":
                self._manifest["note"] = "reused job: not checked again"
            info = {"job_id": job.job_id, "backend": backend, "user": user["name"],
                    "compute_profile": compute_profile, "priority": int(priority), "uploaded_media": 0,
                    "reused": True}
            if not wait_for_completion:
                return (job.job_id, "\n".join(job.result_paths),
                        json.dumps({**info, "status": job.status}, indent=2))
            return self._wait(job, info, backend, timeout_minutes, unique_id)

        adapter = get_adapter(backend)
        _progress_event(unique_id, 0.05, "pre-flight node check")
        preflight.validate_backend_nodes(prompt, adapter, backend)  # fatality preventer
        if dependency_check != "off":
            _progress_event(unique_id, 0.07, "dependency check")
            try:
                oi = _backend_object_info(adapter)
            except Exception as exc:  # noqa: BLE001 - reported, and blocking in block mode
                oi = None
                self._manifest["note"] = f"could not read the backend's /object_info: {str(exc)[:200]}"
            if oi is None:
                self._manifest.setdefault("note", "this backend adapter cannot report its node list")
                if dependency_check == "block":
                    raise RuntimeError(f"C2C Farm: dependency_check=block but backend '{backend}' did not report "
                                       f"its nodes ({self._manifest['note']}). Nothing was uploaded or spooled.")
            else:
                self._manifest = check_dependencies(prompt, oi, self._manifest)
                if dependency_check == "block" and not self._manifest["ok"]:
                    raise RuntimeError(_dependency_error(backend, self._manifest))

        _progress_event(unique_id, 0.10, "courier: media upload")
        prompt, uploads = courier.prepare_prompt(prompt, backend_cfg)

        job = Job(prompt_json=prompt, backend_name=backend, user=user["name"],
                  project_name=project_name, priority=int(priority),
                  compute_profile=compute_profile, fingerprint=fp)
        qm.submit(job)
        info = {"job_id": job.job_id, "backend": backend, "user": user["name"],
                "compute_profile": compute_profile, "priority": int(priority),
                "uploaded_media": len(uploads)}

        if not wait_for_completion:
            _progress_event(unique_id, 1.0, f"spooled {job.job_id}")
            return (job.job_id, "", json.dumps({**info, "status": "spooled"}, indent=2))
        return self._wait(job, info, backend, timeout_minutes, unique_id, cancel=qm.cancel)

    @staticmethod
    def _wait(job, info, backend, timeout_minutes, unique_id, cancel=None):
        deadline = time.time() + timeout_minutes * 60
        while not job.done.wait(timeout=2.0):
            if _cancel_requested():
                # L9.190: Cancel in ComfyUI stops the wait. A job THIS run submitted is cancelled on the farm too
                # (it may be billed cloud time); a reused job belongs to an earlier submission and keeps running.
                if not info.get("reused") and cancel is not None:
                    remote_ok = cancel(job.job_id)
                    log.info("C2C Farm: job %s cancelled by the user (farm cancel %s)", job.job_id,
                             "confirmed" if remote_ok else "NOT confirmed - check the C2C Farm dashboard")
                else:
                    log.info("C2C Farm: stopped waiting for reused job %s (it keeps running)", job.job_id)
                _raise_interrupted()
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
        }, "optional": {
            # L9.174 (A25), appended: defaults = the behaviour of every saved workflow
            "telemetry": ("BOOLEAN", {"default": False, "tooltip": "Add derived telemetry per backend (queue depth, "
                          "free workers, total / free / used VRAM, largest free GPU) and cluster totals to "
                          "cluster_json. CPU, RAM and storage are listed as not reported: the backend does not "
                          "send them."}),
            "forecast_vram_gb": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1024.0, "step": 0.5, "tooltip":
                                 "Capacity forecast: GPU memory a proposed job needs, in GB. Each backend gets "
                                 "fits yes / no / unknown with the reason and the queue ahead, plus the best "
                                 "backend. 0 = off."}),
            "forecast_workers": ("INT", {"default": 1, "min": 1, "max": 64, "tooltip": "Capacity forecast: free "
                                 "worker slots the job needs at once."}),
        }}

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("cluster_json", "summary")
    OUTPUT_TOOLTIPS = ("Per backend: reachable, queue depth, VRAM (or the error), as JSON. Refreshes about every 5 s.",
                       "One readable line per backend (queue, free workers, free VRAM, forecast) plus a cluster "
                       "line.")
    FUNCTION = "status"
    CATEGORY = "MEC/RenderFarm"
    DESCRIPTION = "Live render-farm capacity (queue depth, VRAM, reachability) as JSON."

    @classmethod
    def IS_CHANGED(cls, backend, **_kw):
        # Live cluster state genuinely changes — refresh every ~5s bucket.
        return int(time.time() // 5)

    def status(self, backend, telemetry=False, forecast_vram_gb=0.0, forecast_workers=1):
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
        out = {"backends": report}
        if telemetry:
            for e in report:
                if e.get("reachable"):
                    e["telemetry"] = _telemetry(e)
            ts = [e["telemetry"] for e in report if "telemetry" in e]

            def _sum(k):
                v = [t[k] for t in ts if t[k] is not None]
                return round(sum(v), 3) if v else None
            out["cluster"] = {"backends": len(report), "reachable": sum(1 for e in report if e.get("reachable")),
                              "unreachable": sum(1 for e in report if not e.get("reachable")),
                              "queue_depth": _sum("queue_depth"), "free_workers": _sum("free_workers"),
                              "vram_total_gb": _sum("vram_total_gb"), "vram_free_gb": _sum("vram_free_gb"),
                              "not_reported": ["cpu", "ram", "storage"]}
        if float(forecast_vram_gb) > 0:
            for e in report:
                e["forecast"] = _forecast(e, float(forecast_vram_gb), int(forecast_workers))
            cands = sorted(report, key=lambda e: (e["forecast"]["fits"] is True,
                                                  e["forecast"].get("free_workers") or 0,
                                                  (e.get("telemetry") or _telemetry(e))["largest_gpu_free_gb"] or 0
                                                  if e.get("reachable") else 0), reverse=True)
            best = next((e["backend"] for e in cands if e["forecast"]["fits"] is True), None)
            out["forecast"] = {"vram_gb": float(forecast_vram_gb), "workers": int(forecast_workers),
                               "best_backend": best,
                               "candidates": [{"backend": e.get("backend"), **e["forecast"]} for e in cands]}
        return (json.dumps(out, indent=2), _cluster_summary(report))


class C2C_JobHistory:
    """Query the SQLite audit log (who ran what, where, how long)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "limit": ("INT", {"default": 50, "min": 1, "max": 1000, "tooltip": "Most recent jobs to list."}),
            "user_filter": ("STRING", {"default": "", "tooltip": "Empty = all users."}),
            "project_filter": ("STRING", {"default": "", "tooltip": "Empty = all projects."}),
        }, "optional": {
            # L9.174 (A25), appended: empty = the behaviour of every saved workflow
            "job_id": ("STRING", {"default": "", "tooltip": "Execution trace of one job: stages with times, "
                       "duration (or elapsed while running), error text, outputs and the workflow's node types when "
                       "the job ran in this session. Result in job_report. Empty = off."}),
            "compare_job_id": ("STRING", {"default": "", "tooltip": "Run comparison: a second job id. job_report "
                               "then lists the configuration differences (backend, profile, priority, project, "
                               "user), status, duration delta / ratio, output counts and - when both jobs ran in "
                               "this session - the workflow differences node by node."}),
        }}

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("stats_json", "jobs_table", "job_report")
    OUTPUT_TOOLTIPS = ("Job counts per user and per status plus total / average duration, over ALL jobs (the filters do not apply), as JSON.",
                       "The most recent jobs (up to limit) as a text table.",
                       "job_id set: the job's execution trace; with compare_job_id: the comparison of the two "
                       "jobs, as JSON. Empty when off.")
    FUNCTION = "history"
    CATEGORY = "MEC/RenderFarm"
    DESCRIPTION = "Render-farm audit log: per-user/status stats + recent jobs table."

    @classmethod
    def IS_CHANGED(cls, limit, user_filter, project_filter, job_id="", compare_job_id="", **_kw):
        # Re-run only when the audit DB actually changed on disk.
        import os
        from ..renderfarm.logging.audit_db import DB_PATH
        try:
            m = os.path.getmtime(DB_PATH)
        except OSError:
            m = 0
        if not (job_id or compare_job_id):
            return m                                              # the saved-workflow fingerprint, unchanged
        live = int(time.time() // 5)                              # a live job's in-memory state changes too
        return f"{m}|{job_id}|{compare_job_id}|{live}"

    def history(self, limit, user_filter, project_filter, job_id="", compare_job_id=""):
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
        report = ""
        jid, cid = (job_id or "").strip(), (compare_job_id or "").strip()
        if cid and not jid:
            raise ValueError("C2C Job History: compare_job_id needs job_id (the first job) as well.")
        if jid and cid:
            report = json.dumps(_job_compare(db.path, jid, cid), indent=2, default=str)
        elif jid:
            report = json.dumps(_job_trace(db.path, jid), indent=2, default=str)
        return (json.dumps(db.stats(), indent=2), "\n".join(lines), report)


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
