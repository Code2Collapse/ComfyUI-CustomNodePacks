"""C2C Farm backend adapter interface — the compute plane contract.

Every backend is a FULL ComfyUI instance (Docker on AKS, RunPod, vast.ai,
LAN). Adapters translate the spooler's verbs into that backend's transport.
"""

from __future__ import annotations


class BackendAdapter:
    """Contract used by the spooler; all methods may raise RuntimeError with
    an actionable message (unreachable host, bad token, missing job…)."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.name = cfg.get("name", "unnamed")

    def submit(self, prompt_json: dict, compute_profile: str, priority: int) -> str:
        """Dispatch the workflow; returns the backend's job id."""
        raise NotImplementedError

    def get_status(self, job_id: str) -> dict:
        """{status: queued|running|complete|error|unknown,
            progress_pct: float 0..1 | None, error: str | None}"""
        raise NotImplementedError

    def get_result(self, job_id: str) -> list[str]:
        """Download the job's output media; returns local file paths."""
        raise NotImplementedError

    def cancel(self, job_id: str) -> bool:
        raise NotImplementedError

    def supports_preview(self) -> bool:
        return False

    def get_preview(self, job_id: str) -> str | None:
        """Latest live preview as a base64 data-URI payload, or None."""
        return None

    # ── optional capability probes used by pre-flight / dashboard ────
    def installed_node_classes(self) -> set[str] | None:
        """Set of node class_types installed on the backend (None = unknown)."""
        return None

    def capacity(self) -> dict:
        """Best-effort live capacity summary for the dashboard."""
        return {"backend": self.name, "reachable": None}
