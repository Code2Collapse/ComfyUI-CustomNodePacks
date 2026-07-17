"""
RIB — "Render In Background" Tractor-style farm manager for ComfyUI.
====================================================================
Turns this local ComfyUI into a Pixar-Tractor-style spooler/dashboard that
dispatches full workflow JSON to remote **ComfyUI instances in Docker**
(AKS / AWS / GCP / RunPod / LAN), with:

  * control plane — SQLite audit log (async writer) + priority spooler
    (`logging/audit_db.py`, `spooler/queue_manager.py`)
  * data plane   — "invisible courier" cloud-storage handoff for media
    (`storage/`, `courier.py`)
  * compute plane — backend adapters wrapping the remote ComfyUI HTTP API
    (`backends/`), compute-profile routing, pre-flight node validation
    (`preflight.py`)
  * UI           — sidebar dashboard (js/rib_dashboard.js) fed by /rib/*
    routes (`api_routes.py`)

Everything here is import-light: heavy/optional SDKs (boto3, azure, gcs,
websocket-client) are lazy-imported at point of use with actionable errors.
Kubernetes/Terraform and model serving are explicitly out of scope — the
backends are full ComfyUI containers managed by the infra team.
"""

__all__ = ["__version__"]
__version__ = "0.1.0"
