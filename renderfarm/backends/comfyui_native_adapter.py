"""ComfyUI-native backend adapter — AKS Docker, RunPod, vast.ai, LAN boxes.

Speaks the standard ComfyUI HTTP API of the remote instance/gateway:
    POST /prompt        submit prompt JSON (+ compute_profile in body & header
                        so the infra team's gateway can route node pools)
    GET  /history/{id}  terminal status + outputs
    GET  /queue         running/pending introspection
    GET  /view          output file download
    POST /interrupt, /queue {"delete": [...]}   cancel
    WS   /ws?clientId=  live progress + binary preview frames (optional —
                        needs `pip install websocket-client`; degrades to
                        history-only polling without it)

Auth: if the backend config names `api_key_env`, that env var MUST be set
(fail-loud) and rides as a Bearer token.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import struct
import threading
import time
import uuid

from .base_adapter import BackendAdapter

log = logging.getLogger("RIB.backend")


class ComfyUINativeAdapter(BackendAdapter):
    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.base_url = (cfg.get("base_url") or "").rstrip("/")
        if not self.base_url:
            raise RuntimeError(f"RIB: backend '{self.name}' has no base_url in backends.json.")
        self.headers = {}
        key_env = cfg.get("api_key_env") or ""
        if key_env:
            token = os.environ.get(key_env, "")
            if not token:
                raise RuntimeError(
                    f"RIB: backend '{self.name}' requires the API token env var "
                    f"'{key_env}', which is not set. Export it and restart ComfyUI."
                )
            self.headers["Authorization"] = f"Bearer {token}"
        self.client_id = f"rib-{uuid.uuid4().hex[:8]}"
        self._object_info_cache: tuple[float, set[str]] | None = None
        self._ws_tap: _WsTap | None = None
        self._ws_lock = threading.Lock()

    # ── http helpers ─────────────────────────────────────────────────
    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def _request(self, method: str, path: str, timeout=(5, 60), **kw):
        import requests
        try:
            r = requests.request(method, self._url(path), headers=self.headers,
                                 timeout=timeout, **kw)
        except requests.exceptions.RequestException as exc:
            raise RuntimeError(
                f"RIB: backend '{self.name}' unreachable at {self.base_url} ({exc.__class__.__name__}). "
                f"Is the container/gateway running and the URL in backends.json correct?"
            ) from exc
        if r.status_code == 401 or r.status_code == 403:
            raise RuntimeError(
                f"RIB: backend '{self.name}' rejected the API token (HTTP {r.status_code}). "
                f"Check the '{self.cfg.get('api_key_env')}' env var."
            )
        r.raise_for_status()
        return r

    # ── BackendAdapter contract ──────────────────────────────────────
    def submit(self, prompt_json: dict, compute_profile: str, priority: int) -> str:
        body = {
            "prompt": prompt_json,
            "client_id": self.client_id,
            # Routing hints for the infra team's API gateway (heavy vs light
            # node pools). Sent in the body AND a header so either layer can
            # route without parsing the prompt.
            "compute_profile": compute_profile,
            "extra_data": {"rib": {"compute_profile": compute_profile,
                                   "priority": int(priority)}},
        }
        headers = {"X-Compute-Profile": compute_profile}
        import requests
        try:
            r = requests.post(self._url("/prompt"), json=body,
                              headers={**self.headers, **headers}, timeout=(5, 120))
        except requests.exceptions.RequestException as exc:
            raise RuntimeError(
                f"RIB: submit to backend '{self.name}' failed ({exc.__class__.__name__}) — "
                f"gateway {self.base_url} unreachable."
            ) from exc
        if r.status_code >= 400:
            raise RuntimeError(
                f"RIB: backend '{self.name}' rejected the workflow (HTTP {r.status_code}): "
                f"{r.text[:500]}"
            )
        remote_id = r.json().get("prompt_id")
        if not remote_id:
            raise RuntimeError(f"RIB: backend '{self.name}' returned no prompt_id: {r.text[:300]}")
        self._ensure_ws_tap()
        return remote_id

    def get_status(self, job_id: str) -> dict:
        hist = self._history(job_id)
        if hist is not None:
            status = hist.get("status", {})
            if status.get("status_str") == "success" or status.get("completed"):
                return {"status": "complete", "progress_pct": 1.0, "error": None}
            err = self._extract_error(status)
            return {"status": "error", "progress_pct": None,
                    "error": err or "backend reported execution error"}
        q = self._request("GET", "/queue", timeout=(5, 20)).json()
        running = {e[1] for e in q.get("queue_running", []) if len(e) > 1}
        pending = {e[1] for e in q.get("queue_pending", []) if len(e) > 1}
        tap = self._ws_tap
        pct = tap.progress.get(job_id) if tap else None
        if job_id in running:
            return {"status": "running", "progress_pct": pct, "error": None}
        if job_id in pending:
            return {"status": "queued", "progress_pct": None, "error": None}
        return {"status": "unknown", "progress_pct": None, "error": None}

    def get_result(self, job_id: str) -> list[str]:
        hist = self._history(job_id)
        if hist is None:
            raise RuntimeError(f"RIB: backend '{self.name}' has no history for job {job_id}.")
        from ..courier import download_results, results_dir
        paths: list[str] = []
        url_outputs: list[str] = []
        dest = results_dir(job_id)
        for node_out in (hist.get("outputs") or {}).values():
            for media_list in node_out.values():
                if not isinstance(media_list, list):
                    continue
                for item in media_list:
                    if isinstance(item, str) and item.startswith(("http://", "https://")):
                        url_outputs.append(item)
                    elif isinstance(item, dict) and item.get("filename"):
                        paths.append(self._download_view(item, dest))
        if url_outputs:  # cloud-save nodes on the backend emit URLs
            paths.extend(download_results(url_outputs, job_id))
        return paths

    def cancel(self, job_id: str) -> bool:
        ok = True
        try:
            self._request("POST", "/queue", json={"delete": [job_id]}, timeout=(5, 20))
        except Exception:  # noqa: BLE001 — may already be running
            ok = False
        try:
            st = self.get_status(job_id)
            if st["status"] == "running":
                self._request("POST", "/interrupt", timeout=(5, 20))
                ok = True
        except Exception:  # noqa: BLE001
            ok = False
        return ok

    def supports_preview(self) -> bool:
        try:
            import websocket  # noqa: F401
            return True
        except ImportError:
            return False

    def get_preview(self, job_id: str) -> str | None:
        tap = self._ws_tap
        if not tap:
            return None
        return tap.previews.get(job_id) or tap.latest_preview

    # ── capability probes ────────────────────────────────────────────
    def installed_node_classes(self) -> set[str] | None:
        now = time.time()
        if self._object_info_cache and now - self._object_info_cache[0] < 300:
            return self._object_info_cache[1]
        classes = set(self._request("GET", "/object_info", timeout=(5, 60)).json().keys())
        self._object_info_cache = (now, classes)
        return classes

    def capacity(self) -> dict:
        out = {"backend": self.name, "base_url": self.base_url, "reachable": False,
               "running": None, "pending": None, "gpus": None,
               "max_concurrent_jobs": self.cfg.get("max_concurrent_jobs", 1),
               "compute_profiles": self.cfg.get("compute_profiles", [])}
        try:
            q = self._request("GET", "/queue", timeout=(3, 10)).json()
            out["running"] = len(q.get("queue_running", []))
            out["pending"] = len(q.get("queue_pending", []))
            out["reachable"] = True
            stats = self._request("GET", "/system_stats", timeout=(3, 10)).json()
            out["gpus"] = [
                {"name": d.get("name"), "vram_total": d.get("vram_total"),
                 "vram_free": d.get("vram_free")}
                for d in stats.get("devices", [])
            ]
        except Exception as exc:  # noqa: BLE001 — capacity is best-effort
            out["error"] = str(exc)[:200]
        return out

    # ── internals ────────────────────────────────────────────────────
    def _history(self, job_id: str) -> dict | None:
        data = self._request("GET", f"/history/{job_id}", timeout=(5, 30)).json()
        return data.get(job_id)

    @staticmethod
    def _extract_error(status: dict) -> str | None:
        for msg in status.get("messages", []) or []:
            if isinstance(msg, (list, tuple)) and len(msg) > 1 and msg[0] == "execution_error":
                d = msg[1] or {}
                return f"{d.get('node_type', '?')}: {d.get('exception_message', 'error')}".strip()
        return None

    def _download_view(self, item: dict, dest_dir: str) -> str:
        import requests
        params = {"filename": item["filename"],
                  "subfolder": item.get("subfolder", ""),
                  "type": item.get("type", "output")}
        local = os.path.join(dest_dir, item["filename"])
        os.makedirs(os.path.dirname(local), exist_ok=True)
        with requests.get(self._url("/view"), params=params, headers=self.headers,
                          stream=True, timeout=(10, 600)) as r:
            r.raise_for_status()
            with open(local, "wb") as fh:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    fh.write(chunk)
        return local

    def _ensure_ws_tap(self):
        """Start the live progress/preview websocket tap (optional dep)."""
        with self._ws_lock:
            if self._ws_tap is not None and self._ws_tap.alive:
                return
            try:
                import websocket  # noqa: F401
            except ImportError:
                log.info("websocket-client not installed — backend '%s' progress will "
                         "poll history only (pip install websocket-client for live "
                         "progress + previews).", self.name)
                return
            self._ws_tap = _WsTap(self.base_url, self.client_id, self.headers)


class _WsTap:
    """Background websocket listener: per-prompt progress + latest preview."""

    def __init__(self, base_url: str, client_id: str, headers: dict):
        self.progress: dict[str, float] = {}
        self.previews: dict[str, str] = {}
        self.latest_preview: str | None = None
        self._current_prompt: str | None = None
        self.alive = True
        ws_url = base_url.replace("https://", "wss://").replace("http://", "ws://")
        self._url = f"{ws_url}/ws?clientId={client_id}"
        self._headers = [f"{k}: {v}" for k, v in headers.items()]
        threading.Thread(target=self._loop, name="rib-ws-tap", daemon=True).start()

    def _loop(self):
        import websocket
        while self.alive:
            try:
                ws = websocket.create_connection(self._url, header=self._headers, timeout=30)
                while self.alive:
                    frame = ws.recv()
                    if isinstance(frame, bytes):
                        self._on_binary(frame)
                    else:
                        self._on_text(frame)
            except Exception as exc:  # noqa: BLE001 — reconnect with backoff
                log.debug("ws tap reconnect: %s", exc)
                time.sleep(5)

    def _on_text(self, frame: str):
        try:
            msg = json.loads(frame)
        except Exception:  # noqa: BLE001
            return
        t, d = msg.get("type"), msg.get("data", {})
        if t == "executing":
            self._current_prompt = d.get("prompt_id") or self._current_prompt
        elif t == "progress":
            pid = d.get("prompt_id") or self._current_prompt
            if pid and d.get("max"):
                self.progress[pid] = round(float(d["value"]) / float(d["max"]), 4)
        elif t == "execution_start":
            self._current_prompt = d.get("prompt_id")

    def _on_binary(self, frame: bytes):
        # ComfyUI binary preview: uint32 event (1 = PREVIEW_IMAGE),
        # uint32 format (1 = JPEG, 2 = PNG), then the image bytes.
        if len(frame) < 8:
            return
        event, fmt = struct.unpack(">II", frame[:8])
        if event != 1:
            return
        mime = "image/png" if fmt == 2 else "image/jpeg"
        b64 = base64.b64encode(frame[8:]).decode("ascii")
        data_uri = f"data:{mime};base64,{b64}"
        self.latest_preview = data_uri
        if self._current_prompt:
            self.previews[self._current_prompt] = data_uri
            if len(self.previews) > 32:
                self.previews.pop(next(iter(self.previews)))

