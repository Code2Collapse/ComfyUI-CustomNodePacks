"""RIB storage adapter interface — the "invisible courier" data plane.

4K video / latents cannot ride a REST body without HTTP timeouts, so media
is handed off through cloud object storage: upload → presigned URL → the
backend's loader pulls the URL; outputs come back the same way.

Contract:
    upload(local_path)              -> remote_url (time-limited GET URL)
    download(remote_url, local_path)-> local_path

All SDKs are OPTIONAL lazy imports (boto3 / azure-storage-blob /
google-cloud-storage) — a missing SDK or env var raises a RuntimeError that
names the exact `pip install` and variables to set.
"""

from __future__ import annotations

import os


class BaseStorage:
    """Abstract courier. Subclasses implement upload(); generic HTTP download
    works for any presigned/SAS/signed URL out of the box."""

    name = "base"

    def upload(self, local_path: str) -> str:
        raise NotImplementedError

    def download(self, remote_url: str, local_path: str) -> str:
        """Default: stream any http(s) URL (presigned S3 / SAS / signed GCS)."""
        return _http_download(remote_url, local_path)


def _http_download(url: str, local_path: str) -> str:
    import requests
    os.makedirs(os.path.dirname(os.path.abspath(local_path)), exist_ok=True)
    with requests.get(url, stream=True, timeout=(10, 600)) as r:
        r.raise_for_status()
        with open(local_path, "wb") as fh:
            for chunk in r.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
    return local_path


def require_env(*names: str) -> list[str]:
    """Return env values; raise a fail-loud error naming every missing var."""
    missing = [n for n in names if not os.environ.get(n)]
    if missing:
        raise RuntimeError(
            f"RIB storage: required environment variable(s) missing: {missing}. "
            f"Set them before submitting jobs that carry local media."
        )
    return [os.environ[n] for n in names]
