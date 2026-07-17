"""Google Cloud Storage courier (google-cloud-storage, lazy).

Env:
    C2C_GCS_BUCKET                 (required)  bucket name
    GOOGLE_APPLICATION_CREDENTIALS (required)  service-account JSON path
    C2C_GCS_URL_TTL                (optional)  signed GET lifetime seconds, default 86400
"""

from __future__ import annotations

import datetime
import os
import uuid

from .base_storage import BaseStorage, require_env


class GCSStorage(BaseStorage):
    name = "gcs"

    def __init__(self, client=None):
        (self.bucket_name,) = require_env("C2C_GCS_BUCKET")
        require_env("GOOGLE_APPLICATION_CREDENTIALS")
        self.ttl = int(os.environ.get("C2C_GCS_URL_TTL", "86400"))
        if client is not None:  # test injection
            self.client = client
            return
        try:
            from google.cloud import storage as gcs
        except ImportError as exc:
            raise RuntimeError(
                "C2C Farm storage: google-cloud-storage is not installed. "
                "Run: pip install google-cloud-storage"
            ) from exc
        self.client = gcs.Client()

    def upload(self, local_path: str) -> str:
        blob_name = f"c2c-farm/{uuid.uuid4().hex[:10]}/{os.path.basename(local_path)}"
        blob = self.client.bucket(self.bucket_name).blob(blob_name)
        blob.upload_from_filename(local_path)
        return blob.generate_signed_url(
            version="v4",
            expiration=datetime.timedelta(seconds=self.ttl),
            method="GET",
        )
