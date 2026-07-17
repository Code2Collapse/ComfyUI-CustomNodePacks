"""S3-compatible courier: AWS S3, Cloudflare R2, MinIO (boto3, lazy).

Env:
    RIB_S3_BUCKET        (required)  bucket name
    RIB_S3_ENDPOINT_URL  (optional)  set for R2 / MinIO / other S3-compatibles
    RIB_S3_REGION        (optional)  default us-east-1
    RIB_S3_PREFIX        (optional)  key prefix, default "rib/"
    RIB_S3_URL_TTL       (optional)  presigned GET lifetime seconds, default 86400
Credentials ride the standard boto3 chain (env AWS_ACCESS_KEY_ID/…, profile,
instance role).
"""

from __future__ import annotations

import os
import uuid

from .base_storage import BaseStorage, require_env


class S3CompatibleStorage(BaseStorage):
    name = "s3"

    def __init__(self, client=None):
        (self.bucket,) = require_env("RIB_S3_BUCKET")
        self.prefix = os.environ.get("RIB_S3_PREFIX", "rib/")
        self.ttl = int(os.environ.get("RIB_S3_URL_TTL", "86400"))
        if client is not None:  # test injection
            self.client = client
            return
        try:
            import boto3
        except ImportError as exc:
            raise RuntimeError(
                "RIB storage: boto3 is not installed. Run: pip install boto3 "
                "(covers AWS S3, Cloudflare R2, MinIO)."
            ) from exc
        self.client = boto3.client(
            "s3",
            endpoint_url=os.environ.get("RIB_S3_ENDPOINT_URL") or None,
            region_name=os.environ.get("RIB_S3_REGION", "us-east-1"),
        )

    def _key(self, local_path: str) -> str:
        return f"{self.prefix}{uuid.uuid4().hex[:10]}/{os.path.basename(local_path)}"

    def upload(self, local_path: str) -> str:
        key = self._key(local_path)
        self.client.upload_file(local_path, self.bucket, key)
        return self.client.generate_presigned_url(
            "get_object", Params={"Bucket": self.bucket, "Key": key}, ExpiresIn=self.ttl
        )
