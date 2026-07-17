"""RIB storage courier factory.

Provider chosen by the RIB_STORAGE_PROVIDER env var:
    s3 | r2 | minio  -> S3CompatibleStorage (boto3)
    azure            -> AzureBlobStorage    (azure-storage-blob)
    gcs              -> GCSStorage          (google-cloud-storage)

Unset provider only errors when a workflow actually carries local media —
URL-only workflows never touch storage.
"""

from __future__ import annotations

import os

from .base_storage import BaseStorage  # noqa: F401 (re-export)

_PROVIDERS = {"s3", "r2", "minio", "azure", "gcs"}


def get_storage() -> BaseStorage:
    provider = os.environ.get("RIB_STORAGE_PROVIDER", "").strip().lower()
    if not provider:
        raise RuntimeError(
            "RIB storage: this workflow references local media files, which must be "
            "handed to the backend via cloud storage — but RIB_STORAGE_PROVIDER is not "
            f"set. Set it to one of {sorted(_PROVIDERS)} plus that provider's env vars "
            "(see renderfarm/storage/*.py headers)."
        )
    if provider in ("s3", "r2", "minio"):
        from .s3_compatible_storage import S3CompatibleStorage
        return S3CompatibleStorage()
    if provider == "azure":
        from .azure_blob_storage import AzureBlobStorage
        return AzureBlobStorage()
    if provider == "gcs":
        from .gcs_storage import GCSStorage
        return GCSStorage()
    raise RuntimeError(
        f"RIB storage: unknown RIB_STORAGE_PROVIDER '{provider}'. "
        f"Valid values: {sorted(_PROVIDERS)}."
    )
