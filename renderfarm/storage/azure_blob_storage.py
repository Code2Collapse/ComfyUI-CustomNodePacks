"""Azure Blob courier (azure-storage-blob, lazy).

Env:
    AZURE_STORAGE_CONNECTION_STRING (required)  standard Azure connection string
    C2C_AZURE_CONTAINER             (required)  container name
    C2C_AZURE_URL_TTL               (optional)  SAS GET lifetime seconds, default 86400
"""

from __future__ import annotations

import datetime
import os
import uuid

from .base_storage import BaseStorage, require_env


class AzureBlobStorage(BaseStorage):
    name = "azure"

    def __init__(self, service_client=None):
        conn, self.container = require_env("AZURE_STORAGE_CONNECTION_STRING",
                                           "C2C_AZURE_CONTAINER")
        self.ttl = int(os.environ.get("C2C_AZURE_URL_TTL", "86400"))
        try:
            from azure.storage.blob import BlobServiceClient
        except ImportError as exc:
            raise RuntimeError(
                "C2C Farm storage: azure-storage-blob is not installed. "
                "Run: pip install azure-storage-blob"
            ) from exc
        self.service = service_client or BlobServiceClient.from_connection_string(conn)

    def upload(self, local_path: str) -> str:
        from azure.storage.blob import BlobSasPermissions, generate_blob_sas
        blob_name = f"c2c-farm/{uuid.uuid4().hex[:10]}/{os.path.basename(local_path)}"
        blob = self.service.get_blob_client(container=self.container, blob=blob_name)
        with open(local_path, "rb") as fh:
            blob.upload_blob(fh, overwrite=True)
        sas = generate_blob_sas(
            account_name=self.service.account_name,
            container_name=self.container,
            blob_name=blob_name,
            account_key=self.service.credential.account_key,
            permission=BlobSasPermissions(read=True),
            expiry=datetime.datetime.utcnow() + datetime.timedelta(seconds=self.ttl),
        )
        return f"{blob.url}?{sas}"
