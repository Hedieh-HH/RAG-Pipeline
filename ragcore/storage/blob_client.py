import hashlib
import logging
from pathlib import Path

from azure.core.exceptions import (
    AzureError,
    ClientAuthenticationError,
    ServiceRequestError,
)
from azure.storage.blob import BlobServiceClient

from ragcore.models import LocalDocument

logger = logging.getLogger(__name__)


class BlobStorageClient:
    """
    Context manager wrapping the Azure Blob Storage SDK.

    Always use via `with`:

        with BlobStorageClient(connection_string, container_name, pdf_dir) as client:
            ...

    This ensures the underlying HTTP session is properly closed.
    """

    def __init__(
        self,
        connection_string: str,
        container_name: str,
        pdf_dir: Path,
    ) -> None:
        self._connection_string = connection_string
        self._container_name = container_name
        self._pdf_dir = pdf_dir
        self._client: BlobServiceClient | None = None

    def __enter__(self) -> "BlobStorageClient":
        self._client = BlobServiceClient.from_connection_string(self._connection_string)
        return self

    def __exit__(self, *_: object) -> None:
        if self._client:
            self._client.close()

    def fetch_all(self) -> list[LocalDocument]:
        """
        List all PDF blobs in the container and download each one to local disk.

        Returns a list of LocalDocuments — one per successfully downloaded PDF.
        Non-PDF blobs are silently skipped.
        Per-blob failures are logged and skipped — they do not stop the pipeline.
        Container-level failures raise immediately as there is nothing to process.
        """
        self._pdf_dir.mkdir(parents=True, exist_ok=True)
        documents = []

        # Container-level: unrecoverable — raise immediately.
        try:
            container_client = self._client.get_container_client(self._container_name)
            blobs = list(container_client.list_blobs())
        except ClientAuthenticationError as e:
            logger.error(
                "Authentication failed — check your connection string and permissions: %s",
                e,
            )
            raise
        except ServiceRequestError as e:
            logger.error(
                "Network error reaching Azure Blob Storage — check connectivity: %s", e
            )
            raise
        except AzureError as e:
            logger.error(
                "Failed to list blobs in container '%s': %s", self._container_name, e
            )
            raise

        for blob in blobs:
            if not blob.name.lower().endswith(".pdf"):
                continue

            dest_path = self._pdf_dir / Path(blob.name).name
            logger.info("Downloading '%s' → %s", blob.name, dest_path)

            # Azure-side error: network can drop between listing and downloading.
            try:
                blob_client = self._client.get_blob_client(
                    container=self._container_name,
                    blob=blob.name,
                )
                data = blob_client.download_blob().readall()
            except ServiceRequestError as e:
                logger.error("[Azure] Network error downloading '%s': %s", blob.name, e)
                continue
            except AzureError as e:
                logger.error("[Azure] Failed to download '%s': %s", blob.name, e)
                continue

            # Local error: problem with the filesystem.
            try:
                hasher = hashlib.sha256()
                hasher.update(data)
                with open(dest_path, "wb") as f:
                    f.write(data)
            except OSError as e:
                logger.error("[Local] Failed to write '%s' to disk: %s", dest_path, e)
                continue

            logger.info(
                "Downloaded '%s', sha256=%s...", blob.name, hasher.hexdigest()[:16]
            )

            documents.append(
                LocalDocument(
                    blob_name=blob.name,
                    container_name=self._container_name,
                    etag=blob.etag.strip('"'),
                    local_path=dest_path.resolve(),
                    sha256=hasher.hexdigest(),
                )
            )

        return documents
