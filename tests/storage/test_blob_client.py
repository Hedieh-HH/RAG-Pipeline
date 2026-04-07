"""
Tests for BlobStorageClient.

Require real Azure credentials in .env:
    BLOB__CONNECTION_STRING=...
    BLOB__CONTAINER_NAME=...

Skipped by default. To run:
    RUN_INTEGRATION=1 python -m unittest tests/storage/test_blob_client.py -v
"""

import os
import sys
import unittest
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent.parent / ".env")

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ragcore.models import LocalDocument
from ragcore.storage.blob_client import BlobStorageClient


@unittest.skipUnless(os.getenv("RUN_INTEGRATION"), "Set RUN_INTEGRATION=1 to run")
class TestBlobClientIntegration(unittest.TestCase):

    def setUp(self) -> None:
        from ragcore.config import BlobSettings, LocalStorageSettings

        blob_settings = BlobSettings()  # type: ignore[call-arg]
        local_settings = LocalStorageSettings(pdf_dir=Path("./data/pdfs"))
        self.client = BlobStorageClient(
            connection_string=blob_settings.connection_string,
            container_name=blob_settings.container_name,
            pdf_dir=local_settings.pdf_dir,
        )
        self.client.__enter__()

    def tearDown(self) -> None:
        self.client.__exit__(None, None, None)

    def test_fetch_all_returns_list(self) -> None:
        """fetch_all should always return a list, even if the container is empty."""
        documents = self.client.fetch_all()
        self.assertIsInstance(documents, list)

    def test_fetch_all_returns_local_documents(self) -> None:
        """Every item returned must be a LocalDocument."""
        documents = self.client.fetch_all()
        for doc in documents:
            self.assertIsInstance(doc, LocalDocument)

    def test_fetch_all_files_exist_on_disk(self) -> None:
        """Every downloaded file must actually exist at the reported local path."""
        documents = self.client.fetch_all()
        for doc in documents:
            self.assertTrue(
                doc.local_path.exists(), f"Expected file not found: {doc.local_path}"
            )

    def test_fetch_all_local_paths_are_pdfs(self) -> None:
        """All downloaded files must have a .pdf extension."""
        documents = self.client.fetch_all()
        for doc in documents:
            self.assertEqual(doc.local_path.suffix.lower(), ".pdf")

    def test_fetch_all_etag_is_populated(self) -> None:
        """ETag must be a non-empty string for every document."""
        documents = self.client.fetch_all()
        for doc in documents:
            self.assertIsInstance(doc.etag, str)
            self.assertGreater(
                len(doc.etag), 0, f"Empty ETag for blob: {doc.blob_name}"
            )

    def test_fetch_all_sha256_is_valid_hex(self) -> None:
        """SHA-256 must be a 64-character hex string."""
        documents = self.client.fetch_all()
        for doc in documents:
            self.assertEqual(
                len(doc.sha256), 64, f"Invalid SHA-256 for blob: {doc.blob_name}"
            )
            self.assertTrue(
                all(c in "0123456789abcdef" for c in doc.sha256),
                f"SHA-256 contains non-hex characters: {doc.sha256}",
            )

    def test_fetch_all_container_name_is_populated(self) -> None:
        """container_name must be set on every document."""
        documents = self.client.fetch_all()
        for doc in documents:
            self.assertIsInstance(doc.container_name, str)
            self.assertGreater(len(doc.container_name), 0)

    def test_fetch_all_source_blob_contains_container(self) -> None:
        """container_name must match the configured container."""
        from ragcore.config import BlobSettings

        container = BlobSettings().container_name  # type: ignore[call-arg]
        documents = self.client.fetch_all()
        for doc in documents:
            self.assertEqual(
                doc.container_name,
                container,
                f"Expected container '{container}', got '{doc.container_name}'",
            )


if __name__ == "__main__":
    unittest.main()
