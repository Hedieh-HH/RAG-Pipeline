"""
Tests for QdrantIndexer and QdrantPayload.

Pure function tests — no credentials required, always run:
    python -m unittest tests/vector_store/test_qdrant_indexer.py -v

Integration tests require a running Qdrant instance and RUN_INTEGRATION=1:
    RUN_INTEGRATION=1 python -m unittest tests/vector_store/test_qdrant_indexer.py -v
"""

import os
import sys
import unittest
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent.parent / ".env")

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ragcore.models import Chunk, ChunkMetadata, EmbeddedChunk, QdrantPayload
from ragcore.vector_store.qdrant import QdrantIndexer


# Helpers
def make_embedded_chunk(
    content: str = "Test content.",
    vector_size: int = 8,
    chunk_index: int = 0,
    chunk_total: int = 1,
) -> EmbeddedChunk:
    chunk = Chunk(
        content=content,
        token_count=len(content.split()),
        metadata=ChunkMetadata(
            doc_id="test.pdf",
            source_blob="container/test.pdf",
            etag="test-etag",
            chunk_index=chunk_index,
            chunk_total=chunk_total,
            section_heading="Introduction",
            start_page=1,
            end_page=1,
        ),
    )
    return EmbeddedChunk(
        chunk=chunk,
        embedding=[0.1] * vector_size,
        embedding_model="text-embedding-3-large",
    )


# Integration tests — skipped unless RUN_INTEGRATION=1
@unittest.skipUnless(os.getenv("RUN_INTEGRATION"), "Set RUN_INTEGRATION=1 to run")
class TestQdrantIndexerIntegration(unittest.TestCase):

    _TEST_COLLECTION = f"test_{uuid4().hex[:8]}"
    _VECTOR_SIZE = 8

    @classmethod
    def setUpClass(cls) -> None:
        from ragcore.config import get_settings

        settings = get_settings()
        cls.url = settings.qdrant.url
        cls.indexer = QdrantIndexer(
            url=cls.url,
            collection_name=cls._TEST_COLLECTION,
            vector_size=cls._VECTOR_SIZE,
        )
        cls.indexer.__enter__()
        try:
            cls.indexer.ensure_collection()
        except Exception as e:
            raise unittest.SkipTest(
                f"Qdrant is not reachable at {cls.url} — start it before running integration tests. ({e})"
            )

    @classmethod
    def tearDownClass(cls) -> None:
        from qdrant_client import QdrantClient

        cls.indexer.__exit__(None, None, None)
        client = QdrantClient(url=cls.url)
        client.delete_collection(cls._TEST_COLLECTION)
        client.close()

    def test_ensure_collection_is_idempotent(self) -> None:
        # Calling again on an existing collection must not raise.
        self.indexer.ensure_collection()

    def test_upsert_stores_points(self) -> None:
        chunks = [
            make_embedded_chunk(f"Content {i}.", vector_size=self._VECTOR_SIZE)
            for i in range(3)
        ]
        self.indexer.upsert(chunks)

        from qdrant_client import QdrantClient

        client = QdrantClient(url=self.url)
        ids = [str(ec.chunk.id) for ec in chunks]
        points = client.retrieve(
            collection_name=self._TEST_COLLECTION,
            ids=ids,
            with_payload=True,
        )
        client.close()
        self.assertEqual(len(points), 3)

    def test_upsert_payload_fields_present(self) -> None:
        ec = make_embedded_chunk("Payload check.", vector_size=self._VECTOR_SIZE)
        self.indexer.upsert([ec])

        from qdrant_client import QdrantClient

        client = QdrantClient(url=self.url)
        points = client.retrieve(
            collection_name=self._TEST_COLLECTION,
            ids=[str(ec.chunk.id)],
            with_payload=True,
        )
        client.close()
        payload = points[0].payload
        self.assertIn("content", payload)
        self.assertIn("doc_id", payload)
        self.assertIn("section_heading", payload)
        self.assertIn("chunk_id", payload)

    def test_upsert_empty_does_not_raise(self) -> None:
        self.indexer.upsert([])


if __name__ == "__main__":
    unittest.main(verbosity=2)
