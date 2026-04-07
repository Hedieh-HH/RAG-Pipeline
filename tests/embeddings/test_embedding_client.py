"""
Tests for EmbeddingClient.

Pure function tests — no credentials required, always run:
    python -m unittest tests/embeddings/test_embedding_client.py -v

Integration tests require real Azure credentials and RUN_INTEGRATION=1:
    RUN_INTEGRATION=1 python -m unittest tests/embeddings/test_embedding_client.py -v
"""

import os
import sys
import unittest
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent.parent / ".env")

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ragcore.embeddings.client import EmbeddingClient
from ragcore.models import Chunk, ChunkMetadata, EmbeddedChunk


# Helpers
def make_chunk(content: str = "Test content.") -> Chunk:
    return Chunk(
        content=content,
        token_count=len(content.split()),
        metadata=ChunkMetadata(
            doc_id="test.pdf",
            source_blob="container/test.pdf",
            etag="test-etag",
            chunk_index=0,
            chunk_total=1,
            section_heading="Introduction",
            start_page=1,
            end_page=1,
        ),
    )


# EmbeddingClient.__init__ validation
class TestEmbeddingClientInit(unittest.TestCase):

    def test_zero_batch_size_raises(self) -> None:
        with self.assertRaises(ValueError):
            EmbeddingClient(
                endpoint="https://fake.openai.azure.com/",
                api_key="fake-key",
                api_version="2024-02-01",
                deployment="text-embedding-3-large",
                batch_size=0,
            )

    def test_negative_batch_size_raises(self) -> None:
        with self.assertRaises(ValueError):
            EmbeddingClient(
                endpoint="https://fake.openai.azure.com/",
                api_key="fake-key",
                api_version="2024-02-01",
                deployment="text-embedding-3-large",
                batch_size=-1,
            )

    def test_zero_dimensions_raises(self) -> None:
        with self.assertRaises(ValueError):
            EmbeddingClient(
                endpoint="https://fake.openai.azure.com/",
                api_key="fake-key",
                api_version="2024-02-01",
                deployment="text-embedding-3-large",
                dimensions=0,
            )

    def test_negative_dimensions_raises(self) -> None:
        with self.assertRaises(ValueError):
            EmbeddingClient(
                endpoint="https://fake.openai.azure.com/",
                api_key="fake-key",
                api_version="2024-02-01",
                deployment="text-embedding-3-large",
                dimensions=-1,
            )

    def test_valid_params_do_not_raise(self) -> None:
        EmbeddingClient(
            endpoint="https://fake.openai.azure.com/",
            api_key="fake-key",
            api_version="2024-02-01",
            deployment="text-embedding-3-large",
            dimensions=3072,
            batch_size=16,
        )


# Integration tests — skipped unless RUN_INTEGRATION=1
@unittest.skipUnless(os.getenv("RUN_INTEGRATION"), "Set RUN_INTEGRATION=1 to run")
class TestEmbeddingClientIntegration(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        from ragcore.config import get_settings

        settings = get_settings()
        cls.client = EmbeddingClient(
            endpoint=settings.openai.endpoint,
            api_key=settings.openai.api_key,
            api_version="2024-02-01",
            deployment=settings.openai.embedding_deployment_name,
            dimensions=settings.openai.embedding_dimensions,
            batch_size=settings.openai.embedding_batch_size,
        )
        cls.client.__enter__()
        cls.dimensions = settings.openai.embedding_dimensions

    @classmethod
    def tearDownClass(cls) -> None:
        cls.client.__exit__(None, None, None)

    def test_embed_returns_embedded_chunks(self) -> None:
        chunks = [
            make_chunk("Knowledge distillation transfers knowledge from a large model.")
        ]
        result = self.client.embed(chunks)
        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], EmbeddedChunk)

    def test_embed_vectors_have_correct_dimension(self) -> None:
        chunks = [make_chunk("Sample text for embedding.")]
        result = self.client.embed(chunks)
        self.assertEqual(len(result[0].embedding), self.dimensions)

    def test_embed_preserves_order(self) -> None:
        chunks = [make_chunk(f"Sentence number {i}.") for i in range(5)]
        result = self.client.embed(chunks)
        for i, ec in enumerate(result):
            self.assertIn(str(i), ec.chunk.content)

    def test_embed_empty_list_returns_empty(self) -> None:
        result = self.client.embed([])
        self.assertEqual(result, [])

    def test_embed_multiple_batches(self) -> None:
        small_client = EmbeddingClient(
            endpoint=self.client._endpoint,
            api_key=self.client._api_key,
            api_version=self.client._api_version,
            deployment=self.client._deployment,
            dimensions=self.client._dimensions,
            batch_size=2,
        )
        chunks = [make_chunk(f"Chunk {i} content.") for i in range(5)]
        with small_client as c:
            result = c.embed(chunks)
        self.assertEqual(len(result), 5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
