import logging

from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import Distance, PointStruct, VectorParams

from myrag.models import EmbeddedChunk, QdrantPayload

logger = logging.getLogger(__name__)

_DEFAULT_BATCH_SIZE = 100


class QdrantIndexer:
    """
    Context manager wrapping the Qdrant client.

    Always use via `with`:

        with QdrantIndexer(url, collection_name, vector_size) as indexer:
            indexer.ensure_collection()
            indexer.upsert(embedded_chunks)

    Args:
        url: Qdrant server URL (e.g. "http://localhost:6333").
        collection_name: Name of the collection to write vectors into.
        vector_size: Dimensionality of the embedding vectors (default 3072).
        batch_size: Number of points per upsert call (default 100).
    """

    def __init__(
        self,
        url: str,
        collection_name: str,
        vector_size: int = 3072,
        batch_size: int = _DEFAULT_BATCH_SIZE,
    ) -> None:
        if vector_size <= 0:
            raise ValueError(
                f"vector_size must be a positive integer, got {vector_size}"
            )
        if batch_size <= 0:
            raise ValueError(f"batch_size must be a positive integer, got {batch_size}")

        self._url = url
        self._collection_name = collection_name
        self._vector_size = vector_size
        self._batch_size = batch_size
        self._client: QdrantClient | None = None

    def __enter__(self) -> "QdrantIndexer":
        self._client = QdrantClient(url=self._url)
        logger.debug("Connected to Qdrant at '%s'", self._url)
        return self

    def __exit__(self, *_: object) -> None:
        if self._client:
            self._client.close()

    def ensure_collection(self) -> None:
        """
        Create the collection if it does not already exist.

        Uses cosine distance — standard for text embedding similarity.

        Raises:
            RuntimeError: if called outside of a `with` block.
            UnexpectedResponse: on Qdrant API errors.
        """
        self._require_client()

        if self._client.collection_exists(self._collection_name):
            logger.debug(
                "Collection '%s' already exists — skipping creation.",
                self._collection_name,
            )
            return

        self._client.create_collection(
            collection_name=self._collection_name,
            vectors_config=VectorParams(
                size=self._vector_size,
                distance=Distance.COSINE,
            ),
        )
        logger.info(
            "Created Qdrant collection '%s' (size=%d, distance=cosine).",
            self._collection_name,
            self._vector_size,
        )

    def upsert(self, embedded_chunks: list[EmbeddedChunk]) -> None:
        """
        Upsert a list of EmbeddedChunks into the collection.

        Converts each EmbeddedChunk to a Qdrant PointStruct and upserts in
        batches. Existing points with the same chunk_id are overwritten.

        Raises:
            RuntimeError: if called outside of a `with` block.
            UnexpectedResponse: on Qdrant API errors.
        """
        self._require_client()

        if not embedded_chunks:
            logger.warning("upsert() called with empty list — nothing to index.")
            return

        logger.info(
            "Upserting %d points into collection '%s' in batches of %d.",
            len(embedded_chunks),
            self._collection_name,
            self._batch_size,
        )

        for batch_start in range(0, len(embedded_chunks), self._batch_size):
            batch = embedded_chunks[batch_start : batch_start + self._batch_size]
            points = [self._to_point(ec) for ec in batch]

            try:
                self._client.upsert(
                    collection_name=self._collection_name,
                    points=points,
                )
            except UnexpectedResponse as e:
                logger.error(
                    "Qdrant upsert failed for batch starting at index %d (doc_id: '%s'): %s",
                    batch_start,
                    batch[0].chunk.metadata.doc_id,
                    e,
                )
                raise

            logger.debug(
                "Upserted batch %d/%d (%d points).",
                batch_start // self._batch_size + 1,
                -(-len(embedded_chunks) // self._batch_size),
                len(points),
            )

        logger.info("Upsert complete — %d points indexed.", len(embedded_chunks))

    def _to_point(self, ec: EmbeddedChunk) -> PointStruct:
        """Convert an EmbeddedChunk to a Qdrant PointStruct."""
        payload = QdrantPayload.from_embedded_chunk(ec)
        return PointStruct(
            id=payload.chunk_id,
            vector=ec.embedding,
            payload=payload.model_dump(),
        )

    def _require_client(self) -> None:
        if self._client is None:
            raise RuntimeError("QdrantIndexer must be used as a context manager.")
