import logging
from openai import AzureOpenAI
from openai import APIConnectionError, APIStatusError, RateLimitError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)

from myrag.models import Chunk, EmbeddedChunk

logger = logging.getLogger(__name__)

# Retry on transient errors only — connection issues and rate limits.
# APIStatusError covers 5xx server errors.
_RETRY_EXCEPTIONS = (APIConnectionError, RateLimitError, APIStatusError)


class EmbeddingClient:
    """
    Context manager wrapping the Azure OpenAI embeddings API.

    Always use via `with`:

        with EmbeddingClient(endpoint, api_key, api_version, deployment) as client:
            embedded_chunks = client.embed(chunks)

    Args:
        endpoint: Azure OpenAI resource endpoint.
        api_key: Azure OpenAI API key.
        api_version: API version string (e.g. "2024-02-01").
        deployment: Name of the embedding model deployment.
        dimensions: Output vector dimensionality (default 3072).
        batch_size: Number of chunks per API call (default 16).
    """

    def __init__(
        self,
        endpoint: str,
        api_key: str,
        api_version: str,
        deployment: str,
        dimensions: int = 3072,
        batch_size: int = 16,
    ) -> None:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be a positive integer, got {batch_size}")
        if dimensions <= 0:
            raise ValueError(f"dimensions must be a positive integer, got {dimensions}")

        self._endpoint = endpoint
        self._api_key = api_key
        self._api_version = api_version
        self._deployment = deployment
        self._dimensions = dimensions
        self._batch_size = batch_size
        self._client: AzureOpenAI | None = None

    def __enter__(self) -> "EmbeddingClient":
        self._client = AzureOpenAI(
            azure_endpoint=self._endpoint,
            api_key=self._api_key,
            api_version=self._api_version,
        )
        return self

    def __exit__(self, *_: object) -> None:
        if self._client:
            self._client.close()

    def embed(self, chunks: list[Chunk]) -> list[EmbeddedChunk]:
        """
        Embed a list of chunks and return EmbeddedChunk objects in the same order.

        Raises:
            RuntimeError: if called outside of a `with` block.
            APIConnectionError: on persistent connection failures after retries.
            APIStatusError: on persistent server errors after retries.
        """
        if self._client is None:
            raise RuntimeError("EmbeddingClient must be used as a context manager.")

        if not chunks:
            logger.warning(
                "embed() called with empty chunk list — returning empty list."
            )
            return []

        logger.info(
            "Embedding %d chunks in batches of %d (deployment: '%s')",
            len(chunks),
            self._batch_size,
            self._deployment,
        )

        embedded: list[EmbeddedChunk] = []
        for batch_start in range(0, len(chunks), self._batch_size):
            batch = chunks[batch_start : batch_start + self._batch_size]
            embedded.extend(self._embed_batch(batch))
            logger.debug(
                "Embedded batch %d/%d",
                batch_start // self._batch_size + 1,
                -(-len(chunks) // self._batch_size),  # ceiling division
            )

        logger.info("Embedding complete — %d chunks embedded.", len(embedded))
        return embedded

    @retry(
        retry=retry_if_exception_type(_RETRY_EXCEPTIONS),
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        before_sleep=before_sleep_log(
            logger, logging.WARNING
        ),  # every time a retry is about to happen, it logs a WARNING message
        reraise=True,
    )
    def _embed_batch(self, batch: list[Chunk]) -> list[EmbeddedChunk]:
        """
        Call the Azure OpenAI embeddings API for a single batch.

        Retries up to 4 attempts with exponential backoff on transient errors.
        """
        texts = [chunk.content for chunk in batch]
        response = self._client.embeddings.create(
            model=self._deployment,
            input=texts,
            dimensions=self._dimensions,
        )

        return [
            EmbeddedChunk(
                chunk=chunk,
                embedding=item.embedding,
                embedding_model=self._deployment,
            )
            for chunk, item in zip(batch, response.data)
        ]
