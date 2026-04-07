import logging
from dataclasses import dataclass

import pymongo
import pymongo.collection

from myrag.chunking.chunker import Chunker
from myrag.config import Settings
from myrag.embeddings.client import EmbeddingClient
from myrag.models import LocalDocument, DocumentRecord
from myrag.parsing.parser import DocumentIntelligenceParser
from myrag.storage.blob_client import BlobStorageClient
from myrag.vector_store.qdrant import QdrantIndexer

logger = logging.getLogger(__name__)


class DocumentStatusStore:
    """
    Reads and writes document_status records in MongoDB.

    One record per blob, keyed by blob_name. Upserts on save so re-runs
    update the existing record rather than inserting duplicates.
    """

    def __init__(self, collection: pymongo.collection.Collection) -> None:
        self._col = collection

    def get(self, blob_name: str) -> DocumentRecord | None:
        """Return the document_status entry for a blob, or None if not seen before."""
        doc = self._col.find_one({"blob_name": blob_name})
        if doc is None:
            return None
        doc.pop("_id", None)
        return DocumentRecord.model_validate(doc)

    def save(self, entry: DocumentRecord) -> None:
        """Upsert a manifest entry keyed by blob_name."""
        self._col.update_one(
            {"blob_name": entry.blob_name},
            {"$set": entry.model_dump(mode="json")},
            upsert=True,
        )


@dataclass
class PipelineResult:
    """Summary returned after a pipeline run."""

    total: int = 0
    skipped: int = 0
    indexed: int = 0
    failed: int = 0


class Pipeline:
    """
    Orchestrates the full RAG ingestion pipeline.

    Args:
        settings: Validated application settings loaded from .env.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def run(self) -> PipelineResult:
        """
        Run the full pipeline: download → parse → chunk → embed → index.

        Returns a PipelineResult with counts of skipped, indexed, and failed documents.
        """
        s = self._settings
        result = PipelineResult()

        # Set up MongoDB manifest
        mongo_client = pymongo.MongoClient(s.mongo.uri)
        db = mongo_client[s.mongo.database]
        document_status_collection = DocumentStatusStore(db[s.mongo.collection])

        # Download all blobs
        logger.info("Starting pipeline run.")
        with BlobStorageClient(
            connection_string=s.blob.connection_string,
            container_name=s.blob.container_name,
            pdf_dir=s.local.pdf_dir,
        ) as blob_client:
            documents = blob_client.fetch_all()

        result.total = len(documents)
        logger.info("Downloaded %d documents.", result.total)

        if not documents:
            logger.warning(
                "No documents found in container '%s'.", s.blob.container_name
            )
            mongo_client.close()
            return result

        # Process each document
        with (
            DocumentIntelligenceParser(
                endpoint=s.doc_intelligence.endpoint,
                api_key=s.doc_intelligence.api_key,
                model_id=s.doc_intelligence.model_id,
            ) as parser,
            EmbeddingClient(
                endpoint=s.openai.endpoint,
                api_key=s.openai.api_key,
                api_version=s.openai.api_version,
                deployment=s.openai.embedding_deployment_name,
                dimensions=s.openai.embedding_dimensions,
                batch_size=s.openai.embedding_batch_size,
            ) as embedder,
            QdrantIndexer(
                url=s.qdrant.url,
                collection_name=s.qdrant.collection_name,
                vector_size=s.qdrant.vector_size,
                batch_size=s.qdrant.batch_size,
            ) as indexer,
        ):
            indexer.ensure_collection()
            chunker = Chunker(
                chunk_size_tokens=s.chunking.chunk_size_tokens,
                overlap_tokens=s.chunking.chunk_overlap_tokens,
            )

            for doc in documents:
                try:
                    self._process_document(
                        doc,
                        parser,
                        chunker,
                        embedder,
                        indexer,
                        document_status_collection,
                        result,
                    )
                except Exception as e:
                    logger.error(
                        "Unexpected error processing '%s': %s", doc.blob_name, e
                    )
                    entry = document_status_collection.get(
                        doc.blob_name
                    ) or DocumentRecord(
                        blob_name=doc.blob_name,
                        etag=doc.etag,
                        local_path=doc.local_path,
                    )
                    entry.mark_failed(str(e))
                    document_status_collection.save(entry)
                    result.failed += 1

        mongo_client.close()
        logger.info(
            "Pipeline complete — total: %d, indexed: %d, skipped: %d, failed: %d",
            result.total,
            result.indexed,
            result.skipped,
            result.failed,
        )
        return result

    def _process_document(
        self,
        doc: LocalDocument,
        parser: DocumentIntelligenceParser,
        chunker: Chunker,
        embedder: EmbeddingClient,
        indexer: QdrantIndexer,
        document_status_collection: DocumentStatusStore,
        result: PipelineResult,
    ) -> None:
        """Process a single document through all pipeline stages."""
        # Check manifest — skip if already indexed with the same ETag
        entry = document_status_collection.get(doc.blob_name)
        if entry and entry.is_up_to_date and entry.etag == doc.etag:
            logger.info(
                "Skipping '%s' — already indexed (ETag unchanged).", doc.blob_name
            )
            result.skipped += 1
            return

        # Create or reset the manifest entry for this run
        entry = DocumentRecord(
            blob_name=doc.blob_name,
            etag=doc.etag,
            local_path=doc.local_path,
        )
        entry.mark_downloaded()
        document_status_collection.save(entry)

        # Parse
        logger.info("Parsing '%s'.", doc.blob_name)
        parsed = parser.parse(doc)
        entry.mark_parsed()
        document_status_collection.save(entry)

        # Chunk
        chunks = chunker.chunk(parsed)
        if not chunks:
            logger.warning("'%s' produced no chunks — skipping.", doc.blob_name)
            result.skipped += 1
            return
        entry.mark_chunked(len(chunks))
        document_status_collection.save(entry)

        # Embed
        logger.info("Embedding %d chunks for '%s'.", len(chunks), doc.blob_name)
        embedded = embedder.embed(chunks)

        # Index
        indexer.upsert(embedded)
        entry.mark_indexed()
        document_status_collection.save(entry)

        result.indexed += 1
        logger.info("Indexed '%s' (%d chunks).", doc.blob_name, len(chunks))
