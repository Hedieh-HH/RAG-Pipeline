from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from myrag.utils import utcnow


class LocalDocument(BaseModel):
    """A PDF that has been downloaded to local disk."""

    blob_name: str
    container_name: str
    etag: str
    local_path: Path
    sha256: str


@dataclass
class ParsedElement:
    """A cleaned element produced by the parser, before section assembly.
    Currently covers paragraphs; will extend to tables in future.
    """

    type: str  # "title", "heading", "paragraph", "footnote", "table"
    content: str
    bbox: list  # list of BoundingRegion from Azure SDK (multiple entries when element spans pages)
    offset: int
    index: (
        int | list[int]
    )  # paragraph index for paragraphs; list of paragraph indices for tables


@dataclass
class Section:
    """A group of ParsedElements under a single heading."""

    heading: str
    level: int  # 0 for root (pre-heading content), 1+ for headings
    elements: list[ParsedElement]


class ParsedDocument(BaseModel):
    """The full structured representation of a PDF after Document Intelligence parsing."""

    model_config = {"arbitrary_types_allowed": True}

    doc_id: str = Field(description="Stable identifier — the blob name.")
    doc_title: str | None = Field(
        default=None,
        description="Document title extracted from the TITLE element, if present.",
    )
    source_blob: str = Field(description="Full blob path: '<container>/<name>'.")
    etag: str = Field(description="ETag of the source blob at parse time.")
    local_path: Path = Field(description="Path to the source PDF on local disk.")
    parsed_at: datetime = Field(default_factory=utcnow)
    page_count: int = Field(ge=1)
    sections: list[Section] = Field(
        description="Ordered list of sections extracted from the document."
    )


class ChunkMetadata(BaseModel):
    """
    Provenance metadata attached to every chunk.

    Stored as the Qdrant point payload to enable filtered search,
    retrieval-time context stitching, and auditability.
    """

    doc_id: str
    doc_title: str | None = None
    source_blob: str
    etag: str
    chunk_index: int = Field(
        ge=0, description="Zero-based position within the document."
    )
    chunk_total: int = Field(
        ge=1, description="Total chunks produced from this document."
    )
    section_heading: str = Field(
        description="Heading of the section this chunk belongs to."
    )
    start_page: int = Field(ge=1)
    end_page: int = Field(ge=1)


class Chunk(BaseModel):
    """
    A semantic chunk ready for embedding.

    `prev_chunk_id` and `next_chunk_id` form an explicit doubly-linked list
    within the document, enabling retrieval-time context stitching.
    """

    id: UUID = Field(default_factory=uuid4)
    content: str = Field(description="Text content to be embedded.")
    token_count: int = Field(ge=1)
    metadata: ChunkMetadata
    prev_chunk_id: UUID | None = Field(default=None)
    next_chunk_id: UUID | None = Field(default=None)


class EmbeddedChunk(BaseModel):
    """A Chunk paired with its embedding vector, ready for Qdrant upsert."""

    chunk: Chunk
    embedding: list[float] = Field(description="Dense embedding vector.")
    embedding_model: str = Field(
        description="Deployment name of the embedding model used."
    )
    embedded_at: datetime = Field(default_factory=utcnow)


class QdrantPayload(BaseModel):
    """
    The payload stored alongside each vector in Qdrant.

    All UUIDs are stored as strings for JSON compatibility.
    """

    # Identity
    chunk_id: str = Field(
        description="UUID of this chunk, used as the Qdrant point ID."
    )
    prev_chunk_id: str | None = Field(
        default=None, description="UUID of the preceding chunk in the same document."
    )
    next_chunk_id: str | None = Field(
        default=None, description="UUID of the following chunk in the same document."
    )

    # Content
    content: str = Field(description="Text content that was embedded.")
    token_count: int = Field(description="Number of tokens in the content.")

    # Provenance
    doc_id: str = Field(
        description="Blob filename — stable identifier for the source document."
    )
    doc_title: str | None = Field(
        default=None,
        description="Human-readable title extracted from the document, if present.",
    )
    source_blob: str = Field(
        description="Full Azure Blob Storage path: '<container>/<blob-name>'."
    )
    etag: str = Field(
        description="Azure blob ETag at index time — used to detect stale chunks on re-runs."
    )

    # Position
    chunk_index: int = Field(
        description="Zero-based position of this chunk within its document."
    )
    chunk_total: int = Field(
        description="Total number of chunks produced from this document."
    )
    section_heading: str = Field(
        description="Normalised heading of the section this chunk belongs to."
    )
    start_page: int = Field(description="First page this chunk's content appears on.")
    end_page: int = Field(description="Last page this chunk's content appears on.")

    # Audit
    embedding_model: str = Field(
        description="Deployment name of the embedding model used."
    )
    embedded_at: str = Field(
        description="ISO 8601 UTC timestamp of when the embedding was created."
    )

    @classmethod
    def from_embedded_chunk(cls, ec: EmbeddedChunk) -> "QdrantPayload":
        c = ec.chunk
        m = c.metadata
        return cls(
            chunk_id=str(c.id),
            prev_chunk_id=str(c.prev_chunk_id) if c.prev_chunk_id else None,
            next_chunk_id=str(c.next_chunk_id) if c.next_chunk_id else None,
            content=c.content,
            token_count=c.token_count,
            doc_id=m.doc_id,
            doc_title=m.doc_title,
            source_blob=m.source_blob,
            etag=m.etag,
            chunk_index=m.chunk_index,
            chunk_total=m.chunk_total,
            section_heading=m.section_heading,
            start_page=m.start_page,
            end_page=m.end_page,
            embedding_model=ec.embedding_model,
            embedded_at=ec.embedded_at.isoformat(),
        )


class ProcessingStage(StrEnum):
    """Pipeline stages used in the manifest for state tracking."""

    PENDING = "pending"
    DOWNLOADED = "downloaded"
    PARSED = "parsed"
    CHUNKED = "chunked"
    INDEXED = "indexed"
    FAILED = "failed"


class DocumentRecord(BaseModel):
    """Per-blob processing record in MongoDB for idempotent re-runs."""

    blob_name: str
    etag: str = Field(description="ETag at the time of last successful download.")
    local_path: Path
    stage: ProcessingStage = ProcessingStage.PENDING
    chunk_count: int = Field(default=0)
    error: str | None = Field(default=None)
    downloaded_at: datetime | None = None
    parsed_at: datetime | None = None
    chunked_at: datetime | None = None
    indexed_at: datetime | None = None
    last_updated: datetime = Field(default_factory=utcnow)

    def mark_downloaded(self) -> None:
        self.stage = ProcessingStage.DOWNLOADED
        self.downloaded_at = utcnow()
        self.last_updated = utcnow()
        self.error = None

    def mark_parsed(self) -> None:
        self.stage = ProcessingStage.PARSED
        self.parsed_at = utcnow()
        self.last_updated = utcnow()

    def mark_chunked(self, chunk_count: int) -> None:
        self.stage = ProcessingStage.CHUNKED
        self.chunk_count = chunk_count
        self.chunked_at = utcnow()
        self.last_updated = utcnow()

    def mark_indexed(self) -> None:
        self.stage = ProcessingStage.INDEXED
        self.indexed_at = utcnow()
        self.last_updated = utcnow()

    def mark_failed(self, error: str) -> None:
        self.stage = ProcessingStage.FAILED
        self.error = error
        self.last_updated = utcnow()

    @property
    def is_up_to_date(self) -> bool:
        return self.stage == ProcessingStage.INDEXED
