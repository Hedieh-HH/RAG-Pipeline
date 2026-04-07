from pathlib import Path

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class BlobSettings(BaseSettings):
    """Azure Blob Storage connection details."""

    model_config = SettingsConfigDict(env_prefix="BLOB__")

    connection_string: str = Field(
        ...,
        description="Azure Storage connection string.",
    )
    container_name: str = Field(
        ...,
        description="Name of the blob container that holds the source PDFs.",
    )


class DocIntelligenceSettings(BaseSettings):
    """Azure Document Intelligence settings."""

    model_config = SettingsConfigDict(env_prefix="DOC_INTELLIGENCE__")

    endpoint: str = Field(
        ...,
        description="Azure Document Intelligence endpoint URL.",
    )
    api_key: str = Field(
        ...,
        description="Azure Document Intelligence API key.",
    )
    model_id: str = Field(
        default="prebuilt-layout",
        description=(
            "Document Intelligence model to use. "
            "'prebuilt-layout' extracts paragraphs, tables, sections and headings."
        ),
    )


class OpenAISettings(BaseSettings):
    """Azure OpenAI settings for text embeddings."""

    model_config = SettingsConfigDict(env_prefix="OPENAI__")

    endpoint: str = Field(
        ...,
        description="Azure OpenAI resource endpoint",
    )
    api_key: str = Field(
        ...,
        description="Azure OpenAI API key.",
    )
    api_version: str = Field(
        default="2024-12-01",
        description="Azure OpenAI API version string.",
    )
    embedding_deployment_name: str = Field(
        default="text-embedding-3-large",
        description="Name of the deployed embedding model in your Azure OpenAI resource.",
    )
    embedding_dimensions: int = Field(
        default=3072,
        ge=256,
        le=3072,
        description=("Output vector dimensionality. "),
    )
    embedding_batch_size: int = Field(
        default=16,
        ge=1,
        le=2048,
        description="Number of text chunks to embed in a single API call.",
    )


class QdrantSettings(BaseSettings):
    """Qdrant vector store connection settings."""

    model_config = SettingsConfigDict(env_prefix="QDRANT__")

    url: str = Field(
        default="http://localhost:6333",
        description="Qdrant server URL",
    )
    collection_name: str = Field(
        ...,
        description="Name of the Qdrant collection to write vectors into.",
    )
    vector_size: int = Field(
        default=3072,
        ge=1,
        description="Dimensionality of the embedding vectors. Must match the embedding model output.",
    )
    batch_size: int = Field(
        default=100,
        ge=1,
        description="Number of points per upsert call to Qdrant.",
    )


class LocalStorageSettings(BaseSettings):
    """Local filesystem settings for caching downloaded PDFs."""

    model_config = SettingsConfigDict(env_prefix="LOCAL__")

    pdf_dir: Path = Field(
        default=Path("./data/pdfs"),
        description="Directory where downloaded PDFs are stored.",
    )


class MongoSettings(BaseSettings):
    """MongoDB settings for pipeline state tracking."""

    model_config = SettingsConfigDict(env_prefix="MONGO__")

    uri: str = Field(
        default="mongodb://localhost:27017",
        description="MongoDB connection URI.",
    )
    database: str = Field(
        ...,
        description="Database name.",
    )
    collection: str = Field(
        default="document_status",
        description="Collection that stores per-blob processing state.",
    )


class ChunkingSettings(BaseSettings):
    """Chunking strategy parameters."""

    model_config = SettingsConfigDict(env_prefix="CHUNKING__")

    chunk_size_tokens: int = Field(
        default=512,
        ge=64,
        le=8192,
        description="Target maximum token count per chunk.",
    )
    chunk_overlap_tokens: int = Field(
        default=100,
        ge=0,
        description="Number of tokens to overlap between consecutive chunks.",
    )

    @model_validator(mode="after")
    def overlap_must_be_less_than_chunk_size(self) -> "ChunkingSettings":
        if self.chunk_overlap_tokens >= self.chunk_size_tokens:
            raise ValueError(
                f"chunk_overlap_tokens ({self.chunk_overlap_tokens}) must be "
                f"less than chunk_size_tokens ({self.chunk_size_tokens})."
            )
        return self


class LoggingSettings(BaseSettings):
    """Structured logging configuration."""

    model_config = SettingsConfigDict(env_prefix="LOG__")

    level: str = Field(
        default="INFO",
        description="Log level: DEBUG, INFO, WARNING, ERROR, CRITICAL.",
    )
    format: str = Field(
        default="console",
        description=(
            "'console' for human-readable output (development); "
            "'json' for structured JSON (production / log aggregators)."
        ),
    )

    @field_validator("level")
    @classmethod
    def normalise_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in allowed:
            raise ValueError(f"log level must be one of {allowed}, got '{v}'")
        return upper

    @field_validator("format")
    @classmethod
    def normalise_format(cls, v: str) -> str:
        allowed = {"console", "json"}
        lower = v.lower()
        if lower not in allowed:
            raise ValueError(f"log format must be one of {allowed}, got '{v}'")
        return lower


class Settings(BaseSettings):
    """
    Root settings object.  Load once at application startup via `get_settings()`.

    .env file lookup order:
      1. .env  (project root)
      2. Environment variables (takes precedence over .env)
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_nested_delimiter="__",  # X__Y maps to X.Y
        case_sensitive=False,
        extra="ignore",
    )

    blob: BlobSettings
    doc_intelligence: DocIntelligenceSettings
    openai: OpenAISettings
    qdrant: QdrantSettings
    mongo: MongoSettings
    local: LocalStorageSettings = Field(default_factory=LocalStorageSettings)
    chunking: ChunkingSettings = Field(default_factory=ChunkingSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)


# Singleton accessor (cached after first call)
_settings: Settings | None = None


def get_settings() -> Settings:
    """Return the validated Settings singleton, initialising it on first call."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
