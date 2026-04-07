# MyRAG

A production-style **Retrieval-Augmented Generation (RAG)** ingestion pipeline built on Azure services. It reads PDF documents from Azure Blob Storage, parses them with Azure Document Intelligence, chunks them into semantically coherent pieces, embeds them with Azure OpenAI, and indexes them in a Qdrant vector store — ready to be queried by any RAG application.

---

## Architecture

```
Azure Blob Storage
       │
       ▼
  BlobStorageClient          download PDFs → local disk
       │
       ▼
DocumentIntelligenceParser   extract sections, paragraphs, tables
       │
       ▼
     Chunker                 sentence-aware sliding window chunking
       │
       ▼
  EmbeddingClient            batch embed via Azure OpenAI
       │
       ▼
   QdrantIndexer             upsert vectors + metadata payload
```

Pipeline state is tracked per-document in **MongoDB** so re-runs skip already-indexed files (ETag-based idempotency).

---

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (package manager)
- Running [Qdrant](https://qdrant.tech/) instance (local or cloud)
- Running MongoDB instance (local or Atlas)
- Azure resources:
  - Azure Blob Storage container with PDFs
  - Azure Document Intelligence resource
  - Azure OpenAI resource with an embedding deployment

---

## Starting Qdrant

The pipeline requires a running Qdrant instance. The easiest way to run one locally is with Docker, mounting a local volume so your data persists across container restarts:

```bash
docker run -d \
  --name qdrant \
  -p 6333:6333 \
  -v $(pwd)/qdrant_storage:/qdrant/storage \
  qdrant/qdrant
```

| Flag | Purpose |
|---|---|
| `-p 6333:6333` | REST API (used by this pipeline) |
| `-v $(pwd)/qdrant_storage:/qdrant/storage` | Persists collections and vectors to `./qdrant_storage` on your host |

The Qdrant dashboard will be available at `http://localhost:6333/dashboard` once the container is running.

Set `QDRANT__URL=http://localhost:6333` in your `.env` file.

---

## Setup

```bash
# Clone and enter the project
git clone <repo-url>
cd MyRag

# Install dependencies
uv sync

# Download the spaCy language model (used for sentence splitting)
uv run python -m spacy download en_core_web_sm

# Copy and fill in your environment variables
cp .env.example .env
```

Edit `.env` with your Azure credentials and service URLs (see `.env.example` for all required variables).

---

## Running the pipeline

```bash
uv run python main.py
```

This runs the full ingestion pipeline and prints a summary:

```
Pipeline complete
  Total   : 5
  Indexed : 4
  Skipped : 1
  Failed  : 0
```

---

## Running tests

Pure function tests (no credentials required):

```bash
uv run python -m unittest discover -s tests -p "test_*.py" -v
```

Integration tests (require real Azure credentials and running services):

```bash
RUN_INTEGRATION=1 uv run python -m unittest discover -s tests -p "test_*.py" -v
```

---

## Project structure

```
myrag/
├── storage/        BlobStorageClient — download PDFs from Azure
├── parsing/        DocumentIntelligenceParser — extract structure from PDFs
├── chunking/       Chunker — sentence-aware sliding window
├── embeddings/     EmbeddingClient — batch embed via Azure OpenAI
├── vector_store/   QdrantIndexer — upsert vectors into Qdrant
├── pipeline.py     Orchestrates all stages end-to-end
├── models.py       Pydantic models shared across stages
└── config.py       Settings loaded from .env

tests/
├── chunking/
├── parsing/
├── storage/
├── embeddings/
└── vector_store/

main.py             Entry point — run the ingestion pipeline
```

---

## Environment variables

| Variable | Description |
|---|---|
| `BLOB__CONNECTION_STRING` | Azure Storage connection string |
| `BLOB__CONTAINER_NAME` | Container holding the PDFs |
| `DOC_INTELLIGENCE__ENDPOINT` | Azure Document Intelligence endpoint |
| `DOC_INTELLIGENCE__API_KEY` | Azure Document Intelligence API key |
| `DOC_INTELLIGENCE__MODEL_ID` | Document Intelligence model (e.g. `prebuilt-layout`) |
| `OPENAI__ENDPOINT` | Azure OpenAI endpoint |
| `OPENAI__API_KEY` | Azure OpenAI API key |
| `OPENAI__API_VERSION` | Azure OpenAI API version |
| `OPENAI__EMBEDDING_DEPLOYMENT_NAME` | Embedding model deployment name |
| `OPENAI__EMBEDDING_DIMENSIONS` | Embedding vector dimensions |
| `OPENAI__EMBEDDING_BATCH_SIZE` | Number of texts to embed per API call |
| `QDRANT__URL` | Qdrant server URL |
| `QDRANT__COLLECTION_NAME` | Qdrant collection to upsert vectors into |
| `MONGO__URI` | MongoDB connection URI |
| `MONGO__DATABASE` | MongoDB database name |
| `MONGO__COLLECTION` | Collection used to track pipeline state |
| `LOCAL__PDF_DIR` | Local directory for caching downloaded PDFs |
| `CHUNKING__CHUNK_SIZE_TOKENS` | Target chunk size in tokens |
| `CHUNKING__CHUNK_OVERLAP_TOKENS` | Overlap between consecutive chunks in tokens |
| `LOG__LEVEL` | Logging level (e.g. `INFO`, `DEBUG`) |
| `LOG__FORMAT` | Log format (`console` or `json`) |
