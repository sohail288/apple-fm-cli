# Apple Foundation Models - RAG Demo

This example demonstrates a local Retrieval-Augmented Generation (RAG) pipeline using:
- **Apple Intelligence** (via `apple-fm-cli`) for generation.
- **Native Apple Embeddings** for semantic search.
- **Qdrant** as the vector database.

## Prerequisites

1. **macOS 15+** with Apple Intelligence enabled.
2. **Docker** installed and running.
3. **Python 3.14+**.

## Setup

1. **Start Qdrant** (from this `examples/rag_demo` directory, uses `docker-compose.yml`):

   ```bash
   docker compose up -d
   ```

   - HTTP API: `http://localhost:6333` (gRPC/REST) — the demo uses the Python client on this port.
   - Dashboard: `http://localhost:6334` (if exposed by the image)
   - Stop: `docker compose down`
   - Remove data volume (fresh start): `docker compose down -v`

   If you do not use Compose, a one-liner is equivalent:  
   `docker run -d --name qdrant -p 6333:6333 -p 6334:6334 qdrant/qdrant:v1.13.1`

2. **Install dependencies** (from this directory):

   ```bash
   pip install -r requirements.txt
   ```

## Running the demo

With Qdrant already listening on **localhost:6333**:

```bash
# from examples/rag_demo — script has PEP 723 deps (qdrant-client, httpx, apple-fm-cli)
uv run rag_experiment.py
```

Or with **pip** and this folder’s `requirements.txt`:

```bash
pip install -r requirements.txt
python rag_experiment.py
```

## What happens?
1. Three "private" facts are embedded locally using the native Apple Natural Language framework.
2. These embeddings are stored in Qdrant.
3. A query is made about these facts.
4. The system retrieves the relevant context and uses Apple Intelligence to generate a grounded answer.

## In-memory retrieval (no Qdrant)

For a minimal pipeline or unit tests, the SDK exposes **cosine similarity** and **top‑k** over precomputed 512-d vectors: `from apple_fm_sdk import cosine_similarity, retrieve_top_k` (see `src/apple_fm_sdk/retrieval.py`).

In this repo, run **`uv run pytest tests/test_retrieval.py`** to exercise both deterministic vector tests and a live native-embedding check that a vault-related query ranks above unrelated passages.
