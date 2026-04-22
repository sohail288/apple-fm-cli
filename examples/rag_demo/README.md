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

1. **Start the Vector Database (Qdrant)**:
   ```bash
   docker run -d -p 6333:6333 -p 6334:6334 qdrant/qdrant
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Demo

Run the experiment script:
```bash
python rag_experiment.py
```

Alternatively, if you have `uv` installed, you can run it without manual setup:
```bash
uv run rag_experiment.py
```

## What happens?
1. Three "private" facts are embedded locally using the native Apple Natural Language framework.
2. These embeddings are stored in Qdrant.
3. A query is made about these facts.
4. The system retrieves the relevant context and uses Apple Intelligence to generate a grounded answer.
