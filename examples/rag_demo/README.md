# Apple Foundation Models - RAG Demo

This example demonstrates a local Retrieval-Augmented Generation (RAG) pipeline using:
- **Apple Intelligence** (via `apple-fm-cli`) for generation.
- **Native Apple Embeddings** for semantic search.
- **Qdrant** as the vector database.

There are two entry points: a **multi-format ingest + ask** flow (text, Markdown, CSV, and PDF) and a **single-script** minimal experiment (`rag_experiment.py`).

**Longer write-up (architecture, what we built, and how to explain it to a manager):** see [`RAG_PIPELINE_EXPLAINED.md`](RAG_PIPELINE_EXPLAINED.md).

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

3. (Optional) **Regenerate the sample PDF** if you changed the legal text or do not have the committed `sample_corpus/04_legal/legal_handbook_excerpt.pdf` (requires `fpdf2`):

   ```bash
   uv run --with fpdf2 python generate_sample_corpus.py
   ```

## Multi-format RAG (sample corpus)

The `sample_corpus/` tree mixes formats so ingest exercises **chunking**, CSV rows, and **PDF text extraction** (via `pypdf`).

**Chunking strategy (two layers):**

1. **Hierarchical parents** — **PDFs** are split one *page* at a time (no leaf crosses pages). **Markdown** is split on `#` / `##` / `###` into section parents. **Plain text and CSV** use a single document parent. Under each parent, a **recursive character** leaf pass runs (via `langchain-text-splitters`: paragraphs → lines → `". "` → words → characters, with **overlap** between leaves).

2. **Contextual vectors** — each leaf’s embedding is computed from a **short preamble** (source id, hierarchy path, and a truncated *parent* excerpt) plus the raw *Passage*, so the vector “knows” *which file and which section/page* the text comes from, similar in spirit to contextual retrieval patterns. The stored payload still includes the **raw passage** and **parent scope** for the answer model. Tune with `rag_ingest.py --chunk-size` and `--overlap`.

3. **Optional: `--semantic-within-block`** — for long parents, **sentence** boundaries can be pre-clustered with embedding **cosine similarity** before leaf chunking (extra embedding work; use `--max-semantic-embeds-per-block` to cap it).

| Path | Type | Role |
|------|------|------|
| `01_product/product_briefing.txt` | Text | Product / support / policy references |
| `02_ops/incidents_q2.csv` | CSV | Operational incident rows (embedded as one chunk per row) |
| `03_people/remote_work_policy.md` | Markdown | Internal HR / hybrid work rules |
| `04_legal/legal_handbook_excerpt.pdf` | PDF | Legal retention, whistleblower line |

With Qdrant on **localhost:6333** (and Apple Intelligence available), from `examples/rag_demo`:

```bash
# Ingest: recreate collection "apple_fm_rag_corpus" with 512-d cosine vectors
uv run rag_ingest.py

# Ingest the repo corpus plus a PDF outside the tree (each PDF is read fully, then chunked)
uv run rag_ingest.py --include ~/Downloads/posture_aware.pdf

# Ask a question (retrieves top-k chunks, then Foundation Models)
uv run rag_ask.py "Where are NDA originals stored and for how long?"
```

Useful flags: `uv run rag_ingest.py --corpus /path/to/docs --no-recreate` to append without dropping the collection; repeat `--include` for more files. **`rag_ask.py`** (by default) runs **RRF** over two query embeddings, then **reorders** chunks for the prompt so price/table text is less likely to be cut off by the context cap; use `--no-fusion` to disable, or `--per-query-limit` to widen each search. Adjust `--top-k` (default 12) with the same cap in mind.

**Inline dependencies:** `rag_ingest.py` and `rag_ask.py` declare PEP 723 dependencies (`langchain-text-splitters` is included for `chunking.py`); `uv run` will resolve them. For `generate_sample_corpus.py` only, pass `--with fpdf2` (see step 3 above) or add `fpdf2` to your environment.

## Minimal single-script demo

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

## What `rag_experiment` does
1. Three "private" facts are embedded locally using the native Apple Natural Language framework.
2. These embeddings are stored in Qdrant (collection `apple_fm_knowledge` in that script).
3. A query is made about these facts.
4. The system retrieves the relevant context and uses Apple Intelligence to generate a grounded answer.

## In-memory retrieval (no Qdrant)

For a minimal pipeline or unit tests, the SDK exposes **cosine similarity** and **top‑k** over precomputed 512-d vectors: `from apple_fm_sdk import cosine_similarity, retrieve_top_k` (see `src/apple_fm_sdk/retrieval.py`).

In this repo, run **`uv run pytest tests/test_retrieval.py`** to exercise both deterministic vector tests and a live native-embedding check that a vault-related query ranks above unrelated passages.
