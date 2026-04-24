#!/usr/bin/env python3
# /// script
# requires-python = ">=3.14"
# dependencies = [
#   "qdrant-client",
#   "httpx",
#   "apple-fm-cli",
# ]
# ///
"""
RAG over Qdrant: embed the question, retrieve top-k chunks, answer with a local
Foundation Models session. Requires prior `python rag_ingest.py` with Qdrant up
(`docker compose up -d`).

Retrieval uses **reciprocal rank fusion (RRF)** over two vector queries: the
original question plus a short *expansion* phrasing that emphasizes concrete
details (prices, tiers, names, numbers). That reduces misses when a single
embedding ranks the right chunk just below ``top_k`` (e.g. fact tables vs prose).
A **total context budget** caps combined prompt size so the session does not
exceed the model window.
"""

from __future__ import annotations

import argparse
import asyncio
import re
from collections import defaultdict

from qdrant_client import QdrantClient

import apple_fm_sdk as fm

# Hierarchical payloads can include long parent_scope; cap per hit so the session fits.
_MAX_PARENT_CHARS = 320
_MAX_PASSAGE_CHARS = 850
# Cap retrieved context (approx chars) so Foundation Models window is not exceeded.
_MAX_CONTEXT_SECTION_CHARS = 10_500

# RRF: merge ranks from two query embeddings (original + expansion). Standard k=60.
_RRF_K = 60.0
# Heuristic string so the second query vector aligns better with passages that list
# prices, product names, tiers, etc. (helps table-like chunks surface).
_RETRIEVAL_EXPANSION = (
    " Key concrete details: specific prices, subscription tiers, plan names, "
    "dollar amounts, per-user or per-month costs, product and company names, dates, "
    "and bullet lists of features."
)

_FINANCIAL_Q_HINTS = frozenset(
    ("price", "pricing", "cost", "tier", "plan", "subscription", "dollar", "pay", "fee", "how much")
)
_NUMERIC_Q_HINTS = frozenset(
    ("how many", "number", "list three", "three ", "all ", "each ", "per user", "per month")
)


def _retrieval_text_boost(question: str, text: str) -> int:
    """Deprioritize ordering only for building the prompt: surface numeric/tier lines early."""
    q = question.lower()
    t = (text or "").lower()
    b = 0
    if any(h in q for h in _FINANCIAL_Q_HINTS):
        if "$" in t:
            b += 5
        if "per user" in t or "per month" in t or "/month" in t:
            b += 2
        if any(s in t for s in ("starter", "pro", "enterprise", "tier", "b2b saas", "b2b")):
            b += 2
    if any(h in q for h in _NUMERIC_Q_HINTS) and re.search(r"\$|\d{2,4}\s*(usd|/mo|month)?", t):
        b += 2
    return b


def reciprocal_rank_fusion(ranked_lists: list[list], *, k: float = _RRF_K) -> list:
    """Merge ordered hit lists by RRF so two query vectors share credit per point id."""
    scores: defaultdict[object, float] = defaultdict(float)
    by_id: dict[object, object] = {}
    for ranked in ranked_lists:
        for rank, pt in enumerate(ranked, 1):
            pid = pt.id
            scores[pid] += 1.0 / (k + rank)
            by_id[pid] = pt
    ordered_ids = sorted(scores.keys(), key=lambda x: -scores[x])
    return [by_id[i] for i in ordered_ids]


def reorder_hits_for_prompt(question: str, hits: list) -> list:
    """Stabilize RRF order with query-aware score so price/table chunks are not cut off
    from the final prompt when a character budget applies."""
    if not hits:
        return hits

    with_scores: list[tuple[tuple[int, int], object]] = []
    for idx, h in enumerate(hits):
        text = (getattr(h, "payload", None) or {}).get("text", "")
        boost = _retrieval_text_boost(question, str(text))
        # Sort key: higher boost first; preserve RRF order on ties.
        with_scores.append(((-boost, idx), h))
    with_scores.sort(key=lambda p: p[0])
    return [h for _, h in with_scores]


def retrieve_hits(
    client: QdrantClient,
    collection: str,
    question: str,
    *,
    top_k: int,
    use_fusion: bool,
    per_query_limit: int,
) -> list:
    if not use_fusion:
        qv = fm.get_sentence_embedding(question)
        return client.query_points(
            collection_name=collection, query=qv, limit=top_k
        ).points
    q1 = question.strip()
    q2 = (question + _RETRIEVAL_EXPANSION).strip()
    h1 = client.query_points(
        collection_name=collection,
        query=fm.get_sentence_embedding(q1),
        limit=per_query_limit,
    ).points
    h2 = client.query_points(
        collection_name=collection,
        query=fm.get_sentence_embedding(q2),
        limit=per_query_limit,
    ).points
    if not h1 and not h2:
        return []
    if not h1:
        return h2[:top_k]
    if not h2:
        return h1[:top_k]
    merged = reciprocal_rank_fusion([h1, h2])
    return merged[:top_k]


def _format_hit_block(i: int, h: object) -> str:
    p = getattr(h, "payload", None) or {}
    src = p.get("source", "?")
    hpath = p.get("hierarchy_path", "")
    text = (p.get("text", "") or "").strip()
    if len(text) > _MAX_PASSAGE_CHARS:
        text = text[: _MAX_PASSAGE_CHARS - 1] + "…"
    parent = (p.get("parent_scope", "") or "").strip()
    if len(parent) > _MAX_PARENT_CHARS:
        parent = parent[: _MAX_PARENT_CHARS - 1] + "…"
    loc = f"\nPosition: {hpath}" if hpath else ""
    par = f"\nSection/page context:\n{parent}\n" if parent else ""
    return f"[{i}] ({src}){loc}{par}\nPassage:\n{text}"


def build_context_sections(hits: list) -> str:
    """Include ordered hits until approx char budget (best-first from retrieval)."""
    parts: list[str] = []
    total = 0
    for i, h in enumerate(hits, 1):
        block = _format_hit_block(i, h)
        sep = 0 if not parts else len("\n\n---\n\n")
        if total + sep + len(block) > _MAX_CONTEXT_SECTION_CHARS and parts:
            break
        if parts:
            total += sep
        parts.append(block)
        total += len(block)
    return "\n\n---\n\n".join(parts)


async def run_query(
    question: str,
    *,
    host: str,
    port: int,
    collection: str,
    top_k: int,
    use_fusion: bool = True,
    per_query_limit: int = 48,
) -> str:
    model = fm.SystemLanguageModel()
    ok, reason = model.is_available()
    if not ok:
        raise RuntimeError(f"Foundation Models unavailable: {reason}")

    client = QdrantClient(host=host, port=port)
    hits = retrieve_hits(
        client,
        collection,
        question,
        top_k=top_k,
        use_fusion=use_fusion,
        per_query_limit=per_query_limit,
    )

    if not hits:
        return "No retrieval hits — run rag_ingest.py first with the same Qdrant settings."

    hits = reorder_hits_for_prompt(question, list(hits))
    context = build_context_sections(hits)
    prompt = f"""You are a precise internal assistant. Use ONLY the context below; if the answer
is not in the context, say you do not have enough information. Cite which source file(s) you used.

Context:
{context}

Question: {question}
Answer:"""

    session = fm.LanguageModelSession(
        instructions="Be concise. Prefer bullet points for multi-part answers."
    )
    return str(await session.respond(prompt))


def main() -> None:
    ap = argparse.ArgumentParser(description="RAG Q&A against ingested Qdrant collection")
    ap.add_argument(
        "query",
        nargs="?",
        default=(
            "What is the emergency support line, where must NDAs be stored physically, "
            "and what is the hotline for ethics concerns?"
        ),
        help="Question to ask (default: a multi-hop demo question)",
    )
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=6333)
    ap.add_argument("--collection", default="apple_fm_rag_corpus")
    ap.add_argument(
        "--top-k",
        type=int,
        default=12,
        help="Final number of chunks after RRF (trimmed to context budget)",
    )
    ap.add_argument(
        "--no-fusion",
        action="store_true",
        help="Use a single query embedding (no RRF; older behavior)",
    )
    ap.add_argument(
        "--per-query-limit",
        type=int,
        default=48,
        metavar="N",
        help="How many points each of the two vector searches considers before RRF (fusion only)",
    )
    args = ap.parse_args()

    answer = asyncio.run(
        run_query(
            args.query,
            host=args.host,
            port=args.port,
            collection=args.collection,
            top_k=args.top_k,
            use_fusion=not args.no_fusion,
            per_query_limit=args.per_query_limit,
        )
    )
    print(answer)


if __name__ == "__main__":
    main()
