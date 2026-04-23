# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.

"""In-memory vector retrieval utilities (e.g. for RAG without a vector database)."""

from __future__ import annotations

import math
from collections.abc import Sequence


def _l2_norm(vector: Sequence[float]) -> float:
    return math.sqrt(sum(x * x for x in vector))


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity in [-1, 1] for two vectors of equal length (e.g. sentence embeddings)."""
    if len(a) != len(b):
        raise ValueError("vectors must have the same length")
    dot = 0.0
    for x, y in zip(a, b, strict=True):
        dot += x * y
    na = _l2_norm(a)
    nb = _l2_norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def retrieve_top_k(
    query: Sequence[float],
    corpus: list[tuple[str, Sequence[float]]],
    k: int,
) -> list[tuple[str, float]]:
    """
    Return up to ``k`` document ids with highest cosine similarity to ``query``,
    sorted by score descending.
    """
    if k < 1:
        return []
    scored: list[tuple[str, float]] = [
        (doc_id, cosine_similarity(query, vector)) for doc_id, vector in corpus
    ]
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[:k]
