"""Tests for in-memory vector retrieval and optional native embedding pipelines."""

from __future__ import annotations

import math

import pytest

from apple_fm_sdk.retrieval import cosine_similarity, retrieve_top_k

ONE = [0.0] * 40 + [1.0] + [0.0] * 471
TWO = [0.0] * 41 + [1.0] + [0.0] * 470
THREE = [0.0] * 42 + [1.0] + [0.0] * 469


def test_cosine_identical() -> None:
    v = [0.1, 0.2, -0.3]
    assert cosine_similarity(v, v) == pytest.approx(1.0)


def test_cosine_opposite() -> None:
    a = [1.0, 0.0]
    b = [-1.0, 0.0]
    assert cosine_similarity(a, b) == pytest.approx(-1.0)


def test_cosine_orthogonal_512d() -> None:
    assert cosine_similarity(ONE, TWO) == pytest.approx(0.0, abs=1e-9)


def test_cosine_length_mismatch() -> None:
    with pytest.raises(ValueError, match="same length"):
        cosine_similarity([1.0, 0.0], [1.0])


def test_retrieve_top_k_orthogonal() -> None:
    corpus: list[tuple[str, list[float]]] = [
        ("a", list(ONE)),
        ("b", list(TWO)),
        ("c", list(THREE)),
    ]
    out = retrieve_top_k(list(ONE), corpus, k=2)
    assert [x[0] for x in out] == ["a", "b"]
    assert out[0][1] == pytest.approx(1.0)
    assert out[1][1] == pytest.approx(0.0, abs=1e-9)


def test_retrieve_top_k_k_zero() -> None:
    out = retrieve_top_k(ONE, [("a", ONE)], k=0)
    assert out == []


def test_retrieve_top_k_tiny_corpus() -> None:
    out = retrieve_top_k(ONE, [("only", list(ONE))], k=5)
    assert len(out) == 1
    assert out[0][0] == "only"
    assert out[0][1] == pytest.approx(1.0)


@pytest.fixture
def require_sentence_embeddings() -> None:
    import apple_fm_sdk as fm

    try:
        v = fm.get_sentence_embedding("bootstrap")
    except (RuntimeError, OSError) as e:
        pytest.skip(f"native embeddings not available: {e}")
    if len(v) != 512:
        pytest.fail(f"unexpected embedding length {len(v)}")


@pytest.mark.usefixtures("require_sentence_embeddings")
def test_retrieval_ranks_relevant_doc_first() -> None:
    """
    With real NLEnglish 512-d embeddings, a query about the "vault code" should
    score the vault-related passage above clearly unrelated text.
    """
    import apple_fm_sdk as fm

    passages = {
        "vault": "The secret vault access codeword is RAG-512 for emergencies and audits.",
        "pizza": "Chicago is known for deep dish pizza and long winters near the lake.",
        "ocean": "Coral reefs host diverse species in warm shallow marine waters worldwide.",
    }
    query = "What is the secret codeword for vault access?"

    corpus: list[tuple[str, list[float]]] = [
        (name, fm.get_sentence_embedding(text)) for name, text in passages.items()
    ]
    qv = fm.get_sentence_embedding(query)
    ranked = retrieve_top_k(qv, corpus, k=3)
    by_id = {doc_id: score for doc_id, score in ranked}

    assert ranked[0][0] == "vault", f"expected vault first, got: {ranked}"
    assert by_id["vault"] > by_id["pizza"] + 0.01
    assert by_id["vault"] > by_id["ocean"] + 0.01
    for _, score in ranked:
        assert -1.01 <= score <= 1.01
        assert not math.isnan(score)
