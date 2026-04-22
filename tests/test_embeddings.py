import apple_fm_sdk as fm
import pytest

def test_get_sentence_embedding() -> None:
    text = "Apple Intelligence is powerful."
    vector = fm.get_sentence_embedding(text)
    
    assert isinstance(vector, list)
    assert len(vector) == 512
    assert all(isinstance(x, float) for x in vector)
    
    # Test that different sentences have different embeddings
    vector2 = fm.get_sentence_embedding("Something else entirely.")
    assert vector != vector2

def test_embedding_consistency() -> None:
    text = "Consistency check."
    v1 = fm.get_sentence_embedding(text)
    v2 = fm.get_sentence_embedding(text)
    assert v1 == v2
