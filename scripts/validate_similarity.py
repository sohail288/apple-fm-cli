import math

import apple_fm_sdk as fm


def dot_product(v1: list[float], v2: list[float]) -> float:
    return sum(x * y for x, y in zip(v1, v2, strict=True))


def magnitude(v: list[float]) -> float:
    return math.sqrt(sum(x * x for x in v))


def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    mag1 = magnitude(v1)
    mag2 = magnitude(v2)
    if mag1 == 0 or mag2 == 0:
        return 0
    return dot_product(v1, v2) / (mag1 * mag2)


def validate_embeddings() -> None:
    sentences = [
        ("The cat sat on the mat.", "A feline rested on the rug."),  # Very similar
        ("I love playing soccer.", "Football is my favorite sport."),  # Similar
        (
            "The stock market is volatile today.",
            "I enjoy eating green apples.",
        ),  # Dissimilar
        (
            "Apple Intelligence is coming to Mac.",
            "The weather is nice in Cupertino.",
        ),  # Slightly related
    ]

    print(f"{'Sentence A':<40} | {'Sentence B':<40} | {'Similarity':<10}")
    print("-" * 95)

    for s1, s2 in sentences:
        v1 = fm.get_sentence_embedding(s1)
        v2 = fm.get_sentence_embedding(s2)
        sim = cosine_similarity(v1, v2)
        print(f"{s1:<40} | {s2:<40} | {sim:.4f}")


if __name__ == "__main__":
    validate_embeddings()
