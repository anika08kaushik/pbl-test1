from typing import List

from sentence_transformers import SentenceTransformer

# Pure semantic model — no keyword, no fuzzy, no NER/spaCy
# all-MiniLM-L6-v2 is fast, accurate, and ideal for resume-JD matching
MODEL_NAME = "all-MiniLM-L6-v2"

_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    """Lazy-load the model once and reuse."""
    global _model
    if _model is None:
        print(f"[embedder] Loading model: {MODEL_NAME}")
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Embed a list of strings into dense vectors.
    Returns list of float vectors (one per input text).
    No keyword extraction, no fuzzy matching — pure context similarity.
    """
    model = get_model()
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return embeddings.tolist()


def embed_single(text: str) -> List[float]:
    """Embed a single string."""
    return embed_texts([text])[0]


def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """
    Compute cosine similarity between two vectors.
    Returns float in [-1, 1]. For well-formed sentence embeddings this is [0, 1].
    """
    import math
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    mag_a = math.sqrt(sum(a * a for a in vec_a))
    mag_b = math.sqrt(sum(b * b for b in vec_b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)
