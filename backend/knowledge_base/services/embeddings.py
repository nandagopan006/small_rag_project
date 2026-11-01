from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"

_model = None

def get_embedding_model():
    """Load the embedding model once and reuse it."""
    global _model

    if _model is None:
        try:
            _model = SentenceTransformer(MODEL_NAME, local_files_only=True)
        except Exception:
            _model = SentenceTransformer(MODEL_NAME)

    return _model

def embed_text(text: str) -> list[float]:
    """Convert one text string into a normalized embedding vector."""
    if not text or not text.strip():
        raise ValueError("Text cannot be empty.")

    model = get_embedding_model()

    embedding = model.encode(
        text.strip(),
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    return embedding.tolist()

def embed_texts(texts: list[str]) -> list[list[float]]:
    """Convert multiple text strings into embedding vectors."""
    if not texts:
        return []

    if any(not text or not text.strip() for text in texts):
        raise ValueError("All texts must be non-empty.")

    model = get_embedding_model()

    embeddings = model.encode(
        [text.strip() for text in texts],
        batch_size=32,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    return embeddings.tolist()
