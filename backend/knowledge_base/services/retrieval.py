"""Retrieval service for the College Policy RAG assistant."""

from typing import Any, TypedDict

from .vector_store import search_similar_chunks


class ChunkSource(TypedDict, total=False):
    """Metadata describing the origin of a retrieved chunk."""

    document_id: str | None
    chunk_index: int | None
    page_number: int | None
    section_title: str | None


class RetrievedChunk(TypedDict):
    """Structured result for a retrieved policy chunk."""

    chunk_id: str
    content: str
    source: ChunkSource
    similarity: float
    distance: float


def retrieve_relevant_chunks(
    query_text: str,
    top_k: int = 5,
    document_id: int | str | None = None,
    min_similarity: float | None = None,
) -> list[dict[str, Any]]:
    """
    Retrieve relevant policy chunks matching the query.

    Acts as a clean abstraction over the vector store service without
    recomputing embeddings or interacting directly with ChromaDB.

    Args:
        query_text: The search query or student/staff policy question.
        top_k: Number of top chunks to retrieve (must be >= 1).
        document_id: Optional ID to restrict retrieval to a specific document.
        min_similarity: Optional similarity threshold between 0.0 and 1.0.
            Chunks with similarity lower than this value are filtered out.
            Defaults to None.

    Returns:
        List of structured chunk dictionaries with chunk_id, content, source,
        similarity, and distance.

    Raises:
        ValueError: If query_text is empty, top_k is less than 1, or
            min_similarity is not within [0.0, 1.0].
    """
    if not query_text or not isinstance(query_text, str) or not query_text.strip():
        raise ValueError("Search query cannot be empty.")

    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k must be an integer greater than or equal to 1.")

    if min_similarity is not None:
        if (
            isinstance(min_similarity, bool)
            or not isinstance(min_similarity, (int, float))
            or not (0.0 <= float(min_similarity) <= 1.0)
        ):
            raise ValueError("min_similarity must be a number between 0.0 and 1.0.")

    raw_chunks = search_similar_chunks(
        query_text=query_text.strip(),
        n_results=top_k,
        document_id=document_id,
    )

    if not raw_chunks:
        return []

    results: list[dict[str, Any]] = []

    for item in raw_chunks:
        similarity = float(item.get("similarity", 0.0))

        if min_similarity is not None and similarity < min_similarity:
            continue

        raw_metadata = item.get("metadata") or {}
        source: ChunkSource = {
            "document_id": raw_metadata.get("document_id"),
            "chunk_index": raw_metadata.get("chunk_index"),
            "page_number": raw_metadata.get("page_number"),
            "section_title": raw_metadata.get("section_title"),
        }

        results.append(
            {
                "chunk_id": item["id"],
                "content": item["content"],
                "source": source,
                "similarity": similarity,
                "distance": float(item.get("distance", 0.0)),
            }
        )

    return results
