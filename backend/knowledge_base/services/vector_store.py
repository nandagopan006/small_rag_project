
"""ChromaDB client and collection management."""

from pathlib import Path
from collections.abc import Iterable
from ..models import DocumentChunk
from .embeddings import embed_text, embed_texts

import chromadb
from django.conf import settings


COLLECTION_NAME = "college_policy_chunks"

EMBEDDING_MODEL = (
    "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
)

EMBEDDING_DIMENSION = 384

_client = None
_collection = None

CHROMA_BATCH_SIZE = 100


def get_chroma_client():
    """Initialize and reuse the persistent ChromaDB client."""

    global _client

    if _client is None:
        persist_directory = Path(settings.BASE_DIR) / "chroma_data"

        _client = chromadb.PersistentClient(
            path=str(persist_directory)
        )

    return _client

#creeate the collection
def get_policy_collection():
    """Get or create the college policy vector collection."""

    global _collection

    if _collection is None:
        client = get_chroma_client()

        _collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            configuration={
                "hnsw": {
                    "space": "cosine",
                }
            },
            metadata={
                "description": "College policy document chunks",
                "embedding_model": EMBEDDING_MODEL,
                "embedding_dimension": EMBEDDING_DIMENSION,
            },
        )

    return _collection


def generate_chunk_id(chunk: DocumentChunk) -> str:
    """Generate a stable ID for a document chunk."""

    return (
        f"document-{chunk.document_id}"
        f"-chunk-{chunk.chunk_index}"
    )

#adding to the vector db with id,vectors,text,metadata
def upsert_chunks(chunks: Iterable[DocumentChunk]) -> int:
    """Generate and store embeddings for Django document chunks."""

    chunk_list = list(chunks)

    if not chunk_list:
        return 0

    collection = get_policy_collection()

    for start in range(0, len(chunk_list), CHROMA_BATCH_SIZE):
        batch = chunk_list[start:start + CHROMA_BATCH_SIZE]

        documents = [
            chunk.content.strip()
            for chunk in batch
        ]

        if any(not content for content in documents):
            raise ValueError("Chunk content cannot be empty.")

        embeddings = embed_texts(documents)

        ids = [
            generate_chunk_id(chunk)
            for chunk in batch
        ]

        metadatas = [
            {
                "document_id": str(chunk.document_id),
                "chunk_index": chunk.chunk_index,
                "page_number": chunk.page_number,
                "section_title": chunk.section_title,
            }
            for chunk in batch
        ]

        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    return len(chunk_list)

def delete_document_vectors(document_id: int | str) -> None:
    """Delete all vectors belonging to a policy document."""

    collection = get_policy_collection()

    collection.delete(
        where={
            "document_id": str(document_id)
        }
    )

def sync_document_chunks(document_id: int | str) -> int:
    """Synchronize one document's chunks with ChromaDB."""

    collection = get_policy_collection()

    chunks = list(
        DocumentChunk.objects.filter(
            document_id=document_id
        ).order_by("chunk_index")
    )

    existing_records = collection.get(
        where={
            "document_id": str(document_id)
        },
        include=["metadatas"],
    )

    existing_ids = set(existing_records["ids"])

    if chunks:
        upsert_chunks(chunks)

    expected_ids = {
        generate_chunk_id(chunk)
        for chunk in chunks
    }

    stale_ids = existing_ids - expected_ids

    if stale_ids:
        collection.delete(ids=list(stale_ids))

    return len(chunks)

def search_similar_chunks(
    query_text: str,
    n_results: int = 5,
    document_id: int | str | None = None,
) -> list[dict]:
    """Retrieve the most relevant policy chunks for a question."""

    if not query_text or not query_text.strip():
        raise ValueError("Search query cannot be empty.")

    if n_results < 1:
        raise ValueError("n_results must be at least 1.")

    collection = get_policy_collection()

    collection_count = collection.count()

    if collection_count == 0:
        return []

    query_embedding = embed_text(query_text)

    query_params = {
        "query_embeddings": [query_embedding],
        "n_results": min(n_results, collection_count),
        "include": [
            "documents",
            "metadatas",
            "distances",
        ],
    }

    if document_id is not None:
        query_params["where"] = {
            "document_id": str(document_id)
        }

    results = collection.query(**query_params)

    retrieved_chunks = []

    for chunk_id, content, metadata, distance in zip(
        results["ids"][0],
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        retrieved_chunks.append({
            "id": chunk_id,
            "content": content,
            "metadata": metadata,
            "distance": float(distance),
            "similarity": 1.0 - float(distance),
        })

    return retrieved_chunks