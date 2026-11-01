

from .retrieval import retrieve_relevant_chunks
from .context_builder import build_context
from .llm import generate_answer


NO_ANSWER_MESSAGE = (
    "I could not find sufficient information in the "
    "available college policies to answer your question."
)


def ask_policy_question(
    question,
    top_k=5,
    document_id=None,
    min_similarity=None,
    max_context_chars=12000,
):
    """Retrieve policy information and generate an answer."""

    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question cannot be empty.")

    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k must be an integer greater than or equal to 1.")

    if (
        isinstance(max_context_chars, bool)
        or not isinstance(max_context_chars, int)
        or max_context_chars <= 0
    ):
        raise ValueError("max_context_chars must be a positive integer.")

    question = question.strip()

    #Retrieve relevant policy chunks.
    retrieved_chunks = retrieve_relevant_chunks(
        query_text=question,
        top_k=top_k,
        document_id=document_id,
        min_similarity=min_similarity,
    )


    if not retrieved_chunks:
        return {
            "answer": NO_ANSWER_MESSAGE,
            "sources": [],
            "retrieved_chunk_count": 0,
            "included_chunk_count": 0,
            "excluded_chunk_count": 0,
        }

    #  retrieved chunks.
    context_data = build_context(
        retrieved_chunks=retrieved_chunks,
        max_context_chars=max_context_chars,
    )

    context = context_data["context"]


    if not context.strip():
        return {
            "answer": NO_ANSWER_MESSAGE,
            "sources": [],
            "retrieved_chunk_count": len(retrieved_chunks),
            "included_chunk_count": 0,
            "excluded_chunk_count": context_data[
                "excluded_chunk_count"
            ],
        }

    #Generate an answer using OpenRouter.
    answer = generate_answer(
        question=question,
        context=context,
    )

    # Return the answer and its sources.
    return {
        "answer": answer,
        "sources": context_data["sources"],
        "retrieved_chunk_count": len(retrieved_chunks),
        "included_chunk_count": context_data[
            "included_chunk_count"
        ],
        "excluded_chunk_count": context_data[
            "excluded_chunk_count"
        ],
    }
