"""Context Builder service for the College Policy RAG assistant.

Prepares retrieved policy document chunks into a structured, source-aware
context string suitable for LLM answer generation, strictly adhering to a
character-based context budget.

Security Note:
    All retrieved chunk text is treated as untrusted reference data. Retained
    content is clearly demarcated with strict context boundaries and separate
    source headers. This module never interprets, executes, or mixes retrieved
    text with system instructions. Delimiters assist in structural parsing but
    do not replace robust system-prompt isolation at the generation layer.
"""

from typing import Any, TypedDict

CONTEXT_START_DELIMITER = "[BEGIN RETRIEVED POLICY CONTEXT]"
CONTEXT_END_DELIMITER = "[END RETRIEVED POLICY CONTEXT]"
SOURCE_BLOCK_SEPARATOR = "\n\n\n"
DEFAULT_MAX_CONTEXT_CHARS = 12000


class ContextSource(TypedDict, total=False):
    """Metadata describing a single policy chunk cited in the context."""

    source_label: str
    chunk_id: str | None
    document_id: str | None
    page_number: int | None
    section_title: str | None
    similarity: float | None


class BuiltContext(TypedDict):
    """Output structure returned by the context builder service."""

    context: str
    sources: list[ContextSource]
    included_chunk_count: int
    excluded_chunk_count: int


def _extract_chunk_content(chunk: dict[str, Any]) -> str | None:
    """Extract and validate the text content of a chunk."""
    content = chunk.get("content")
    if not isinstance(content, str) or not content.strip():
        return None
    return content


def _extract_chunk_id(chunk: dict[str, Any]) -> str | None:
    """Extract and validate the stable chunk ID.

    Returns a stripped string if valid. Missing, non-string, or whitespace-only
    IDs return None so that unrelated chunks lacking valid IDs are not
    mistakenly deduplicated against each other.
    """
    raw_id = chunk.get("chunk_id")
    if raw_id is None:
        raw_id = chunk.get("id")

    if isinstance(raw_id, str) and raw_id.strip():
        return raw_id.strip()
    return None


def _extract_metadata(
    chunk: dict[str, Any],
) -> tuple[str | None, int | None, str | None, float | None]:
    """Safely extract optional source metadata fields.

    Extracts (document_id, page_number, section_title, similarity) from either
    the nested 'source' / 'metadata' dict or directly from the chunk record.
    Returns None for any missing, malformed, or unparseable field.
    """
    raw_source = chunk.get("source")
    if not isinstance(raw_source, dict):
        raw_source = chunk.get("metadata")
    if not isinstance(raw_source, dict):
        raw_source = {}

    # document_id
    doc_id = raw_source.get("document_id")
    if doc_id is None:
        doc_id = chunk.get("document_id")
    valid_doc_id = str(doc_id).strip() if doc_id is not None and str(doc_id).strip() else None

    # page_number
    page = raw_source.get("page_number")
    if page is None:
        page = raw_source.get("page")
    if page is None:
        page = chunk.get("page_number")

    valid_page: int | None = None
    if page is not None and not isinstance(page, bool):
        if isinstance(page, int):
            valid_page = page
        elif isinstance(page, str) and page.strip().isdigit():
            valid_page = int(page.strip())

    # section_title
    section = raw_source.get("section_title")
    if section is None:
        section = chunk.get("section_title")
    valid_section = section.strip() if isinstance(section, str) and section.strip() else None

    # similarity
    sim = chunk.get("similarity")
    if sim is None:
        sim = raw_source.get("similarity")

    valid_similarity: float | None = None
    if sim is not None and not isinstance(sim, bool) and isinstance(sim, (int, float)):
        valid_similarity = float(sim)

    return valid_doc_id, valid_page, valid_section, valid_similarity


def _format_source_block(
    source_label: str,
    content: str,
    chunk_id: str | None,
    document_id: str | None,
    page_number: int | None,
    section_title: str | None,
    similarity: float | None,
) -> str:
    """Format a single chunk into a delimited source block with available metadata."""
    lines = [f"[{source_label}]"]

    if chunk_id is not None:
        lines.append(f"Chunk ID: {chunk_id}")
    if document_id is not None:
        lines.append(f"Document ID: {document_id}")
    if page_number is not None:
        lines.append(f"Page: {page_number}")
    if section_title is not None:
        lines.append(f"Section: {section_title}")
    if similarity is not None:
        lines.append(f"Similarity: {similarity:.2f}")

    header = "\n".join(lines)
    return f"{header}\n\n{content}"


def build_context(
    retrieved_chunks: list[dict[str, Any]],
    max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
) -> BuiltContext:
    """Build structured, source-aware context for an LLM.

    Processes raw or structured chunks returned by the retrieval layer:
    validates input, deduplicates chunks by stable chunk ID, formats each
    into a sequential source block, and strictly enforces a character budget.

    Args:
        retrieved_chunks: List of chunk dictionaries from the retrieval service.
        max_context_chars: Maximum allowed characters in the final context string.
            Must be a positive integer greater than zero. Defaults to 12000.

    Returns:
        BuiltContext containing:
            - context: Formatted context string enclosed within delimiters, or
              an empty string if no chunks fit or the input is empty.
            - sources: Traceable list of source metadata for included chunks.
            - included_chunk_count: Number of unique chunks included in context.
            - excluded_chunk_count: Number of supplied records excluded (duplicates,
              malformed, empty-content, or over-budget).

    Raises:
        TypeError: If retrieved_chunks is not a list, or max_context_chars is not an int.
        ValueError: If max_context_chars <= 0.
    """
    if not isinstance(retrieved_chunks, list):
        raise TypeError(
            f"retrieved_chunks must be a list of dictionaries, got {type(retrieved_chunks).__name__}."
        )

    if isinstance(max_context_chars, bool) or not isinstance(max_context_chars, int):
        raise TypeError(
            f"max_context_chars must be an integer, got {type(max_context_chars).__name__}."
        )

    if max_context_chars <= 0:
        raise ValueError(
            f"max_context_chars must be a positive integer greater than zero, got {max_context_chars}."
        )

    if not retrieved_chunks:
        return {
            "context": "",
            "sources": [],
            "included_chunk_count": 0,
            "excluded_chunk_count": 0,
        }

    seen_chunk_ids: set[str] = set()
    included_blocks: list[str] = []
    included_sources: list[ContextSource] = []
    excluded_chunk_count = 0
    current_context_length = 0

    delimiters_overhead = (
        len(CONTEXT_START_DELIMITER)
        + len("\n\n")
        + len("\n\n")
        + len(CONTEXT_END_DELIMITER)
    )

    for item in retrieved_chunks:
        if not isinstance(item, dict):
            excluded_chunk_count += 1
            continue

        content = _extract_chunk_content(item)
        if content is None:
            excluded_chunk_count += 1
            continue

        chunk_id = _extract_chunk_id(item)
        if chunk_id is not None:
            if chunk_id in seen_chunk_ids:
                excluded_chunk_count += 1
                continue
            seen_chunk_ids.add(chunk_id)

        doc_id, page, section, similarity = _extract_metadata(item)

        next_source_index = len(included_sources) + 1
        source_label = f"Source {next_source_index}"

        candidate_block = _format_source_block(
            source_label=source_label,
            content=content,
            chunk_id=chunk_id,
            document_id=doc_id,
            page_number=page,
            section_title=section,
            similarity=similarity,
        )

        if not included_blocks:
            projected_length = delimiters_overhead + len(candidate_block)
        else:
            projected_length = (
                current_context_length
                + len(SOURCE_BLOCK_SEPARATOR)
                + len(candidate_block)
            )

        if projected_length <= max_context_chars:
            included_blocks.append(candidate_block)
            current_context_length = projected_length
            included_sources.append(
                {
                    "source_label": source_label,
                    "chunk_id": chunk_id,
                    "document_id": doc_id,
                    "page_number": page,
                    "section_title": section,
                    "similarity": round(similarity, 4) if similarity is not None else None,
                }
            )
        else:
            # Over-budget: cannot fit complete block. Exclude and continue checking later chunks.
            excluded_chunk_count += 1

    if not included_blocks:
        formatted_context = ""
    else:
        formatted_context = (
            f"{CONTEXT_START_DELIMITER}\n\n"
            f"{SOURCE_BLOCK_SEPARATOR.join(included_blocks)}"
            f"\n\n{CONTEXT_END_DELIMITER}"
        )

    return {
        "context": formatted_context,
        "sources": included_sources,
        "included_chunk_count": len(included_sources),
        "excluded_chunk_count": excluded_chunk_count,
    }
