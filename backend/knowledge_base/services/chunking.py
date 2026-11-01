"""Structure-aware recursive text chunking utilities."""
import re
from collections.abc import Sequence 
from typing import TypedDict

class ExtractedPage(TypedDict):
    """Represents one extracted PDF page."""
    page:int
    text:str
    
class TextChunk(TypedDict):
    """Represents a chunk with its source information."""

    chunk_index: int
    page: int
    section_title: str
    text: str
    
# Initial character-based configuration.
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 150

SEPARATORS = (
    "\n\n",
    "\n",
    ". ",
    "; ",
    " ",
    "",
)

NUMBERED_HEADING_PATTERN = re.compile(
    r"^(?:(?:section|chapter)\s+\d+|\d+(?:\.\d+)*[.)]?)\s+.{2,100}$",
    re.IGNORECASE,
)

POLICY_ID_PATTERN = re.compile(
    r"^[A-Z]{2,8}-\d{1,4}\s*[:\-–—]?\s+.{2,100}$"
)

SMALL_TITLE_WORDS = {
    "a", "an", "and", "as", "at", "by", "for",
    "in", "of", "on", "or", "the", "to", "with",
}

def _is_heading(line: str) -> bool:
    """Identify likely section headings in extracted PDF text."""

    line = line.strip()

    if not line or len(line) > 100:
        return False

    if NUMBERED_HEADING_PATTERN.match(line):
        return True

    if POLICY_ID_PATTERN.match(line):
        return True
    
    # Detect short, uppercase headings.
    if (
        any(char.isalpha() for char in line)
        and line.isupper()
        and len(line.split()) <= 12
        and not line.endswith((".", "?", "!"))
    ):
        return True
    
    # Detect likely title-case headings.
    words = line.split()
    
    if (
        2 <= len(words) <= 8
        and not line.endswith((".", "?", "!", ",", ";"))
    ):
        return all(
            word[0].isupper() or word.lower() in SMALL_TITLE_WORDS
            for word in words
            if word
        )

    return False

def _extract_sections(text: str) -> list[dict[str, str]]:
    """
    Divide a page into sections using detected headings.

    Text before the first recognized heading is assigned
    to the General section.
    """

    sections = []
    current_title = "General"
    current_lines = []

    def save_section():
        section_text = "\n".join(current_lines).strip()

        if section_text:
            sections.append(
                {
                    "section_title": current_title,
                    "text": section_text,
                }
            )

    for line in text.splitlines():
        stripped_line = line.strip()

        if stripped_line and _is_heading(stripped_line):
            save_section()

            current_title = stripped_line
            current_lines = []
        else:
            current_lines.append(line)

    save_section()

    return sections

def _split_preserving_separator(
    text: str,
    separator: str,
) -> list[str]:
    """Split text while retaining separators at the end of each part."""

    parts = text.split(separator)

    result = [
        part + separator
        for part in parts[:-1]
    ]

    result.append(parts[-1])

    return [part for part in result if part]


def _recursive_split(
    text: str,
    separators: Sequence[str],
    chunk_size: int,
) -> list[str]:
    """
    Recursively split text using progressively smaller boundaries.

    This function creates smaller text units. Overlap is added
    separately by the merging function.
    """

    if len(text) <= chunk_size:
        return [text]

    if not separators:
        return [
            text[index:index + chunk_size]
            for index in range(0, len(text), chunk_size)
        ]

    for index, separator in enumerate(separators):
        if separator == "":
            return [
                text[start:start + chunk_size]
                for start in range(0, len(text), chunk_size)
            ]

        if separator in text:
            parts = _split_preserving_separator(text, separator)
            smaller_parts = []

            for part in parts:
                if len(part) > chunk_size:
                    smaller_parts.extend(
                        _recursive_split(
                            part,
                            separators[index + 1:],
                            chunk_size,
                        )
                    )
                else:
                    smaller_parts.append(part)

            return smaller_parts

    return [text]


def _get_overlap(text: str, overlap_size: int) -> str:
    """Get trailing overlap without intentionally starting mid-word."""

    if overlap_size <= 0:
        return ""

    overlap_text = text[-overlap_size:]

    boundary = next(
        (
            index
            for index, character in enumerate(overlap_text)
            if character.isspace()
        ),
        None,
    )

    if boundary is not None:
        overlap_text = overlap_text[boundary + 1:]

    return overlap_text


def _merge_chunks(
    parts: list[str],
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    """Combine smaller parts into chunks with overlapping context."""

    chunks = []
    current_chunk = ""

    for part in parts:
        if len(current_chunk) + len(part) <= chunk_size:
            current_chunk += part
            continue

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        available_overlap = max(0, chunk_size - len(part))

        overlap = _get_overlap(
            current_chunk,
            min(chunk_overlap, available_overlap),
        )

        current_chunk = overlap + part

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


def recursive_split_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    """Split one section into overlapping recursive text chunks."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative.")

    if chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be smaller than chunk_size."
        )

    if not text or not text.strip():
        return []

    parts = _recursive_split(
        text=text.strip(),
        separators=SEPARATORS,
        chunk_size=chunk_size,
    )

    return _merge_chunks(
        parts=parts,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


def chunk_extracted_pages(
    pages: Sequence[ExtractedPage],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[TextChunk]:
    """
    Generate structure-aware chunks from extracted PDF pages.

    Preserves the original page number and detected section heading.
    Chunks do not cross page or section boundaries.
    """

    chunks: list[TextChunk] = []

    for page_data in pages:
        sections = _extract_sections(page_data["text"])

        for section in sections:
            section_title = section["section_title"]

            section_chunks = recursive_split_text(
                text=section["text"],
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )

            for chunk_text in section_chunks:
                # Include the heading in the text so its context
                # remains available during embedding and retrieval.
                contextual_text = (
                    f"{section_title}\n\n{chunk_text}"
                    if section_title != "General"
                    else chunk_text
                )

                chunks.append(
                    {
                        "chunk_index": len(chunks),
                        "page": page_data["page"],
                        "section_title": section_title,
                        "text": contextual_text,
                    }
                )

    return chunks
