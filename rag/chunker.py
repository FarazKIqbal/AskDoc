"""Split page text into overlapping chunks for embedding.

Each chunk carries the page number it came from, so answers can cite
"page N" later. A fixed chunk size keeps things simple and predictable,
and a small overlap prevents facts that straddle a page boundary from
being split apart.
"""

from __future__ import annotations

import re
from typing import NamedTuple

from rag.loader import Page

# A "word" here is a run of non-whitespace characters. Using a regex keeps
# this fast and dependency-free compared to a tokenizer library.
_WORD_RE = re.compile(r"\S+")

# Defaults: ~500 words per chunk with ~50 words of overlap.
DEFAULT_CHUNK_WORDS = 500
DEFAULT_OVERLAP_WORDS = 50


class Chunk(NamedTuple):
    """A piece of text ready for embedding, with its source page."""

    text: str
    page_number: int


def _word_count(text: str) -> int:
    """Count words in text using the same definition as the splitter."""
    return len(_WORD_RE.findall(text))


def chunk_pages(
    pages: list[Page],
    chunk_words: int = DEFAULT_CHUNK_WORDS,
    overlap_words: int = DEFAULT_OVERLAP_WORDS,
) -> list[Chunk]:
    """Split page text into overlapping chunks.

    Args:
        pages: Page objects from rag.loader.load_pdf.
        chunk_words: Target number of words per chunk.
        overlap_words: Number of words shared between consecutive chunks
            so context at chunk boundaries is not lost.

    Returns:
        A list of Chunk objects, each with text and the page it came from.

    Raises:
        ValueError: if chunk_words or overlap_words are invalid.
    """
    if chunk_words <= 0:
        raise ValueError("chunk_words must be positive")
    if overlap_words < 0:
        raise ValueError("overlap_words must not be negative")
    if overlap_words >= chunk_words:
        raise ValueError("overlap_words must be smaller than chunk_words")

    chunks: list[Chunk] = []
    for page in pages:
        words = _WORD_RE.findall(page.text)
        if not words:
            continue

        # Slide a window across the words. Each step moves forward by
        # (chunk_words - overlap_words), so consecutive chunks share the
        # last overlap_words words of the previous chunk.
        step = chunk_words - overlap_words
        for start in range(0, len(words), step):
            window = words[start : start + chunk_words]
            if not window:
                break
            chunks.append(Chunk(text=" ".join(window), page_number=page.page_number))

    return chunks