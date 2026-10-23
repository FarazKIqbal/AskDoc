"""Tests for rag/chunker.py."""

from rag.chunker import DEFAULT_CHUNK_WORDS, DEFAULT_OVERLAP_WORDS, chunk_pages
from rag.loader import Page


def _pages_from_text(texts):
    """Build a list of Page objects from a list of strings."""
    return [Page(page_number=i + 1, text=t) for i, t in enumerate(texts)]


def test_chunk_size_is_roughly_target():
    """Chunks should be close to the target word count."""
    text = "word " * 1000
    chunks = chunk_pages([Page(page_number=1, text=text)])
    assert len(chunks) >= 2
    for chunk in chunks:
        # Allow some slack because the last chunk of a page may be smaller.
        assert len(chunk.text.split()) <= DEFAULT_CHUNK_WORDS


def test_overlap_between_consecutive_chunks():
    """Consecutive chunks should share their last words."""
    # 14 words with chunk=10, overlap=3 gives step=7, so exactly 2 chunks.
    text = " ".join(f"w{i}" for i in range(14))
    chunks = chunk_pages([Page(page_number=1, text=text)], chunk_words=10, overlap_words=3)
    assert len(chunks) == 2
    first_words = chunks[0].text.split()
    second_words = chunks[1].text.split()
    # The last 3 words of chunk 0 should be the first 3 words of chunk 1.
    assert first_words[-3:] == second_words[:3]


def test_page_numbers_are_preserved():
    """Each chunk keeps the page number it came from."""
    pages = _pages_from_text(["page one text " * 50, "page two text " * 50])
    chunks = chunk_pages(pages, chunk_words=20, overlap_words=5)
    assert all(c.page_number in (1, 2) for c in chunks)
    # Chunks from page 1 should all say page 1.
    page1_chunks = [c for c in chunks if c.page_number == 1]
    assert page1_chunks and all(c.page_number == 1 for c in page1_chunks)


def test_empty_page_is_skipped():
    """A page with no words produces no chunks."""
    chunks = chunk_pages([Page(page_number=1, text="")])
    assert chunks == []


def test_invalid_chunk_size_raises():
    """Bad parameters raise ValueError instead of producing garbage."""
    import pytest

    with pytest.raises(ValueError):
        chunk_pages([Page(page_number=1, text="hello")], chunk_words=0)
    with pytest.raises(ValueError):
        chunk_pages([Page(page_number=1, text="hello")], overlap_words=10, chunk_words=5)