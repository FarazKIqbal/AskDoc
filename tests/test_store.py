"""Tests for rag/store.py.

These tests load the real embedding model, so they are slower than the
chunker tests. They use a temporary directory so no real Chroma data is
left behind.
"""

import shutil
import tempfile
from pathlib import Path

import pytest

from rag.chunker import chunk_pages
from rag.loader import Page
from rag.store import add_chunks, query


@pytest.fixture(scope="module")
def embedder():
    """Load the embedding model once for the whole test module."""
    from rag.store import _make_embedder

    return _make_embedder()


@pytest.fixture
def temp_chroma():
    """A fresh empty directory for Chroma, cleaned up after the test."""
    path = Path(tempfile.mkdtemp())
    yield path
    shutil.rmtree(path, ignore_errors=True)


def test_add_and_query_returns_relevant_chunks(temp_chroma, embedder):
    """Storing chunks then querying should return the most similar ones."""
    pages = [
        Page(page_number=1, text="The cat sat on the mat. Cats like naps."),
        Page(page_number=2, text="Dogs are loyal animals. They enjoy walks."),
    ]
    chunks = chunk_pages(pages, chunk_words=20, overlap_words=5)
    add_chunks(chunks, temp_chroma, embedder=embedder)

    results = query("Where does the cat sit?", temp_chroma, embedder=embedder)
    assert len(results) >= 1
    # The most relevant chunk should mention the cat sitting on the mat.
    assert "cat" in results[0].text.lower()
    assert results[0].page_number == 1


def test_empty_chunks_raises(temp_chroma, embedder):
    """Storing nothing should raise rather than create an empty collection."""
    with pytest.raises(ValueError):
        add_chunks([], temp_chroma, embedder=embedder)


def test_query_before_ingest_raises(temp_chroma, embedder):
    """Querying a collection that doesn't exist should raise a clear error."""
    with pytest.raises(ValueError):
        query("anything", temp_chroma, embedder=embedder)


def test_page_number_survives_round_trip(temp_chroma, embedder):
    """The page number stored with a chunk should come back unchanged."""
    pages = [Page(page_number=7, text="A short sentence about page seven.")]
    chunks = chunk_pages(pages, chunk_words=20, overlap_words=5)
    add_chunks(chunks, temp_chroma, embedder=embedder)

    results = query("page seven", temp_chroma, embedder=embedder)
    assert results[0].page_number == 7