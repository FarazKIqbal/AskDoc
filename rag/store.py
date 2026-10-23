"""Embed chunks with sentence-transformers and store them in ChromaDB.

ChromaDB is a local, persistent vector database. We keep one collection
per document so a future version could support multiple PDFs; for now the
collection name is fixed so the pipeline is simple to reason about.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Sequence

import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer

from rag.chunker import Chunk

logger = logging.getLogger(__name__)

# A small, fast, free embedding model that runs on CPU.
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
# One collection per project. Overridable via env var so a second document
# can live in its own collection without touching code.
COLLECTION_NAME = os.environ.get("RAG_COLLECTION_NAME", "rag-doc-qa")
# How many chunks to return for a question.
TOP_K = 4


def _make_client(persist_dir: Path) -> chromadb.Client:
    """Create a Chroma client that stores its data on disk.

    PersistentClient is the modern Chroma API: it writes the database
    files to disk and reuses them on the next run, so re-uploading a PDF
    is instant instead of rebuilding the index from scratch.
    """
    return chromadb.PersistentClient(path=str(persist_dir))


def _make_embedder() -> SentenceTransformer:
    """Load the sentence-transformer model (downloads once, then cached)."""
    logger.info("Loading embedding model: %s", EMBEDDING_MODEL)
    return SentenceTransformer(EMBEDDING_MODEL)


def add_chunks(
    chunks: Sequence[Chunk],
    persist_dir: Path,
    collection_name: str = COLLECTION_NAME,
    embedder: SentenceTransformer | None = None,
) -> int:
    """Embed chunks and store them in Chroma, replacing any existing data.

    Args:
        chunks: Chunk objects from rag.chunker.chunk_pages.
        persist_dir: Directory where Chroma stores its database files.
        collection_name: Name of the Chroma collection to use.
        embedder: Optional pre-loaded SentenceTransformer (helps tests).

    Returns:
        The number of chunks stored.

    Raises:
        ValueError: if chunks is empty.
    """
    if not chunks:
        raise ValueError("Cannot store an empty list of chunks")

    persist_dir = Path(persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)

    client = _make_client(persist_dir)
    # Delete the old collection so re-ingestion starts fresh instead of
    # accumulating duplicate vectors.
    try:
        client.delete_collection(name=collection_name)
        logger.info("Deleted existing collection %s", collection_name)
    except Exception:
        pass  # Collection doesn't exist yet — that's fine.

    collection = client.get_or_create_collection(name=collection_name)

    own_embedder = embedder is None
    if own_embedder:
        embedder = _make_embedder()

    texts = [chunk.text for chunk in chunks]
    ids = [f"chunk-{i}" for i in range(len(chunks))]
    metadatas = [{"page_number": chunk.page_number} for chunk in chunks]

    embeddings = embedder.encode(texts, show_progress_bar=False).tolist()

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )

    logger.info("Stored %d chunks in collection %s", len(chunks), collection_name)
    return len(chunks)


def query(
    question: str,
    persist_dir: Path,
    collection_name: str = COLLECTION_NAME,
    top_k: int = TOP_K,
    embedder: SentenceTransformer | None = None,
) -> list[Chunk]:
    """Return the top_k chunks most similar to the question.

    Args:
        question: The user's question.
        persist_dir: Directory where Chroma stores its database files.
        collection_name: Name of the Chroma collection to query.
        top_k: Number of chunks to return.
        embedder: Optional pre-loaded SentenceTransformer (helps tests).

    Returns:
        A list of Chunk objects, most similar first.

    Raises:
        ValueError: if the collection is empty or doesn't exist.
    """
    persist_dir = Path(persist_dir)
    client = _make_client(persist_dir)

    try:
        collection = client.get_collection(name=collection_name)
    except Exception as exc:
        raise ValueError(
            f"Collection {collection_name!r} not found in {persist_dir}. "
            "Run ingest() first."
        ) from exc

    count = collection.count()
    if count == 0:
        raise ValueError(f"Collection {collection_name!r} is empty.")

    own_embedder = embedder is None
    if own_embedder:
        embedder = _make_embedder()

    query_embedding = embedder.encode([question], show_progress_bar=False).tolist()[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, count),
        include=["documents", "metadatas", "distances"],
    )

    # Chroma returns cosine *distance* (0 = identical). Convert to a
    # similarity score in [0, 1] so callers can threshold on it.
    chunks: list[Chunk] = []
    for doc, meta, distance in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        chunks.append(
            Chunk(
                text=doc,
                page_number=int(meta["page_number"]),
            )
        )

    logger.info("Retrieved %d chunks for question", len(chunks))
    return chunks


def query_with_scores(
    question: str,
    persist_dir: Path,
    collection_name: str = COLLECTION_NAME,
    top_k: int = TOP_K,
    embedder: SentenceTransformer | None = None,
) -> list[tuple[Chunk, float]]:
    """Like query(), but also returns each chunk's similarity score.

    The score is cosine similarity between the question and the chunk
    (1.0 = identical). Callers use it to decide whether the retrieved
    chunks are actually relevant, which is how we avoid answering
    questions the document doesn't cover.

    Args:
        question: The user's question.
        persist_dir: Directory where Chroma stores its database files.
        collection_name: Name of the Chroma collection to query.
        top_k: Number of chunks to return.
        embedder: Optional pre-loaded SentenceTransformer (helps tests).

    Returns:
        A list of (Chunk, similarity) tuples, most similar first.
    """
    persist_dir = Path(persist_dir)
    client = _make_client(persist_dir)

    try:
        collection = client.get_collection(name=collection_name)
    except Exception as exc:
        raise ValueError(
            f"Collection {collection_name!r} not found in {persist_dir}. "
            "Run ingest() first."
        ) from exc

    count = collection.count()
    if count == 0:
        raise ValueError(f"Collection {collection_name!r} is empty.")

    own_embedder = embedder is None
    if own_embedder:
        embedder = _make_embedder()

    query_embedding = embedder.encode([question], show_progress_bar=False).tolist()[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, count),
        include=["documents", "metadatas", "distances"],
    )

    scored: list[tuple[Chunk, float]] = []
    for doc, meta, distance in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        similarity = 1.0 - float(distance) / 2.0
        scored.append(
            (
                Chunk(text=doc, page_number=int(meta["page_number"])),
                similarity,
            )
        )

    logger.info(
        "Retrieved %d chunk(s) for question, best similarity %.2f",
        len(scored),
        scored[0][1] if scored else 0.0,
    )
    return scored


def mean_similarity(
    question: str,
    persist_dir: Path,
    collection_name: str = COLLECTION_NAME,
    embedder: SentenceTransformer | None = None,
) -> float:
    """Average cosine similarity between the question and every chunk.

    This is the "background" score for the document: how similar is the
    question to the collection as a whole, not just to its best match. It
    is used to tell "the document actually contains this" from "the
    document just happens to share some vocabulary."

    Returns:
        The mean cosine similarity across all chunks (0.0 if empty).
    """
    persist_dir = Path(persist_dir)
    client = _make_client(persist_dir)

    try:
        collection = client.get_collection(name=collection_name)
    except Exception as exc:
        raise ValueError(
            f"Collection {collection_name!r} not found in {persist_dir}. "
            "Run ingest() first."
        ) from exc

    count = collection.count()
    if count == 0:
        return 0.0

    own_embedder = embedder is None
    if own_embedder:
        embedder = _make_embedder()

    query_embedding = embedder.encode([question], show_progress_bar=False).tolist()[0]

    # Fetch every chunk's embedding and average the cosine similarities.
    # Chroma stores vectors normalized, so the dot product IS cosine sim.
    # Vectorized with numpy: one matmul instead of a Python loop per chunk.
    all_embeddings = collection.get(include=["embeddings"])["embeddings"]
    if len(all_embeddings) == 0:
        return 0.0

    matrix = np.array(all_embeddings)
    similarities = matrix @ np.array(query_embedding)
    return float(similarities.sum()) / len(all_embeddings)