"""The full RAG pipeline: PDF -> chunks -> embeddings -> answer.

This module ties together loader, chunker, store, and generator into two
simple functions, `ingest` and `ask`. Keeping the pipeline in one place
means the UI and the eval script can both call the same code path.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from rag.chunker import DEFAULT_CHUNK_WORDS, DEFAULT_OVERLAP_WORDS, chunk_pages
from rag.chunker import Chunk
from rag.generator import generate_answer
from rag.loader import load_pdf
from rag.local_llm import LocalLLM
from rag.store import COLLECTION_NAME, add_chunks, query, mean_similarity, query_with_scores

logger = logging.getLogger(__name__)

# Where Chroma stores its database files, relative to the project root.
DEFAULT_CHROMA_DIR = Path(__file__).resolve().parent.parent / "chroma_db"

# Which model answers questions. "claude" (API, needs ANTHROPIC_API_KEY) or
# "openai" (uses LLM_API_KEY / LLM_BASE_URL). "local" runs the offline
# Qwen2.5 model — no API key, no network.
DEFAULT_PROVIDER = os.environ.get("RAG_PROVIDER", "openai")

# Out-of-scope detection: if the best-matching chunk's similarity to the
# question divided by the mean similarity across the document is below this
# ratio, the question is treated as out of scope and we return "I couldn't
# find that in the document" instead of letting the model guess.
MIN_SIMILARITY_RATIO = float(os.environ.get("RAG_MIN_SIMILARITY_RATIO", "1.15"))

# Cache the local LLM so the model loads once per session instead of on
# every question. Loading is the slow part; inference is fast.
_local_llm_cache = None


def _get_local_llm():
    """Return the shared LocalLLM instance, loading it on first use."""
    global _local_llm_cache
    if _local_llm_cache is None:
        _local_llm_cache = LocalLLM()
    return _local_llm_cache


def ingest(
    pdf_path: Path | str,
    chroma_dir: Path | None = None,
    chunk_words: int = DEFAULT_CHUNK_WORDS,
    overlap_words: int = DEFAULT_OVERLAP_WORDS,
) -> list[Chunk]:
    """Load a PDF, chunk it, embed it, and store it in Chroma.

    Args:
        pdf_path: Path to the PDF file.
        chroma_dir: Directory for the Chroma database. Defaults to
            <project>/chroma_db.
        chunk_words: Words per chunk (tunable for eval comparisons).
        overlap_words: Words shared between chunks (tunable too).

    Returns:
        The list of Chunk objects that were stored.

    Raises:
        FileNotFoundError: if the PDF doesn't exist.
        ValueError: if the PDF has no extractable text.
    """
    chroma_dir = Path(chroma_dir) if chroma_dir else DEFAULT_CHROMA_DIR

    pages = load_pdf(pdf_path)
    chunks = chunk_pages(pages, chunk_words=chunk_words, overlap_words=overlap_words)
    add_chunks(chunks, chroma_dir)

    logger.info("Ingestion complete: %d page(s) -> %d chunk(s)", len(pages), len(chunks))
    return chunks


def ask(
    question: str,
    chroma_dir: Path | None = None,
    top_k: int = 4,
    model: str | None = None,
    provider: str | None = None,
) -> tuple[str, list[Chunk]]:
    """Retrieve relevant chunks and answer the question with an LLM.

    Args:
        question: The user's question.
        chroma_dir: Directory for the Chroma database. Defaults to
            <project>/chroma_db.
        top_k: Number of chunks to retrieve.
        model: Optional model override (only used with provider="claude").
        provider: "claude" (API, needs ANTHROPIC_API_KEY) or "openai"
            (uses LLM_API_KEY / LLM_BASE_URL). Defaults to the
            RAG_PROVIDER env var or "claude".

    Returns:
        A tuple of (answer_text, retrieved_chunks) so the UI can show the
        source passages alongside the answer.
    """
    chroma_dir = Path(chroma_dir) if chroma_dir else DEFAULT_CHROMA_DIR
    provider = provider or DEFAULT_PROVIDER

    # Out-of-scope detection: compare the best-matching chunk's similarity
    # to the average similarity across the whole document. If the best
    # match is not clearly better than the background, the question is
    # treated as out of scope and we return "I couldn't find that in the
    # document" instead of letting the LLM guess.
    scored = query_with_scores(question, chroma_dir, top_k=top_k)
    chunks = [chunk for chunk, _score in scored]
    best_score = scored[0][1] if scored else 0.0
    background = mean_similarity(question, chroma_dir)
    ratio = (best_score / background) if background > 0 else 0.0

    if ratio < MIN_SIMILARITY_RATIO:
        logger.info(
            "Best chunk similarity %.2f vs background %.2f (ratio %.2f) "
            "is below threshold %.2f; treating question as out of scope",
            best_score,
            background,
            ratio,
            MIN_SIMILARITY_RATIO,
        )
        return "I couldn't find that in the document.", chunks

    if provider == "local":
        llm = _get_local_llm()
        answer = llm.generate_answer(chunks, question)
    else:
        answer = generate_answer(chunks, question, model=model)

    return answer, chunks