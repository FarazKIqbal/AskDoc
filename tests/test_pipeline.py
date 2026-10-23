"""Tests for rag/pipeline.py — the two functions the whole app depends on.

ingest() and ask() are the only entry points the UI and the eval script
share, and until now neither had a test. These cover them with a real
(temp) PDF and Chroma store, mocking only the network call.
"""

import pytest

from rag.chunker import Chunk
from rag.pipeline import ask, ingest


def _make_pdf(path, text: str) -> None:
    """Write a one-page PDF with the given text (pypdf, no fixtures needed)."""
    from pypdf import PdfWriter
    from pypdf.generic import DictionaryObject, NameObject, StreamObject

    writer = PdfWriter()
    page = writer.add_blank_page(width=720, height=720)
    # pypdf v4 has no direct text API, so we inject a content stream.
    # A font resource is required or extract_text() returns garbage.
    font = DictionaryObject()
    font[NameObject("/Type")] = NameObject("/Font")
    font[NameObject("/Subtype")] = NameObject("/Type1")
    font[NameObject("/BaseFont")] = NameObject("/Helvetica")
    resources = DictionaryObject()
    resources[NameObject("/Font")] = DictionaryObject()
    resources[NameObject("/Font")][NameObject("/F1")] = font
    page[NameObject("/Resources")] = resources

    content = f"BT /F1 12 Tf 72 700 Td ({text}) Tj ET"
    stream = StreamObject()
    stream.set_data(content.encode("latin-1", errors="replace"))
    page.replace_contents(stream)
    with open(path, "wb") as f:
        writer.write(f)


def test_ingest_and_ask_round_trip(tmp_path, monkeypatch):
    """ingest() stores chunks; ask() returns the model's answer + sources."""
    pdf = tmp_path / "doc.pdf"
    _make_pdf(pdf, "The sky is blue. Employees get 15 days of paid time off per year.")

    chroma = tmp_path / "chroma"
    chunks = ingest(pdf, chroma_dir=chroma)
    assert chunks  # something was stored
    assert all(isinstance(c, Chunk) for c in chunks)

    # Avoid the network: the pipeline should pass our chunks straight through.
    monkeypatch.setattr(
        "rag.pipeline.generate_answer",
        lambda chunks, question, model=None: "15 days",
    )
    # A single-chunk document gives best==background (ratio 1.0), which the
    # out-of-scope guard would block. Mock the background so this test
    # exercises the answer path; the guard itself has its own test below.
    monkeypatch.setattr("rag.pipeline.mean_similarity", lambda *a, **k: 0.1)
    answer, returned_chunks = ask("How much PTO?", chroma_dir=chroma)
    assert answer == "15 days"
    assert returned_chunks  # sources come back for the UI to display


def test_ask_returns_not_found_when_out_of_scope(tmp_path, monkeypatch):
    """A question with no real match should hit the out-of-scope guard."""
    pdf = tmp_path / "doc.pdf"
    _make_pdf(pdf, "The sky is blue. Employees get 15 days of paid time off per year.")

    chroma = tmp_path / "chroma"
    ingest(pdf, chroma_dir=chroma)

    # Force the retrieval to look irrelevant: best score barely above background.
    fake_chunk = Chunk(text="irrelevant", page_number=1)
    monkeypatch.setattr(
        "rag.pipeline.query_with_scores",
        lambda *a, **k: [(fake_chunk, 0.4)],
    )
    monkeypatch.setattr(
        "rag.pipeline.mean_similarity",
        lambda *a, **k: 0.35,  # ratio = 0.4/0.35 = 1.14 < 1.15 threshold
    )
    answer, chunks = ask("What is the company stock price?", chroma_dir=chroma)
    assert answer == "I couldn't find that in the document."
    assert chunks  # the UI still gets the (irrelevant) sources to show


def test_ask_uses_local_provider(tmp_path, monkeypatch):
    """provider='local' routes through the cached LocalLLM, not the API."""
    pdf = tmp_path / "doc.pdf"
    _make_pdf(pdf, "The sky is blue.")

    chroma = tmp_path / "chroma"
    ingest(pdf, chroma_dir=chroma)

    called: dict = {}

    class _FakeLocal:
        def generate_answer(self, chunks, question):
            called["question"] = question
            return "local answer"

    monkeypatch.setattr("rag.pipeline._local_llm_cache", _FakeLocal())
    monkeypatch.setattr("rag.pipeline.mean_similarity", lambda *a, **k: 0.1)
    answer, _chunks = ask("What color is the sky?", chroma_dir=chroma, provider="local")
    assert answer == "local answer"
    assert called["question"] == "What color is the sky?"