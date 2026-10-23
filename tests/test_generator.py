"""Tests for rag/generator.py.

The live LLM call requires an API key, so we test the parts that don't:
prompt construction, the missing-key error, and the empty-answer fallback.
"""

import os

import pytest

from rag.chunker import Chunk
from rag.generator import (
    DEFAULT_MODEL,
    SYSTEM_PROMPT,
    _format_context,
    _make_openai_client,
    build_prompt,
    generate_answer,
)


def test_system_prompt_rules():
    """The system prompt must contain the three required rules."""
    lowered = SYSTEM_PROMPT.lower()
    assert "using only the document passages" in lowered
    assert "i couldn't find that in the document" in lowered
    assert "page" in lowered


def test_format_context_labels_pages():
    """Each chunk should be labeled with its page number."""
    chunks = [
        Chunk(text="Cats like naps.", page_number=1),
        Chunk(text="Dogs like walks.", page_number=2),
    ]
    formatted = _format_context(chunks)
    assert "[page 1]" in formatted
    assert "[page 2]" in formatted
    assert "Cats like naps." in formatted
    assert "Dogs like walks." in formatted


def test_format_context_empty():
    """An empty chunk list should produce a clear placeholder."""
    assert _format_context([]) == "(no relevant passages found)"


def test_build_prompt_contains_context_and_question():
    """The user message should have the passages and the question."""
    chunks = [Chunk(text="The cat sat on the mat.", page_number=1)]
    prompt = build_prompt(chunks, "Where did the cat sit?")
    assert "The cat sat on the mat." in prompt
    assert "[page 1]" in prompt
    assert "Where did the cat sit?" in prompt


def test_make_client_raises_without_key(monkeypatch):
    """Missing API key should raise RuntimeError, not crash."""
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        _make_openai_client()


def test_generate_answer_empty_chunks_uses_placeholder(monkeypatch):
    """With no chunks the prompt should still be well-formed."""
    # Avoid making a real API call: fake the client.
    class _FakeMessage:
        content = ""

    class _FakeChoice:
        message = _FakeMessage()

    class _FakeResponse:
        choices = [_FakeChoice()]

    class _FakeCompletions:
        def create(self, **kwargs):
            return _FakeResponse()

    class _FakeClient:
        chat = type("Chat", (), {"completions": _FakeCompletions()})()

    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    answer = generate_answer([], "anything", client=_FakeClient())
    # Empty content -> the documented "not found" fallback.
    assert answer == "I couldn't find that in the document."


def test_default_model_is_not_hardcoded(monkeypatch):
    """The default model name should be a named constant, not a magic string.

    This checks the actual fallback chain (env var -> constant) by capturing
    the model name the fake client receives, instead of just asserting the
    constant starts with "gpt-".
    """
    monkeypatch.delenv("LLM_MODEL", raising=False)
    assert DEFAULT_MODEL  # the constant exists and is non-empty

    captured: dict = {}

    class _FakeMessage:
        content = ""  # empty -> triggers the documented "not found" fallback

    class _FakeChoice:
        message = _FakeMessage()

    class _FakeResponse:
        choices = [_FakeChoice()]

    class _FakeCompletions:
        def create(self, **kwargs):
            captured["model"] = kwargs.get("model")
            return _FakeResponse()

    class _FakeClient:
        chat = type("Chat", (), {"completions": _FakeCompletions()})()

    answer = generate_answer([], "anything", client=_FakeClient())
    assert answer == "I couldn't find that in the document."
    assert captured["model"] == DEFAULT_MODEL