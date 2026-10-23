"""Build a prompt from retrieved chunks and call an LLM API for an answer.

This module is provider-agnostic. It reads three environment variables:

    LLM_API_KEY    - your API key (OpenAI, DeepSeek, Groq, etc.)
    LLM_BASE_URL   - optional endpoint override (for OpenAI-compatible APIs)
    LLM_MODEL      - model name, e.g. "gpt-4o-mini", "deepseek-chat"

If LLM_BASE_URL points at an Anthropic endpoint, Claude is used instead.
The system prompt is a named constant at the top of the file so it's easy
to find and tweak. It tells the model three things:

1. Answer only from the provided context.
2. Say "I couldn't find that in the document" when the context doesn't
   contain the answer, instead of guessing.
3. Include each chunk labeled with its page number so answers can cite them.
"""

from __future__ import annotations

import logging
import os
from typing import Sequence

from anthropic import Anthropic
from openai import OpenAI

from rag.chunker import Chunk

logger = logging.getLogger(__name__)

# Sensible defaults. All three are overridable via environment variables so
# the model and endpoint can be swapped without touching code.
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_BASE_URL = None  # OpenAI's default endpoint

# The system prompt. Kept as a constant so it's easy to find and tweak.
SYSTEM_PROMPT = """\
You are a careful reading assistant. Answer the user's question using ONLY \
the document passages provided below. Do not use any outside knowledge.

Rules:
- If the passages do not contain enough information to answer, say exactly: \
"I couldn't find that in the document." Do not guess or fill in the blanks.
- Be concise. One or two short sentences is usually enough.
- Cite the page number of each passage you use, like (page 3). If you use \
more than one passage, cite each one.
- If the answer is not in the passages, say so rather than making something up.
"""


def _format_context(chunks: Sequence[Chunk]) -> str:
    """Turn retrieved chunks into the text block we feed to the model.

    Each chunk is labeled with its page number so the model can cite it and
    so the UI can show the user where the answer came from.
    """
    if not chunks:
        return "(no relevant passages found)"

    parts = []
    for chunk in chunks:
        parts.append(f"[page {chunk.page_number}]\n{chunk.text}")
    return "\n\n---\n\n".join(parts)


def build_prompt(chunks: Sequence[Chunk], question: str) -> str:
    """Assemble the user message: context first, then the question.

    Putting the context before the question helps the model treat the
    passages as the source of truth rather than something to reconcile with
    its own pretraining knowledge.
    """
    context = _format_context(chunks)
    return (
        "Document passages:\n\n"
        f"{context}\n\n"
        "---\n\n"
        f"Question: {question}"
    )


def _make_openai_client() -> OpenAI:
    """Create an OpenAI-compatible client from environment variables.

    Raises:
        RuntimeError: if LLM_API_KEY is not set.
    """
    api_key = os.environ.get("LLM_API_KEY")
    if not api_key:
        raise RuntimeError(
            "LLM_API_KEY is not set. Copy .env.example to .env and add "
            "your key, or set the LLM_API_KEY environment variable."
        )
    base_url = os.environ.get("LLM_BASE_URL") or None
    return OpenAI(api_key=api_key, base_url=base_url)


def _make_anthropic_client() -> Anthropic:
    """Create an Anthropic client from environment variables.

    Raises:
        RuntimeError: if ANTHROPIC_API_KEY is not set.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add "
            "your key, or set the ANTHROPIC_API_KEY environment variable."
        )
    return Anthropic(api_key=api_key)


def _is_anthropic_endpoint(base_url: str | None) -> bool:
    """Heuristic: does this endpoint look like Anthropic?"""
    if not base_url:
        return False
    return "anthropic" in base_url.lower()


def generate_answer(
    chunks: Sequence[Chunk],
    question: str,
    model: str | None = None,
    client=None,
) -> str:
    """Ask an LLM to answer the question using only the retrieved chunks.

    Args:
        chunks: Retrieved chunks from rag.store.query.
        question: The user's question.
        model: Model name. Defaults to LLM_MODEL env var or a sensible
            default. Never hardcode a model name in production.
        client: Optional pre-built client (OpenAI or Anthropic), for tests.

    Returns:
        The model's answer as a string.

    Raises:
        RuntimeError: if the API key is missing or the API call fails.
    """
    model_name = model or os.environ.get("LLM_MODEL") or DEFAULT_MODEL
    base_url = os.environ.get("LLM_BASE_URL") or DEFAULT_BASE_URL

    own_client = client is None
    if own_client:
        if _is_anthropic_endpoint(base_url):
            client = _make_anthropic_client()
        else:
            client = _make_openai_client()

    user_prompt = build_prompt(chunks, question)

    logger.info(
        "Asking %s with %d chunk(s) for question: %r",
        model_name,
        len(chunks),
        question[:80],
    )

    try:
        if isinstance(client, Anthropic):
            message = client.messages.create(
                model=model_name,
                max_tokens=512,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
            )
            answer = "".join(
                block.text for block in message.content if hasattr(block, "text")
            ).strip()
        else:
            # OpenAI-compatible chat completion.
            response = client.chat.completions.create(
                model=model_name,
                max_tokens=512,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            )
            # Some local servers return a raw string instead of a
            # ChatCompletion object. Handle both cases.
            if isinstance(response, str):
                answer = response
            else:
                answer = response.choices[0].message.content or ""
            answer = answer.strip()
    except Exception as exc:
        logger.error("LLM API error: %s", exc)
        raise RuntimeError(
            "Could not get an answer from the LLM. Please check your API "
            "key, model name, and connection, then try again."
        ) from exc

    if not answer:
        logger.warning("LLM returned an empty answer")
        return "I couldn't find that in the document."

    return answer