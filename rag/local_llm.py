"""Run a local open-source LLM (via transformers + torch) instead of Claude.

This is the offline / no-cost path: a small open model runs on your own
machine, so the app works with no API key and no monthly bill. The model
is downloaded from HuggingFace once and cached on disk.

The prompt contract is identical to rag/generator.py — same system prompt,
same "answer only from context" rules — so the rest of the pipeline does
not need to know which model produced the answer.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rag.chunker import Chunk
from rag.generator import SYSTEM_PROMPT, build_prompt

logger = logging.getLogger(__name__)

# A small, free, permissively-licensed instruction model (~500M params).
# Runs on CPU in a few seconds per answer; good enough for grounded Q&A.
DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
# Maximum tokens for the answer.
DEFAULT_MAX_TOKENS = 512


class LocalLLM:
    """Wraps a local causal LM with the same interface as Claude.

    Usage:
        llm = LocalLLM()
        answer = llm.generate_answer(chunks, question)
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        device: str | None = None,
    ) -> None:
        """Load (or download) the model and tokenizer.

        Args:
            model_name: HuggingFace model id.
            max_tokens: Maximum length of the answer.
            device: "cpu" (default), "cuda", or "mps". Auto-detected if None.
        """
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None
        self._tokenizer = None

    def _ensure_loaded(self) -> None:
        """Load the model and tokenizer on first use (lazy)."""
        if self._model is not None:
            return

        logger.info(
            "Loading local LLM %s on %s", self.model_name, self.device
        )
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float32,
        ).to(self.device)
        self._model.eval()

    def generate_answer(
        self,
        chunks: Sequence[Chunk],
        question: str,
    ) -> str:
        """Answer the question using only the retrieved chunks.

        Args:
            chunks: Retrieved chunks from rag.store.query.
            question: The user's question.

        Returns:
            The model's answer as a string.
        """
        self._ensure_loaded()
        assert self._model is not None and self._tokenizer is not None

        user_prompt = build_prompt(chunks, question)

        logger.info("Asking local LLM for question: %r", question[:80])

        # Use the model's own chat template so system/user/assistant roles
        # are formatted exactly as the model was trained to expect.
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self._tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output = self._model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                do_sample=False,  # Deterministic answers are better for eval.
                pad_token_id=self._tokenizer.eos_token_id,
            )

        # Slice off the prompt; the model only generated the new tokens.
        new_tokens = output[0][inputs["input_ids"].shape[1]:]
        answer = self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        if not answer:
            logger.warning("Local LLM returned an empty answer")
            return "I couldn't find that in the document."

        return answer