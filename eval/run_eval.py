"""Run the evaluation set against the RAG pipeline and report accuracy.

Usage:
    python eval/run_eval.py                    # use defaults
    python eval/run_eval.py --chunk-words 300  # compare settings

Each question in eval/questions.json has an expected key fact. An answer is
marked correct if it contains that fact (case-insensitive substring match).
This is intentionally simple -- it's v1. An LLM-as-judge is the next step.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Allow running as `python eval/run_eval.py` from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

from rag.pipeline import DEFAULT_CHROMA_DIR, ask, ingest

# Load environment variables from .env (API keys, RAG_PROVIDER, etc.).
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
QUESTIONS_PATH = PROJECT_ROOT / "eval" / "questions.json"
SAMPLE_PDF = PROJECT_ROOT / "data" / "sample.pdf"


def _safe(text: str) -> str:
    """Encode text for the terminal, replacing characters it can't show.

    Some models return Unicode like a narrow no-break space (U+202F); the
    default cp1252 console then crashes on print(). Replace, don't fail.
    """
    return text.encode("ascii", errors="replace").decode("ascii")


# A few LLM responses leak narrow no-break spaces (U+202F) and similar
# invisible Unicode whitespace chars that break a simple substring match.
# Replace them with normal spaces so "90{u202f}day" still matches "90 day".
_INVISIBLE_WHITESPACE = "​‌‍             "


def load_questions(path: Path) -> list[dict]:
    """Load the question set from JSON."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def is_correct(answer: str, expected: str) -> bool:
    """A simple substring match (case-insensitive) for v1 scoring."""
    answer_n = "".join(c.replace(c, " ") if c in _INVISIBLE_WHITESPACE else c for c in answer)
    expected_n = "".join(c.replace(c, " ") if c in _INVISIBLE_WHITESPACE else c for c in expected)
    return expected_n.lower() in answer_n.lower()


def run(
    questions: list[dict],
    top_k: int,
    provider: str,
) -> tuple[int, int, list[dict]]:
    """Ask every question and return (correct, total, results)."""
    correct = 0
    results = []
    for q in questions:
        answer, chunks = ask(
            q["question"],
            top_k=top_k,
            provider=provider,
        )
        ok = is_correct(answer, q["expected_answer"])
        correct += int(ok)
        results.append(
            {
                "question": q["question"],
                "expected": q["expected_answer"],
                "answer": answer,
                "correct": ok,
                "pages": sorted({c.page_number for c in chunks}),
            }
        )
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] Q: {_safe(q['question'])}")
        print(f"          expected: {_safe(q['expected_answer'])}")
        print(f"          got:      {_safe(answer)}")
    return correct, len(questions), results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the RAG eval set.")
    parser.add_argument(
        "--chunk-words",
        type=int,
        default=500,
        help="Words per chunk (default: 500).",
    )
    parser.add_argument(
        "--overlap-words",
        type=int,
        default=50,
        help="Overlap between chunks (default: 50).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=4,
        help="Chunks to retrieve per question (default: 4).",
    )
    parser.add_argument(
        "--provider",
        choices=["openai", "claude", "local"],
        default="openai",
        help="Which model answers questions (default: openai).",
    )
    parser.add_argument(
        "--questions",
        type=Path,
        default=QUESTIONS_PATH,
        help="Path to the questions JSON file.",
    )
    args = parser.parse_args()

    if not SAMPLE_PDF.exists():
        print(f"Sample PDF not found: {SAMPLE_PDF}")
        print("Generate it or point --questions at a real PDF first.")
        sys.exit(1)

    questions = load_questions(args.questions)
    print(f"\nIngesting {SAMPLE_PDF} "
          f"(chunk={args.chunk_words}, overlap={args.overlap_words})...")
    ingest(
        SAMPLE_PDF,
        chroma_dir=DEFAULT_CHROMA_DIR,
        chunk_words=args.chunk_words,
        overlap_words=args.overlap_words,
    )

    print(f"\nAsking {len(questions)} question(s) with provider={args.provider}...\n")
    correct, total, results = run(
        questions,
        top_k=args.top_k,
        provider=args.provider,
    )

    accuracy = 100.0 * correct / total if total else 0.0
    print(f"\n{'='*60}")
    print(f"Accuracy: {correct}/{total} = {accuracy:.1f}%")
    print(f"{'='*60}")

    failed = [r for r in results if not r["correct"]]
    if failed:
        print(f"\nFailed questions ({len(failed)}):")
        for r in failed:
            print(f"  - {_safe(r['question'])}")
            print(f"    expected: {_safe(r['expected'])}")
            print(f"    got:      {_safe(r['answer'])}")
    else:
        print("\nAll questions passed.")


if __name__ == "__main__":
    main()
