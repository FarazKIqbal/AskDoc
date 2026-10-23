"""Extract text from a PDF, keeping the page number for each chunk of text.

The chunker (rag/chunker.py) later splits this text into overlapping pieces,
so we don't try to split here — we just hand the raw page text forward along
with the page it came from.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import NamedTuple

from pypdf import PdfReader

logger = logging.getLogger(__name__)


class Page(NamedTuple):
    """One page of extracted text, with its page number.

    Page numbers are 1-based (page 1 is the first page) so they match the
    numbers a human sees when opening the PDF.
    """

    page_number: int
    text: str


def load_pdf(pdf_path: Path | str) -> list[Page]:
    """Read every page of a PDF and return a list of Page objects.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        A list of Page objects, one per page that had extractable text.
        Pages with no text (e.g. scanned images) are skipped but do not
        raise — the caller can decide what to do with an empty result.

    Raises:
        FileNotFoundError: if the file does not exist.
        ValueError: if the file is not a valid PDF.
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    logger.info("Loading PDF: %s", path)
    reader = PdfReader(str(path))

    if len(reader.pages) == 0:
        raise ValueError(f"PDF has no pages: {path}")

    pages: list[Page] = []
    for index, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = text.strip()
        if not text:
            # Scanned page, image-only page, or blank page — skip it.
            logger.warning("Page %d has no extractable text; skipping", index)
            continue
        pages.append(Page(page_number=index, text=text))

    logger.info("Extracted text from %d page(s)", len(pages))
    return pages