"""Module 1 - document processing (PRD 3.3)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from trustqa.config import Config
from trustqa.schemas import Chunk


def process_pdfs(pdf_paths: Sequence[str | Path], config: Config | None = None) -> list[Chunk]:
    """Turn PDFs into canonical chunk records.

    Assigns a stable document_id per file (DOC01, DOC02, ... in input order), extracts
    text page by page (PyMuPDF), falls back to Tesseract OCR on low-text pages, extracts
    tables (pdfplumber), and chunks without crossing page boundaries.

    Returns chunks with unique chunk_ids (e.g. DOC01_P4_C1) and 1-indexed page_number.
    Serialise with Chunk.to_dict() for data/processed/*.jsonl.

    Raises ValueError for unreadable / encrypted PDFs (NFR-09).
    """
    raise NotImplementedError("Phase 1/2")
