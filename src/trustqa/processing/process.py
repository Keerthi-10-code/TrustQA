"""Module 1 - document processing (PRD 3.3)."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

from trustqa.config import Config, load_config
from trustqa.processing.chunking import chunk_pages
from trustqa.processing.extract import DocumentRegistry, extract_pages
from trustqa.schemas import Chunk

logger = logging.getLogger("trustqa")


def process_pdfs(pdf_paths: Sequence[str | Path], config: Config | None = None) -> list[Chunk]:
    """Turn PDFs into canonical chunk records.

    Assigns a stable document_id per file (DOC01, DOC02, ... in input order), extracts
    text page by page (PyMuPDF), falls back to Tesseract OCR on low-text pages, extracts
    tables (pdfplumber), and chunks without crossing page boundaries.

    Returns chunks with unique chunk_ids (e.g. DOC01_P4_C1) and 1-indexed page_number.
    Serialise with Chunk.to_dict() for data/processed/*.jsonl.

    Raises ValueError for unreadable / encrypted PDFs (NFR-09).

    Phase 1 status: digital text layer only. OCR fallback and tables arrive in Phase 2;
    pages without a text layer currently yield no chunks (a warning is logged). The same
    file passed twice (by content hash) is processed once.
    """
    config = config or load_config()
    registry = DocumentRegistry()
    chunks: list[Chunk] = []
    for path in pdf_paths:
        entry, is_new = registry.register(path) if Path(path).is_file() else (None, True)
        if entry is None:
            raise ValueError(f"{Path(path).name}: file not found ({path})")
        if not is_new:
            logger.warning("%s is a duplicate of %s (%s); skipped", Path(path).name,
                           entry.document_name, entry.document_id)
            continue
        pages = extract_pages(entry.path, entry.document_id, entry.document_name)
        doc_chunks = chunk_pages(pages, config.chunk_size, config.chunk_overlap)
        logger.info("%s -> %s: %d pages, %d chunks", entry.document_name,
                    entry.document_id, len(pages), len(doc_chunks))
        chunks.extend(doc_chunks)

    ids = [c.chunk_id for c in chunks]
    if len(ids) != len(set(ids)):  # cannot happen by construction; guard the contract
        raise AssertionError("duplicate chunk_ids produced")
    return chunks
