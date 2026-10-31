"""Module 4 - integration (PRD 3.3)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from trustqa.config import Config
from trustqa.schemas import Result


def answer_question(
    question: str, uploaded_docs: Sequence[str | Path], config: Config | None = None
) -> Result:
    """End to end: process_pdfs -> build_index -> retrieve -> generate_and_verify.

    Processed chunks and the index are cached per document set so repeated questions
    over the same uploads do not reprocess. Returns the final Result for the UI/CLI.
    """
    raise NotImplementedError("Phase 1")
