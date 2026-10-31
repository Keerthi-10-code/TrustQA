"""Module 3 - grounded generation and verification (PRD 3.3)."""

from __future__ import annotations

from collections.abc import Sequence

from trustqa.llm import LLM
from trustqa.schemas import Result, RetrievedChunk


def generate_and_verify(
    question: str, passages: Sequence[RetrievedChunk], llm: LLM | None = None
) -> Result:
    """Generate an answer from `passages` only, then verify it.

    - The LLM may cite only chunk_ids present in `passages`; others are rejected with an
      INVALID_CHUNK_ID warning (FR-09).
    - Citation document_name/page_number are resolved from chunk metadata, never from
      LLM output.
    - Claims are checked for support (FR-10) and numeric consistency (FR-11).
    - Status follows PRD 2.3; with no supporting evidence the result is
      INSUFFICIENT_EVIDENCE with no guessed answer.

    `llm` defaults to the configured provider; pass FakeLLM in tests.
    """
    raise NotImplementedError("Phase 1/3")
