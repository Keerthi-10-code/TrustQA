"""Module 3 - grounded generation and verification (PRD 3.3)."""

from __future__ import annotations

from collections.abc import Sequence

from trustqa import llm as llm_mod
from trustqa.generation import ABSTENTION_ANSWER, generate, resolve_citations
from trustqa.llm import LLM
from trustqa.schemas import Result, RetrievedChunk, Status, Warning, WarningCode


class _DefaultLLM:
    """Delegates to trustqa.llm.complete so set_llm() overrides are honoured."""

    def complete(self, prompt: str, system: str | None = None, json: bool = False) -> str:
        return llm_mod.complete(prompt, system=system, json=json)


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

    Phase 1 (baseline system A) status rules - no support/numeric checks yet:
    - model abstains, gives no answer, or cites no retrieved chunk -> INSUFFICIENT_EVIDENCE
      (answer replaced by a fixed abstention message, no citations);
    - otherwise -> PARTIALLY_SUPPORTED. Never VERIFIED: PRD 2.3 requires a separate
      evidence-support check, which arrives in Phase 3.
    """
    out = generate(question, passages, llm or _DefaultLLM())
    warnings = list(out.warnings)

    if out.abstained or not out.answer or not out.cited_ids:
        if out.answer and not out.abstained and not out.cited_ids:
            warnings.append(Warning(
                WarningCode.UNSUPPORTED_CLAIM,
                "the model's answer cited no retrieved passage, so it was withheld"))
        return Result(question=question, answer=ABSTENTION_ANSWER,
                      status=Status.INSUFFICIENT_EVIDENCE, citations=[], warnings=warnings)

    return Result(question=question, answer=out.answer, status=Status.PARTIALLY_SUPPORTED,
                  citations=resolve_citations(out.cited_ids, passages), warnings=warnings)
