"""Module 3 - grounded generation and verification (PRD 3.3, Phase 3).

Pipeline (each step is its own module and testable alone):
  abstention gate 1 (retrieval score)          abstention.retrieval_gate
  grounded generation                           generation.generate
  citation registry + provenance (FR-09)        provenance.check_provenance
  claim decomposition                           claims.decompose_claims
  evidence-support judge + quote check (FR-10)  support.check_support          [LLM call]
  numeric / unit / period checks (FR-11)        numeric.check_numbers          [LLM call only
                                                                                for calculations]
  cross-document conflicts                      conflicts.check_conflicts      [LLM call only for
                                                                                multi-doc numeric]
  OCR uncertainty                               ocr_check.check_ocr
  status rules + abstention gate 2 (PRD 2.3)    status.decide_status

All LLM outputs are strict JSON and only ever *proposals*: chunk IDs are checked against the
registry, quotes against chunk text, calculations are evaluated in Python, and extracted facts
must occur in their chunk. Document name/page always come from chunk metadata.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import replace

from trustqa import llm as llm_mod
from trustqa.config import Config, load_config
from trustqa.generation import ABSTENTION_ANSWER, generate
from trustqa.llm import LLM
from trustqa.schemas import Citation, Result, RetrievedChunk, Status, Warning, WarningCode
from trustqa.verification.abstention import retrieval_gate
from trustqa.verification.claims import decompose_claims
from trustqa.verification.conflicts import CONFLICT_PROMPT_VERSION, check_conflicts
from trustqa.verification.numeric import CALC_PROMPT_VERSION, check_numbers, extract_numbers
from trustqa.verification.ocr_check import check_ocr
from trustqa.verification.provenance import CitationRegistry, check_provenance
from trustqa.verification.status import decide_status
from trustqa.verification.support import SUPPORT_PROMPT_VERSION, SupportResult, Verdict, check_support

logger = logging.getLogger("trustqa")

VERIFIER_VERSION = (f"verify-v1+{SUPPORT_PROMPT_VERSION}+{CALC_PROMPT_VERSION}"
                    f"+{CONFLICT_PROMPT_VERSION}")


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

    `llm` defaults to the configured provider; pass FakeLLM in tests. Settings come from
    load_config(); pipeline.answer_question passes its own config via run_verification.
    """
    return run_verification(question, passages, llm or _DefaultLLM(), load_config())


def run_verification(question: str, passages: Sequence[RetrievedChunk], llm: LLM,
                     config: Config) -> Result:
    """generate_and_verify with explicit settings (thresholds and ablation switches)."""
    gate = retrieval_gate(passages, config.abstain_min_score, question,
                          config.gate_lexical_pass)
    if gate:
        logger.info("abstaining before generation: %s", gate)
        return _abstain(question, [])

    out = generate(question, passages, llm)
    warnings: list[Warning] = list(out.warnings)
    if out.abstained or not out.answer or not out.cited_ids:
        if out.answer and not out.abstained and not out.cited_ids:
            warnings.append(Warning(
                WarningCode.UNSUPPORTED_CLAIM,
                "the model's answer cited no retrieved passage, so it was withheld"))
        return _abstain(question, warnings)

    # Item 1 - provenance
    upstream = sum(1 for w in out.warnings if w.code is WarningCode.INVALID_CHUNK_ID)
    prov = check_provenance(out.answer, out.cited_ids, passages, upstream_unresolved=upstream)
    warnings.extend(prov.warnings)
    if not prov.accepted_ids:
        warnings.append(Warning(
            WarningCode.UNSUPPORTED_CLAIM,
            "none of the answer's citations resolve to a trusted passage, so it was withheld"))
        return _abstain(question, warnings)
    chunks = dict(CitationRegistry.from_passages(passages).chunks)

    # Item 2 - claims
    claims = decompose_claims(prov.answer, prov.accepted_ids)

    # Item 3 - support
    support = None
    if config.verify_support:
        support, w = check_support(question, claims, chunks, llm, config.quote_min_similarity)
        warnings.extend(w)

    # Item 4 - numbers
    numeric = None
    if config.verify_numeric:
        numeric, w = check_numbers(claims, chunks, llm, question)
        warnings.extend(w)
        numeric_ids = {cid for cid, r in numeric.items() if r.numeric}
    else:
        numeric_ids = {c.claim_id for c in claims if extract_numbers(c.text)}

    # Item 5 - conflicts (only matters for answers that state numbers)
    conflict_found = conflict_failed = False
    if config.verify_conflicts and numeric_ids:
        retrieved = list({p.chunk.chunk_id: p.chunk for p in passages
                          if p.chunk.chunk_id in chunks}.values())
        rep = check_conflicts(question, retrieved, prov.accepted_ids, llm)
        warnings.extend(rep.warnings)
        conflict_found, conflict_failed = bool(rep.conflicts), rep.failed

    # Item 6 - OCR uncertainty
    ocr = check_ocr(claims, chunks, numeric_ids, config.ocr_warn_confidence)
    warnings.extend(ocr.warnings)

    # Items 7 + 8 - status and abstention gate 2
    decision = decide_status(claims, support, numeric, prov.all_resolved,
                             conflict_found=conflict_found, conflict_check_failed=conflict_failed,
                             ocr_capped_claim_ids=ocr.capped_claim_ids)
    logger.info("status %s: %s", decision.status.value, "; ".join(decision.reasons) or "all checks passed")
    if decision.status is Status.INSUFFICIENT_EVIDENCE:
        logger.info("withheld answer: %r", prov.answer)  # kept in the log for error analysis
        return _abstain(question, warnings)
    return Result(question=question, answer=prov.answer, status=decision.status,
                  citations=_with_quotes(prov.citations, support), warnings=warnings)


def _with_quotes(citations: list[Citation], support: dict[int, SupportResult] | None
                 ) -> list[Citation]:
    """Fill each citation's supporting_quote with the validated quotes from its chunks."""
    if not support:
        return citations
    out = []
    for c in citations:
        quotes = list(dict.fromkeys(
            r.quote for r in support.values()
            if r.verdict is Verdict.SUPPORTED and r.quote and r.chunk_id in c.chunk_ids))
        out.append(replace(c, supporting_quote=" ... ".join(quotes)) if quotes else c)
    return out


def _abstain(question: str, warnings: list[Warning]) -> Result:
    return Result(question=question, answer=ABSTENTION_ANSWER,
                  status=Status.INSUFFICIENT_EVIDENCE, citations=[], warnings=warnings)
