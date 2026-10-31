"""Checks that an answer is about what the question asks (failures.md F10, F13). Plain Python.

Scope (config.verify_scope, F13): the question's identifying names (trustqa/scope.py) say
which documents it is about.
- every citation from another document -> the answer is withheld (it is about something
  else: e.g. the RAG paper's V100 GPUs offered for "What GPUs were used to train the
  Self-RAG models?");
- some citations from another document -> warning, status capped at Partially Supported;
- a capitalised name in the question that no document mentions at all -> withheld (the
  documents cannot answer a question about it).

Answer type (config.verify_answers_question, F10): a question asking for a quantity
("how many", "what ... score", "accuracy", "nDCG@10", ...) is not Verified by an answer that
contains no number (T06: "We consider four popular open-domain QA datasets ..." for an
exact-match score). This only caps the status, it never withholds. (Asking the support judge
for an "answers_question" verdict is a separate switch, config.judge_answers_question.)
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from trustqa.schemas import Citation, Warning, WarningCode
from trustqa.scope import DocumentScope

_QUANTITY_Q = re.compile(
    r"\b(how many|how much|how long|how large|how big|how fast|what percentage|what percent|"
    r"what proportion|what fraction|score|accuracy|bleu|ndcg@?\d*|recall@\d+|precision@\d+|"
    r"mrr|f1|exact match|latency|throughput|perplexity)\b", re.I)
_NUMBER_WORDS = re.compile(
    r"\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|dozen|hundred|"
    r"thousand|million|billion)\b", re.I)
_SCORE_Q = re.compile(r"\b(score|accuracy|bleu|ndcg|recall@|precision@|mrr|f1|exact match|"
                      r"percentage|percent|latency|perplexity)", re.I)


def expects_number(question: str) -> bool:
    return bool(_QUANTITY_Q.search(question))


def answer_has_number(question: str, answer: str) -> bool:
    """Digits, or (for counts only, not scores) a number word such as 'six'."""
    if re.search(r"\d", answer):
        return True
    return not _SCORE_Q.search(question) and bool(_NUMBER_WORDS.search(answer))


def check_answer_type(question: str, answer: str) -> list[Warning]:
    answer = re.sub(r"\[[^\[\]]*\]", " ", answer)  # citation markers carry digits (DOC02_P7_C1)
    if expects_number(question) and not answer_has_number(question, answer):
        return [Warning(WarningCode.UNSUPPORTED_CLAIM,
                        "the question asks for a quantity but the answer states no number; "
                        "it may not answer the question")]
    return []


def repair_neighbour_citations(claims, citations: Sequence[Citation], chunks: dict,
                               passages) -> tuple[list, list[Citation], list[str]]:
    """A small model often cites the chunk next to the one that states a number (same page).
    For a claim whose numbers are not all in its cited chunks, attach a retrieved chunk from
    the same document and page whose text contains every number exactly. The document and
    page shown do not change; the claim is then checked against the added chunk as usual.
    Returns (claims, citations, added chunk ids)."""
    import dataclasses

    from trustqa.verification.support import _number_tokens

    pages = {(c.document_id, c.page_number) for c in citations}
    neighbours = [p.chunk for p in passages
                  if (p.chunk.document_id, p.chunk.page_number) in pages]
    added: list[str] = []
    out = []
    for cl in claims:
        nums = _number_tokens(cl.text)
        cited_text = " ".join(chunks[c].text for c in cl.chunk_ids if c in chunks)
        if nums and not all(n in set(_number_tokens(cited_text)) for n in nums):
            cited_pages = {(chunks[c].document_id, chunks[c].page_number) for c in cl.chunk_ids if c in chunks}
            for ch in neighbours:
                if (ch.chunk_id not in cl.chunk_ids
                        and (ch.document_id, ch.page_number) in cited_pages
                        and all(n in set(_number_tokens(ch.text)) for n in nums)):
                    cl = dataclasses.replace(cl, chunk_ids=(*cl.chunk_ids, ch.chunk_id))
                    added.append(ch.chunk_id)
                    break
        out.append(cl)
    if not added:
        return list(claims), list(citations), []
    by_chunk = {ch.chunk_id: ch for ch in neighbours}
    new_cits = []
    for c in citations:
        extra = [cid for cid in added if (by_chunk[cid].document_id, by_chunk[cid].page_number)
                 == (c.document_id, c.page_number) and cid not in c.chunk_ids]
        new_cits.append(dataclasses.replace(c, chunk_ids=[*c.chunk_ids, *dict.fromkeys(extra)]))
    return out, new_cits, list(dict.fromkeys(added))


def check_scope(question: str, citations: Sequence[Citation], scope: DocumentScope
                ) -> tuple[list[Warning], bool, bool]:
    """(warnings, withhold, cap_at_partial)."""
    missing = scope.missing(question)
    if missing:
        return ([Warning(WarningCode.UNSUPPORTED_CLAIM,
                         f"the question is about {', '.join(missing)}, which none of the "
                         "uploaded documents mention, so the answer was withheld")], True, True)
    allowed = scope.allowed_documents(question)
    if allowed is None or not citations:
        return [], False, False
    outside = [c for c in citations if c.document_id not in allowed]
    if not outside:
        return [], False, False
    names = ", ".join(scope.identifying(question))
    docs = ", ".join(sorted({c.document_name for c in outside}))
    if len(outside) == len(citations):
        return ([Warning(WarningCode.UNSUPPORTED_CLAIM,
                         f"the question is about {names}, but the answer cites only {docs}, "
                         "which is about something else, so the answer was withheld",
                         chunk_id=outside[0].chunk_ids[0])], True, True)
    return ([Warning(WarningCode.UNSUPPORTED_CLAIM,
                     f"the question is about {names}, but part of the answer cites {docs}",
                     chunk_id=outside[0].chunk_ids[0])], False, True)
