"""Phase 3 item 7 - abstention gate. Plain Python.

Two gates, both ending in INSUFFICIENT_EVIDENCE with no guessed answer:
1. Before generation: no passages, or the best retrieval score is below `min_score`. This
   skips the LLM call for questions the collection clearly cannot answer. The default 0.30 is
   for normalised MiniLM cosine scores: on the pilot set (10 answerable, 7 unanswerable
   questions) the lowest answerable top-1 score was 0.352 and 3/7 unanswerable questions
   scored below 0.30. Scores alone do not separate the classes (an unanswerable question
   scored 0.685), so this gate is a cheap first filter, not the decision; re-calibrate it
   for another embedding model.
   With `lexical_pass` (config.gate_lexical_pass) a question below the threshold still goes
   ahead when one retrieved passage - its text plus section path - contains every content
   word of the question (at least 2 words). Short generic questions such as "list the core
   technologies" score low on cosine although the passage under the heading "Core
   Technologies" is retrieved. Passing the gate only lets generation and verification run;
   an answer is still shown only if its claims are supported.
2. After verification: no material claim is supported by its cited evidence (every
   material claim was judged not_supported or contradicted, or failed its numeric check).
   Claims the judge could not assess ("unverified") do not trigger this gate.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

from trustqa.retrieval.index import index_text
from trustqa.schemas import RetrievedChunk
from trustqa.verification.claims import Claim
from trustqa.verification.numeric import NumericResult
from trustqa.verification.support import SupportResult, Verdict


_STOPWORDS = frozenset(
    "a an the of on in for to and or is are was were be been what which who whom whose how "
    "many much does do did list give name show describe explain tell me all about with by "
    "from as at that this these those their its it there between than into section say says "
    "said summarize summarise summary document paper mention mentions".split())


def content_terms(question: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", question.lower())
            if len(w) > 1 and w not in _STOPWORDS}


def _term_in(term: str, words: set[str]) -> bool:
    """Exact, or a one-letter typo / swap for words of 6+ letters ("technolgies")."""
    if term in words:
        return True
    if len(term) < 6:
        return False
    from rapidfuzz.distance import DamerauLevenshtein

    return any(abs(len(w) - len(term)) <= 1 and DamerauLevenshtein.distance(term, w) <= 1
               for w in words)


def lexical_cover(question: str, passages: Sequence[RetrievedChunk]) -> bool:
    """True if some passage (text + section) contains all of >= 2 question content words
    (tolerating one typo in long words)."""
    terms = content_terms(question)
    if len(terms) < 2:
        return False
    for p in passages:
        words = set(re.findall(r"[a-z0-9]+", index_text(p.chunk).lower()))
        if all(_term_in(t, words) for t in terms):
            return True
    return False


def retrieval_gate(passages: Sequence[RetrievedChunk], min_score: float,
                   question: str | None = None, lexical_pass: bool = False) -> str | None:
    """Reason to abstain before generation, or None to go ahead."""
    if not passages:
        return "no passages were retrieved"
    best = max(p.score for p in passages)
    if best < min_score:
        if lexical_pass and question and lexical_cover(question, passages):
            return None
        return f"best retrieval score {best:.3f} is below the abstention threshold {min_score:g}"
    return None


def claim_supported(claim: Claim, support: Mapping[int, SupportResult] | None,
                    numeric: Mapping[int, NumericResult] | None) -> bool | None:
    """True/False once judged; None if support was not assessed (check off or failed)."""
    if numeric is not None and claim.claim_id in numeric and not numeric[claim.claim_id].ok:
        return False
    if support is None or claim.claim_id not in support:
        return None
    v = support[claim.claim_id].verdict
    if v is Verdict.UNVERIFIED:
        return None
    return v is Verdict.SUPPORTED


def nothing_supported(claims: Sequence[Claim], support: Mapping[int, SupportResult] | None,
                      numeric: Mapping[int, NumericResult] | None) -> bool:
    """True if there are no material claims, or every one was judged and none is supported."""
    material = [c for c in claims if c.material]
    if not material:
        return True
    states = [claim_supported(c, support, numeric) for c in material]
    return all(s is False for s in states)
