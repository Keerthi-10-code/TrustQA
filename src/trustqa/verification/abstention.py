"""Phase 3 item 7 - abstention gate. Plain Python.

Two gates, both ending in INSUFFICIENT_EVIDENCE with no guessed answer:
1. Before generation: no passages, or the best retrieval score is below `min_score`. This
   skips the LLM call for questions the collection clearly cannot answer. The default 0.30 is
   for normalised MiniLM cosine scores: on the pilot set (10 answerable, 7 unanswerable
   questions) the lowest answerable top-1 score was 0.352 and 3/7 unanswerable questions
   scored below 0.30. Scores alone do not separate the classes (an unanswerable question
   scored 0.685), so this gate is a cheap first filter, not the decision; re-calibrate it
   for another embedding model.
2. After verification: no material claim is supported by its cited evidence (every
   material claim was judged not_supported or contradicted, or failed its numeric check).
   Claims the judge could not assess ("unverified") do not trigger this gate.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from trustqa.schemas import RetrievedChunk
from trustqa.verification.claims import Claim
from trustqa.verification.numeric import NumericResult
from trustqa.verification.support import SupportResult, Verdict


def retrieval_gate(passages: Sequence[RetrievedChunk], min_score: float) -> str | None:
    """Reason to abstain before generation, or None to go ahead."""
    if not passages:
        return "no passages were retrieved"
    best = max(p.score for p in passages)
    if best < min_score:
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
