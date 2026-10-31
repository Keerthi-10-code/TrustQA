"""Phase 3 item 8 - status rules (PRD 2.3). Plain Python.

- Verified: every material claim is supported by its cited evidence (support verdict
  "supported" with a validated quote, and its numbers/periods found or verified as
  calculations) AND every citation resolves to the correct document and page
  (ProvenanceReport.all_resolved) AND no cross-document conflict touches the answer AND no
  numeric claim rests on uncertain OCR text. Every check must actually have run.
- Insufficient Evidence: no material claim is supported (abstention gate 2), or an earlier
  gate abstained.
- Partially Supported: anything in between. The warnings explain the limitation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from trustqa.schemas import Status
from trustqa.verification.abstention import claim_supported, nothing_supported
from trustqa.verification.claims import Claim
from trustqa.verification.numeric import NumericResult
from trustqa.verification.support import SupportResult


@dataclass
class StatusDecision:
    status: Status
    reasons: list[str] = field(default_factory=list)  # why it is not Verified ([] if Verified)


def decide_status(
    claims: Sequence[Claim],
    support: Mapping[int, SupportResult] | None,
    numeric: Mapping[int, NumericResult] | None,
    citations_resolved: bool,
    conflict_found: bool = False,
    conflict_check_failed: bool = False,
    ocr_capped_claim_ids: Sequence[int] = (),
) -> StatusDecision:
    """`support` / `numeric` are None when that check is switched off (ablations)."""
    if nothing_supported(claims, support, numeric):
        return StatusDecision(Status.INSUFFICIENT_EVIDENCE,
                              ["no material claim is supported by the cited evidence"])
    reasons: list[str] = []
    material = [c for c in claims if c.material]
    if support is None:
        reasons.append("support check is switched off")
    if numeric is None:
        reasons.append("numeric check is switched off")
    for c in material:
        state = claim_supported(c, support, numeric)
        if state is False:
            reasons.append(f"claim {c.claim_id} is not supported")
        elif state is None and support is not None:
            reasons.append(f"claim {c.claim_id} could not be checked")
    if not citations_resolved:
        reasons.append("not every citation resolves to the correct document and page")
    if conflict_found:
        reasons.append("sources conflict")
    if conflict_check_failed:
        reasons.append("conflict check could not run")
    if ocr_capped_claim_ids:
        reasons.append("numeric claim(s) "
                       f"{', '.join(map(str, ocr_capped_claim_ids))} rely on uncertain OCR text")
    return StatusDecision(Status.PARTIALLY_SUPPORTED if reasons else Status.VERIFIED, reasons)
