"""Phase 3 item 6 - OCR-uncertainty warning. Plain Python.

A cited OCR chunk is uncertain when
- its mean word confidence is below `min_confidence` (default 90), or
- it contains OCR garbage near digits, which the mean confidence misses (failures.md F1:
  "β1 = 0.9, β2 = 0.98, ϵ = 10−9" read as "3; = 0.9, 52 = 0.98 « = 10~°" at confidence 92.5):
  two or more stray symbols (« » ~ ° ¢ § ¥ £ | �), or a digit followed by ; or : and "="
  ("3; =").
Calibration on the pilot scan (45 OCR chunks vs ground truth): the rule flags the F1 chunk
(DOC01_P7_C5) and both misread Table 2 chunks (P8_C1 "37.3"->"373", P8_C2), and no chunk
with a character error rate below 5%.

Effect: every uncertain cited chunk gets an OCR_UNCERTAINTY warning; a claim that states a
number and relies on an uncertain chunk caps the status at Partially Supported (the digits
themselves may be misread). Non-numeric claims are warned about but not capped.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from trustqa.schemas import Chunk, Warning, WarningCode
from trustqa.verification.claims import Claim

_STRAY_RE = re.compile(r"[«»~°¢§¥£|�]")
_GLUE_RE = re.compile(r"\d[;:]\s*=")


def ocr_uncertainty(chunk: Chunk, min_confidence: float = 90.0) -> str | None:
    """Reason a chunk's OCR text is unreliable, or None (also None for non-OCR chunks)."""
    if not chunk.ocr_used:
        return None
    reasons = []
    if chunk.ocr_confidence is not None and chunk.ocr_confidence < min_confidence:
        reasons.append(f"mean OCR confidence {chunk.ocr_confidence:.1f} < {min_confidence:g}")
    stray = _STRAY_RE.findall(chunk.text)
    if len(stray) >= 2:
        reasons.append(f"{len(stray)} stray OCR symbols ({''.join(dict.fromkeys(stray))})")
    if _GLUE_RE.search(chunk.text):
        reasons.append("garbled symbols before '=' (e.g. Greek letters misread as digits)")
    return "; ".join(reasons) or None


@dataclass
class OcrReport:
    warnings: list[Warning] = field(default_factory=list)
    uncertain_chunks: dict[str, str] = field(default_factory=dict)  # chunk_id -> reason
    capped_claim_ids: list[int] = field(default_factory=list)  # numeric claims on uncertain OCR


def check_ocr(claims: Sequence[Claim], chunks: dict[str, Chunk], numeric_claim_ids: set[int],
              min_confidence: float = 90.0) -> OcrReport:
    rep = OcrReport()
    cited = list(dict.fromkeys(cid for c in claims for cid in c.chunk_ids if cid in chunks))
    for cid in cited:
        reason = ocr_uncertainty(chunks[cid], min_confidence)
        if reason:
            rep.uncertain_chunks[cid] = reason
    for c in claims:
        if c.claim_id in numeric_claim_ids and any(cid in rep.uncertain_chunks for cid in c.chunk_ids):
            rep.capped_claim_ids.append(c.claim_id)
    for cid, reason in rep.uncertain_chunks.items():
        capped = [c.claim_id for c in claims
                  if c.claim_id in rep.capped_claim_ids and cid in c.chunk_ids]
        effect = (f"; numbers in claim(s) {', '.join(map(str, capped))} may be misread"
                  if capped else "")
        ch = chunks[cid]
        rep.warnings.append(Warning(
            WarningCode.OCR_UNCERTAINTY,
            f"{cid} ({ch.document_name} p.{ch.page_number}) is OCR text that may contain errors: "
            f"{reason}{effect}", chunk_id=cid))
    return rep
