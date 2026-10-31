"""Phase 3 items 6-8: OCR uncertainty, abstention gates, status rules."""

import pytest

from trustqa.schemas import Chunk, ExtractionMethod, RetrievedChunk, Status, WarningCode
from trustqa.verification.abstention import nothing_supported, retrieval_gate
from trustqa.verification.claims import Claim
from trustqa.verification.numeric import NumericResult
from trustqa.verification.ocr_check import check_ocr, ocr_uncertainty
from trustqa.verification.status import decide_status
from trustqa.verification.support import SupportResult, Verdict


def ocr_chunk(cid, text, conf):
    return Chunk("DOC01", "scan.pdf", 7, cid, text, extraction_method=ExtractionMethod.TESSERACT,
                 ocr_used=True, ocr_confidence=conf)


F1 = ocr_chunk("DOC01_P7_C5", "We used the Adam optimizer [20] with 3; = 0.9, 52 = 0.98 and « = 10~°.", 92.5)
CLEAN = ocr_chunk("DOC01_P7_C1", "The model was trained for 100,000 steps on 8 GPUs.", 95.0)
LOW = ocr_chunk("DOC01_P8_C2", "big model BLEU 28.4 and 41.8 at a cost of 373 hours", 80.5)
DIGITAL = Chunk("DOC02", "paper.pdf", 3, "DOC02_P3_C1", "Plain digital text 12.")


# --- item 6: OCR uncertainty ---

def test_low_confidence_chunk_is_uncertain():
    assert "80.5 < 90" in ocr_uncertainty(LOW)


def test_f1_garbled_symbols_flagged_despite_high_confidence():
    reason = ocr_uncertainty(F1)
    assert reason and "stray" in reason and "'='" in reason


def test_clean_ocr_and_digital_chunks_are_not_flagged():
    assert ocr_uncertainty(CLEAN) is None and ocr_uncertainty(DIGITAL) is None


def test_numeric_claim_on_uncertain_ocr_is_capped_and_warned():
    chunks = {c.chunk_id: c for c in (LOW, CLEAN)}
    claims = [Claim(1, "BLEU was 28.4.", ("DOC01_P8_C2",), True, True),
              Claim(2, "Training used GPUs.", ("DOC01_P7_C1",), True, True)]
    rep = check_ocr(claims, chunks, numeric_claim_ids={1})
    assert rep.capped_claim_ids == [1]
    assert [w.code for w in rep.warnings] == [WarningCode.OCR_UNCERTAINTY]
    assert "may be misread" in rep.warnings[0].message and "scan.pdf p.7" in rep.warnings[0].message


def test_non_numeric_claim_on_uncertain_ocr_warns_but_is_not_capped():
    rep = check_ocr([Claim(1, "Adam was used.", ("DOC01_P7_C5",), True, True)], {F1.chunk_id: F1},
                    numeric_claim_ids=set())
    assert rep.capped_claim_ids == [] and len(rep.warnings) == 1


# --- item 7: abstention gates ---

def rc(score):
    return RetrievedChunk(DIGITAL, score=score, rank=1)


@pytest.mark.parametrize("passages, abstain", [([], True), ([rc(0.2)], True), ([rc(0.35)], False)])
def test_retrieval_gate(passages, abstain):
    assert (retrieval_gate(passages, 0.30) is not None) is abstain


C1 = Claim(1, "a", ("X",), True, True)
C2 = Claim(2, "b", ("X",), True, True)
BOILER = Claim(3, "Based on the passages:", ("X",), True, False)


def sup(**v):
    return {int(k[1:]): SupportResult(int(k[1:]), Verdict(x)) for k, x in v.items()}


def test_nothing_supported():
    assert nothing_supported([C1, C2], sup(c1="not_supported", c2="contradicted"), None)
    assert not nothing_supported([C1, C2], sup(c1="supported", c2="not_supported"), None)
    assert not nothing_supported([C1], sup(c1="unverified"), None)  # unknown is not "no"
    assert nothing_supported([BOILER], sup(c3="supported"), None)    # nothing material


# --- item 8: status rules ---

def ok_num(*ids):
    return {i: NumericResult(i, checked=["1"]) for i in ids}


def test_verified_needs_everything():
    d = decide_status([C1, C2, BOILER], sup(c1="supported", c2="supported", c3="not_supported"),
                      ok_num(1, 2), citations_resolved=True)
    assert d.status is Status.VERIFIED and d.reasons == []


@pytest.mark.parametrize("kw, reason", [
    (dict(citations_resolved=False), "citation"),
    (dict(citations_resolved=True, conflict_found=True), "conflict"),
    (dict(citations_resolved=True, conflict_check_failed=True), "conflict check"),
    (dict(citations_resolved=True, ocr_capped_claim_ids=[1]), "OCR"),
])
def test_each_failed_condition_blocks_verified(kw, reason):
    d = decide_status([C1], sup(c1="supported"), ok_num(1), **kw)
    assert d.status is Status.PARTIALLY_SUPPORTED and any(reason in r for r in d.reasons)


def test_one_unsupported_claim_gives_partial():
    d = decide_status([C1, C2], sup(c1="supported", c2="not_supported"), ok_num(1, 2), True)
    assert d.status is Status.PARTIALLY_SUPPORTED and "claim 2 is not supported" in d.reasons


def test_numeric_mismatch_makes_claim_unsupported():
    num = {1: NumericResult(1, checked=["5"], unmatched=["5"])}
    assert decide_status([C1], sup(c1="supported"), num, True).status is Status.INSUFFICIENT_EVIDENCE
    d = decide_status([C1, C2], sup(c1="supported", c2="supported"), {**num, 2: NumericResult(2)}, True)
    assert d.status is Status.PARTIALLY_SUPPORTED


def test_unverified_claims_give_partial_not_insufficient():
    d = decide_status([C1], sup(c1="unverified"), ok_num(1), True)
    assert d.status is Status.PARTIALLY_SUPPORTED and "could not be checked" in d.reasons[0]


def test_all_unsupported_gives_insufficient():
    d = decide_status([C1, C2], sup(c1="not_supported", c2="contradicted"), ok_num(1, 2), True)
    assert d.status is Status.INSUFFICIENT_EVIDENCE


def test_switched_off_checks_cannot_verify():
    assert decide_status([C1], None, ok_num(1), True).status is Status.PARTIALLY_SUPPORTED
    assert decide_status([C1], sup(c1="supported"), None, True).status is Status.PARTIALLY_SUPPORTED
