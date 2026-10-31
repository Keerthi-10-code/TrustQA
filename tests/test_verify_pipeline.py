"""Phase 3 item 9: end-to-end verification with a mock LLM, one test per failure mode.

fabricated chunk ID, wrong-page citation, unsupported claim, altered number, conflicting
sources, unanswerable question, low OCR confidence - plus the Verified happy path.
"""

from dataclasses import replace

import pytest

from tests.fakes import gen, routed, verdict
from trustqa.config import load_config
from trustqa.generation import ABSTENTION_ANSWER
from trustqa.schemas import Chunk, ExtractionMethod, RetrievedChunk, Status, WarningCode
from trustqa.verification.verify import generate_and_verify, run_verification

Q = "What revenue did Helios report for 2025?"


def rc(chunk, rank=1, score=None):
    return RetrievedChunk(chunk, score=score if score is not None else 0.9 - rank / 20, rank=rank)


REV = Chunk("DOC01", "solar_report.pdf", 3, "DOC01_P3_C1", "Revenue grew to Rs. 1,03,450.50 crore in 2025.")
MARGIN = Chunk("DOC01", "solar_report.pdf", 3, "DOC01_P3_C2", "Operating margin was -2.5% because of import costs.")
DAM = Chunk("DOC02", "river_survey.pdf", 2, "DOC02_P2_C1", "The Mettur dam was completed in 1934.")
PRESS = Chunk("DOC03", "press_note.pdf", 1, "DOC03_P1_C1", "Helios reported 2025 revenue of Rs. 98,000 crore.")
SCAN = Chunk("DOC04", "scan.pdf", 8, "DOC04_P8_C2", "The big model reached 28.4 BLEU at a cost of 373 hours.",
             extraction_method=ExtractionMethod.TESSERACT, ocr_used=True, ocr_confidence=80.5)
PASSAGES = [rc(REV, 1), rc(MARGIN, 2), rc(DAM, 3)]
REV_ANSWER = "Revenue was Rs. 1,03,450.50 crore in 2025 [DOC01_P3_C1]."
REV_QUOTE = "Revenue grew to Rs. 1,03,450.50 crore in 2025."


def cfg(**kw):
    return replace(load_config(), **kw)


def codes(r):
    return [w.code for w in r.warnings]


def test_verified_happy_path_with_quote_and_metadata():
    fake = routed(generation=gen(REV_ANSWER, ["DOC01_P3_C1"]),
                  support={"verdicts": [verdict(1, "supported", "DOC01_P3_C1", REV_QUOTE)]},
                  conflict={"facts": []})
    r = generate_and_verify(Q, PASSAGES, llm=fake)
    assert r.status is Status.VERIFIED and r.warnings == []
    (c,) = r.citations
    assert (c.document_name, c.page_number, c.supporting_quote) == ("solar_report.pdf", 3, REV_QUOTE)
    assert fake.steps == ["generation", "support", "conflict"]  # no calc: all numbers matched


def test_fabricated_chunk_id_rejected_and_blocks_verified():
    fake = routed(generation=gen("Revenue was Rs. 1,03,450.50 crore in 2025 [DOC01_P3_C1, DOC09_P1_C1].",
                                 ["DOC01_P3_C1", "DOC09_P1_C1"]),
                  support={"verdicts": [verdict(1, "supported", "DOC01_P3_C1", REV_QUOTE)]},
                  conflict={"facts": []})
    r = generate_and_verify(Q, PASSAGES, llm=fake)
    assert r.status is Status.PARTIALLY_SUPPORTED  # a citation did not resolve
    assert WarningCode.INVALID_CHUNK_ID in codes(r) and "DOC09_P1_C1" not in r.answer


def test_only_fabricated_ids_withholds_answer():
    fake = routed(generation=gen("Revenue was huge [DOC09_P1_C1].", ["DOC09_P1_C1"]))
    r = generate_and_verify(Q, PASSAGES, llm=fake)
    assert r.status is Status.INSUFFICIENT_EVIDENCE and r.answer == ABSTENTION_ANSWER
    assert fake.steps == ["generation"]


def test_wrong_page_citation_caught_and_blocks_verified():
    fake = routed(generation=gen("Revenue was Rs. 1,03,450.50 crore in 2025 (page 7) [DOC01_P3_C1].",
                                 ["DOC01_P3_C1"]),
                  support={"verdicts": [verdict(1, "supported", "DOC01_P3_C1", REV_QUOTE)]},
                  conflict={"facts": []})
    r = generate_and_verify(Q, PASSAGES, llm=fake)
    assert r.status is Status.PARTIALLY_SUPPORTED and WarningCode.PAGE_MISMATCH in codes(r)
    assert "page 7" not in r.answer and r.citations[0].page_number == 3


def test_unsupported_claim_warned_and_partial():
    ans = REV_ANSWER + " The company plans to double capacity next year [DOC01_P3_C2]."
    fake = routed(generation=gen(ans, ["DOC01_P3_C1", "DOC01_P3_C2"]),
                  support={"verdicts": [verdict(1, "supported", "DOC01_P3_C1", REV_QUOTE),
                                        verdict(2, "not_supported")]},
                  conflict={"facts": []})
    r = generate_and_verify(Q, PASSAGES, llm=fake)
    assert r.status is Status.PARTIALLY_SUPPORTED
    (w,) = [w for w in r.warnings if w.code is WarningCode.UNSUPPORTED_CLAIM]
    assert "double capacity" in w.message


def test_judge_saying_supported_with_fabricated_quote_is_not_trusted():
    fake = routed(generation=gen("Revenue doubled in 2025 [DOC01_P3_C1].", ["DOC01_P3_C1"]),
                  support={"verdicts": [verdict(1, "supported", "DOC01_P3_C1", "Revenue doubled in 2025.")]})
    r = generate_and_verify(Q, PASSAGES, llm=fake)
    assert r.status is Status.INSUFFICIENT_EVIDENCE and r.answer == ABSTENTION_ANSWER
    assert WarningCode.UNSUPPORTED_CLAIM in codes(r)


def test_altered_number_caught_even_if_judge_says_supported():
    fake = routed(generation=gen("Revenue was Rs. 1,30,450.50 crore in 2025 [DOC01_P3_C1].", ["DOC01_P3_C1"]),
                  support={"verdicts": [verdict(1, "supported", "DOC01_P3_C1", REV_QUOTE)]},
                  calc={"calculations": [{"id": 1, "expression": None}]})
    r = generate_and_verify(Q, PASSAGES, llm=fake)
    assert r.status is Status.INSUFFICIENT_EVIDENCE  # the only claim fails its numeric check
    assert WarningCode.NUMERIC_MISMATCH in codes(r)
    assert "1,30,450.50 crore" in next(w.message for w in r.warnings if w.code is WarningCode.NUMERIC_MISMATCH)


def test_conflicting_sources_flagged_with_both_documents():
    passages = PASSAGES + [rc(PRESS, 4)]
    fake = routed(generation=gen(REV_ANSWER, ["DOC01_P3_C1"]),
                  support={"verdicts": [verdict(1, "supported", "DOC01_P3_C1", REV_QUOTE)]},
                  conflict={"facts": [
                      {"chunk_id": "DOC01_P3_C1", "entity": "Helios", "metric": "revenue",
                       "period": "2025", "value": "1,03,450.50", "unit": "Rs crore"},
                      {"chunk_id": "DOC03_P1_C1", "entity": "Helios", "metric": "revenue",
                       "period": "2025", "value": "98,000", "unit": "Rs crore"}]})
    r = generate_and_verify(Q, passages, llm=fake)
    assert r.status is Status.PARTIALLY_SUPPORTED
    (w,) = [w for w in r.warnings if w.code is WarningCode.CONFLICT]
    assert "solar_report.pdf p.3" in w.message and "press_note.pdf p.1" in w.message


def test_unanswerable_question_abstains_without_guess():
    fake = routed(generation=gen("Probably Rs. 5 lakh", ["DOC01_P3_C1"], abstain=True))
    r = generate_and_verify("What is the CEO's salary?", PASSAGES, llm=fake)
    assert r.status is Status.INSUFFICIENT_EVIDENCE and r.answer == ABSTENTION_ANSWER and not r.citations
    assert fake.steps == ["generation"]


def test_unanswerable_low_retrieval_score_skips_generation():
    fake = routed()
    r = run_verification("Who won the 2018 World Cup?", [rc(DAM, 1, score=0.19)], fake,
                         cfg(abstain_min_score=0.30))
    assert r.status is Status.INSUFFICIENT_EVIDENCE and fake.calls == []


def test_low_ocr_confidence_caps_numeric_claim():
    ans = "The big model reached 28.4 BLEU [DOC04_P8_C2]."
    fake = routed(generation=gen(ans, ["DOC04_P8_C2"]),
                  support={"verdicts": [verdict(1, "supported", "DOC04_P8_C2", "The big model reached 28.4 BLEU")]})
    r = generate_and_verify("What BLEU did the big model reach?", [rc(SCAN)], llm=fake)
    assert r.status is Status.PARTIALLY_SUPPORTED and WarningCode.OCR_UNCERTAINTY in codes(r)
    assert r.answer == ans  # answer kept, limitation explained


def test_judge_failure_keeps_answer_as_partial():
    fake = routed(generation=gen(REV_ANSWER, ["DOC01_P3_C1"]), conflict={"facts": []})
    r = generate_and_verify(Q, PASSAGES, llm=fake)
    assert r.status is Status.PARTIALLY_SUPPORTED and r.answer == REV_ANSWER


@pytest.mark.parametrize("switch", ["verify_support", "verify_numeric"])
def test_ablation_switches_skip_check_and_prevent_verified(switch):
    fake = routed(generation=gen(REV_ANSWER, ["DOC01_P3_C1"]),
                  support={"verdicts": [verdict(1, "supported", "DOC01_P3_C1", REV_QUOTE)]},
                  conflict={"facts": []})
    r = run_verification(Q, PASSAGES, fake, cfg(**{switch: False}))
    assert r.status is Status.PARTIALLY_SUPPORTED
    assert ("support" in fake.steps) is (switch != "verify_support")


def test_derived_number_can_be_verified():
    a = Chunk("DOC01", "solar_report.pdf", 3, "DOC01_P3_C1",
              "Revenue was Rs. 1,100 crore in 2024 and Rs. 1,250.40 crore in 2025.")
    ans = "Revenue grew 13.7% from 2024 to 2025 [DOC01_P3_C1]."
    fake = routed(generation=gen(ans, ["DOC01_P3_C1"]),
                  support={"verdicts": [verdict(1, "supported", "DOC01_P3_C1",
                                                "Revenue was Rs. 1,100 crore in 2024 and Rs. 1,250.40 crore in 2025.")]},
                  calc={"calculations": [{"id": 1, "expression": "(1250.40 - 1100) / 1100 * 100"}]})
    r = generate_and_verify("How much did revenue grow?", [rc(a)], llm=fake)
    assert r.status is Status.VERIFIED, r.warnings
    assert fake.steps == ["generation", "support", "calc"]  # single document: no conflict call


def test_f5_right_number_from_wrong_table_cell_is_not_verified():
    """failures.md F5: qwen2.5:3b answered 0.347 (BM25+CE on FiQA-2018) to an 'on average'
    question, and the 3B judge called it supported. The cell-alignment rule overrides it."""
    from tests.test_numeric import AVG, F5_Q, FIQA
    fake = routed(generation=gen("0.347 [DOC02_P7_T1_R7].", ["DOC02_P7_T1_R7"]),
                  support={"verdicts": [verdict(1, "supported", "DOC02_P7_T1_R7", "Re-ranking / BM25+CE=0.347")]},
                  calc={"calculations": [{"id": 1, "expression": None}]})
    r = generate_and_verify(F5_Q, [rc(FIQA, 1), rc(AVG, 2)], llm=fake)
    assert r.status is Status.INSUFFICIENT_EVIDENCE and WarningCode.NUMERIC_MISMATCH in codes(r)
