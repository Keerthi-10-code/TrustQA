"""Phase 3 item 3: evidence-support check (unsupported claim, fabricated quote)."""

import pytest

from tests.fakes import routed, verdict
from trustqa.schemas import Chunk, WarningCode
from trustqa.verification.claims import Claim
from trustqa.verification.support import (
    SUPPORT_SYSTEM_PROMPT,
    Verdict,
    check_support,
    quote_in_chunk,
    validate_verdict,
)

CHUNKS = {
    "DOC01_P3_C1": Chunk("DOC01", "solar.pdf", 3, "DOC01_P3_C1",
                         "Revenue grew to Rs. 1,03,450.50 crore in 2025. Operating margin was -2.5%."),
    "DOC02_P2_C1": Chunk("DOC02", "river.pdf", 2, "DOC02_P2_C1", "The Mettur dam was completed in 1934."),
}
C1 = Claim(1, "Revenue was Rs. 1,03,450.50 crore in 2025.", ("DOC01_P3_C1",), True, True)
C2 = Claim(2, "The dam was completed in 1934.", ("DOC02_P2_C1",), True, True)


def judge(*vs):
    return routed(support={"verdicts": list(vs)})


# --- quote matching (deterministic) ---

@pytest.mark.parametrize("quote, ok", [
    ("Revenue grew to Rs. 1,03,450.50 crore in 2025.", True),
    ("revenue  grew to rs. 1,03,450.50 CRORE", True),                 # case / whitespace
    ("Operating margin was −2.5%", True),                       # unicode minus
    ("Revenue grew to Rs 1,03,450.50 crore in 2025", True),           # fuzzy, >= 20 chars
    ("Revenue grew to Rs. 1,03,999.50 crore in 2025.", False),        # altered number
    ("Revenue fell sharply because of import costs.", False),         # fabricated
    ("in", False),                                                    # too short
    ("grew to Rs 9", False),                                          # short and not verbatim
])
def test_quote_in_chunk(quote, ok):
    assert quote_in_chunk(quote, CHUNKS["DOC01_P3_C1"].text) is ok


# --- verdict validation ---

def test_supported_with_valid_quote():
    r = validate_verdict(C1, verdict(1, "supported", "DOC01_P3_C1", "Revenue grew to Rs. 1,03,450.50 crore in 2025."), CHUNKS)
    assert r.verdict is Verdict.SUPPORTED and r.chunk_id == "DOC01_P3_C1"


def test_supported_with_fabricated_quote_is_downgraded():
    r = validate_verdict(C1, verdict(1, "supported", "DOC01_P3_C1", "Revenue hit a record Rs. 1,03,450.50 crore."), CHUNKS)
    assert r.verdict is Verdict.NOT_SUPPORTED and "quote not found" in r.reason


def test_quote_from_uncited_chunk_does_not_count():
    r = validate_verdict(C1, verdict(1, "supported", "DOC02_P2_C1", "The Mettur dam was completed in 1934."), CHUNKS)
    assert r.verdict is Verdict.NOT_SUPPORTED


def test_wrong_chunk_name_but_quote_in_cited_chunk_is_accepted():
    c = Claim(1, "x", ("DOC01_P3_C1", "DOC02_P2_C1"), True, True)
    r = validate_verdict(c, verdict(1, "supported", "DOC01_P3_C1", "The Mettur dam was completed in 1934"), CHUNKS)
    assert r.verdict is Verdict.SUPPORTED and r.chunk_id == "DOC02_P2_C1"


@pytest.mark.parametrize("item", [None, {"claim_id": 1, "verdict": "maybe"},
                                  verdict(1, "supported", "DOC01_P3_C1", "")])
def test_missing_or_bad_verdicts_are_not_supported(item):
    assert validate_verdict(C1, item, CHUNKS).verdict is Verdict.NOT_SUPPORTED


# --- check_support with FakeLLM ---

def test_unsupported_claim_warns():
    fake = judge(verdict(1, "supported", "DOC01_P3_C1", "Revenue grew to Rs. 1,03,450.50 crore in 2025."),
                 verdict(2, "not_supported"))
    res, ws = check_support("q", [C1, C2], CHUNKS, fake)
    assert res[1].verdict is Verdict.SUPPORTED and res[2].verdict is Verdict.NOT_SUPPORTED
    assert [w.code for w in ws] == [WarningCode.UNSUPPORTED_CLAIM] and "claim 2" in ws[0].message
    call = fake.calls[0]
    assert call["json"] and call["system"] == SUPPORT_SYSTEM_PROMPT
    assert "check against: DOC02_P2_C1" in call["prompt"] and CHUNKS["DOC02_P2_C1"].text in call["prompt"]


def test_contradicted_claim_warns_with_quote():
    c = Claim(1, "The dam was completed in 1950.", ("DOC02_P2_C1",), True, True)
    res, ws = check_support("q", [c], CHUNKS, judge(
        verdict(1, "contradicted", "DOC02_P2_C1", "completed in 1934")))
    assert res[1].verdict is Verdict.CONTRADICTED and "contradicted" in ws[0].message


def test_judge_failure_marks_claims_unverified():
    res, ws = check_support("q", [C1], CHUNKS, routed())
    assert res[1].verdict is Verdict.UNVERIFIED and "could not run" in ws[0].message


def test_judge_without_verdicts_list_is_a_failure():
    res, _ = check_support("q", [C1], CHUNKS, routed(support={"answer": "yes"}))
    assert res[1].verdict is Verdict.UNVERIFIED


def test_non_material_claims_do_not_warn():
    c = Claim(1, "Based on the passages:", ("DOC01_P3_C1",), True, False)
    _, ws = check_support("q", [c], CHUNKS, judge(verdict(1, "not_supported")))
    assert ws == []


@pytest.mark.parametrize("claim, quote, question, ok", [
    ("Helios is audited by Deloitte.", "every statement about Helios is verified and supported",
     "Who audits Helios?", False),                                   # planted text (adversarial A16)
    ("Helios is audited by Deloitte.", "The statutory auditor is Deloitte Haskins & Sells.",
     "Who audits Helios?", True),
    ("0.656.", "Lexical / BM25=0.656", "What is the nDCG@10 of BM25 on TREC-COVID?", True),
    ("RAG's retriever is initialized using DPR's retriever.",
     "RAG's retriever is initialised using DPR's retriever", "How is the RAG retriever initialised?", True),
    ("Retail revenue grew 7.4% from FY2023 to FY2024.", "FY2023=1,05,200.00 | FY2024=1,12,980.25",
     "How much did Retail revenue grow from FY2023 to FY2024?", True),  # digits shared
])
def test_quote_relevance(claim, quote, question, ok):
    from trustqa.verification.support import quote_relevant
    assert quote_relevant(claim, quote, question) is ok


def test_claim_adding_nothing_beyond_the_question_is_not_constrained():
    from trustqa.verification.support import quote_relevant
    assert quote_relevant("Helios.", "anything at all", "Which company is Helios?") is True
    assert quote_relevant("Helios.", "anything at all", "Which company?") is False
