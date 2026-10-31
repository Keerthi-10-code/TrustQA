import json

import pytest

from trustqa import llm as llm_mod
from trustqa.generation import (
    ABSTENTION_ANSWER,
    SYSTEM_PROMPT,
    answer_unverified,
    build_prompt,
    parse_generation,
    strip_invalid_markers,
)
from trustqa.llm import FakeLLM, LLMError
from trustqa.schemas import Chunk, RetrievedChunk, Status, WarningCode
from trustqa.verification.verify import generate_and_verify

Q = "What revenue did Helios report for 2025?"


def rc(cid, doc, name, page, text, rank):
    return RetrievedChunk(Chunk(doc, name, page, cid, text), score=1.0 - rank / 10, rank=rank)


PASSAGES = [
    rc("DOC01_P3_C1", "DOC01", "solar_report.pdf", 3, "Revenue grew to Rs. 1,03,450.50 crore in 2025.", 1),
    rc("DOC01_P3_C2", "DOC01", "solar_report.pdf", 3, "Operating margin was -2.5%.", 2),
    rc("DOC02_P2_C1", "DOC02", "river_survey.pdf", 2, "The Mettur dam was completed in 1934.", 3),
]


def reply(**kw):
    base = {"answer": "", "chunk_ids": [], "insufficient_evidence": False}
    base.update(kw)
    return json.dumps(base)


def test_prompt_contains_passages_ids_and_rules():
    fake = FakeLLM([reply(answer="Rs. 1,03,450.50 crore [DOC01_P3_C1].", chunk_ids=["DOC01_P3_C1"])])
    generate_and_verify(Q, PASSAGES, llm=fake)
    call = fake.calls[0]
    assert call["json"] is True and call["system"] == SYSTEM_PROMPT
    for p in PASSAGES:
        assert f"[{p.chunk.chunk_id}]" in call["prompt"] and p.chunk.text in call["prompt"]
    assert Q in call["prompt"]
    assert "ONLY" in SYSTEM_PROMPT and "insufficient_evidence" in SYSTEM_PROMPT


def test_grounded_answer_cites_pages_from_metadata():
    # The model claims page 99 and a different file name in prose and extra fields;
    # the citation must still come from chunk metadata.
    raw = json.dumps({
        "answer": "Revenue was Rs. 1,03,450.50 crore (annual.pdf, page 99) [DOC01_P3_C1].",
        "chunk_ids": ["DOC01_P3_C1"], "insufficient_evidence": False,
        "page_number": 99, "document_name": "annual.pdf"})
    r = generate_and_verify(Q, PASSAGES, llm=FakeLLM([raw]))
    base = answer_unverified(Q, PASSAGES, FakeLLM([raw]))
    for res in (r, base):
        assert res.status is Status.PARTIALLY_SUPPORTED  # one scripted reply: judge cannot run
        assert len(res.citations) == 1
        c = res.citations[0]
        assert (c.document_id, c.document_name, c.page_number, c.chunk_ids) == (
            "DOC01", "solar_report.pdf", 3, ["DOC01_P3_C1"])
    # Phase 3 item 1: the verified path flags the false source claims in the prose;
    # the frozen baseline (systems A/B) does not.
    assert [w.code for w in r.warnings if w.code is WarningCode.PAGE_MISMATCH] == [
        WarningCode.PAGE_MISMATCH] * 2
    assert r.answer == "Revenue was Rs. 1,03,450.50 crore [DOC01_P3_C1]."
    assert base.warnings == [] and "page 99" in base.answer


def test_citations_grouped_by_page_in_cite_order():
    fake = FakeLLM([reply(answer="Dam 1934 [DOC02_P2_C1]. Revenue up [DOC01_P3_C1, DOC01_P3_C2].",
                          chunk_ids=["DOC02_P2_C1", "DOC01_P3_C1", "DOC01_P3_C2"])])
    r = generate_and_verify(Q, PASSAGES, llm=fake)
    assert [(c.document_id, c.page_number, c.chunk_ids) for c in r.citations] == [
        ("DOC02", 2, ["DOC02_P2_C1"]), ("DOC01", 3, ["DOC01_P3_C1", "DOC01_P3_C2"])]


def test_inline_markers_count_when_list_is_empty():
    fake = FakeLLM([reply(answer="Margin was -2.5% [DOC01_P3_C2].", chunk_ids=[])])
    r = generate_and_verify(Q, PASSAGES, llm=fake)
    assert r.citations[0].chunk_ids == ["DOC01_P3_C2"]


def test_fabricated_chunk_id_rejected_and_removed_from_answer():
    fake = FakeLLM([reply(answer="Revenue was Rs. 1,03,450.50 crore [DOC01_P3_C1, DOC09_P1_C1].",
                          chunk_ids=["DOC01_P3_C1", "DOC09_P1_C1"])])
    r = generate_and_verify(Q, PASSAGES, llm=fake)
    assert [c.chunk_ids for c in r.citations] == [["DOC01_P3_C1"]]
    assert "DOC09_P1_C1" not in r.answer and "[DOC01_P3_C1]" in r.answer
    bad = [w for w in r.warnings if w.code is WarningCode.INVALID_CHUNK_ID]
    assert [w.chunk_id for w in bad] == ["DOC09_P1_C1"]


def test_passage_text_in_chunk_ids_is_ignored_with_warning():
    # Observed in the Phase 0 smoke test with qwen2.5:3b.
    fake = FakeLLM([reply(answer="Revenue was Rs. 1,03,450.50 crore.",
                          chunk_ids=["Revenue grew to Rs. 1,03,450.50 crore in 2025."])])
    r = generate_and_verify(Q, PASSAGES, llm=fake)
    assert r.status is Status.INSUFFICIENT_EVIDENCE  # no valid citation -> withheld
    assert r.answer == ABSTENTION_ANSWER and r.citations == []
    codes = {w.code for w in r.warnings}
    assert WarningCode.INVALID_CHUNK_ID in codes and WarningCode.UNSUPPORTED_CLAIM in codes


def test_ids_embedded_in_strings_are_salvaged():
    fake = FakeLLM([reply(answer="Dam completed in 1934.", chunk_ids=["[DOC02_P2_C1] passage"])])
    r = generate_and_verify(Q, PASSAGES, llm=fake)
    assert r.citations[0].chunk_ids == ["DOC02_P2_C1"]


def test_model_abstention_gives_no_guessed_answer():
    fake = FakeLLM([reply(answer="Probably Rs. 5 crore", chunk_ids=["DOC01_P3_C1"],
                          insufficient_evidence=True)])
    r = generate_and_verify("What is the CEO's salary?", PASSAGES, llm=fake)
    assert r.status is Status.INSUFFICIENT_EVIDENCE
    assert r.answer == ABSTENTION_ANSWER and r.citations == []


def test_no_passages_abstains_without_llm_call():
    fake = FakeLLM([])
    r = generate_and_verify(Q, [], llm=fake)
    assert r.status is Status.INSUFFICIENT_EVIDENCE and fake.calls == []


def test_non_object_json_is_an_explicit_error():
    with pytest.raises(LLMError):
        generate_and_verify(Q, PASSAGES, llm=FakeLLM(["[1, 2]"]))


def test_default_llm_honours_set_llm():
    fake = FakeLLM([reply(answer="1934 [DOC02_P2_C1].", chunk_ids=["DOC02_P2_C1"])])
    llm_mod.set_llm(fake)
    r = generate_and_verify("When was the dam completed?", PASSAGES)
    assert r.citations[0].page_number == 2
    assert fake.calls[0]["system"] == SYSTEM_PROMPT  # generation went through set_llm()


def test_strip_invalid_markers_leaves_other_brackets():
    s = strip_invalid_markers("As shown in [12], X [DOC09_P1_C1]. Y [DOC01_P3_C1].",
                              {"DOC01_P3_C1"})
    assert s == "As shown in [12], X. Y [DOC01_P3_C1]."


def test_parse_handles_string_flag_and_scalar_ids():
    out = parse_generation(json.dumps({"answer": "a", "chunk_ids": "DOC01_P3_C1",
                                       "insufficient_evidence": "false"}), PASSAGES)
    assert out.cited_ids == ["DOC01_P3_C1"] and not out.abstained


def test_build_prompt_lists_valid_ids():
    assert "Valid passage IDs: DOC01_P3_C1, DOC01_P3_C2, DOC02_P2_C1" in build_prompt(Q, PASSAGES)
