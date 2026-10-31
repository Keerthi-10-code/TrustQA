"""Document scope (F13), answer type (F10), numeric quote fallback (F14), entity-scoped
retrieval (F12), list-aware prompt - and the Phase 3 adversarial suite with all of them on."""

import dataclasses

import pytest

from tests.fakes import gen, routed, verdict
from tests.test_adversarial import CASES
from trustqa.config import load_config
from trustqa.generation import ABSTENTION_ANSWER, SYSTEM_PROMPT_V2
from trustqa.schemas import Chunk, RetrievedChunk, Status, WarningCode
from trustqa.scope import DocumentScope, question_names, squash, title_region
from trustqa.verification.answer_checks import answer_has_number, expects_number
from trustqa.verification.verify import run_verification

ALL_ON = dict(verify_scope=True, verify_answers_question=True, support_number_fallback=True,
              citation_repair=True, support_retry_missing=True,
              generation_prompt="grounded-v2", abstain_min_score=0.0)

RAG_T = Chunk("DOC01", "rag_lewis2020.pdf", 1, "DOC01_P1_C1",
              "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks Patrick Lewis Abstract "
              "Large pre-trained language models store knowledge. We call this RAG.")
RAG_GPU = Chunk("DOC01", "rag_lewis2020.pdf", 5, "DOC01_P5_C2",
                "We train with mixed precision, distributing training across 8, 32GB NVIDIA V100 GPUs.")
SR_T = Chunk("DOC02", "selfrag_asai2023.pdf", 1, "DOC02_P1_C1",
             "SELF-RAG: LEARNING TO RETRIEVE, GENERATE, AND CRITIQUE Akari Asai Abstract Retrieval-"
             "Augmented Generation (RAG) helps. We introduce Self-RAG.")
SR_GPU = Chunk("DOC02", "selfrag_asai2023.pdf", 18, "DOC02_P18_C1",
               "We train our models on 4 Nvidia A100 with 80GB memory for 3 epochs.")
SR_POP = Chunk("DOC02", "selfrag_asai2023.pdf", 7, "DOC02_P7_C1",
               "For PopQA, we use the long-tail subset, consisting of 1,399 rare entity queries. "
               "We report accuracy.")
BEIR_T = Chunk("DOC03", "beir_thakur2021.pdf", 1, "DOC03_P1_C1",
               "BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation Abstract We evaluate BM25.")
SCOPE = DocumentScope.from_chunks([RAG_T, RAG_GPU, SR_T, SR_GPU, SR_POP, BEIR_T])
GPU_Q = "What GPUs were used to train the Self-RAG models?"


def rc(*chunks):
    return [RetrievedChunk(c, score=0.9 - i / 20, rank=i + 1) for i, c in enumerate(chunks)]


def run(question, passages, replies, **over):
    cfg = load_config(**{**ALL_ON, **over})
    return run_verification(question, passages, routed(**replies), cfg, SCOPE)


# ---------------------------------------------------------------- scope primitives

def test_names_titles_and_matching():
    assert question_names(GPU_Q) == ["GPUs", "Self-RAG"]
    assert question_names("What does RAG's retriever use?") == ["RAG"]
    assert question_names("What nDCG@10 does BM25 get on Touche-2020?") == ["nDCG@10", "BM25", "Touche-2020"]
    assert squash("Touché-2020") == squash("Touche-2020") == "touche2020"
    assert title_region("My Title by A. Author Abstract The body") == "My Title by A. Author "
    # 'Self-RAG' identifies only the Self-RAG paper (title), 'RAG' only the RAG paper (file
    # name): token-aligned, so 'rag' does not match inside 'self-rag'; 'GPUs' (body) identifies none
    assert SCOPE.identifying(GPU_Q) == {"Self-RAG": {"DOC02"}}
    assert SCOPE.identifying("How is RAG's retriever initialised?") == {"RAG": {"DOC01"}}
    assert SCOPE.allowed_documents("list the core technologies") is None
    assert SCOPE.missing("What did Infosys report?") == ["Infosys"]
    assert SCOPE.missing("What was beta2?") == []  # lowercase symbols are not names
    assert SCOPE.missing("What did the Lewis paper find?") == []  # mentioned in a chunk
    from trustqa.schemas import ContentType, TableData
    row = Chunk("DOC09", "annual.pdf", 7, "DOC09_P7_T2_R1", "Table 2 | Segment=Retail | FY2024=1,12,980.25",
                content_type=ContentType.TABLE,
                table_data=TableData("DOC09_P7_T2", ["Segment", "FY2024"], [["Retail", "1,12,980.25"]]))
    assert DocumentScope.from_chunks([row]).missing("What was Retail revenue in FY2024?") == []


def test_answer_type_rules():
    assert expects_number("What exact match score does RAG-Sequence get on NQ?")
    assert expects_number("How many queries does PopQA contain?")
    assert not expects_number("Which retriever does Self-RAG use?")
    assert not answer_has_number("What exact match score does it get?", "We consider four datasets.")
    assert answer_has_number("How many layers?", "six identical layers")
    assert answer_has_number("What BLEU score?", "28.4 BLEU")


# ---------------------------------------------------------------- verification with the checks on

def test_f13_answer_from_the_wrong_document_is_withheld():
    r = run(GPU_Q, rc(RAG_GPU, SR_GPU), dict(
        generation=gen("The models were trained on 8, 32GB NVIDIA V100 GPUs [DOC01_P5_C2].", ["DOC01_P5_C2"]),
        support={"verdicts": [verdict(1, "supported", "DOC01_P5_C2", "8, 32GB NVIDIA V100 GPUs")]},
        conflict={"facts": []}))
    assert r.status is Status.INSUFFICIENT_EVIDENCE and r.answer == ABSTENTION_ANSWER
    assert any("Self-RAG" in w.message and "rag_lewis2020.pdf" in w.message for w in r.warnings)
    # without the check the same scripted run is Verified (the Phase 4/6 failure)
    off = run(GPU_Q, rc(RAG_GPU, SR_GPU), dict(
        generation=gen("The models were trained on 8, 32GB NVIDIA V100 GPUs [DOC01_P5_C2].", ["DOC01_P5_C2"]),
        support={"verdicts": [verdict(1, "supported", "DOC01_P5_C2", "8, 32GB NVIDIA V100 GPUs")]},
        conflict={"facts": []}), verify_scope=False)
    assert off.status is Status.VERIFIED


def test_f13_right_document_still_verified_and_mixed_is_capped():
    ok = run(GPU_Q, rc(SR_GPU, RAG_GPU), dict(
        generation=gen("They used 4 Nvidia A100 with 80GB memory [DOC02_P18_C1].", ["DOC02_P18_C1"]),
        support={"verdicts": [verdict(1, "supported", "DOC02_P18_C1", "4 Nvidia A100 with 80GB memory")]},
        conflict={"facts": []}))
    assert ok.status is Status.VERIFIED
    mixed = run(GPU_Q, rc(SR_GPU, RAG_GPU), dict(
        generation=gen("They used 4 Nvidia A100 with 80GB memory [DOC02_P18_C1]. "
                       "Training was distributed across GPUs [DOC01_P5_C2].", ["DOC02_P18_C1", "DOC01_P5_C2"]),
        support={"verdicts": [verdict(1, "supported", "DOC02_P18_C1", "4 Nvidia A100 with 80GB memory"),
                              verdict(2, "supported", "DOC01_P5_C2", "distributing training across 8, 32GB NVIDIA V100 GPUs")]},
        conflict={"facts": []}))
    assert mixed.status is Status.PARTIALLY_SUPPORTED
    assert any("part of the answer cites rag_lewis2020.pdf" in w.message for w in mixed.warnings)


def test_question_about_something_no_document_mentions_is_withheld():
    r = run("What revenue did Infosys report?", rc(SR_POP), dict(
        generation=gen("It reported 1,399 [DOC02_P7_C1].", ["DOC02_P7_C1"]),
        support={"verdicts": [verdict(1, "supported", "DOC02_P7_C1", "1,399 rare entity queries")]}))
    assert r.status is Status.INSUFFICIENT_EVIDENCE
    assert any("Infosys" in w.message for w in r.warnings)


def test_f10_quantity_question_without_a_number_is_not_verified():
    q = "What accuracy does Self-RAG report on PopQA?"
    r = run(q, rc(SR_POP), dict(
        generation=gen("For PopQA, we use the long-tail subset [DOC02_P7_C1].", ["DOC02_P7_C1"]),
        support={"verdicts": [verdict(1, "supported", "DOC02_P7_C1", "For PopQA, we use the long-tail subset")]}))
    assert r.status is Status.PARTIALLY_SUPPORTED
    assert any("states no number" in w.message for w in r.warnings)


def test_f10_judge_says_answer_does_not_answer_the_question():
    q = "Which subset of PopQA does Self-RAG use?"
    replies = dict(
        generation=gen("Self-RAG reports accuracy [DOC02_P7_C1].", ["DOC02_P7_C1"]),
        support={"answers_question": False,
                 "verdicts": [verdict(1, "supported", "DOC02_P7_C1", "We report accuracy")]})
    assert run(q, rc(SR_POP), replies).status is Status.VERIFIED  # judge field off by default
    r = run(q, rc(SR_POP), dict(
        generation=gen("Self-RAG reports accuracy [DOC02_P7_C1].", ["DOC02_P7_C1"]),
        support={"answers_question": False,
                 "verdicts": [verdict(1, "supported", "DOC02_P7_C1", "We report accuracy")]}),
        judge_answers_question=True)
    assert r.status is Status.PARTIALLY_SUPPORTED
    assert any("may not give what the question asks" in w.message for w in r.warnings)


def test_f14_correct_number_with_a_quote_that_omits_it():
    q = "How many queries does Self-RAG's PopQA long-tail subset contain?"
    replies = dict(generation=gen("1,399 [DOC02_P7_C1]", ["DOC02_P7_C1"]),
                   support={"verdicts": [verdict(1, "supported", "DOC02_P7_C1",
                                                 "For PopQA, we use the long-tail subset")]})
    assert run(q, rc(SR_POP), replies, support_number_fallback=False).status is Status.INSUFFICIENT_EVIDENCE
    r = run(q, rc(SR_POP), replies)
    assert r.status is Status.VERIFIED and r.answer.startswith("1,399")
    assert "1,399 rare entity queries" in r.citations[0].supporting_quote
    # a number the chunk does not state never gets a fallback quote
    wrong = dict(generation=gen("1,400 [DOC02_P7_C1]", ["DOC02_P7_C1"]), support=replies["support"])
    assert run(q, rc(SR_POP), wrong).status is Status.INSUFFICIENT_EVIDENCE


def test_citation_repair_uses_the_neighbouring_chunk_on_the_same_page():
    c1 = Chunk("DOC02", "selfrag_asai2023.pdf", 7, "DOC02_P7_C1", "For PopQA, we use the long-tail subset.")
    c2 = Chunk("DOC02", "selfrag_asai2023.pdf", 7, "DOC02_P7_C2", "It consists of 1,399 rare entity queries.")
    other = Chunk("DOC02", "selfrag_asai2023.pdf", 9, "DOC02_P9_C1", "Another table lists 1,400 queries.")
    q = "How many queries does Self-RAG's PopQA long-tail subset contain?"
    replies = dict(generation=gen("1,399 [DOC02_P7_C1]", ["DOC02_P7_C1"]),
                   support={"verdicts": [verdict(1, "supported", "DOC02_P7_C2", "1,399 rare entity queries")]})
    assert run(q, rc(c1, c2, other), replies, citation_repair=False).status is Status.INSUFFICIENT_EVIDENCE
    r = run(q, rc(c1, c2, other), replies)
    assert r.status is Status.VERIFIED and r.citations[0].page_number == 7
    assert r.citations[0].chunk_ids == ["DOC02_P7_C1", "DOC02_P7_C2"]
    # a number only found on another page is never attached
    wrong = dict(generation=gen("1,400 [DOC02_P7_C1]", ["DOC02_P7_C1"]),
                 support={"verdicts": [verdict(1, "supported", "DOC02_P9_C1", "1,400 queries")]})
    assert run(q, rc(c1, c2, other), wrong).status is Status.INSUFFICIENT_EVIDENCE


def test_judge_is_asked_again_for_skipped_claims():
    import json as _json
    from trustqa.llm import FakeLLM
    from trustqa.generation import SYSTEM_PROMPT_V2 as GEN
    from trustqa.verification.support import SUPPORT_SYSTEM_PROMPT as SUP

    calls = []

    def fn(prompt, system, json_mode):
        if system == GEN:
            return _json.dumps(gen("They used 4 Nvidia A100 [DOC02_P18_C1]. Training ran for 3 epochs "
                                   "[DOC02_P18_C1].", ["DOC02_P18_C1"]))
        calls.append(prompt)
        if len(calls) == 1:  # first judge call answers claim 1 only
            return _json.dumps({"verdicts": [verdict(1, "supported", "DOC02_P18_C1", "4 Nvidia A100")]})
        return _json.dumps({"verdicts": [verdict(2, "supported", "DOC02_P18_C1", "for 3 epochs")]})

    q = "What GPUs and how many epochs were used to train the Self-RAG models?"
    cfg = load_config(**{**ALL_ON, "verify_conflicts": False})
    r = run_verification(q, rc(SR_GPU), FakeLLM(fn), cfg, SCOPE)
    assert r.status is Status.VERIFIED and len(calls) == 2
    assert "Claim 1 " not in calls[1] and "Claim 2 " in calls[1]  # only the skipped claim


def test_v2_prompt_asks_for_every_item_and_v1_is_unchanged():
    from trustqa.generation import SYSTEM_PROMPT
    assert "give every item the passages state" in SYSTEM_PROMPT_V2
    assert "5. Keep the answer concise." in SYSTEM_PROMPT and "every item" not in SYSTEM_PROMPT
    b = load_config(generation_prompt="grounded-v2", baseline_b=True)
    assert b.generation_prompt == "grounded-v1"  # baselines keep the frozen prompt


@pytest.mark.parametrize("case", CASES, ids=[c.name.split()[0] for c in CASES])
def test_adversarial_cases_hold_with_all_new_checks_on(case):
    scope = DocumentScope.from_chunks([p.chunk for p in case.passages])
    r = run_verification(case.question, case.passages, routed(**case.replies),
                         load_config(**ALL_ON), scope)
    codes = {w.code for w in r.warnings}
    # the new checks may only make a result stricter, never more trusting
    order = [Status.INSUFFICIENT_EVIDENCE, Status.PARTIALLY_SUPPORTED, Status.VERIFIED]
    assert order.index(r.status) <= order.index(case.status), (case.name, r.status, [w.message for w in r.warnings])
    if r.status is case.status:
        assert case.must_warn <= codes, case.name
    for s in case.hidden:
        assert s not in r.answer, case.name


def test_entity_scoped_retrieval_brings_in_the_named_paper(fake_embedder):
    from trustqa.retrieval.index import SearchIndex
    from trustqa.retrieval.search import retrieve_with_config

    filler = [dataclasses.replace(RAG_GPU, chunk_id=f"DOC01_P6_C{i}", page_number=6,
                                  text=f"GPUs were used for training run {i} with NVIDIA GPUs.")
              for i in range(1, 7)]
    chunks = [RAG_T, RAG_GPU, *filler, SR_T, SR_GPU, SR_POP, BEIR_T]
    idx = SearchIndex.from_chunks(chunks, fake_embedder, "fake")
    base = load_config(top_k=3)
    plain = [p.chunk.document_id for p in retrieve_with_config(GPU_Q, 3, idx, base)]
    scoped = retrieve_with_config(GPU_Q, 3, idx, load_config(top_k=3, retrieval_entity_scope=True))
    assert "DOC02" not in plain  # RAG's GPU sentences crowd out the Self-RAG paper
    assert any(p.chunk.chunk_id == "DOC02_P18_C1" for p in scoped)
    # deeper retrieval keeps the same first top_k (eval retrieves 10 and slices 5)
    deep = retrieve_with_config(GPU_Q, 6, idx, load_config(top_k=3, retrieval_entity_scope=True))
    assert [p.chunk.chunk_id for p in deep[:3]] == [p.chunk.chunk_id for p in scoped]
