"""Phase 3 item 1: citation registry + provenance check (FR-09, PRD 2.3 "citations resolve").

Failure cases this item must catch:
- fabricated chunk ID (not in the retrieved set)            -> rejected, INVALID_CHUNK_ID
- citation entry that is not an ID at all (passage text)     -> unresolved, INVALID_CHUNK_ID
- wrong page / wrong document stated in the answer prose     -> PAGE_MISMATCH
- chunk whose ID disagrees with its own metadata (corrupt)   -> not citable, PAGE_MISMATCH
Document name/page always come from the registry (trusted chunk metadata).
"""

import json

import pytest

from trustqa.llm import FakeLLM
from trustqa.schemas import Chunk, ContentType, RetrievedChunk, Status, TableData, WarningCode
from trustqa.verification.provenance import (
    CitationRegistry,
    check_provenance,
    find_source_mentions,
)
from trustqa.verification.verify import generate_and_verify

Q = "What revenue did Helios report for 2025?"


def rc(cid, doc, name, page, text, rank=1, **kw):
    return RetrievedChunk(Chunk(doc, name, page, cid, text, **kw), score=1.0 - rank / 10, rank=rank)


PASSAGES = [
    rc("DOC01_P3_C1", "DOC01", "solar_report.pdf", 3, "Revenue grew to Rs. 1,03,450.50 crore in 2025.", 1),
    rc("DOC01_P3_C2", "DOC01", "solar_report.pdf", 3, "Operating margin was -2.5%.", 2),
    rc("DOC02_P2_C1", "DOC02", "river_survey.pdf", 2, "The Mettur dam was completed in 1934.", 3),
    rc("DOC02_P5_C1", "DOC02", "river_survey.pdf", 5, "See page 12 of the 1950 report for flows.", 4),
]


def codes(ws):
    return [w.code for w in ws]


def reply(**kw):
    base = {"answer": "", "chunk_ids": [], "insufficient_evidence": False}
    base.update(kw)
    return json.dumps(base)


# --- registry -----------------------------------------------------------------------------

def test_registry_resolves_only_retrieved_ids():
    reg = CitationRegistry.from_passages(PASSAGES)
    assert reg.resolve("DOC01_P3_C1").document_name == "solar_report.pdf"
    assert reg.resolve("DOC09_P1_C1") is None
    assert "DOC02_P2_C1" in reg and "DOC09_P1_C1" not in reg
    assert reg.problems == {}


def test_registry_accepts_table_row_ids():
    td = TableData("T1", ["Model", "nDCG"], [["BM25", "0.656"]])
    p = rc("DOC02_P7_T1_R1", "DOC02", "beir.pdf", 7, "Table 1 | Model=BM25 | nDCG=0.656",
           content_type=ContentType.TABLE, table_data=td)
    reg = CitationRegistry.from_passages([p])
    assert reg.resolve("DOC02_P7_T1_R1") is not None and reg.problems == {}


@pytest.mark.parametrize("cid, doc, page, why", [
    ("DOC01_P4_C1", "DOC01", 3, "page"),      # ID says page 4, metadata says page 3
    ("DOC01_P3_C1", "DOC02", 3, "document"),  # ID says DOC01, metadata says DOC02
    ("chunk-17", "DOC01", 3, "format"),       # not a chunk ID at all
])
def test_registry_excludes_chunks_whose_id_disagrees_with_metadata(cid, doc, page, why):
    reg = CitationRegistry.from_passages([rc(cid, doc, "x.pdf", page, "text")])
    assert reg.resolve(cid) is None
    assert why in reg.problems[cid]


def test_registry_excludes_duplicate_ids_with_conflicting_metadata():
    dup = [rc("DOC01_P3_C1", "DOC01", "a.pdf", 3, "one"), rc("DOC01_P3_C1", "DOC01", "b.pdf", 3, "two")]
    reg = CitationRegistry.from_passages(dup)
    assert reg.resolve("DOC01_P3_C1") is None and "DOC01_P3_C1" in reg.problems


def test_registry_tolerates_identical_duplicates():
    same = [PASSAGES[0], PASSAGES[0]]
    reg = CitationRegistry.from_passages(same)
    assert reg.resolve("DOC01_P3_C1") is not None and reg.problems == {}


# --- source mentions in prose -------------------------------------------------------------

def test_find_source_mentions_binds_to_following_marker():
    ans = "Revenue was high (page 9) [DOC01_P3_C1]. The dam (river_survey.pdf, p. 2) [DOC02_P2_C1]."
    ms = find_source_mentions(ans)
    assert [(m.kind, m.value, m.scope) for m in ms] == [
        ("page", (9, 9), ["DOC01_P3_C1"]),
        ("document", "river_survey.pdf", ["DOC02_P2_C1"]),
        ("page", (2, 2), ["DOC02_P2_C1"]),
    ]


def test_find_source_mentions_ranges_and_unbound():
    ms = find_source_mentions("Covered on pages 3-5 and pp. 7–8.")
    assert [(m.value, m.scope) for m in ms] == [((3, 5), []), ((7, 8), [])]


# --- check_provenance ---------------------------------------------------------------------

def test_clean_citations_resolve_from_metadata():
    rep = check_provenance("Revenue was Rs. 1,03,450.50 crore [DOC01_P3_C1].", ["DOC01_P3_C1"], PASSAGES)
    assert rep.all_resolved and rep.warnings == [] and rep.rejected_ids == []
    c = rep.citations[0]
    assert (c.document_id, c.document_name, c.page_number, c.chunk_ids) == (
        "DOC01", "solar_report.pdf", 3, ["DOC01_P3_C1"])


def test_fabricated_chunk_id_rejected():
    rep = check_provenance("Revenue [DOC01_P3_C1].", ["DOC01_P3_C1", "DOC09_P1_C1"], PASSAGES)
    assert rep.accepted_ids == ["DOC01_P3_C1"] and rep.rejected_ids == ["DOC09_P1_C1"]
    assert codes(rep.warnings) == [WarningCode.INVALID_CHUNK_ID]
    assert rep.warnings[0].chunk_id == "DOC09_P1_C1"
    assert not rep.all_resolved
    assert [c.chunk_ids for c in rep.citations] == [["DOC01_P3_C1"]]


def test_upstream_unresolved_entries_make_citations_unresolved():
    # e.g. parse_generation already dropped a passage-text "ID" and warned about it
    rep = check_provenance("Revenue [DOC01_P3_C1].", ["DOC01_P3_C1"], PASSAGES, upstream_unresolved=1)
    assert not rep.all_resolved and rep.warnings == []


def test_wrong_page_in_prose_is_caught():
    rep = check_provenance("Revenue was Rs. 1,03,450.50 crore (page 99) [DOC01_P3_C1].",
                           ["DOC01_P3_C1"], PASSAGES)
    assert codes(rep.warnings) == [WarningCode.PAGE_MISMATCH]
    assert "99" in rep.warnings[0].message and "3" in rep.warnings[0].message
    assert rep.citations[0].page_number == 3  # still from metadata
    assert not rep.all_resolved


def test_wrong_page_scoped_to_the_sentence_citation():
    # page 2 is right for the dam chunk but wrong for the revenue chunk it is attached to
    ans = "Revenue grew (page 2) [DOC01_P3_C1]. The dam was built in 1934 [DOC02_P2_C1]."
    rep = check_provenance(ans, ["DOC01_P3_C1", "DOC02_P2_C1"], PASSAGES)
    assert codes(rep.warnings) == [WarningCode.PAGE_MISMATCH]


def test_wrong_document_in_prose_is_caught():
    rep = check_provenance("Revenue grew (annual_report.pdf) [DOC01_P3_C1].", ["DOC01_P3_C1"], PASSAGES)
    assert codes(rep.warnings) == [WarningCode.PAGE_MISMATCH]
    assert "annual_report.pdf" in rep.warnings[0].message


def test_correct_page_and_document_in_prose_pass():
    ans = "Revenue grew (Solar_Report.pdf, page 3) [DOC01_P3_C1]. Dam: 1934 (p. 2) [DOC02_P2_C1]."
    rep = check_provenance(ans, ["DOC01_P3_C1", "DOC02_P2_C1"], PASSAGES)
    assert rep.warnings == [] and rep.all_resolved


def test_page_range_containing_cited_page_passes():
    rep = check_provenance("Revenue grew (pages 2-4) [DOC01_P3_C1].", ["DOC01_P3_C1"], PASSAGES)
    assert rep.warnings == []


def test_unbound_mention_checked_against_all_cited_pages():
    ok = check_provenance("Per page 2, the dam [DOC02_P2_C1] ...", ["DOC02_P2_C1"], PASSAGES)
    assert ok.warnings == []
    bad = check_provenance("Revenue grew [DOC01_P3_C1]. More on page 8.", ["DOC01_P3_C1"], PASSAGES)
    assert codes(bad.warnings) == [WarningCode.PAGE_MISMATCH]


def test_page_reference_quoted_from_the_chunk_is_not_a_mismatch():
    rep = check_provenance("Flows are in page 12 of the 1950 report [DOC02_P5_C1].",
                           ["DOC02_P5_C1"], PASSAGES)
    assert rep.warnings == []


def test_false_source_parenthetical_removed_from_answer():
    rep = check_provenance("Revenue was Rs. 1,03,450.50 crore (source: annual.pdf, page 99) "
                           "[DOC01_P3_C1].", ["DOC01_P3_C1"], PASSAGES)
    assert rep.answer == "Revenue was Rs. 1,03,450.50 crore [DOC01_P3_C1]."
    assert codes(rep.warnings) == [WarningCode.PAGE_MISMATCH] * 2  # document and page
    assert all("removed from the answer" in w.message for w in rep.warnings)
    assert "document" in rep.warnings[0].message


def test_correct_source_parenthetical_kept():
    ans = "Revenue grew (solar_report.pdf, page 3) [DOC01_P3_C1]."
    assert check_provenance(ans, ["DOC01_P3_C1"], PASSAGES).answer == ans


def test_mixed_parenthetical_with_wrong_page_removed_whole():
    rep = check_provenance("Revenue grew (solar_report.pdf, p. 7) [DOC01_P3_C1].",
                           ["DOC01_P3_C1"], PASSAGES)
    assert rep.answer == "Revenue grew [DOC01_P3_C1]."
    assert codes(rep.warnings) == [WarningCode.PAGE_MISMATCH]


@pytest.mark.parametrize("ans", [
    "As stated on page 99, revenue grew [DOC01_P3_C1].",          # running prose
    "Revenue grew (see Table 2 on page 99) [DOC01_P3_C1].",       # parenthetical with content
])
def test_false_mention_in_prose_kept_but_warned(ans):
    rep = check_provenance(ans, ["DOC01_P3_C1"], PASSAGES)
    assert rep.answer == ans
    assert codes(rep.warnings) == [WarningCode.PAGE_MISMATCH]
    assert "removed" not in rep.warnings[0].message


def test_rejected_markers_stripped_from_report_answer():
    rep = check_provenance("Revenue [DOC01_P3_C1, DOC09_P1_C1].", ["DOC01_P3_C1", "DOC09_P1_C1"],
                           PASSAGES)
    assert rep.answer == "Revenue [DOC01_P3_C1]."


def test_citing_a_corrupt_chunk_is_rejected():
    bad = rc("DOC01_P4_C1", "DOC01", "solar_report.pdf", 3, "Revenue grew.")
    rep = check_provenance("Revenue grew [DOC01_P4_C1].", ["DOC01_P4_C1"], [bad])
    assert rep.accepted_ids == [] and rep.citations == [] and not rep.all_resolved
    assert codes(rep.warnings) == [WarningCode.PAGE_MISMATCH]


def test_no_citations_is_not_resolved():
    assert not check_provenance("Revenue grew.", [], PASSAGES).all_resolved


def test_citations_grouped_per_document_page_in_cite_order():
    rep = check_provenance("x", ["DOC02_P2_C1", "DOC01_P3_C1", "DOC01_P3_C2", "DOC01_P3_C1"], PASSAGES)
    assert [(c.document_id, c.page_number, c.chunk_ids) for c in rep.citations] == [
        ("DOC02", 2, ["DOC02_P2_C1"]), ("DOC01", 3, ["DOC01_P3_C1", "DOC01_P3_C2"])]


# --- wired into generate_and_verify -------------------------------------------------------

def test_generate_and_verify_flags_wrong_page_claim():
    fake = FakeLLM([reply(answer="Revenue was Rs. 1,03,450.50 crore (solar_report.pdf, page 99) "
                                 "[DOC01_P3_C1].", chunk_ids=["DOC01_P3_C1"],
                          page_number=99)])
    r = generate_and_verify(Q, PASSAGES, llm=fake)
    assert r.status is Status.PARTIALLY_SUPPORTED
    assert r.citations[0].page_number == 3
    assert WarningCode.PAGE_MISMATCH in codes(r.warnings)
    assert r.answer == "Revenue was Rs. 1,03,450.50 crore [DOC01_P3_C1]."


def test_generate_and_verify_rejects_fabricated_id_once():
    fake = FakeLLM([reply(answer="Revenue [DOC01_P3_C1, DOC09_P1_C1].",
                          chunk_ids=["DOC01_P3_C1", "DOC09_P1_C1"])])
    r = generate_and_verify(Q, PASSAGES, llm=fake)
    bad = [w for w in r.warnings if w.code is WarningCode.INVALID_CHUNK_ID]
    assert [w.chunk_id for w in bad] == ["DOC09_P1_C1"]  # not duplicated by the second check
    assert "DOC09_P1_C1" not in r.answer


def test_generate_and_verify_withholds_answer_citing_only_corrupt_chunk():
    bad = rc("DOC01_P4_C1", "DOC01", "solar_report.pdf", 3, "Revenue grew.")
    fake = FakeLLM([reply(answer="Revenue grew [DOC01_P4_C1].", chunk_ids=["DOC01_P4_C1"])])
    r = generate_and_verify(Q, [bad], llm=fake)
    assert r.status is Status.INSUFFICIENT_EVIDENCE and r.citations == []
    assert WarningCode.PAGE_MISMATCH in codes(r.warnings)
