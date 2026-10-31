"""Phase 3 item 5: conflict detection across documents."""

from tests.fakes import routed
from trustqa.schemas import Chunk, WarningCode
from trustqa.verification.conflicts import check_conflicts, find_conflicts, parse_facts

A = Chunk("DOC01", "annual_2025.pdf", 4, "DOC01_P4_C1", "Helios revenue in FY2025 was Rs. 12,450 crore.")
B = Chunk("DOC02", "press_note.pdf", 1, "DOC02_P1_C1", "Helios reported FY2025 revenue of Rs. 13,100 crore.")
C = Chunk("DOC02", "press_note.pdf", 2, "DOC02_P2_C1", "Helios revenue in FY2024 was Rs. 11,000 crore.")
D = Chunk("DOC01", "annual_2025.pdf", 5, "DOC01_P5_C1", "Helios employs 4,200 people.")
CH = {c.chunk_id: c for c in (A, B, C, D)}


def fact(cid, value, period="FY2025", metric="revenue", entity="Helios", unit="Rs crore"):
    return {"chunk_id": cid, "entity": entity, "metric": metric, "period": period,
            "value": value, "unit": unit}


def facts(*fs):
    import json
    return parse_facts(json.dumps({"facts": list(fs)}), CH)


def test_conflicting_values_across_documents_flagged():
    cs = find_conflicts(facts(fact("DOC01_P4_C1", "12,450"), fact("DOC02_P1_C1", "13,100",
                                                                      metric="total revenue")),
                        CH, ["DOC01_P4_C1"])
    assert len(cs) == 1 and {f.chunk_id for f in cs[0].facts} == {"DOC01_P4_C1", "DOC02_P1_C1"}


def test_same_value_different_formatting_is_not_a_conflict():
    b = Chunk("DOC02", "x.pdf", 1, "DOC02_P1_C1", "Helios FY2025 revenue: Rs. 12450.0 crore.")
    ch = {**CH, b.chunk_id: b}
    import json
    fs = parse_facts(json.dumps({"facts": [fact("DOC01_P4_C1", "12,450"), fact("DOC02_P1_C1", "12450.0")]}), ch)
    assert find_conflicts(fs, ch, ["DOC01_P4_C1"]) == []


def test_different_periods_are_not_a_conflict():
    assert find_conflicts(facts(fact("DOC01_P4_C1", "12,450"), fact("DOC02_P2_C1", "11,000", period="FY2024")),
                          CH, ["DOC01_P4_C1"]) == []


def test_same_document_is_not_a_cross_document_conflict():
    a2 = Chunk("DOC01", "annual_2025.pdf", 9, "DOC01_P9_C1", "Helios FY2025 revenue was Rs. 12,900 crore.")
    ch = {**CH, a2.chunk_id: a2}
    import json
    fs = parse_facts(json.dumps({"facts": [fact("DOC01_P4_C1", "12,450"), fact("DOC01_P9_C1", "12,900")]}), ch)
    assert find_conflicts(fs, ch, ["DOC01_P4_C1"]) == []


def test_conflict_must_touch_a_cited_chunk():
    assert find_conflicts(facts(fact("DOC01_P4_C1", "12,450"), fact("DOC02_P1_C1", "13,100")),
                          CH, ["DOC01_P5_C1"]) == []


def test_hallucinated_fact_value_is_dropped():
    assert facts(fact("DOC02_P1_C1", "15,000")) == []          # not in the chunk
    assert facts(fact("DOC09_P1_C1", "12,450")) == []          # unknown chunk


def test_check_conflicts_warns_with_both_sources_from_metadata():
    fake = routed(conflict={"facts": [fact("DOC01_P4_C1", "12,450"), fact("DOC02_P1_C1", "13,100")]})
    rep = check_conflicts("What was Helios revenue in FY2025?", [A, B], ["DOC01_P4_C1"], fake)
    (w,) = rep.warnings
    assert w.code is WarningCode.CONFLICT and not rep.failed
    assert "annual_2025.pdf p.4" in w.message and "press_note.pdf p.1" in w.message


def test_single_document_skips_the_llm_call():
    fake = routed()
    rep = check_conflicts("q", [A, D], ["DOC01_P4_C1"], fake)
    assert rep.conflicts == [] and not rep.failed and fake.calls == []


def test_extraction_failure_is_reported_not_hidden():
    rep = check_conflicts("q", [A, B], ["DOC01_P4_C1"], routed())
    assert rep.failed and rep.warnings[0].code is WarningCode.CONFLICT
