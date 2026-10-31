"""Phase 6 harness: metrics on tiny hand-made examples, question validation, LLM cache,
judge parsing, per-question scoring and the report."""

import json

import pytest

from eval import metrics
from eval.judge import judge_answer
from eval.questions import category_counts, is_short_answer, validate
from eval.run_all import score_row
from eval.systems import SYSTEMS, CachingLLM, config_for
from trustqa.llm import FakeLLM, LLMError
from trustqa.schemas import Citation, Result, Status, Warning, WarningCode

G = [{"pages": [["a.pdf", 3]]}, {"pages": [["b.pdf", 7], ["b.pdf", 8]]}]


# ---------------------------------------------------------------- answers

def test_exact_match_and_f1():
    assert metrics.exact_match("The BART-large.", "bart-large") == 1.0
    assert metrics.exact_match("BART", ["BART-large", "BART"]) == 1.0
    assert metrics.exact_match("BART-base", "BART-large") == 0.0
    # pred "dpr retriever" vs ref "dprs retriever ..." -> handmade overlap
    assert metrics.token_f1("the big dog", "a big dog") == 1.0
    assert metrics.token_f1("big cat", "big dog") == pytest.approx(0.5)
    assert metrics.token_f1("", "x") == 0.0


def test_numeric_scores_value_unit_period():
    s = metrics.numeric_scores("Revenue was Rs. 1,03,450.50 crore in FY2024.", value="103450.50 crore",
                               unit="crore", period="FY2024")
    assert s == {"value": True, "unit": True, "period": True}
    s = metrics.numeric_scores("Revenue was Rs. 1,03,450.50 lakh in FY2023.", value="103450.50 crore",
                               unit="crore", period="FY2024")
    assert s == {"value": False, "unit": False, "period": False}
    assert metrics.numeric_scores("Margin -2.5%", value="-2.5%")["value"] is True
    assert metrics.numeric_scores("Margin 2.5%", value="-2.5%")["value"] is False
    # no explicit label: every number of the reference must appear
    assert metrics.numeric_scores("28.4 BLEU", reference="28.4 BLEU.")["value"] is True
    assert metrics.numeric_scores("27.3", reference="28.4 BLEU.")["value"] is False
    assert metrics.numeric_scores("BART", reference="BART-large.") == \
        {"value": None, "unit": None, "period": None}


# ---------------------------------------------------------------- retrieval / provenance

def test_retrieval_scores_by_hand():
    ranked = [("x.pdf", 1), ("b.pdf", 8), ("a.pdf", 3), ("x.pdf", 2), ("y.pdf", 1)]
    s = metrics.retrieval_scores(ranked, G, ks=(1, 3, 5))
    assert s["recall@1"] == 0 and s["recall@3"] == 1.0 and s["all_groups@3"] == 1.0
    assert s["precision@3"] == pytest.approx(2 / 3) and s["precision@5"] == pytest.approx(2 / 5)
    assert s["rr"] == 0.5
    s = metrics.retrieval_scores([("b.pdf", 7)], G, ks=(1,))
    assert s["recall@1"] == 0.5 and s["all_groups@1"] == 0.0 and s["rr"] == 1.0
    assert metrics.retrieval_scores([], G, ks=(1,))["rr"] == 0.0
    with pytest.raises(ValueError):
        metrics.retrieval_scores([], [])


def test_provenance_by_hand():
    assert metrics.provenance([("a.pdf", 3), ("x.pdf", 1)], G) == \
        {"provenance_hit": 1.0, "citation_precision": 0.5}
    assert metrics.provenance([("x.pdf", 1)], G)["provenance_hit"] == 0.0
    assert metrics.provenance([], G) == {"provenance_hit": None, "citation_precision": None}


def test_support_scores_by_hand():
    rows = [{"status": "Verified", "correct": True}, {"status": "Verified", "correct": False},
            {"status": "Partially Supported", "correct": False},
            {"status": "Insufficient Evidence", "correct": False}]
    s = metrics.support_scores(rows)
    assert s["verified_precision"] == 0.5 and s["verified_n"] == 2
    assert s["incorrect_not_verified"] == 0.5 and s["incorrect_n"] == 2


def test_cer_wer():
    s = metrics.cer_wer("the  cat sat", "the cat sat")
    assert s == {"cer": 0.0, "wer": 0.0}
    s = metrics.cer_wer("abcd", "abxd")
    assert s["cer"] == 0.25 and s["wer"] == 1.0


# ---------------------------------------------------------------- scoring rows

def _q(**kw):
    q = {"id": "T1", "category": "table", "question": "BM25 on TREC-COVID?",
         "reference_answer": "0.656", "evidence_groups": [{"pages": [["beir.pdf", 7]]}]}
    q.update(kw)
    return q


def _res(answer, status, pages=(), warnings=()):
    cits = [Citation("DOC01", d, p, [f"DOC01_P{p}_C1"]) for d, p in pages]
    return Result("q", answer, status, citations=cits, warnings=list(warnings))


def test_score_row_numbers_override_judge_and_abstention():
    r = score_row(_q(), _res("0.656", Status.VERIFIED, [("beir.pdf", 7)]), [("beir.pdf", 7)], 1.0,
                  "correct")
    assert r["correct"] and r["provenance"]["provenance_hit"] == 1.0 and r["exact_match"] == 1.0
    r = score_row(_q(numeric={"value": "0.656"}), _res("0.665", Status.VERIFIED, [("beir.pdf", 7)]),
                  [("beir.pdf", 7)], 1.0, "correct")
    assert r["correct"] is False  # judge said correct, labelled value disagrees
    r = score_row(_q(reference_answer="21M (100-word chunks)"), _res("21M", Status.VERIFIED, [("beir.pdf", 7)]),
                  [("beir.pdf", 7)], 1.0, "correct")
    assert r["correct"] is True and r["numeric"]["value"] is False  # no label: judge decides
    r = score_row(_q(), _res("Insufficient", Status.INSUFFICIENT_EVIDENCE), [("x.pdf", 1)], 1.0)
    assert r["correct"] is False and r["numeric"]["value"] is False and r["answer"] == ""
    u = score_row({"id": "U1", "category": "unanswerable", "question": "CEO?",
                   "expected_abstention": True},
                  _res("Insufficient", Status.INSUFFICIENT_EVIDENCE), [], 1.0)
    assert u["correct"] is True and not u["answerable"]
    c = score_row(_q(expected_conflict=True, category="conflicting"),
                  _res("0.656", Status.PARTIALLY_SUPPORTED, [("beir.pdf", 7)],
                       [Warning(WarningCode.CONFLICT, "two values")]), [], 1.0, "correct")
    assert c["conflict_flagged"] and c["expected_conflict"]


def test_aggregate_rates():
    rows = [
        score_row(_q(id="1"), _res("0.656", Status.VERIFIED, [("beir.pdf", 7)]), [("beir.pdf", 7)], 2, "correct"),
        score_row(_q(id="2"), _res("0.5", Status.PARTIALLY_SUPPORTED, [("x.pdf", 1)]), [("x.pdf", 1)], 4, "incorrect"),
        score_row(_q(id="3"), _res("", Status.INSUFFICIENT_EVIDENCE), [("beir.pdf", 7)], 6),
        score_row({"id": "4", "category": "unanswerable", "question": "?", "expected_abstention": True},
                  _res("x", Status.PARTIALLY_SUPPORTED, [("x.pdf", 1)]), [], 8),
    ]
    a = metrics.aggregate(rows, ks=(5,))
    assert a["accuracy"] == pytest.approx(1 / 3) and a["answered_rate"] == pytest.approx(2 / 3)
    assert a["unsupported_answer_rate"] == 0.5 and a["unanswerable_answered"] == 1
    assert a["correct_abstention_rate"] == 0.0 and a["over_abstention_rate"] == pytest.approx(1 / 3)
    assert a["recall@5"] == pytest.approx(2 / 3) and a["verified_precision"] == 1.0
    assert a["conflict_recall"] is None and a["conflict_n"] == 0  # no labelled conflicts: n/a, not 0
    assert a["latency_mean_s"] == 5.0
    rows[0]["latency_valid"] = False  # served from the LLM cache: not a latency measurement
    assert metrics.aggregate(rows, ks=(5,))["latency_n"] == 3


# ---------------------------------------------------------------- questions

def test_validate_question_set():
    good = [_q(id="T1"),
            {"id": "U1", "category": "unanswerable", "question": "?", "expected_abstention": True}]
    assert validate(good, {"beir.pdf": 24}) == ([], [])
    bad = [_q(id="T1"), _q(id="T1"), _q(id="X", category="trivia"),
           _q(id="P", evidence_groups=[{"pages": [["beir.pdf", 0]]}]),
           _q(id="R", evidence_groups=[{"pages": [["beir.pdf", 99]]}]),
           _q(id="D", evidence_groups=[{"pages": [["missing.pdf", 1]]}]),
           _q(id="N", reference_answer=""),
           _q(id="C", category="conflicting"),
           {"id": "U", "category": "unanswerable", "question": "?"}]
    errors, _ = validate(bad, {"beir.pdf": 24})
    text = "\n".join(errors)
    for needle in ["T1: duplicate id", "X: unknown category", "P: evidence group 1: page entry",
                   "R: evidence group 1: beir.pdf has 24 pages", "D: evidence group 1: document",
                   "N: missing reference_answer", "C: conflicting questions need",
                   "U: unanswerable questions need"]:
        assert needle in text
    _, warnings = validate([_q(id="M", category="multi_doc")])
    assert "fewer than 2 documents" in warnings[0]


def test_category_counts_and_answer_type():
    counts = category_counts([_q(), _q(category="direct")])
    assert counts["table"] == (1, 10) and counts["conflicting"] == (0, 5)
    assert is_short_answer(_q()) and not is_short_answer(_q(reference_answer="one two three four five six seven"))
    assert not is_short_answer(_q(answer_type="long"))


# ---------------------------------------------------------------- systems / cache / judge

def test_systems_differ_in_one_switch_and_pin_temperature(monkeypatch):
    monkeypatch.setenv("VERIFY_SUPPORT", "0")  # a stray .env setting must not leak into C
    c = config_for("C")
    assert c.verify_support and c.llm_temperature == 0.0 and c.system_name == "C-trustqa"
    assert config_for("C-no_support").verify_support is False
    assert config_for("C-no_tables").flatten_tables and config_for("C-no_tables").verify
    assert config_for("C-no_ocr").ocr_enabled is False
    assert config_for("A").system_name == "A-baseline" and config_for("B").system_name == "B-ocr-plain-rag"
    assert set(SYSTEMS) >= {"A", "B", "C", "C-no_tables", "C-no_support", "C-no_numeric", "C-no_ocr"}


def test_caching_llm_hits_by_prompt_and_model(tmp_path):
    inner = FakeLLM(lambda p, s, j: f"out:{p}")
    llm = CachingLLM(inner, tmp_path / "c.sqlite", {"model": "m1"})
    assert llm.complete("hi") == "out:hi" and llm.complete("hi") == "out:hi"
    assert (llm.hits, llm.misses, len(inner.calls)) == (1, 1, 1)
    llm.complete("hi", system="other")
    assert len(inner.calls) == 2  # system prompt is part of the key
    other = CachingLLM(inner, tmp_path / "c.sqlite", {"model": "m2"})
    other.complete("hi")
    assert len(inner.calls) == 3  # another model never reuses m1's output
    again = CachingLLM(FakeLLM([]), tmp_path / "c.sqlite", {"model": "m1"})
    assert again.complete("hi") == "out:hi"  # persisted on disk

    def boom(p, s, j):
        raise LLMError("down")
    failing = CachingLLM(FakeLLM(boom), tmp_path / "c.sqlite", {"model": "m3"})
    with pytest.raises(LLMError):
        failing.complete("x")
    assert failing.misses == 0  # errors are not cached
    fresh = CachingLLM(inner, tmp_path / "c.sqlite", {"model": "m1"}, read=False)
    fresh.complete("hi")
    assert len(inner.calls) == 4 and fresh.hits == 0  # --fresh: model called despite the cache


def test_judge_parses_strict_json_only():
    ok = FakeLLM([json.dumps({"verdict": "Correct", "reason": "matches"})])
    assert judge_answer("q", "0.656", "0.656", ok) == ("correct", "matches")
    bad = FakeLLM([json.dumps({"verdict": "mostly right"})])
    assert judge_answer("q", "r", "a", bad)[0] is None
    assert judge_answer("q", "r", "a", FakeLLM(["not json"]))[0] is None


def test_report_builds_tables_from_runs(tmp_path):
    from eval import error_analysis, report

    rows = [score_row(_q(id="1"), _res("0.656", Status.VERIFIED, [("beir.pdf", 7)]), [("beir.pdf", 7)], 1, "correct"),
            score_row(_q(id="2"), _res("0.9", Status.VERIFIED, [("beir.pdf", 7)]), [("beir.pdf", 7)], 1, "incorrect")]
    runs = tmp_path / "runs"
    runs.mkdir()
    for name in ("C", "C-no_support"):
        (runs / f"{name}__r0.json").write_text(json.dumps(
            {"system": name, "description": "", "repeat": 0, "complete": True, "rows": rows}))
    s = report.build(tmp_path, tmp_path / "plots", ["1", "2"])
    assert s["systems"]["C"]["metrics"]["accuracy"] == 0.5
    assert s["ablation_deltas_vs_C"]["C-no_support"]["accuracy"] == 0.0
    assert (tmp_path / "summary.csv").is_file() and (tmp_path / "plots" / "ablation_deltas.png").is_file()
    md = error_analysis.write(tmp_path, s).read_text(encoding="utf-8")
    assert "[verified_but_incorrect]" in md and "**2**" in md
    assert (tmp_path / "spot_check.csv").is_file()
