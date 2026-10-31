"""Phase 3 item 4: numeric checks (altered number, unit, period, calculations)."""

from decimal import Decimal
from fractions import Fraction

import pytest

from tests.fakes import routed
from trustqa.schemas import Chunk, ContentType, TableData, WarningCode
from trustqa.verification.claims import Claim
from trustqa.verification.numeric import (
    ExpressionError,
    check_calculation,
    check_numbers,
    evaluate_expression,
    extract_numbers,
    extract_periods,
    number_in_evidence,
)

TEXT = Chunk("DOC01", "solar.pdf", 3, "DOC01_P3_C1",
             "Revenue grew to Rs. 1,03,450.50 crore in 2025. Operating margin was -2.5%. "
             "The board approved a dividend of 12 percent.")
TD = TableData("T2", ["Segment", "FY2023", "FY2024", "FY2025"],
               [["Retail", "1,05,200.00", "1,12,980.25", "-1,250.40"]], units="Rs crore",
               caption="Table 2: Segment revenue (Rs crore)")
TABLE = Chunk("DOC03", "annual.pdf", 7, "DOC03_P7_T2_R1",
              "Table T2 (Segment revenue, Rs crore) | Segment=Retail | FY2023=1,05,200.00 | "
              "FY2024=1,12,980.25 | FY2025=-1,250.40", content_type=ContentType.TABLE, table_data=TD)
BEIR = Chunk("DOC02", "beir.pdf", 7, "DOC02_P7_T1_R20",
             "Table 2 | Model=BM25+CE | nDCG@10 avg=0.413 | vs BM25=+ 11% | cost 2.3 · 10^19")
CHUNKS = {c.chunk_id: c for c in (TEXT, TABLE, BEIR)}


def num(s):
    (n,) = extract_numbers(s)
    return n


def claim(text, *ids):
    return Claim(1, text, ids, True, True)


# --- extraction / normalisation ---

@pytest.mark.parametrize("text, value, pct, neg", [
    ("Rs. 1,03,450.50 crore", Decimal("1034505000000"), False, False),
    ("1,234,567", Decimal("1234567"), False, False),
    ("−2.5%", Decimal("-2.5"), True, True),
    ("+ 11%", Decimal("11"), True, False),
    ("12 percent", Decimal("12"), True, False),
    ("2.3 · 10^19", Decimal("2.3e19"), False, False),
    ("10^20", Decimal("1e20"), False, False),
    ("1.03 lakh crore", Decimal("1.03e12"), False, False),
    ("5 million", Decimal("5e6"), False, False),
])
def test_extract_numbers_normalises(text, value, pct, neg):
    n = num(text)
    assert n.scaled == value and n.percent is pct and n.negative is neg


def test_identifiers_and_reference_labels_are_skipped():
    assert extract_numbers("BM25 on nDCG@10 in Table 2 on page 3, COVID-19, T5, FY2024") == []


def test_periods_normalised():
    assert extract_periods("FY2024 and FY 2023-24 and Q3 FY24") == {"FY24", "Q3FY24", "Q3"}


# --- matching ---

@pytest.mark.parametrize("text, chunk, ok", [
    ("Rs. 1,03,450.50 crore", TEXT, True),
    ("1,03,450.50", TEXT, True),                # unit omitted
    ("about 1.03 lakh crore", TEXT, True),      # rounded, re-scaled
    ("Rs. 1,03,450.5 crore", TEXT, True),
    ("Rs. 1,03,540.50 crore", TEXT, False),     # transposed digits
    ("-2.5%", TEXT, True),
    ("2.5%", TEXT, True),                       # sign in words ("fell by 2.5%")
    ("-12 percent", TEXT, False),               # sign flipped
    ("12%", TEXT, True),                        # % vs percent
    ("12 crore", TEXT, False),                  # scale word the chunk never uses... with a bare 12
    ("1,12,980.25 crore", TABLE, True),         # bare table cell, caption says Rs crore
    ("-1,250.40", TABLE, True),
    ("1,250.40", TABLE, True),
    ("11%", BEIR, True),
    ("0.347", BEIR, False),                     # failures.md F5: value not in the row
    ("0.41", BEIR, True),                       # rounding with 2 significant digits
    ("0.4", BEIR, False),                       # 1 significant digit: too coarse
])
def test_number_in_evidence(text, chunk, ok):
    assert number_in_evidence(num(text), [chunk]) is ok


def test_altered_number_is_caught():
    res, ws = check_numbers([claim("Revenue was Rs. 1,03,540.50 crore in 2025.", "DOC01_P3_C1")], CHUNKS)
    assert not res[1].ok and res[1].unmatched == ["1,03,540.50 crore"]
    assert [w.code for w in ws] == [WarningCode.NUMERIC_MISMATCH]


def test_correct_numbers_pass_and_mark_claim_numeric():
    res, ws = check_numbers([claim("Revenue was Rs. 1,03,450.50 crore in 2025.", "DOC01_P3_C1")], CHUNKS)
    assert res[1].ok and res[1].numeric and ws == []


def test_number_from_another_chunk_is_not_evidence():
    res, _ = check_numbers([claim("Margin was 11%.", "DOC01_P3_C1")], CHUNKS)
    assert not res[1].ok


def test_missing_period_is_caught():
    res, ws = check_numbers([claim("Retail revenue was 1,12,980.25 crore in FY2026.", "DOC03_P7_T2_R1")], CHUNKS)
    assert res[1].missing_periods == ["FY26"] and "period" in ws[0].message
    ok, _ = check_numbers([claim("Retail revenue was 1,12,980.25 crore in FY24.", "DOC03_P7_T2_R1")], CHUNKS)
    assert ok[1].ok


def test_non_numeric_claim():
    res, ws = check_numbers([claim("Revenue grew.", "DOC01_P3_C1")], CHUNKS)
    assert res[1].ok and not res[1].numeric and ws == []


# --- calculations ---

def test_evaluate_expression_exact_and_safe():
    assert evaluate_expression("(1250.4 - 1100) / 1100 * 100") == Fraction("150.4") / 1100 * 100
    assert evaluate_expression("0.1 + 0.2") == evaluate_expression("0.3")
    for bad in ["__import__('os')", "2 ** 10", "1 / 0", "", "abs(-1)"]:
        with pytest.raises(ExpressionError):
            evaluate_expression(bad)


def test_check_calculation():
    n = num("7.4%")  # (1,12,980.25 - 1,05,200.00) / 1,05,200.00 = 7.396%
    assert check_calculation(n, "(112980.25 - 105200.00) / 105200.00 * 100", [TABLE]) is None
    assert check_calculation(n, "(112980.25 - 105200) / 105200", [TABLE]) is None  # ratio, claim in %
    assert "operand 99999" in check_calculation(n, "(99999 - 105200) / 105200 * 100", [TABLE])
    assert "not 7.4%" in check_calculation(n, "112980.25 / 105200 * 100", [TABLE])


def test_derived_number_verified_with_llm_expression():
    c = claim("Retail revenue grew 7.4% from FY2023 to FY2024.", "DOC03_P7_T2_R1")
    fake = routed(calc={"calculations": [{"id": 1, "expression": "(112980.25 - 105200.00) / 105200.00 * 100"}]})
    res, ws = check_numbers([c], CHUNKS, fake)
    assert res[1].ok and res[1].derived == {"7.4%": "(112980.25 - 105200.00) / 105200.00 * 100"}
    assert ws == [] and fake.steps == ["calc"]


def test_wrong_calculation_is_caught():
    c = claim("Retail revenue grew 9.1% from FY2023 to FY2024.", "DOC03_P7_T2_R1")
    fake = routed(calc={"calculations": [{"id": 1, "expression": "(112980.25 - 105200.00) / 105200.00 * 100"}]})
    res, ws = check_numbers([c], CHUNKS, fake)
    assert res[1].unmatched == ["9.1%"] and "rejected" in ws[0].message


def test_calc_llm_not_called_when_all_numbers_match():
    fake = routed()
    check_numbers([claim("Margin was -2.5%.", "DOC01_P3_C1")], CHUNKS, fake)
    assert fake.calls == []


def test_calc_llm_failure_leaves_mismatch():
    res, ws = check_numbers([claim("Margin was 9%.", "DOC01_P3_C1")], CHUNKS, routed())
    assert not res[1].ok and "unavailable" in ws[0].message


# --- table-cell alignment (failures.md F5) --------------------------------------------------

HEADERS = ["Model (→) / Dataset (↓)", "Lexical / BM25", "Dense / TAS-B", "Re-ranking / BM25+CE"]
CAPTION = "Table 2: In-domain and zero-shot performances on BEIR benchmark. (nDCG@10)"


def beir_row(rid, row):
    td = TableData("T1", HEADERS, [row], caption=CAPTION)
    text = CAPTION + " | " + " | ".join(f"{h}={v}" for h, v in zip(HEADERS, row) if v)
    return Chunk("DOC02", "beir.pdf", 7, f"DOC02_P7_T1_{rid}", text,
                 content_type=ContentType.TABLE, table_data=td)


FIQA = beir_row("R7", ["FiQA-2018", "0.236", "0.300", "0.347"])
COVID = beir_row("R2", ["TREC-COVID", "0.656", "0.481", "0.757"])
AVG = beir_row("R20", ["Avg. Performance vs. BM25", "", "- 2.8%", "+ 11%"])
ROWS = {c.chunk_id: c for c in (FIQA, COVID, AVG)}
F5_Q = "On average, how much better than BM25 is the BM25+CE re-ranker on BEIR?"


def test_f5_right_number_wrong_row_is_caught():
    c = Claim(1, "0.347.", ("DOC02_P7_T1_R7", "DOC02_P7_T1_R20"), True, True)
    res, ws = check_numbers([c], ROWS, question=F5_Q)
    assert res[1].unmatched == ["0.347"] and ws
    no_q, _ = check_numbers([c], ROWS)  # without the question there is no context to align
    assert no_q[1].ok


def test_f5_correct_cell_passes():
    c = Claim(1, "BM25+CE is 11% better than BM25 on average.", ("DOC02_P7_T1_R20",), True, True)
    assert check_numbers([c], ROWS, question=F5_Q)[0][1].ok


@pytest.mark.parametrize("q, claim_text, ok", [
    ("What is the nDCG@10 of BM25 on TREC-COVID?", "0.656.", True),
    ("What is the nDCG@10 of BM25 on TREC-COVID?", "0.757.", False),    # BM25+CE column... no "ce" in q
    ("What does TAS-B score on TREC-COVID?", "0.481.", True),
    ("What does TAS-B score on FiQA-2018?", "0.481.", False),           # value from the COVID row
])
def test_cell_alignment(q, claim_text, ok):
    c = Claim(1, claim_text, tuple(ROWS), True, True)
    assert check_numbers([c], ROWS, question=q)[0][1].ok is ok


def test_alignment_handles_period_headers():
    c = Claim(1, "1,12,980.25 crore.", ("DOC03_P7_T2_R1",), True, True)
    for q in ["What was Retail revenue in FY2024?", "Retail revenue in FY24?", "Retail revenue in 2024?"]:
        assert check_numbers([c], CHUNKS, question=q)[0][1].ok, q
    assert not check_numbers([c], CHUNKS, question="Retail revenue in FY2025?")[0][1].ok


@pytest.mark.parametrize("expr", ["0.347", "0.347 * 1", "(0.347 + 0) / 1"])
def test_restating_a_passage_number_is_not_a_calculation(expr):
    assert check_calculation(num("0.347"), expr, [FIQA]) is not None  # rejected either way


def test_f5_not_rescued_by_trivial_calculation():
    c = Claim(1, "0.347", ("DOC02_P7_T1_R7",), True, True)
    fake = routed(calc={"calculations": [{"id": 1, "expression": "0.347"}]})
    res, _ = check_numbers([c], ROWS, fake, question=F5_Q)
    assert res[1].unmatched == ["0.347"]
