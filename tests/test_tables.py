"""Table extraction: headers/rows preserved, numeric fidelity (NFR-03), row-aware chunks."""

import re
from pathlib import Path

import pdfplumber
import pymupdf
import pytest

from trustqa.config import load_config
from trustqa.processing.process import process_pdfs, process_pdfs_detailed
from trustqa.processing.tables import detect_units, find_tables, row_text
from trustqa.schemas import Chunk, ContentType, ExtractionMethod

NUM_RE = re.compile(r"[-+−]?\(?\d[\d,]*(?:\.\d+)?\)?%?")

INTRO = ("Segment results. Helios Energy reports revenue for each business segment below. "
         "Figures are unaudited and stated in crore rupees.")
HEADERS = ["Segment", "Revenue / Q1 FY25", "Revenue / Q2 FY25", "Margin %"]
ROWS = [
    ["Retail", "1,03,450.50", "98,210.75", "-2.5"],
    ["Utility scale", "45,000.00", "-1,250.40", "+11%"],
    ["Exports", "0.0700", "(3.2)", "-0.35"],
]
CAPTION = "Table 3: Quarterly revenue by segment (Rs crore). Negative values are losses."


def make_booktabs_pdf(path: Path, caption: str | None = CAPTION) -> Path:
    """Academic-style table: three long rules, a partial rule under the 'Revenue' group
    header, no vertical rules, caption below."""
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    page.insert_textbox(pymupdf.Rect(60, 40, 540, 90), INTRO, fontsize=10)
    xs = [70, 220, 320, 430]
    page.draw_line((60, 110), (540, 110), width=0.8)
    page.insert_text((250, 124), "Revenue", fontsize=9)
    page.draw_line((215, 128), (400, 128), width=0.5)  # \cmidrule under "Revenue"
    for x, h in zip(xs, ["Segment", "Q1 FY25", "Q2 FY25", "Margin %"]):
        page.insert_text((x, 140), h, fontsize=9)
    page.draw_line((60, 146), (540, 146), width=0.5)
    y = 162
    for row in ROWS:
        for x, v in zip(xs, row):
            page.insert_text((x, y), v, fontsize=9)
        y += 16
    page.draw_line((60, y - 8), (540, y - 8), width=0.8)
    if caption:
        page.insert_text((60, y + 10), caption, fontsize=9)
    page.insert_textbox(pymupdf.Rect(60, y + 40, 540, y + 120),
                        "The retail segment had the largest revenue in both quarters.",
                        fontsize=10)
    doc.save(path)
    doc.close()
    return path


def make_grid_pdf(path: Path) -> Path:
    """Fully ruled grid with a merged 'Sales' header cell over two year columns."""
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    xs, ys = [60, 200, 320, 440], [100, 120, 140, 160, 180]
    for y in ys:
        page.draw_line((xs[0], y), (xs[-1], y), width=0.6)
    for x in xs:
        if x == 320:  # no vertical line inside the merged header cell
            page.draw_line((x, ys[1]), (x, ys[-1]), width=0.6)
        else:
            page.draw_line((x, ys[0]), (x, ys[-1]), width=0.6)
    cells = [[(65, "Region"), (205, "Sales")],
             [(205, "2024"), (325, "2025")],
             [(65, "North"), (205, "1,200.50"), (325, "-3.75")],
             [(65, "South"), (205, "0.08"), (325, "+12%")]]
    for r, row in enumerate(cells):
        for x, t in row:
            page.insert_text((x, ys[r] + 14), t, fontsize=9)
    doc.save(path)
    doc.close()
    return path


@pytest.fixture
def cfg(tmp_path):
    env = tmp_path / ".env"
    env.write_text("")
    return load_config(env, ocr_enabled=False)


@pytest.fixture
def booktabs(tmp_path):
    return make_booktabs_pdf(tmp_path / "segments.pdf")


def _tables(pdf: Path):
    with pdfplumber.open(pdf) as p:
        return find_tables(p.pages[0], 1, "DOC01")


# ------------------------------------------------------------------ structure


def test_booktabs_headers_and_rows_preserved(booktabs):
    (t,) = _tables(booktabs)
    assert t.headers == HEADERS  # group header spread over its \cmidrule columns
    assert t.rows == ROWS
    assert (t.label, t.method, t.table_id) == ("Table 3", "rules", "DOC01_P1_T1")
    assert t.caption == CAPTION
    assert t.units == "Rs crore"


def test_grid_with_merged_header(tmp_path):
    (t,) = _tables(make_grid_pdf(tmp_path / "grid.pdf"))
    assert t.method == "lines"
    assert t.headers == ["Region", "Sales / 2024", "Sales / 2025"]
    assert t.rows == [["North", "1,200.50", "-3.75"], ["South", "0.08", "+12%"]]


def test_ruled_box_without_caption_is_not_a_table(tmp_path):
    (t,) = _tables(make_booktabs_pdf(tmp_path / "nocap.pdf", caption=None)) or (None,)
    assert t is None  # e.g. algorithm boxes: rules but no "Table N" caption


def test_superscript_exponent_is_marked_not_merged(tmp_path):
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    page.draw_line((60, 100), (400, 100))
    page.insert_text((70, 115), "Model", fontsize=10)
    page.insert_text((250, 115), "Cost", fontsize=10)
    page.draw_line((60, 121), (400, 121))
    for y, (m, base, exp) in zip((138, 154), [("Big", "2.3 · 10", "19"), ("Base", "3.3 · 10", "18")]):
        page.insert_text((70, y), m, fontsize=10)
        page.insert_text((250, y), base, fontsize=10)
        end = 250 + pymupdf.get_text_length(base, fontsize=10)
        page.insert_text((end + 0.3, y - 4), exp, fontsize=7)
    page.draw_line((60, 162), (400, 162))
    page.insert_text((60, 178), "Table 1: Training cost in FLOPs.", fontsize=9)
    doc.save(tmp_path / "exp.pdf")
    doc.close()
    (t,) = _tables(tmp_path / "exp.pdf")
    assert t.rows == [["Big", "2.3 · 10^19"], ["Base", "3.3 · 10^18"]]


@pytest.mark.parametrize("caption,headers,units", [
    ("Table 2: Revenue (Rs crore)", [], "Rs crore"),
    ("Table 1: Sales in USD millions", [], "USD millions"),
    ("Table 2: All scores denote nDCG@10. Best in bold.", [], "nDCG@10"),
    ("Table 4: Results", ["Model", "Latency (ms)"], "ms"),
    ("Table 5: Datasets", ["Name", "Size"], None),
])
def test_unit_detection(caption, headers, units):
    assert detect_units(caption, headers) == units


# ------------------------------------------------------------------ chunks


def test_table_chunks_are_row_aware_with_context(booktabs, cfg):
    chunks = process_pdfs([booktabs], cfg)
    rows = [c for c in chunks if c.content_type is ContentType.TABLE]
    assert [c.chunk_id for c in rows] == ["DOC01_P1_T1_R1", "DOC01_P1_T1_R2", "DOC01_P1_T1_R3"]
    for c, row in zip(rows, ROWS):
        assert c.extraction_method is ExtractionMethod.PDFPLUMBER and not c.ocr_used
        assert c.section == "Table 3" and c.page_number == 1
        td = c.table_data
        assert (td.table_id, td.headers, td.rows) == ("DOC01_P1_T1", HEADERS, [row])
        assert td.units == "Rs crore" and td.caption == CAPTION
        # caption, units and every header=value pair in every row chunk
        assert c.text.startswith("Table 3: Quarterly revenue by segment (Rs crore). |")
        for h, v in zip(HEADERS, row):
            assert f"{h}={v}" in c.text
        assert Chunk.from_json(c.to_json()) == c
    assert rows[0].text == row_text_expected(ROWS[0])


def row_text_expected(row):
    pairs = " | ".join(f"{h}={v}" for h, v in zip(HEADERS, row))
    return f"Table 3: Quarterly revenue by segment (Rs crore). | {pairs}"


def test_table_body_removed_from_page_text_but_prose_kept(booktabs, cfg):
    chunks = process_pdfs([booktabs], cfg)
    text = " ".join(c.text for c in chunks if c.content_type is ContentType.TEXT)
    assert "Helios Energy reports revenue" in text
    assert "largest revenue in both quarters" in text
    assert "Table 3: Quarterly revenue" in text  # caption stays searchable as prose
    assert "1,03,450.50" not in text and "98,210.75" not in text  # no table-soup duplicate


def test_flatten_tables_leaves_table_in_page_text(booktabs, tmp_path):
    env = tmp_path / "b.env"
    env.write_text("")
    cfg_b = load_config(env, ocr_enabled=False, flatten_tables=True)
    chunks = process_pdfs([booktabs], cfg_b)
    assert all(c.content_type is ContentType.TEXT for c in chunks)
    text = " ".join(c.text for c in chunks)
    assert "1,03,450.50" in text and "-1,250.40" in text


# ------------------------------------------------------------------ numeric fidelity


def _pdf_numbers(pdf: Path, clip) -> list[str]:
    with pymupdf.open(pdf) as doc:
        words = doc[0].get_text("words", clip=clip)
    return [w[4] for w in words if NUM_RE.fullmatch(w[4])]


def test_every_number_in_table_survives_verbatim(booktabs, cfg):
    result = process_pdfs_detailed([booktabs], cfg)
    (t,) = result.extracted_tables
    in_pdf = _pdf_numbers(booktabs, pymupdf.Rect(*t.bbox))
    cells = [c for row in t.rows for c in row]
    assert sorted(in_pdf) == sorted(c for c in cells if NUM_RE.fullmatch(c))
    texts = " ".join(c.text for c in result.chunks if c.content_type is ContentType.TABLE)
    for n in in_pdf:
        assert n in texts, n


# ------------------------------------------------------------------ pilot table PDF

BEIR = Path(__file__).resolve().parents[1] / "data" / "raw_pdfs" / "tables" / "beir_thakur2021.pdf"


@pytest.mark.skipif(not BEIR.is_file(), reason="pilot PDF not present (git-ignored)")
def test_beir_results_table():
    with pdfplumber.open(BEIR) as pdf:
        (t,) = find_tables(pdf.pages[6], 7, "DOC02")
    assert t.label == "Table 2" and t.units == "nDCG@10" and len(t.rows) == 20
    assert t.headers[:3] == ["Model (→) / Dataset (↓)", "Lexical / BM25", "Sparse / DeepCT"]
    assert "Dense / TAS-B" in t.headers and "Re-ranking / BM25+CE" in t.headers
    by_label = {r[0]: dict(zip(t.headers, r)) for r in t.rows}
    assert by_label["TREC-COVID"]["Lexical / BM25"] == "0.656"
    assert by_label["MS MARCO"]["Sparse / DeepCT"] == "0.296‡"
    assert by_label["Touché-2020"]["Lexical / BM25"] == "0.367"
    assert by_label["Avg. Performance vs. BM25"]["Dense / DPR"] == "- 47.7%"
    assert by_label["Avg. Performance vs. BM25"]["Re-ranking / BM25+CE"] == "+ 11%"


def test_small_caps_unmarked_but_subscript_marked(tmp_path):
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    page.draw_line((60, 100), (400, 100))
    page.insert_text((70, 115), "Token", fontsize=10)
    page.insert_text((250, 115), "Cost", fontsize=10)
    page.draw_line((60, 121), (400, 121))
    for y, (big, small, dy, rest) in zip((138, 154), [("I", "SREL", 0, ""), ("O(log", "k", 2.5, "(n))")]):
        page.insert_text((70, y), big, fontsize=10)
        x = 70 + pymupdf.get_text_length(big, fontsize=10)
        page.insert_text((x, y + dy), small, fontsize=7)  # dy=0: small caps; dy>0: subscript
        if rest:
            x += pymupdf.get_text_length(small, fontsize=7)
            page.insert_text((x, y), rest, fontsize=10)
        page.insert_text((250, y), "1.0", fontsize=10)
    page.draw_line((60, 162), (400, 162))
    page.insert_text((60, 178), "Table 1: Tokens.", fontsize=9)
    doc.save(tmp_path / "caps.pdf")
    doc.close()
    (t,) = _tables(tmp_path / "caps.pdf")
    assert [r[0] for r in t.rows] == ["ISREL", "O(log_k(n))"]


def test_full_table_rebuilt_from_row_chunks(booktabs, cfg):
    from trustqa.processing.tables import rebuild_table

    chunks = process_pdfs([booktabs], cfg)
    shuffled = list(reversed(chunks))  # order of retrieval / index must not matter
    td = rebuild_table(shuffled, "DOC01_P1_T1")
    assert (td.headers, td.rows, td.units, td.caption) == (HEADERS, ROWS, "Rs crore", CAPTION)
    assert rebuild_table(chunks, "DOC01_P9_T9") is None
