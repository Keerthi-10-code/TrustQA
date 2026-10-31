"""Phase 5 evidence panel: page rendering with the cited text highlighted."""

import pytest

from trustqa.evidence import align_words, highlight_target, render_page_evidence
from trustqa.schemas import Chunk, ContentType, TableData


def test_align_words_needs_consecutive_runs_and_enough_overlap():
    page = "the board approved a dividend of 12 percent in the annual report".split()
    assert align_words(page, "approved a dividend of 12 percent".split()) == set(range(2, 8))
    assert align_words(page, "the cat sat on the mat today".split()) == set()  # stray words only
    assert align_words(page, []) == set()


def test_highlight_target_prefers_table_cells_then_quote_then_chunk_text():
    td = TableData("DOC01_P1_T1", ["Model", "Score"], [["BM25", "0.656"]])
    row = Chunk("DOC01", "a.pdf", 1, "DOC01_P1_T1_R1", "Table 1 | Model=BM25 | Score=0.656",
                content_type=ContentType.TABLE, table_data=td)
    assert highlight_target(row, "Model=BM25") == (["BM25 0.656"], "table row")
    text = Chunk("DOC01", "a.pdf", 1, "DOC01_P1_C1", "Some passage text.")
    assert highlight_target(text, "first part ... second part") == (
        ["first part", "second part"], "quote")
    assert highlight_target(text, "") == (["Some passage text."], "chunk text")
    assert highlight_target(None, None) == ([], "none")


def test_render_highlights_quote_on_digital_page(solar_pdf):
    ev = render_page_evidence(solar_pdf, 3, None, "The board approved a dividend of 12 percent.")
    assert ev.png.startswith(b"\x89PNG") and ev.page_number == 3
    assert ev.located and ev.highlighted_words >= 7 and ev.note is None


def test_render_reports_quote_not_on_page(solar_pdf):
    ev = render_page_evidence(solar_pdf, 1, None, "Completely unrelated sentence about tides.")
    assert not ev.located and "could not be located" in ev.note


def test_render_rejects_bad_page(solar_pdf):
    with pytest.raises(ValueError, match="out of range"):
        render_page_evidence(solar_pdf, 9, None, "x")
