from pathlib import Path

import pymupdf
import pytest

from trustqa.processing.detect import page_quality

MIN, RATIO = 50, 0.6


def test_normal_prose_is_ok():
    text = ("Revenue grew to Rs. 1,03,450.50 crore in 2025. Operating margin was -2.5% "
            "because of panel import costs. The board approved a dividend of 12 percent.")
    q = page_quality(text, MIN, RATIO)
    assert not q.needs_ocr and q.reason == "ok"


@pytest.mark.parametrize("text", ["", "   \n ", "Page 3", "CONFIDENTIAL - scanned copy 12"])
def test_empty_or_stamp_only_page_needs_ocr(text):
    q = page_quality(text, MIN, RATIO)
    assert q.needs_ocr and q.reason == "low_text"


def test_symbol_garbage_needs_ocr():
    text = "a#$%&* " * 80  # broken font encoding: mostly symbols, 80 letters
    q = page_quality(text, MIN, RATIO)
    assert q.alnum_chars >= MIN
    assert q.needs_ocr and q.reason == "garbage_text"


def test_undecodable_glyphs_count_as_garbage():
    # "(cid:51)" contains letters and digits but is one undecodable glyph, as is U+FFFD
    text = "word " * 15 + "(cid:51)(cid:52) " * 60 + "�" * 50
    q = page_quality(text, MIN, RATIO)
    assert q.reason == "garbage_text"


def test_table_of_contents_with_dot_leaders_is_not_garbage():
    # Self-RAG appendix TOC (page 16) has ratio ~0.49: real text, must not be OCR'd
    line = "A.1 Reflection Tokens . . . . . . . . . . . . . . . . . . . . . . . 17\n"
    q = page_quality(line * 12, MIN, RATIO)
    assert not q.needs_ocr


PILOT = Path(__file__).resolve().parents[1] / "data" / "raw_pdfs"


@pytest.mark.skipif(not (PILOT / "text").is_dir() or not any(PILOT.rglob("*.pdf")),
                    reason="pilot PDFs not present (git-ignored)")
def test_pilot_detection_matches_document_type():
    """Digital pilot pages never trigger OCR; every page of the synthetic scan does."""
    for pdf in sorted(PILOT.rglob("*.pdf")):
        with pymupdf.open(pdf) as doc:
            flags = [page_quality(p.get_text(), MIN, RATIO).needs_ocr for p in doc]
        if pdf.parent.name == "scanned":
            assert all(flags), pdf.name
        else:
            assert not any(flags), (pdf.name, [i + 1 for i, f in enumerate(flags) if f])
