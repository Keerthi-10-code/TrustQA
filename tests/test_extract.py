import shutil

import pymupdf
import pytest

from trustqa.processing.extract import DocumentRegistry, extract_pages, normalize_text


def test_pages_are_one_indexed_and_in_order(solar_pdf):
    pages = extract_pages(solar_pdf, "DOC01")
    assert [p.page_number for p in pages] == [1, 2, 3, 4]
    assert all(p.document_id == "DOC01" and p.document_name == "solar_report.pdf"
               for p in pages)
    assert "Helios Energy" in pages[0].text
    assert pages[1].text == ""  # blank page kept, so later page numbers stay true
    assert "Pavagada" in pages[3].text


def test_numbers_survive_extraction(solar_pdf):
    text = extract_pages(solar_pdf, "DOC01")[2].text
    assert "Rs. 1,03,450.50 crore" in text
    assert "-2.5%" in text


def test_normalize_text_is_conservative():
    vocab = {"retrieval", "files"}
    assert normalize_text("retrie-\nval of ﬁles", vocab) == "retrieval of files"
    # compound not known as one word -> hyphen kept, nothing lost
    assert normalize_text("knowledge-\nintensive", vocab) == "knowledge-intensive"
    assert normalize_text("retrie-\nval") == "retrie-val"  # no vocabulary -> never join
    assert normalize_text("range 10-\n20 units") == "range 10- 20 units"  # digits untouched
    assert normalize_text("x²  = -3.50\n\n") == "x² = -3.50"  # no NFKC on superscripts


def test_registry_assigns_sequential_ids_and_dedupes_by_content(solar_pdf, river_pdf, tmp_path):
    reg = DocumentRegistry()
    a, new_a = reg.register(solar_pdf)
    b, new_b = reg.register(river_pdf)
    copy = shutil.copy(solar_pdf, tmp_path / "copy_of_solar.pdf")
    c, new_c = reg.register(copy)
    assert (a.document_id, b.document_id) == ("DOC01", "DOC02")
    assert new_a and new_b and not new_c
    assert c.document_id == "DOC01" and len(reg) == 2


def test_registry_is_deterministic(solar_pdf, river_pdf):
    ids = []
    for _ in range(2):
        reg = DocumentRegistry()
        ids.append([reg.register(p)[0].document_id for p in (river_pdf, solar_pdf)])
    assert ids[0] == ids[1] == ["DOC01", "DOC02"]


def test_unreadable_and_missing_files_raise_value_error(tmp_path):
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"this is not a pdf")
    with pytest.raises(ValueError, match="bad.pdf"):
        extract_pages(bad, "DOC01")
    with pytest.raises(ValueError, match="not found"):
        extract_pages(tmp_path / "missing.pdf", "DOC01")


def test_encrypted_pdf_raises_value_error(tmp_path):
    path = tmp_path / "locked.pdf"
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), "secret")
    doc.save(path, encryption=pymupdf.PDF_ENCRYPT_AES_256, owner_pw="o", user_pw="u")
    doc.close()
    with pytest.raises(ValueError, match="encrypted"):
        extract_pages(path, "DOC01")
