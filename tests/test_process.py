import shutil
from pathlib import Path

import pytest

from trustqa.config import load_config
from trustqa.processing.process import process_pdfs
from trustqa.schemas import Chunk

PILOT_DIGITAL = sorted(
    p for sub in ("text", "tables")
    for p in (Path(__file__).resolve().parents[1] / "data" / "raw_pdfs" / sub).glob("*.pdf")
)


@pytest.fixture
def cfg(tmp_path):
    env = tmp_path / ".env"
    env.write_text("")
    return load_config(env, chunk_size=120, chunk_overlap=30)


def test_ids_unique_pages_one_indexed(solar_pdf, river_pdf, cfg):
    chunks = process_pdfs([solar_pdf, river_pdf], cfg)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))
    assert all(c.page_number >= 1 for c in chunks)
    assert {c.document_id for c in chunks} == {"DOC01", "DOC02"}
    assert {c.document_name for c in chunks if c.document_id == "DOC02"} == {"river_survey.pdf"}
    # page numbers match the PDF: the 600 MW fact is on page 4 of the solar report
    hit = [c for c in chunks if "600 MW" in c.text]
    assert hit and all(c.page_number == 4 and c.document_id == "DOC01" for c in hit)


def test_input_order_defines_ids(solar_pdf, river_pdf, cfg):
    chunks = process_pdfs([river_pdf, solar_pdf], cfg)
    assert {c.document_name for c in chunks if c.document_id == "DOC01"} == {"river_survey.pdf"}


def test_duplicate_file_processed_once(solar_pdf, tmp_path, cfg):
    dup = shutil.copy(solar_pdf, tmp_path / "dup.pdf")
    chunks = process_pdfs([solar_pdf, dup], cfg)
    assert {c.document_id for c in chunks} == {"DOC01"}


def test_chunks_round_trip_json(solar_pdf, cfg):
    for c in process_pdfs([solar_pdf], cfg):
        assert Chunk.from_json(c.to_json()) == c


def test_missing_file_raises(tmp_path, cfg):
    with pytest.raises(ValueError, match="not found"):
        process_pdfs([tmp_path / "nope.pdf"], cfg)


@pytest.mark.skipif(not PILOT_DIGITAL, reason="pilot PDFs not present (git-ignored)")
def test_pilot_pdfs_default_config():
    cfg = load_config()
    chunks = process_pdfs(PILOT_DIGITAL, cfg)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))
    assert all(1 <= len(c.text) <= cfg.chunk_size for c in chunks)
    assert all(c.page_number >= 1 for c in chunks)
    assert len({c.document_id for c in chunks}) == len(PILOT_DIGITAL)
