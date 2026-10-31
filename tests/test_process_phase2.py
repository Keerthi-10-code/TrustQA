"""Phase 2 processing: OCR only on low-text pages, config switches, baseline B, JSONL."""

import json
from pathlib import Path

import pymupdf
import pytest

from tests.conftest import make_pdf
from trustqa import llm as llm_mod
from trustqa import pipeline
from trustqa.config import ConfigError, load_config
from trustqa.llm import FakeLLM
from trustqa.processing import process as process_mod
from trustqa.processing.extract import EXTRACTOR_VERSION
from trustqa.processing.ocr import OcrUnavailableError, assemble_words, tesseract_version
from trustqa.processing.process import (
    process_pdfs,
    process_pdfs_detailed,
    processing_signature,
    read_jsonl,
)
from trustqa.schemas import ContentType, ExtractionMethod, Status

DIGITAL = ("Helios Energy annual report. The Pavagada plant has an installed capacity of "
           "600 MW and maintenance is done by drones every quarter.")
SCANNED = "Scanned annex. The Kaveri basin receives 1,234.56 mm of rain from June to September."


def make_mixed_pdf(path: Path) -> Path:
    """Page 1 digital text, page 2 image-only (a 'scan' of text), page 3 blank."""
    src = pymupdf.open()
    sp = src.new_page(width=595, height=842)
    sp.insert_textbox(pymupdf.Rect(50, 50, 545, 400), SCANNED, fontsize=14)
    pix = sp.get_pixmap(dpi=200, colorspace=pymupdf.csGRAY)

    doc = pymupdf.open()
    p1 = doc.new_page(width=595, height=842)
    p1.insert_textbox(pymupdf.Rect(50, 50, 545, 792), DIGITAL, fontsize=11)
    p2 = doc.new_page(width=595, height=842)
    p2.insert_image(p2.rect, pixmap=pix)
    doc.new_page(width=595, height=842)
    doc.save(path)
    doc.close()
    src.close()
    return path


@pytest.fixture
def cfg(tmp_path):
    env = tmp_path / ".env"
    env.write_text("")
    return load_config(env, chunk_size=200, chunk_overlap=40,
                       indexes_dir=tmp_path / "idx", processed_dir=tmp_path / "processed")


@pytest.fixture
def mixed_pdf(tmp_path):
    return make_mixed_pdf(tmp_path / "mixed.pdf")


@pytest.fixture
def fake_ocr(monkeypatch):
    """Record which pages are OCR'd; return the known page text with fixed confidences."""
    calls = []

    def ocr_page(page, **kw):
        calls.append(page.number + 1)
        text, words = assemble_words([[(w, 88.0 if i % 2 else 72.0)
                                       for i, w in enumerate(SCANNED.split())]])
        from trustqa.processing.ocr import OcrPage

        return OcrPage(text=text, words=words, mean_confidence=80.0)

    monkeypatch.setattr(process_mod, "ocr_page", ocr_page)
    monkeypatch.setattr(process_mod, "tesseract_version", lambda cmd=None: "fake")
    return calls


def test_ocr_runs_only_on_the_low_text_page(mixed_pdf, cfg, fake_ocr):
    result = process_pdfs_detailed([mixed_pdf], cfg)
    assert fake_ocr == [2]  # not the digital page 1, not the blank page 3
    by_page = {}
    for c in result.chunks:
        by_page.setdefault(c.page_number, []).append(c)
    assert set(by_page) == {1, 2}
    for c in by_page[1]:
        assert (c.extraction_method, c.ocr_used, c.ocr_confidence) == \
               (ExtractionMethod.PYMUPDF, False, None)
    for c in by_page[2]:
        assert c.extraction_method is ExtractionMethod.TESSERACT and c.ocr_used
        assert 72.0 <= c.ocr_confidence <= 88.0
    assert "1,234.56" in " ".join(c.text for c in by_page[2])
    rep = result.documents[0]
    assert rep.ocr_pages == [2]
    assert [p.quality for p in rep.pages] == ["ok", "low_text", "blank"]


def test_per_chunk_confidence_comes_from_its_own_words(mixed_pdf, cfg, fake_ocr):
    small = load_config(cfg_env(cfg), chunk_size=40, chunk_overlap=0,
                        processed_dir=cfg.processed_dir)
    ocr_chunks = [c for c in process_pdfs([mixed_pdf], small) if c.ocr_used]
    assert len(ocr_chunks) > 1
    assert len({c.ocr_confidence for c in ocr_chunks}) > 1  # not just the page mean


def cfg_env(cfg):
    p = Path(cfg.processed_dir).parent / ".env"
    p.write_text("")
    return p


def test_ocr_disabled_skips_scanned_page(mixed_pdf, cfg, fake_ocr):
    off = load_config(cfg_env(cfg), ocr_enabled=False)
    chunks = process_pdfs([mixed_pdf], off)
    assert fake_ocr == []
    assert {c.page_number for c in chunks} == {1}


def test_garbage_text_layer_is_ocrd_and_bad_ocr_rejected(tmp_path, cfg, monkeypatch, fake_ocr):
    garbage = make_pdf(tmp_path / "garbage.pdf", ["a#$%&*@! " * 60, DIGITAL])
    result = process_pdfs_detailed([garbage], cfg)
    assert fake_ocr == [1]
    assert result.documents[0].pages[0].method == "tesseract"

    # an OCR result worse than the text layer is not used
    from trustqa.processing.ocr import OcrPage

    monkeypatch.setattr(process_mod, "ocr_page",
                        lambda page, **kw: OcrPage(text="", words=[], mean_confidence=None))
    low = make_pdf(tmp_path / "low.pdf", ["Figure 3", DIGITAL])
    result = process_pdfs_detailed([low], cfg)
    pr = result.documents[0].pages[0]
    assert pr.ocr_attempted and pr.ocr_rejected and pr.method == "pymupdf"
    assert any(c.text == "Figure 3" for c in result.chunks)


def test_missing_tesseract_keeps_digital_pages(mixed_pdf, cfg, monkeypatch):
    def unavailable(cmd=None):
        raise OcrUnavailableError("Tesseract not found; set TESSERACT_CMD")

    monkeypatch.setattr(process_mod, "tesseract_version", unavailable)
    result = process_pdfs_detailed([mixed_pdf], cfg)
    assert {c.page_number for c in result.chunks} == {1}
    assert any("Tesseract" in w for w in result.documents[0].warnings)


def test_missing_tesseract_on_fully_scanned_pdf_is_an_error(tmp_path, cfg, monkeypatch):
    def unavailable(cmd=None):
        raise OcrUnavailableError("Tesseract not found; set TESSERACT_CMD")

    monkeypatch.setattr(process_mod, "tesseract_version", unavailable)
    scan = make_mixed_pdf(tmp_path / "m.pdf")
    only_scan = tmp_path / "scan.pdf"
    with pymupdf.open(scan) as d:
        d.select([1])
        d.save(only_scan)
    with pytest.raises(ValueError, match="no usable text layer"):
        process_pdfs([only_scan], cfg)


def _tesseract_available() -> bool:
    try:
        tesseract_version(load_config().tesseract_cmd)
        return True
    except OcrUnavailableError:
        return False


@pytest.mark.skipif(not _tesseract_available(), reason="Tesseract not installed")
def test_real_tesseract_on_image_page(mixed_pdf, tmp_path):
    env = tmp_path / "real.env"
    env.write_text(f"TESSERACT_CMD={load_config().tesseract_cmd or ''}\n")
    result = process_pdfs_detailed([mixed_pdf], load_config(env))
    ocr = [c for c in result.chunks if c.ocr_used]
    assert ocr and all(c.page_number == 2 for c in ocr)
    text = " ".join(c.text for c in ocr)
    assert "Kaveri" in text and "1,234.56" in text
    assert all(c.ocr_confidence is not None and c.ocr_confidence > 60 for c in ocr)


# ------------------------------------------------------------------ config / systems


def test_baseline_b_preset_forces_flat_tables_and_no_verification(tmp_path, monkeypatch):
    for k, v in {"BASELINE_B": "1", "FLATTEN_TABLES": "0", "VERIFY": "1"}.items():
        monkeypatch.setenv(k, v)  # setenv, not a .env file, so nothing leaks
    env = tmp_path / ".env"
    env.write_text("")
    b = load_config(env)
    assert (b.flatten_tables, b.verify, b.ocr_enabled) == (True, False, True)
    assert b.system_name == "B-ocr-plain-rag"
    a = load_config(env, baseline_b=False, flatten_tables=True, verify=False, ocr_enabled=False)
    assert a.system_name == "A-baseline"
    assert load_config(env, baseline_b=False).system_name == "C-trustqa"


@pytest.mark.parametrize("name,value", [("OCR_ENABLED", "maybe"), ("OCR_DPI", "20"),
                                        ("OCR_MAX_NONALNUM_RATIO", "0")])
def test_invalid_phase2_settings_rejected(tmp_path, name, value, monkeypatch):
    # setenv (not a .env file): load_dotenv would leak the bad value into os.environ
    monkeypatch.setenv(name, value)
    env = tmp_path / ".env"
    env.write_text("")
    with pytest.raises(ConfigError):
        load_config(env)


def test_system_a_processing_signature_is_unchanged(cfg):
    a = load_config(cfg_env(cfg), ocr_enabled=False, flatten_tables=True, verify=False)
    assert processing_signature(a) == EXTRACTOR_VERSION  # frozen A index key still valid
    assert processing_signature(cfg) != EXTRACTOR_VERSION


A_JSON = Path(__file__).resolve().parents[1] / "eval" / "configs" / "system_A.json"
RAW = Path(__file__).resolve().parents[1] / "data" / "raw_pdfs"


@pytest.mark.skipif(not any(RAW.rglob("*.pdf")), reason="pilot PDFs not present")
def test_system_a_index_key_reproduces_frozen_value():
    frozen = json.loads(A_JSON.read_text())
    docs = [RAW / d["path"] for d in frozen["document_set"]["documents"]]
    if not all(p.is_file() for p in docs):
        pytest.skip("pilot set differs from the frozen system A set")
    s = frozen["settings"]
    a = load_config(ocr_enabled=False, flatten_tables=True, verify=False,
                    chunk_size=s["chunk_size"], chunk_overlap=s["chunk_overlap"],
                    embedding_model=s["embedding_model"])
    assert pipeline.index_key(docs, a) == frozen["document_set"]["index_key"]


# ------------------------------------------------------------------ pipeline


@pytest.fixture(autouse=True)
def _clear_caches():
    pipeline._memory_cache.clear()
    yield
    pipeline._memory_cache.clear()


def _cite_first(prompt, system, json_mode):
    import re

    cid = re.findall(r"^\[(DOC\d+_P\d+_[A-Z]\d+(?:_R\d+)?)\]", prompt, re.M)[0]
    return json.dumps({"answer": f"Answer [{cid}].", "chunk_ids": [cid],
                       "insufficient_evidence": False})


def test_baseline_b_skips_verification(solar_pdf, cfg, fake_embedder, monkeypatch):
    b = load_config(cfg_env(cfg), baseline_b=True, indexes_dir=cfg.indexes_dir,
                    processed_dir=cfg.processed_dir)

    def boom(*a, **kw):
        raise AssertionError("verification must not run for baseline B")

    monkeypatch.setattr(pipeline, "run_verification", boom)
    llm_mod.set_llm(FakeLLM(_cite_first))
    r = pipeline.answer_question("installed capacity?", [solar_pdf], b)
    assert r.status is Status.PARTIALLY_SUPPORTED and r.citations


def test_build_writes_canonical_jsonl(solar_pdf, cfg, fake_embedder):
    from tests.test_tables import make_booktabs_pdf

    table_pdf = make_booktabs_pdf(Path(cfg.processed_dir).parent / "segments.pdf")
    idx = pipeline.get_or_build_index([solar_pdf, table_pdf], cfg)
    (out,) = [p for p in Path(cfg.processed_dir).iterdir() if p.is_dir()]
    records = read_jsonl(out / "chunks.jsonl")
    assert [c.chunk_id for c in records] == [c.chunk_id for c in idx.chunks]
    assert {c.content_type for c in records} == {ContentType.TEXT, ContentType.TABLE}
    lines = (out / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
    assert all(set(json.loads(x)) >= {"chunk_id", "page_number", "content_type", "table_data",
                                      "extraction_method", "ocr_used", "ocr_confidence"}
               for x in lines)
    docs = json.loads((out / "documents.json").read_text(encoding="utf-8"))
    assert [d["tables"] for d in docs["documents"]] == [0, 1]
    run = json.loads((out / "run_config.json").read_text(encoding="utf-8"))
    assert run["system"] == "C-trustqa"
