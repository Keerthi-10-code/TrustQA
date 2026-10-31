"""Phase 5 demo back end: corpus preparation, caching, per-document errors, systems."""

import json
import re

import pymupdf
import pytest

from trustqa import corpus as corpus_mod
from trustqa import llm as llm_mod
from trustqa.config import ConfigError, load_config
from trustqa.corpus import (
    DUPLICATE,
    ENCRYPTED,
    NO_TEXT,
    UNREADABLE,
    CorpusError,
    prepare_corpus,
    save_upload,
    system_config,
)
from trustqa.llm import FakeLLM, LLMError, check_available
from trustqa.schemas import Status
from tests.conftest import make_pdf

ID_IN_PROMPT = re.compile(r"^\[(DOC\d+_P\d+_C\d+)\]", re.M)


@pytest.fixture
def dirs(tmp_path):
    return dict(chunk_size=200, chunk_overlap=40, top_k=3, ocr_enabled=False,
                processed_dir=tmp_path / "processed", indexes_dir=tmp_path / "idx",
                abstain_min_score=0.0)


def cite_top(prompt, system, json_mode):
    top = ID_IN_PROMPT.findall(prompt)[0]
    return json.dumps({"answer": f"From the passage [{top}].", "chunk_ids": [top],
                       "insufficient_evidence": False})


def encrypted_pdf(path):
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), "secret text")
    doc.save(path, encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="pw", owner_pw="pw")
    doc.close()
    return path


def test_systems_pin_their_switches(dirs, monkeypatch):
    monkeypatch.setenv("BASELINE_B", "1")  # stray .env settings must not change a system
    monkeypatch.setenv("OCR_ENABLED", "0")
    dirs = {k: v for k, v in dirs.items() if k != "ocr_enabled"}
    c = system_config("C", **dirs)
    assert c.verify and not c.flatten_tables and c.system_name == "C-trustqa"
    assert system_config("B", **dirs).system_name == "B-ocr-plain-rag"
    a = system_config("A", **dirs)
    assert a.system_name == "A-baseline" and not a.ocr_enabled
    with pytest.raises(ValueError):
        system_config("Z")


def test_prepare_reports_per_document_status_and_skips_bad_files(
        solar_pdf, river_pdf, tmp_path, dirs, fake_embedder):
    bad = tmp_path / "broken.pdf"
    bad.write_bytes(b"not a pdf at all")
    locked = encrypted_pdf(tmp_path / "locked.pdf")
    dup = tmp_path / "copy.pdf"
    dup.write_bytes(solar_pdf.read_bytes())
    blank = make_pdf(tmp_path / "blank.pdf", ["", ""])
    cfg = system_config("C", **dirs)
    events = []
    c = prepare_corpus([solar_pdf, bad, locked, dup, river_pdf, blank], cfg,
                       progress=lambda f, m: events.append((f, m)))
    by_name = {d.name: d for d in c.documents}
    assert [d.name for d in c.ready] == ["solar_report.pdf", "river_survey.pdf"]
    assert [d.document_id for d in c.ready] == ["DOC01", "DOC02"]
    assert by_name["broken.pdf"].error_kind == UNREADABLE
    assert by_name["locked.pdf"].error_kind == ENCRYPTED
    assert by_name["copy.pdf"].error_kind == DUPLICATE
    assert by_name["blank.pdf"].error_kind == NO_TEXT
    solar = by_name["solar_report.pdf"]
    assert solar.pages == 4 and solar.ocr_pages == [] and solar.text_chunks > 0
    assert events[-1] == (1.0, "Ready") and all(0 <= f <= 1 for f, _ in events)
    assert c.document_path("DOC02") == river_pdf
    assert c.chunk(c.index.chunks[0].chunk_id) is c.index.chunks[0]


def test_no_usable_document_raises(tmp_path, dirs):
    bad = tmp_path / "broken.pdf"
    bad.write_bytes(b"%PDF-garbage")
    with pytest.raises(CorpusError, match="broken.pdf"):
        prepare_corpus([bad], system_config("C", **dirs))
    with pytest.raises(CorpusError):
        prepare_corpus([], system_config("C", **dirs))


def test_documents_and_index_are_cached_by_file_hash(solar_pdf, river_pdf, dirs, fake_embedder,
                                                     monkeypatch):
    cfg = system_config("C", **dirs)
    first = prepare_corpus([solar_pdf, river_pdf], cfg)
    assert not any(d.from_cache for d in first.documents) and not first.index_from_cache

    processed, builds = [], []
    real_proc, real_build = corpus_mod.process_document, corpus_mod.build_index
    monkeypatch.setattr(corpus_mod, "process_document",
                        lambda *a, **k: processed.append(1) or real_proc(*a, **k))
    monkeypatch.setattr(corpus_mod, "build_index",
                        lambda ch, c: builds.append(1) or real_build(ch, c))
    again = prepare_corpus([solar_pdf, river_pdf], cfg)
    assert processed == [] and builds == []
    assert all(d.from_cache for d in again.documents) and again.index_from_cache
    assert [c.chunk_id for c in again.index.chunks] == [c.chunk_id for c in first.index.chunks]

    # adding a document reuses the cached processing of the others; only the index is new
    extra = make_pdf(solar_pdf.parent / "extra.pdf", ["A third document about tides."])
    third = prepare_corpus([solar_pdf, river_pdf, extra], cfg)
    assert len(processed) == 1 and len(builds) == 1
    assert [d.from_cache for d in third.documents] == [True, True, False]


def test_save_upload_is_content_addressed(tmp_path, dirs):
    cfg = system_config("C", **dirs)
    a = save_upload("../../evil name?.pdf", b"%PDF-1.4 x", cfg)
    b = save_upload("../../evil name?.pdf", b"%PDF-1.4 x", cfg)
    c = save_upload("report", b"%PDF-1.4 y", cfg)
    assert a == b and a.read_bytes() == b"%PDF-1.4 x"
    assert a.name == "evil name_.pdf" and a.parent.parent == corpus_mod.uploads_dir(cfg)
    assert c.name == "report.pdf" and c.parent != a.parent


def test_ask_runs_system_c_and_baseline(solar_pdf, river_pdf, dirs, fake_embedder):
    llm_mod.set_llm(FakeLLM(cite_top))
    c = prepare_corpus([solar_pdf, river_pdf], system_config("C", **dirs), "C")
    b = prepare_corpus([solar_pdf, river_pdf], system_config("A", **dirs), "A")
    q = "When was the Mettur dam completed?"
    ans_b = b.ask(q)
    assert ans_b.result.status is Status.PARTIALLY_SUPPORTED and ans_b.system == "A"
    assert ans_b.passages and ans_b.result.citations[0].document_name == "river_survey.pdf"
    # system C runs verification; the fake judge reply is not valid, so it cannot be Verified
    ans_c = c.ask(q)
    assert ans_c.result.status is not Status.VERIFIED
    with pytest.raises(ValueError):
        c.ask("  ")


def test_check_available_reports_missing_key_and_unreachable_ollama(dirs):
    with pytest.raises(ConfigError, match="GROQ_API_KEY"):
        check_available(load_config(llm_provider="groq", groq_api_key=None, **dirs))
    with pytest.raises(LLMError, match="cannot reach Ollama"):
        check_available(load_config(llm_provider="ollama", ollama_host="http://127.0.0.1:9",
                                    **dirs), timeout_s=1)
    llm_mod.set_llm(FakeLLM(["{}"]))
    assert check_available(load_config(llm_provider="groq", groq_api_key=None, **dirs)) == "override"
