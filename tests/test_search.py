from pathlib import Path

import pytest

from trustqa.config import load_config
from trustqa.processing.process import process_pdfs
from trustqa.retrieval import index as index_mod
from trustqa.retrieval.index import SearchIndex, build_index, mark_latest
from trustqa.retrieval.search import retrieve

ROOT = Path(__file__).resolve().parents[1]
SELFRAG = ROOT / "data" / "raw_pdfs" / "text" / "selfrag_asai2023.pdf"


@pytest.fixture
def cfg(tmp_path):
    env = tmp_path / ".env"
    env.write_text("")
    return load_config(env, chunk_size=200, chunk_overlap=40, indexes_dir=tmp_path / "idx")


@pytest.fixture
def index(solar_pdf, river_pdf, cfg, fake_embedder):
    return build_index(process_pdfs([solar_pdf, river_pdf], cfg), cfg)


def test_known_question_retrieves_known_chunk(index):
    res = retrieve("When was the Mettur dam completed?", top_k=3, index=index)
    top = res[0].chunk
    assert (top.document_id, top.page_number, top.document_name) == ("DOC02", 2, "river_survey.pdf")
    assert "1934" in top.text


def test_results_are_original_records_ranked(index):
    res = retrieve("Pavagada plant installed capacity", top_k=4, index=index)
    assert [r.rank for r in res] == [1, 2, 3, 4]
    assert [r.score for r in res] == sorted((r.score for r in res), reverse=True)
    for r in res:
        assert r.chunk in index.chunks  # unmodified record from the index
    assert res[0].chunk.chunk_id.startswith("DOC01_P4_")


def test_default_index_loaded_from_latest_pointer(index, cfg, monkeypatch, fake_embedder):
    target = Path(cfg.indexes_dir) / "abc123"
    index.save(target)
    mark_latest(cfg.indexes_dir, target)
    monkeypatch.setattr(index_mod, "load_config", lambda: cfg)
    loaded = index_mod.load_default_index()
    assert loaded.chunks == index.chunks


def test_missing_default_index_is_explicit(cfg, monkeypatch):
    monkeypatch.setattr(index_mod, "load_config", lambda: cfg)
    with pytest.raises(FileNotFoundError, match="no default index"):
        retrieve("anything")


def test_empty_question_rejected(index):
    with pytest.raises(ValueError):
        retrieve("   ", index=index)


# --- real embedding model (skipped if it is not available locally) -----------------

@pytest.fixture(scope="module")
def real_embedder():
    try:
        return index_mod.get_embedder(load_config().embedding_model)
    except Exception as e:  # no network and not cached
        pytest.skip(f"embedding model unavailable: {e}")


def test_real_model_known_question(solar_pdf, river_pdf, cfg, real_embedder):
    chunks = process_pdfs([solar_pdf, river_pdf], cfg)
    idx = SearchIndex.from_chunks(chunks, real_embedder, cfg.embedding_model)
    top = retrieve("What is the dividend approved by the board?", 1, idx)[0].chunk
    assert (top.document_id, top.page_number) == ("DOC01", 3)
    top = retrieve("Which months bring monsoon rain to the river basin?", 1, idx)[0].chunk
    assert (top.document_id, top.page_number) == ("DOC02", 1)


@pytest.mark.skipif(not SELFRAG.is_file(), reason="pilot PDF not present (git-ignored)")
def test_real_model_pilot_pdf(real_embedder):
    cfg = load_config()
    idx = SearchIndex.from_chunks(process_pdfs([SELFRAG], cfg), real_embedder,
                                  cfg.embedding_model)
    res = retrieve("Definitions of the reflection tokens used by Self-RAG", 3, idx)
    # Appendix A.1 "Reflection tokens" is on PDF page 17
    assert 17 in [r.chunk.page_number for r in res]
