import json
import re

import pytest

from trustqa import llm as llm_mod
from trustqa import pipeline
from trustqa.cli import main
from trustqa.config import load_config
from trustqa.llm import FakeLLM
from trustqa.retrieval import index as index_mod
from trustqa.schemas import Status

ID_IN_PROMPT = re.compile(r"^\[(DOC\d+_P\d+_C\d+)\]", re.M)


@pytest.fixture
def cfg(tmp_path):
    env = tmp_path / ".env"
    env.write_text("")
    return load_config(env, chunk_size=200, chunk_overlap=40, top_k=3,
                       indexes_dir=tmp_path / "idx", raw_pdfs_dir=tmp_path,
                       processed_dir=tmp_path / "processed")


@pytest.fixture(autouse=True)
def _clear_caches():
    pipeline._memory_cache.clear()
    yield
    pipeline._memory_cache.clear()


def cite_top_passage(prompt, system, json_mode):
    """Fake LLM that answers from, and cites, the first passage in the prompt."""
    top = ID_IN_PROMPT.findall(prompt)[0]
    return json.dumps({"answer": f"From the passage [{top}].", "chunk_ids": [top],
                       "insufficient_evidence": False})


def test_answer_question_end_to_end(solar_pdf, river_pdf, cfg, fake_embedder):
    llm_mod.set_llm(FakeLLM(cite_top_passage))
    r = pipeline.answer_question("When was the Mettur dam completed?", [solar_pdf, river_pdf], cfg)
    assert r.status is Status.PARTIALLY_SUPPORTED
    c = r.citations[0]
    assert (c.document_id, c.document_name, c.page_number) == ("DOC02", "river_survey.pdf", 2)


def test_index_is_cached_in_memory_and_on_disk(solar_pdf, river_pdf, cfg, fake_embedder,
                                               monkeypatch):
    llm_mod.set_llm(FakeLLM(cite_top_passage))
    builds = []
    real_build = pipeline.build_index
    monkeypatch.setattr(pipeline, "build_index", lambda ch, c: builds.append(1) or real_build(ch, c))
    docs = [solar_pdf, river_pdf]
    pipeline.answer_question("dam?", docs, cfg)
    pipeline.answer_question("revenue?", docs, cfg)
    assert len(builds) == 1  # second question reused the in-memory index
    pipeline._memory_cache.clear()
    pipeline.answer_question("dividend?", docs, cfg)
    assert len(builds) == 1  # loaded from data/indexes/<key>
    key = pipeline.index_key(docs, cfg)
    assert (cfg.indexes_dir / key / "index.faiss").is_file()
    assert (cfg.indexes_dir / index_mod.LATEST_POINTER).read_text() == key


def test_cache_key_changes_with_chunking_and_docs(solar_pdf, river_pdf, cfg):
    k = pipeline.index_key([solar_pdf], cfg)
    assert k != pipeline.index_key([solar_pdf, river_pdf], cfg)
    assert k != pipeline.index_key([solar_pdf], load_config(chunk_size=300, chunk_overlap=40))


def test_abstention_flows_through(solar_pdf, cfg, fake_embedder):
    llm_mod.set_llm(FakeLLM([json.dumps({"answer": "", "chunk_ids": [],
                                         "insufficient_evidence": True})]))
    r = pipeline.answer_question("Who won the 1998 World Cup?", [solar_pdf], cfg)
    assert r.status is Status.INSUFFICIENT_EVIDENCE and not r.citations


def test_text_less_pdf_gives_clear_error(tmp_path, cfg, fake_embedder):
    from tests.conftest import make_pdf

    blank = make_pdf(tmp_path / "scan.pdf", ["", ""])
    with pytest.raises(ValueError, match="no extractable text"):
        pipeline.answer_question("q", [blank], cfg)


def test_cli_prints_metadata_sources(solar_pdf, river_pdf, cfg, fake_embedder, monkeypatch,
                                     capsys):
    llm_mod.set_llm(FakeLLM(cite_top_passage))
    monkeypatch.setattr("trustqa.cli.load_config", lambda **kw: cfg)
    code = main(["Installed capacity of the Pavagada plant?", "--docs", str(solar_pdf),
                 str(river_pdf)])
    out = capsys.readouterr().out
    assert code == 0
    assert "Status: Partially Supported" in out
    assert "solar_report.pdf, page 4" in out


def test_cli_json_and_error_exit(solar_pdf, cfg, fake_embedder, monkeypatch, capsys, tmp_path):
    llm_mod.set_llm(FakeLLM(cite_top_passage))
    monkeypatch.setattr("trustqa.cli.load_config", lambda **kw: cfg)
    assert main(["dividend?", "--docs", str(solar_pdf), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "Partially Supported"
    assert main(["q", "--docs", str(tmp_path / "missing.pdf")]) == 2
    assert "not found" in capsys.readouterr().err
