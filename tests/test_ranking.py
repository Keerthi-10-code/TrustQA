"""Phase 4 retrieval upgrades: BM25 tokenisation, RRF, per-document quota, MMR, reranking,
and their wiring in retrieve_with_config (flags off == original dense retrieval)."""

from __future__ import annotations

import numpy as np
import pytest

from trustqa.config import ConfigError, load_config
from trustqa.processing.process import process_pdfs
from trustqa.retrieval import ranking
from trustqa.retrieval.index import build_index
from trustqa.retrieval.search import retrieve, retrieve_with_config


@pytest.fixture
def cfg(tmp_path):
    env = tmp_path / ".env"
    env.write_text("")
    return load_config(env, chunk_size=200, chunk_overlap=40, indexes_dir=tmp_path / "idx")


@pytest.fixture
def index(solar_pdf, river_pdf, cfg, fake_embedder):
    return build_index(process_pdfs([solar_pdf, river_pdf], cfg), cfg)


def with_flags(cfg, **flags):
    from dataclasses import replace
    return replace(cfg, **flags)


# --- pure functions -------------------------------------------------------------------

def test_tokenize_keeps_compounds_parts_and_decimals():
    toks = ranking.tokenize("nDCG@10 of TAS-B on FiQA-2018 is 0.300 (Touché-2020)")
    for t in ("ndcg@10", "ndcg", "10", "tas-b", "tas", "fiqa-2018", "fiqa", "2018", "0.300",
              "touche-2020", "touche"):
        assert t in toks
    assert "of" not in toks and "is" not in toks
    assert "300" not in toks  # decimals are never split


def test_rrf_sums_reciprocal_ranks_and_breaks_ties_deterministically():
    fused = ranking.rrf([[1, 2, 3], [3, 1, 4]], k=60)
    order = [p for p, _ in fused]
    assert order[0] == 1  # 1/61 + 1/62 beats 3: 1/63 + 1/61
    assert order[1] == 3
    assert set(order) == {1, 2, 3, 4}
    # 2 (rank 2 in first list) and 4 (rank 3 in second) differ; equal scores keep list order
    tie = ranking.rrf([[5, 6], [6, 5]], k=60)
    assert [p for p, _ in tie] == [5, 6]


def test_doc_quota_demotes_but_never_drops():
    docs = ["A", "A", "A", "B", "C"]
    assert ranking.doc_quota([0, 1, 2, 3, 4], docs, 2) == [0, 1, 3, 4, 2]
    assert ranking.doc_quota([0, 1, 2, 3, 4], docs, 1) == [0, 3, 4, 1, 2]


def test_mmr_prefers_a_diverse_second_pick():
    vecs = np.array([[1, 0], [1, 0], [0, 1]], dtype=float)  # 0 and 1 are duplicates
    assert ranking.mmr([0, 1, 2], [1.0, 0.95, 0.9], vecs, lam=0.5)[:2] == [0, 2]
    assert ranking.mmr([0, 1, 2], [1.0, 0.95, 0.9], vecs, lam=1.0) == [0, 1, 2]


# --- wiring ---------------------------------------------------------------------------

def test_flags_default_off_and_retrieve_is_unchanged(index, cfg):
    assert not (cfg.retrieval_hybrid or cfg.rerank) and cfg.retrieval_diversify == "none"
    a = retrieve("Pavagada plant installed capacity", 4, index)
    b = retrieve_with_config("Pavagada plant installed capacity", 4, index, cfg)
    assert [(r.chunk.chunk_id, r.score, r.rank) for r in a] == \
           [(r.chunk.chunk_id, r.score, r.rank) for r in b]
    assert [r.score for r in a] == sorted((r.score for r in a), reverse=True)


@pytest.mark.parametrize("flags", [
    dict(retrieval_hybrid=True),
    dict(retrieval_diversify="quota", doc_quota=1),
    dict(retrieval_diversify="mmr"),
    dict(retrieval_hybrid=True, retrieval_diversify="quota"),
])
def test_upgraded_scores_are_dense_cosine_and_prefix_consistent(index, cfg, flags):
    c = with_flags(cfg, **flags)
    q = "When was the Mettur dam completed?"
    long = retrieve_with_config(q, 6, index, c)
    short = retrieve_with_config(q, 3, index, c)
    assert [r.chunk.chunk_id for r in short] == [r.chunk.chunk_id for r in long[:3]]
    assert [r.rank for r in long] == list(range(1, len(long) + 1))
    qv = index.embed_query(q)
    pos = {ch.chunk_id: i for i, ch in enumerate(index.chunks)}
    for r in long:  # score is the chunk's cosine, whatever ordered it (abstention gate)
        assert r.score == pytest.approx(float(index.vectors()[pos[r.chunk.chunk_id]] @ qv),
                                        abs=1e-5)
        assert r.chunk in index.chunks


def test_hybrid_finds_exact_term_and_keeps_known_answer(index, cfg):
    c = with_flags(cfg, retrieval_hybrid=True)
    top = retrieve_with_config("When was the Mettur dam completed?", 3, index, c)[0].chunk
    assert (top.document_name, top.page_number) == ("river_survey.pdf", 2)


def test_quota_limits_chunks_per_document_in_top_k(index, cfg):
    c = with_flags(cfg, retrieval_diversify="quota", doc_quota=1)
    res = retrieve_with_config("solar revenue dividend capacity", 2, index, c)
    assert len({r.chunk.document_id for r in res}) == 2


def test_rerank_reorders_candidates(index, cfg, monkeypatch):
    texts_seen = []

    def fake_scores(model, query, texts):
        texts_seen.append(len(texts))
        return np.array([1.0 if "Mettur" in t else 0.0 for t in texts])

    monkeypatch.setattr(ranking, "rerank_scores", fake_scores)
    c = with_flags(cfg, rerank=True)
    top = retrieve_with_config("solar farms in Karnataka", 2, index, c)[0].chunk
    assert "Mettur" in top.text
    assert texts_seen and texts_seen[0] == len(index)  # whole pool (< retrieval_candidates)


def test_config_validation_and_baseline_b_forces_upgrades_off(tmp_path):
    env = tmp_path / ".env"
    env.write_text("")
    with pytest.raises(ConfigError):
        load_config(env, retrieval_diversify="cluster")
    with pytest.raises(ConfigError):
        load_config(env, mmr_lambda=1.5)
    b = load_config(env, baseline_b=True, retrieval_hybrid=True, rerank=True,
                    retrieval_diversify="mmr")
    assert not b.retrieval_hybrid and not b.rerank and b.retrieval_diversify == "none"


def test_env_flags_are_read(tmp_path, monkeypatch):
    # setenv (not a .env file): load_dotenv would leave the flags in os.environ for later tests
    env = tmp_path / ".env"
    env.write_text("")
    for k, v in {"RETRIEVAL_HYBRID": "1", "RETRIEVAL_DIVERSIFY": "quota", "DOC_QUOTA": "3",
                 "RRF_K": "10"}.items():
        monkeypatch.setenv(k, v)
    c = load_config(env)
    assert (c.retrieval_hybrid, c.retrieval_diversify, c.doc_quota, c.rrf_k) == \
           (True, "quota", 3, 10)


# --- baselines A and B are protected -------------------------------------------------

def test_baseline_a_also_forces_upgrades_off(tmp_path):
    env = tmp_path / ".env"
    env.write_text("")
    a = load_config(env, ocr_enabled=False, flatten_tables=True, verify=False,
                    retrieval_hybrid=True, retrieval_decompose=True, retrieval_diversify="quota")
    assert a.system_name == "A-baseline" and a.is_frozen_baseline
    assert not (a.retrieval_hybrid or a.retrieval_decompose or a.rerank)
    assert a.retrieval_diversify == "none"
    c = load_config(env, retrieval_hybrid=True, retrieval_decompose=True)
    assert c.system_name == "C-trustqa" and c.retrieval_hybrid and c.retrieval_decompose


def test_tests_ignore_developer_env_flags():
    # conftest pins the flags, so a .env with RETRIEVAL_HYBRID=1 cannot change test results
    c = load_config()
    assert not (c.retrieval_hybrid or c.rerank or c.retrieval_decompose)
    assert c.retrieval_diversify == "none"


# --- query decomposition (F12) ---------------------------------------------------------

from trustqa.llm import FakeLLM, LLMError  # noqa: E402
from trustqa.retrieval.decompose import decompose, parse_subqueries  # noqa: E402

Q = "Compare the dam completion year and the solar plant capacity."


@pytest.mark.parametrize("raw, expected", [
    ('{"queries": ["When was the Mettur dam completed?", "Pavagada plant capacity"]}',
     ["When was the Mettur dam completed?", "Pavagada plant capacity"]),
    (f'{{"queries": ["{Q}"]}}', []),                       # one query: no decomposition
    ('{"queries": ["only a rewrite"]}', []),                # single rewrite: ignored
    ('not json', []),
    ('{"queries": "a string"}', []),
    ('{"queries": ["a", 3]}', []),
    ('{"queries": ["a", ""]}', []),
    ('{"queries": ["a", "b", "c", "d"]}', []),              # more than max_n
    ('{"queries": ["' + "x" * 400 + '", "b"]}', []),        # overlong
    (f'{{"queries": ["{Q}", "Pavagada capacity", "pavagada  capacity"]}}',
     ["Pavagada capacity"]),                                # echo and duplicates dropped
])
def test_parse_subqueries_validates_llm_output(raw, expected):
    assert parse_subqueries(raw, Q, 3) == expected


def test_decompose_llm_failure_falls_back():
    def boom(prompt, system, json):
        raise LLMError("ollama down")
    assert decompose(Q, FakeLLM(boom)) == []


def test_decompose_retrieves_both_documents(index, cfg):
    q = "Compare the reservoir of the dam and the dividend of the energy company"
    fake = FakeLLM(['{"queries": ["When was the Mettur dam completed and what does its reservoir '
                    'support?", "What dividend did the board of Helios Energy approve?"]}'])
    c = with_flags(cfg, retrieval_decompose=True)
    res = retrieve_with_config(q, 2, index, c, llm=fake)
    assert len(fake.calls) == 1 and fake.calls[0]["json"]
    assert {r.chunk.document_name for r in res} == {"solar_report.pdf", "river_survey.pdf"}
    qv = index.embed_query(q)  # scores stay the cosine to the original question
    pos = {ch.chunk_id: i for i, ch in enumerate(index.chunks)}
    for r in res:
        assert r.score == pytest.approx(float(index.vectors()[pos[r.chunk.chunk_id]] @ qv),
                                        abs=1e-5)


def test_decompose_without_split_equals_base_retrieval(index, cfg):
    q = "When was the Mettur dam completed?"
    fake = FakeLLM([f'{{"queries": ["{q}"]}}'])
    with_dec = retrieve_with_config(q, 4, index, with_flags(cfg, retrieval_decompose=True), llm=fake)
    # decompose alone enables the upgraded path: same order as dense over the candidate pool
    plain = retrieve_with_config(q, 4, index, cfg)
    assert [r.chunk.chunk_id for r in with_dec] == [r.chunk.chunk_id for r in plain]


def test_decompose_uses_installed_llm_by_default(index, cfg):
    from trustqa import llm as llm_mod
    fake = FakeLLM(['{"queries": ["x"]}'])
    llm_mod.set_llm(fake)
    retrieve_with_config("anything about dams", 2, index, with_flags(cfg, retrieval_decompose=True))
    assert len(fake.calls) == 1
