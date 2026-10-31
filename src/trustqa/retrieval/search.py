"""Module 2b - retrieval (PRD 3.3).

Default: dense retrieval (cosine over normalised embeddings, FAISS). Phase 4 upgrades, each
behind a config flag and all off by default (see retrieval/ranking.py):
  retrieval_hybrid      BM25 + dense candidates merged by reciprocal rank fusion
  rerank                cross-encoder reorders the candidate pool
  retrieval_diversify   "quota" (max doc_quota chunks per document) or "mmr"
  retrieval_decompose   LLM sub-queries per entity, each retrieved, fused with the question
Order of application: (decompose) -> candidates per query -> fusion -> rerank -> diversify
-> top_k.
Every RetrievedChunk.score is the dense cosine of that chunk, whatever produced the order,
so score thresholds (abstention gate) mean the same with or without the upgrades; rank is
the position in the final order.
"""

from __future__ import annotations

import re

from trustqa import llm as llm_mod
from trustqa.config import Config, load_config
from trustqa.retrieval import ranking
from trustqa.retrieval.decompose import decompose
from trustqa.retrieval.index import SearchIndex, index_text, load_default_index
from trustqa.schemas import RetrievedChunk


def upgrades_enabled(config: Config) -> bool:
    return (config.retrieval_hybrid or config.rerank or config.retrieval_decompose
            or config.retrieval_diversify != "none" or config.retrieval_entity_scope)


_llm_cache: dict[tuple, llm_mod.LLM] = {}


def _default_llm(config: Config) -> llm_mod.LLM:
    override = llm_mod.overridden_llm()
    if override is not None:
        return override
    key = (config.llm_provider, config.llm_model, config.llm_temperature, config.ollama_host)
    if key not in _llm_cache:
        _llm_cache[key] = llm_mod.get_llm(config)
    return _llm_cache[key]


def retrieve(
    question: str, top_k: int = 5, index: SearchIndex | None = None
) -> list[RetrievedChunk]:
    """Return the top_k chunks for `question`, best first (rank 1).

    Each result carries the original, unmodified Chunk record and its similarity score.
    If `index` is None the default index under config.indexes_dir is loaded. Retrieval
    upgrades follow load_config() (.env); use retrieve_with_config to pass a Config.
    """
    return retrieve_with_config(question, top_k, index, None)


def retrieve_with_config(
    question: str, top_k: int = 5, index: SearchIndex | None = None,
    config: Config | None = None, llm: llm_mod.LLM | None = None,
) -> list[RetrievedChunk]:
    """retrieve() with an explicit Config selecting the Phase 4 upgrades (None ->
    load_config(); the Config defaults are all off). With every upgrade off this is the
    original dense retrieval, unchanged. `llm` is used only for retrieval_decompose
    (default: the configured provider, or the one installed with llm.set_llm)."""
    if not question or not question.strip():
        raise ValueError("question must be non-empty")
    if top_k < 1:
        raise ValueError(f"top_k must be >= 1, got {top_k}")
    question = question.strip()
    if index is None:
        index = load_default_index()
    config = config or load_config()
    if not upgrades_enabled(config):
        return [
            RetrievedChunk(chunk=chunk, score=score, rank=rank)
            for rank, (chunk, score) in enumerate(index.search(question, top_k), start=1)
        ]
    subqueries: list[str] = []
    if config.retrieval_decompose:
        subqueries = decompose(question, llm or _default_llm(config), config.max_subqueries)
    positions, cosine = _ranked_positions(question, index, config, top_k, subqueries)
    if config.retrieval_entity_scope:
        positions = _entity_scoped(question, index, config, positions, cosine)
    return [
        RetrievedChunk(chunk=index.chunks[p], score=float(cosine[p]), rank=rank)
        for rank, p in enumerate(positions[:top_k], start=1)
    ]


def _entity_scoped(question: str, index: SearchIndex, config: Config, positions: list[int],
                   cosine) -> list[int]:
    """Give each document-identifying name in the question (trustqa/scope.py) a share of the
    first config.top_k slots, filled from its documents' best chunks (same dense / hybrid
    fusion over the whole index, queried with the question minus the *other* documents'
    names); the rest follow the normal order. Only the first top_k slots change, so a
    deeper retrieval keeps the same top_k (the eval retrieves 10 and slices 5)."""
    from trustqa.scope import scope_for

    groups: dict[frozenset, list[str]] = {}
    for name, docs in scope_for(index).identifying(question).items():
        groups.setdefault(frozenset(docs), []).append(name)
    if not groups:
        return positions
    k = min(config.top_k, len(positions))
    n = len(index)
    order_all, _ = _first_stage(question, index, config, n, cosine)
    doc_of = [c.document_id for c in index.chunks]
    lists = []
    for g, names in groups.items():
        others = [nm for g2, nms in groups.items() if g2 != g for nm in nms]
        query = question
        for nm in others:
            query = re.sub(r"(?<!\w)" + re.escape(nm) + r"(?:'s|’s)?(?!\w)", " ", query)
        query = re.sub(r"\s+", " ", query).strip()
        order_g = _first_stage(query, index, config, n)[0] if others else order_all
        lists.append([p for p in order_g if doc_of[p] in g])
    per = max(1, -(-k // (len(groups) + 1)))  # ceil(k / (groups + 1))
    picks: list[int] = []
    for r in range(per):  # round robin, so every group gets its first pick
        for lst in lists:
            if r < len(lst) and lst[r] not in picks and len(picks) < k:
                picks.append(lst[r])
    for p in positions:
        if len(picks) >= k:
            break
        if p not in picks:
            picks.append(p)
    rank = {p: i for i, p in enumerate(order_all)}
    head = sorted(picks, key=lambda p: rank.get(p, n))
    return head + [p for p in positions if p not in head]


def _first_stage(query: str, index: SearchIndex, config: Config, n: int, cosine=None):
    """Candidate order and relevance for one query: dense, or dense + BM25 fused by RRF."""
    if cosine is None:
        cosine = index.vectors() @ index.embed_query(query)
    dense = ranking.top_positions(cosine, n)
    if config.retrieval_hybrid:
        bm25 = ranking.top_positions(ranking.bm25_scores(index, query), n, positive_only=True)
        fused = ranking.rrf([dense, bm25], k=config.rrf_k)
        return [p for p, _ in fused], [s for _, s in fused]
    return dense, [float(cosine[p]) for p in dense]


def _ranked_positions(question: str, index: SearchIndex, config: Config, top_k: int,
                      subqueries: list[str] | None = None):
    vectors = index.vectors()
    cosine = vectors @ index.embed_query(question)  # reported scores: always vs. the question
    n = min(max(config.retrieval_candidates, top_k), len(index))

    order, relevance = _first_stage(question, index, config, n, cosine)
    if subqueries:
        lists = [order] + [_first_stage(q, index, config, n)[0] for q in subqueries]
        fused = ranking.rrf(lists, k=config.rrf_k)
        order = [p for p, _ in fused]
        relevance = [s for _, s in fused]

    if config.rerank:
        ce = ranking.rerank_scores(config.reranker_model, question,
                                   [index_text(index.chunks[p]) for p in order])
        pairs = sorted(zip(order, ce), key=lambda t: -t[1])  # stable: ties keep prior order
        order = [p for p, _ in pairs]
        relevance = [float(s) for _, s in pairs]

    if config.retrieval_diversify == "quota":
        order = ranking.doc_quota(order, [c.document_id for c in index.chunks], config.doc_quota)
    elif config.retrieval_diversify == "mmr":
        order = ranking.mmr(order, relevance, vectors, config.mmr_lambda)
    return order, cosine
