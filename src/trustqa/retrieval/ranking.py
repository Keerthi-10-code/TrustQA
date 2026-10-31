"""Phase 4 retrieval upgrades: BM25, reciprocal rank fusion, per-document quota, MMR and an
optional cross-encoder reranker. Each step works on chunk positions (indices into
SearchIndex.chunks) and is greedy over a fixed candidate pool, so the first k results do not
depend on how many are requested (evaluating at depth 10 gives the same top 5 as top_k=5).

All of this only reorders chunks. search.retrieve() still reports the dense cosine as the
score, so thresholds calibrated on it (abstention gate) keep their meaning.
"""

from __future__ import annotations

import re
import unicodedata
import weakref
from collections.abc import Sequence
from functools import lru_cache

import numpy as np

from trustqa.retrieval.index import index_text

_TOKEN = re.compile(r"[a-z0-9]+(?:[.\-+@/_^][a-z0-9]+)*")
_SPLIT = re.compile(r"[.\-+@/_^]")
_NUMBER = re.compile(r"\d+(?:\.\d+)?")
STOPWORDS = frozenset("""
a an and are as at be been but by can could did do does for from had has have how in into is
it its of on or our over than that the their them then there these they this those to under
up was we were what when where which while who whom why will with would you your
""".split())


def tokenize(text: str) -> list[str]:
    """Lowercase, accent-stripped tokens for BM25.

    Compound tokens such as "fiqa-2018", "tas-b", "bm25+ce" and "ndcg@10" are kept whole
    *and* split into their parts, so both exact names and their pieces match. Decimal
    numbers ("0.300") are kept whole and never split.
    """
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    out: list[str] = []
    for m in _TOKEN.finditer(text):
        tok = m.group()
        if tok in STOPWORDS:
            continue
        out.append(tok)
        if not _NUMBER.fullmatch(tok):
            parts = [p for p in _SPLIT.split(tok) if p and p not in STOPWORDS]
            if len(parts) > 1:
                out.extend(parts)
    return out


_bm25_cache: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def bm25_scores(index, query: str) -> np.ndarray:
    """BM25 (Okapi) score of every chunk in `index` for `query`; built once per index."""
    bm25 = _bm25_cache.get(index)
    if bm25 is None:
        from rank_bm25 import BM25Okapi

        bm25 = BM25Okapi([tokenize(index_text(c)) or ["_empty_"] for c in index.chunks])
        _bm25_cache[index] = bm25
    q = tokenize(query)
    if not q:
        return np.zeros(len(index.chunks))
    return np.asarray(bm25.get_scores(q), dtype=float)


def top_positions(scores: np.ndarray, n: int, positive_only: bool = False) -> list[int]:
    """Positions of the n highest scores, best first; ties broken by position (stable)."""
    order = np.argsort(-scores, kind="stable")[:n]
    if positive_only:
        order = [i for i in order if scores[i] > 0]
    return [int(i) for i in order]


def rrf(rankings: Sequence[Sequence[int]], k: int = 60) -> list[tuple[int, float]]:
    """Reciprocal rank fusion: sum over rankings of 1 / (k + rank), rank 1-based.

    Returns (position, fused score) best first. Ties are broken by the earliest appearance in
    the first ranking, then the second, ... (deterministic).
    """
    fused: dict[int, float] = {}
    first_seen: dict[int, tuple[int, int]] = {}
    for r_i, ranking in enumerate(rankings):
        for rank, pos in enumerate(ranking, start=1):
            fused[pos] = fused.get(pos, 0.0) + 1.0 / (k + rank)
            first_seen.setdefault(pos, (r_i, rank))
    return sorted(fused.items(), key=lambda kv: (-kv[1], first_seen[kv[0]]))


def doc_quota(order: Sequence[int], doc_of: Sequence[str], quota: int) -> list[int]:
    """Greedy pass keeping at most `quota` chunks per document; chunks over the quota are
    appended afterwards in their original order (so nothing is dropped, only demoted)."""
    kept, demoted, counts = [], [], {}
    for pos in order:
        d = doc_of[pos]
        if counts.get(d, 0) < quota:
            counts[d] = counts.get(d, 0) + 1
            kept.append(pos)
        else:
            demoted.append(pos)
    return kept + demoted


def mmr(order: Sequence[int], relevance: Sequence[float], vectors: np.ndarray,
        lam: float) -> list[int]:
    """Maximal marginal relevance over `order`.

    relevance[i] is the relevance of order[i] (any scale; min-max normalised here);
    vectors are unit-length chunk embeddings (cosine = dot product). Greedily picks
    argmax lam * rel - (1 - lam) * max cosine to the already picked chunks.
    """
    if not order:
        return []
    rel = np.asarray(relevance, dtype=float)
    span = rel.max() - rel.min()
    rel = (rel - rel.min()) / span if span > 0 else np.ones_like(rel)
    vecs = vectors[list(order)]
    sims = vecs @ vecs.T
    picked: list[int] = []
    max_sim = np.full(len(order), -np.inf)
    remaining = np.ones(len(order), dtype=bool)
    for _ in range(len(order)):
        redundancy = np.where(np.isfinite(max_sim), max_sim, 0.0)
        mmr_score = np.where(remaining, lam * rel - (1 - lam) * redundancy, -np.inf)
        j = int(np.argmax(mmr_score))  # first max -> earlier candidate wins ties
        picked.append(j)
        remaining[j] = False
        max_sim = np.maximum(max_sim, sims[j])
    return [order[j] for j in picked]


@lru_cache(maxsize=2)
def get_cross_encoder(model_name: str):
    """Load (once per process) a sentence-transformers CrossEncoder."""
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name)


def rerank_scores(model_name: str, query: str, texts: Sequence[str]) -> np.ndarray:
    if not texts:
        return np.zeros(0)
    model = get_cross_encoder(model_name)
    return np.asarray(model.predict([(query, t) for t in texts], show_progress_bar=False),
                      dtype=float)
