"""Module 2b - retrieval (PRD 3.3)."""

from __future__ import annotations

from trustqa.retrieval.index import SearchIndex, load_default_index
from trustqa.schemas import RetrievedChunk


def retrieve(
    question: str, top_k: int = 5, index: SearchIndex | None = None
) -> list[RetrievedChunk]:
    """Return the top_k chunks for `question`, best first (rank 1).

    Each result carries the original, unmodified Chunk record and its similarity score.
    If `index` is None the default index under config.indexes_dir is loaded.
    """
    if not question or not question.strip():
        raise ValueError("question must be non-empty")
    if index is None:
        index = load_default_index()
    return [
        RetrievedChunk(chunk=chunk, score=score, rank=rank)
        for rank, (chunk, score) in enumerate(index.search(question.strip(), top_k), start=1)
    ]
