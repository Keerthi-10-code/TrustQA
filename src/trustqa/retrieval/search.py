"""Module 2b - retrieval (PRD 3.3)."""

from __future__ import annotations

from trustqa.retrieval.index import SearchIndex
from trustqa.schemas import RetrievedChunk


def retrieve(
    question: str, top_k: int = 5, index: SearchIndex | None = None
) -> list[RetrievedChunk]:
    """Return the top_k chunks for `question`, best first (rank 1).

    Each result carries the original, unmodified Chunk record and its similarity score.
    If `index` is None the default index under config.indexes_dir is loaded.
    """
    raise NotImplementedError("Phase 1")
