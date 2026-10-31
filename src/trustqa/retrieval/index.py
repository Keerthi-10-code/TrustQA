"""Module 2a - embeddings and vector index (PRD 3.3)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from trustqa.config import Config
from trustqa.schemas import Chunk


class SearchIndex:
    """FAISS inner-product index over normalised embeddings, with chunks stored aligned
    to vector positions so retrieval returns the original records."""

    def save(self, directory: str | Path) -> None:
        raise NotImplementedError("Phase 1")

    @classmethod
    def load(cls, directory: str | Path) -> SearchIndex:
        raise NotImplementedError("Phase 1")


def build_index(chunks: Sequence[Chunk], config: Config | None = None) -> SearchIndex:
    """Embed chunk.text with Sentence-Transformers (config.embedding_model, normalised)
    and build a SearchIndex. Raises ValueError on an empty chunk list or duplicate
    chunk_ids."""
    raise NotImplementedError("Phase 1")
