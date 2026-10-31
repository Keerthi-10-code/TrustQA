"""Module 2a - embeddings and vector index (PRD 3.3).

Embeddings are L2-normalised, so FAISS inner product == cosine similarity. Chunk records
are stored in a list aligned with vector positions (vector i <-> chunks[i]), so a search
returns the original records unchanged.

On disk (one directory per index):
  index.faiss   serialised FAISS IndexFlatIP
  chunks.jsonl  one Chunk.to_json() per line, in vector order
  meta.json     embedding model, dimension, count, chunk_id order
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
from typing import Protocol

import faiss
import numpy as np

from trustqa.config import Config, load_config
from trustqa.schemas import Chunk, ContentType

logger = logging.getLogger("trustqa")

INDEX_FORMAT_VERSION = 1


def index_text(chunk: Chunk) -> str:
    """Text that retrieval (embeddings, BM25, reranker) sees for a chunk: the chunk text,
    prefixed by its section path for text chunks that carry one (config.section_headings).
    Table chunks already carry caption/headers in their text."""
    if chunk.section and chunk.content_type is ContentType.TEXT:
        return f"{chunk.section}\n{chunk.text}"
    return chunk.text


class Embedder(Protocol):
    def encode(self, texts: list[str]) -> np.ndarray:
        """Return a (len(texts), dim) float array."""
        ...


class SentenceTransformerEmbedder:
    """Sentence-Transformers model producing normalised float32 embeddings."""

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self._model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        return self._model.encode(
            list(texts), batch_size=32, normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=False,
        ).astype(np.float32)


@lru_cache(maxsize=2)
def get_embedder(model_name: str) -> Embedder:
    """Load (once per process) the embedding model named in config."""
    return SentenceTransformerEmbedder(model_name)


def _as_unit_rows(vectors: np.ndarray) -> np.ndarray:
    v = np.ascontiguousarray(np.asarray(vectors, dtype=np.float32))
    if v.ndim != 2:
        raise ValueError(f"embedder returned shape {v.shape}, expected 2-D")
    faiss.normalize_L2(v)  # idempotent for already-normalised vectors
    return v


class SearchIndex:
    """FAISS inner-product index over normalised embeddings, with chunks stored aligned
    to vector positions so retrieval returns the original records."""

    def __init__(self, index: faiss.Index, chunks: list[Chunk], embedding_model: str,
                 embedder: Embedder | None = None):
        if index.ntotal != len(chunks):
            raise ValueError(f"index has {index.ntotal} vectors but {len(chunks)} chunks")
        self.index = index
        self.chunks = chunks
        self.embedding_model = embedding_model
        self._embedder = embedder

    def __len__(self) -> int:
        return len(self.chunks)

    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = get_embedder(self.embedding_model)
        return self._embedder

    @classmethod
    def from_chunks(cls, chunks: Sequence[Chunk], embedder: Embedder,
                    embedding_model: str) -> SearchIndex:
        chunks = list(chunks)
        if not chunks:
            raise ValueError("cannot build an index from zero chunks")
        ids = [c.chunk_id for c in chunks]
        if len(set(ids)) != len(ids):
            dupes = sorted({i for i in ids if ids.count(i) > 1})
            raise ValueError(f"duplicate chunk_ids: {dupes[:10]}")
        vectors = _as_unit_rows(embedder.encode([index_text(c) for c in chunks]))
        if len(vectors) != len(chunks):
            raise ValueError("embedder returned a different number of vectors than chunks")
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        return cls(index, chunks, embedding_model, embedder)

    def search(self, query: str, top_k: int) -> list[tuple[Chunk, float]]:
        """Best-first (chunk, cosine score) pairs; at most min(top_k, len(self))."""
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        q = _as_unit_rows(self.embedder.encode([query]))
        if q.shape[1] != self.index.d:
            raise ValueError(f"query dim {q.shape[1]} != index dim {self.index.d}; "
                             f"was the index built with {self.embedding_model}?")
        scores, idx = self.index.search(q, min(top_k, len(self)))
        return [(self.chunks[i], float(s)) for i, s in zip(idx[0], scores[0]) if i >= 0]

    def embed_query(self, query: str) -> np.ndarray:
        """Unit-length query vector, shape (dim,)."""
        q = _as_unit_rows(self.embedder.encode([query]))
        if q.shape[1] != self.index.d:
            raise ValueError(f"query dim {q.shape[1]} != index dim {self.index.d}; "
                             f"was the index built with {self.embedding_model}?")
        return q[0]

    def vectors(self) -> np.ndarray:
        """All stored chunk vectors, (len(self), dim), aligned with self.chunks (cached)."""
        v = getattr(self, "_vectors", None)
        if v is None:
            v = self.index.reconstruct_n(0, self.index.ntotal)
            self._vectors = v
        return v

    def save(self, directory: str | Path) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        # serialise via numpy rather than faiss.write_index: avoids FAISS path-encoding
        # issues on Windows.
        (d / "index.faiss").write_bytes(faiss.serialize_index(self.index).tobytes())
        with open(d / "chunks.jsonl", "w", encoding="utf-8") as f:
            for c in self.chunks:
                f.write(c.to_json() + "\n")
        meta = {
            "format_version": INDEX_FORMAT_VERSION,
            "embedding_model": self.embedding_model,
            "dim": self.index.d,
            "count": len(self.chunks),
            "metric": "inner_product_on_l2_normalised",
            "chunk_ids": [c.chunk_id for c in self.chunks],
        }
        (d / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, directory: str | Path, embedder: Embedder | None = None) -> SearchIndex:
        d = Path(directory)
        for name in ("index.faiss", "chunks.jsonl", "meta.json"):
            if not (d / name).is_file():
                raise FileNotFoundError(f"index at {d} is missing {name}")
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        if meta.get("format_version") != INDEX_FORMAT_VERSION:
            raise ValueError(f"index at {d} has format {meta.get('format_version')}, "
                             f"expected {INDEX_FORMAT_VERSION}; rebuild it")
        index = faiss.deserialize_index(
            np.frombuffer((d / "index.faiss").read_bytes(), dtype=np.uint8))
        with open(d / "chunks.jsonl", encoding="utf-8") as f:
            chunks = [Chunk.from_json(line) for line in f if line.strip()]
        if [c.chunk_id for c in chunks] != meta["chunk_ids"] or index.ntotal != len(chunks):
            raise ValueError(f"index at {d} is inconsistent (vectors/chunks misaligned)")
        return cls(index, chunks, meta["embedding_model"], embedder)


LATEST_POINTER = "LATEST"


def mark_latest(indexes_dir: str | Path, index_dir: str | Path) -> None:
    """Record `index_dir` (a subdirectory of indexes_dir) as the default index."""
    root = Path(indexes_dir)
    root.mkdir(parents=True, exist_ok=True)
    (root / LATEST_POINTER).write_text(Path(index_dir).name, encoding="utf-8")


def load_default_index(config: Config | None = None) -> SearchIndex:
    """Load the index most recently built by the pipeline under config.indexes_dir."""
    config = config or load_config()
    pointer = Path(config.indexes_dir) / LATEST_POINTER
    if not pointer.is_file():
        raise FileNotFoundError(
            f"no default index under {config.indexes_dir}; build one first, e.g. "
            f'python -m trustqa.cli "your question"')
    return SearchIndex.load(Path(config.indexes_dir) / pointer.read_text(encoding="utf-8").strip())


def build_index(chunks: Sequence[Chunk], config: Config | None = None) -> SearchIndex:
    """Embed chunk.text with Sentence-Transformers (config.embedding_model, normalised)
    and build a SearchIndex. Raises ValueError on an empty chunk list or duplicate
    chunk_ids."""
    config = config or load_config()
    embedder = get_embedder(config.embedding_model)
    idx = SearchIndex.from_chunks(chunks, embedder, config.embedding_model)
    logger.info("built index: %d chunks, dim %d, model %s", len(idx), idx.index.d,
                config.embedding_model)
    return idx
