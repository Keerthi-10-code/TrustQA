"""Module 4 - integration (PRD 3.3)."""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Sequence
from pathlib import Path

from trustqa import llm as llm_mod
from trustqa.config import Config, load_config, log_run_config
from trustqa.generation import PROMPT_VERSION
from trustqa.processing.extract import EXTRACTOR_VERSION, file_sha256
from trustqa.processing.process import process_pdfs
from trustqa.retrieval.index import (
    INDEX_FORMAT_VERSION,
    SearchIndex,
    build_index,
    mark_latest,
)
from trustqa.retrieval.search import retrieve
from trustqa.schemas import Result
from trustqa.verification.verify import generate_and_verify

logger = logging.getLogger("trustqa")

CHUNKER_VERSION = "sentence-v1"
SYSTEM_NAME = "A-baseline"  # Phase 1: plain-text RAG, no OCR/tables/verification

_memory_cache: dict[str, SearchIndex] = {}
_llm_cache: dict[tuple, llm_mod.LLM] = {}


def index_key(doc_paths: Sequence[str | Path], config: Config) -> str:
    """Cache key: document contents (in order) + every setting that changes the index."""
    parts = {
        "docs": [file_sha256(p) for p in doc_paths],
        "chunk_size": config.chunk_size,
        "chunk_overlap": config.chunk_overlap,
        "embedding_model": config.embedding_model,
        "extractor": EXTRACTOR_VERSION,
        "chunker": CHUNKER_VERSION,
        "index_format": INDEX_FORMAT_VERSION,
    }
    return hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()[:16]


def get_or_build_index(doc_paths: Sequence[str | Path], config: Config,
                       rebuild: bool = False) -> SearchIndex:
    """Return the index for this document set, from memory, disk or a fresh build.

    Built indexes are saved to config.indexes_dir/<key>/ and marked as the default index
    used by retrieve(index=None).
    """
    paths = [Path(p) for p in doc_paths]
    if not paths:
        raise ValueError("no documents given")
    for p in paths:
        if not p.is_file():
            raise ValueError(f"{p.name}: file not found ({p})")
    key = index_key(paths, config)
    target = Path(config.indexes_dir) / key

    if not rebuild and key in _memory_cache:
        return _memory_cache[key]
    if not rebuild and (target / "meta.json").is_file():
        try:
            idx = SearchIndex.load(target)
            logger.info("loaded cached index %s (%d chunks)", key, len(idx))
            _memory_cache[key] = idx
            mark_latest(config.indexes_dir, target)
            return idx
        except (OSError, ValueError) as e:
            logger.warning("cached index %s unusable (%s); rebuilding", key, e)

    chunks = process_pdfs(paths, config)
    if not chunks:
        raise ValueError("no extractable text in the given documents (scanned/image-only "
                         "PDFs need OCR, which arrives in Phase 2)")
    idx = build_index(chunks, config)
    idx.save(target)
    manifest = [{"path": str(p), "name": p.name} for p in paths]
    (target / "documents.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    mark_latest(config.indexes_dir, target)
    _memory_cache[key] = idx
    return idx


def _llm_for(config: Config) -> llm_mod.LLM:
    override = llm_mod.overridden_llm()
    if override is not None:
        return override
    key = (config.llm_provider, config.llm_model, config.llm_temperature, config.llm_effort,
           config.llm_max_tokens, config.llm_timeout_s, config.ollama_host)
    if key not in _llm_cache:
        _llm_cache[key] = llm_mod.get_llm(config)
    return _llm_cache[key]


def run_record(config: Config) -> dict:
    """Everything needed to reproduce a run (NFR-04)."""
    return log_run_config(config, extra={
        "system": SYSTEM_NAME,
        "extractor_version": EXTRACTOR_VERSION,
        "chunker_version": CHUNKER_VERSION,
        "index_format_version": INDEX_FORMAT_VERSION,
        "prompt_version": PROMPT_VERSION,
    })


def answer_question(
    question: str, uploaded_docs: Sequence[str | Path], config: Config | None = None
) -> Result:
    """End to end: process_pdfs -> build_index -> retrieve -> generate_and_verify.

    Processed chunks and the index are cached per document set so repeated questions
    over the same uploads do not reprocess. Returns the final Result for the UI/CLI.
    """
    if not question or not question.strip():
        raise ValueError("question must be non-empty")
    config = config or load_config()
    run_record(config)
    index = get_or_build_index(uploaded_docs, config)
    passages = retrieve(question, config.top_k, index)
    return generate_and_verify(question, passages, llm=_llm_for(config))
