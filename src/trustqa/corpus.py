"""Document sets for interactive use (Phase 5 Streamlit demo).

`prepare_corpus(paths, config, progress)` turns uploaded PDFs into a searchable Corpus:

- every file is checked first; unreadable, encrypted and duplicate files are reported per
  document (DocumentStatus.error / error_kind) and left out, the rest are processed;
- processing is cached per document under processed_dir/documents/<key>/, keyed by file
  hash, assigned DOC ID, file name, processing signature and chunking settings, so
  re-uploading a file (or adding one more) never repeats OCR or table extraction;
- the index is cached per document set exactly like pipeline.get_or_build_index (same
  index_key), and the canonical chunk JSONL of the set is written to processed_dir.

`Corpus.ask(question)` retrieves and answers with the corpus's config (system A, B or C,
see system_config), returning the Result plus the retrieved passages and timing.
The UI only displays what this module and evidence.py return.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from trustqa.config import Config, load_config
from trustqa.pipeline import CHUNKER_VERSION, _llm_for, answer_from_passages, index_key, run_record
from trustqa.processing.extract import (
    DocumentRegistry,
    EncryptedPdfError,
    UnreadablePdfError,
    file_sha256,
    open_pdf,
)
from trustqa.processing.process import (
    DocumentReport,
    ProcessingResult,
    process_document,
    processed_key,
    processing_signature,
    read_jsonl,
    write_jsonl,
    write_processed,
)
from trustqa.processing.tables import rebuild_table
from trustqa.retrieval.index import SearchIndex, build_index, mark_latest
from trustqa.retrieval.search import retrieve_with_config
from trustqa.schemas import Chunk, Result, RetrievedChunk, TableData

logger = logging.getLogger("trustqa")

DOC_CACHE_VERSION = 1

# Evaluation systems (phases.md / PRD). Overrides pin the switches that define a system so
# a stray .env setting (e.g. BASELINE_B=1) cannot turn the TrustQA view into a baseline.
SYSTEMS: dict[str, tuple[str, dict]] = {
    "C": ("TrustQA (system C)", {"baseline_b": False, "flatten_tables": False, "verify": True}),
    "B": ("Baseline B: OCR + plain RAG", {"baseline_b": True, "ocr_enabled": True}),
    "A": ("Baseline A: plain-text RAG",
          {"baseline_b": False, "ocr_enabled": False, "flatten_tables": True, "verify": False}),
}


def system_config(system: str, **overrides) -> Config:
    """Config for evaluation system A, B or C (other settings from .env)."""
    if system not in SYSTEMS:
        raise ValueError(f"unknown system {system!r}; use one of {sorted(SYSTEMS)}")
    return load_config(**{**SYSTEMS[system][1], **overrides})


class CorpusError(ValueError):
    """No usable document in the set (every file failed or had no text)."""


# Why a document was left out. The UI maps these to user guidance.
UNREADABLE, ENCRYPTED, EXTRACTION_FAILED, NO_TEXT, DUPLICATE = (
    "unreadable", "encrypted", "extraction_failed", "no_text", "duplicate")


@dataclass
class DocumentStatus:
    """Per-document processing outcome shown in the UI."""

    name: str
    path: str
    sha256: str
    document_id: str | None = None
    pages: int = 0
    ocr_pages: list[int] = field(default_factory=list)
    ocr_mean_confidence: float | None = None
    tables: int = 0
    text_chunks: int = 0
    table_chunks: int = 0
    warnings: list[str] = field(default_factory=list)
    from_cache: bool = False
    seconds: float = 0.0
    error: str | None = None
    error_kind: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    @classmethod
    def from_report(cls, report: DocumentReport, from_cache: bool, seconds: float
                    ) -> DocumentStatus:
        s = report.summary()
        return cls(name=report.document_name, path=report.path, sha256=report.sha256,
                   document_id=report.document_id, pages=s["pages"], ocr_pages=s["ocr_pages"],
                   ocr_mean_confidence=s["ocr_mean_confidence"], tables=s["tables"],
                   text_chunks=s["text_chunks"], table_chunks=s["table_chunks"],
                   warnings=s["warnings"], from_cache=from_cache, seconds=round(seconds, 2))


@dataclass
class Answer:
    result: Result
    passages: list[RetrievedChunk]
    seconds: float
    system: str


@dataclass
class Corpus:
    system: str
    config: Config
    documents: list[DocumentStatus]  # every input file, in input order (incl. failures)
    index: SearchIndex
    key: str
    index_from_cache: bool = False
    _by_id: dict[str, Chunk] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self._by_id = {c.chunk_id: c for c in self.index.chunks}

    @property
    def ready(self) -> list[DocumentStatus]:
        return [d for d in self.documents if d.ok]

    @property
    def failed(self) -> list[DocumentStatus]:
        return [d for d in self.documents if not d.ok]

    def chunk(self, chunk_id: str) -> Chunk | None:
        return self._by_id.get(chunk_id)

    def table(self, table_id: str) -> TableData | None:
        """The full table a row chunk belongs to (row chunks carry only their own row)."""
        return rebuild_table(self.index.chunks, table_id)

    def document_path(self, document_id: str) -> Path | None:
        for d in self.ready:
            if d.document_id == document_id:
                return Path(d.path)
        return None

    def ask(self, question: str) -> Answer:
        """Retrieve top_k passages and answer (verified for C, plain RAG for A/B)."""
        if not question or not question.strip():
            raise ValueError("Please enter a question.")
        start = time.perf_counter()
        passages = retrieve_with_config(question.strip(), self.config.top_k, self.index,
                                        self.config, _llm_for(self.config))
        result = answer_from_passages(question.strip(), passages, self.config, self.index)
        return Answer(result, passages, round(time.perf_counter() - start, 2), self.system)


# ---------------------------------------------------------------- uploads

_UNSAFE = re.compile(r"[^A-Za-z0-9._ ()+-]+")


def uploads_dir(config: Config) -> Path:
    return Path(config.processed_dir).parent / "uploads"


def save_upload(name: str, data: bytes, config: Config) -> Path:
    """Store uploaded bytes under data/uploads/<sha16>/<name>; same content -> same path,
    so caches keyed by file hash hit on re-upload."""
    digest = hashlib.sha256(data).hexdigest()
    safe = _UNSAFE.sub("_", Path(name).name).strip() or "document.pdf"
    if not safe.lower().endswith(".pdf"):
        safe += ".pdf"
    target = uploads_dir(config) / digest[:16] / safe
    if not target.is_file():
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(".part")
        tmp.write_bytes(data)
        tmp.replace(target)
    return target


# ---------------------------------------------------------------- per-document cache

def _doc_cache_dir(sha256: str, document_id: str, name: str, config: Config) -> Path:
    parts = {"sha256": sha256, "document_id": document_id, "name": name,
             "signature": processing_signature(config), "chunk_size": config.chunk_size,
             "chunk_overlap": config.chunk_overlap, "chunker": CHUNKER_VERSION,
             "version": DOC_CACHE_VERSION}
    key = hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()[:16]
    return Path(config.processed_dir) / "documents" / key


def _load_doc_cache(d: Path) -> tuple[list[Chunk], DocumentReport] | None:
    try:
        report = DocumentReport.from_dict(json.loads((d / "report.json").read_text("utf-8")))
        return read_jsonl(d / "chunks.jsonl"), report
    except (OSError, ValueError, KeyError, TypeError) as e:
        if d.exists():
            logger.warning("document cache %s unusable (%s); reprocessing", d.name, e)
        return None


def _save_doc_cache(d: Path, chunks: list[Chunk], report: DocumentReport) -> None:
    write_jsonl(chunks, d / "chunks.jsonl")
    (d / "report.json").write_text(json.dumps(report.to_dict(), indent=1, ensure_ascii=False),
                                   encoding="utf-8")


# ---------------------------------------------------------------- prepare

ProgressFn = Callable[[float, str], None]


def prepare_corpus(paths: Sequence[str | Path], config: Config, system: str = "C",
                   progress: ProgressFn | None = None) -> Corpus:
    """Check, process (cached per document) and index a set of PDFs.

    Files that cannot be used are reported in Corpus.documents with error/error_kind and
    skipped. Raises CorpusError if no document yields any text. `progress(fraction,
    message)` is called as work advances (fraction in [0, 1]).
    """
    report_progress = progress or (lambda f, m: None)
    paths = [Path(p) for p in paths]
    if not paths:
        raise CorpusError("No documents uploaded.")
    run_record(config)

    statuses: list[DocumentStatus] = []
    registry = DocumentRegistry()
    accepted: list[tuple[int, object]] = []  # (position in statuses, DocumentEntry)
    for p in paths:
        st = DocumentStatus(name=p.name, path=str(p), sha256="")
        statuses.append(st)
        try:
            st.sha256 = file_sha256(p)
            open_pdf(p).close()
        except EncryptedPdfError as e:
            st.error, st.error_kind = str(e), ENCRYPTED
            continue
        except (UnreadablePdfError, OSError) as e:
            st.error, st.error_kind = str(e), UNREADABLE
            continue
        entry, is_new = registry.register(p)
        if not is_new:
            st.error, st.error_kind = (f"{p.name}: same content as {entry.document_name} "
                                       f"({entry.document_id}); skipped", DUPLICATE)
            continue
        st.document_id = entry.document_id
        accepted.append((len(statuses) - 1, entry))

    n = max(len(accepted), 1)
    chunks: list[Chunk] = []
    reports: list[DocumentReport] = []
    ready_paths: list[Path] = []
    for i, (pos, entry) in enumerate(accepted):
        st = statuses[pos]
        base, span = 0.9 * i / n, 0.9 / n
        report_progress(base, f"Processing {entry.document_name} ({i + 1}/{len(accepted)})")
        cache = _doc_cache_dir(entry.sha256, entry.document_id, entry.document_name, config)
        start = time.perf_counter()
        cached = _load_doc_cache(cache)
        if cached is not None:
            doc_chunks, report = cached
        else:
            def on_page(done: int, total: int, _b=base, _s=span, _name=entry.document_name):
                report_progress(_b + _s * done / max(total, 1),
                                f"Processing {_name}: page {done}/{total}")
            try:
                doc_chunks, report, _ = process_document(entry, config, on_page=on_page)
            except ValueError as e:  # e.g. scanned PDF and Tesseract unavailable
                st.error, st.error_kind = str(e), EXTRACTION_FAILED
                continue
            except Exception as e:  # never let one bad file take down the whole set
                logger.exception("processing %s failed", entry.document_name)
                st.error = f"{entry.document_name}: extraction failed ({type(e).__name__}: {e})"
                st.error_kind = EXTRACTION_FAILED
                continue
        seconds = time.perf_counter() - start
        st = DocumentStatus.from_report(report, cached is not None, seconds)
        st.path = statuses[pos].path  # the file actually given (cache may hold an older path)
        statuses[pos] = st
        if not doc_chunks:
            hint = "" if config.ocr_enabled else " (this system runs without OCR)"
            st.error = f"{entry.document_name}: no text could be extracted{hint}"
            st.error_kind = NO_TEXT
            continue
        if cached is None:
            _save_doc_cache(cache, doc_chunks, report)
        chunks.extend(doc_chunks)
        reports.append(report)
        ready_paths.append(Path(st.path))

    if not chunks:
        problems = "; ".join(s.error for s in statuses if s.error) or "no text found"
        raise CorpusError(f"None of the uploaded documents could be used: {problems}")

    key = index_key(ready_paths, config)
    target = Path(config.indexes_dir) / key
    index, from_cache = None, False
    if (target / "meta.json").is_file():
        try:
            index = SearchIndex.load(target)
            from_cache = [c.chunk_id for c in index.chunks] == [c.chunk_id for c in chunks]
            if not from_cache:
                index = None
        except (OSError, ValueError) as e:
            logger.warning("cached index %s unusable (%s); rebuilding", key, e)
            index = None
    if index is None:
        report_progress(0.92, f"Embedding {len(chunks)} chunks and building the index")
        result = ProcessingResult(chunks, reports, processing_signature(config))
        write_processed(result, Path(config.processed_dir) / processed_key(result, config),
                        run_record(config))
        index = build_index(chunks, config)
        index.save(target)
        manifest = [{"path": str(p), "name": p.name} for p in ready_paths]
        (target / "documents.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    mark_latest(config.indexes_dir, target)
    report_progress(1.0, "Ready")
    return Corpus(system=system, config=config, documents=statuses, index=index, key=key,
                  index_from_cache=from_cache)


__all__ = ["SYSTEMS", "system_config", "Corpus", "CorpusError", "DocumentStatus", "Answer",
           "prepare_corpus", "save_upload", "uploads_dir", "UNREADABLE", "ENCRYPTED",
           "EXTRACTION_FAILED", "NO_TEXT", "DUPLICATE"]
