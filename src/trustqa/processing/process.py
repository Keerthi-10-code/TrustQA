"""Module 1 - document processing (PRD 3.3).

Per document: register (stable DOC ID) -> per page: digital text layer (PyMuPDF), quality
check (detect.py), OCR fallback for scanned / garbage pages (ocr.py), table extraction on
digital pages (tables.py) -> sentence chunks for text, one row-aware chunk per table row.

Switches (config.py): ocr_enabled, ocr_* thresholds, flatten_tables (no table extraction:
table content stays in the page text as a plain-text RAG sees it), baseline_b preset.
With ocr_enabled=False and flatten_tables=True the output is identical to baseline A.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable, Collection, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

from trustqa.config import Config, load_config
from trustqa.processing.chunking import chunk_pages
from trustqa.processing.detect import page_quality
from trustqa.processing.extract import (
    EXTRACTOR_VERSION,
    DocumentEntry,
    DocumentRegistry,
    PageText,
    document_vocabulary,
    normalize_text,
    open_pdf,
    page_text_excluding,
    raw_page_text,
)
from trustqa.processing.ocr import (
    OCR_VERSION,
    OcrPage,
    OcrUnavailableError,
    configure_tesseract,
    is_blank_page,
    ocr_page,
    tesseract_version,
)
from trustqa.processing.sections import SECTIONS_VERSION, find_headings, section_marks
from trustqa.processing.tables import TABLES_VERSION, ExtractedTable, find_tables, table_chunks
from trustqa.schemas import Chunk, ContentType, ExtractionMethod

logger = logging.getLogger("trustqa")


def processing_signature(config: Config) -> str:
    """Identifies everything that changes processing output (index cache key, NFR-04).

    Equals the Phase 1 EXTRACTOR_VERSION when OCR and table extraction are both off, so
    baseline A keeps its frozen index key.
    """
    parts = [EXTRACTOR_VERSION]
    if config.ocr_enabled:
        parts.append(f"{OCR_VERSION}(min={config.ocr_min_chars},ratio="
                     f"{config.ocr_max_nonalnum_ratio},dpi={config.ocr_dpi},lang="
                     f"{config.ocr_lang},psm={config.ocr_psm},pre={int(config.ocr_preprocess)})")
    if not config.flatten_tables:
        parts.append(TABLES_VERSION)
    if config.section_headings:
        parts.append(SECTIONS_VERSION)
    return "+".join(parts)


@dataclass
class PageReport:
    page_number: int
    method: str  # pymupdf | tesseract
    quality: str  # ok | low_text | garbage_text | blank (of the digital text layer)
    alnum_chars: int
    nonalnum_ratio: float
    ocr_attempted: bool = False
    ocr_confidence: float | None = None
    ocr_rejected: bool = False  # OCR ran but the text layer was kept
    tables: int = 0
    text_chunks: int = 0
    table_chunks: int = 0


@dataclass
class DocumentReport:
    document_id: str
    document_name: str
    path: str
    sha256: str
    pages: list[PageReport] = field(default_factory=list)
    tables: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    # in-memory only (not serialised): OCR word confidences per page, for inspection
    ocr_results: dict[int, OcrPage] = field(default_factory=dict, repr=False)

    @property
    def ocr_pages(self) -> list[int]:
        return [p.page_number for p in self.pages if p.method == "tesseract"]

    def summary(self) -> dict:
        confs = [p.ocr_confidence for p in self.pages
                 if p.method == "tesseract" and p.ocr_confidence is not None]
        return {
            "document_id": self.document_id,
            "document_name": self.document_name,
            "pages": len(self.pages),
            "ocr_pages": self.ocr_pages,
            "ocr_mean_confidence": round(sum(confs) / len(confs), 2) if confs else None,
            "low_text_pages_kept": [p.page_number for p in self.pages
                                    if p.quality != "ok" and p.method == "pymupdf"],
            "tables": len(self.tables),
            "text_chunks": sum(p.text_chunks for p in self.pages),
            "table_chunks": sum(p.table_chunks for p in self.pages),
            "warnings": list(self.warnings),
        }

    def to_dict(self) -> dict:
        d = self.summary()
        d.update(path=self.path, sha256=self.sha256,
                 page_reports=[asdict(p) for p in self.pages], table_list=self.tables)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> DocumentReport:
        """Inverse of to_dict (OCR word data is not serialised and comes back empty)."""
        return cls(document_id=d["document_id"], document_name=d["document_name"],
                   path=d["path"], sha256=d["sha256"],
                   pages=[PageReport(**p) for p in d.get("page_reports", [])],
                   tables=list(d.get("table_list", [])), warnings=list(d.get("warnings", [])))


@dataclass
class ProcessingResult:
    chunks: list[Chunk]
    documents: list[DocumentReport]
    signature: str
    extracted_tables: list[ExtractedTable] = field(default_factory=list)


def _accept_ocr(digital_alnum: int, quality_reason: str, ocr_text: str, config: Config) -> bool:
    """Use the OCR text unless it is worse than what the text layer already had."""
    q = page_quality(ocr_text, config.ocr_min_chars, config.ocr_max_nonalnum_ratio)
    if quality_reason == "garbage_text":
        return q.alnum_chars > 0 and not q.needs_ocr
    return q.alnum_chars > digital_alnum


def process_document(entry: DocumentEntry, config: Config,
                     pages: Collection[int] | None = None,
                     on_page: Callable[[int, int], None] | None = None,
                     ) -> tuple[list[Chunk], DocumentReport, list[ExtractedTable]]:
    """Process one registered PDF. `pages` (1-indexed) limits work to those pages.
    `on_page(done, total)` is called after each selected page (progress display)."""
    name = entry.document_name
    report = DocumentReport(entry.document_id, name, str(entry.path), entry.sha256)
    all_tables: list[ExtractedTable] = []
    with open_pdf(entry.path, name) as doc:
        selected = [i for i in range(len(doc)) if pages is None or (i + 1) in pages]
        raws = {i: raw_page_text(doc[i], name, i + 1) for i in range(len(doc))}
        vocab = document_vocabulary(list(raws.values()))
        quality = {i: page_quality(raws[i], config.ocr_min_chars,
                                   config.ocr_max_nonalnum_ratio) for i in selected}
        blank = {i for i in selected if config.ocr_enabled and quality[i].needs_ocr
                 and quality[i].reason == "low_text" and is_blank_page(doc[i])}
        ocr_ok = config.ocr_enabled
        if ocr_ok and any(q.needs_ocr and i not in blank for i, q in quality.items()):
            try:
                tesseract_version(config.tesseract_cmd)
                configure_tesseract(config.tesseract_cmd)
            except OcrUnavailableError as e:
                if all(q.needs_ocr for q in quality.values()):
                    raise ValueError(f"{name}: no usable text layer and {e}") from e
                ocr_ok = False  # keep the digital pages; say what was skipped
                report.warnings.append(f"OCR skipped (Tesseract unavailable): {e}")
                logger.warning("%s: OCR skipped, Tesseract unavailable", name)

        plumber = None
        if not config.flatten_tables:
            import pdfplumber

            plumber = pdfplumber.open(entry.path)
        try:
            page_texts: list[PageText] = []
            page_tables: dict[int, list[ExtractedTable]] = {}
            sec_stack: list = []  # open section headings, carried across pages
            for done, i in enumerate(selected):
                if on_page is not None and done:
                    on_page(done, len(selected))
                pn = i + 1
                q = quality[i]
                pr = PageReport(pn, "pymupdf", q.reason, q.alnum_chars,
                                round(q.nonalnum_ratio, 4))
                report.pages.append(pr)
                if i in blank:
                    pr.quality = "blank"
                elif ocr_ok and q.needs_ocr:
                    pr.ocr_attempted = True
                    res = ocr_page(doc[i], dpi=config.ocr_dpi, lang=config.ocr_lang,
                                   psm=config.ocr_psm, use_preprocess=config.ocr_preprocess,
                                   vocab=vocab or None)
                    pr.ocr_confidence = res.mean_confidence
                    report.ocr_results[pn] = res
                    if _accept_ocr(q.alnum_chars, q.reason, res.text, config):
                        pr.method = "tesseract"
                        marks = ()
                        if config.section_headings:  # no font info: carry the open section
                            marks, sec_stack = section_marks([], res.text, sec_stack)
                        page_texts.append(PageText(
                            entry.document_id, name, pn, res.text,
                            extraction_method=ExtractionMethod.TESSERACT, ocr_used=True,
                            ocr_confidence=res.mean_confidence, ocr=res,
                            section_marks=marks))
                        continue
                    pr.ocr_rejected = True
                    report.warnings.append(f"page {pn}: OCR output not better than the text "
                                           f"layer ({q.reason}); kept text layer")
                tables: list[ExtractedTable] = []
                if plumber is not None:
                    try:
                        tables = find_tables(plumber.pages[i], pn, entry.document_id)
                    except Exception as e:  # never lose the page over a table failure
                        report.warnings.append(f"page {pn}: table extraction failed ({e})")
                        logger.warning("%s p%d: table extraction failed: %s", name, pn, e)
                raw = page_text_excluding(doc[i], [t.bbox for t in tables]) if tables \
                    else raws[i]
                text = normalize_text(raw, vocab)
                marks = ()
                if config.section_headings:
                    marks, sec_stack = section_marks(find_headings(doc[i]), text, sec_stack)
                page_texts.append(PageText(entry.document_id, name, pn, text,
                                           section_marks=marks))
                page_tables[pn] = tables
                pr.tables = len(tables)
        finally:
            if plumber is not None:
                plumber.close()
        if on_page is not None:
            on_page(len(selected), len(selected))

    chunks: list[Chunk] = []
    by_page = {p.page_number: p for p in report.pages}
    for pt in page_texts:
        text_chunks = chunk_pages([pt], config.chunk_size, config.chunk_overlap)
        tab_chunks: list[Chunk] = []
        for t in page_tables.get(pt.page_number, []):
            tab_chunks.extend(table_chunks(t, entry.document_id, name, config.chunk_size))
            all_tables.append(t)
            report.tables.append({"table_id": t.table_id, "label": t.label, "page": t.page_number,
                                  "rows": len(t.rows), "columns": len(t.headers),
                                  "method": t.method, "units": t.units,
                                  "bbox": [round(v, 1) for v in t.bbox], "notes": t.notes})
        by_page[pt.page_number].text_chunks = len(text_chunks)
        by_page[pt.page_number].table_chunks = len(tab_chunks)
        chunks.extend(text_chunks)
        chunks.extend(tab_chunks)

    empty = [p.page_number for p in report.pages
             if p.quality != "blank" and not p.text_chunks and not p.table_chunks]
    if empty:
        report.warnings.append(f"no text extracted from page(s) {empty}")
        logger.warning("%s: no text extracted from page(s) %s", name, empty)
    return chunks, report, all_tables


def process_pdfs_detailed(pdf_paths: Sequence[str | Path], config: Config | None = None
                          ) -> ProcessingResult:
    """process_pdfs plus per-document reports (pages, OCR pages, tables, warnings)."""
    config = config or load_config()
    registry = DocumentRegistry()
    chunks: list[Chunk] = []
    reports: list[DocumentReport] = []
    tables: list[ExtractedTable] = []
    for path in pdf_paths:
        if not Path(path).is_file():
            raise ValueError(f"{Path(path).name}: file not found ({path})")
        entry, is_new = registry.register(path)
        if not is_new:
            logger.warning("%s is a duplicate of %s (%s); skipped", Path(path).name,
                           entry.document_name, entry.document_id)
            continue
        doc_chunks, report, doc_tables = process_document(entry, config)
        s = report.summary()
        logger.info("%s -> %s: %d pages (OCR %s), %d tables, %d text + %d table chunks",
                    entry.document_name, entry.document_id, s["pages"], s["ocr_pages"] or "none",
                    s["tables"], s["text_chunks"], s["table_chunks"])
        chunks.extend(doc_chunks)
        reports.append(report)
        tables.extend(doc_tables)

    ids = [c.chunk_id for c in chunks]
    if len(ids) != len(set(ids)):  # cannot happen by construction; guard the contract
        raise AssertionError("duplicate chunk_ids produced")
    return ProcessingResult(chunks, reports, processing_signature(config), tables)


def process_pdfs(pdf_paths: Sequence[str | Path], config: Config | None = None) -> list[Chunk]:
    """Turn PDFs into canonical chunk records.

    Assigns a stable document_id per file (DOC01, DOC02, ... in input order), extracts
    text page by page (PyMuPDF), falls back to Tesseract OCR on low-text / garbage pages,
    extracts tables (pdfplumber), and chunks without crossing page boundaries.

    Returns chunks with unique chunk_ids (DOC01_P4_C1 for text, DOC01_P4_T1_R3 for table
    row 3 of table 1 on page 4) and 1-indexed page_number. Serialise with
    Chunk.to_dict() / write_jsonl() for data/processed/.

    Raises ValueError for unreadable / encrypted PDFs, or when OCR is needed but
    Tesseract is unavailable (NFR-09). The same file passed twice (by content hash) is
    processed once.
    """
    return process_pdfs_detailed(pdf_paths, config).chunks


# ---------------------------------------------------------------- canonical JSONL

def write_jsonl(chunks: Sequence[Chunk], path: str | Path) -> Path:
    """One Chunk.to_json() per line (canonical PRD 3.2 records)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        for c in chunks:
            f.write(c.to_json() + "\n")
    tmp.replace(p)
    return p


def read_jsonl(path: str | Path) -> list[Chunk]:
    with open(path, encoding="utf-8") as f:
        return [Chunk.from_json(line) for line in f if line.strip()]


def processed_key(result: ProcessingResult, config: Config) -> str:
    """Folder name under data/processed/: documents (content hash, in order) + processing
    signature + chunking, so the same inputs and settings always land in the same place."""
    parts = {"docs": [r.sha256 for r in result.documents], "signature": result.signature,
             "chunk_size": config.chunk_size, "chunk_overlap": config.chunk_overlap}
    return hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()[:16]


def write_processed(result: ProcessingResult, out_dir: str | Path,
                    run_config: dict | None = None) -> Path:
    """Write chunks.jsonl + documents.json (+ run_config.json) into out_dir."""
    out = Path(out_dir)
    write_jsonl(result.chunks, out / "chunks.jsonl")
    docs = {"processing_signature": result.signature,
            "documents": [r.to_dict() for r in result.documents]}
    (out / "documents.json").write_text(json.dumps(docs, indent=1, ensure_ascii=False),
                                        encoding="utf-8")
    if run_config is not None:
        (out / "run_config.json").write_text(json.dumps(run_config, indent=1, sort_keys=True),
                                             encoding="utf-8")
    return out


def count_by_type(chunks: Sequence[Chunk]) -> dict[str, int]:
    return {t.value: sum(c.content_type is t for c in chunks) for t in ContentType}


__all__ = ["process_pdfs", "process_pdfs_detailed", "process_document", "processing_signature",
           "write_jsonl", "read_jsonl", "write_processed", "processed_key", "ProcessingResult", "DocumentReport",
           "PageReport", "count_by_type"]
