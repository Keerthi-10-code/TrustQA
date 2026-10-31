"""Page-by-page text extraction from digital PDFs (PyMuPDF) and the document-ID registry.

`extract_pages` reads the digital text layer only (baseline system A). Pages with no
extractable text come back empty; process.py decides per page whether to OCR them
(processing/detect.py, processing/ocr.py) and removes extracted table regions from the
page text (processing/tables.py).

Text normalisation is deliberately conservative so numbers, signs and decimals are never
altered (NFR-03): ligatures are expanded from an explicit map (no NFKC, which would turn
superscript digits into ordinary digits), words split by a line-end hyphen are re-joined
only between letters, and whitespace runs are collapsed.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import pymupdf

from trustqa.schemas import ExtractionMethod

if TYPE_CHECKING:
    from trustqa.processing.ocr import OcrPage

logger = logging.getLogger("trustqa")

# Bump when extraction/normalisation output changes, so cached indexes are rebuilt.
EXTRACTOR_VERSION = "pymupdf-text-v1"

_LIGATURES = {
    "ﬀ": "ff",
    "ﬁ": "fi",
    "ﬂ": "fl",
    "ﬃ": "ffi",
    "ﬄ": "ffl",
    "ﬅ": "st",
    "ﬆ": "st",
}
_LIGATURE_RE = re.compile("|".join(_LIGATURES))
# Word split by a line-end hyphen: letters on both sides, never digits.
_HYPHEN_BREAK_RE = re.compile(r"([A-Za-z]+)-[ \t]*\n[ \t]*([a-z]+)")
_SOFT_HYPHEN = "­"
_WS_RE = re.compile(r"\s+")
_WORD_RE = re.compile(r"[A-Za-z]+")


def _fix_ligatures(text: str) -> str:
    text = text.replace(_SOFT_HYPHEN + "\n", "").replace(_SOFT_HYPHEN, "")
    return _LIGATURE_RE.sub(lambda m: _LIGATURES[m.group(0)], text)


def document_vocabulary(raw_texts: list[str]) -> set[str]:
    """Lower-cased words of a document, used to decide line-end hyphen joins."""
    vocab: set[str] = set()
    for t in raw_texts:
        vocab.update(w.lower() for w in _WORD_RE.findall(_fix_ligatures(t)))
    return vocab


def normalize_text(text: str, vocab: set[str] | None = None) -> str:
    """Clean PyMuPDF page text without touching numbers or symbols.

    A line-end hyphen between letters ("retrie-\\nval") is joined only when the joined
    word occurs elsewhere in `vocab` (the document's words); otherwise the hyphen is kept
    ("knowledge-\\nintensive" -> "knowledge-intensive"), so no character is lost.
    """
    text = _fix_ligatures(text)

    def join(m: re.Match) -> str:
        left, right = m.group(1), m.group(2)
        if vocab is not None and (left + right).lower() in vocab:
            return left + right
        return f"{left}-{right}"

    text = _HYPHEN_BREAK_RE.sub(join, text)
    return _WS_RE.sub(" ", text).strip()


@dataclass(frozen=True)
class PageText:
    """Text of one PDF page. page_number is 1-indexed.

    For OCR pages `ocr` keeps the per-word confidences and character spans so chunks can
    report the confidence of exactly the words they contain.
    """

    document_id: str
    document_name: str
    page_number: int
    text: str
    extraction_method: ExtractionMethod = ExtractionMethod.PYMUPDF
    ocr_used: bool = False
    ocr_confidence: float | None = None  # page mean, 0-100
    ocr: OcrPage | None = field(default=None, compare=False, repr=False)
    # (offset in text, section path) where the section changes; empty unless
    # config.section_headings (processing/sections.py)
    section_marks: tuple[tuple[int, str], ...] = field(default=(), compare=False, repr=False)


@dataclass(frozen=True)
class DocumentEntry:
    document_id: str
    document_name: str
    path: Path
    sha256: str


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class DocumentRegistry:
    """Assigns document IDs DOC01, DOC02, ... in registration (input) order.

    Keyed by file content hash, so registering the same file twice (even under another
    path) returns the existing entry instead of creating a duplicate document.
    """

    def __init__(self) -> None:
        self._by_hash: dict[str, DocumentEntry] = {}
        self._entries: list[DocumentEntry] = []

    def register(self, path: str | Path) -> tuple[DocumentEntry, bool]:
        """Return (entry, is_new)."""
        p = Path(path)
        digest = file_sha256(p)
        if digest in self._by_hash:
            return self._by_hash[digest], False
        entry = DocumentEntry(
            document_id=f"DOC{len(self._entries) + 1:02d}",
            document_name=p.name,
            path=p,
            sha256=digest,
        )
        self._by_hash[digest] = entry
        self._entries.append(entry)
        return entry, True

    @property
    def entries(self) -> list[DocumentEntry]:
        return list(self._entries)

    def __len__(self) -> int:
        return len(self._entries)


class UnreadablePdfError(ValueError):
    """The file is missing, damaged or not a PDF (NFR-09)."""


class EncryptedPdfError(UnreadablePdfError):
    """The PDF needs a password to open (NFR-09)."""


def open_pdf(path: str | Path, document_name: str | None = None) -> pymupdf.Document:
    """Open a PDF for reading. Raises UnreadablePdfError (a ValueError) if missing,
    unreadable or not a PDF, and EncryptedPdfError if it is password-protected (NFR-09)."""
    p = Path(path)
    name = document_name or p.name
    if not p.is_file():
        raise UnreadablePdfError(f"{name}: file not found ({p})")
    try:
        doc = pymupdf.open(p)
    except Exception as e:  # PyMuPDF raises several types for corrupt input
        raise UnreadablePdfError(f"{name}: not a readable PDF ({e})") from e
    if not doc.is_pdf:
        doc.close()
        raise UnreadablePdfError(f"{name}: not a PDF file")
    if doc.needs_pass:
        doc.close()
        raise EncryptedPdfError(f"{name}: PDF is encrypted / password-protected")
    return doc


def raw_page_text(page: pymupdf.Page, name: str, page_number: int) -> str:
    try:
        return page.get_text("text")
    except Exception as e:
        raise ValueError(f"{name}: text extraction failed on page {page_number} ({e})") from e


def page_text_excluding(page: pymupdf.Page, rects: list[tuple[float, float, float, float]]
                        ) -> str:
    """Raw page text without the lines whose centre falls inside any of `rects`
    (x0, top, x1, bottom in PDF points), e.g. extracted table bodies."""
    if not rects:
        return page.get_text("text")
    blocks = []
    for b in page.get_text("dict")["blocks"]:
        if b.get("type") != 0:
            continue
        kept = []
        for line in b["lines"]:
            x0, y0, x1, y1 = line["bbox"]
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            if any(r[0] - 2 <= cx <= r[2] + 2 and r[1] - 2 <= cy <= r[3] + 2 for r in rects):
                continue
            kept.append("".join(span["text"] for span in line["spans"]))
        if kept:
            blocks.append("\n".join(kept))
    return "\n".join(blocks)


def extract_pages(path: str | Path, document_id: str, document_name: str | None = None
                  ) -> list[PageText]:
    """Extract normalised text for every page of a PDF, in page order (text layer only).

    Raises ValueError if the file is missing, not a readable PDF, or encrypted (NFR-09).
    """
    name = document_name or Path(path).name
    with open_pdf(path, name) as doc:
        raw_pages = [raw_page_text(page, name, i + 1) for i, page in enumerate(doc)]
    vocab = document_vocabulary(raw_pages)
    pages = [PageText(document_id, name, i + 1, normalize_text(raw, vocab))
             for i, raw in enumerate(raw_pages)]
    empty = [pg.page_number for pg in pages if not pg.text]
    if empty:
        logger.warning("%s: %d/%d page(s) have no text layer: %s",
                       name, len(empty), len(pages), empty)
    return pages
