"""Page-by-page text extraction from digital PDFs (PyMuPDF) and the document-ID registry.

Phase 1 handles the digital text layer only. Pages with no extractable text are returned
with empty text (they become OCR candidates in Phase 2) and produce no chunks.

Text normalisation is deliberately conservative so numbers, signs and decimals are never
altered (NFR-03): ligatures are expanded from an explicit map (no NFKC, which would turn
superscript digits into ordinary digits), words split by a line-end hyphen are re-joined
only between letters, and whitespace runs are collapsed.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import pymupdf

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
    """Text of one PDF page. page_number is 1-indexed."""

    document_id: str
    document_name: str
    page_number: int
    text: str


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


def extract_pages(path: str | Path, document_id: str, document_name: str | None = None
                  ) -> list[PageText]:
    """Extract normalised text for every page of a PDF, in page order.

    Raises ValueError if the file is missing, not a readable PDF, or encrypted (NFR-09).
    """
    p = Path(path)
    name = document_name or p.name
    if not p.is_file():
        raise ValueError(f"{name}: file not found ({p})")
    try:
        doc = pymupdf.open(p)
    except Exception as e:  # PyMuPDF raises several types for corrupt input
        raise ValueError(f"{name}: not a readable PDF ({e})") from e
    with doc:
        if not doc.is_pdf:
            raise ValueError(f"{name}: not a PDF file")
        if doc.needs_pass:
            raise ValueError(f"{name}: PDF is encrypted / password-protected")
        raw_pages: list[str] = []
        for i, page in enumerate(doc):
            try:
                raw_pages.append(page.get_text("text"))
            except Exception as e:
                raise ValueError(f"{name}: text extraction failed on page {i + 1} ({e})") from e
    vocab = document_vocabulary(raw_pages)
    pages = [PageText(document_id, name, i + 1, normalize_text(raw, vocab))
             for i, raw in enumerate(raw_pages)]
    empty = [pg.page_number for pg in pages if not pg.text]
    if empty:
        logger.warning("%s: %d/%d page(s) have no text layer (OCR is Phase 2): %s",
                       name, len(empty), len(pages), empty)
    return pages
