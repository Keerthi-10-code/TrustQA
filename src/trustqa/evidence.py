"""Evidence rendering for the demo UI: a PDF page image with the cited text highlighted.

The text to highlight is chosen from trusted data only: the validated supporting quote(s)
of a citation, else the cited chunk's own text; for a table row chunk, the row's cells
(the row-aware chunk text "Table 2 | Model=BM25 | ..." is not printed on the page).

Words are located on the page by token alignment (difflib) against the page's own word
boxes: PyMuPDF words for digital pages, Tesseract word boxes for OCR'd pages (re-run at a
lower DPI on that single page; nothing from the LLM is used). Only aligned runs of
consecutive words are highlighted, so isolated common words elsewhere on the page are not.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

import pymupdf

from trustqa.processing.extract import open_pdf
from trustqa.schemas import Chunk, ContentType

logger = logging.getLogger("trustqa")

_TOKEN_RE = re.compile(r"[a-z0-9]+")
HIGHLIGHT_RGB = (1.0, 0.85, 0.0)


@dataclass
class PageEvidence:
    png: bytes
    page_number: int
    highlighted_words: int  # page words highlighted
    target_words: int  # tokens in the text we tried to locate
    source: str  # "quote" | "chunk text" | "table row" | "none"
    note: str | None = None  # why nothing (or only part) was highlighted

    @property
    def located(self) -> bool:
        return self.highlighted_words > 0


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def highlight_target(chunk: Chunk | None, quote: str | None) -> tuple[list[str], str]:
    """Text segments to locate on the page, and where they came from."""
    if chunk is not None and chunk.content_type is ContentType.TABLE and chunk.table_data:
        cells = [c for row in chunk.table_data.rows for c in row if c and c.strip()]
        return [" ".join(cells)], "table row"
    if quote and quote.strip():
        return [q for q in re.split(r"\s*\.\.\.\s*", quote) if q.strip()], "quote"
    if chunk is not None and chunk.text.strip():
        return [chunk.text], "chunk text"
    return [], "none"


def align_words(page_tokens: list[str], segment_tokens: list[str]) -> set[int]:
    """Indices of page words aligned with a segment, keeping runs of >= 2 consecutive
    words (>= 1 for one- or two-token segments). Empty unless at least 40% of the
    segment's tokens align."""
    if not segment_tokens or not page_tokens:
        return set()
    sm = SequenceMatcher(None, page_tokens, segment_tokens, autojunk=False)
    min_run = 1 if len(segment_tokens) <= 2 else 2
    hits: set[int] = set()
    matched = 0
    for block in sm.get_matching_blocks():
        if block.size >= min_run:
            hits.update(range(block.a, block.a + block.size))
            matched += block.size
    if matched < 0.4 * len(segment_tokens):
        return set()
    return hits


def _digital_words(page: pymupdf.Page) -> list[tuple[pymupdf.Rect, str]]:
    return [(pymupdf.Rect(w[:4]), w[4]) for w in page.get_text("words", sort=True)]


def _ocr_words(page: pymupdf.Page, dpi: int, tesseract_cmd: str | None, lang: str
               ) -> list[tuple[pymupdf.Rect, str]]:
    import pytesseract

    from trustqa.processing.ocr import configure_tesseract, render_page

    configure_tesseract(tesseract_cmd)
    img = render_page(page, dpi)
    data = pytesseract.image_to_data(img, lang=lang, output_type=pytesseract.Output.DICT)
    scale = 72.0 / dpi
    out = []
    for text, x, y, w, h in zip(data["text"], data["left"], data["top"], data["width"],
                                data["height"]):
        if text and text.strip():
            out.append((pymupdf.Rect(x * scale, y * scale, (x + w) * scale, (y + h) * scale),
                        text))
    return out


def render_page_evidence(pdf_path: str | Path, page_number: int, chunk: Chunk | None = None,
                         quote: str | None = None, *, ocr: bool | None = None,
                         dpi: int = 110, ocr_dpi: int = 200,
                         tesseract_cmd: str | None = None, ocr_lang: str = "eng"
                         ) -> PageEvidence:
    """Render page `page_number` (1-indexed) of `pdf_path` as PNG with the cited text
    highlighted. `ocr` defaults to chunk.ocr_used. Raises ValueError for unreadable PDFs
    or out-of-range pages."""
    segments, source = highlight_target(chunk, quote)
    use_ocr = chunk.ocr_used if ocr is None and chunk is not None else bool(ocr)
    note = None
    with open_pdf(pdf_path) as doc:
        if not 1 <= page_number <= len(doc):
            raise ValueError(f"{Path(pdf_path).name}: page {page_number} out of range "
                             f"(document has {len(doc)} pages)")
        page = doc[page_number - 1]
        words: list[tuple[pymupdf.Rect, str]] = []
        if segments:
            if use_ocr:
                try:
                    words = _ocr_words(page, ocr_dpi, tesseract_cmd, ocr_lang)
                except Exception as e:  # Tesseract missing/broken: still show the page
                    note = f"could not locate words on this scanned page ({e})"
                    logger.warning("evidence OCR failed: %s", e)
            else:
                words = _digital_words(page)

        # tokenise words, remembering which word each token came from
        page_tokens: list[str] = []
        owner: list[int] = []
        for wi, (_, text) in enumerate(words):
            for t in _tokens(text):
                page_tokens.append(t)
                owner.append(wi)
        hit_words: set[int] = set()
        target = 0
        for seg in segments:
            seg_tokens = _tokens(seg)
            target += len(seg_tokens)
            hit_words.update(owner[i] for i in align_words(page_tokens, seg_tokens))

        for wi in sorted(hit_words):
            r = words[wi][0]
            page.draw_rect(r + (-1, -1, 1, 1), color=None, fill=HIGHLIGHT_RGB,
                           fill_opacity=0.4, overlay=True)
        if segments and not hit_words and note is None:
            note = ("the cited text could not be located on the rendered page "
                    "(it may be reworded, hyphenated or an OCR misread)")
        png = page.get_pixmap(dpi=dpi, alpha=False).tobytes("png")
    return PageEvidence(png=png, page_number=page_number, highlighted_words=len(hit_words),
                        target_words=target, source=source, note=note)


__all__ = ["PageEvidence", "render_page_evidence", "highlight_target", "align_words"]
