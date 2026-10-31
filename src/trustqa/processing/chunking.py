"""Sentence-aware chunking that never crosses a page boundary.

Sizes are in characters (config.chunk_size / config.chunk_overlap). Each page is chunked
independently: sentences are packed greedily up to chunk_size, and the next chunk starts
with the trailing whole sentences of the previous one whose combined length fits within
chunk_overlap (so overlap is at most chunk_overlap and may be 0 if the last sentence is
longer). A single sentence longer than chunk_size is split at word boundaries.

chunk_id = f"{document_id}_P{page_number}_C{n}", n starting at 1 on each page.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from trustqa.processing.extract import PageText
from trustqa.schemas import Chunk, ContentType, ExtractionMethod

# Tokens that end in "." without ending a sentence (compared lower-case, without the dot).
_ABBREVIATIONS = {
    "e.g", "i.e", "al", "etc", "vs", "cf", "fig", "figs", "eq", "eqs", "sec", "tab",
    "no", "nos", "vol", "pp", "dr", "mr", "mrs", "ms", "prof", "inc", "ltd", "co", "st",
    "rs", "approx", "resp", "dept", "jan", "feb", "mar", "apr", "jun", "jul", "aug",
    "sep", "sept", "oct", "nov", "dec",
}
# Candidate boundary: terminal punctuation, optional closing quotes/brackets, whitespace,
# then something that can start a sentence.
_BOUNDARY_RE = re.compile(r"[.!?][\"'”’)\]]*\s+(?=[A-Z0-9\"'“‘(\[])")
_LAST_TOKEN_RE = re.compile(r"(\S+)$")


def chunk_id_for(document_id: str, page_number: int, n: int) -> str:
    return f"{document_id}_P{page_number}_C{n}"


def split_sentences(text: str) -> list[str]:
    """Split normalised text into sentences, guarding common abbreviations/initials."""
    sentences: list[str] = []
    start = 0
    for m in _BOUNDARY_RE.finditer(text):
        end = m.start() + 1  # include the terminal punctuation
        tok = _LAST_TOKEN_RE.search(text[start:end])
        word = tok.group(1).rstrip(".!?").lower() if tok else ""
        if text[m.start()] == "." and (
            word in _ABBREVIATIONS
            or (len(word) == 1 and word.isalpha())  # initials: "J. Smith"
        ):
            continue
        piece = text[start:m.end()].strip()
        if piece:
            sentences.append(piece)
        start = m.end()
    tail = text[start:].strip()
    if tail:
        sentences.append(tail)
    return sentences


def _split_long(sentence: str, size: int) -> list[str]:
    """Split one over-long sentence at word boundaries (hard-cut giant tokens)."""
    parts: list[str] = []
    cur = ""
    for word in sentence.split(" "):
        while len(word) > size:  # e.g. a URL longer than chunk_size
            if cur:
                parts.append(cur)
                cur = ""
            parts.append(word[:size])
            word = word[size:]
        if not word:
            continue
        if cur and len(cur) + 1 + len(word) > size:
            parts.append(cur)
            cur = word
        else:
            cur = f"{cur} {word}" if cur else word
    if cur:
        parts.append(cur)
    return parts


def _joined_len(parts: list[str]) -> int:
    return sum(len(p) for p in parts) + max(len(parts) - 1, 0)


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Pack sentences into chunks of at most chunk_size characters."""
    if chunk_size < 1 or not 0 <= chunk_overlap < chunk_size:
        raise ValueError(f"invalid chunk_size={chunk_size} / chunk_overlap={chunk_overlap}")
    pieces: list[str] = []
    for s in split_sentences(text):
        pieces.extend(_split_long(s, chunk_size) if len(s) > chunk_size else [s])

    chunks: list[str] = []
    cur: list[str] = []
    for s in pieces:
        if cur and _joined_len(cur) + 1 + len(s) > chunk_size:
            chunks.append(" ".join(cur))
            tail: list[str] = []
            for prev in reversed(cur):
                if _joined_len([prev, *tail]) > chunk_overlap:
                    break
                tail.insert(0, prev)
            while tail and _joined_len([*tail, s]) > chunk_size:
                tail.pop(0)
            cur = tail
        cur.append(s)
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def chunk_page(page: PageText, chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    """Chunk one page. Empty pages produce no chunks.

    Chunks inherit the page's extraction_method / ocr_used. For OCR pages each chunk's
    ocr_confidence is the mean confidence of the OCR words inside that chunk (chunks are
    substrings of the page text), falling back to the page mean.
    """
    if not page.text:
        return []
    chunks = []
    search_from = 0
    for n, text in enumerate(chunk_text(page.text, chunk_size, chunk_overlap), start=1):
        conf = page.ocr_confidence
        if page.ocr is not None:
            pos = page.text.find(text, search_from)
            if pos >= 0:
                span_conf = page.ocr.span_confidence(pos, pos + len(text))
                conf = span_conf if span_conf is not None else conf
                search_from = pos + 1
        chunks.append(Chunk(
            document_id=page.document_id,
            document_name=page.document_name,
            page_number=page.page_number,
            chunk_id=chunk_id_for(page.document_id, page.page_number, n),
            text=text,
            content_type=ContentType.TEXT,
            extraction_method=ExtractionMethod(page.extraction_method),
            ocr_used=page.ocr_used,
            ocr_confidence=conf if page.ocr_used else None,
        ))
    return chunks


def chunk_pages(pages: Iterable[PageText], chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    out: list[Chunk] = []
    for page in pages:
        out.extend(chunk_page(page, chunk_size, chunk_overlap))
    return out
