"""Section headings for digital pages (config.section_headings, default off).

A list such as "A. Core Technologies: Transformer ... BERT ... T5 ..." is often split over
two chunks, and only the first one says "Core Technologies"; a question naming the section
then cannot retrieve the rest. With this switch on, every text chunk records the section
it starts in (Chunk.section, e.g. "I. LITERATURE SURVEY > A. Core Technologies"), and
retrieval / generation see that path next to the chunk text (retrieval.index.index_text,
generation.format_passages). The chunk text itself is unchanged, so quotes, numeric checks
and page highlights work exactly as before.

Headings are found from PyMuPDF font information, conservatively. A line is a heading when
- it starts with section numbering ("I.", "A.", "2", "2.1", "3.2.1") followed by a short
  title (<= 12 words, <= 90 chars) starting with a capital and not ending in . , ; :
  (run-in paragraph headings and list items end with punctuation or are long), or it is a
  bare number line ("2.1") directly followed by such a title line in the same block;
- it is styled differently from body text: bold/medium or italic for the whole line, or
  larger than the page's body size, or (Roman numbering only) an all-caps title;
- it is not smaller than body text (excludes figure labels such as "9 Tasks").
Levels: Roman -> 1, capital letter -> 2, Arabic -> number of dot-separated parts.
OCR pages have no font information: they carry the section in effect from earlier pages.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass

import pymupdf

SECTIONS_VERSION = "sections-v1"

_NUMBERED = re.compile(r"^(?:(?P<roman>[IVX]{1,5})\.|(?P<letter>[A-Z])\.|"
                       r"(?P<arabic>\d{1,2}(?:\.\d{1,2}){0,3})\.?)\s+(?P<title>\S.*)$")
_BARE_NUMBER = re.compile(r"^\d{1,2}(?:\.\d{1,2}){0,3}\.?$")
_MAX_WORDS, _MAX_CHARS = 12, 90


@dataclass(frozen=True)
class Heading:
    text: str
    level: int


def _styled(spans: list[dict]) -> bool:
    def bold(s):
        return bool(s["flags"] & 16) or any(k in s["font"] for k in ("Bold", "Medi", "Semibold", "Heavy", "Black"))

    def italic(s):
        return bool(s["flags"] & 2) or any(k in s["font"] for k in ("Ital", "Obli"))

    return all(bold(s) for s in spans) or all(italic(s) for s in spans)


def _title_ok(title: str) -> bool:
    title = title.strip()
    return (bool(title) and title[0].isupper() and any(ch.isalpha() for ch in title)
            and len(title) <= _MAX_CHARS and len(title.split()) <= _MAX_WORDS
            and title[-1] not in ".,;:")


def _level(m: re.Match) -> int:
    if m.group("roman"):
        return 1
    if m.group("letter"):
        return 2
    return m.group("arabic").rstrip(".").count(".") + 1


def find_headings(page: pymupdf.Page) -> list[Heading]:
    """Numbered section headings on a digital page, in reading order."""
    data = page.get_text("dict")
    sizes: list[float] = []
    blocks: list[list[tuple[str, list[dict], float]]] = []
    for b in data.get("blocks", []):
        lines = []
        for line in b.get("lines", []):
            spans = [s for s in line["spans"] if s["text"].strip()]
            if not spans:
                continue
            text = " ".join("".join(s["text"] for s in line["spans"]).split())
            size = max(s["size"] for s in spans)
            sizes.extend(s["size"] for s in spans for _ in range(len(s["text"].strip())))
            lines.append((text, spans, size))
        if lines:
            blocks.append(lines)
    if not sizes:
        return []
    body = statistics.median(sizes)
    out: list[Heading] = []
    for lines in blocks:
        i = 0
        while i < len(lines):
            text, spans, size = lines[i]
            cand = None
            if (_BARE_NUMBER.match(text) and i + 1 < len(lines)
                    and (_styled(spans) or size > body + 0.5)):
                nxt_text, nxt_spans, nxt_size = lines[i + 1]
                joined = f"{text} {nxt_text}"
                m = _NUMBERED.match(joined)
                if (m and _title_ok(m.group("title")) and nxt_size >= body - 0.5
                        and (_styled(nxt_spans) or nxt_size > body + 0.5)):
                    cand = Heading(joined, _level(m))
                    i += 1
            else:
                m = _NUMBERED.match(text)
                if m and _title_ok(m.group("title")) and size >= body - 0.5:
                    title = m.group("title")
                    caps = m.group("roman") and title.upper() == title
                    if _styled(spans) or size > body + 0.5 or caps:
                        cand = Heading(text, _level(m))
            if cand:
                out.append(cand)
            i += 1
    return out


def _position(text: str, heading: str, start: int) -> int | None:
    tokens = heading.split()
    pattern = r"\s*".join(re.escape(t) for t in tokens)
    m = re.compile(pattern).search(text, start)
    return m.start() if m else None


def section_marks(headings: list[Heading], text: str, stack: list[Heading]
                  ) -> tuple[tuple[tuple[int, str], ...], list[Heading]]:
    """Character offsets in the page text where the section path changes.

    `stack` is the open heading hierarchy from earlier pages (updated copy returned). The
    first mark (offset 0) is the section carried in from earlier pages, if any. A heading
    that cannot be located in the page text (e.g. removed with a table) still updates the
    hierarchy, at the position of the previous mark.
    """
    stack = list(stack)
    marks: list[tuple[int, str]] = []
    if stack:
        marks.append((0, " > ".join(h.text for h in stack)))
    cursor = 0
    for h in headings:
        while stack and stack[-1].level >= h.level:
            stack.pop()
        stack.append(h)
        pos = _position(text, h.text, cursor)
        if pos is not None:
            cursor = pos
        path = " > ".join(x.text for x in stack)
        if marks and marks[-1][0] == cursor:
            marks[-1] = (cursor, path)
        else:
            marks.append((cursor, path))
    return tuple(marks), stack


def section_at(marks: tuple[tuple[int, str], ...], pos: int) -> str | None:
    """The section path in effect at character `pos` (None before the first heading)."""
    current = None
    for offset, path in marks:
        if offset <= pos:
            current = path
        else:
            break
    return current
