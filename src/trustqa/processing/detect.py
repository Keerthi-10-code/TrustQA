"""Per-page detection of scanned (no / too little text) and garbage text layers.

A page is sent to OCR when either
- it has fewer than `min_chars` alphanumeric characters (image-only scan, or a scan with
  only a stamp/header in its text layer), or
- more than `max_nonalnum_ratio` of its non-whitespace characters are not alphanumeric
  (broken font encodings that decode to symbols).

Undecodable glyphs count as non-alphanumeric even when their textual form contains
letters: U+FFFD, PDF "(cid:NN)" placeholders, private-use-area code points and control
characters. Ordinary punctuation is allowed generously (default 0.6) because legitimate
pages such as tables of contents with dot leaders reach ~0.5 (Self-RAG page 16: 0.49).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_CID_RE = re.compile(r"\(cid:\d+\)")


def _is_garbage_char(ch: str) -> bool:
    cp = ord(ch)
    return (ch == "�" or 0xE000 <= cp <= 0xF8FF or cp >= 0xF0000
            or (cp < 32 and ch not in "\t\n\r"))


@dataclass(frozen=True)
class PageQuality:
    alnum_chars: int
    nonalnum_ratio: float  # over non-whitespace chars; 1.0 for an empty page
    needs_ocr: bool
    reason: str  # "ok" | "low_text" | "garbage_text"


def page_quality(text: str, min_chars: int, max_nonalnum_ratio: float) -> PageQuality:
    """Classify one page's text layer. Pure function of the raw text and thresholds."""
    cids = _CID_RE.findall(text)
    cleaned = _CID_RE.sub("", text)
    alnum = other = 0
    for ch in cleaned:
        if ch.isspace():
            continue
        if ch.isalnum() and not _is_garbage_char(ch):
            alnum += 1
        else:
            other += 1
    other += len(cids)  # each (cid:N) counts as one undecodable glyph
    total = alnum + other
    ratio = other / total if total else 1.0
    if alnum < min_chars:
        return PageQuality(alnum, ratio, True, "low_text")
    if ratio > max_nonalnum_ratio:
        return PageQuality(alnum, ratio, True, "garbage_text")
    return PageQuality(alnum, ratio, False, "ok")
