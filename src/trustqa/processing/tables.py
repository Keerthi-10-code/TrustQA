"""Table extraction (pdfplumber) and row-aware table chunks (FR-04, NFR-03).

Two detectors run on every digital page:

1. Rule-bounded tables (academic "booktabs" style: horizontal rules, few or no vertical
   rules). A region is two or more long horizontal rules sharing the same x-extent with a
   "Table N" caption directly above or below; the caption requirement keeps algorithm
   boxes and framed figures out. Cells are rebuilt from pdfplumber words:
   - body lines are clustered by vertical position, words merged into phrases;
   - column boundaries are the x-gaps no body phrase covers, plus vertical rules;
   - the header block is everything above the first long rule inside the region; group
     headers ("Sparse" over DeepCT/SPARTA/docT5query) are mapped onto the columns under
     their partial rule (\\cmidrule) or, without one, onto the columns they overlap.
     Multi-level headers are joined top-down with " / " ("Sparse / DeepCT");
   - wrapped text lines (no numbers, mostly empty, first cell empty) are merged into the
     row above.
2. Fully ruled grids via pdfplumber's built-in lines strategy, kept only when the grid
   looks clean (few empty cells, no cells holding several numbers or several lines) and it
   does not overlap a rule-bounded table.

Cell text is the PDF's own characters: words are joined with a space only where the PDF
has a visible gap, nothing is re-formatted, so numbers, signs and decimals are preserved
exactly. The only substitution is undecodable ZapfDingbats check/cross glyphs
("(cid:51)" -> "✓", "(cid:55)" -> "✗").

Known limits: tables on OCR pages are not extracted (their text is still OCR'd as page
text); borderless tables without a caption and rules are not detected; row labels that a
PDF typesets as one cell spanning several rows stay split per line.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field

import numpy as np

from trustqa.schemas import Chunk, ContentType, ExtractionMethod, TableData

TABLES_VERSION = "tables-v1"

# "Table 3:" / "Tab. 3." / "TABLE 3" - punctuation required for mixed case so prose such
# as "Table 4 shows ..." is not taken for a caption
CAPTION_RE = re.compile(r"^(?:(?:Table|Tab\.)\s*([A-Z]?\d+[a-z]?)\s*[:.]|TABLE\s+([A-Z]?\d+[a-z]?)\b[:.]?)")
_NUMBER_RE = re.compile(r"[-+−]?\d[\d,]*(?:\.\d+)?%?")
_DINGBATS = {"(cid:51)": "✓", "(cid:52)": "✔", "(cid:53)": "✕", "(cid:54)": "✖",
             "(cid:55)": "✗", "(cid:56)": "✘"}
_UNIT_PATTERNS = [
    re.compile(r"\(\s*(?:in\s+)?((?:Rs\.?|INR|USD|US\$|\$|€|£|EUR)\s*(?:in\s+)?"
               r"(?:crores?|lakhs?|millions?|billions?|thousands?|mn|bn|cr)?)\s*\)", re.I),
    re.compile(r"\b[Ii]n\s+((?:Rs\.?|INR|USD|\$|€|£)?\s*(?:crores?|lakhs?|millions?|"
               r"billions?|thousands?))\b"),
    re.compile(r"\b(?:[Ss]cores?|[Vv]alues?|[Nn]umbers?|[Rr]esults?)\s+(?:denote|are"
               r"(?:\s+reported)?\s+in|in)\s+([A-Za-z][\w@.%/-]*[\w@%])"),
    re.compile(r"\((%|percent|ms|seconds|MB|GB|kg|km|MW|GW|kWh)\)"),
]


@dataclass
class ExtractedTable:
    table_id: str  # DOC02_P7_T1
    page_number: int
    bbox: tuple[float, float, float, float]  # x0, top, x1, bottom (PDF points)
    headers: list[str]
    rows: list[list[str]]
    caption: str | None = None
    label: str | None = None  # "Table 2" from the caption
    units: str | None = None
    method: str = "rules"  # rules | lines
    notes: list[str] = field(default_factory=list)

    def to_table_data(self, rows: list[list[str]] | None = None) -> TableData:
        return TableData(table_id=self.table_id, headers=list(self.headers),
                         rows=[list(r) for r in (self.rows if rows is None else rows)],
                         units=self.units, caption=self.caption)


# ---------------------------------------------------------------- geometry helpers

@dataclass
class _Word:
    text: str
    x0: float
    x1: float
    top: float
    bottom: float
    size: float

    @property
    def cy(self) -> float:
        return (self.top + self.bottom) / 2


@dataclass
class _Phrase:
    text: str
    x0: float
    x1: float
    top: float
    bottom: float


@dataclass
class _Line:
    words: list[_Word]

    @property
    def top(self) -> float:
        return min(w.top for w in self.words)

    @property
    def bottom(self) -> float:
        return max(w.bottom for w in self.words)

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words)


@dataclass
class _Rule:
    x0: float
    x1: float
    y: float

    @property
    def width(self) -> float:
        return self.x1 - self.x0


def _words(page, bbox=None) -> list[_Word]:
    src = page.crop(bbox, strict=False) if bbox else page
    out = []
    # Tolerance relative to glyph size: word spaces in small table fonts are ~0.25 em
    # (1.4pt at 5.7pt), below pdfplumber's absolute 3pt default.
    for w in src.extract_words(x_tolerance_ratio=0.15, y_tolerance=2, keep_blank_chars=False,
                               use_text_flow=False, extra_attrs=["size", "fontname"]):
        text = w["text"]
        if "Dingbat" in w.get("fontname", "") and "(cid:" in text:
            for k, v in _DINGBATS.items():
                text = text.replace(k, v)
        out.append(_Word(text, w["x0"], w["x1"], w["top"], w["bottom"], w.get("size", 8.0)))
    return out


def _cluster_lines(words: list[_Word]) -> list[_Line]:
    lines: list[_Line] = []
    for w in sorted(words, key=lambda w: (w.cy, w.x0)):
        if lines:
            cur = lines[-1]
            ref = statistics.median(x.cy for x in cur.words)
            line_size = max(x.size for x in cur.words)  # sub/superscripts are smaller
            if abs(w.cy - ref) <= 0.45 * max(w.size, line_size, 4.0):
                cur.words.append(w)
                continue
        lines.append(_Line([w]))
    for ln in lines:
        ln.words.sort(key=lambda w: w.x0)
    return lines


def _segments(lines: list[_Line], gap_em: float = 0.6) -> list[_Line]:
    """Split page lines at wide horizontal gaps, so text from side-by-side columns or a
    figure next to a table does not share a "line" with a caption. Column gutters can be
    as narrow as 0.8 em (BEIR p8: 8pt at 10pt); word spaces stay below ~0.4 em."""
    out: list[_Line] = []
    for ln in lines:
        cur = [ln.words[0]]
        for prev, w in zip(ln.words, ln.words[1:]):
            if w.x0 - prev.x1 > gap_em * max(prev.size, w.size):
                out.append(_Line(cur))
                cur = []
            cur.append(w)
        out.append(_Line(cur))
    return sorted(out, key=lambda s: (s.top, s.words[0].x0))


def _h_rules(page) -> list[_Rule]:
    raw = sorted((_Rule(e["x0"], e["x1"], e["top"]) for e in page.edges
                  if e["orientation"] == "h" and e["x1"] - e["x0"] >= 8),
                 key=lambda r: (r.y, r.x0))
    merged: list[_Rule] = []
    for r in raw:  # a filled thin rectangle yields two edges ~0.4pt apart
        if merged and abs(r.y - merged[-1].y) <= 1.0 and abs(r.x0 - merged[-1].x0) <= 2 \
                and abs(r.x1 - merged[-1].x1) <= 2:
            continue
        merged.append(r)
    return merged


@dataclass
class _VRule:
    x: float
    segments: list[tuple[float, float]]  # (top, bottom) pieces

    @property
    def height(self) -> float:
        return sum(b - a for a, b in self.segments)

    def crosses(self, y: float) -> bool:
        return any(a - 0.5 <= y <= b + 0.5 for a, b in self.segments)


def _v_rules(page, bbox) -> list[_VRule]:
    """Vertical rule segments inside bbox, grouped by x."""
    x0, top, x1, bottom = bbox
    out: list[_VRule] = []
    for e in page.edges:
        if e["orientation"] != "v":
            continue
        x = round((e["x0"] + e["x1"]) / 2, 1)
        if not x0 + 1 < x < x1 - 1:
            continue
        a, b = max(e["top"], top), min(e["bottom"], bottom)
        if b <= a:
            continue
        vr = next((v for v in out if abs(v.x - x) <= 1.0), None)
        if vr is None:
            out.append(_VRule(x, [(a, b)]))
        else:
            vr.segments.append((a, b))
    return sorted(out, key=lambda v: v.x)


def _join(prev: _Word, w: _Word) -> str:
    """Separator between two adjacent words of one phrase.

    No space where the glyphs touch (font changes such as "(", "→", ")"); a raised or
    lowered smaller alphanumeric glyph is marked "^"/"_" so 2.3·10^19 or n^2 never
    collapse into 1019 / n2. Symbols such as ‡ * † attach directly.
    """
    size = max(prev.size, w.size)
    if w.x0 - prev.x1 > 0.12 * size:
        return " "
    # Compare estimated baselines. pdfplumber boxes are exactly `size` tall with the
    # baseline ~0.21*size above the bottom (measured on CM and Times fonts), so a smaller
    # glyph on the same baseline (small caps: IsRel) gets no marker.
    smaller = w.size < 0.85 * prev.size
    if smaller and w.text[:1].isalnum():
        base_prev = prev.bottom - 0.21 * prev.size
        base_w = w.bottom - 0.21 * w.size
        if base_w < base_prev - 0.15 * prev.size:
            return "^"
        if base_w > base_prev + 0.08 * prev.size:
            return "_"
    return ""


def _phrases(line: _Line, vrules: list[_VRule]) -> list[_Phrase]:
    out: list[_Phrase] = []
    cur: list[_Word] = []
    texts: list[str] = []
    cy = statistics.median(w.cy for w in line.words)
    vxs = [v.x for v in vrules if v.crosses(cy)]

    def flush():
        if cur:
            out.append(_Phrase("".join(texts), cur[0].x0, max(w.x1 for w in cur),
                               min(w.top for w in cur), max(w.bottom for w in cur)))

    for w in line.words:
        if cur:
            prev = cur[-1]
            gap = w.x0 - prev.x1
            if gap > 0.5 * max(prev.size, w.size) or any(prev.x1 <= x <= w.x0 for x in vxs):
                flush()
                cur, texts = [], []
            else:
                texts.append(_join(prev, w))
        cur.append(w)
        texts.append(w.text)
    flush()
    return out


def _is_numeric(text: str) -> bool:
    return bool(_NUMBER_RE.search(text))


# ---------------------------------------------------------------- captions and units

_MAX_CAPTION_ABOVE = 70.0  # pt from caption end to first rule (room for an unruled header)


def _caption_candidates(lines: list[_Line], region: tuple[float, float, float, float]
                        ) -> list[tuple[float, int, str, str, bool, float]]:
    """(distance, line index, label, caption text, is_above, caption bottom) for
    "Table N" captions directly below or above the region. Distance is measured from the
    caption block's nearest edge (its last line, for a caption above)."""
    x0, top, x1, bottom = region
    out = []
    for i, ln in enumerate(lines):
        m = CAPTION_RE.match(ln.text)
        if not m:
            continue
        if min(x1, max(w.x1 for w in ln.words)) - max(x0, min(w.x0 for w in ln.words)) <= 0:
            continue
        if bottom - 1 <= ln.top <= bottom + 30:
            above = False
        elif top - 150 <= ln.bottom <= top + 1:
            above = True
        else:
            continue
        # continuation lines: the segments below in the same column
        cx0, cx1 = ln.words[0].x0, max(w.x1 for w in ln.words)
        parts = [ln]
        for nxt in lines[i + 1:]:
            nx0, nx1 = nxt.words[0].x0, max(w.x1 for w in nxt.words)
            if min(cx1, nx1) - max(cx0, nx0) < 0.5 * min(cx1 - cx0, nx1 - nx0):
                continue  # other column
            h = parts[-1].bottom - parts[-1].top
            if nxt.top - parts[-1].bottom > max(4.0, 0.8 * h) or CAPTION_RE.match(nxt.text):
                break
            if (above and nxt.bottom > top + 1) or len(parts) >= 8:
                break
            parts.append(nxt)
        cap_bottom = parts[-1].bottom
        if above:
            dist = top - cap_bottom
            if dist > _MAX_CAPTION_ABOVE:
                continue
        else:
            dist = ln.top - bottom
        text = re.sub(r"\s+", " ", " ".join(p.text for p in parts)).strip()
        out.append((max(dist, 0.0), i, f"Table {m.group(1) or m.group(2)}", text, above,
                    cap_bottom))
    return out


def _assign_captions(regions: list[tuple], lines: list[_Line]
                     ) -> dict[int, tuple[str, str, bool, float]]:
    """Greedy nearest-first matching: each caption and each region used at most once.
    Returns region index -> (label, caption text, caption is above, caption bottom)."""
    cands = sorted((c[0], ri, c[1], c[2:]) for ri, reg in enumerate(regions)
                   for c in _caption_candidates(lines, reg))
    used_r: set[int] = set()
    used_l: set[int] = set()
    out: dict[int, tuple[str, str, bool, float]] = {}
    for _, ri, li, info in cands:
        if ri in used_r or li in used_l:
            continue
        used_r.add(ri)
        used_l.add(li)
        out[ri] = info
    return out


def detect_units(caption: str | None, headers: list[str]) -> str | None:
    found: list[str] = []
    for text in [caption or "", *headers]:
        for pat in _UNIT_PATTERNS:
            for m in pat.finditer(text):
                u = re.sub(r"\s+", " ", m.group(1)).strip()
                if u and u not in found:
                    found.append(u)
    return ", ".join(found) if found else None


# ---------------------------------------------------------------- rule-bounded tables

def _rule_regions(rules: list[_Rule], lines: list[_Line], page_width: float
                  ) -> list[list[_Rule]]:
    long_rules = [r for r in rules if r.width >= max(100.0, 0.25 * page_width)]
    by_extent: list[list[_Rule]] = []
    for r in long_rules:
        for grp in by_extent:
            if abs(grp[0].x0 - r.x0) <= 3 and abs(grp[0].x1 - r.x1) <= 3:
                grp.append(r)
                break
        else:
            by_extent.append([r])
    caption_tops = [ln.top for ln in lines if CAPTION_RE.match(ln.text)]
    groups: list[list[_Rule]] = []
    for grp in by_extent:
        grp.sort(key=lambda r: r.y)
        cur = [grp[0]]
        for r in grp[1:]:
            gap_has_caption = any(cur[-1].y < t < r.y for t in caption_tops)
            if r.y - cur[-1].y > 250 or gap_has_caption:
                groups.append(cur)
                cur = [r]
            else:
                cur.append(r)
        groups.append(cur)
    return [g for g in groups if g[-1].y - g[0].y > 10]


def _columns(body_lines: list[list[_Phrase]], x0: float, x1: float,
             vrules: list[_VRule], body_height: float) -> list[tuple[float, float]]:
    """Column intervals from the x-gaps that no body phrase covers.

    A gap crossed only by a few phrases that each span it completely (merged
    \\multicolumn cells) is still a gap; such phrases go to their largest-overlap column.
    A vertical rule spanning >= 30% of the body is a boundary too: inside an empty gap it
    replaces the gap's midpoint; it may also cut through a few overhanging phrases (a long
    row label running past the rule), which are then assigned by largest overlap.
    """
    step = 0.25
    n = int((x1 - x0) / step) + 2
    cov = np.zeros(n, dtype=np.int32)
    multi = [ps for ps in body_lines if len(ps) >= 2] or body_lines
    spans: list[tuple[int, int]] = []
    for ps in multi:
        for p in ps:
            a = max(0, int((p.x0 - x0) / step))
            b = min(n, int(np.ceil((p.x1 - x0) / step)))
            cov[a:b] += 1
            spans.append((a, b))
    covered = np.nonzero(cov)[0]
    if covered.size == 0:
        return [(x0, x1)]
    strong = [v.x for v in vrules if v.height >= 0.3 * body_height]
    max_cut = max(1, int(0.2 * len(multi)))
    bounds: list[float] = []
    i, last = covered[0], covered[-1]
    while i <= last:
        zero = cov[i] == 0
        low = 0 < cov[i] <= max_cut
        if not (zero or low):
            i += 1
            continue
        j = i
        while j <= last and ((cov[j] == 0) if zero else (0 < cov[j] <= max_cut)):
            j += 1
        if zero:
            accept = True
        else:  # low run: a gap only if bounded by busier columns and fully spanned
            crossing = [(a, b) for a, b in spans if a < j and b > i]
            accept = (i > covered[0] and j <= last and cov[i - 1] > max_cut
                      and cov[j] > max_cut and all(a <= i and b >= j for a, b in crossing))
        if accept:
            lo, hi = x0 + i * step, x0 + j * step
            inside = [vx for vx in strong if lo - step <= vx <= hi + step]
            bounds.append(inside[0] if inside else (lo + hi) / 2)
        i = j
    max_cut = max(1, int(0.15 * len(multi)))
    for vx in strong:
        idx = int((vx - x0) / step)
        if 0 <= idx < n and 0 < cov[idx] <= max_cut:
            bounds.append(vx)
    bounds = sorted(bounds)
    dedup: list[float] = []
    for b in bounds:
        if not dedup or b - dedup[-1] > 2.0:
            dedup.append(b)
    edges = [x0, *dedup, x1]
    return [(edges[k], edges[k + 1]) for k in range(len(edges) - 1)]


def _col_for(p: _Phrase, cols: list[tuple[float, float]]) -> int:
    best, best_ov = 0, -1e9
    for j, (a, b) in enumerate(cols):
        ov = min(b, p.x1) - max(a, p.x0)
        if ov > best_ov:
            best, best_ov = j, ov
    return best


def _build_rule_table(page, group: list[_Rule], rules: list[_Rule],
                      top_override: float | None = None
                      ) -> tuple[list[str], list[list[str]], tuple, list[str]] | None:
    """Build headers/rows for a rule-bounded region. `top_override` (above the first rule)
    covers tables without a top rule whose header sits between caption and first rule."""
    x0, x1 = min(r.x0 for r in group), max(r.x1 for r in group)
    top, bottom = group[0].y, group[-1].y
    if top_override is not None and top_override < top:
        top = top_override
    bbox = (x0 - 2, top, x1 + 2, bottom + 1)
    words = [w for w in _words(page, bbox) if w.cy > top and w.cy < bottom]
    if not words:
        return None
    lines = _cluster_lines(words)
    # header separator: the first long rule inside the region with text above it
    sep = next((r.y for r in group[:-1]
                if r.y > top + 2 and any(ln.bottom <= r.y + 1 for ln in lines)), None)
    if sep is not None:
        head_lines = [ln for ln in lines if ln.bottom <= sep + 1]
        body = [ln for ln in lines if ln.top >= sep - 1]
    else:
        head_lines, body = lines[:1], lines[1:]
    if len(body) < 2:
        return None
    vr = _v_rules(page, (x0, top, x1, bottom))
    body_ph = [_phrases(ln, vr) for ln in body]
    cols = _columns(body_ph, x0, x1, vr, bottom - (sep or top))
    ncols = len(cols)
    if ncols < 2:
        return None
    notes: list[str] = []
    extents: list[list[float]] = [[np.inf, -np.inf] for _ in range(ncols)]

    # body rows (with wrapped-line merging)
    rows: list[list[str]] = []
    row_bottoms: list[float] = []
    inner_rules = [r.y for r in rules if top < r.y < bottom]
    for ln, ps in zip(body, body_ph):
        cells = [""] * ncols
        for p in ps:
            j = _col_for(p, cols)
            cells[j] = f"{cells[j]} {p.text}".strip()
            a, b = cols[j]
            if len(ps) >= 2 and p.x0 >= a - 1 and p.x1 <= b + 1:  # not overhanging
                extents[j] = [min(extents[j][0], p.x0), max(extents[j][1], p.x1)]
        filled = [j for j, c in enumerate(cells) if c]
        is_wrap = (rows and not cells[0] and len(filled) <= max(1, ncols // 3)
                   and not any(_is_numeric(cells[j]) for j in filled)
                   and ln.top - row_bottoms[-1] <= 0.8 * (ln.bottom - ln.top)
                   and not any(row_bottoms[-1] - 1 < y < ln.top + 1 for y in inner_rules))
        if is_wrap:
            for j in filled:
                rows[-1][j] = f"{rows[-1][j]} {cells[j]}".strip()
            row_bottoms[-1] = ln.bottom
            continue
        rows.append(cells)
        row_bottoms.append(ln.bottom)

    # headers, with group headers spread over the columns under their partial rule
    # content extent of each column (where its cells actually sit); fall back to bounds
    ext = [(a, b) if a < b else cols[j] for j, (a, b) in enumerate(extents)]
    parts: list[list[tuple[int, str]]] = [[] for _ in range(ncols)]
    for li, ln in enumerate(head_lines):
        for p in _phrases(ln, vr):
            cmid = next((r for r in rules
                         if p.bottom - 1 <= r.y <= p.bottom + 8 and r.width < 0.9 * (x1 - x0)
                         and min(r.x1, p.x1) - max(r.x0, p.x0) >= 0.5 * (p.x1 - p.x0)), None)
            if cmid is not None:
                targets = [j for j, (a, b) in enumerate(ext)
                           if cmid.x0 - 1 <= (a + b) / 2 <= cmid.x1 + 1]
            else:  # columns whose content the phrase covers substantially
                targets = [j for j, (a, b) in enumerate(ext)
                           if min(b, p.x1) - max(a, p.x0) >= 0.3 * min(b - a, p.x1 - p.x0)]
            if not targets:
                targets = [_col_for(p, cols)]
            if len(targets) > 1:
                notes.append(f"header {p.text!r} spans {len(targets)} columns")
            for j in targets:
                if parts[j] and parts[j][-1][0] == li:
                    parts[j][-1] = (li, f"{parts[j][-1][1]} {p.text}")
                else:
                    parts[j].append((li, p.text))
    headers = [" / ".join(t for _, t in parts[j]) or f"Column {j + 1}" for j in range(ncols)]
    # drop columns that are empty in every body row and have no header of their own
    keep = [j for j in range(ncols) if any(r[j] for r in rows) or parts[j]]
    if len(keep) < ncols:
        headers = [headers[j] for j in keep]
        rows = [[r[j] for j in keep] for r in rows]
    if _looks_like_figure(headers, rows):
        return None
    return headers, rows, (x0, top, x1, bottom), notes


_PLAIN_NUMBER_RE = re.compile(r"^[-+]?\d+(?:\.\d+)?$")
_YEAR_RE = re.compile(r"^(?:19|20)\d\d$")


def _looks_like_figure(headers: list[str], rows: list[list[str]]) -> bool:
    """Plots captioned as tables: axis ticks become numeric 'headers', cells mostly empty."""
    ticks = [h for h in headers for part in h.split(" / ")[-1:]
             if _PLAIN_NUMBER_RE.match(part) and not _YEAR_RE.match(part)]
    cells = [c for r in rows for c in r]
    fill = sum(bool(c) for c in cells) / max(1, len(cells))
    return len(ticks) >= 2 or fill < 0.3


# ---------------------------------------------------------------- lines-strategy grids

def _clean_lines_table(raw: list[list[str | None]]) -> tuple[list[str], list[list[str]]] | None:
    if len(raw) < 2 or max(len(r) for r in raw) < 2:
        return None
    grid = [[(c or "").strip() for c in r] for r in raw]
    cells = [c for r in grid for c in r]
    nonempty = [c for c in cells if c]
    if len(nonempty) < 0.6 * len(cells):
        return None
    if sum("\n" in c for c in nonempty) > 0.25 * len(nonempty):
        return None
    if sum(len(_NUMBER_RE.findall(c)) >= 2 for c in nonempty) > 0.25 * len(nonempty):
        return None  # several numbers per cell: columns were merged
    raw0 = list(raw[0])
    head0: list[str] = []
    for c in raw0:  # merged header cells come back as None: carry the label right
        head0.append((c or "").replace("\n", " ").strip() if c is not None
                     else (head0[-1] if head0 else ""))
    headers, body = head0, grid[1:]
    second = grid[1]
    # a merged (spanning) header cell means a two-level header; the second level may be
    # numeric (years), so only a second row of plain values (decimals, %) stops it
    looks_like_data = any(_is_numeric(c) and not _YEAR_RE.match(c) for c in second if c)
    if any(c is None for c in raw0) and not looks_like_data:
        headers = [" / ".join(x for x in (a, b.replace("\n", " ")) if x)
                   for a, b in zip(head0, second)]
        body = grid[2:]
    body = [[c.replace("\n", " ") for c in r] for r in body]
    if not body:
        return None
    headers = [h or f"Column {j + 1}" for j, h in enumerate(headers)]
    return headers, body


def _overlap_ratio(a: tuple, b: tuple) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    area = max(1e-6, min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1])))
    return ix * iy / area


# ---------------------------------------------------------------- public API

def find_tables(page, page_number: int, document_id: str) -> list[ExtractedTable]:
    """Detect and extract tables on one pdfplumber page, in top-to-bottom order."""
    rules = _h_rules(page)
    page_words = _words(page)
    page_lines = _segments(_cluster_lines(page_words))
    found: list[ExtractedTable] = []

    groups = _rule_regions(rules, page_lines, float(page.width))
    regions = [(min(r.x0 for r in g), g[0].y, max(r.x1 for r in g), g[-1].y) for g in groups]
    captions = _assign_captions(regions, page_lines)
    for ri, group in enumerate(groups):
        if ri not in captions:
            continue  # rules without a "Table N" caption: algorithm box, framed figure...
        label, caption, above, cap_bottom = captions[ri]
        built = _build_rule_table(page, group, rules,
                                  top_override=cap_bottom + 1 if above else None)
        if built is None:
            continue
        headers, rows, bbox, notes = built
        found.append(ExtractedTable("", page_number, bbox, headers, rows, caption=caption,
                                    label=label, method="rules", notes=notes))

    grids = []
    for t in page.find_tables():
        if any(_overlap_ratio(t.bbox, f.bbox) > 0.2 for f in found):
            continue
        cleaned = _clean_lines_table(t.extract())
        if cleaned is not None:
            grids.append((tuple(t.bbox), cleaned))
    grid_caps = _assign_captions([b for b, _ in grids], page_lines)
    for gi, (bbox, (headers, rows)) in enumerate(grids):
        cap = grid_caps.get(gi)
        found.append(ExtractedTable("", page_number, bbox, headers, rows,
                                    caption=cap[1] if cap else None,
                                    label=cap[0] if cap else None, method="lines"))

    found.sort(key=lambda t: (t.bbox[1], t.bbox[0]))
    for k, t in enumerate(found, start=1):
        t.table_id = f"{document_id}_P{page_number}_T{k}"
        t.units = detect_units(t.caption, t.headers)
    return found


def _short_caption(table: ExtractedTable, limit: int = 160) -> str:
    cap = table.caption or ""
    m = CAPTION_RE.match(cap)
    if m:
        cap = cap[m.end():].strip()
    first = re.split(r"(?<=[.!?])\s+", cap, maxsplit=1)[0] if cap else ""
    if len(first) > limit:
        first = first[:limit].rsplit(" ", 1)[0] + " ..."
    return first


def row_text(table: ExtractedTable, row: list[str], with_caption: bool = True) -> str:
    """Row-aware text: table label, caption and units context, then header=value pairs.

    e.g. "Table 2: In-domain and zero-shot performances on BEIR benchmark. (nDCG@10) |
          Model (→) / Dataset (↓)=TREC-COVID | Lexical / BM25=0.656 | ..."
    Cell values are inserted verbatim. Empty cells are omitted.
    """
    label = table.label or table.table_id
    ctx = label
    if with_caption and table.caption:
        short = _short_caption(table)
        if short:
            ctx = f"{label}: {short}"
    if table.units and table.units not in ctx:
        ctx = f"{ctx} ({table.units})"
    pairs = [f"{h}={v}" for h, v in zip(table.headers, row) if v]
    return " | ".join([ctx, *pairs])


def table_chunks(table: ExtractedTable, document_id: str, document_name: str,
                 max_chars: int | None = None) -> list[Chunk]:
    """One chunk per data row, each carrying the table context and its row as table_data.

    If a row's text exceeds max_chars the caption is dropped from the context (the label
    and units stay); cells are never cut.
    """
    chunks = []
    for r, row in enumerate(table.rows, start=1):
        if not any(row):
            continue
        text = row_text(table, row)
        if max_chars and len(text) > max_chars:
            text = row_text(table, row, with_caption=False)
        chunks.append(Chunk(
            document_id=document_id,
            document_name=document_name,
            page_number=table.page_number,
            chunk_id=f"{table.table_id}_R{r}",
            text=text,
            content_type=ContentType.TABLE,
            table_data=table.to_table_data(rows=[row]),
            extraction_method=ExtractionMethod.PDFPLUMBER,
            ocr_used=False,
            section=table.label,
        ))
    return chunks


_ROW_SUFFIX_RE = re.compile(r"_R(\d+)$")


def rebuild_table(chunks: list[Chunk], table_id: str) -> TableData | None:
    """Reassemble a full table from its row chunks (e.g. all chunks in an index).

    Row chunks deliberately carry only their own row in table_data, so a cited row is
    exactly the evidence checked; use this to show the whole table (Phase 5 UI).
    Rows come back in their original order (by the _R<n> chunk-id suffix).
    """
    rows = [c for c in chunks if c.table_data is not None and c.table_data.table_id == table_id]
    if not rows:
        return None

    def order(c: Chunk) -> int:
        m = _ROW_SUFFIX_RE.search(c.chunk_id)
        return int(m.group(1)) if m else 0

    rows.sort(key=order)
    first = rows[0].table_data
    return TableData(table_id=table_id, headers=list(first.headers),
                     rows=[r for c in rows for r in c.table_data.rows],
                     units=first.units, caption=first.caption)
