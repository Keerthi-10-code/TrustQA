"""Phase 3 item 1 - citation registry and provenance check (FR-09, NFR-01, PRD 2.3).

Answers "does every citation resolve to the correct document and page?", separately from
"does the cited evidence support the claim?" (item 3). Plain Python, no LLM calls.

Trust boundary
- CitationRegistry is built only from the retrieved passages (trusted chunk metadata).
  A chunk is citable only if its chunk_id is well formed and agrees with its own
  metadata (DOCnn prefix == document_id, Pnn == page_number); a disagreement means the
  record is corrupt, so it is excluded rather than trusted.
- The LLM contributes only chunk IDs and answer prose. IDs outside the registry are
  rejected (INVALID_CHUNK_ID). Document names and page numbers the model writes in its
  prose are never used for citations; they are only compared against the metadata of the
  chunks they are attached to, and a disagreement is reported (PAGE_MISMATCH).
- So users never see two conflicting sources, a false source note is removed from the
  displayed answer when it is a source-only parenthetical such as "(annual.pdf, page 99)".
  A false mention inside running prose ("as stated on page 99, ...") is left in place,
  since cutting it would garble the sentence; the warning covers it. The frozen schema has
  no document-mismatch code, so a wrong document name is also reported as PAGE_MISMATCH
  and the message says which kind it is.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from trustqa.generation import CHUNK_ID_RE, strip_invalid_markers
from trustqa.schemas import Chunk, Citation, RetrievedChunk, Warning, WarningCode

_ID_PARTS_RE = re.compile(r"^(DOC\d+)_P(\d+)(?:_[A-Z]+\d+)+$")
_MARKER_RE = re.compile(r"\[([^\[\]]*)\]")
# "page 3", "pages 3-5", "pages 3 to 5", "p. 3", "pp. 3-5", "p3"
_PAGE_RE = re.compile(
    r"\b(?:pages?|pp?\.?)\s*(\d+)(?:\s*(?:-|–|—|to)\s*(\d+))?\b", re.IGNORECASE)
_DOC_RE = re.compile(r"[\w.\-]+\.pdf\b", re.IGNORECASE)
# What may remain in a parenthetical that only names a source, e.g. "(see source: x.pdf, page 3)".
_SOURCE_FILLER_RE = re.compile(
    r"\b(?:source|sources|see|from|in|on|at|of|the|document|file|and)\b|[\s,;:.\-–—]",
    re.IGNORECASE)


@dataclass(frozen=True)
class CitationRegistry:
    """chunk_id -> trusted Chunk for the retrieved set. `problems` lists excluded IDs."""

    chunks: Mapping[str, Chunk]
    problems: Mapping[str, str] = field(default_factory=dict)

    @classmethod
    def from_passages(cls, passages: Sequence[RetrievedChunk]) -> CitationRegistry:
        chunks: dict[str, Chunk] = {}
        problems: dict[str, str] = {}
        for rc in passages:
            c = rc.chunk
            cid = c.chunk_id
            if cid in problems:
                continue
            issue = _id_metadata_issue(c)
            if issue is None and cid in chunks and chunks[cid] != c:
                issue = "duplicate chunk ID with different metadata in the retrieved set"
            if issue is not None:
                problems[cid] = issue
                chunks.pop(cid, None)
            else:
                chunks[cid] = c
        return cls(chunks=chunks, problems=problems)

    def __contains__(self, chunk_id: object) -> bool:
        return chunk_id in self.chunks

    def resolve(self, chunk_id: str) -> Chunk | None:
        return self.chunks.get(chunk_id)

    def citations(self, chunk_ids: Sequence[str]) -> list[Citation]:
        """One Citation per (document, page), in first-cited order; IDs must be resolvable."""
        grouped: dict[tuple[str, int], list[str]] = {}
        for cid in dict.fromkeys(chunk_ids):
            c = self.chunks[cid]
            grouped.setdefault((c.document_id, c.page_number), []).append(cid)
        return [Citation(document_id=doc, document_name=self.chunks[ids[0]].document_name,
                         page_number=page, chunk_ids=ids)
                for (doc, page), ids in grouped.items()]


def _id_metadata_issue(c: Chunk) -> str | None:
    m = _ID_PARTS_RE.match(c.chunk_id)
    if m is None:
        return "chunk ID format is not recognised"
    if m.group(1) != c.document_id:
        return f"chunk ID names document {m.group(1)} but metadata says {c.document_id}"
    if int(m.group(2)) != c.page_number:
        return f"chunk ID names page {int(m.group(2))} but metadata says page {c.page_number}"
    return None


@dataclass(frozen=True)
class SourceMention:
    """A page or document the model names in its prose, and the IDs it is attached to.

    scope = chunk IDs in the first citation marker after the mention; [] if none follows.
    """

    kind: str  # "page" | "document"
    value: tuple[int, int] | str  # inclusive page range, or a file name
    text: str
    scope: list[str]
    start: int = 0
    end: int = 0


def find_source_mentions(answer: str) -> list[SourceMention]:
    markers = [(m.start(), CHUNK_ID_RE.findall(m.group(1))) for m in _MARKER_RE.finditer(answer)]
    markers = [(pos, ids) for pos, ids in markers if ids]

    def scope_at(pos: int) -> list[str]:
        return next((list(ids) for mpos, ids in markers if mpos >= pos), [])

    found: list[tuple[int, SourceMention]] = []
    for m in _DOC_RE.finditer(answer):
        found.append((m.start(), SourceMention("document", m.group(0), m.group(0),
                                               scope_at(m.end()), m.start(), m.end())))
    for m in _PAGE_RE.finditer(answer):
        lo = int(m.group(1))
        hi = int(m.group(2)) if m.group(2) else lo
        found.append((m.start(), SourceMention("page", (min(lo, hi), max(lo, hi)), m.group(0),
                                               scope_at(m.end()), m.start(), m.end())))
    return [s for _, s in sorted(found, key=lambda t: t[0])]


@dataclass
class ProvenanceReport:
    """`answer` is the answer to display: rejected citation markers and false
    source-only parentheticals removed."""

    answer: str
    accepted_ids: list[str]
    rejected_ids: list[str]
    citations: list[Citation]
    warnings: list[Warning]
    upstream_unresolved: int = 0

    @property
    def all_resolved(self) -> bool:
        """PRD 2.3 citation condition for Verified: something was cited and every citation
        (and every prose source claim) resolves to the correct document and page."""
        return (bool(self.accepted_ids) and not self.rejected_ids
                and self.upstream_unresolved == 0
                and not any(w.code is WarningCode.PAGE_MISMATCH for w in self.warnings))


def check_provenance(
    answer: str,
    cited_ids: Sequence[str],
    passages: Sequence[RetrievedChunk],
    upstream_unresolved: int = 0,
) -> ProvenanceReport:
    """Validate cited IDs against the retrieved set and check prose source claims.

    `cited_ids`: chunk IDs the model cited (list and inline markers, any order).
    `upstream_unresolved`: citation entries an earlier parser already rejected and warned
    about (e.g. passage text in chunk_ids); they count against all_resolved without being
    warned twice.
    """
    reg = CitationRegistry.from_passages(passages)
    warnings: list[Warning] = []
    accepted: list[str] = []
    rejected: list[str] = []
    for cid in dict.fromkeys(cited_ids):
        if cid in reg:
            accepted.append(cid)
            continue
        rejected.append(cid)
        if cid in reg.problems:
            warnings.append(Warning(WarningCode.PAGE_MISMATCH,
                                    f"cited chunk {cid} cannot be trusted: {reg.problems[cid]}",
                                    chunk_id=cid))
        else:
            warnings.append(Warning(WarningCode.INVALID_CHUNK_ID,
                                    f"cited chunk {cid} is not among the retrieved passages; "
                                    "ignored", chunk_id=cid))

    spans: list[tuple[int, int]] = []
    for mention, warning in _mention_mismatches(answer, accepted, reg):
        span = _source_parenthetical(answer, mention)
        if span is not None:
            spans.append(span)
            warning.message += "; removed from the answer"
        warnings.append(warning)
    cleaned = strip_invalid_markers(_cut(answer, spans), set(accepted))
    return ProvenanceReport(answer=cleaned, accepted_ids=accepted, rejected_ids=rejected,
                            citations=reg.citations(accepted), warnings=warnings,
                            upstream_unresolved=upstream_unresolved)


def _mention_mismatches(answer: str, accepted: list[str], reg: CitationRegistry
                        ) -> list[tuple[SourceMention, Warning]]:
    out: list[tuple[SourceMention, Warning]] = []
    if not accepted:
        return out
    for mention in find_source_mentions(answer):
        scope = [cid for cid in mention.scope if cid in reg] or accepted
        chunks = [reg.chunks[cid] for cid in scope]
        if any(mention.text.lower() in c.text.lower() for c in chunks):
            continue  # the model is quoting the passage, not naming its source
        if mention.kind == "page":
            lo, hi = mention.value  # type: ignore[misc]
            pages = sorted({c.page_number for c in chunks})
            if any(lo <= p <= hi for p in pages):
                continue
            out.append((mention, Warning(
                WarningCode.PAGE_MISMATCH,
                f"answer says '{mention.text}' but the cited passage(s) {', '.join(scope)} "
                f"are on page {', '.join(map(str, pages))}", chunk_id=scope[0])))
        else:
            names = sorted({c.document_name for c in chunks})
            if mention.value.lower() in (n.lower() for n in names):  # type: ignore[union-attr]
                continue
            out.append((mention, Warning(
                WarningCode.PAGE_MISMATCH,
                f"answer names document '{mention.text}' but the cited passage(s) "
                f"{', '.join(scope)} are from {', '.join(names)}", chunk_id=scope[0])))
    return out


def _source_parenthetical(answer: str, mention: SourceMention) -> tuple[int, int] | None:
    """Span (with one leading space) of the (...) around `mention` if it only names sources."""
    open_ = answer.rfind("(", 0, mention.start)
    close = answer.find(")", mention.end)
    if (open_ < 0 or close < 0 or ")" in answer[open_:mention.start]
            or "(" in answer[mention.end:close]):
        return None
    inner = answer[open_ + 1:close]
    if _SOURCE_FILLER_RE.sub("", _PAGE_RE.sub("", _DOC_RE.sub("", inner))):
        return None  # carries content beyond a source note
    start = open_ - 1 if open_ > 0 and answer[open_ - 1] == " " else open_
    return start, close + 1


def _cut(text: str, spans: list[tuple[int, int]]) -> str:
    out, pos = [], 0
    for start, end in sorted(set(spans)):  # two mentions in one parenthetical share a span
        out.append(text[pos:start])
        pos = end
    out.append(text[pos:])
    return "".join(out)
