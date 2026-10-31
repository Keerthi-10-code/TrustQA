"""Grounded answer generation (FR-08) with metadata-resolved citations (FR-09 basics).

The LLM sees only the retrieved passages, each labelled with its chunk ID, and must
return JSON:

    {"answer": "... [DOC01_P4_C1]", "chunk_ids": ["DOC01_P4_C1"], "insufficient_evidence": false}

Trust boundary: the LLM contributes only the answer text and a list of chunk IDs. IDs are
accepted only if they belong to the retrieved set; document name and page shown to users
are looked up from those chunks' metadata. Anything else the model writes about sources
(file names, page numbers, extra JSON fields) is ignored.

Phase 1 does no claim-support checking: this module answers "which retrieved chunks did
the model cite", not "do they support the answer" (Phase 3).
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from trustqa.llm import LLM, LLMError
from trustqa.schemas import Chunk, Citation, RetrievedChunk, Warning, WarningCode

PROMPT_VERSION = "grounded-v1"

ABSTENTION_ANSWER = (
    "Insufficient evidence: the retrieved passages do not contain enough information "
    "to answer this question."
)

# DOC01_P4_C1 (text) and later forms such as DOC02_P7_T1_R1 (table rows, Phase 2).
CHUNK_ID_RE = re.compile(r"\bDOC\d+_P\d+(?:_[A-Z]+\d+)+\b")
_BRACKET_RE = re.compile(r"\s?\[([^\[\]]*)\]")

SYSTEM_PROMPT = """You answer questions using ONLY the numbered passages you are given.

Rules:
1. Use only facts stated in the passages. Do not use outside knowledge, even if you know the answer.
2. After every sentence of your answer, cite the passage(s) it comes from with their IDs in square brackets, exactly as written, e.g. [DOC01_P4_C1] or [DOC01_P4_C1, DOC02_P7_C3].
3. "chunk_ids" must list every passage ID you cited. Each item is only an ID such as "DOC01_P4_C1" - never passage text, file names or page numbers.
4. If the passages do not contain enough information to answer, set "insufficient_evidence" to true, "answer" to "" and "chunk_ids" to []. Do not guess.
5. Keep the answer concise.

Return a JSON object with exactly these keys:
{"answer": string, "chunk_ids": [string], "insufficient_evidence": boolean}"""


def format_passages(passages: Sequence[RetrievedChunk]) -> str:
    blocks = []
    for rc in passages:
        c = rc.chunk
        blocks.append(f"[{c.chunk_id}] (source: {c.document_name}, page {c.page_number})\n"
                      f"{c.text}")
    return "\n\n".join(blocks)


def build_prompt(question: str, passages: Sequence[RetrievedChunk]) -> str:
    ids = ", ".join(rc.chunk.chunk_id for rc in passages)
    return (f"Passages:\n\n{format_passages(passages)}\n\n"
            f"Valid passage IDs: {ids}\n\n"
            f"Question: {question}\n\n"
            "Answer in JSON as instructed.")


@dataclass
class GenerationOutput:
    """Parsed, ID-validated model output (before Result assembly)."""

    answer: str
    cited_ids: list[str]  # valid IDs, in first-cited order
    abstained: bool
    warnings: list[Warning] = field(default_factory=list)
    raw: str = ""


def _ids_in(value: object) -> list[str]:
    return CHUNK_ID_RE.findall(value) if isinstance(value, str) else []


def parse_generation(raw: str, passages: Sequence[RetrievedChunk]) -> GenerationOutput:
    """Validate model JSON against the retrieved set. Raises LLMError on non-object JSON."""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise LLMError(f"generation output is not JSON: {e}") from e
    if not isinstance(data, dict):
        raise LLMError(f"generation output must be a JSON object, got {type(data).__name__}")

    valid = {rc.chunk.chunk_id for rc in passages}
    warnings: list[Warning] = []
    answer = data.get("answer")
    answer = answer.strip() if isinstance(answer, str) else ""
    flag = data.get("insufficient_evidence", False)
    abstained = flag is True or (isinstance(flag, str) and flag.strip().lower() == "true")

    listed = data.get("chunk_ids", [])
    if isinstance(listed, str):
        listed = [listed]
    if not isinstance(listed, list):
        listed = []

    candidates: list[str] = []
    for item in listed:
        found = _ids_in(item)
        if not found:
            snippet = str(item)[:80]
            warnings.append(Warning(WarningCode.INVALID_CHUNK_ID,
                                    f"ignored citation entry that is not a chunk ID: {snippet!r}"))
        candidates.extend(found)
    candidates.extend(_ids_in(answer))  # inline [ID] markers count as citations too

    cited: list[str] = []
    rejected: list[str] = []
    for cid in candidates:
        if cid in valid:
            if cid not in cited:
                cited.append(cid)
        elif cid not in rejected:
            rejected.append(cid)
    for cid in rejected:
        warnings.append(Warning(WarningCode.INVALID_CHUNK_ID,
                                f"cited chunk {cid} is not among the retrieved passages; ignored",
                                chunk_id=cid))

    answer = strip_invalid_markers(answer, valid)
    return GenerationOutput(answer=answer, cited_ids=cited, abstained=abstained,
                            warnings=warnings, raw=raw)


def strip_invalid_markers(answer: str, valid: set[str]) -> str:
    """Remove unretrieved IDs from inline [..] markers; other brackets are untouched."""

    def fix(m: re.Match) -> str:
        ids = CHUNK_ID_RE.findall(m.group(1))
        if not ids:
            return m.group(0)  # e.g. a "[1]" reference in quoted text
        keep = [i for i in dict.fromkeys(ids) if i in valid]
        return f" [{', '.join(keep)}]" if keep else ""

    return _BRACKET_RE.sub(fix, answer).strip()


def resolve_citations(cited_ids: Sequence[str], passages: Sequence[RetrievedChunk]
                      ) -> list[Citation]:
    """Build citations from trusted chunk metadata, one per (document, page), in cite order."""
    by_id: dict[str, Chunk] = {rc.chunk.chunk_id: rc.chunk for rc in passages}
    grouped: dict[tuple[str, int], list[str]] = {}
    for cid in cited_ids:
        c = by_id[cid]  # KeyError here would mean an unvalidated ID slipped through
        grouped.setdefault((c.document_id, c.page_number), []).append(cid)
    citations = []
    for (doc_id, page), ids in grouped.items():
        c = by_id[ids[0]]
        citations.append(Citation(document_id=doc_id, document_name=c.document_name,
                                  page_number=page, chunk_ids=ids, supporting_quote=""))
    return citations


def generate(question: str, passages: Sequence[RetrievedChunk], llm: LLM) -> GenerationOutput:
    """Ask the LLM for a grounded answer. With no passages, abstains without a call."""
    if not passages:
        return GenerationOutput(answer="", cited_ids=[], abstained=True)
    raw = llm.complete(build_prompt(question, passages), system=SYSTEM_PROMPT, json=True)
    return parse_generation(raw, passages)
