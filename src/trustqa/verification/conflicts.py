"""Phase 3 item 5 - conflict detection across documents (FR-10 "conflicting claims").

Two steps:
1. Fact extraction (LLM, strict JSON): from the retrieved passages, list the numeric facts
   relevant to the question as {chunk_id, entity, metric, period, value, unit}.
2. Comparison (plain Python): a fact is kept only if its chunk was retrieved and its value
   occurs in that chunk's text (a hallucinated fact cannot create a conflict). Facts are
   grouped by normalised entity + metric + period; a group is a conflict when it holds
   values from different documents that differ beyond rounding, in compatible units, and at
   least one value comes from a chunk the answer cites (otherwise the conflict does not bear
   on the answer). Both sources are surfaced with document name and page from metadata.

The extraction call runs only when the retrieved passages span two or more documents and
the answer states a number, so single-document questions cost nothing extra.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal

from rapidfuzz import fuzz

from trustqa.llm import LLM, LLMError
from trustqa.schemas import Chunk, Warning, WarningCode
from trustqa.verification.numeric import Num, extract_numbers

CONFLICT_PROMPT_VERSION = "conflict-v1"

_STOP = {"the", "a", "an", "of", "in", "for", "on", "at", "to", "by", "and", "total", "overall",
         "reported", "value", "its", "their", "company", "s"}


@dataclass(frozen=True)
class Fact:
    chunk_id: str
    entity: str
    metric: str
    period: str
    value: str
    unit: str
    num: Num


@dataclass(frozen=True)
class Conflict:
    facts: tuple[Fact, ...]  # at least two, from different documents


CONFLICT_SYSTEM_PROMPT = """You extract numeric facts from passages. List every numeric fact in the passages that is relevant to the question: what entity it is about, which metric or quantity, which time period (or "" if none), the value exactly as written in the passage, and the unit (or "").

Use the passage ID exactly as given. Copy values character for character; do not convert or round them. Do not include facts that are not in the passages.

Return a JSON object: {"facts": [{"chunk_id": string, "entity": string, "metric": string, "period": string, "value": string, "unit": string}]}"""


def build_conflict_prompt(question: str, chunks: Sequence[Chunk]) -> str:
    passages = "\n\n".join(f"[{c.chunk_id}] ({c.document_name})\n{c.text}" for c in chunks)
    return f"Passages:\n\n{passages}\n\nQuestion: {question}\n\nAnswer in JSON as instructed."


def _key_tokens(s: str) -> frozenset[str]:
    return frozenset(t for t in re.findall(r"[a-z0-9]+", s.lower()) if t not in _STOP)


def _same(a: str, b: str) -> bool:
    ta, tb = _key_tokens(a), _key_tokens(b)
    if ta == tb:
        return True
    if not ta or not tb:
        return False
    return fuzz.token_set_ratio(" ".join(sorted(ta)), " ".join(sorted(tb))) >= 90 and (
        ta <= tb or tb <= ta)


def _norm_unit(f: Fact) -> str:
    u = (f.unit or "").lower().replace("percent", "%").replace("per cent", "%").strip()
    if f.num.percent:
        u = "%"
    for w in f.num.scale_words:
        u = u.replace(w, "").strip()
    return re.sub(r"[^a-z%$₹]", "", u)


def _values_equal(a: Num, b: Num) -> bool:
    if a.scaled == b.scaled:
        return True
    dec = min(a.decimals, b.decimals)
    mult = min(a.mult, b.mult) or 1
    q = Decimal(1).scaleb(-dec)
    return (a.scaled / mult).quantize(q) == (b.scaled / mult).quantize(q)


def parse_facts(raw: str, chunks: dict[str, Chunk]) -> list[Fact]:
    """Validate the extractor's JSON against the retrieved chunks (plain Python)."""
    data = json.loads(raw)
    items = data.get("facts") if isinstance(data, dict) else None
    if not isinstance(items, list):
        raise LLMError("conflict extractor output has no 'facts' list")
    facts: list[Fact] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        cid, value = it.get("chunk_id"), it.get("value")
        if not isinstance(cid, str) or cid not in chunks or not isinstance(value, (str, int, float)):
            continue
        nums = extract_numbers(str(value))
        if len(nums) != 1:
            continue
        n = nums[0]
        chunk_nums = extract_numbers(chunks[cid].text)
        if not any(abs(e.base) == abs(n.base) for e in chunk_nums):
            continue  # value is not in the chunk: the extractor made it up
        facts.append(Fact(chunk_id=cid, entity=str(it.get("entity") or ""),
                          metric=str(it.get("metric") or ""), period=str(it.get("period") or ""),
                          value=str(value), unit=str(it.get("unit") or ""), num=n))
    return facts


def find_conflicts(facts: Sequence[Fact], chunks: dict[str, Chunk], cited_ids: Sequence[str]
                   ) -> list[Conflict]:
    """Group facts by entity/metric/period and return cross-document value disagreements."""
    cited = set(cited_ids)
    conflicts: list[Conflict] = []
    seen: set[frozenset[str]] = set()
    for i, f in enumerate(facts):
        if not _key_tokens(f.metric):
            continue
        group = [f]
        for g in facts[i + 1:]:
            if (_same(f.entity, g.entity) and _same(f.metric, g.metric)
                    and _key_tokens(f.period) == _key_tokens(g.period)
                    and _norm_unit(f) == _norm_unit(g)):
                group.append(g)
        docs = {chunks[x.chunk_id].document_id for x in group}
        if len(docs) < 2 or not any(x.chunk_id in cited for x in group):
            continue
        anchor = next(x for x in group if x.chunk_id in cited)
        other = [x for x in group if chunks[x.chunk_id].document_id != chunks[anchor.chunk_id].document_id
                 and not _values_equal(x.num, anchor.num)]
        if not other:
            continue
        key = frozenset([anchor.chunk_id, *(x.chunk_id for x in other)])
        if key in seen:
            continue
        seen.add(key)
        conflicts.append(Conflict(facts=(anchor, *other)))
    return conflicts


def _describe(f: Fact, chunks: dict[str, Chunk]) -> str:
    c = chunks[f.chunk_id]
    unit = f" {f.unit}" if f.unit and f.unit not in f.value else ""
    return f"{f.value}{unit} in {c.document_name} p.{c.page_number} ({f.chunk_id})"


@dataclass
class ConflictReport:
    conflicts: list[Conflict]
    warnings: list[Warning]
    failed: bool = False  # extraction was needed but could not run: no claim of "no conflict"


def check_conflicts(question: str, passages: Sequence[Chunk], cited_ids: Sequence[str], llm: LLM
                    ) -> ConflictReport:
    """Extract facts with the LLM, compare them in Python, warn per conflict.

    Runs only for multi-document passages. If extraction fails, the report is `failed`
    and carries a CONFLICT warning, so the answer cannot be Verified unchecked.
    """
    chunks = {c.chunk_id: c for c in passages}
    if len({c.document_id for c in passages}) < 2:
        return ConflictReport([], [])
    try:
        raw = llm.complete(build_conflict_prompt(question, passages),
                           system=CONFLICT_SYSTEM_PROMPT, json=True)
        facts = parse_facts(raw, chunks)
    except (LLMError, json.JSONDecodeError) as e:
        return ConflictReport([], [Warning(
            WarningCode.CONFLICT, f"cross-document conflict check could not run ({e})")],
            failed=True)
    conflicts = find_conflicts(facts, chunks, cited_ids)
    warnings = []
    for cf in conflicts:
        a = cf.facts[0]
        label = " ".join(x for x in (a.entity, a.metric, a.period) if x).strip()
        warnings.append(Warning(
            WarningCode.CONFLICT,
            f"sources disagree on {label}: " + " vs ".join(_describe(f, chunks) for f in cf.facts),
            chunk_id=a.chunk_id))
    return ConflictReport(conflicts, warnings)
