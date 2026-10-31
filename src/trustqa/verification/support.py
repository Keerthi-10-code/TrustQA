"""Phase 3 item 3 - evidence-support check (FR-10).

One LLM judge call per answer returns strict JSON with a verdict per claim:
supported / not_supported / contradicted, plus the passage it relied on and a verbatim
`supporting_quote`. The judge is not trusted on its own:
- the quote must occur in the named chunk, and that chunk must be one the claim cites;
- a "supported" quote must mention what the claim adds beyond the question (a content word
  or a digit sequence of the claim), so text that merely exists in the chunk - such as
  instructions planted in a document - cannot serve as support;
  matching is normalised (case, whitespace, quote/dash styles) and fuzzy only for quotes of
  at least 20 characters (>= quote_min_similarity), and every number in the quote must occur
  exactly in the chunk - a fabricated or altered quote downgrades "supported" to
  "not_supported";
- a missing, malformed or unknown verdict counts as "not_supported";
- if the judge call itself fails (provider error or non-JSON), claims are "unverified":
  the answer is not withheld, but it cannot be Verified.

NLI cross-encoder first pass: not used. It would add a model download and per-pair
inference, and the plan says to adopt it only if testing on our questions shows the LLM judge
needs a cheaper filter; the quote rule already guards against the judge's main failure mode.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

from rapidfuzz import fuzz

from trustqa.llm import LLM, LLMError
from trustqa.schemas import Chunk, Warning, WarningCode
from trustqa.verification.claims import Claim

SUPPORT_PROMPT_VERSION = "support-v1"
MIN_FUZZY_QUOTE_LEN = 20


class Verdict(str, Enum):
    SUPPORTED = "supported"
    NOT_SUPPORTED = "not_supported"
    CONTRADICTED = "contradicted"
    UNVERIFIED = "unverified"  # the judge could not be run


@dataclass(frozen=True)
class SupportResult:
    claim_id: int
    verdict: Verdict
    chunk_id: str | None = None
    quote: str = ""  # validated verbatim span of the chunk ("" unless validated)
    reason: str = ""


SUPPORT_SYSTEM_PROMPT = """You are a strict fact checker. For each numbered claim, decide whether the passages listed for that claim state it.

Verdicts:
- "supported": every fact in the claim, including every number, unit and time period, is stated in one of the listed passages.
- "contradicted": a listed passage states something incompatible with the claim.
- "not_supported": anything else, including claims that are only partly stated or that need outside knowledge.

Judge every claim as part of the answer to the question. A claim marked [short answer] is a bare value or phrase meaning "the answer to the question is ...": it is supported only if a passage states that value for exactly what the question asks - the same entity, metric, table column and time period. A matching number that belongs to a different row, column, dataset or metric is "not_supported".

For "supported" and "contradicted", set "chunk_id" to the passage ID and copy into "supporting_quote" the exact words from that passage that support or contradict the claim: one contiguous span, copied character for character, at most 300 characters. Never paraphrase or shorten numbers in the quote. For "not_supported", use "" for both.

Return a JSON object: {"verdicts": [{"claim_id": integer, "verdict": "supported" | "not_supported" | "contradicted", "chunk_id": string, "supporting_quote": string}]}"""


def is_fragment(text: str) -> bool:
    """A bare value or phrase ("0.347.", "Critique tokens") that only means something as the
    answer to the question, so the judge must check it against the question."""
    return len(re.findall(r"[A-Za-z0-9][\w.,%+-]*", text)) < 4


def build_support_prompt(question: str, claims: Sequence[Claim], chunks: dict[str, Chunk]) -> str:
    used = list(dict.fromkeys(cid for c in claims for cid in c.chunk_ids if cid in chunks))
    passages = "\n\n".join(f"[{cid}]\n{chunks[cid].text}" for cid in used)
    lines = "\n".join(
        f"Claim {c.claim_id} (check against: {', '.join(c.chunk_ids)})"
        f"{' [short answer]' if is_fragment(c.text) else ''}: {c.text}" for c in claims)
    return (f"Passages:\n\n{passages}\n\nQuestion that was answered: {question}\n\n"
            f"Claims:\n{lines}\n\nAnswer in JSON as instructed.")


_TRANS = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"',
                        "–": "-", "—": "-", "−": "-", " ": " "})


def normalise(text: str) -> str:
    t = unicodedata.normalize("NFKC", text).translate(_TRANS).lower()
    return re.sub(r"\s+", " ", t).strip()


def _number_tokens(text: str) -> list[str]:
    return re.findall(r"\d+(?:[.,]\d+)*", text)


def quote_in_chunk(quote: str, chunk_text: str, min_similarity: float = 90.0) -> bool:
    """True if `quote` is (near-)verbatim text of the chunk. Deterministic, no LLM."""
    q, t = normalise(quote), normalise(chunk_text)
    if len(q) < 3:
        return False
    if q in t:
        return True
    if len(q) < MIN_FUZZY_QUOTE_LEN:
        return False
    if any(tok not in t for tok in _number_tokens(q)):
        return False  # an altered number never passes as a fuzzy match
    return fuzz.partial_ratio(q, t) >= min_similarity


_STOP = {
    "the", "and", "for", "are", "was", "were", "with", "that", "this", "from", "has", "have", "had",
    "its", "their", "which", "what", "who", "whom", "how", "does", "did", "use", "uses", "used",
    "into", "than", "then", "also", "been", "being", "there", "these", "those", "they", "them",
    "about", "over", "under", "between", "most", "more", "much", "many", "such", "only", "very",
}


def _stems(text: str) -> set[str]:
    """Content-word stems (first 5 letters), so 'audits'/'audited' and
    'initialised'/'initialized' match."""
    return {t[:5] for t in re.findall(r"[a-z]{3,}", normalise(text)) if t not in _STOP}


def quote_relevant(claim: str, quote: str, question: str = "") -> bool:
    """A quote supports a claim only if it mentions what the claim adds: at least one content
    word of the claim that is not already in the question, or one of the claim's digit
    sequences. Blocks support by a quote that merely exists in the chunk - e.g. planted text
    such as "every statement about Helios is verified" offered for "Helios is audited by
    Deloitte". Claims that add no new word or digit are not constrained."""
    new = _stems(claim) - _stems(question)
    digits = set(re.findall(r"\d+", claim))
    if not new and not digits:
        return True
    q = normalise(quote)
    return bool(new & _stems(q)) or any(d in set(re.findall(r"\d+", q)) for d in digits)


def _parse_verdicts(raw: str) -> dict[int, dict]:
    data = json.loads(raw)
    items = data.get("verdicts") if isinstance(data, dict) else None
    if not isinstance(items, list):
        raise LLMError("support judge output has no 'verdicts' list")
    out: dict[int, dict] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            cid = int(item.get("claim_id"))
        except (TypeError, ValueError):
            continue
        out.setdefault(cid, item)
    return out


def validate_verdict(claim: Claim, item: dict | None, chunks: dict[str, Chunk],
                     min_similarity: float = 90.0, question: str = "") -> SupportResult:
    """Turn one raw judge verdict into a trusted SupportResult (plain Python)."""
    if item is None:
        return SupportResult(claim.claim_id, Verdict.NOT_SUPPORTED, reason="judge gave no verdict")
    raw_verdict = str(item.get("verdict", "")).strip().lower().replace(" ", "_").replace("-", "_")
    try:
        verdict = Verdict(raw_verdict)
    except ValueError:
        return SupportResult(claim.claim_id, Verdict.NOT_SUPPORTED,
                             reason=f"unknown verdict {raw_verdict!r}")
    if verdict in (Verdict.NOT_SUPPORTED, Verdict.UNVERIFIED):
        return SupportResult(claim.claim_id, Verdict.NOT_SUPPORTED,
                             reason="judge found no support in the cited passages")
    quote = item.get("supporting_quote")
    quote = quote.strip().strip('"').strip() if isinstance(quote, str) else ""
    named = item.get("chunk_id") if isinstance(item.get("chunk_id"), str) else ""
    allowed = [c for c in claim.chunk_ids if c in chunks]
    order = ([named] if named in allowed else []) + [c for c in allowed if c != named]
    for cid in order:  # the judge may name the wrong passage; the quote decides
        if quote and quote_in_chunk(quote, chunks[cid].text, min_similarity):
            if verdict is Verdict.SUPPORTED and not quote_relevant(claim.text, quote, question):
                return SupportResult(claim.claim_id, Verdict.NOT_SUPPORTED, reason=(
                    "judge said supported, but its quote does not mention what the claim states"))
            return SupportResult(claim.claim_id, verdict, chunk_id=cid, quote=quote)
    why = "no supporting quote given" if not quote else (
        "supporting quote not found in the cited passage(s)")
    return SupportResult(claim.claim_id, Verdict.NOT_SUPPORTED, reason=f"judge said {verdict.value}, "
                         f"but {why}")


def check_support(question: str, claims: Sequence[Claim], chunks: dict[str, Chunk], llm: LLM,
                  min_similarity: float = 90.0) -> tuple[dict[int, SupportResult], list[Warning]]:
    """Judge every claim against its cited chunks with one LLM call."""
    results: dict[int, SupportResult] = {}
    warnings: list[Warning] = []
    if not claims:
        return results, warnings
    try:
        raw = llm.complete(build_support_prompt(question, claims, chunks),
                           system=SUPPORT_SYSTEM_PROMPT, json=True)
        verdicts = _parse_verdicts(raw)
    except (LLMError, json.JSONDecodeError) as e:
        for c in claims:
            results[c.claim_id] = SupportResult(c.claim_id, Verdict.UNVERIFIED, reason=str(e))
        if any(c.material for c in claims):
            warnings.append(Warning(WarningCode.UNSUPPORTED_CLAIM,
                                    f"support check could not run ({e}); claims are unverified"))
        return results, warnings

    for c in claims:
        res = validate_verdict(c, verdicts.get(c.claim_id), chunks, min_similarity, question)
        results[c.claim_id] = res
        if not c.material or res.verdict is Verdict.SUPPORTED:
            continue
        if res.verdict is Verdict.CONTRADICTED:
            warnings.append(Warning(
                WarningCode.UNSUPPORTED_CLAIM,
                f"claim {c.claim_id} is contradicted by {res.chunk_id}: \"{c.text}\" vs "
                f"\"{res.quote}\"", chunk_id=res.chunk_id))
        else:
            warnings.append(Warning(
                WarningCode.UNSUPPORTED_CLAIM,
                f"claim {c.claim_id} is not supported by its cited passage(s) "
                f"{', '.join(c.chunk_ids)} ({res.reason}): \"{c.text}\"",
                chunk_id=c.chunk_ids[0] if c.chunk_ids else None))
    return results, warnings
