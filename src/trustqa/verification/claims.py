"""Phase 3 item 2 - claim decomposition. Plain Python, no LLM call.

A claim is one sentence of the answer, mapped to the chunk IDs cited in it. Sentence level is
deliberate: the support judge (item 3) must find *every* fact of a claim in the evidence, and
the numeric check (item 4) checks *every* number in it, so a sentence that mixes a supported
and an unsupported fact still fails. Finer, LLM-made "atomic" claims would only change how
the failure is reported, at the cost of an extra model call that can itself drop facts.

Mapping rules
- Citation markers belong to the sentence they close: "A [ID1]. B [ID2]." and the
  after-the-period style "A. [ID1] B. [ID2]" both map A->ID1, B->ID2.
- A sentence with no marker of its own (explicit=False) is checked against every accepted
  chunk of the answer: the prompt asks for a citation per sentence, but small models often
  cite once at the end. It still has to pass the support check against that evidence.
- Boilerplate with no factual content ("Based on the passages:") is not material and does
  not affect the status.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from trustqa.generation import CHUNK_ID_RE

_MARKER_RE = re.compile(r"\s*\[([^\[\]]*)\]")
_ABBREV = {
    "rs", "mr", "mrs", "ms", "dr", "prof", "e.g", "i.e", "al", "fig", "figs", "eq", "eqs", "no",
    "nos", "vs", "approx", "p", "pp", "inc", "ltd", "co", "corp", "sec", "vol", "ch", "cf", "etc",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec", "st",
}
# Candidate boundary: sentence punctuation, optional closing quote/bracket, then whitespace.
_BOUNDARY_RE = re.compile(r"[.!?][\"')”’]?(?=\s+)")
_BOILERPLATE_RE = re.compile(
    r"^(?:(?:based on|according to|from|using) (?:the )?(?:provided |given |retrieved )?"
    r"(?:passages?|sources?|documents?|context|evidence)|here is (?:the|my) answer|in summary|"
    r"to summari[sz]e|answer)\s*[:,.]?$", re.IGNORECASE)


@dataclass(frozen=True)
class Claim:
    claim_id: int  # 1-based, in answer order
    text: str  # the sentence without citation markers
    chunk_ids: tuple[str, ...]  # evidence this claim must be checked against
    explicit: bool  # True if the sentence carries its own citation marker
    material: bool  # False for boilerplate without factual content


def split_sentences(answer: str) -> list[str]:
    """Split answer prose into sentences, keeping each sentence's trailing citation markers."""
    text = answer.strip()
    if not text:
        return []
    cuts = []
    for m in _BOUNDARY_RE.finditer(text):
        end = m.end()
        word = re.search(r"([A-Za-z.]+)\.$", text[max(0, m.start() - 12):m.start() + 1])
        if text[m.start()] == "." and word and word.group(1).lower().rstrip(".") in _ABBREV:
            continue  # "Rs. 5", "et al. 2020", "p. 3"
        nxt = text[end:].lstrip()
        if nxt[:1] == "[":  # "A. [ID1] B." -> the marker closes A
            mk = re.match(r"(?:\s*\[[^\[\]]*\])+", text[end:])
            if mk and CHUNK_ID_RE.search(mk.group(0)):
                end += mk.end()
                nxt = text[end:].lstrip()
        if not nxt or not (nxt[0].isupper() or nxt[0].isdigit() or nxt[0] in "\"'(“"):
            continue
        cuts.append(end)
    parts, start = [], 0
    for c in cuts:
        parts.append(text[start:c].strip())
        start = c
    parts.append(text[start:].strip())
    return [p for p in parts if p]


def _strip_markers(sentence: str) -> str:
    out = _MARKER_RE.sub(lambda m: "" if CHUNK_ID_RE.search(m.group(1)) else m.group(0), sentence)
    return re.sub(r"\s+([.,;:!?])", r"\1", out).strip()


def is_material(text: str) -> bool:
    core = text.strip().strip("*-• ").strip()
    if not re.search(r"[A-Za-z0-9]", core):
        return False
    return not _BOILERPLATE_RE.match(core)


def decompose_claims(answer: str, accepted_ids: Sequence[str]) -> list[Claim]:
    """Split `answer` into sentence claims mapped to their cited chunks.

    Only IDs in `accepted_ids` (provenance-checked, item 1) are kept; a sentence whose only
    markers were rejected is treated as uncited.
    """
    accepted = list(dict.fromkeys(accepted_ids))
    ok = set(accepted)
    claims: list[Claim] = []
    for sentence in split_sentences(answer):
        ids = [i for m in _MARKER_RE.finditer(sentence) for i in CHUNK_ID_RE.findall(m.group(1))]
        own = tuple(dict.fromkeys(i for i in ids if i in ok))
        text = _strip_markers(sentence)
        if not text:
            continue
        claims.append(Claim(claim_id=len(claims) + 1, text=text,
                            chunk_ids=own or tuple(accepted), explicit=bool(own),
                            material=is_material(text)))
    return claims
