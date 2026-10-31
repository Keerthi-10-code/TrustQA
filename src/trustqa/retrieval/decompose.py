"""Phase 4 - query decomposition for multi-document questions (failures.md F12).

A comparison question ("Compare the GPUs used to train the Transformer and Self-RAG") is one
query vector / one BM25 query, dominated by whichever document's wording matches better, so
the other document's evidence may not be retrieved at all. Reordering a candidate pool (quota,
MMR) cannot fix that. Here the LLM proposes one self-contained sub-query per entity; each is
retrieved separately and the lists are fused with the original question's list (RRF).

The LLM output is a proposal checked in Python: anything malformed falls back to the original
question alone (= retrieval without decomposition), and an LLM failure never fails retrieval.
"""

from __future__ import annotations

import json
import logging

from trustqa.llm import LLM, LLMError

logger = logging.getLogger("trustqa")

DECOMPOSE_VERSION = "decompose-v1"
MAX_SUBQUERY_CHARS = 300

SYSTEM_PROMPT = (
    "You prepare search queries for a document search engine. "
    "Reply with JSON only: {\"queries\": [\"...\"]}."
)

PROMPT_TEMPLATE = """Question: {question}

If the question asks about two or more different things (papers, models, systems, entities) \
that may be described in different documents, write one self-contained search query per \
thing. Each query must name its thing and repeat what is asked about it. Write at most \
{max_n} queries.
If the question is about one thing, return exactly one query: the question itself.
Do not answer the question. JSON: {{"queries": ["..."]}}"""


def parse_subqueries(raw: str, question: str, max_n: int) -> list[str]:
    """Validated sub-queries, or [] if the output is unusable or adds nothing."""
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return []
    queries = data.get("queries") if isinstance(data, dict) else None
    if not isinstance(queries, list):
        return []
    out: list[str] = []
    seen = {question.strip().lower()}
    for q in queries:
        if not isinstance(q, str):
            return []
        q = " ".join(q.split())
        if not q or len(q) > MAX_SUBQUERY_CHARS:
            return []
        if q.lower() not in seen:
            seen.add(q.lower())
            out.append(q)
    if len(queries) < 2 or len(out) > max_n:
        return []  # a single query (or the question echoed) means "no decomposition"
    return out


def decompose(question: str, llm: LLM, max_n: int = 3) -> list[str]:
    """Sub-queries for `question` (empty list = do not decompose)."""
    try:
        raw = llm.complete(PROMPT_TEMPLATE.format(question=question, max_n=max_n),
                           system=SYSTEM_PROMPT, json=True)
    except LLMError as e:
        logger.warning("query decomposition failed (%s); retrieving the question as is", e)
        return []
    subs = parse_subqueries(raw, question, max_n)
    if subs:
        logger.info("decomposed %r into %s", question, subs)
    return subs
