"""Which documents a question is about - deterministic, from document text only.

A question often names its subject: "What GPUs were used to train the Self-RAG models?".
A name *identifies* documents when it occurs in the title region (page-1 text before
"Abstract", else its first 400 characters) or the file name of some but not all documents:
"Self-RAG" identifies the Self-RAG paper, "BEIR" the BEIR paper, "RAG" the RAG paper
(rag_lewis2020.pdf). Abstracts are deliberately excluded: they mention datasets, metrics
and hardware ("Natural Questions", "BLEU", "GPUs") that are not what a document is about.

Used by
- verification (config.verify_scope, failures.md F13): an answer whose citations all come
  from documents the question's identifying names do not point to is withheld - e.g. the
  RAG paper's "8, 32GB NVIDIA V100 GPUs" offered for the Self-RAG question; a mix of in- and
  out-of-scope citations, or a name that no document mentions at all, caps the status at
  Partially Supported with a warning;
- retrieval (config.retrieval_entity_scope, F12): each identifying name gets a share of the
  top-k from its documents, so a two-paper question retrieves evidence from both.

Names are question tokens that look like proper names: capitalised (other than the first
word), containing an inner capital ("SELF-RAG", "nDCG@10"), or mixing letters and digits
("BM25", "GPT-4"). Matching is case-, hyphen- and space-insensitive ("Self-RAG" = "SELF-RAG"
= "Self RAG") and token-aligned, so "RAG" does not match inside "Self-RAG".
"""

from __future__ import annotations

import re
import unicodedata
import weakref
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from trustqa.schemas import Chunk

SCOPE_VERSION = "scope-v1"
TITLE_FALLBACK_CHARS = 400  # title region when page 1 has no "Abstract" heading
_MAX_JOIN = 3  # a name may be split over up to 3 tokens in PDF text ("S ELF -R AG")
_STRIP = ".,;:?!()[]{}\"'“”‘’*"
_NOT_NAMES = {"i", "table", "figure", "fig", "section", "appendix", "eq", "equation", "page"}


def squash(text: str) -> str:
    """Lower-case ASCII letters and digits only: 'Self-RAG' -> 'selfrag', 'nDCG@10' ->
    'ndcg10', 'Touché-2020' -> 'touche2020'."""
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]", "", text.lower())


def title_region(page1_text: str) -> str:
    m = re.search(r"\babstract\b", page1_text, flags=re.I)
    return page1_text[:m.start()] if m and m.start() > 0 else page1_text[:TITLE_FALLBACK_CHARS]


_SEPARATORS = re.compile(r"[\s=|:;,/()\[\]{}]+")  # table chunks: "Segment=Retail | FY2024=..."


def _keys(text: str) -> set[str]:
    toks = [squash(t) for t in _SEPARATORS.split(text)]
    toks = [t for t in toks if t]
    keys = set(toks)
    for n in range(2, _MAX_JOIN + 1):
        keys.update("".join(toks[i:i + n]) for i in range(len(toks) - n + 1))
    return keys


def question_names(question: str) -> list[str]:
    """Proper-name-like tokens of a question, in order, without duplicates."""
    out: list[str] = []
    for i, raw in enumerate(question.split()):
        tok = raw.strip(_STRIP)
        tok = re.sub(r"['’]s$", "", tok)
        key = squash(tok)
        if len(key) < 2 or key.isdigit() or key in _NOT_NAMES:
            continue
        inner_cap = any(ch.isupper() for ch in tok[1:])
        mixed = any(ch.isdigit() for ch in tok) and any(ch.isalpha() for ch in tok)
        if (i > 0 and tok[:1].isupper()) or inner_cap or mixed:
            if tok not in out:
                out.append(tok)
    return out


@dataclass
class DocumentScope:
    profiles: dict[str, set[str]]  # document_id -> keys of its first-page text + file name
    names: dict[str, str]  # document_id -> document_name
    corpus: set[str]  # keys of every chunk's text

    @classmethod
    def from_chunks(cls, chunks: Sequence[Chunk]) -> DocumentScope:
        first: dict[str, list[str]] = {}
        names: dict[str, str] = {}
        corpus: set[str] = set()
        for c in chunks:
            names.setdefault(c.document_id, c.document_name)
            corpus |= _keys(c.text)
            if c.section:
                corpus |= _keys(c.section)
            if c.page_number == 1 and c.content_type.value == "text":
                first.setdefault(c.document_id, []).append(c.text)
        profiles = {}
        for doc, name in names.items():
            text = title_region(" ".join(first.get(doc, [])))
            name_keys = _keys(re.sub(r"[_\-.]", " ", name))
            profiles[doc] = _keys(text) | name_keys
            corpus |= name_keys  # a file named helios_annual_2025.pdf mentions Helios
        return cls(profiles, names, corpus)

    def identifying(self, question: str) -> dict[str, set[str]]:
        """Question name -> the documents it identifies (some, not all, documents)."""
        n = len(self.profiles)
        out = {}
        for name in question_names(question):
            key = squash(name)
            docs = {d for d, keys in self.profiles.items() if key in keys}
            if 0 < len(docs) < n:
                out[name] = docs
        return out

    def allowed_documents(self, question: str) -> set[str] | None:
        """Documents the question points to, or None when it names no identifying name."""
        ident = self.identifying(question)
        return set().union(*ident.values()) if ident else None

    def missing(self, question: str) -> list[str]:
        """Capitalised question names that occur nowhere in the collection (lowercase symbols
        such as "beta2" are skipped: OCR often garbles them)."""
        return [n for n in question_names(question)
                if n[:1].isupper() and squash(n) not in self.corpus]


_cache: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def scope_for(index) -> DocumentScope:
    """DocumentScope of a SearchIndex (built once per index object)."""
    s = _cache.get(index)
    if s is None:
        s = DocumentScope.from_chunks(index.chunks)
        _cache[index] = s
    return s


def scope_from(chunks: Iterable[Chunk]) -> DocumentScope:
    return DocumentScope.from_chunks(list(chunks))
