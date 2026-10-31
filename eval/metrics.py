"""Evaluation metrics (Phase 6). Pure functions over plain data, unit-tested on tiny
hand-made examples in tests/test_eval_metrics.py. No LLM calls here (the judge is judge.py).

Definitions (what each number means; reported per system by report.py):

Answer quality
- exact_match / token_f1: SQuAD-style normalisation (lowercase, strip punctuation and
  articles); the max over the reference strings given. Meant for short answers.
- numeric_scores: value / unit / period correctness of a numeric answer against labels.
  Values are compared as numbers (extract_numbers from the verifier: "1,03,450.50 crore",
  "12%", "-2.5" all normalised); unit is a case-insensitive word match; period uses the
  verifier's FY/quarter/year reading. A field is None when the question has no label for it.

Retrieval (page-level, a chunk is relevant if its (document, page) is a labelled page)
- recall_at_k: share of evidence groups with a relevant chunk in the top k (a multi-doc
  question has one group per document needed); all_groups_at_k: 1 if every group is hit.
- precision_at_k: relevant chunks in the top k / k.
- reciprocal_rank: 1 / rank of the first relevant chunk (0 if none in the list).

Provenance (citations only, reported separately from support)
- provenance: hit = some citation is on a labelled page; precision = share of the
  citations that are on labelled pages.

Support / status (does the status the system shows agree with answer correctness?)
- see support_scores: verified_precision (Verified answers that are correct) and
  incorrect_not_verified (incorrect answers that the system did not mark Verified).

Rates over a set of rows (aggregate): unsupported_answer_rate = answers shown to the
user (status not Insufficient Evidence) that are judged incorrect; correct_abstention_rate
(unanswerable -> Insufficient Evidence); over_abstention_rate (answerable -> Insufficient
Evidence); conflict recall on questions labelled expected_conflict, and the false conflict
rate on the others.

OCR: cer / wer with jiwer after whitespace normalisation.
"""

from __future__ import annotations

import re
import statistics
import string
from collections import Counter
from collections.abc import Iterable, Sequence

from trustqa.verification.numeric import extract_numbers, extract_periods

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = str.maketrans("", "", string.punctuation)
_WS = re.compile(r"\s+")


# ---------------------------------------------------------------- answers

def normalize_answer(text: str) -> str:
    text = (text or "").lower().translate(_PUNCT)
    return _WS.sub(" ", _ARTICLES.sub(" ", text)).strip()


def exact_match(prediction: str, references: str | Sequence[str]) -> float:
    refs = [references] if isinstance(references, str) else list(references)
    pred = normalize_answer(prediction)
    return float(any(pred == normalize_answer(r) for r in refs))


def _f1(pred: list[str], ref: list[str]) -> float:
    if not pred or not ref:
        return float(pred == ref)
    common = Counter(pred) & Counter(ref)
    same = sum(common.values())
    if same == 0:
        return 0.0
    p, r = same / len(pred), same / len(ref)
    return 2 * p * r / (p + r)


def token_f1(prediction: str, references: str | Sequence[str]) -> float:
    refs = [references] if isinstance(references, str) else list(references)
    pred = normalize_answer(prediction).split()
    return max(_f1(pred, normalize_answer(r).split()) for r in refs)


def _values(text: str) -> set:
    out = set()
    for n in extract_numbers(text or ""):
        v = -n.scaled if n.negative and n.scaled > 0 else n.scaled
        out.add(v.normalize())
    return out


def numeric_scores(answer: str, value: str | None = None, unit: str | None = None,
                   period: str | None = None, reference: str | None = None) -> dict:
    """{"value", "unit", "period"} -> True/False, or None when not labelled.

    With no explicit `value`, every number in `reference` must appear in the answer
    (None if the reference has no numbers)."""
    got = _values(answer)
    if value is not None:
        want = _values(value)
        value_ok = bool(want) and want <= got
    elif reference is not None and _values(reference):
        value_ok = _values(reference) <= got
    else:
        value_ok = None
    unit_ok = None
    if unit:
        unit_ok = re.search(r"(?<![a-z])" + re.escape(unit.lower()) + r"(?![a-z])",
                            (answer or "").lower()) is not None
    period_ok = None
    if period:
        want_p = extract_periods(period) or {period.lower()}
        period_ok = bool(want_p & extract_periods(answer or ""))
    return {"value": value_ok, "unit": unit_ok, "period": period_ok}


# ---------------------------------------------------------------- retrieval

Page = tuple[str, int]


def _groups(evidence_groups: Sequence[dict]) -> list[set[Page]]:
    return [{(d, int(p)) for d, p in g["pages"]} for g in evidence_groups]


def retrieval_scores(retrieved: Sequence[Page], evidence_groups: Sequence[dict],
                     ks: Iterable[int] = (1, 3, 5, 10)) -> dict:
    """Page-level retrieval metrics for one answerable question. `retrieved` is the ranked
    list of (document_name, page_number) of the retrieved chunks."""
    groups = _groups(evidence_groups)
    if not groups:
        raise ValueError("retrieval metrics need at least one evidence group")
    relevant = set().union(*groups)
    ranked = [(d, int(p)) for d, p in retrieved]
    out: dict = {}
    for k in ks:
        top = ranked[:k]
        hit = [any(r in g for r in top) for g in groups]
        out[f"recall@{k}"] = sum(hit) / len(groups)
        out[f"all_groups@{k}"] = float(all(hit))
        out[f"precision@{k}"] = sum(r in relevant for r in top) / k
    first = next((i for i, r in enumerate(ranked, 1) if r in relevant), None)
    out["rr"] = 1.0 / first if first else 0.0
    return out


# ---------------------------------------------------------------- provenance / support

def provenance(cited: Sequence[Page], evidence_groups: Sequence[dict]) -> dict:
    """Citation-level provenance against labelled pages. None values when nothing is cited."""
    relevant = set().union(*_groups(evidence_groups)) if evidence_groups else set()
    pages = [(d, int(p)) for d, p in cited]
    if not pages:
        return {"provenance_hit": None, "citation_precision": None}
    on = [p in relevant for p in pages]
    return {"provenance_hit": float(any(on)), "citation_precision": sum(on) / len(on)}


VERIFIED, PARTIAL, INSUFFICIENT = "Verified", "Partially Supported", "Insufficient Evidence"


def support_scores(rows: Sequence[dict]) -> dict:
    """Agreement between the status shown and answer correctness, over answered rows with a
    correctness label (row["correct"] in {True, False})."""
    labelled = [r for r in rows if r.get("status") != INSUFFICIENT and r.get("correct") is not None]
    ver = [r for r in labelled if r["status"] == VERIFIED]
    wrong = [r for r in labelled if not r["correct"]]
    return {
        "verified_precision": _mean(float(r["correct"]) for r in ver),
        "verified_n": len(ver),
        "incorrect_not_verified": _mean(float(r["status"] != VERIFIED) for r in wrong),
        "incorrect_n": len(wrong),
    }


# ---------------------------------------------------------------- OCR

def _norm_ws(text: str) -> str:
    return _WS.sub(" ", text or "").strip()


def cer_wer(reference: str, hypothesis: str) -> dict:
    import jiwer

    ref, hyp = _norm_ws(reference), _norm_ws(hypothesis)
    if not ref:
        raise ValueError("empty reference text")
    return {"cer": jiwer.cer(ref, hyp), "wer": jiwer.wer(ref, hyp)}


# ---------------------------------------------------------------- aggregation

def _mean(values: Iterable[float | None]) -> float | None:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def aggregate(rows: Sequence[dict], ks: Iterable[int] = (1, 3, 5, 10)) -> dict:
    """System-level metrics over per-question rows produced by run_all.score_row."""
    ans = [r for r in rows if r["answerable"]]
    una = [r for r in rows if not r["answerable"]]
    shown = [r for r in rows if r["status"] != INSUFFICIENT]
    out: dict = {"n": len(rows), "n_answerable": len(ans), "n_unanswerable": len(una)}
    for k in ks:
        for m in ("recall", "all_groups", "precision"):
            out[f"{m}@{k}"] = _mean(r["retrieval"].get(f"{m}@{k}") for r in ans)
    out["mrr"] = _mean(r["retrieval"].get("rr") for r in ans)
    out["exact_match"] = _mean(r.get("exact_match") for r in ans if r["short_answer"])
    out["token_f1"] = _mean(r.get("token_f1") for r in ans if r["short_answer"])
    out["accuracy"] = _mean(_f(r.get("correct")) for r in ans)  # unanswered count as wrong
    out["answered_rate"] = _mean(float(r["status"] != INSUFFICIENT) for r in ans)
    out["verified_rate"] = _mean(float(r["status"] == VERIFIED) for r in ans)
    for key in ("value", "unit", "period"):
        out[f"numeric_{key}"] = _mean(_f(r["numeric"].get(key)) for r in ans if r.get("numeric"))
        out[f"numeric_{key}_n"] = sum(r.get("numeric", {}).get(key) is not None for r in ans)
    out["provenance_hit"] = _mean(r["provenance"]["provenance_hit"] for r in ans)
    out["citation_precision"] = _mean(r["provenance"]["citation_precision"] for r in ans)
    out.update(support_scores(ans))
    out["unsupported_answer_rate"] = _mean(float(r["correct"] is False) for r in shown
                                           if r["answerable"] and r.get("correct") is not None)
    out["unanswerable_answered"] = sum(1 for r in shown if not r["answerable"])
    out["correct_abstention_rate"] = _mean(float(r["status"] == INSUFFICIENT) for r in una)
    out["over_abstention_rate"] = _mean(float(r["status"] == INSUFFICIENT) for r in ans)
    conf = [r for r in rows if r.get("expected_conflict")]
    rest = [r for r in rows if not r.get("expected_conflict")]
    out["conflict_recall"] = _mean(float(r["conflict_flagged"]) for r in conf)
    out["conflict_n"] = len(conf)
    out["false_conflict_rate"] = _mean(float(r["conflict_flagged"]) for r in rest)
    # rows whose LLM calls were served from the cache have no real latency (latency_valid False)
    lat = [r["latency_s"] for r in rows
           if r.get("latency_s") is not None and r.get("latency_valid", True)]
    out["latency_n"] = len(lat)
    out["latency_mean_s"] = statistics.fmean(lat) if lat else None
    out["latency_p50_s"] = statistics.median(lat) if lat else None
    out["latency_p90_s"] = (statistics.quantiles(lat, n=10)[-1] if len(lat) >= 2
                            else (lat[0] if lat else None))
    return out


def _f(v: bool | None) -> float | None:
    return None if v is None else float(v)
