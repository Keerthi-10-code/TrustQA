"""Question-set schema and validation (Phase 6, step 1).

One JSON object per line in eval/questions.jsonl:

  id                 unique string
  question           non-empty string
  category           direct | multi_doc | table | unanswerable | conflicting | scanned
  reference_answer   non-empty string (all categories except unanswerable)
  expected_abstention  true for unanswerable questions (and only for them)
  evidence_groups    list of {"pages": [[document_name, page], ...], "pattern": regex?}:
                     one group per piece of evidence needed (a multi-doc question has one
                     group per document); pages are 1-indexed; required unless unanswerable
  numeric            optional {"value": "0.656", "unit": "%", "period": "FY2024"}
  expected_conflict  optional bool; required true for category conflicting
  answer_type        optional short | long (default: short if the reference has <= 6 words)
  source, notes      optional free text

Usage: python -m eval.questions [eval/questions.jsonl]  (prints errors, warnings and the
category counts against the target 10/10/10/10/5/5).
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUESTIONS = ROOT / "eval" / "questions.jsonl"
DEV_QUESTIONS = ROOT / "eval" / "dev_questions.jsonl"

TARGETS = {"direct": 10, "multi_doc": 10, "table": 10, "unanswerable": 10,
           "conflicting": 5, "scanned": 5}
KNOWN_KEYS = {"id", "question", "category", "reference_answer", "expected_abstention",
              "evidence_groups", "numeric", "expected_conflict", "answer_type", "source", "notes"}


class QuestionSetError(ValueError):
    pass


def load_questions(path: str | Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise QuestionSetError(f"{path}:{n}: invalid JSON ({e})") from e
    return rows


def validate(questions: list[dict], page_counts: dict[str, int] | None = None
             ) -> tuple[list[str], list[str]]:
    """Return (errors, warnings). `page_counts` (document_name -> pages) enables checks that
    labelled documents exist and pages are in range."""
    errors: list[str] = []
    warnings: list[str] = []
    seen: set[str] = set()
    for i, q in enumerate(questions, 1):
        qid = q.get("id") or f"line {i}"
        err = lambda m: errors.append(f"{qid}: {m}")  # noqa: E731
        if not isinstance(q.get("id"), str) or not q["id"].strip():
            err("missing id")
        elif q["id"] in seen:
            err("duplicate id")
        seen.add(q.get("id"))
        if not isinstance(q.get("question"), str) or not q["question"].strip():
            err("missing question")
        cat = q.get("category")
        if cat not in TARGETS:
            err(f"unknown category {cat!r} (use one of {sorted(TARGETS)})")
        unknown = set(q) - KNOWN_KEYS
        if unknown:
            warnings.append(f"{qid}: unknown field(s) {sorted(unknown)}")
        groups = q.get("evidence_groups") or []
        if cat == "unanswerable":
            if q.get("expected_abstention") is not True:
                err("unanswerable questions need expected_abstention: true")
            if groups:
                err("unanswerable questions must not have evidence_groups")
            continue
        if q.get("expected_abstention"):
            err("expected_abstention is only for category unanswerable")
        if not isinstance(q.get("reference_answer"), str) or not q["reference_answer"].strip():
            err("missing reference_answer")
        if not groups:
            err("missing evidence_groups (relevant document + page)")
        for gi, g in enumerate(groups, 1):
            pages = g.get("pages") if isinstance(g, dict) else None
            if not pages:
                err(f"evidence group {gi} has no pages")
                continue
            for p in pages:
                if (not isinstance(p, list) or len(p) != 2 or not isinstance(p[0], str)
                        or not isinstance(p[1], int) or isinstance(p[1], bool) or p[1] < 1):
                    err(f"evidence group {gi}: page entry {p!r} must be [document_name, page>=1]")
                elif page_counts is not None:
                    if p[0] not in page_counts:
                        err(f"evidence group {gi}: document {p[0]!r} is not in the document set")
                    elif p[1] > page_counts[p[0]]:
                        err(f"evidence group {gi}: {p[0]} has {page_counts[p[0]]} pages, "
                            f"labelled page {p[1]}")
            if g.get("pattern"):
                try:
                    re.compile(g["pattern"])
                except re.error as e:
                    err(f"evidence group {gi}: invalid pattern ({e})")
        if cat == "multi_doc":
            docs = {p[0] for g in groups for p in (g.get("pages") or []) if isinstance(p, list)}
            if len(groups) < 2 or len(docs) < 2:
                warnings.append(f"{qid}: multi_doc question labels fewer than 2 documents")
        if cat == "conflicting" and q.get("expected_conflict") is not True:
            err("conflicting questions need expected_conflict: true")
        num = q.get("numeric")
        if num is not None and (not isinstance(num, dict) or not str(num.get("value", "")).strip()):
            err("numeric must be an object with a non-empty value")
        if q.get("answer_type") not in (None, "short", "long"):
            err("answer_type must be short or long")
    return errors, warnings


def category_counts(questions: list[dict]) -> dict[str, tuple[int, int]]:
    c = Counter(q.get("category") for q in questions)
    return {cat: (c.get(cat, 0), target) for cat, target in TARGETS.items()}


def is_short_answer(q: dict) -> bool:
    if q.get("answer_type"):
        return q["answer_type"] == "short"
    return len((q.get("reference_answer") or "").split()) <= 6


def document_page_counts(doc_paths) -> dict[str, int]:
    import pymupdf

    out = {}
    for p in doc_paths:
        with pymupdf.open(p) as d:
            out[Path(p).name] = len(d)
    return out


def format_counts(questions: list[dict]) -> str:
    lines = [f"{'category':<14}{'have':>6}{'target':>8}"]
    for cat, (have, target) in category_counts(questions).items():
        flag = "" if have >= target else f"  (-{target - have})"
        lines.append(f"{cat:<14}{have:>6}{target:>8}{flag}")
    lines.append(f"{'total':<14}{len(questions):>6}{sum(TARGETS.values()):>8}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    from trustqa.cli import default_docs
    from trustqa.config import load_config

    args = sys.argv[1:] if argv is None else argv
    path = Path(args[0]) if args else DEFAULT_QUESTIONS
    if not path.is_file():
        print(f"error: {path} not found. Create it (schema in eval/questions.py), or validate "
              f"the dev set: python -m eval.questions {DEV_QUESTIONS.relative_to(ROOT)}")
        return 2
    qs = load_questions(path)
    errors, warnings = validate(qs, document_page_counts(default_docs(load_config().raw_pdfs_dir)))
    print(format_counts(qs))
    for w in warnings:
        print("warning:", w)
    for e in errors:
        print("error:", e)
    print(f"{len(qs)} questions, {len(errors)} error(s), {len(warnings)} warning(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
