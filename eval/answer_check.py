"""Answer-level check of Phase 4 retrieval flags: run the full system C pipeline (generation +
verification) on the dev questions under each variant and record status, answer, cited pages
and latency. Retrieval metrics alone do not show whether answers change.

Per question it records, from metadata only:
  answered        status is not Insufficient Evidence
  cites_evidence  some citation is on a labelled evidence page (provenance hit)
  ref_in_answer   every number in reference_answer appears in the answer (numeric questions
                  only; None otherwise) - a rough correctness signal, read the answers too.
Unanswerable questions are correct when the status is Insufficient Evidence.

Usage: python eval/answer_check.py --out eval/results/phase4_answers.json [--variants dense,hybrid]
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

from retrieval_eval import default_questions_path, load_questions  # noqa: E402
from trustqa.cli import default_docs  # noqa: E402
from trustqa.config import load_config  # noqa: E402
from trustqa.llm import LLMError  # noqa: E402
from trustqa.pipeline import answer_question, run_record  # noqa: E402
from trustqa.schemas import Status  # noqa: E402

# every flag set explicitly, so a .env with RETRIEVAL_HYBRID=1 cannot turn "dense" into hybrid
FLAGS_OFF = dict(retrieval_hybrid=False, retrieval_diversify="none", rerank=False,
                 retrieval_decompose=False)
VARIANTS = {
    "dense": {},
    "hybrid": dict(retrieval_hybrid=True),
    "hybrid+decompose": dict(retrieval_hybrid=True, retrieval_decompose=True),
}
_NUM = re.compile(r"\d+(?:[.,]\d+)*")


def numbers(text: str) -> list[str]:
    return [n.replace(",", "") for n in _NUM.findall(text or "")]


def check(q: dict, result) -> dict:
    status = result.status
    row = {"id": q["id"], "category": q["category"], "status": status.value,
           "answer": result.answer,
           "cited_pages": sorted({(c.document_name, c.page_number) for c in result.citations}),
           "warnings": [w.code.value for w in result.warnings],
           "warning_messages": [w.message for w in result.warnings]}
    if not q.get("evidence_groups"):
        row["correct_abstention"] = status is Status.INSUFFICIENT_EVIDENCE
        return row
    labelled = {(d, int(p)) for g in q["evidence_groups"] for d, p in g["pages"]}
    row["answered"] = status is not Status.INSUFFICIENT_EVIDENCE
    row["cites_evidence"] = row["answered"] and any(tuple(cp) in labelled
                                                    for cp in row["cited_pages"])
    ref = numbers(q.get("reference_answer", ""))
    ans = set(numbers(result.answer)) if row["answered"] else set()
    row["ref_in_answer"] = (all(n in ans for n in ref) if ref else None)
    return row


def summary(rows: list[dict]) -> dict:
    ans = [r for r in rows if "answered" in r]
    una = [r for r in rows if "correct_abstention" in r]
    num = [r for r in ans if r["ref_in_answer"] is not None]
    return {
        "answerable": len(ans),
        "answered": sum(r["answered"] for r in ans),
        "verified": sum(r["status"] == Status.VERIFIED.value for r in ans),
        "cites_evidence": sum(r["cites_evidence"] for r in ans),
        "numeric_ref_in_answer": f"{sum(r['ref_in_answer'] for r in num)}/{len(num)}",
        "unanswerable_correctly_abstained": f"{sum(r['correct_abstention'] for r in una)}/{len(una)}",
        "latency_s_mean": round(sum(r["latency_s"] for r in rows) / len(rows), 1),
    }


class _Withheld(logging.Handler):
    """Collects the answers verification withheld (logged as 'withheld answer: ...')."""

    def __init__(self) -> None:
        super().__init__(logging.INFO)
        self.records: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        if record.msg == "withheld answer: %r" and record.args:
            self.records.append(str(record.args[0]))


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--questions", type=Path, default=default_questions_path())
    ap.add_argument("--variants", default="dense,hybrid")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--resume", action="store_true",
                    help="keep finished questions of a partial run in --out and do the rest")
    args = ap.parse_args(argv)
    questions = load_questions(args.questions)
    report = {"created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "questions_file": (args.questions.resolve().relative_to(ROOT).as_posix()
                                 if args.questions.resolve().is_relative_to(ROOT)
                                 else str(args.questions)), "variants": {}}
    withheld = _Withheld()
    tq_logger = logging.getLogger("trustqa")
    tq_logger.addHandler(withheld)
    if tq_logger.level == logging.NOTSET or tq_logger.level > logging.INFO:
        tq_logger.setLevel(logging.INFO)
    if args.resume and args.out.exists():
        report = json.loads(args.out.read_text(encoding="utf-8"))
    for name in args.variants.split(","):
        config = load_config(**{**FLAGS_OFF, **VARIANTS[name]})
        docs = default_docs(config.raw_pdfs_dir)
        done = {r["id"]: r for r in report["variants"].get(name, {}).get("per_question", [])}
        rows = []
        for q in questions:
            if q["id"] in done:
                rows.append(done[q["id"]])
                continue
            t0 = time.perf_counter()
            for attempt in (1, 2):  # one retry: a local LLM timeout under load is not a result
                try:
                    result = answer_question(q["question"], docs, config)
                    break
                except LLMError as e:
                    if attempt == 2:
                        raise
                    print(f"[{name}] {q['id']} LLM error, retrying once: {e}", flush=True)
            row = check(q, result)
            if withheld.records:  # verification logs the answer it withheld; keep it for analysis
                row["withheld_answer"] = withheld.records[-1]
            withheld.records.clear()
            row["latency_s"] = round(time.perf_counter() - t0, 1)
            rows.append(row)
            print(f"[{name}] {q['id']} {row['status']:<22} {row['latency_s']:>5}s  "
                  f"{(row['answer'] or '')[:90]!r}", flush=True)
            # partial save after every question: a killed run keeps what it finished
            report["variants"][name] = {"flags": VARIANTS[name], "partial": True,
                                        "per_question": rows}
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(report, indent=1, ensure_ascii=False, default=list),
                                encoding="utf-8")
        report["variants"][name] = {"flags": VARIANTS[name], "run_config": run_record(config),
                                    "summary": summary(rows), "per_question": rows}
        print(name, report["variants"][name]["summary"], flush=True)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=1, ensure_ascii=False, default=list),
                            encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
