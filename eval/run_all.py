"""One command for the whole evaluation (Phase 6):  python -m eval.run_all

Steps: validate the question set -> run every system (A, B, C and the C ablations) on every
question with LLM outputs cached by (prompt, model) hash -> judge answer correctness ->
OCR CER/WER against ground truth -> tables (JSON/CSV/Markdown) and plots -> error-analysis
dump. A killed run resumes where it stopped (results are saved after every question), and
a re-run with a warm cache makes no model calls, so tables and plots are reproducible.

  python -m eval.run_all                                   # eval/questions.jsonl, everything
  python -m eval.run_all --questions eval/dev_questions.jsonl
  python -m eval.run_all --systems C --repeats 3           # variance runs (fresh LLM calls)
  python -m eval.run_all --report-only                     # rebuild tables/plots from runs/

Outputs: eval/results/<question-file-stem>/ (runs/, summary.*, by_category.csv,
per_question.csv, ocr_quality.json, error_analysis.md, spot_check.csv) and
eval/plots/<question-file-stem>/*.png.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from eval import metrics
from eval.judge import JUDGE_VERSION, judge_answer
from eval.questions import (
    DEFAULT_QUESTIONS,
    ROOT,
    document_page_counts,
    format_counts,
    is_short_answer,
    load_questions,
    validate,
)
from eval.systems import ABLATIONS, MAIN_SYSTEMS, SYSTEMS, caching_llm, config_for
from trustqa import llm as llm_mod
from trustqa.cli import default_docs
from trustqa.config import Config
from trustqa.llm import LLMError, check_available
from trustqa.pipeline import answer_from_passages, get_or_build_index, run_record
from trustqa.retrieval.search import retrieve_with_config
from trustqa.schemas import Result, Status, WarningCode

RETRIEVAL_DEPTH = 10
KS = (1, 3, 5, 10)
CACHE_PATH = ROOT / "eval" / "cache" / "llm_cache.sqlite"

logger = logging.getLogger("trustqa")


class _Withheld(logging.Handler):
    """Collects answers that verification withheld (logged as 'withheld answer: ...')."""

    def __init__(self) -> None:
        super().__init__(logging.INFO)
        self.records: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        if record.msg == "withheld answer: %r" and record.args:
            self.records.append(str(record.args[0]))


# ---------------------------------------------------------------- scoring one question

def score_row(q: dict, result: Result, retrieved: list[tuple[str, int]], latency_s: float,
              verdict: str | None = None, judge_reason: str = "", use_judge: bool = True) -> dict:
    """Per-question record: raw outputs (from metadata only) plus every per-question metric."""
    status = result.status.value
    answered = result.status is not Status.INSUFFICIENT_EVIDENCE
    answerable = q["category"] != "unanswerable"
    cited = sorted({(c.document_name, c.page_number) for c in result.citations})
    row = {
        "id": q["id"], "category": q["category"], "question": q["question"],
        "reference_answer": q.get("reference_answer"), "answerable": answerable,
        "short_answer": answerable and is_short_answer(q),
        "expected_conflict": bool(q.get("expected_conflict")),
        "status": status, "answer": result.answer if answered else "",
        "cited_pages": [list(p) for p in cited],
        "citations": [c.to_dict() for c in result.citations],
        "warnings": [w.to_dict() for w in result.warnings],
        "conflict_flagged": any(w.code is WarningCode.CONFLICT for w in result.warnings),
        "retrieved_pages": [list(p) for p in retrieved],
        "latency_s": round(latency_s, 2),
        "judge_verdict": verdict, "judge_reason": judge_reason,
    }
    if not answerable:
        row.update(retrieval={}, provenance={"provenance_hit": None, "citation_precision": None},
                   numeric={}, correct=not answered, correct_source="abstention")
        return row
    groups = q["evidence_groups"]
    row["labelled_pages"] = sorted({(d, p) for g in groups for d, p in g["pages"]})
    row["retrieval"] = metrics.retrieval_scores(retrieved, groups, KS)
    row["provenance"] = metrics.provenance(cited, groups)
    ref = q["reference_answer"]
    num = q.get("numeric") or {}
    # an unanswered question scores False on every labelled field (empty answer)
    row["numeric"] = metrics.numeric_scores(row["answer"], value=num.get("value"),
                                            unit=num.get("unit"), period=num.get("period"),
                                            reference=ref)
    if row["short_answer"]:
        row["exact_match"] = metrics.exact_match(row["answer"], ref)
        row["token_f1"] = metrics.token_f1(row["answer"], ref)
    if not answered:
        row["correct"], row["correct_source"] = False, "not answered"
    elif verdict is not None:
        ok = verdict == "correct"
        if ok and num.get("value") and row["numeric"]["value"] is False:
            ok = False  # a labelled value must match exactly, whatever the judge says
        row["correct"], row["correct_source"] = ok, "judge"
    elif row["numeric"]["value"] is not None:
        row["correct"], row["correct_source"] = row["numeric"]["value"], "numeric"
    elif row["short_answer"] and not use_judge:
        row["correct"], row["correct_source"] = row["token_f1"] >= 0.8, "token_f1>=0.8"
    else:
        row["correct"], row["correct_source"] = None, "unlabelled (judge failed)"
    return row


# ---------------------------------------------------------------- running a system

def _run_file(out_dir: Path, system: str, repeat: int) -> Path:
    return out_dir / "runs" / f"{system}__r{repeat}.json"


def run_system(system: str, repeat: int, questions: list[dict], docs: list[Path],
               out_dir: Path, use_judge: bool, judge_config: Config, fresh: bool = False) -> dict:
    config = config_for(system)
    path = _run_file(out_dir, system, repeat)
    record = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
    qids = [q["id"] for q in questions]
    judge_id = _judge_id(judge_config) if use_judge else None
    done = {r["id"]: r for r in (record or {}).get("rows", [])
            if (record or {}).get("judge") == judge_id}
    if record and record.get("complete") and list(done) == qids:
        print(f"[{system} r{repeat}] complete, {len(done)} questions (cached)", flush=True)
        return record

    llm = caching_llm(config, CACHE_PATH, repeat, read=not fresh)
    judge_llm = caching_llm(judge_config, CACHE_PATH) if use_judge else None
    llm_mod.set_llm(llm)
    withheld = _Withheld()
    logger.addHandler(withheld)
    t_index = time.perf_counter()
    try:
        index = get_or_build_index(docs, config)
        index_s = time.perf_counter() - t_index
        rows = []
        for q in questions:
            if q["id"] in done:
                rows.append(done[q["id"]])
                continue
            t0 = time.perf_counter()
            hits0, misses0 = llm.hits, llm.misses
            for attempt in (1, 2):  # one retry: a local-LLM timeout under load is not a result
                try:
                    retrieved = retrieve_with_config(q["question"], max(RETRIEVAL_DEPTH, config.top_k),
                                                     index, config, llm)
                    result = answer_from_passages(q["question"], retrieved[:config.top_k], config,
                                                  index)
                    break
                except LLMError as e:
                    if attempt == 2:
                        raise
                    print(f"[{system} r{repeat}] {q['id']} LLM error, retrying once: {e}", flush=True)
            latency = time.perf_counter() - t0
            cached_calls, fresh_calls = llm.hits - hits0, llm.misses - misses0
            pages = [(p.chunk.document_name, p.chunk.page_number) for p in retrieved]
            verdict, reason = None, ""
            answered = result.status is not Status.INSUFFICIENT_EVIDENCE
            if use_judge and answered and q["category"] != "unanswerable":
                verdict, reason = judge_answer(q["question"], q["reference_answer"],
                                               result.answer, judge_llm)
            row = score_row(q, result, pages, latency, verdict, reason, use_judge)
            # latency is only a measurement when no answer-side LLM call came from the cache
            row["llm_calls"] = {"fresh": fresh_calls, "cached": cached_calls}
            row["latency_valid"] = cached_calls == 0
            if withheld.records:
                row["withheld_answer"] = withheld.records[-1]
            withheld.records.clear()
            rows.append(row)
            print(f"[{system} r{repeat}] {q['id']:<5} {row['status']:<22} {row['latency_s']:>6.1f}s "
                  f"correct={row['correct']!s:<5} {(row['answer'] or '')[:70]!r}", flush=True)
            _save(path, system, repeat, config, rows, False, llm, judge_id, index_s)
        return _save(path, system, repeat, config, rows, True, llm, judge_id, index_s)
    finally:
        logger.removeHandler(withheld)
        llm_mod.set_llm(None)
        llm.close()
        if judge_llm is not None:
            judge_llm.close()


def _judge_id(config: Config) -> str:
    return f"{JUDGE_VERSION}:{config.llm_provider}:{config.llm_model}"


def _save(path: Path, system: str, repeat: int, config: Config, rows: list[dict],
          complete: bool, llm, judge_id: str | None, index_s: float) -> dict:
    record = {
        "system": system, "description": SYSTEMS[system][1], "repeat": repeat,
        "complete": complete, "judge": judge_id,
        "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run_config": run_record(config), "index_seconds": round(index_s, 1),
        "llm_cache": {"hits": llm.hits, "misses": llm.misses},
        "rows": rows,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(record, indent=1, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)
    return record


def rescore(out_dir: Path, questions: list[dict]) -> int:
    """Recompute every stored row's metrics from its saved outputs (status, answer,
    citations, warnings, retrieved pages, judge verdict) with the current scoring code.
    No model calls. Rows saved before per-row cache tracking get latency_valid False."""
    by_id = {q["id"]: q for q in questions}
    n = 0
    for f in sorted((out_dir / "runs").glob("*__r*.json")):
        rec = json.loads(f.read_text(encoding="utf-8"))
        new_rows = []
        for r in rec["rows"]:
            q = by_id.get(r["id"])
            if q is None:
                new_rows.append(r)
                continue
            result = Result.from_dict({"question": r["question"], "answer": r["answer"] or "-",
                                       "status": r["status"], "citations": r["citations"],
                                       "warnings": r["warnings"]})
            row = score_row(q, result, [tuple(p) for p in r["retrieved_pages"]], r["latency_s"],
                            r.get("judge_verdict"), r.get("judge_reason", ""),
                            rec.get("judge") is not None)
            for k in ("withheld_answer", "llm_calls"):
                if k in r:
                    row[k] = r[k]
            row["latency_valid"] = r.get("latency_valid", False)
            new_rows.append(row)
            n += 1
        rec["rows"] = new_rows
        f.write_text(json.dumps(rec, indent=1, ensure_ascii=False), encoding="utf-8")
    return n


# ---------------------------------------------------------------- OCR quality

def ocr_quality(config: Config, out_dir: Path) -> dict:
    """CER/WER of the OCR fallback on every scanned PDF that has per-page ground truth in
    data/ocr_ground_truth/<name>/page_NNN.txt (<name> = file stem without '_scanned')."""
    from trustqa.processing.process import process_pdfs_detailed, processing_signature

    gt_root = ROOT / "data" / "ocr_ground_truth"
    target = out_dir / "ocr_quality.json"
    sig = processing_signature(config)
    if target.is_file():
        cached = json.loads(target.read_text(encoding="utf-8"))
        if cached.get("processing_signature") == sig:
            return cached
    docs = []
    for pdf in sorted((Path(config.raw_pdfs_dir) / "scanned").glob("*.pdf")):
        gt = gt_root / pdf.stem.removesuffix("_scanned")
        if not gt.is_dir():
            continue
        print(f"[ocr] {pdf.name}: OCR vs ground truth", flush=True)
        report = process_pdfs_detailed([pdf], config).documents[0]
        pages = []
        for pn, res in sorted(report.ocr_results.items()):
            truth = gt / f"page_{pn:03d}.txt"
            if truth.is_file():
                s = metrics.cer_wer(truth.read_text(encoding="utf-8"), res.text)
                pages.append({"page": pn, "cer": round(s["cer"], 4), "wer": round(s["wer"], 4),
                              "ocr_confidence": res.mean_confidence})
        if pages:
            docs.append({"document": pdf.name, "pages": pages,
                         "cer_mean": round(sum(p["cer"] for p in pages) / len(pages), 4),
                         "wer_mean": round(sum(p["wer"] for p in pages) / len(pages), 4)})
    out = {"processing_signature": sig, "documents": docs}
    target.write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


# ---------------------------------------------------------------- main

def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(prog="python -m eval.run_all",
                                 description=__doc__.split("\n")[0])
    ap.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    ap.add_argument("--systems", default=",".join(MAIN_SYSTEMS + ABLATIONS),
                    help=f"comma-separated, from {list(SYSTEMS)}")
    ap.add_argument("--repeats", type=int, default=1,
                    help="runs per system; repeat 0 is shared with earlier runs via the cache")
    ap.add_argument("--no-judge", action="store_true", help="skip the LLM correctness judge")
    ap.add_argument("--judge-provider", help="LLM provider for the judge (default: as generation)")
    ap.add_argument("--judge-model", help="LLM model for the judge (default: as generation)")
    ap.add_argument("--no-ocr-quality", action="store_true", help="skip CER/WER")
    ap.add_argument("--report-only", action="store_true", help="only rebuild tables and plots")
    ap.add_argument("--fresh", action="store_true",
                    help="do not read the LLM cache for answers (latency runs); outputs are stored")
    ap.add_argument("--rescore", action="store_true",
                    help="recompute metrics of saved runs with the current scoring (no model calls)")
    ap.add_argument("--limit", type=int, help="first N questions only (smoke test)")
    ap.add_argument("--out", type=Path, help="results directory (default eval/results/<stem>)")
    args = ap.parse_args(argv)
    # trustqa logs at INFO (the withheld-answer record is captured from it); only warnings
    # reach the console
    console = logging.StreamHandler(sys.stderr)
    console.setLevel(logging.WARNING)
    console.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    logger.addHandler(console)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not args.questions.is_file():
        print(f"error: {args.questions} not found. Label questions in eval/questions.jsonl "
              "(schema: eval/questions.py), or pass --questions eval/dev_questions.jsonl.")
        return 2
    stem = args.questions.stem + (f"_first{args.limit}" if args.limit else "")
    out_dir = args.out or ROOT / "eval" / "results" / stem
    plots_dir = ROOT / "eval" / "plots" / stem
    questions = load_questions(args.questions)
    if args.limit:
        questions = questions[:args.limit]
    systems = [s.strip() for s in args.systems.split(",") if s.strip()]
    unknown = [s for s in systems if s not in SYSTEMS]
    if unknown:
        print(f"error: unknown system(s) {unknown}; use {list(SYSTEMS)}")
        return 2

    from eval import error_analysis, report

    if args.rescore:
        print(f"rescored {rescore(out_dir, questions)} rows")
    if not (args.report_only or args.rescore):
        base = config_for("C")
        docs = default_docs(base.raw_pdfs_dir)
        errors, warnings = validate(questions, document_page_counts(docs))
        print(format_counts(questions))
        for w in warnings:
            print("warning:", w)
        if errors:
            for e in errors:
                print("error:", e)
            print("question set is invalid; fix it before evaluating")
            return 2
        judge_cfg = config_for("C", **{k: v for k, v in (("llm_provider", args.judge_provider),
                                                          ("llm_model", args.judge_model)) if v})
        try:
            print("LLM:", check_available(base))
            if not args.no_judge:
                print("judge:", check_available(judge_cfg))
        except (LLMError, ValueError) as e:
            print(f"error: {e}")
            return 2
        print(f"documents: {[d.name for d in docs]}")
        for system in systems:
            for r in range(args.repeats):
                run_system(system, r, questions, docs, out_dir, not args.no_judge, judge_cfg,
                           args.fresh)
        if not args.no_ocr_quality:
            ocr_quality(base, out_dir)

    summary = report.build(out_dir, plots_dir, [q["id"] for q in questions])
    error_analysis.write(out_dir, summary)
    print(report.format_summary(summary))
    print(f"results: {out_dir.relative_to(ROOT)}  plots: {plots_dir.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
