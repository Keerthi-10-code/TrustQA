"""Dump the worst failures per system for hand analysis (Phase 6, step 6), and a spot-check
sheet for validating the LLM judge.

  python -m eval.error_analysis [--results eval/results/<stem>] [--per-system 15]

Writes <results>/error_analysis.md: per system, failures ranked by severity
  1. verified_but_incorrect     Verified, but judged incorrect (the worst case for TrustQA)
  2. unanswerable_answered      an answer shown for an unanswerable question
  3. shown_incorrect            a non-Verified answer that is judged incorrect
  4. abstained_with_evidence    abstained although a labelled page was retrieved
  5. retrieval_miss             abstained or wrong, no labelled page in the top 5
with question, reference, answer, status, warnings, cited vs labelled pages, top retrieved
pages and any answer verification withheld. It lists cases; the analysis (why) is yours.

Also writes <results>/spot_check.csv (once; existing human labels are kept): a fixed random
sample of judged answers with an empty human_verdict column. Fill it in (correct /
partially_correct / incorrect) and re-run `python -m eval.run_all --report-only` to get the
judge agreement.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

from eval.report import load_runs
from eval.questions import ROOT

SEVERITY = ["verified_but_incorrect", "unanswerable_answered", "shown_incorrect",
            "abstained_with_evidence", "retrieval_miss"]
IE, VERIFIED = "Insufficient Evidence", "Verified"


def classify(r: dict) -> str | None:
    if not r["answerable"]:
        return "unanswerable_answered" if r["status"] != IE else None
    hit = (r["retrieval"].get("recall@5") or 0) > 0
    if r["status"] == VERIFIED and r["correct"] is False:
        return "verified_but_incorrect"
    if r["status"] != IE and r["correct"] is False:
        return "shown_incorrect" if hit else "retrieval_miss"
    if r["status"] == IE:
        return "abstained_with_evidence" if hit else "retrieval_miss"
    return None


def _case(r: dict) -> str:
    lab = ", ".join(f"{d} p{p}" for d, p in r.get("labelled_pages", [])) or "-"
    cited = ", ".join(f"{d} p{p}" for d, p in r["cited_pages"]) or "-"
    top = ", ".join(f"{d.split('_')[0]} p{p}" for d, p in r["retrieved_pages"][:5])
    lines = [f"**{r['id']}** ({r['category']}) - {r['question']}",
             f"- reference: {r.get('reference_answer') or '(should abstain)'}",
             f"- status: {r['status']}; correct: {r['correct']} ({r['correct_source']})"
             + (f"; judge: {r['judge_verdict']} - {r['judge_reason']}" if r.get("judge_verdict") else ""),
             f"- answer: {r['answer'] or '(none)'}",
             f"- cited: {cited} | labelled: {lab}",
             f"- top-5 retrieved: {top}"]
    if r.get("withheld_answer"):
        lines.append(f"- withheld by verification: {r['withheld_answer']}")
    for w in r["warnings"]:
        lines.append(f"- warning [{w['code']}] {w['message']}")
    lines.append("- why (fill in): ")
    return "\n".join(lines)


def write(out_dir: Path, summary: dict | None = None, per_system: int = 15,
          spot_check_n: int = 20, seed: int = 0) -> Path:
    runs = load_runs(out_dir, qids=_all_ids(out_dir))
    md = ["# Error analysis (cases for hand analysis)", "",
          "Ranked by severity per system (repeat 0). Fill in 'why' for 10-15 cases.", ""]
    for name, recs in runs.items():
        rows = recs[0]["rows"]
        cases = sorted(((classify(r), r) for r in rows if classify(r)),
                       key=lambda t: SEVERITY.index(t[0]))
        counts = {s: sum(1 for c, _ in cases if c == s) for s in SEVERITY}
        md += [f"## {name} - {recs[0]['description']}", "",
               "Counts: " + ", ".join(f"{k} {v}" for k, v in counts.items()), ""]
        for kind, r in cases[:per_system]:
            md += [f"### [{kind}]", _case(r), ""]
    path = out_dir / "error_analysis.md"
    path.write_text("\n".join(md), encoding="utf-8")
    _spot_check(out_dir, runs, spot_check_n, seed)
    return path


def _all_ids(out_dir: Path) -> list[str]:
    ids: list[str] = []
    for f in sorted((out_dir / "runs").glob("*__r*.json")):
        for r in json.loads(f.read_text(encoding="utf-8"))["rows"]:
            if r["id"] not in ids:
                ids.append(r["id"])
    return ids


def _spot_check(out_dir: Path, runs: dict, n: int, seed: int) -> None:
    path = out_dir / "spot_check.csv"
    if path.is_file():
        return  # keep the sample (and any human labels) stable
    pool = [(name, r) for name, recs in runs.items() for r in recs[0]["rows"]
            if r.get("judge_verdict")]
    random.Random(seed).shuffle(pool)
    fields = ["system", "id", "question", "reference_answer", "answer", "judge_verdict",
              "judge_reason", "human_verdict"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for name, r in pool[:n]:
            w.writerow({"system": name, "id": r["id"], "question": r["question"],
                        "reference_answer": r["reference_answer"], "answer": r["answer"],
                        "judge_verdict": r["judge_verdict"], "judge_reason": r["judge_reason"],
                        "human_verdict": ""})


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", type=Path, default=ROOT / "eval" / "results" / "questions")
    ap.add_argument("--per-system", type=int, default=15)
    args = ap.parse_args(argv)
    if not (args.results / "runs").is_dir():
        print(f"error: no runs under {args.results}; run python -m eval.run_all first")
        return 2
    print(f"wrote {write(args.results, per_system=args.per_system)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
