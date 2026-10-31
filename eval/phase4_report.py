"""Build the Phase 4 tables from the result JSON files (no number is typed by hand).

Usage:
  python eval/phase4_report.py --answers eval/results/phase4_answers.json \
      eval/results/phase4_answers_hybrid.json > eval/results/phase4_tables.md

Inputs: eval/results/retrieval_baseline.json, eval/results/retrieval_grid.json and any number of
answer_check.py outputs (their variants are merged; a later file wins on a name clash).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "eval" / "results"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

from answer_check import summary  # noqa: E402

MAIN = ["dense (baseline)", "hybrid", "quota", "mmr", "hybrid+quota", "hybrid+mmr", "rerank",
        "hybrid+rerank", "hybrid+rerank+quota", "decompose", "hybrid+decompose"]
NAMED = [("T01", "T01 TAS-B on FiQA-2018 (F3)"), ("S02", "S02 Adam beta2, scanned (F4)"),
         ("T03", "T03 (F5)"), ("M02", "M02 Transformer vs Self-RAG GPUs (F12)")]
NAMED_VARIANTS = ["dense (baseline)", "hybrid", "rerank", "hybrid+rerank", "decompose",
                  "hybrid+decompose"]


def f3(x) -> str:
    return "-" if x is None else f"{x:.3f}"


def load(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def baseline_table(base: dict) -> list[str]:
    out = ["| System | R@1 | R@5 | R@10 | MRR | direct R@5 | table R@5 | scanned R@5 | "
           "multi-doc R@5 | multi-doc all@5 |", "|" + "---|" * 10]
    for name, s in base["systems"].items():
        o, c = s["overall"], s["by_category"]
        out.append(f"| {name} | {f3(o['recall@1'])} | {f3(o['recall@5'])} | {f3(o['recall@10'])} | "
                   f"{f3(o['mrr'])} | {f3(c['direct']['recall@5'])} | {f3(c['table']['recall@5'])} | "
                   f"{f3(c['scanned']['recall@5'])} | {f3(c['multi_doc']['recall@5'])} | "
                   f"{f3(c['multi_doc']['all_recall@5'])} |")
    return out


def vs(v: dict) -> tuple[str, str]:
    b = v.get("vs_baseline")
    if not b:
        return "-", "-"
    r, m = b["recall@5"], b["mrr"]
    return (f"+{r['wins']}/-{r['losses']}",
            f"{m['mean_diff']:+.3f} [{m['ci95'][0]:+.3f}, {m['ci95'][1]:+.3f}]")


def grid_table(grid: dict, names: list[str]) -> list[str]:
    out = ["| Variant | R@1 | R@5 | R@10 | MRR | multi-doc all@5 | table R@5 | scanned R@5 | "
           "direct R@5 | R@5 vs dense (+/-) | MRR diff [95% CI] | latency mean / p95 ms |",
           "|" + "---|" * 12]
    for n in names:
        v = grid["variants"].get(n)
        if v is None:
            continue
        o, c = v["overall"], v["by_category"]
        r5, md = vs(v)
        lat = v["latency_ms"]
        out.append(f"| {n} | {f3(o['recall@1'])} | {f3(o['recall@5'])} | {f3(o['recall@10'])} | "
                   f"{f3(o['mrr'])} | {f3(c['multi_doc']['all_recall@5'])} | "
                   f"{f3(c['table']['recall@5'])} | {f3(c['scanned']['recall@5'])} | "
                   f"{f3(c['direct']['recall@5'])} | {r5} | {md} | "
                   f"{lat['mean']:.0f} / {lat['p95']:.0f} |")
    return out


def group_ranks(v: dict, qid: str) -> list:
    for q in v["per_question"]:
        if q["id"] == qid:
            return q["group_ranks"]
    return []


def named_table(grid: dict) -> list[str]:
    names = [n for n in NAMED_VARIANTS if n in grid["variants"]]
    out = ["| Question | " + " | ".join(names) + " |", "|" + "---|" * (len(names) + 1)]
    for qid, label in NAMED:
        cells = []
        for n in names:
            rs = group_ranks(grid["variants"][n], qid)
            cells.append(" / ".join("-" if r is None else str(r) for r in rs) or "?")
        out.append(f"| {label} | " + " | ".join(cells) + " |")
    return out


def answer_tables(variants: dict) -> list[str]:
    out = ["| Variant | answered (of answerable) | Verified | cites a labelled page | "
           "reference numbers in answer | unanswerable abstained | latency mean s |",
           "|" + "---|" * 7]
    for n, v in variants.items():
        s = v["summary"]
        out.append(f"| {n} | {s['answered']}/{s['answerable']} | {s['verified']} | "
                   f"{s['cites_evidence']} | {s['numeric_ref_in_answer']} | "
                   f"{s['unanswerable_correctly_abstained']} | {s['latency_s_mean']} |")
    cats = sorted({r["category"] for v in variants.values() for r in v["per_question"]})
    out += ["", "Per category, answered / Verified / cites a labelled page:", "",
            "| Variant | " + " | ".join(cats) + " |", "|" + "---|" * (len(cats) + 1)]
    for n, v in variants.items():
        cells = []
        for c in cats:
            rows = [r for r in v["per_question"] if r["category"] == c]
            if rows and "answered" not in rows[0]:
                cells.append(f"abstained {sum(r['correct_abstention'] for r in rows)}/{len(rows)}")
                continue
            cells.append(f"{sum(r['answered'] for r in rows)} / "
                         f"{sum(r['status'] == 'Verified' for r in rows)} / "
                         f"{sum(r['cites_evidence'] for r in rows)} (of {len(rows)})")
        out.append(f"| {n} | " + " | ".join(cells) + " |")
    return out


def answer_diff(variants: dict, a: str, b: str) -> list[str]:
    ra = {r["id"]: r for r in variants[a]["per_question"]}
    rb = {r["id"]: r for r in variants[b]["per_question"]}
    keys = ("status", "cites_evidence", "ref_in_answer", "correct_abstention")
    out = [f"| Question | {a}: status / cites / ref | {b}: status / cites / ref |", "|---|---|---|"]
    n = 0
    for qid in ra:
        if qid not in rb:
            continue
        x, y = ra[qid], rb[qid]
        if all(x.get(k) == y.get(k) for k in keys):
            continue
        n += 1

        def cell(r):
            if "correct_abstention" in r:
                return f"{r['status']} (abstained: {r['correct_abstention']})"
            ref = {None: "n/a", True: "yes", False: "no"}[r["ref_in_answer"]]
            return f"{r['status']} / {'yes' if r['cites_evidence'] else 'no'} / {ref}"
        out.append(f"| {qid} | {cell(x)} | {cell(y)} |")
    return out if n else [f"No question changed status, citation hit or reference match ({a} vs {b})."]


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--baseline", type=Path, default=RES / "retrieval_baseline.json")
    ap.add_argument("--grid", type=Path, default=RES / "retrieval_grid.json")
    ap.add_argument("--answers", type=Path, nargs="*", default=[])
    args = ap.parse_args(argv)
    base, grid = load(args.baseline), load(args.grid)
    answers: dict = {}
    for p in args.answers:
        for n, v in load(p)["variants"].items():
            answers[n] = v
    note = ""
    if answers:  # compare variants only on the questions every run finished (paired)
        common = set.intersection(*({r["id"] for r in v["per_question"]} for v in answers.values()))
        n_all = max(len(v["per_question"]) for v in answers.values())
        for n, v in answers.items():
            rows = [r for r in v["per_question"] if r["id"] in common]
            answers[n] = {**v, "per_question": rows, "summary": summary(rows)}
        if len(common) < n_all:
            partial = [n for n, v in answers.items() if v.get("partial")]
            note = (f"**Partial:** {', '.join(partial)} finished {len(common)} of {n_all} questions "
                    f"(the run was stopped); every variant below is scored on those {len(common)} "
                    f"questions only: {', '.join(sorted(common))}.")
    sens = [n for n in grid["variants"] if n not in MAIN]
    lines = [f"<!-- generated by eval/phase4_report.py from {args.baseline.name} "
             f"({base['created_at']}), {args.grid.name} ({grid['created_at']})"
             + "".join(f", {p.name}" for p in args.answers) + " -->", "",
             "### Baseline (before, flags off)", "", *baseline_table(base), "",
             "### Flag combinations on system C", "", *grid_table(grid, MAIN), "",
             "### Sensitivity (not used for the recommendation)", "", *grid_table(grid, sens), "",
             "### Named failures: rank of the first chunk on a labelled page, per evidence group "
             "(\"-\" = not in the top 10)", "", *named_table(grid), ""]
    if answers:
        lines += ["### Answer level (full system C pipeline: generation + verification)", ""]
        if note:
            lines += [note, ""]
        lines += [*answer_tables(answers), ""]
        names = list(answers)
        for other in names[1:]:
            lines += [f"Questions that changed, {names[0]} -> {other}:", "",
                      *answer_diff(answers, names[0], other), ""]
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
