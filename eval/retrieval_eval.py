"""Retrieval evaluation (Phase 4): Recall@K and MRR on a labelled question set.

Labels are page-level: each answerable question has one or more *evidence groups*, one per
document (or fact) the answer needs. A group lists the (filename, page) pairs that state that
part of the answer; it is found at K when any of the top-K chunks lies on one of its pages.
Pages are labelled by filename + 1-indexed page, never DOC ID (DOC IDs follow input order).

Metrics, macro-averaged over answerable questions (unanswerable ones are skipped):
  recall@K       fraction of a question's evidence groups found in the top K
  all_recall@K   1 if every evidence group is found in the top K (complete evidence)
  mrr            1 / rank of the first chunk on any labelled page (0 if not in the top max(K))
  latency_ms     wall time of one retrieve() call (model already loaded), mean and p95

Usage:
  python eval/retrieval_eval.py                          # system C index, default questions
  python eval/retrieval_eval.py --systems C,B,A --out eval/results/retrieval_baseline.json
  python eval/retrieval_eval.py --grid --out eval/results/retrieval_grid.json   # flag combos on C
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trustqa.cli import default_docs  # noqa: E402
from trustqa.config import Config, load_config  # noqa: E402
from trustqa.pipeline import get_or_build_index, index_key  # noqa: E402
from trustqa.llm import get_llm  # noqa: E402
from trustqa.retrieval.decompose import DECOMPOSE_VERSION, parse_subqueries  # noqa: E402
from trustqa.retrieval.search import retrieve, retrieve_with_config  # noqa: E402

KS = (1, 3, 5, 10)
# Phase 4 flag combinations (retrieval-time only; the index is the same for all of them).
# Parameters are the standard defaults (RRF k=60, quota 2 per document at top_k=5, MMR
# lambda 0.7), fixed before looking at grid results; SENSITIVITY varies them separately.
GRID = {
    "dense (baseline)": {},
    "hybrid": dict(retrieval_hybrid=True),
    "quota": dict(retrieval_diversify="quota"),
    "mmr": dict(retrieval_diversify="mmr"),
    "hybrid+quota": dict(retrieval_hybrid=True, retrieval_diversify="quota"),
    "hybrid+mmr": dict(retrieval_hybrid=True, retrieval_diversify="mmr"),
}
RERANK_GRID = {
    "rerank": dict(rerank=True),
    "hybrid+rerank": dict(retrieval_hybrid=True, rerank=True),
    "hybrid+rerank+quota": dict(retrieval_hybrid=True, rerank=True, retrieval_diversify="quota"),
}
DECOMPOSE_GRID = {  # needs the configured LLM (Ollama); one extra call per question
    "decompose": dict(retrieval_decompose=True),
    "hybrid+decompose": dict(retrieval_hybrid=True, retrieval_decompose=True),
}
SENSITIVITY = {
    "hybrid rrf_k=10": dict(retrieval_hybrid=True, rrf_k=10),
    "hybrid+quota q=1": dict(retrieval_hybrid=True, retrieval_diversify="quota", doc_quota=1),
    "hybrid+quota q=3": dict(retrieval_hybrid=True, retrieval_diversify="quota", doc_quota=3),
    "hybrid+mmr l=0.5": dict(retrieval_hybrid=True, retrieval_diversify="mmr", mmr_lambda=0.5),
}
# All Phase 4 flags off: variants and --systems runs set every flag explicitly, so a developer
# .env (e.g. RETRIEVAL_HYBRID=1 for system C) cannot change what "dense" or a baseline means.
FLAGS_OFF = dict(retrieval_hybrid=False, retrieval_diversify="none", rerank=False,
                 retrieval_decompose=False)
SYSTEMS = {
    "A": dict(ocr_enabled=False, flatten_tables=True, verify=False),
    "B": dict(baseline_b=True),
    "C": dict(),
}


def default_questions_path() -> Path:
    full = ROOT / "eval" / "questions.jsonl"
    return full if full.is_file() else ROOT / "eval" / "dev_questions.jsonl"


def load_questions(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def group_hits(retrieved: list[tuple[str, int]], groups: list[dict]) -> list[int | None]:
    """For each evidence group, the 1-based rank of its first hit, or None."""
    out: list[int | None] = []
    for g in groups:
        pages = {(d, int(p)) for d, p in g["pages"]}
        out.append(next((r for r, dp in enumerate(retrieved, 1) if dp in pages), None))
    return out


def score_question(retrieved: list[tuple[str, int]], groups: list[dict],
                   ks: tuple[int, ...]) -> dict:
    hits = group_hits(retrieved, groups)
    first = min((h for h in hits if h is not None), default=None)
    row = {"mrr": 1.0 / first if first else 0.0, "group_ranks": hits}
    for k in ks:
        found = [h is not None and h <= k for h in hits]
        row[f"recall@{k}"] = sum(found) / len(found)
        row[f"all_recall@{k}"] = float(all(found))
    return row


def aggregate(rows: list[dict], ks: tuple[int, ...]) -> dict:
    if not rows:
        return {"n": 0}
    keys = ["mrr"] + [f"{m}@{k}" for k in ks for m in ("recall", "all_recall")]
    agg = {"n": len(rows)}
    agg.update({k: round(statistics.fmean(r[k] for r in rows), 4) for k in keys})
    return agg


def evaluate(questions: list[dict], index, config: Config, ks: tuple[int, ...] = KS,
             retrieve_fn=None) -> dict:
    """Run retrieval for every answerable question and compute metrics.

    `retrieve_fn(question, top_k, index)` defaults to trustqa.retrieval.search.retrieve.
    """
    retrieve_fn = retrieve_fn or (lambda q, k, idx: retrieve(q, k, idx))
    depth = max(ks)
    answerable = [q for q in questions if q.get("evidence_groups")]
    retrieve_fn(answerable[0]["question"], depth, index)  # warm-up (model load)
    per_q, latencies = [], []
    for q in answerable:
        t0 = time.perf_counter()
        res = retrieve_fn(q["question"], depth, index)
        latencies.append((time.perf_counter() - t0) * 1000)
        retrieved = [(r.chunk.document_name, r.chunk.page_number) for r in res]
        row = score_question(retrieved, q["evidence_groups"], ks)
        row.update(id=q["id"], category=q["category"],
                   top5=[r.chunk.chunk_id for r in res[:5]])
        per_q.append(row)
    cats = sorted({r["category"] for r in per_q})
    return {
        "overall": aggregate(per_q, ks),
        "by_category": {c: aggregate([r for r in per_q if r["category"] == c], ks) for c in cats},
        "latency_ms": {"mean": round(statistics.fmean(latencies), 1),
                       "p95": round(sorted(latencies)[int(0.95 * (len(latencies) - 1))], 1)},
        "skipped_unanswerable": len(questions) - len(answerable),
        "per_question": per_q,
    }


def load_system(system: str, **overrides) -> tuple[Config, object, list[Path]]:
    config = load_config(**SYSTEMS[system], **{**FLAGS_OFF, **overrides})
    docs = default_docs(config.raw_pdfs_dir)
    return config, get_or_build_index(docs, config), docs


def paired(base: list[dict], other: list[dict], metric: str, n_boot: int = 10000,
           seed: int = 0) -> dict:
    """Per-question comparison of `other` vs `base` on one metric: wins/losses/ties, mean
    difference and a paired bootstrap 95% CI (fixed seed, so reruns give the same CI)."""
    import random

    b = {r["id"]: r[metric] for r in base}
    diffs = [r[metric] - b[r["id"]] for r in other]
    rng = random.Random(seed)
    boots = sorted(statistics.fmean(rng.choices(diffs, k=len(diffs))) for _ in range(n_boot))
    return {"wins": sum(d > 0 for d in diffs), "losses": sum(d < 0 for d in diffs),
            "ties": sum(d == 0 for d in diffs), "mean_diff": round(statistics.fmean(diffs), 4),
            "ci95": [round(boots[int(0.025 * n_boot)], 4), round(boots[int(0.975 * n_boot)], 4)]}


def print_row(name: str, res: dict) -> None:
    o, c = res["overall"], res["by_category"]
    md = c.get("multi_doc", {})
    print(f"{name:<22} R@1={o['recall@1']:.3f} R@5={o['recall@5']:.3f} "
          f"R@10={o['recall@10']:.3f} MRR={o['mrr']:.3f} | multi-doc all@5="
          f"{md.get('all_recall@5', float('nan')):.3f} | {res['latency_ms']['mean']:.1f} ms")


class RecordingLLM:
    """Wraps an LLM and keeps the last raw output per question (for the report)."""

    def __init__(self, inner):
        self.inner, self.outputs = inner, {}

    def complete(self, prompt: str, system: str | None = None, json: bool = False) -> str:
        out = self.inner.complete(prompt, system=system, json=json)
        q = prompt.split("\n", 1)[0].removeprefix("Question: ")
        self.outputs[q] = out
        return out


def run_grid(args, questions: list[dict], report: dict) -> int:
    system = args.systems.split(",")[0]
    _, index, docs = load_system(system)
    variants = dict(GRID)
    if args.rerank:
        variants.update(RERANK_GRID)
    if args.decompose:
        variants.update(DECOMPOSE_GRID)
    if args.sensitivity:
        variants.update(SENSITIVITY)
    report.update(system=system, variants={})
    for name, flags in variants.items():
        config = load_config(**SYSTEMS[system], **{**FLAGS_OFF, **flags})
        llm = RecordingLLM(get_llm(config)) if config.retrieval_decompose else None
        res = evaluate(questions, index, config,
                       retrieve_fn=lambda q, k, idx, c=config, m=llm:
                       retrieve_with_config(q, k, idx, c, m))
        res["flags"] = flags
        if llm is not None:
            res["llm"] = {"provider": config.llm_provider, "model": config.llm_model,
                          "decompose_version": DECOMPOSE_VERSION}
            by_q = {q["id"]: q["question"] for q in questions}
            for row in res["per_question"]:
                raw = llm.outputs.get(by_q[row["id"]], "")
                row["subqueries"] = parse_subqueries(raw, by_q[row["id"]], config.max_subqueries)
        base = report["variants"].get("dense (baseline)")
        if base is not None and name != "dense (baseline)":
            res["vs_baseline"] = {m: paired(base["per_question"], res["per_question"], m)
                                  for m in ("recall@5", "mrr", "all_recall@5")}
        res["index_key"] = index_key(docs, config)
        report["variants"][name] = res
        print_row(name, res)
        if "vs_baseline" in res:
            v = res["vs_baseline"]
            print(f"{'':<22} vs dense: R@5 +{v['recall@5']['wins']}/-{v['recall@5']['losses']} "
                  f"CI{v['recall@5']['ci95']}  MRR +{v['mrr']['wins']}/-{v['mrr']['losses']} "
                  f"diff={v['mrr']['mean_diff']:+.3f} CI{v['mrr']['ci95']}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"wrote {args.out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--questions", type=Path, default=default_questions_path())
    ap.add_argument("--systems", default="C", help="comma-separated: A,B,C")
    ap.add_argument("--out", type=Path, default=None, help="write results JSON here")
    ap.add_argument("--grid", action="store_true",
                    help="evaluate Phase 4 flag combinations on the first system instead")
    ap.add_argument("--rerank", action="store_true", help="with --grid: add reranker variants")
    ap.add_argument("--decompose", action="store_true",
                    help="with --grid: add LLM query-decomposition variants")
    ap.add_argument("--sensitivity", action="store_true",
                    help="with --grid: add parameter-sensitivity variants")
    args = ap.parse_args(argv)

    questions = load_questions(args.questions)
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "questions_file": args.questions.relative_to(ROOT).as_posix(),
        "n_questions": len(questions),
        "ks": list(KS),
        "systems": {},
    }
    if args.grid:
        return run_grid(args, questions, report)
    for s in args.systems.split(","):
        config, index, docs = load_system(s)
        res = evaluate(questions, index, config,
                       retrieve_fn=lambda q, k, idx, c=config: retrieve_with_config(q, k, idx, c))
        res["index_key"] = index_key(docs, config)
        res["n_chunks"] = len(index)
        res["settings"] = {"embedding_model": config.embedding_model,
                           "chunk_size": config.chunk_size, "chunk_overlap": config.chunk_overlap}
        report["systems"][s] = res
        o = res["overall"]
        print(f"system {s} ({res['n_chunks']} chunks): " + "  ".join(
            f"{k}={o[k]:.3f}" for k in ("recall@1", "recall@5", "recall@10", "all_recall@5", "mrr"))
            + f"  latency={res['latency_ms']['mean']}ms")
        for c, a in res["by_category"].items():
            print(f"   {c:<10} n={a['n']:<3} R@5={a['recall@5']:.3f} R@10={a['recall@10']:.3f} "
                  f"all@5={a['all_recall@5']:.3f} MRR={a['mrr']:.3f}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
