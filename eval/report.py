"""Tables and plots from eval/results/<stem>/runs/*.json (Phase 6, step 5).

Writes summary.json / summary.csv / summary.md, by_category.csv, per_question.csv and
plots (systems_overview.png, ablation_deltas.png, accuracy_by_category.png). Only measured
values appear; a metric with no labelled questions is shown as "n/a", never as 0.
"""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

from eval import metrics
from eval.systems import ABLATIONS, MAIN_SYSTEMS, SYSTEMS

# metric key, label, higher is better
HEADLINE = [
    ("accuracy", "Answer accuracy (judge)", True),
    ("answered_rate", "Answered (answerable qs)", True),
    ("verified_rate", "Verified (answerable qs)", True),
    ("verified_precision", "Verified answers correct", True),
    ("unsupported_answer_rate", "Shown answers incorrect", False),
    ("exact_match", "Exact match (short)", True),
    ("token_f1", "Token F1 (short)", True),
    ("numeric_value", "Numeric value correct", True),
    ("numeric_unit", "Numeric unit correct", True),
    ("numeric_period", "Numeric period correct", True),
    ("recall@1", "Recall@1", True),
    ("recall@5", "Recall@5", True),
    ("all_groups@5", "All evidence groups@5", True),
    ("precision@5", "Precision@5", True),
    ("mrr", "MRR", True),
    ("provenance_hit", "Provenance: cites a labelled page", True),
    ("citation_precision", "Provenance: citation precision", True),
    ("incorrect_not_verified", "Incorrect answers not Verified", True),
    ("correct_abstention_rate", "Correct abstention (unanswerable)", True),
    ("over_abstention_rate", "Over-abstention (answerable)", False),
    ("conflict_recall", "Conflict detection recall", True),
    ("false_conflict_rate", "False conflict rate", False),
    ("latency_mean_s", "Latency mean (s)", False),
    ("latency_p90_s", "Latency p90 (s)", False),
]
BY_DESIGN = ["verified_rate", "verified_precision", "verified_n", "incorrect_not_verified"]
VARIANCE_KEYS = ["accuracy", "verified_rate", "answered_rate", "recall@5", "provenance_hit"]
OVERVIEW = ["accuracy", "verified_precision", "unsupported_answer_rate", "recall@5",
            "provenance_hit", "correct_abstention_rate", "latency_mean_s"]

# reference palette (dataviz skill, light mode)
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # palette slots 1-4 (adjacent bars, value-labelled)
BETTER, WORSE, NEUTRAL = "#2a78d6", "#e34948", "#f0efec"


def load_runs(out_dir: Path, qids: list[str]) -> dict[str, list[dict]]:
    """system -> run records (by repeat), each restricted to the question ids given."""
    runs: dict[str, list[dict]] = {}
    keep = set(qids)
    for f in sorted((out_dir / "runs").glob("*__r*.json")):
        rec = json.loads(f.read_text(encoding="utf-8"))
        rec["rows"] = [r for r in rec["rows"] if r["id"] in keep]
        if rec["rows"]:
            runs.setdefault(rec["system"], []).append(rec)
    for recs in runs.values():
        recs.sort(key=lambda r: r["repeat"])
    order = [s for s in SYSTEMS if s in runs]
    return {s: runs[s] for s in order}


def _agreement(recs: list[dict]) -> float | None:
    """Share of questions with identical status and answer text in every repeat."""
    if len(recs) < 2:
        return None
    by_id = [{r["id"]: (r["status"], r["answer"]) for r in rec["rows"]} for rec in recs]
    common = set.intersection(*(set(b) for b in by_id))
    if not common:
        return None
    return sum(len({b[i] for b in by_id}) == 1 for i in common) / len(common)


def judge_agreement(out_dir: Path) -> dict | None:
    """Agreement of the LLM judge with human labels filled into spot_check.csv."""
    path = out_dir / "spot_check.csv"
    if not path.is_file():
        return None
    with open(path, encoding="utf-8", newline="") as f:
        rows = [r for r in csv.DictReader(f) if (r.get("human_verdict") or "").strip()]
    rows = [r for r in rows if r.get("judge_verdict")]
    if not rows:
        return {"n": 0, "agreement": None, "note": "spot_check.csv has no human_verdict filled in yet"}
    norm = lambda v: v.strip().lower().replace(" ", "_")  # noqa: E731
    agree = sum(norm(r["human_verdict"]) == norm(r["judge_verdict"]) for r in rows)
    binary = sum((norm(r["human_verdict"]) == "correct") == (norm(r["judge_verdict"]) == "correct")
                 for r in rows)
    return {"n": len(rows), "agreement": agree / len(rows), "agreement_correct_vs_not": binary / len(rows)}


def build(out_dir: Path, plots_dir: Path, qids: list[str]) -> dict:
    runs = load_runs(out_dir, qids)
    systems: dict[str, dict] = {}
    for name, recs in runs.items():
        per_rep = [metrics.aggregate(r["rows"]) for r in recs]
        mean = {}
        for k in per_rep[0]:
            vals = [p[k] for p in per_rep if isinstance(p.get(k), (int, float))]
            mean[k] = statistics.fmean(vals) if vals else None
        var = {k: {"values": [p.get(k) for p in per_rep],
                   "std": (statistics.stdev([p[k] for p in per_rep if p.get(k) is not None])
                           if sum(p.get(k) is not None for p in per_rep) >= 2 else None)}
               for k in VARIANCE_KEYS} if len(per_rep) > 1 else None
        cfg = recs[0].get("run_config") or {}
        if cfg.get("verify") and not (cfg.get("verify_support") and cfg.get("verify_numeric")):
            # Phase 3 rule: with a check switched off an answer cannot be Verified, so these
            # metrics are 0 / undefined by construction, not by measurement
            for k in BY_DESIGN:
                mean[k] = None
        # latency: mean over repeats that measured it (cached rows excluded in aggregate)
        # a repeat counts only if it measured at least half the questions (the rest were
        # served from the cache, so a small n would describe an unrepresentative subset)
        lat_reps = [p for p in per_rep if (p.get("latency_n") or 0) >= p["n"] / 2]
        mean["latency_n"] = sum(p["latency_n"] for p in lat_reps)
        for k in ("latency_mean_s", "latency_p50_s", "latency_p90_s"):
            mean[k] = statistics.fmean(p[k] for p in lat_reps) if lat_reps else None
        rows0 = recs[0]["rows"]
        cats = sorted({r["category"] for r in rows0})
        systems[name] = {
            "description": SYSTEMS[name][1], "repeats": len(recs),
            "complete": all(r["complete"] and len(r["rows"]) == len(qids) for r in recs),
            "metrics": mean, "variance": var, "answer_agreement_across_repeats": _agreement(recs),
            "by_category": {c: metrics.aggregate([r for r in rows0 if r["category"] == c])
                            for c in cats},
            "llm_cache": [r.get("llm_cache") for r in recs],
            "run_config": recs[0].get("run_config"),
        }
    deltas = {}
    if "C" in systems:
        base = systems["C"]["metrics"]
        for a in ABLATIONS:
            if a in systems:
                deltas[a] = {k: (systems[a]["metrics"][k] - base[k]
                                 if systems[a]["metrics"].get(k) is not None and base.get(k) is not None
                                 else None) for k, _, _ in HEADLINE}
    ocr_path = out_dir / "ocr_quality.json"
    summary = {"questions": len(qids), "systems": systems, "ablation_deltas_vs_C": deltas,
               "judge_agreement": judge_agreement(out_dir),
               "ocr_quality": json.loads(ocr_path.read_text()) if ocr_path.is_file() else None}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
    _write_csvs(out_dir, summary, runs)
    (out_dir / "summary.md").write_text(format_summary(summary, markdown=True), encoding="utf-8")
    if systems:
        _plots(plots_dir, summary)
    return summary


def _fmt(v, key: str = "") -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.1f}" if key.endswith("_s") else f"{v:.3f}"
    return str(v)


def _write_csvs(out_dir: Path, summary: dict, runs: dict[str, list[dict]]) -> None:
    names = list(summary["systems"])
    with open(out_dir / "summary.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", *names])
        for k, label, _ in HEADLINE:
            w.writerow([k, *(_fmt(summary["systems"][n]["metrics"].get(k), k) for n in names)])
    with open(out_dir / "by_category.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        keys = ["n", "accuracy", "answered_rate", "verified_rate", "recall@5", "provenance_hit",
                "correct_abstention_rate"]
        w.writerow(["system", "category", *keys])
        for n in names:
            for c, m in summary["systems"][n]["by_category"].items():
                w.writerow([n, c, *(_fmt(m.get(k), k) for k in keys)])
    with open(out_dir / "per_question.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["system", "repeat", "id", "category", "status", "correct", "correct_source",
                    "judge_verdict", "recall@5", "rr", "provenance_hit", "cited_pages",
                    "warnings", "latency_s", "answer"])
        for n, recs in runs.items():
            for rec in recs:
                for r in rec["rows"]:
                    w.writerow([n, rec["repeat"], r["id"], r["category"], r["status"], r["correct"],
                                r["correct_source"], r.get("judge_verdict"),
                                r["retrieval"].get("recall@5"), r["retrieval"].get("rr"),
                                r["provenance"]["provenance_hit"],
                                "; ".join(f"{d} p{p}" for d, p in r["cited_pages"]),
                                "; ".join(x["code"] for x in r["warnings"]), r["latency_s"],
                                r["answer"]])


def format_summary(summary: dict, markdown: bool = False) -> str:
    names = list(summary["systems"])
    if not names:
        return "no runs found"
    lines = []
    if markdown:
        lines += ["# Evaluation summary", "",
                  f"{summary['questions']} questions. Values are measured; n/a = no labelled "
                  "question for that metric. Generated by `python -m eval.run_all`.", ""]
    head = ["metric", *names]
    rows = [[label, *(_fmt(summary["systems"][n]["metrics"].get(k), k) for n in names)]
            for k, label, _ in HEADLINE]
    rows.append(["Latency: questions measured", *(_fmt(summary["systems"][n]["metrics"].get("latency_n"))
                                                  for n in names)])
    rows.append(["repeats", *(str(summary["systems"][n]["repeats"]) for n in names)])
    rows.append(["complete", *(str(summary["systems"][n]["complete"]) for n in names)])
    if markdown:
        lines += ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
        lines += ["| " + " | ".join(r) + " |" for r in rows]
    else:
        width = max(len(r[0]) for r in rows) + 2
        lines.append(head[0].ljust(width) + "".join(h.rjust(14) for h in head[1:]))
        lines += [r[0].ljust(width) + "".join(c.rjust(14) for c in r[1:]) for r in rows]
    var = {n: s for n, s in summary["systems"].items() if s["variance"]}
    if var:
        lines += ["", "Variance across repeats (std; answer agreement = same status and text):"]
        for n, s in var.items():
            parts = [f"{k} {_fmt(v['std'])}" for k, v in s["variance"].items()]
            lines.append(f"- {n} ({s['repeats']} runs): " + ", ".join(parts)
                         + f", answer agreement {_fmt(s['answer_agreement_across_repeats'])}")
    ja = summary.get("judge_agreement")
    lines += ["", "Judge agreement with human spot check: " + (
        f"{_fmt(ja['agreement'])} exact, {_fmt(ja.get('agreement_correct_vs_not'))} correct-vs-not "
        f"(n={ja['n']})" if ja and ja.get("n") else
        "not measured yet (fill human_verdict in spot_check.csv and re-run with --report-only)")]
    oq = summary.get("ocr_quality")
    if oq and oq.get("documents"):
        for d in oq["documents"]:
            lines.append(f"OCR {d['document']}: CER {d['cer_mean']:.2%}, WER {d['wer_mean']:.2%} "
                         f"over {len(d['pages'])} pages")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- plots

def _style(ax, title: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def _plots(plots_dir: Path, summary: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots_dir.mkdir(parents=True, exist_ok=True)
    systems = summary["systems"]
    main = [s for s in MAIN_SYSTEMS if s in systems]
    labels = dict((k, lbl) for k, lbl, _ in HEADLINE)
    better = dict((k, b) for k, _, b in HEADLINE)

    # 1. A/B/C overview: small multiples, one metric per panel (each on its own scale)
    if main:
        keys = [k for k in OVERVIEW if any(systems[s]["metrics"].get(k) is not None for s in main)]
        cols = 4
        nrows = (len(keys) + cols - 1) // cols
        fig, axes = plt.subplots(nrows, cols, figsize=(12, 2.8 * nrows), facecolor=SURFACE,
                                 squeeze=False)
        for ax, k in zip(axes.flat, keys):
            vals = [systems[s]["metrics"].get(k) for s in main]
            xs = range(len(main))
            ax.bar(xs, [v or 0 for v in vals], width=0.55, color=SERIES[0], edgecolor=SURFACE,
                   linewidth=2)
            for x, v in zip(xs, vals):
                ax.text(x, (v or 0), _fmt(v, k), ha="center", va="bottom", fontsize=8, color=INK2)
            ax.set_xticks(list(xs), main)
            top = max([v for v in vals if v is not None] or [1])
            ax.set_ylim(0, top * 1.2 if k.endswith("_s") else max(1.0, top * 1.15))
            _style(ax, labels[k] + ("" if better[k] else " (lower\nis better)"))
        for ax in list(axes.flat)[len(keys):]:
            ax.axis("off")
        fig.suptitle(f"Systems A / B / C ({summary['questions']} questions)", x=0.01, ha="left",
                     color=INK, fontsize=12)
        fig.tight_layout()
        fig.savefig(plots_dir / "systems_overview.png", dpi=150, facecolor=SURFACE)
        plt.close(fig)

    # 2. ablation deltas vs C, coloured by better / worse (direction-aware)
    deltas = summary["ablation_deltas_vs_C"]
    if deltas:
        keys = [k for k in ["accuracy", "verified_rate", "verified_precision",
                            "unsupported_answer_rate", "numeric_value", "recall@5",
                            "provenance_hit", "correct_abstention_rate"]  # rates only: seconds would need their own axis
                if any(d.get(k) is not None for d in deltas.values())]
        # one x-scale for every panel, so bar lengths compare across ablations
        lim = max([abs(v) for d in deltas.values() for k, v in d.items()
                   if k in keys and v is not None] + [0.05]) * 1.6
        fig, axes = plt.subplots(1, len(deltas), figsize=(3.4 * len(deltas), 0.42 * len(keys) + 1.4),
                                 facecolor=SURFACE, sharey=True, squeeze=False)
        for ax, (name, d) in zip(axes[0], deltas.items()):
            vals = [d.get(k) for k in keys]
            colors = [NEUTRAL if not v else (BETTER if (v > 0) == better[k] else WORSE) for k, v in zip(keys, vals)]
            ys = range(len(keys))
            ax.barh(ys, [v or 0 for v in vals], color=colors, height=0.6, edgecolor=SURFACE, linewidth=2)
            for y, v, k in zip(ys, vals, keys):
                ax.text(v or 0, y, " " + ("n/a" if v is None else f"{v:+.2f}") + " ",
                        va="center", ha="left" if (v or 0) >= 0 else "right", fontsize=7, color=INK2)
            ax.axvline(0, color=AXIS, linewidth=1)
            ax.set_yticks(list(ys), [labels[k] for k in keys])
            ax.invert_yaxis()
            ax.set_xlim(-lim, lim)
            _style(ax, name)
            ax.grid(axis="y", visible=False)
            ax.grid(axis="x", color=GRID, linewidth=0.6)
        fig.suptitle("Ablations: change vs C (blue = better, red = worse; n/a for Verified-based "
                     "metrics = cannot be Verified by design)", x=0.01, ha="left", color=INK,
                     fontsize=10)
        fig.tight_layout()
        fig.savefig(plots_dir / "ablation_deltas.png", dpi=150, facecolor=SURFACE)
        plt.close(fig)

    # 3. accuracy by category for A/B/C (three categorical slots, legend + value labels)
    if main:
        cats = sorted({c for s in main for c in systems[s]["by_category"]})
        fig, ax = plt.subplots(figsize=(max(6, 1.6 * len(cats)), 3.6), facecolor=SURFACE)
        w = 0.8 / len(main)
        for i, s in enumerate(main):
            vals = []
            for c in cats:
                m = systems[s]["by_category"].get(c, {})
                vals.append(m.get("correct_abstention_rate") if c == "unanswerable" else m.get("accuracy"))
            xs = [j + (i - (len(main) - 1) / 2) * w for j in range(len(cats))]
            ax.bar(xs, [v or 0 for v in vals], width=w, color=SERIES[i], label=s,
                   edgecolor=SURFACE, linewidth=2)
            for x, v in zip(xs, vals):
                ax.text(x, v or 0, _fmt(v), ha="center", va="bottom", fontsize=7, color=INK2)
        ax.set_xticks(range(len(cats)),
                      [f"{c}\n(n={systems[main[0]]['by_category'].get(c, {}).get('n', 0)})" for c in cats])
        ax.set_ylim(0, 1.15)
        _style(ax, "Accuracy by category (unanswerable: correct abstention)")
        ax.legend(frameon=False, fontsize=8, labelcolor=INK2, ncol=len(main), loc="upper right")
        fig.tight_layout()
        fig.savefig(plots_dir / "accuracy_by_category.png", dpi=150, facecolor=SURFACE)
        plt.close(fig)
