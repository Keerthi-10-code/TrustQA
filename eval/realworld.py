"""Real-world check on documents never used during development:  python -m eval.realworld

Downloads two public papers (DPR, Karpukhin et al. 2020; ColBERT, Khattab & Zaharia 2020) into
data/realworld/, makes an image-only scanned copy of DPR pages 1-6 (scripts/make_scanned_pdf.py),
and asks the questions in eval/realworld_questions.jsonl through the same code path as the app
(trustqa.corpus). Sets: "unseen" (DPR + ColBERT), "scanned" (the scanned DPR copy alone, OCR),
"mixed" (the 4 pilot PDFs + DPR + ColBERT).

Expected answers were read from the PDFs' own text (the "evidence" field gives page and passage).
Scoring: an answerable question is correct when the status is not Insufficient Evidence, every
expected pattern is in the answer and (when pages are given) a citation is on an expected page;
an unanswerable one is correct when the status is Insufficient Evidence. Writes
eval/results/realworld/results.json and summary.md.

  python -m eval.realworld                 # system C
  python -m eval.realworld --systems C,C-v0.4,B
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
import urllib.request
from pathlib import Path

from eval.questions import ROOT
from eval.systems import config_for
from trustqa.corpus import prepare_corpus

DATA = ROOT / "data" / "realworld"
PAPERS = {"dpr_karpukhin2020.pdf": "https://arxiv.org/pdf/2004.04906v3",
          "colbert_khattab2020.pdf": "https://arxiv.org/pdf/2004.12832v2"}
QUESTIONS = ROOT / "eval" / "realworld_questions.jsonl"
OUT = ROOT / "eval" / "results" / "realworld"


def fetch() -> dict[str, Path]:
    DATA.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, url in PAPERS.items():
        p = DATA / name
        if not p.is_file():
            print(f"downloading {url}", flush=True)
            urllib.request.urlretrieve(url, p)
        paths[name] = p
    scan = DATA / "scanned" / "dpr_scanned.pdf"
    if not scan.is_file():
        sys.path.insert(0, str(ROOT / "scripts"))
        from make_scanned_pdf import main as make_scan  # noqa: E402

        make_scan([str(paths["dpr_karpukhin2020.pdf"]), str(scan), "--pages", "1-6",
                   "--truth-dir", str(DATA / "scanned" / "truth")])
    paths["scan"] = scan
    return paths


def page_ok(result, pages) -> bool:
    if not pages:
        return True
    cited = {(c.document_name.lower(), c.page_number) for c in result.citations}
    return any(sub in d and p == pg for d, p in cited for sub, pg in pages)


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--systems", default="C")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)
    paths = fetch()
    pilot = sorted((ROOT / "data" / "raw_pdfs").rglob("*.pdf"))
    unseen = [paths["dpr_karpukhin2020.pdf"], paths["colbert_khattab2020.pdf"]]
    doc_sets = {"unseen": unseen, "scanned": [paths["scan"]], "mixed": pilot + unseen}
    qs = [json.loads(line) for line in QUESTIONS.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = []
    for system in [s.strip() for s in args.systems.split(",") if s.strip()]:
        cfg = config_for(system)
        for set_name, docs in doc_sets.items():
            t = time.time()
            corpus = prepare_corpus(docs, cfg, system="C" if system.startswith("C") else system)
            print(f"\n== {system} | {set_name}: {len(corpus.ready)} documents ready "
                  f"({time.time() - t:.1f} s)", flush=True)
            for q in [q for q in qs if q["set"] == set_name]:
                a = corpus.ask(q["question"])
                r = a.result
                ie = r.status.value == "Insufficient Evidence"
                if q["expect"]:
                    hits = [bool(re.search(p, r.answer)) for p in q["expect"]]
                    ok = (not ie) and all(hits) and page_ok(r, q["pages"])
                else:
                    hits, ok = [], ie
                rows.append({"system": system, "set": set_name, "id": q["id"], "question": q["question"],
                             "status": r.status.value, "correct": ok, "answer": r.answer,
                             "cited": [[c.document_name, c.page_number] for c in r.citations],
                             "warnings": [w.message for w in r.warnings], "seconds": a.seconds})
                print(f"{q['id']} {'OK ' if ok else 'BAD'} {r.status.value:22} {a.seconds:5.1f}s "
                      f"{r.answer[:110]!r}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    done = {r["system"] for r in rows}  # merge with earlier runs of other systems
    if (OUT / "results.json").is_file():
        old = json.loads((OUT / "results.json").read_text(encoding="utf-8"))
        rows = [r for r in old if r["system"] not in done] + rows
    (OUT / "results.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
    lines = ["# Real-world check (documents not used during development)", "",
             "| system | set | correct | Verified | Verified but wrong | refused correctly |",
             "|---|---|---|---|---|---|"]
    for system in dict.fromkeys(r["system"] for r in rows):
        for set_name in [*doc_sets, "all"]:
            rs = [r for r in rows if r["system"] == system and (set_name == "all" or r["set"] == set_name)]
            if not rs:
                continue
            una = [r for r in rs if not [q for q in qs if q["id"] == r["id"]][0]["expect"]]
            lines.append(f"| {system} | {set_name} | {sum(r['correct'] for r in rs)}/{len(rs)} | "
                         f"{sum(r['status'] == 'Verified' for r in rs)} | "
                         f"{sum(r['status'] == 'Verified' and not r['correct'] for r in rs)} | "
                         f"{sum(r['correct'] for r in una)}/{len(una)} |")
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
