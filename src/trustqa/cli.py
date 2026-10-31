"""Command line: python -m trustqa.cli "question" [--docs a.pdf b.pdf] [--json] [--baseline-b]

With no --docs, every PDF under config.raw_pdfs_dir (recursively, sorted by path) is used.
Document/page in the output come from chunk metadata, never from the LLM.
--baseline-b reproduces baseline system B (OCR + plain RAG: tables flattened, no
verification); the same as BASELINE_B=1 in .env.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from trustqa.config import ConfigError, load_config
from trustqa.llm import LLMError
from trustqa.pipeline import answer_question, get_or_build_index
from trustqa.retrieval.search import retrieve
from trustqa.schemas import Result


def default_docs(raw_dir: Path) -> list[Path]:
    return sorted(Path(raw_dir).rglob("*.pdf"), key=lambda p: p.relative_to(raw_dir).as_posix())


def format_result(result: Result) -> str:
    lines = [f"Question: {result.question}", "", f"Status: {result.status.value}", "",
             "Answer:", result.answer, ""]
    if result.citations:
        lines.append("Sources:")
        for c in result.citations:
            lines.append(f"  - {c.document_name}, page {c.page_number} "
                         f"[{', '.join(c.chunk_ids)}]")
        lines.append("")
    if result.warnings:
        lines.append("Warnings:")
        lines.extend(f"  - [{w.code.value}] {w.message}" for w in result.warnings)
    return "\n".join(lines).rstrip()


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):  # Windows consoles default to cp1252
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser(prog="python -m trustqa.cli", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("question")
    ap.add_argument("--docs", nargs="+", type=Path, help="PDF files (default: data/raw_pdfs)")
    ap.add_argument("--top-k", type=int, help="override TOP_K")
    ap.add_argument("--json", action="store_true", help="print the Result as JSON")
    ap.add_argument("--show-passages", action="store_true",
                    help="also print the retrieved passages and scores")
    ap.add_argument("--rebuild", action="store_true", help="ignore the cached index")
    ap.add_argument("--baseline-b", action="store_true",
                    help="baseline system B: OCR + plain RAG (tables flattened, no verification)")
    ap.add_argument("-v", "--verbose", action="store_true", help="log progress and run config")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(message)s", stream=sys.stderr)
    try:
        overrides = {"top_k": args.top_k} if args.top_k is not None else {}
        if args.baseline_b:
            overrides["baseline_b"] = True
        config = load_config(**overrides)
        docs = args.docs or default_docs(config.raw_pdfs_dir)
        if not docs:
            raise ValueError(f"no PDFs found under {config.raw_pdfs_dir}; pass --docs")
        if args.rebuild:
            get_or_build_index(docs, config, rebuild=True)
        result = answer_question(args.question, docs, config)
        if args.show_passages:
            index = get_or_build_index(docs, config)
            print("Retrieved passages:")
            for r in retrieve(args.question, config.top_k, index):
                c = r.chunk
                print(f"  {r.rank}. {c.chunk_id} ({c.document_name} p{c.page_number}) "
                      f"score={r.score:.3f}: {c.text[:160]}")
            print()
    except (ConfigError, LLMError, ValueError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(result.to_json(indent=2) if args.json else format_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
