"""Process PDFs into canonical chunk JSONL without building an index.

  python -m trustqa.processing [PDF ...] [--out DIR] [--baseline-b] [--no-ocr]

With no PDFs, every PDF under config.raw_pdfs_dir (sorted by path, as the question CLI
does) is processed. Output (default: data/processed/<key>/, see
process.processed_key; the question CLI writes to the same place):
  chunks.jsonl     one canonical chunk record per line (PRD 3.2)
  documents.json   per-document report: pages, OCR pages + confidence, tables, warnings
  run_config.json  settings used (NFR-04)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from trustqa.config import ConfigError, load_config, log_run_config
from trustqa.processing.process import (
    count_by_type,
    process_pdfs_detailed,
    processed_key,
    write_processed,
)


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(prog="python -m trustqa.processing", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pdfs", nargs="*", type=Path)
    ap.add_argument("--out", type=Path, help="output directory")
    ap.add_argument("--baseline-b", action="store_true", help="flatten tables (system B)")
    ap.add_argument("--no-ocr", action="store_true", help="disable the OCR fallback")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(message)s", stream=sys.stderr)
    try:
        overrides: dict = {}
        if args.baseline_b:
            overrides["baseline_b"] = True
        if args.no_ocr:
            overrides["ocr_enabled"] = False
        config = load_config(**overrides)
        raw = Path(config.raw_pdfs_dir)
        pdfs = args.pdfs or sorted(raw.rglob("*.pdf"), key=lambda p: p.relative_to(raw).as_posix())
        if not pdfs:
            raise ValueError(f"no PDFs found under {raw}")
        result = process_pdfs_detailed(pdfs, config)
        record = log_run_config(config, extra={"system": config.system_name,
                                               "processing_signature": result.signature,
                                               "inputs": [str(p) for p in pdfs]})
        out = args.out or Path(config.processed_dir) / processed_key(result, config)
        write_processed(result, out, record)
    except (ConfigError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    for r in result.documents:
        s = r.summary()
        conf = f" (mean conf {s['ocr_mean_confidence']})" if s["ocr_pages"] else ""
        print(f"{s['document_id']} {s['document_name']}: {s['pages']} pages, "
              f"OCR pages {s['ocr_pages'] or 'none'}{conf}, {s['tables']} tables, "
              f"{s['text_chunks']} text + {s['table_chunks']} table chunks")
        for w in s["warnings"]:
            print(f"    warning: {w}")
    counts = count_by_type(result.chunks)
    print(f"{len(result.chunks)} chunks ({counts}) -> {out / 'chunks.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
