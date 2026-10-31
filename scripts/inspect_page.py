"""Manual QA: show one page's extracted text and tables next to the page image.

  python scripts/inspect_page.py PDF PAGE [--out FILE.html] [--baseline-b] [--no-ocr]
  python scripts/inspect_page.py PDF PAGE --chunks data/processed/<key>/chunks.jsonl

Writes a self-contained HTML file (default: data/processed/inspect/<pdf>_p<PAGE>.html):
left, the rendered page with detected table regions outlined in red; right, the page
report (method, quality, OCR confidence), every text chunk with its metadata, OCR words
below --low-conf highlighted, and each table rebuilt from its row chunks (headers, rows,
units, caption) plus the exact row text that gets embedded.

By default the page is processed live with the current .env settings. With --chunks the
records are read from an existing JSONL instead (no table outlines in that mode, since
table positions are not part of the chunk record).
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import sys
from collections import OrderedDict
from pathlib import Path

import pymupdf
from PIL import ImageDraw

from trustqa.config import load_config
from trustqa.processing.extract import DocumentRegistry
from trustqa.processing.ocr import OcrPage, render_page
from trustqa.processing.process import process_document, read_jsonl
from trustqa.schemas import Chunk, ContentType

CSS = """
:root { color-scheme: light dark; --bg:#fff; --fg:#1b1b1b; --muted:#666; --line:#ddd;
  --card:#f6f6f4; --mark:#ffe08a; --bad:#c0392b; }
@media (prefers-color-scheme: dark) { :root { --bg:#161616; --fg:#e8e8e8; --muted:#9a9a9a;
  --line:#333; --card:#202020; --mark:#7a5d00; --bad:#ff7b6b; } }
body { margin:0; font:14px/1.45 system-ui, sans-serif; background:var(--bg); color:var(--fg); }
header { padding:12px 16px; border-bottom:1px solid var(--line); }
header h1 { font-size:16px; margin:0 0 4px; } header p { margin:0; color:var(--muted); }
main { display:flex; gap:16px; padding:16px; align-items:flex-start; }
.img { flex:0 0 48%; position:sticky; top:8px; }
.img img { width:100%; border:1px solid var(--line); background:#fff; }
.data { flex:1; min-width:0; }
.card { background:var(--card); border:1px solid var(--line); border-radius:6px;
  padding:10px 12px; margin-bottom:12px; overflow-x:auto; }
.meta { color:var(--muted); font-size:12px; margin-bottom:4px; }
.txt { white-space:pre-wrap; word-break:break-word; font-family:ui-monospace,monospace; font-size:12.5px; }
mark { background:var(--mark); color:inherit; }
table { border-collapse:collapse; font-size:12.5px; }
th, td { border:1px solid var(--line); padding:3px 6px; text-align:left; vertical-align:top; }
th { background:var(--bg); }
.warn { color:var(--bad); }
@media (max-width: 900px) { main { flex-direction:column; } .img { position:static; flex-basis:auto; width:100%; } }
"""


def page_png(pdf: Path, page_number: int, dpi: int, boxes: list[tuple]) -> str:
    with pymupdf.open(pdf) as doc:
        img = render_page(doc[page_number - 1], dpi).convert("RGB")
    draw = ImageDraw.Draw(img)
    s = dpi / 72
    for x0, top, x1, bottom in boxes:
        draw.rectangle([x0 * s - 3, top * s - 3, x1 * s + 3, bottom * s + 3],
                       outline=(220, 30, 30), width=3)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


def ocr_html(ocr: OcrPage, low_conf: float) -> str:
    out, pos = [], 0
    for w in ocr.words:
        out.append(html.escape(ocr.text[pos:w.start]))
        word = html.escape(ocr.text[w.start:w.end])
        out.append(f'<mark title="conf {w.confidence:.0f}">{word}</mark>'
                   if w.confidence < low_conf else word)
        pos = w.end
    out.append(html.escape(ocr.text[pos:]))
    return "".join(out)


def tables_html(chunks: list[Chunk]) -> str:
    groups: OrderedDict[str, list[Chunk]] = OrderedDict()
    for c in chunks:
        groups.setdefault(c.table_data.table_id, []).append(c)
    parts = []
    for tid, rows in groups.items():
        td = rows[0].table_data
        head = "".join(f"<th>{html.escape(h)}</th>" for h in td.headers)
        body = "".join(
            "<tr>" + "".join(f"<td>{html.escape(v)}</td>" for v in c.table_data.rows[0]) + "</tr>"
            for c in rows)
        parts.append(
            f'<div class="card"><div class="meta"><b>{html.escape(tid)}</b> '
            f'({html.escape(rows[0].section or "no caption label")}) - {len(rows)} row chunks, '
            f'{len(td.headers)} columns, units: {html.escape(td.units or "none")}</div>'
            f'<div class="meta">caption: {html.escape(td.caption or "-")}</div>'
            f"<table><tr>{head}</tr>{body}</table>"
            f'<div class="meta" style="margin-top:6px">embedded text of {html.escape(rows[0].chunk_id)}:</div>'
            f'<div class="txt">{html.escape(rows[0].text)}</div></div>')
    return "".join(parts)


def build_html(pdf: Path, page_number: int, chunks: list[Chunk], page_info: dict | None,
               ocr: OcrPage | None, boxes: list[tuple], dpi: int, low_conf: float,
               source: str) -> str:
    text_chunks = [c for c in chunks if c.content_type is ContentType.TEXT]
    table_chunks = [c for c in chunks if c.content_type is ContentType.TABLE]
    info = ""
    if page_info:
        info = ", ".join(f"{k}={v}" for k, v in page_info.items())
        info = f'<div class="card"><div class="meta">page report</div>{html.escape(info)}</div>'
    ocr_block = ""
    if ocr is not None:
        ocr_block = (f'<div class="card"><div class="meta">OCR text, mean confidence '
                     f'{ocr.mean_confidence}; words below {low_conf:.0f} highlighted '
                     f'(hover for confidence)</div><div class="txt">{ocr_html(ocr, low_conf)}'
                     f"</div></div>")
    tc = "".join(
        f'<div class="card"><div class="meta"><b>{html.escape(c.chunk_id)}</b> - '
        f"{c.extraction_method.value}, ocr_used={c.ocr_used}, "
        f"ocr_confidence={c.ocr_confidence}, {len(c.text)} chars</div>"
        f'<div class="txt">{html.escape(c.text)}</div></div>' for c in text_chunks)
    if not chunks:
        tc = '<div class="card warn">No chunks extracted from this page.</div>'
    title = f"{pdf.name} - page {page_number}"
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Page inspection</title><style>{CSS}</style></head><body>
<header><h1>{html.escape(title)}</h1><p>{html.escape(source)} - {len(text_chunks)} text chunks,
{len(table_chunks)} table row chunks</p></header>
<main><div class="img"><img alt="page {page_number}" src="data:image/png;base64,{page_png(pdf, page_number, dpi, boxes)}"></div>
<div class="data">{info}{ocr_block}<h3>Tables</h3>{tables_html(table_chunks) or '<div class="card meta">none detected</div>'}
<h3>Text chunks</h3>{tc}</div></main></body></html>"""


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pdf", type=Path)
    ap.add_argument("page", type=int, help="1-indexed page number")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--chunks", type=Path, help="read records from this chunks.jsonl")
    ap.add_argument("--baseline-b", action="store_true")
    ap.add_argument("--no-ocr", action="store_true")
    ap.add_argument("--dpi", type=int, default=110, help="image resolution in the report")
    ap.add_argument("--low-conf", type=float, default=60.0)
    args = ap.parse_args(argv)

    overrides: dict = {}
    if args.baseline_b:
        overrides["baseline_b"] = True
    if args.no_ocr:
        overrides["ocr_enabled"] = False
    config = load_config(**overrides)
    with pymupdf.open(args.pdf) as doc:
        if not 1 <= args.page <= len(doc):
            print(f"error: page must be 1..{len(doc)}", file=sys.stderr)
            return 2

    ocr, boxes, page_info = None, [], None
    if args.chunks:
        chunks = [c for c in read_jsonl(args.chunks)
                  if c.document_name == args.pdf.name and c.page_number == args.page]
        source = f"records from {args.chunks}"
    else:
        entry, _ = DocumentRegistry().register(args.pdf)
        chunks, report, tables = process_document(entry, config, pages={args.page})
        ocr = report.ocr_results.get(args.page)
        boxes = [t.bbox for t in tables]
        pr = report.pages[0]
        page_info = {"method": pr.method, "text_layer_quality": pr.quality,
                     "alnum_chars": pr.alnum_chars, "nonalnum_ratio": pr.nonalnum_ratio,
                     "ocr_attempted": pr.ocr_attempted, "ocr_confidence": pr.ocr_confidence,
                     "ocr_rejected": pr.ocr_rejected, "tables": pr.tables}
        source = f"processed live ({config.system_name}, preprocess={config.ocr_preprocess})"

    out = args.out or (Path(config.processed_dir) / "inspect"
                       / f"{args.pdf.stem}_p{args.page}.html")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_html(args.pdf, args.page, chunks, page_info, ocr, boxes, args.dpi,
                              args.low_conf, source), encoding="utf-8")
    print(f"{len(chunks)} chunks on page {args.page} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
