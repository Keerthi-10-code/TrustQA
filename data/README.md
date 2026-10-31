# Data

PDFs and generated artifacts are git-ignored. This file records where every PDF came from
so the collection can be rebuilt (NFR-04). Check each licence before redistributing.

## Pilot set (Phase 0-1)

| File | Type | Source | Licence / notes |
|---|---|---|---|
| `raw_pdfs/text/rag_lewis2020.pdf` | digital text | https://arxiv.org/abs/2005.11401 (Lewis et al., 2020, RAG) | arXiv; check paper licence |
| `raw_pdfs/text/selfrag_asai2023.pdf` | digital text, related to RAG paper | https://arxiv.org/abs/2310.11511 (Asai et al., 2023, Self-RAG) | arXiv; check paper licence |
| `raw_pdfs/tables/beir_thakur2021.pdf` | table-heavy | https://arxiv.org/abs/2104.08663 (Thakur et al., 2021, BEIR) | arXiv; check paper licence |
| `raw_pdfs/scanned/transformer_vaswani2017_scanned.pdf` | synthetic scan, pages 1-9 | https://arxiv.org/abs/1706.03762 (Vaswani et al., 2017) | Paper grants reproduction of tables/figures for scholarly use; scan made locally |

The two RAG papers are related (multi-document questions). The scanned paper includes a
results table (BLEU scores, page 8), so it also exercises tables on OCR pages.

## Synthetic scans and OCR ground truth

`transformer_vaswani2017_scanned.pdf` is image-only (no text layer), produced by:

```
python scripts/make_scanned_pdf.py data/ocr_ground_truth/transformer_vaswani2017_source.pdf \
  data/raw_pdfs/scanned/transformer_vaswani2017_scanned.pdf --pages 1-9 \
  --truth-dir data/ocr_ground_truth/transformer_vaswani2017
```

Defaults: 150 DPI grayscale, up to 0.6 degrees skew, noise std 8, JPEG quality 70, seed 0.
`ocr_ground_truth/<name>/page_NNN.txt` holds the original digital text per page, used for
CER/WER in Phase 6. Real scanned documents should be added for the final dataset; synthetic
scans are cleaner than real ones.

## Layout
- `raw_pdfs/{text,scanned,tables}/` - input PDFs
- `ocr_ground_truth/` - source PDFs and per-page reference text for synthetic scans
- `processed/` - canonical chunk JSONL (Phase 2)
- `indexes/` - FAISS indexes (Phase 1)
