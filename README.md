# TrustQA

Multi-document RAG with source-verified answers for scanned and tabular PDFs.
Returns answer + citations + status (Verified / Partially Supported / Insufficient Evidence) + warnings.

Spec: `docs/TrustQA_PRD_6_Pages.pdf` | Build plan: `phases.md`

## Layout
- `src/trustqa/processing` - PDF text, OCR fallback, tables, chunking
- `src/trustqa/retrieval` - embeddings, FAISS, optional BM25 hybrid
- `src/trustqa/verification` - generation, citation + claim-support + numeric checks
- `src/trustqa/pipeline.py` - `answer_question()` integration
- `app/` - Streamlit UI
- `eval/` - question set, baselines, metrics, plots
- `data/` - PDFs and generated artifacts (git-ignored)

## Setup
1. Install Tesseract OCR (Windows: UB-Mannheim build; set `TESSERACT_CMD` if not on PATH).
2. Use Python 3.11/3.12 (faiss/torch wheels): `py -3.12 -m venv .venv && .venv\Scripts\activate`
3. `pip install -r requirements.txt` (exact versions used for baseline A: `requirements-lock.txt`),
   then `pip install -e . --no-deps` so `python -m trustqa.cli` can import the package.
4. LLM (free): install [Ollama](https://ollama.com) and run `ollama pull qwen2.5:3b`.
   Optional stronger model: free key from console.groq.com, then set `LLM_PROVIDER=groq`,
   `LLM_MODEL=llama-3.3-70b-versatile`, `GROQ_API_KEY=...` in `.env`.
5. `copy .env.example .env` and set `TESSERACT_CMD` if Tesseract is not on PATH.
6. `pytest`

## Quickstart (Phase 1 baseline)
```
python -m trustqa.cli "Which retriever is RAG's retriever initialized from?"
python -m trustqa.cli "question" --docs a.pdf b.pdf --show-passages --json
```
With no `--docs`, every PDF under `data/raw_pdfs/` is used. The first run builds and caches the
index under `data/indexes/`. Document names and pages shown come from chunk metadata, never from the LLM.
Baseline system A's frozen config: `eval/configs/system_A.json`.

## Processing, OCR and tables (Phase 2)
```
python -m trustqa.processing                      # all PDFs under data/raw_pdfs -> data/processed/<key>/
python -m trustqa.processing a.pdf --baseline-b   # system B: OCR on, tables flattened
python scripts/inspect_page.py data/raw_pdfs/tables/beir_thakur2021.pdf 7   # page image beside text/tables
python -m trustqa.cli "What nDCG@10 does BM25 achieve on TREC-COVID?" --baseline-b
```
`data/processed/<key>/` holds `chunks.jsonl` (one canonical chunk per line), `documents.json`
(pages, OCR pages + confidence, tables, warnings per document) and `run_config.json`.
Scanned/garbage pages are OCR'd with Tesseract (`OCR_*` settings); tables on digital pages become
one chunk per row (`DOC02_P7_T1_R3`) whose text carries the caption, units and every
`header=value` pair, with the structured row kept in `table_data`.
Systems: A = `OCR_ENABLED=0 FLATTEN_TABLES=1 VERIFY=0` (`eval/configs/system_A.json`),
B = `BASELINE_B=1` (`eval/configs/system_B.json`), C = defaults.
