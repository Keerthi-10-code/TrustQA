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
