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
3. `pip install -r requirements.txt`
4. LLM (free): install [Ollama](https://ollama.com) and run `ollama pull qwen2.5:3b`.
   Optional stronger model: free key from console.groq.com, then set `LLM_PROVIDER=groq`,
   `LLM_MODEL=llama-3.3-70b-versatile`, `GROQ_API_KEY=...` in `.env`.
5. `copy .env.example .env` and set `TESSERACT_CMD` if Tesseract is not on PATH.
6. `pytest`
