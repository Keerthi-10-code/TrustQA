# TrustQA - final report outline

Draft structure for the final report and slides. Numbers come only from `eval/results/`
(regenerate with `python -m eval.run_all`); citations marked **VERIFY CITATION** must be
checked against the original papers before submission.

## 1. Problem
- Document collections mix digital text, scans and tables; facts may span several documents
  (PRD 1.1).
- Plain RAG failure modes: OCR noise reaches retrieval and generation; flattened tables lose
  row/column meaning; dense retrieval misses exact names/values; the model makes unsupported
  claims; a model-written citation does not prove the page supports the answer.
- Goal: answers a reader can check - source document, 1-indexed page, supporting quote and a
  verification status (Verified / Partially Supported / Insufficient Evidence), with warnings.

## 2. Related work (placeholders - VERIFY CITATION for every entry)
- Retrieval-augmented generation: Lewis et al., 2020, "Retrieval-Augmented Generation for
  Knowledge-Intensive NLP Tasks" (NeurIPS). **VERIFY CITATION**
- Dense passage retrieval: Karpukhin et al., 2020, "Dense Passage Retrieval for Open-Domain
  Question Answering" (EMNLP). **VERIFY CITATION**
- Zero-shot retrieval benchmark, BM25 as a robust baseline: Thakur et al., 2021, BEIR
  (NeurIPS Datasets and Benchmarks). **VERIFY CITATION**
- Self-reflective RAG (retrieve on demand, critique tokens): Asai et al., 2023, Self-RAG.
  **VERIFY CITATION**
- Late-interaction retrieval: Khattab & Zaharia, 2020, ColBERT (SIGIR). **VERIFY CITATION**
- Attribution / citation verification and NLI-based fact checking for generated text - choose
  2-3 papers the team has read. **VERIFY CITATION**
- OCR: Tesseract (Smith, 2007). **VERIFY CITATION**

## 3. Design (what was built, phases.md 0-6)
- Data contracts frozen in Phase 0 (`schemas.py`): canonical chunk, citation, result.
- Processing: PyMuPDF text; per-page scanned/garbage detection -> Tesseract OCR with per-word
  confidence; pdfplumber + rule-based table reconstruction -> one row-aware chunk per table row
  with structured `table_data`; sentence-aware chunking that never crosses pages; section
  heading paths per chunk (`SECTION_HEADINGS`).
- Retrieval: MiniLM embeddings + FAISS, BM25 fused by reciprocal rank fusion (hybrid),
  entity-scoped slots for questions naming specific documents.
- Generation: answer only from passages, cite chunk IDs, abstain when insufficient.
- Verification (the core contribution, all checks deterministic unless stated):
  provenance (cited IDs must be retrieved; document/page from metadata), claim decomposition,
  LLM support judge whose verbatim quote is validated in Python, numeric/unit/period checks with
  table-cell alignment and Python-evaluated calculations, cross-document conflicts,
  OCR-uncertainty caps, abstention gates, document-scope check (answer must cite the document
  the question names), answer-type check, status rules (PRD 2.3).
- Systems compared: A plain-text RAG, B OCR + plain RAG, C TrustQA; ablations remove one
  component each.
- Demo: Streamlit app with evidence panel (page image with the quote highlighted, full table
  with the cited row) and side-by-side baseline comparison.

## 4. Evaluation setup
- Question set: `eval/questions.jsonl` (target 50: 10 direct, 10 multi-doc, 10 table/numeric,
  10 insufficient-evidence, 5 conflicting, 5 scanned) - **to be labelled by the team**; until
  then the 38-question dev set (`eval/dev_questions.jsonl`, written during development, likely
  favouring the system) is reported and labelled as such.
- Metrics (`eval/metrics.py`): answer accuracy (LLM judge + human spot check), EM / token F1,
  numeric value/unit/period, Recall@K / Precision@K / MRR, provenance and support reported
  separately, unsupported-answer rate, correct- and over-abstention, conflict recall, CER/WER,
  latency. Same LLM (qwen2.5:3b, temperature 0) and embeddings for every system; 3 runs each.

## 5. Results
- Table: systems A / B / C (`eval/results/<set>/summary.md`), plots in `eval/plots/<set>/`.
- Ablations: change vs C per component (`ablation_deltas.png`).
- OCR quality: CER/WER on the scanned pilot paper.
- Real-world check on documents never seen in development (README "Real-world test").
- Judge agreement with the human spot check (`spot_check.csv`) - required before reporting
  judge-based accuracy as final.

## 6. Failure analysis (10-15 cases from `eval/results/<set>/error_analysis.md`)
- Verified but incomplete or not answering the question (F10); answers about the wrong
  document (F13) and how the scope check withholds them; correct answers withheld (F14);
  multi-document evidence missing (F12); OCR symbol garbling (F1); table layout edge cases (F6).
- For each: what happened, why, what changed, and the measured effect (`failures.md`).

## 7. Limitations
- Small local LLM (3B) as generator and judge: incomplete answers, judge errors; a stronger
  model is a configuration change (Groq / Anthropic) and should be measured.
- Synthetic scans are cleaner than real ones; tables inside scans are not reconstructed (F2).
- Dev-set results are not a held-out evaluation; the final set must be labelled independently
  and double-checked.
- Verified means "every claim is supported by a quoted passage of the cited document", not
  "complete" or "true in the world".

## 8. Future work
- Stronger judge/generator and measured judge agreement; table reconstruction on OCR pages;
  per-claim completeness against the question; multi-document decomposition with a stronger
  model; real scanned documents and period-heavy financial tables.
