# TrustQA - Solo Build Plan

Goal: a runnable prototype returning **answer + evidence + source doc + page + verification status**, with a measured comparison against basic RAG.

Principles
- Thin end-to-end slice first, then deepen. Never have a broken pipeline for more than a day.
- Frozen contracts (chunk schema, result schema, function signatures) - change only deliberately.
- Every phase ends with passing tests and a git commit/tag. No phase is "done" without its exit check.
- Do not invent scores. Report measured results and failure examples.
- 1-indexed PDF pages everywhere. Document/page shown to users always comes from metadata, never from LLM output.

How to use the prompts: each phase has a **Claude prompt** block. Start a fresh Claude Code session per phase, paste that block, review the diff and test output, then commit/tag yourself. Keep prompts to one phase at a time (Phase 3: one numbered item at a time).

Rough solo timeline: ~7-9 weeks part-time. Phases 0-1 are the foundation; 6 is where the project's value is demonstrated.

---

## Phase 0 - Foundations and contracts (2-3 days)
Deliverables
- [x] Git repo initialised, venv, deps installed, Tesseract working (`pytesseract.get_tesseract_version()`).
- [x] `src/trustqa/schemas.py`: dataclasses/TypedDicts for `Chunk`, `Citation`, `Result`, `Status` enum. Fields exactly per PRD 3.2 / 3.4 (+ `extraction_method`, `ocr_used`, `section`, `ocr_confidence`).
- [x] `src/trustqa/config.py`: loads `.env`, holds chunk size/overlap, top_k, model names; one place to log run config (NFR-04).
- [x] Function stubs with final signatures: `process_pdfs`, `build_index`, `retrieve`, `generate_and_verify`, `answer_question`.
- [x] Collect 3-5 pilot PDFs (1 digital, 1 scanned, 1 table-heavy, 1-2 related to each other) into `data/raw_pdfs/`. Sources in `data/README.md`.
- [x] LLM wrapper `llm.py` with a single `complete(prompt, system, json=False)` so provider is swappable; add a fake/mock LLM for tests. Providers: Ollama (local, default), Groq (free key), Anthropic (optional).

Phase 0 notes: Python 3.12 venv (3.14 lacks faiss/torch wheels). Local `qwen2.5:3b` runs 100% on the RTX 3050 (~3 s/answer warm) but put passage text in `chunk_ids` on the first smoke test, so Phase 1 prompts need an explicit ID format and FR-09 must reject such citations.

Exit check: `pytest` runs (even with trivial tests); schema round-trips to JSON; one LLM call works.

**Claude prompt - Phase 0**
```text
Read README.md, phases.md and docs/TrustQA_PRD_6_Pages.pdf. We are implementing Phase 0 only (foundations and contracts). Do not build any retrieval, OCR or verification logic.

Do:
1. Create src/trustqa/schemas.py with dataclasses for Chunk, Citation, Result and a Status enum (Verified, Partially Supported, Insufficient Evidence). Fields exactly per PRD 3.2/3.4 plus extraction_method, ocr_used, section, ocr_confidence. Add to_dict/from_dict and JSON round-trip helpers. Pages are 1-indexed.
2. Create src/trustqa/config.py: load .env, hold chunk size/overlap, top_k, model names, paths; add a function that logs the full run config.
3. Add stubs with final signatures and docstrings (raise NotImplementedError): process_pdfs, build_index, retrieve, generate_and_verify, answer_question.
4. Create src/trustqa/llm.py with complete(prompt, system=None, json=False), a real provider implementation, and a FakeLLM for tests (scripted responses).
5. Add tests: schema JSON round-trip, config loading, FakeLLM behaviour.
6. Verify pytesseract.get_tesseract_version() works and report the result.

Constraints: contracts are frozen after this phase, so keep them minimal and explicit. Run pytest and show the output. Do not commit; tell me what to review.
```

## Phase 1 - Baseline end-to-end slice (1 week)
Digital PDF -> chunks -> embeddings -> retrieval -> grounded answer with page source.
- [x] `processing/extract.py`: PyMuPDF page-by-page text; stable `document_id` registry (DOC01...); record filename + page.
- [x] `processing/chunking.py`: configurable size/overlap, sentence-aware splits, never cross page boundaries; unique `chunk_id` = `DOC01_P4_C1`.
- [x] `retrieval/index.py`: Sentence-Transformers embeddings (normalised) + FAISS inner-product index; metadata stored aligned; save/load index to `data/indexes/`.
- [x] `retrieval/search.py`: `retrieve(question, top_k)` returns original chunk records with scores.
- [x] `generation.py`: prompt that forces answer from supplied passages only, cites chunk IDs, abstains if insufficient; JSON output.
- [x] `pipeline.answer_question()` wiring it all; CLI script `python -m trustqa.cli "question"`.
- [x] Tests: chunk IDs unique, page numbers 1-indexed, retrieval returns known chunk for a known question.

Exit check: ask 5 questions over pilot digital PDFs; answers cite correct pages. **Tag `v0.1-baseline`.** This is also baseline system A for evaluation - freeze its config.

Phase 1 notes: frozen config for baseline system A is `eval/configs/system_A.json` (settings, code versions, Ollama model digest, document set with SHA-256 and DOC IDs) plus `requirements-lock.txt`. Status policy for A: cited answers are Partially Supported, never Verified (no support check until Phase 3); answers with no valid retrieved chunk ID are withheld as Insufficient Evidence. DOC IDs follow input order (CLI sorts by path, so adding PDFs renumbers them) - label eval questions by filename + page, not DOC ID. Run `pip install -e . --no-deps` once so `python -m trustqa.cli` works.

Exit-check results (qwen2.5:3b, 4 pilot PDFs, 6 questions): every displayed page came from metadata and matched the cited chunk. Answers: 2/4 answerable correct (RAG retriever init from DPR, p5; BEIR "19 datasets", p20), 1 incomplete (Self-RAG reflection tokens, p17: listed only "Retrieve"), 1 wrong (RAG non-parametric memory: abstract p1 not retrieved), 1 abstention caused by a retrieval miss (BM25 robust baseline, p1 not retrieved); the unanswerable question correctly abstained. Both misses are retrieval failures - candidates for Phase 4 (BM25 + RRF).

**Claude prompt - Phase 1**
```text
Phase 0 is complete. Implement Phase 1 from phases.md: the baseline end-to-end slice for digital PDFs only (no OCR, no tables, no verification yet).

Build in this order, running tests after each step:
1. processing/extract.py - PyMuPDF page-by-page text, stable document_id registry (DOC01...), filename + 1-indexed page.
2. processing/chunking.py - configurable size/overlap, sentence-aware, never crossing page boundaries, chunk_id like DOC01_P4_C1.
3. retrieval/index.py - Sentence-Transformers (normalised) + FAISS inner product, metadata aligned, save/load under data/indexes/.
4. retrieval/search.py - retrieve(question, top_k) returns original chunk records with scores.
5. generation.py - prompt that answers ONLY from supplied passages, cites chunk IDs, abstains if insufficient, JSON output. Document/page shown to users must come from metadata, never from LLM output.
6. pipeline.answer_question() and CLI: python -m trustqa.cli "question".
7. Tests: unique chunk IDs, 1-indexed pages, known question retrieves known chunk (use FakeLLM for generation tests).

Keep the Phase 0 schemas unchanged; if you think a change is needed, stop and ask. At the end, list the config to freeze for baseline system A. Do not commit or tag.
```

## Phase 2 - OCR and tables (1-1.5 weeks)
- [x] Scanned-page detection: text length / char-count threshold per page (also detect garbage text via non-alphanumeric ratio).
- [x] OCR fallback: render page via PyMuPDF at 300 DPI, run Tesseract with per-word confidence (`image_to_data`); store `extraction_method`, `ocr_used`, mean `ocr_confidence`. Light preprocessing (grayscale, deskew, threshold) behind a flag; compare on your scanned PDFs.
- [x] Table extraction with pdfplumber: `table_id`, headers, rows, cells, units, page. Handle merged/multi-line headers; Camelot only if pdfplumber fails on digital tables.
- [x] Table chunks: `content_type="table"`, `table_data` kept structured; `text` is a row-aware representation, e.g. `Table T2 (Revenue, Rs crore) | Year=2025 | Segment=Retail | Value=12`. Also emit header/caption context in every row chunk.
- [x] Canonical JSONL output to `data/processed/` (one record per chunk).
- [x] Inspection tool: small script that dumps a page's extracted text/table next to a page image for manual QA.
- [x] Tests: OCR triggered only on low-text pages; table headers/rows preserved; numbers, signs, decimals unaltered (NFR-03).

Exit check: scanned PDF questions retrieve the right page; table question returns correct value + unit + period. **Tag `v0.2-ocr-tables`.** This is baseline system B (OCR + plain RAG: same pipeline with tables flattened and no verification).

Phase 2 notes: detection thresholds `OCR_MIN_CHARS=50`, `OCR_MAX_NONALNUM_RATIO=0.6` (digital pilot pages: >=290 alnum chars, ratio <=0.15 except the Self-RAG TOC page at 0.49; every synthetic-scan page has 0 chars). Undecodable glyphs (U+FFFD, `(cid:N)`, private use) count as garbage; blank pages are skipped; OCR text replaces the text layer only if it is better. OCR confidence is stored per chunk (mean of that chunk's words). Preprocessing (grayscale + deskew + Otsu) measured on the 9-page scan against ground truth: CER 5.39% / WER 10.2% without vs 5.55% / 11.0% with, so `OCR_PREPROCESS=0` (synthetic skew <=0.6 deg is already handled by Tesseract; re-test on real scans). Tables: caption-anchored rule-bounded grids built from pdfplumber words (booktabs papers), plus pdfplumber's lines strategy for fully ruled grids; Camelot not needed. 24 tables found across the 4 pilot digital sources incl. the Transformer source PDF, all captions correct. Super/subscripts are kept as `10^19` / `log_k`, never merged into digits. Baseline B frozen in `eval/configs/system_B.json` (`BASELINE_B=1`); A's index key is unchanged with OCR and tables off (tested).

Exit-check results (qwen2.5:3b, top_k=5, 4 pilot PDFs; 5 scanned + 6 table questions; correct = right value in the answer and a citation to a page that states it):

| | A (text only) | B (OCR, flat tables) | C (OCR + table chunks) |
|---|---|---|---|
| Scanned questions correct | 0/5 (no text for the scan) | 4/5 | 4/5 |
| Table questions correct | 1/6 | 1/6 | 4/6 |
| A page stating the value in top-5, table qs | 3/6 | 3/6 | 5/6 |

A/B's one table hit (BM25 on TREC-COVID = 0.656) came from BEIR p9, whose flattened Hole@10 table repeats the value; on flattened table text they otherwise answered with wrong numbers from the right page ("20ms" for DPR CPU latency 230ms, "300" for NFCorpus' 323 test queries). Scanned results are identical for B and C (no tables are extracted from OCR pages).

Failures: (1) scanned "Adam beta2" question: page 7 not retrieved, and OCR turned "β1 = 0.9, β2 = 0.98, ϵ = 10−9" into "3; = 0.9, 52 = 0.98 « = 10~°" with chunk confidence 92.5, so confidence alone does not flag symbol errors; (2) "TAS-B on FiQA-2018" (0.300): row chunk not in top-5 under dense retrieval (candidate for BM25, Phase 4), model abstained; (3) "BM25+CE vs BM25 on average": correct row retrieved at rank 1 (`...BM25+CE=+ 11%`) but the model answered 0.347 (generation error, a Phase 3 numeric-check case); (4) OCR'd tables are text only (scanned Table 2: "37.3" read as "373", "10^20" as "107°"); OCR table chunks have visibly lower confidence (80-88 vs 94-96 for prose). Table-extraction limits seen: row labels spanning several rows stay split per line (BEIR Table 1 "Bio-Medical / Information / Retrieval (IR)"); vertically centred cells become their own row (RAG Table 6 FEVER columns); tightly packed math columns merge (Transformer Table 3); merged body cells (\multicolumn) land in one of the spanned columns. Period check (supplement, since no pilot paper has periods): a synthetic annual-report page with a segment x FY2023-FY2025 table (Rs crore) added to the pilot set in a scratch run, 3 questions. Values correct 3/3 in B and 3/3 in C, incl. the sign of -1,250.40; C cites the exact row chunk, whose text and table_data carry the unit (`Rs crore`) and period (`FY2024=1,12,980.25`). qwen2.5:3b restated unit and period in its answer text only once (B, P2); answer wording is a generation concern (prompt frozen for A/B, claim checks in Phase 3).

Phase 2 design decisions (kept deliberately): (1) a table row chunk's `table_data` holds only its own row, so a cited row is exactly the evidence Phase 3 checks; `tables.rebuild_table(chunks, table_id)` reassembles the full table for display; (2) super/subscripts in table cells are written `10^19` / `log_k` - Phase 3 numeric normalisation must read `^` as an exponent; (3) Tesseract missing -> digital pages kept with a warning in `documents.json`, error only if a document has no usable text; (4) baselines A/B answer via `generation.answer_unverified` (VERIFY=0), a frozen copy of the Phase 1 status policy, so Phase 3 changes to `verification/` cannot alter them.

**Claude prompt - Phase 2**
```text
Phase 1 is tagged v0.1-baseline. Implement Phase 2 from phases.md: OCR and table extraction. Do not touch verification.

Steps:
1. Scanned/garbage-page detection per page (char count threshold + non-alphanumeric ratio); thresholds in config.
2. OCR fallback: render at 300 DPI with PyMuPDF, Tesseract image_to_data with per-word confidence; set extraction_method, ocr_used, mean ocr_confidence. Put preprocessing (grayscale, deskew, threshold) behind a flag.
3. Table extraction with pdfplumber: table_id, headers, rows, cells, units, page; handle merged/multi-line headers. Camelot only if pdfplumber clearly fails.
4. Table chunks: content_type="table", structured table_data kept, text is a row-aware representation including caption/header context in every row chunk. Numbers, signs and decimals must never be altered.
5. Write canonical JSONL to data/processed/.
6. Inspection script showing a page's extracted text/table beside the page image.
7. Tests: OCR only on low-text pages; headers/rows preserved; numeric fidelity.

Add a config flag that flattens tables and disables verification so baseline system B can be reproduced. Report results on the pilot scanned and table PDFs, including failures. Do not commit or tag.
```

## Phase 3 - Verification layer (1.5-2 weeks) - core contribution
Build in this order; each piece independently testable.
1. [x] **Citation registry + provenance check (FR-09):** LLM may only cite chunk IDs from the retrieved set; invalid IDs rejected and warned; document name/page resolved from trusted metadata.
2. [x] **Claim decomposition:** split answer into atomic claims, each mapped to its cited chunk(s).
3. [x] **Evidence-support check (FR-10):** per claim, judge supported / not supported / contradicted against cited text. Options (pick by testing on your questions): NLI cross-encoder (e.g. DeBERTa-NLI) as cheap first pass + LLM judge with strict JSON for borderline. Require a `supporting_quote` that must be a substring (fuzzy-matched) of the chunk - rejects fabricated quotes.
4. [x] **Numeric checks (FR-11):** extract numbers/units/periods from claim; verify each appears in cited evidence (normalise commas, lakh/crore, %, signs). For calculations (sum, difference, growth), have LLM emit an expression, evaluate it deterministically in Python, compare.
5. [x] **Conflict detection:** across retrieved chunks from different docs, detect same entity/metric/period with different values; surface both sources.
6. [x] **OCR-uncertainty warning:** if a cited chunk's `ocr_confidence` is below threshold, warn and cap status at Partially Supported when the claim is numeric.
7. [x] **Abstention gate:** retrieval-score threshold + support check; if nothing supports the question -> `Insufficient Evidence` with no guessed answer.
8. [x] **Status rules** per PRD 2.3: Verified only if all material claims supported AND all citations resolve.
9. [x] Tests with mock LLM: fabricated chunk ID rejected; wrong-page citation caught; unsupported claim -> warning; altered number caught; conflicting sources flagged; unanswerable -> abstain.

Exit check: hand-built adversarial cases (10-15) behave correctly. **Tag `v0.3-verification`.**

Phase 3 notes (items 1-9 implemented; exit check passed, see below). Modules in `src/trustqa/verification/`: `provenance.py` (1), `claims.py` (2), `support.py` (3), `numeric.py` (4), `conflicts.py` (5), `ocr_check.py` (6), `abstention.py` (7), `status.py` (8), wired in `verify.py`; `pipeline.answer_question` calls `run_verification` so config thresholds and ablation switches (`VERIFY_SUPPORT`, `VERIFY_NUMERIC`, `VERIFY_CONFLICTS`) apply. LLM calls per answer: generation + support judge, plus a calculation call only when a number is not in the evidence and a fact-extraction call only for numeric answers over passages from 2+ documents. Every LLM output is a proposal checked in Python. Design decisions:
- Claims are sentences, not LLM-made atomic facts: the judge must find every fact of a sentence and the numeric check checks every number, so a mixed sentence still fails; no extra call that could drop facts. Uncited sentences are checked against all of the answer's cited chunks.
- Support: one LLM judge call per answer (strict JSON) with a verbatim quote that must occur in a chunk the claim cites (normalised; fuzzy >= 90 only for quotes >= 20 chars; numbers in a quote must match exactly). Judge failure -> claims "unverified" -> Partially Supported, not withheld. Short answers ("0.347") are marked [short answer] and judged as answers to the question. No NLI cross-encoder: no measured need yet.
- Numbers: normalised (Indian/Western commas, lakh/crore/million, %, signs, 10^19), rounding allowed only with >= 2 significant digits, FY/quarter periods checked. In table-row chunks a value counts only from a cell whose row label and best-matching column header share words with the question/claim (catches F5: right number, wrong cell, which qwen2.5:3b's judge called supported). Calculations: LLM proposes an expression, Python evaluates it exactly; operands must be evidence numbers (plus 1 and 100), and an expression that only restates a passage number is rejected (seen live as a way around the cell rule).
- Conflicts: LLM extracts {entity, metric, period, value, unit}; Python drops facts whose value is not in the chunk, groups by entity/metric/period/unit and flags cross-document disagreements that touch a cited chunk. Extraction failure -> CONFLICT warning, cannot be Verified.
- OCR: uncertain if mean confidence < 90, >= 2 stray symbols, or a "3; =" pattern (F1). Calibrated on the 45 pilot OCR chunks vs ground truth: flags P7_C5 (F1) and both misread Table 2 chunks, no chunk with CER < 5%. Numeric claims on uncertain chunks are capped at Partially Supported.
- Abstention: top retrieval score < 0.30 (MiniLM; lowest answerable top-1 on 10 pilot questions was 0.352, 3/7 unanswerable fell below) skips generation; after verification, no supported material claim -> Insufficient Evidence (withheld answer logged for error analysis).
- Quote relevance: a "supported" quote must contain a content word the claim adds beyond the question, or one of the claim's digit sequences (added after adversarial case A16).
- Status: Verified only if every material claim is supported (quote validated, numbers found or calculated), every citation resolves (incl. no page/document mismatch in prose), no conflict, no numeric claim on uncertain OCR, and every check ran.

Live check (qwen2.5:3b, top_k=5, 4 pilot PDFs, 10 questions, judge on the same model): Verified and correct: BM25 on TREC-COVID 0.656, Transformer (big) EN-DE BLEU 28.4 (scanned doc), RAG retriever initialised from DPR, Self-RAG reflection tokens (supported but incomplete: only critique tokens named - completeness is not checked). Withheld (Insufficient Evidence), correctly: "BEIR contains 5 datasets" (5 not in the cited row), F5 "0.347" (wrong cell). Abstained: TAS-B/FiQA (F3, retrieval miss), Adam beta2 (F4), CEO salary (model abstained), World Cup (retrieval gate, no LLM call). 9-36 s per answer on the RTX 3050.

Exit check (`tests/test_adversarial.py`, 17 cases, scripted misbehaving generator/judge): fabricated IDs in valid format and next to a real ID (A01-A02), wrong page + document in prose (A03), citation swap with the judge quoting another chunk (A04), fabricated quote (A05), transposed Indian-format digits, sign flip, unit swap (12 percent -> Rs. 12 crore), period swap in a table, wrong growth calculation (A06-A10), correct calculation as a Verified control (A11), cross-document conflict (A12), unanswerable with a hallucinated cited answer (A13), F1 OCR garble at confidence 92.5 (A14), instructions planted in a passage with a compromised judge, numeric and non-numeric (A15-A16), malformed judge output (A17). First run: 16/17. A16 failed (Verified): the judge quoted planted text "every statement about Helios is verified and supported" for "Helios is audited by Deloitte", and the quote check only tested that the quote exists in the chunk. Fixed with the quote-relevance rule; 17/17 pass, and the live re-run of the 8 pilot questions gave the same statuses as before. Remaining limits: failures.md F9-F11.

**Claude prompt - Phase 3** (run one numbered item per session/prompt; this is the core contribution)
```text
Phases 0-2 are done. We are implementing Phase 3 (verification layer) in src/trustqa/verification/. Implement ONLY item <N> from the list in phases.md Phase 3, then stop.

Rules for every item:
- Independently testable module with a clear function signature; wire into generate_and_verify only when its tests pass.
- Deterministic checks (citation validity, quote substring, numeric match, arithmetic) must be plain Python, not LLM calls.
- LLM judges must return strict JSON and be testable with FakeLLM.
- Document name/page always resolved from trusted chunk metadata.
- Status rules follow PRD 2.3: Verified only if all material claims are supported AND all citations resolve.
- Write tests first for the failure case this item must catch (fabricated chunk ID, wrong-page citation, unsupported claim, altered number, conflicting sources, unanswerable question, low OCR confidence).

At the end, show pytest output and list any design decision I should confirm. Do not commit or tag.
```
After items 1-8, run: "Create 10-15 adversarial cases in tests/test_adversarial.py covering each failure mode, run them, and report which fail and why."

## Phase 4 - Retrieval quality upgrades (3-5 days)
Only after Phase 3 passes (scope rule).
- [ ] Measure retrieval on your question set first (Recall@K, MRR).
- [ ] Add BM25 + reciprocal rank fusion (FR-15) - helps exact names, dates, values.
- [ ] Multi-document support: ensure top-k isn't dominated by one doc (per-doc quota or MMR) for multi-passage questions.
- [ ] Optional: small cross-encoder reranker only if measured gain justifies latency.
- [ ] Keep each upgrade behind a config flag so ablations are one-line.

Exit check: documented before/after retrieval metrics. Keep only what helps.

**Claude prompt - Phase 4**
```text
Phase 3 is tagged. Implement Phase 4 from phases.md (retrieval upgrades), measurement first.

1. Write eval/retrieval_eval.py computing Recall@K and MRR on the current question set (eval/questions.jsonl, or a small dev set if incomplete). Record the baseline numbers to eval/results/retrieval_baseline.json before changing anything.
2. Add BM25 + reciprocal rank fusion behind a config flag (default off).
3. Add per-document quota or MMR so multi-doc questions aren't dominated by one document, behind a flag.
4. Optional cross-encoder reranker only if steps 2-3 leave a measurable gap; behind a flag and report latency.
5. Produce a before/after table per flag combination. Recommend which flags to keep based on the numbers only; do not invent results.

Do not change schemas or verification code. Do not commit or tag.
```

## Phase 5 - Streamlit demo (4-5 days)
- [ ] `app/streamlit_app.py`: multi-file upload, processing progress, per-doc status (pages, OCR pages, tables found), question box.
- [ ] Result view: answer, colour-coded status badge, warnings list, citations with document, page, supporting quote.
- [ ] Evidence panel: rendered PDF page image with highlighted/quoted text; for table chunks show the structured table.
- [ ] Explicit errors for unreadable PDFs, extraction failure, missing API key (NFR-09).
- [ ] Toggle to compare modes (baseline vs TrustQA) on the same question - great for demos.
- [ ] Cache processed docs/index so reuploads are fast.

Exit check: a new user can upload 3 PDFs and get a verified answer with no terminal. **Tag `v0.4-demo`.**

**Claude prompt - Phase 5**
```text
Implement Phase 5 from phases.md: the Streamlit demo in app/streamlit_app.py. The UI must only call pipeline functions; no business logic in the app.

Include: multi-file upload with processing progress and per-doc status (pages, OCR pages, tables); question box; result view with answer, colour-coded status badge, warnings, and citations (document, page, supporting quote); evidence panel rendering the PDF page image with the quote highlighted, and the structured table for table chunks; a baseline vs TrustQA comparison toggle; caching of processed docs/index keyed by file hash.

Handle explicitly with clear messages: unreadable/encrypted PDF, extraction failure, missing API key, empty results.

Then run the app (use the run skill), upload the pilot PDFs, ask 3 questions including one unanswerable, and report what you saw. Do not commit or tag.
```

## Phase 6 - Dataset and evaluation (1.5-2 weeks)
Start the dataset in parallel from Phase 1 - labelling takes longest.
- [ ] Dataset: 15 public PDFs (5 text, 5 scanned, 5 tables). Record source URL/licence in `data/README`.
- [ ] 50 questions in `eval/questions.jsonl`: 10 direct, 10 multi-doc, 10 table/numeric, 10 insufficient-evidence, 5 conflicting, 5 scanned. Fields: id, question, reference_answer/expected_abstention, relevant doc+page, category. Have the labels double-checked (re-verify a week later or ask a friend for 10-question spot check).
- [ ] Systems: A plain-text RAG, B OCR + plain RAG, C full TrustQA. Same questions, same LLM/embedding settings, configs logged.
- [ ] Metrics (`eval/metrics.py`): EM / token-F1 (short), rubric or LLM-judge with manual spot-check (long); Recall@K, Precision@K, MRR; provenance correctness and support correctness reported **separately**; unsupported-claim rate; correct-abstention rate; conflict detection; numeric value/unit/period correctness; CER/WER on OCR where ground truth exists (jiwer); latency.
- [ ] Ablations: remove one component at a time (no table-aware, no support check, no numeric check, no OCR) to show what each contributes.
- [ ] Outputs: `eval/results/*.json|csv`, plots in `eval/plots/`, and an error analysis of 10-15 failure cases (what failed and why).
- [ ] Run each system more than once if the LLM is non-deterministic (temperature 0, report variance if any).

Exit check: tables + plots reproducible by one command: `python -m eval.run_all`. **Tag `v1.0-eval`.**

**Claude prompt - Phase 6**
```text
Implement the evaluation harness from Phase 6 of phases.md. I (the human) supply the PDFs and labelled questions in eval/questions.jsonl; you build the tooling. Never invent scores or labels.

1. Validate eval/questions.jsonl against a schema (id, question, reference_answer or expected_abstention, relevant doc+page, category) and print category counts versus the target 10/10/10/10/5/5.
2. eval/systems.py: system A (plain-text RAG), B (OCR + plain RAG), C (full TrustQA) with identical LLM/embedding settings, temperature 0, config logged per run.
3. eval/metrics.py: EM, token-F1, Recall@K, Precision@K, MRR, provenance correctness and support correctness reported separately, unsupported-claim rate, correct-abstention rate, conflict detection, numeric value/unit/period correctness, CER/WER via jiwer where ground truth exists, latency. Unit-test metrics on tiny hand-made examples.
4. Ablation switches: no table-aware, no support check, no numeric check, no OCR.
5. eval/run_all.py: one command runs everything, writes eval/results/*.json|csv and eval/plots/, with LLM-output caching by (prompt, model) hash.
6. eval/error_analysis.py: dump the worst failures per system for me to hand-analyse.

Report measured results only, with failure examples. Do not commit or tag.
```

## Phase 7 - Polish, docs, delivery (4-5 days)
- [ ] README: setup (Windows + Colab), quickstart, architecture diagram, screenshots.
- [ ] Colab notebook in `notebooks/` that runs the pipeline end to end.
- [ ] Final report/slides: problem, related work (verify citations against originals), design, results, failures, limitations, future work.
- [ ] 3-minute demo script and a recorded backup video.
- [ ] Security pass: no keys in repo or history; `.env` ignored.
- [ ] Clean `pytest` run on a fresh clone.

**Claude prompt - Phase 7**
```text
Implement Phase 7 from phases.md (polish and delivery).

1. Rewrite README.md: setup for Windows and Colab, quickstart, architecture diagram (mermaid), results table pulled from eval/results (real numbers only), limitations.
2. Create notebooks/trustqa_colab.ipynb running the pipeline end to end on a sample PDF.
3. Draft docs/report_outline.md (problem, related work with placeholders marked "VERIFY CITATION", design, results, failures, limitations, future work) and docs/demo_script.md (3-minute script).
4. Security pass: scan the working tree and (if git is present) history for API keys; confirm .env is in .gitignore and .env.example has placeholders only.
5. Simulate a fresh clone in a temp directory, install requirements, and run pytest; report any failure.

Do not fabricate citations or results. Do not commit or tag.
```

---

## Risks and mitigations
| Risk | Mitigation |
|---|---|
| OCR noise on poor scans | Page-level fallback, 300 DPI, preprocessing flag, confidence propagated to warnings |
| Irregular tables | pdfplumber tuning, Camelot fallback, keep original table image visible in UI |
| LLM judge is itself wrong | Quote-must-be-substring rule, NLI first pass, manual audit of a sample, report judge agreement |
| Label quality | Double-check questions, record page-level evidence, spot-check |
| Solo scope creep | Phase gates; Phase 4 extras only if measured to help |
| API cost/rate limits | Cache LLM outputs by (prompt, model) hash; use a smaller model for judge during dev |
| Python 3.14 locally may lack wheels for faiss-cpu/torch | Use Python 3.11/3.12 venv, or develop in Colab |

## Weekly checkpoint (use every week)
1. Does the end-to-end demo still run? 2. Are tests green? 3. Did I tag the phase? 4. What is the next smallest slice?
