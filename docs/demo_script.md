# TrustQA - 3-minute demo script

Setup before the demo (not shown): `streamlit run app/streamlit_app.py`, Ollama running with
`qwen2.5:3b`, the four pilot PDFs uploaded once so they are cached (re-upload is then instant).
Have a backup screen recording of the same run in case the live model is slow.

| Time | Do | Say |
|---|---|---|
| 0:00-0:20 | Show the empty app. | "TrustQA answers questions over a collection of PDFs - digital, scanned and table-heavy - and every answer comes with its document, page, supporting quote and a verification status. The point is not just an answer, but one you can check." |
| 0:20-0:45 | Drag in the 4 pilot PDFs. Open the **Documents** table. | "Each file is checked first - encrypted or damaged PDFs are reported and skipped. This one is a scan: 9 of 9 pages went through OCR, mean confidence 92.6. The BEIR paper has 9 tables, extracted as real rows and columns, not flattened text." |
| 0:45-1:20 | Ask **"What nDCG@10 does BM25 achieve on TREC-COVID?"** Point at the green **Verified** badge, then the evidence panel. | "0.656, Verified. The citation is BEIR page 7, and the evidence panel shows the full table with the exact row highlighted, plus the page image. The document name and page come from our metadata, never from the language model." |
| 1:20-1:45 | Ask **"What BLEU score does the big Transformer model achieve on English-to-German translation?"** | "This answer comes from the scanned paper. The quote is highlighted on the page image using OCR word positions - 28.4 BLEU, page 8." |
| 1:45-2:20 | Toggle **Compare with a baseline** (B). Ask **"What GPUs were used to train the Self-RAG models?"** | "Plain RAG on the right answers with a confident-looking sentence - but it's the *RAG* paper's V100 GPUs, not Self-RAG's. TrustQA checks that the cited document is the one the question is about, and withholds that answer instead of verifying it." (If TrustQA finds the A100 sentence it shows that instead - either way, point out the cited document.) |
| 2:20-2:45 | Ask **"What was the CEO's salary in 2023?"** | "Nothing in these documents answers this, so TrustQA says Insufficient Evidence rather than guessing." |
| 2:45-3:00 | Close on the status legend / README results table. | "We measured all of this against plain RAG on a labelled question set: higher answer accuracy, citations on the right page far more often, and every unanswerable question correctly refused. Limits we found - like answers that are true but incomplete - are documented in failures.md." |

Fallbacks: if a live answer is slow (>30 s), say "the 3B model runs locally on a laptop GPU" and
switch to the recording. If an answer differs from the script, explain its status and warnings -
the warnings are part of the demo.
