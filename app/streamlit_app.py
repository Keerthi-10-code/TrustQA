"""TrustQA demo (Phase 5). Run: streamlit run app/streamlit_app.py

Display only: every decision (processing, caching, retrieval, verification, status,
quote location) is made by trustqa.corpus / trustqa.evidence / the pipeline. Document
names and page numbers shown come from chunk metadata, never from the LLM.
"""

from __future__ import annotations

import logging

import pandas as pd
import streamlit as st

from trustqa.config import ConfigError
from trustqa.corpus import (
    DUPLICATE,
    ENCRYPTED,
    EXTRACTION_FAILED,
    NO_TEXT,
    SYSTEMS,
    UNREADABLE,
    Answer,
    Corpus,
    CorpusError,
    prepare_corpus,
    save_upload,
    system_config,
)
from trustqa.evidence import render_page_evidence
from trustqa.llm import LLMError, check_available
from trustqa.schemas import ContentType, Result, Status

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s %(message)s")

st.set_page_config(page_title="TrustQA", page_icon=":mag:", layout="wide")

STATUS_STYLE = {  # badge colour + one-line meaning (PRD 2.3)
    Status.VERIFIED: ("green", ":material/verified:",
                      "Every material claim is supported by a cited passage and every citation resolves."),
    Status.PARTIALLY_SUPPORTED: ("orange", ":material/warning:",
                                 "Some claims could not be fully verified against the cited evidence."),
    Status.INSUFFICIENT_EVIDENCE: ("red", ":material/block:",
                                   "The uploaded documents do not contain enough evidence; no answer is given."),
}

ERROR_HELP = {
    ENCRYPTED: "This PDF is password-protected. Remove the password (e.g. open it and "
               "print/save it as a new PDF) and upload it again.",
    UNREADABLE: "This file could not be opened as a PDF. It may be damaged, empty or "
                "not a PDF at all.",
    EXTRACTION_FAILED: "Text extraction failed for this document.",
    NO_TEXT: "No text could be extracted: the pages are blank, or they are scanned and "
             "OCR is unavailable (install Tesseract and set TESSERACT_CMD in .env).",
    DUPLICATE: "Same file content was uploaded twice; the copy was skipped.",
}

WARNING_LABEL = {
    "unsupported_claim": "Unsupported claim",
    "invalid_chunk_id": "Invalid citation",
    "page_mismatch": "Page mismatch",
    "ocr_uncertainty": "OCR uncertainty",
    "numeric_mismatch": "Numeric mismatch",
    "conflict": "Conflicting sources",
}


# ---------------------------------------------------------------- cached helpers

@st.cache_data(ttl=30, show_spinner=False)
def llm_status(system: str) -> tuple[bool, str]:
    """(ok, message) for the configured LLM provider; re-checked every 30 s."""
    try:
        return True, check_available(system_config(system))
    except ConfigError as e:
        return False, f"LLM configuration problem: {e}"
    except LLMError as e:
        return False, str(e)


@st.cache_data(max_entries=64, show_spinner=False)
def page_evidence(path: str, page: int, chunk_json: str | None, quote: str,
                  tesseract_cmd: str | None, ocr_lang: str):
    from trustqa.schemas import Chunk

    chunk = Chunk.from_json(chunk_json) if chunk_json else None
    return render_page_evidence(path, page, chunk, quote or None,
                                tesseract_cmd=tesseract_cmd, ocr_lang=ocr_lang)


def get_corpus(system: str, uploads: list[tuple[str, bytes]]) -> Corpus | None:
    """Prepare (or reuse from this session) the corpus for an upload set and system.
    Disk caches in trustqa.corpus make a re-upload in a new session fast too."""
    config = system_config(system)
    paths = [save_upload(name, data, config) for name, data in uploads]
    key = (system, tuple(str(p) for p in paths))
    cache = st.session_state.setdefault("corpora", {})
    if key in cache:
        return cache[key]
    label = SYSTEMS[system][0]
    bar = st.progress(0.0, text=f"{label}: starting")
    try:
        corpus = prepare_corpus(paths, config, system,
                                progress=lambda f, m: bar.progress(min(f, 1.0), text=f"{label}: {m}"))
    except CorpusError as e:
        bar.empty()
        st.error(f"**{label}: no usable documents.** {e}")
        return None
    except ConfigError as e:
        bar.empty()
        st.error(f"**Configuration problem:** {e}")
        return None
    bar.empty()
    cache[key] = corpus
    return corpus


# ---------------------------------------------------------------- views

def show_documents(corpus: Corpus) -> None:
    rows = []
    for d in corpus.documents:
        if d.ok:
            state = "cached" if d.from_cache else f"processed in {d.seconds:.1f} s"
        else:
            state = "skipped" if d.error_kind == DUPLICATE else "failed"
        conf = d.ocr_mean_confidence
        rows.append({
            "ID": d.document_id or "-",
            "Document": d.name,
            "Pages": str(d.pages) if d.ok else "-",
            "OCR pages": str(len(d.ocr_pages)) if d.ok else "-",
            "OCR conf.": f"{conf:.1f}" if conf is not None else "-",
            "Tables": str(d.tables) if d.ok else "-",
            "Chunks (text + table)": f"{d.text_chunks} + {d.table_chunks}" if d.ok else "",
            "Status": state,
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    for d in corpus.failed:
        msg = f"**{d.name}**: {ERROR_HELP.get(d.error_kind, '')}\n\n`{d.error}`"
        (st.warning if d.error_kind == DUPLICATE else st.error)(msg)
    for d in corpus.ready:
        for w in d.warnings:
            st.caption(f":material/info: {d.name}: {w}")


def show_result(answer: Answer, corpus: Corpus, slot: str) -> None:
    r: Result = answer.result
    colour, icon, meaning = STATUS_STYLE[r.status]
    st.badge(r.status.value, color=colour, icon=icon)
    st.caption(f"{SYSTEMS[answer.system][0]} · {answer.seconds:.1f} s · {meaning}")

    if r.status is Status.INSUFFICIENT_EVIDENCE:
        st.info(f"{r.answer}\n\nNo passage in the uploaded documents supports an answer "
                "to this question, so none is given.")
    else:
        st.markdown(r.answer)

    if r.warnings:
        with st.container(border=True):
            st.markdown(f"**Warnings ({len(r.warnings)})**")
            for w in r.warnings:
                where = f" (`{w.chunk_id}`)" if w.chunk_id else ""
                st.markdown(f"- **{WARNING_LABEL.get(w.code.value, w.code.value)}**{where}: {w.message}")

    if r.citations:
        st.markdown(f"**Citations ({len(r.citations)})**")
        for i, c in enumerate(r.citations, 1):
            with st.container(border=True):
                st.markdown(f"**[{i}] {c.document_name}, page {c.page_number}**  \n"
                            f"<small>chunk {', '.join(c.chunk_ids)}</small>",
                            unsafe_allow_html=True)
                if c.supporting_quote:
                    st.markdown(f"> {c.supporting_quote}")
                elif corpus.config.verify:
                    st.caption("No validated supporting quote for this citation.")
                else:
                    st.caption("Baseline: citations are not checked, so there is no supporting quote.")
        show_evidence(r, corpus, slot)

    with st.expander(f"Retrieved passages (top {len(answer.passages)})"):
        for p in answer.passages:
            ch = p.chunk
            ocr = f", OCR conf {ch.ocr_confidence:.0f}" if ch.ocr_used and ch.ocr_confidence else ""
            st.markdown(f"**{p.rank}.** `{ch.chunk_id}` {ch.document_name} p{ch.page_number} "
                        f"· {ch.content_type.value}{ocr} · score {p.score:.3f}")
            st.caption(ch.text[:400] + ("..." if len(ch.text) > 400 else ""))


def show_evidence(r: Result, corpus: Corpus, slot: str) -> None:
    options = list(range(len(r.citations)))
    pick = st.selectbox(
        "Evidence panel", options, key=f"evidence_{slot}",
        format_func=lambda i: (f"[{i + 1}] {r.citations[i].document_name}, "
                               f"page {r.citations[i].page_number}"))
    c = r.citations[pick]
    path = corpus.document_path(c.document_id)
    chunks = [corpus.chunk(cid) for cid in c.chunk_ids]
    chunks = [ch for ch in chunks if ch is not None]
    first = chunks[0] if chunks else None
    if path is None:
        st.error(f"Source file for {c.document_name} is no longer available.")
        return

    for ch in chunks:
        if ch.content_type is ContentType.TABLE and ch.table_data:
            full = corpus.table(ch.table_data.table_id)
            st.markdown(f"**Structured table {ch.table_data.table_id}**"
                        + (f" ({full.units})" if full and full.units else ""))
            if full and full.caption:
                st.caption(full.caption)
            if full:
                df = pd.DataFrame(full.rows, columns=_unique(full.headers))
                cited_row = ch.table_data.rows[0]
                st.dataframe(df.style.apply(
                    lambda row: ["background-color: rgba(255, 210, 0, .35)"
                                 if list(row.values) == cited_row else "" for _ in row], axis=1),
                    hide_index=True, width="stretch")
            if ch.ocr_used:
                st.caption("This table was read by OCR; check values against the page image.")

    try:
        with st.spinner("Rendering page" + (" (locating words with OCR)" if first and first.ocr_used else "")):
            ev = page_evidence(str(path), c.page_number, first.to_json() if first else None,
                               c.supporting_quote, corpus.config.tesseract_cmd, corpus.config.ocr_lang)
    except ValueError as e:
        st.error(f"Could not render the page: {e}")
        return
    what = {"quote": "supporting quote", "chunk text": "cited passage",
            "table row": "cited table row"}.get(ev.source, ev.source)
    if ev.located:
        st.caption(f"Highlighted: {what} ({ev.highlighted_words} words located on the page).")
    elif ev.note:
        st.caption(f"Not highlighted: {ev.note}.")
    st.image(ev.png, caption=f"{c.document_name}, page {c.page_number}", width="stretch")


def _unique(headers: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    out = []
    for h in headers:
        h = h or "-"
        seen[h] = seen.get(h, 0) + 1
        out.append(h if seen[h] == 1 else f"{h} ({seen[h]})")
    return out


# ---------------------------------------------------------------- page

st.title("TrustQA")
st.caption("Ask questions over your PDFs. Every answer comes with its source document, page, "
           "supporting quote and a verification status.")

with st.sidebar:
    st.header("Settings")
    compare = st.toggle("Compare with a baseline", value=False,
                        help="Answer the same question with a plain RAG baseline as well.")
    baseline = st.radio("Baseline", ["B", "A"], format_func=lambda s: SYSTEMS[s][0],
                        disabled=not compare)
    st.divider()
    ok, msg = llm_status("C")
    if ok:
        st.success(f"LLM: {msg}", icon=":material/check_circle:")
    else:
        st.error(msg, icon=":material/error:")
    cfg = system_config("C")
    st.caption(f"Retrieval: top {cfg.top_k}, {'hybrid BM25 + dense' if cfg.retrieval_hybrid else 'dense'}"
               f" · OCR {'on' if cfg.ocr_enabled else 'off'}")

files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
if not files:
    st.info("Upload one or more PDFs to start. Scanned pages are OCR'd and tables are extracted; "
            "processed documents are cached, so re-uploading the same file is instant.")
    st.stop()

uploads = [(f.name, f.getvalue()) for f in files]
empty = [n for n, data in uploads if not data]
if empty:
    st.error(f"Empty file(s): {', '.join(empty)}. Upload them again.")
    uploads = [(n, d) for n, d in uploads if d]
    if not uploads:
        st.stop()

systems = ["C"] + ([baseline] if compare else [])
corpora: dict[str, Corpus] = {}
for s in systems:
    c = get_corpus(s, uploads)
    if c is not None:
        corpora[s] = c
if "C" not in corpora:
    st.stop()

with st.expander(f"Documents ({len(corpora['C'].ready)} ready, {len(corpora['C'].failed)} not used)",
                 expanded=not st.session_state.get("results")):
    show_documents(corpora["C"])
    if compare and baseline in corpora:
        st.caption(f"{SYSTEMS[baseline][0]} uses its own processing of the same files "
                   f"({'no OCR, ' if baseline == 'A' else ''}tables flattened, no verification).")

with st.form("ask", clear_on_submit=False):
    question = st.text_input("Question", placeholder="e.g. What nDCG@10 does BM25 achieve on TREC-COVID?")
    asked = st.form_submit_button("Ask", type="primary")

if asked:
    if not question.strip():
        st.warning("Please enter a question.")
    elif not ok:
        st.error(f"Cannot answer: {msg}")
    else:
        results = {}
        for s, corpus in corpora.items():
            with st.spinner(f"{SYSTEMS[s][0]}: retrieving, answering"
                            + (" and verifying" if corpus.config.verify else "") + "..."):
                try:
                    results[s] = corpus.ask(question)
                except ConfigError as e:
                    st.error(f"**Configuration problem:** {e}")
                except LLMError as e:
                    st.error(f"**{SYSTEMS[s][0]}: the language model call failed.** {e}")
                except ValueError as e:
                    st.error(str(e))
        st.session_state["results"] = {"question": question, "answers": results,
                                       "corpus": corpora["C"].key}

state = st.session_state.get("results")
if state and state["answers"] and state["corpus"] == corpora["C"].key:
    st.subheader(state["question"])
    shown = [s for s in systems if s in state["answers"]]
    cols = st.columns(len(shown)) if len(shown) > 1 else [st.container()]
    for col, s in zip(cols, shown):
        with col:
            if len(shown) > 1:
                st.markdown(f"#### {SYSTEMS[s][0]}")
            show_result(state["answers"][s], corpora[s], slot=s)
