"""TrustQA demo. Run: streamlit run app/streamlit_app.py

Display only: every decision (processing, caching, retrieval, verification, status, quote
location) is made by trustqa.corpus / trustqa.evidence / the pipeline. Document names and page
numbers shown come from chunk metadata, never from the LLM. This file only arranges the
results: a conversation of questions, each answer with its status, the reasons behind it,
numbered sources and an evidence viewer.
"""

from __future__ import annotations

import html
import logging
import os
import re
from pathlib import Path

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
from trustqa.schemas import ContentType, Result, Status, WarningCode

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s %(message)s")

st.set_page_config(page_title="TrustQA", page_icon=":mag:", layout="wide")

STATUS_STYLE = {  # badge colour, icon, plain-language meaning (PRD 2.3)
    Status.VERIFIED: ("green", ":material/verified:",
                      "Every statement was matched to a quote on the cited page."),
    Status.PARTIALLY_SUPPORTED: ("orange", ":material/warning:",
                                 "An answer was found, but not every part of it could be confirmed - "
                                 "check the notes and the sources."),
    Status.INSUFFICIENT_EVIDENCE: ("red", ":material/block:",
                                   "No answer is shown because the documents did not clearly support one."),
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
    "unsupported_claim": "Not confirmed",
    "invalid_chunk_id": "Invalid citation",
    "page_mismatch": "Page mismatch",
    "ocr_uncertainty": "Scanned text may be misread",
    "numeric_mismatch": "Number not found in source",
    "conflict": "Sources disagree",
    "no_evidence": "Why there is no answer",
}

TIPS = [
    "Use words that appear in the document (names, terms, section titles).",
    "Ask one thing at a time; split comparisons into two questions.",
    "Check the spelling of names and technical terms.",
    "Make sure the document that contains the answer is uploaded.",
]

CSS = """
<style>
.tq-answer {font-size: 1.05rem; line-height: 1.6;}
.tq-answer sup {color: var(--primary-color, #2a78d6); font-weight: 600; padding-left: 1px;}
.tq-quote {border-left: 3px solid rgba(42,120,214,.6); padding: 2px 0 2px 10px; margin: 4px 0;
           opacity: .9; font-size: .93rem;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


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


# ---------------------------------------------------------------- documents

def show_documents(corpus: Corpus) -> None:
    ready = corpus.ready
    cols = st.columns(4)
    cols[0].metric("Documents ready", f"{len(ready)} / {len(corpus.documents)}")
    cols[1].metric("Pages", sum(d.pages for d in ready))
    cols[2].metric("Scanned pages read (OCR)", sum(len(d.ocr_pages) for d in ready))
    cols[3].metric("Tables extracted", sum(d.tables for d in ready))
    for d in corpus.failed:
        msg = f"**{d.name}**: {ERROR_HELP.get(d.error_kind, '')}\n\n`{d.error}`"
        (st.warning if d.error_kind == DUPLICATE else st.error)(msg)
    with st.expander("Per-document details"):
        rows = []
        for d in corpus.documents:
            if d.ok:
                state = "cached" if d.from_cache else f"processed in {d.seconds:.1f} s"
            else:
                state = "skipped" if d.error_kind == DUPLICATE else "failed"
            conf = d.ocr_mean_confidence
            rows.append({
                "ID": d.document_id or "-", "Document": d.name,
                "Pages": str(d.pages) if d.ok else "-",
                "OCR pages": str(len(d.ocr_pages)) if d.ok else "-",
                "OCR conf.": f"{conf:.1f}" if conf is not None else "-",
                "Tables": str(d.tables) if d.ok else "-",
                "Chunks (text + table)": f"{d.text_chunks} + {d.table_chunks}" if d.ok else "",
                "Status": state,
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        for d in ready:
            for w in d.warnings:
                st.caption(f":material/info: {d.name}: {w}")


def starter_questions(corpus: Corpus, limit: int = 4) -> list[str]:
    """Questions built from the documents' own section headings (display of index data);
    subsections first, since they are the most specific."""
    skip = {"introduction", "references", "conclusion", "conclusions", "acknowledgments",
            "acknowledgements", "abstract", "appendix", "related work"}
    found: list[tuple[int, str]] = []
    for ch in corpus.index.chunks:
        if not ch.section or ch.content_type is not ContentType.TEXT:
            continue
        parts = ch.section.split(" > ")
        title = re.sub(r"^(?:[IVX]+|[A-Z]|\d+(?:\.\d+)*)\.?\s+", "", parts[-1]).strip()
        if title.isupper():
            title = title.title()
        if 3 <= len(title) <= 60 and title.lower() not in skip and title not in [t for _, t in found]:
            found.append((len(parts), title))
    found.sort(key=lambda t: -t[0])  # stable: deeper sections first, document order within
    return [f'What does the section "{t}" say?' for _, t in found[:limit]]


# ---------------------------------------------------------------- answers

_ID_GROUP = re.compile(r"\s?\[([^\[\]]*DOC\d+_P\d+[^\[\]]*)\]")


def answer_html(r: Result) -> str:
    """Answer text with chunk-ID markers replaced by numbered source markers."""
    number = {cid: i for i, c in enumerate(r.citations, 1) for cid in c.chunk_ids}
    text = html.escape(r.answer)

    def repl(m: re.Match) -> str:
        ids = [x.strip() for x in m.group(1).split(",")]
        nums = sorted({number[i] for i in ids if i in number})
        return f"<sup>[{', '.join(map(str, nums))}]</sup>" if nums else ""

    return f'<div class="tq-answer">{_ID_GROUP.sub(repl, text)}</div>'


def friendly(message: str, r: Result) -> str:
    """Display wording for a check message: chunk IDs become source numbers, jargon is softened."""
    number = {cid: i for i, c in enumerate(r.citations, 1) for cid in c.chunk_ids}
    msg = re.sub(r"DOC\d+_P\d+(?:_[A-Z]+\d+)+",
                 lambda m: f"source [{number[m.group(0)]}]" if m.group(0) in number else "a passage", message)
    msg = re.sub(r"(source \[(\d+)\])(, source \[\2\])+", r"\1", msg)
    for a, b in (("judge gave no verdict", "the checker gave no verdict"),
                 ("judge found no support", "the checker found no support"),
                 ("judge said supported", "the checker said supported"),
                 ("cited passage(s)", "cited source(s)"), ("claim ", "statement ")):
        msg = msg.replace(a, b)
    return msg[:1].upper() + msg[1:]


def check_chips(r: Result, corpus: Corpus) -> None:
    codes = {w.code for w in r.warnings}
    chips = []
    if r.citations:
        pages = ", ".join(f"p{c.page_number}" for c in r.citations)
        chips.append(f":green-badge[:material/description: {len(r.citations)} source(s): {pages}]")
    if corpus.config.verify:
        if WarningCode.UNSUPPORTED_CLAIM in codes:
            chips.append(":orange-badge[:material/rule: some statements not confirmed]")
        elif r.citations:
            chips.append(":green-badge[:material/format_quote: quotes matched]")
        if WarningCode.NUMERIC_MISMATCH in codes:
            chips.append(":orange-badge[:material/pin: a number not found in source]")
        if WarningCode.OCR_UNCERTAINTY in codes:
            chips.append(":orange-badge[:material/document_scanner: scanned text may be misread]")
        if WarningCode.CONFLICT in codes:
            chips.append(":red-badge[:material/compare_arrows: sources disagree]")
    else:
        chips.append(":gray-badge[baseline: not verified]")
    if chips:
        st.markdown(" ".join(chips))


def show_result(answer: Answer, corpus: Corpus, slot: str) -> None:
    r: Result = answer.result
    colour, icon, meaning = STATUS_STYLE[r.status]
    st.badge(r.status.value, color=colour, icon=icon)
    st.caption(f"{meaning} · {SYSTEMS[answer.system][0]} · {answer.seconds:.1f} s")

    if r.status is Status.INSUFFICIENT_EVIDENCE:
        reasons = [w.message for w in r.warnings if w.code is WarningCode.NO_EVIDENCE]
        details = [w for w in r.warnings if w.code is not WarningCode.NO_EVIDENCE]
        with st.container(border=True):
            st.markdown("**Why there is no answer**")
            for m in reasons or ["The retrieved passages did not contain enough information to answer this question."]:
                st.markdown(f"- {m}")
            if not corpus.config.verify:
                st.markdown("- Baseline system: the model itself said the passages do not answer the question.")
            st.markdown("**What you can try**")
            for t in TIPS:
                st.markdown(f"- {t}")
            if details:
                with st.expander("Details from the checks"):
                    for w in details:
                        st.markdown(f"- **{WARNING_LABEL.get(w.code.value, w.code.value)}**: {friendly(w.message, r)}")
    else:
        st.markdown(answer_html(r), unsafe_allow_html=True)
        check_chips(r, corpus)
        notes = [w for w in r.warnings if w.code is not WarningCode.NO_EVIDENCE]
        if notes:
            with st.expander(f"Notes from the checks ({len(notes)})", expanded=r.status is not Status.VERIFIED):
                for w in notes:
                    st.markdown(f"- **{WARNING_LABEL.get(w.code.value, w.code.value)}**: {friendly(w.message, r)}")

    if r.citations:
        st.markdown("**Sources**")
        for i, c in enumerate(r.citations, 1):
            quote = (f'<div class="tq-quote">{html.escape(c.supporting_quote)}</div>'
                     if c.supporting_quote else
                     ('<div class="tq-quote"><i>no validated quote for this source</i></div>'
                      if corpus.config.verify else
                      '<div class="tq-quote"><i>baseline: sources are not checked</i></div>'))
            st.markdown(f"<b>[{i}]</b> {html.escape(c.document_name)}, page {c.page_number}{quote}",
                        unsafe_allow_html=True)
        with st.expander("View the evidence on the page", expanded=False):
            show_evidence(r, corpus, slot)

    with st.expander(f"What the system read (top {len(answer.passages)} passages)"):
        for p in answer.passages:
            ch = p.chunk
            ocr = f" · OCR conf {ch.ocr_confidence:.0f}" if ch.ocr_used and ch.ocr_confidence else ""
            sec = f" · {ch.section}" if ch.section and ch.content_type is ContentType.TEXT else ""
            st.markdown(f"**{p.rank}.** {ch.document_name}, page {ch.page_number}{sec}"
                        f" · {ch.content_type.value}{ocr} · similarity {p.score:.2f}")
            st.caption(ch.text[:400] + ("..." if len(ch.text) > 400 else ""))


def show_evidence(r: Result, corpus: Corpus, slot: str) -> None:
    options = list(range(len(r.citations)))
    pick = st.selectbox(
        "Source", options, key=f"evidence_{slot}",
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
            st.markdown(f"**Table {ch.table_data.table_id}**"
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
        st.caption(f"Highlighted in yellow: the {what} ({ev.highlighted_words} words located on the page).")
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


def ask_all(question: str, corpora: dict[str, Corpus]) -> dict[str, Answer]:
    results: dict[str, Answer] = {}
    for s, corpus in corpora.items():
        with st.spinner(f"{SYSTEMS[s][0]}: reading the documents, answering"
                        + (" and checking the answer" if corpus.config.verify else "") + "..."):
            try:
                results[s] = corpus.ask(question)
            except ConfigError as e:
                st.error(f"**Configuration problem:** {e}")
            except LLMError as e:
                st.error(f"**{SYSTEMS[s][0]}: the language model call failed.** {e}")
            except ValueError as e:
                st.error(str(e))
    return results


# ---------------------------------------------------------------- page

with st.sidebar:
    st.title("TrustQA")
    ok, msg = llm_status("C")
    if ok:
        st.success(f"Model ready: {msg}", icon=":material/check_circle:")
    else:
        st.error(msg, icon=":material/error:")
    compare = st.toggle("Compare with plain RAG", value=False,
                        help="Answer each question with a plain RAG baseline as well, side by side.")
    baseline = st.radio("Baseline", ["B", "A"], format_func=lambda s: SYSTEMS[s][0],
                        disabled=not compare)
    if st.button("Clear conversation", icon=":material/delete_sweep:", width="stretch"):
        st.session_state["history"] = []
    st.divider()
    st.markdown("**What the status means**")
    st.markdown(":green-badge[:material/verified: Verified] every statement matched a quote on the cited page")
    st.markdown(":orange-badge[:material/warning: Partially Supported] an answer, but part of it is unconfirmed")
    st.markdown(":red-badge[:material/block: Insufficient Evidence] no answer - the reason is shown")
    with st.expander("How an answer is checked"):
        st.markdown(
            "1. Your PDFs are read page by page (scanned pages with OCR, tables as real tables).\n"
            "2. The most relevant passages are found (meaning + keywords).\n"
            "3. A language model answers **only** from those passages and cites them.\n"
            "4. Each statement is checked: its quote must exist on the cited page, numbers must "
            "match the source, and the cited document must be the one the question is about.\n"
            "5. Document names and page numbers always come from the files, never from the model.")
    cfg = system_config("C")
    st.caption(f"Retrieval: top {cfg.top_k}, {'hybrid BM25 + dense' if cfg.retrieval_hybrid else 'dense'}"
               f" · OCR {'on' if cfg.ocr_enabled else 'off'}")

st.header("Ask questions about your PDFs")
st.caption("Every answer shows where it came from - document, page and the exact quote - and how far it "
           "could be checked.")

files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
# TRUSTQA_PRELOAD: PDF paths (separated by os.pathsep) loaded when nothing is uploaded - for
# demos and for the end-to-end UI test (tests/test_app.py).
preload = [Path(x) for x in os.environ.get("TRUSTQA_PRELOAD", "").split(os.pathsep) if x.strip()]
if not files and not preload:
    st.info("Upload one or more PDFs to start. Scanned pages are OCR'd and tables are extracted; "
            "processed documents are cached, so re-uploading the same file is instant.")
    st.stop()

uploads = ([(f.name, f.getvalue()) for f in files] if files else
           [(p.name, p.read_bytes()) for p in preload if p.is_file()])
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

show_documents(corpora["C"])
if compare and baseline in corpora:
    st.caption(f"{SYSTEMS[baseline][0]} uses its own processing of the same files "
               f"({'no OCR, ' if baseline == 'A' else ''}tables flattened, no verification).")

history: list[dict] = st.session_state.setdefault("history", [])
key = corpora["C"].key
turns = [t for t in history if t["corpus"] == key]

if not turns:
    starters = starter_questions(corpora["C"])
    if starters:
        st.markdown("**Try a question**")
        cols = st.columns(len(starters))
        for col, q in zip(cols, starters):
            if col.button(q, key=f"starter_{q}", width="stretch"):
                st.session_state["pending_question"] = q

for n, turn in enumerate(turns):
    with st.chat_message("user"):
        st.markdown(turn["question"])
    with st.chat_message("assistant"):
        shown = [s for s in systems if s in turn["answers"]]
        cols = st.columns(len(shown)) if len(shown) > 1 else [st.container()]
        for col, s in zip(cols, shown):
            with col:
                if len(shown) > 1:
                    st.markdown(f"##### {SYSTEMS[s][0]}")
                show_result(turn["answers"][s], corpora[s], slot=f"{n}_{s}")

typed = st.chat_input("Ask a question about your documents")
question = typed or st.session_state.pop("pending_question", None)
if question:
    if not ok:
        st.error(f"Cannot answer: {msg}")
    elif not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            answers = ask_all(question, corpora)
        if answers:
            history.append({"question": question, "answers": answers, "corpus": key})
            st.rerun()
